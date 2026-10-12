"""Stage A/B count NLL+CRPS; Stage C participant beta NLL with frozen branches."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import FeatureDataset,collate,to_device,select_split,validate_fold
from .model import GSDVQ
from .losses import count_loss,participant_beta_nll
from .checkpoints import save_checkpoint,load_checkpoint,file_hash,save_run_record
from .metrics import rating_report
from . import QUANTIFIERS

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def loader(dataset,config,shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=shuffle,collate_fn=collate,
                      num_workers=0,generator=generator)

def verify_auxiliary_audit(rows,manifest_path,audit_path,policy):
    if policy=="diagnostic":
        return {"exposure_policy":"diagnostic","audit_record":str(audit_path) if audit_path else None}
    if audit_path is None:
        raise ValueError("Strict auxiliary training requires the completed decontamination audit")
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status")!="completed" or audit.get("filtered_manifest_sha256")!=file_hash(manifest_path):
        raise ValueError("Audit record does not correspond to this filtered manifest")
    if not {str(r["id"]) for r in rows} <= set(audit["retained_ids"]):
        raise ValueError("Unfiltered training IDs found")
    # Frozen-backbone exposure cannot be inferred just from head-training data.
    if audit.get("backbone_exposure")!="audited":
        raise ValueError("Unresolved frozen-backbone exposure is diagnostic, as in Appendix B.2")
    return {"exposure_policy":"strict","audit_sha256":file_hash(audit_path),
            "backbone_provenance":audit.get("backbone_provenance",{})}

def fit_auxiliary(rows,manifest_path,feature_root,config,stage,output,audit_path=None):
    seed_everything(config.seed)
    provenance = verify_auxiliary_audit(rows,manifest_path,audit_path,config.checkpoint_policy)
    train_rows = [r for r in rows if r.get("partition")=="train"]
    validation_rows = [r for r in rows if r.get("partition")=="validation"]
    # Held-out counting/expectation test examples are not used for selection.
    model = GSDVQ(config).to(config.device)
    module = model.observation if stage=="A" else model.expectation
    training = loader(FeatureDataset(train_rows,feature_root,config,stage,True),config,True)
    validation = loader(FeatureDataset(validation_rows,feature_root,config,stage,False),config)
    optimizer = make_optimizer(module.parameters(),config)
    epochs = config.stage_a_epochs if stage=="A" else config.stage_b_epochs
    best,stale,history = float("inf"),0,[]
    for epoch in range(epochs):
        module.train()
        losses=[]
        for batch in training:
            batch = to_device(batch,config.device)
            logits = model.observation.logits(batch["counting"]) if stage=="A" else model.expectation.logits(batch["scene"],batch["category"])
            loss = count_loss(logits,batch["count_bin"],config.crps_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite count loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()),len(batch["id"])))
        module.eval()
        valid=[]
        with torch.no_grad():
            for batch in validation:
                batch = to_device(batch,config.device)
                logits = model.observation.logits(batch["counting"]) if stage=="A" else model.expectation.logits(batch["scene"],batch["category"])
                valid.append((float(count_loss(logits,batch["count_bin"],config.crps_weight)),len(batch["id"])))
        score = sum(v*n for v,n in valid)/sum(n for _,n in valid)
        history.append({"epoch":epoch+1,"training_loss":sum(v*n for v,n in losses)/sum(n for _,n in losses),"validation_loss":score})
        if score<best:
            best,stale = score,0
            save_checkpoint(output,module,config,stage,{**provenance,"selection_metric":"validation_count_nll_plus_crps",
                "selection_score":score,"epoch":epoch+1,"manifest_sha256":file_hash(manifest_path),
                "train_ids":[r["id"] for r in train_rows],"validation_ids":[r["id"] for r in validation_rows]})
        else:
            stale += 1
        if stale>=config.patience:
            break
    save_run_record(str(output)+".run.json",{"stage":stage,"config":asdict(config),"config_hash":config.digest(),
        "checkpoint_sha256":file_hash(output),"history":history,**provenance})

def make_optimizer(parameters,config):
    if config.optimizer=="adamw":
        return torch.optim.AdamW(parameters,lr=config.learning_rate,weight_decay=config.weight_decay)
    if config.optimizer=="adam":
        return torch.optim.Adam(parameters,lr=config.learning_rate,weight_decay=config.weight_decay)
    raise ValueError("Unknown reference optimizer")

def fit_decoder(rows,manifest_path,feature_root,config,fold,output,observation_checkpoint=None,expectation_checkpoint=None,gt_observation=False):
    seed_everything(config.seed)
    validate_fold(rows,fold)
    train_rows,validation_rows = select_split(rows,fold,"train"),select_split(rows,fold,"validation")
    model = GSDVQ(config).to(config.device)
    checkpoint_sources = {}
    if config.comparison!="frozen_mlp":
        if not observation_checkpoint or not expectation_checkpoint:
            raise ValueError("Stage C needs frozen Stage A and B checkpoints")
        for name,path,stage in (("observation",observation_checkpoint,"A"),("expectation",expectation_checkpoint,"B")):
            load_checkpoint(path,getattr(model,name),config.device,stage,config.checkpoint_policy=="strict")
            checkpoint_sources[name] = file_hash(path)
    model.freeze_distributions()
    for name,param in model.named_parameters():
        if not name.startswith("decoder.") and param.requires_grad:
            raise AssertionError("VAQUUM optimization must not update count distributions")
    training = loader(FeatureDataset(train_rows,feature_root,config,"C",True,True,gt_observation),config,True)
    validation = loader(FeatureDataset(validation_rows,feature_root,config,"C",False,True,gt_observation),config)
    optimizer = make_optimizer(model.decoder.parameters(),config)
    best,stale,history = float("-inf"),0,[]
    for epoch in range(config.stage_c_epochs):
        model.decoder.train()
        total_nll,total_responses=0.,0
        for batch in training:
            batch = to_device(batch,config.device)
            result = model(batch,batch["count_bin"] if gt_observation else None)
            loss = participant_beta_nll(result["mu"],result["kappa"],batch["ratings"],batch["valid"],config.rating_epsilon)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite beta NLL")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            responses = int(batch["valid"].sum())
            total_nll += float(loss.detach())*responses
            total_responses += responses
        model.decoder.eval()
        predictions,labels = [],[]
        with torch.no_grad():
            for batch in validation:
                batch = to_device(batch,config.device)
                result = model(batch,batch["count_bin"] if gt_observation else None)
                for i,identifier in enumerate(batch["id"]):
                    predictions.append({"id":identifier,"scores":{q:float(result["mu"][i,j]) for j,q in enumerate(QUANTIFIERS)}})
                    labels.append({"id":identifier,"ratings":{q:batch["ratings"][i,j][batch["valid"][i,j]].cpu().tolist() for j,q in enumerate(QUANTIFIERS)}})
        report = rating_report(predictions,labels)
        score = report["macro_r"]
        if score is None:
            raise ValueError("Validation categories must support defined correlations for all five quantifiers")
        history.append({"epoch":epoch+1,"participant_beta_nll":total_nll/total_responses,"validation":report})
        if score>best:
            best,stale = score,0
            save_checkpoint(output,model,config,"C",{"exposure_policy":config.checkpoint_policy,"fold":fold,
                "epoch":epoch+1,"selection_metric":"validation_macro_pearson","selection_score":score,
                "manifest_sha256":file_hash(manifest_path),"distribution_checkpoints":checkpoint_sources,
                "gt_observation_diagnostic":gt_observation})
        else:
            stale += 1
        if stale>=config.patience:
            break
    save_run_record(str(output)+".run.json",{"stage":"C","config":asdict(config),"config_hash":config.digest(),
        "checkpoint_sha256":file_hash(output),"fold":fold,"history":history,"distribution_checkpoints":checkpoint_sources})
