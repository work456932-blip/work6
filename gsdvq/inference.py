"""Write predictions without reading ratings or independent expectation judgments."""
from dataclasses import asdict,replace
from pathlib import Path
import torch
from . import QUANTIFIERS
from .config import Config
from .model import GSDVQ
from .data import FeatureDataset,collate,to_device,write_jsonl
from .checkpoints import load_checkpoint,file_hash
from .training import loader,seed_everything

def predict(rows,feature_root,checkpoint,device="cpu",gt_diagnostic=False,mask_operator=None):
    # Only a locally trusted generated checkpoint should be loaded.
    payload = torch.load(checkpoint,map_location=device,weights_only=True)
    config = Config(**{**payload["config"],"device":device})
    if mask_operator is not None:
        config=replace(config,inference_operator=mask_operator)
    if payload["stage"]!="C":
        raise ValueError("Full prediction requires a Stage C checkpoint")
    if bool(payload["metadata"].get("gt_observation_diagnostic",False))!=gt_diagnostic:
        raise ValueError("Prediction mode must match the checkpoint's GT diagnostic setting")
    seed_everything(config.seed)
    model = GSDVQ(config).to(device)
    model.load_state_dict(payload["state_dict"],strict=True)
    model.freeze_distributions()
    model.eval()
    data = FeatureDataset(rows,feature_root,config,"C",False,False,gt_diagnostic)
    out=[]
    checkpoint_digest = file_hash(checkpoint)
    with torch.no_grad():
        for batch in loader(data,config):
            batch = to_device(batch,device)
            result = model(batch,batch["count_bin"] if gt_diagnostic else None)
            for i,identifier in enumerate(batch["id"]):
                row = {"id":identifier,"image_id":batch["image_id"][i],"category":batch["category_name"][i],
                    "scores":{q:float(result["mu"][i,j]) for j,q in enumerate(QUANTIFIERS)},
                    "concentration":{q:float(result["kappa"][j]) for j,q in enumerate(QUANTIFIERS)},
                    "seed":config.seed,"config_hash":config.digest(),"checkpoint_sha256":checkpoint_digest,
                    "fold":payload["metadata"].get("fold",{}).get("fold"),
                    "exposure_policy":config.checkpoint_policy,"gt_observation_diagnostic":gt_diagnostic}
                for key in ("pobs","pexp","features"):
                    if key in result:
                        row[key] = result[key][i].cpu().tolist()
                for key in ("d_plus","d_minus","s_dom","a_inc","observed_mean","expected_mean"):
                    if key in result:
                        row[key] = float(result[key][i])
                out.append(row)
    return out

def predict_counts(rows,feature_root,checkpoint,device="cpu",mask_operator=None):
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    stage=payload["stage"]
    if stage not in {"A","B"}:
        raise ValueError("Count prediction requires a Stage A or B checkpoint")
    config=Config(**{**payload["config"],"device":device})
    if mask_operator is not None:
        config=replace(config,inference_operator=mask_operator)
    model=GSDVQ(config).to(device)
    module=model.observation if stage=="A" else model.expectation
    module.load_state_dict(payload["state_dict"],strict=True)
    module.eval()
    dataset=FeatureDataset(rows,feature_root,config,stage,False,False,count_supervision=False)
    out=[]
    with torch.no_grad():
        for batch in loader(dataset,config):
            batch=to_device(batch,device)
            p=module(batch["counting"]) if stage=="A" else module(batch["scene"],batch["category"])
            out.extend({"id":identifier,"pmf":p[i].cpu().tolist(),"overflow_value":config.overflow_value}
                       for i,identifier in enumerate(batch["id"]))
    return out

def predict_expectations(rows,feature_root,checkpoint,device="cpu",mask_operator=None):
    payload = torch.load(checkpoint,map_location=device,weights_only=True)
    config = Config(**{**payload["config"],"device":device})
    if mask_operator is not None:
        config=replace(config,inference_operator=mask_operator)
    model = GSDVQ(config).to(device)
    load_checkpoint(checkpoint,model.expectation,device,"B",config.checkpoint_policy=="strict")
    model.expectation.eval()
    data = FeatureDataset(rows,feature_root,config,"E",False,False)
    out=[]
    with torch.no_grad():
        for batch in loader(data,config):
            batch=to_device(batch,device)
            p=model.expectation(batch["scene"],batch["category"])
            for i,identifier in enumerate(batch["id"]):
                out.append({"id":identifier,"pexp":p[i].cpu().tolist(),"checkpoint_sha256":file_hash(checkpoint)})
    return out
