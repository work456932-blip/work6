"""CLI keeps dataset/model integration explicit and never downloads resources."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import Config

def read_rows(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description="Anonymous GSD-VQ staged training and evaluation")
    sub=parser.add_subparsers(dest="command",required=True)
    folds=sub.add_parser("folds")
    folds.add_argument("--data",required=True)
    folds.add_argument("--config",required=True)
    folds.add_argument("--image-disjoint",action="store_true")
    folds.add_argument("--output",required=True)
    audit=sub.add_parser("audit")
    audit.add_argument("--auxiliary",required=True)
    audit.add_argument("--vaquum",required=True)
    audit.add_argument("--image-root",required=True)
    audit.add_argument("--phash-threshold",type=int,default=6)
    audit.add_argument("--review")
    audit.add_argument("--backbone-provenance")
    audit.add_argument("--filtered",required=True)
    audit.add_argument("--output",required=True)
    prepare=sub.add_parser("prepare")
    prepare.add_argument("--data",required=True)
    prepare.add_argument("--image-root",required=True)
    prepare.add_argument("--encoder-factory",required=True)
    prepare.add_argument("--encoder-kwargs",default="{}")
    prepare.add_argument("--inpaint-factory")
    prepare.add_argument("--inpaint-kwargs",default="{}")
    prepare.add_argument("--fixed-k-plan")
    prepare.add_argument("--variants",type=int,default=3)
    prepare.add_argument("--seed",type=int,default=42)
    prepare.add_argument("--output-root",required=True)
    prepare.add_argument("--manifest",required=True)
    fixed=sub.add_parser("fixed-k-plan")
    fixed.add_argument("--training-data",required=True)
    fixed.add_argument("--image-root",required=True)
    fixed.add_argument("--quantile",type=float,default=.99)
    fixed.add_argument("--output",required=True)
    train=sub.add_parser("train")
    train.add_argument("--stage",choices=["A","B","C"],required=True)
    train.add_argument("--config",required=True)
    train.add_argument("--data",required=True)
    train.add_argument("--feature-root",required=True)
    train.add_argument("--audit-record")
    train.add_argument("--folds")
    train.add_argument("--fold",type=int,default=0)
    train.add_argument("--seed",type=int)
    train.add_argument("--observation-checkpoint")
    train.add_argument("--expectation-checkpoint")
    train.add_argument("--gt-observation-diagnostic",action="store_true")
    train.add_argument("--output",required=True)
    predict=sub.add_parser("predict")
    predict.add_argument("--data",required=True)
    predict.add_argument("--feature-root",required=True)
    predict.add_argument("--checkpoint",required=True)
    predict.add_argument("--device",default="cpu")
    predict.add_argument("--mask-operator",choices=["gray","blur","inpaint"])
    predict.add_argument("--folds")
    predict.add_argument("--fold",type=int,default=0)
    predict.add_argument("--expectation-only",action="store_true")
    predict.add_argument("--counting-only",action="store_true")
    predict.add_argument("--gt-observation-diagnostic",action="store_true")
    predict.add_argument("--output",required=True)
    evaluate=sub.add_parser("evaluate")
    evaluate.add_argument("--predictions",required=True)
    evaluate.add_argument("--labels",required=True)
    evaluate.add_argument("--comparison")
    evaluate.add_argument("--paradigm",choices=["rating","production","multiple_choice","expectation","counting"],default="rating")
    evaluate.add_argument("--bootstrap-repeats",type=int,default=2000)
    evaluate.add_argument("--permutation-repeats",type=int,default=5000)
    evaluate.add_argument("--output",required=True)
    merge=sub.add_parser("merge-folds")
    merge.add_argument("--predictions",nargs=5,required=True)
    merge.add_argument("--output",required=True)
    summary=sub.add_parser("summarize-seeds")
    summary.add_argument("--reports",nargs=3,required=True)
    summary.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="folds":
        from .data import category_folds,image_folds
        config=Config.load(args.config)
        function=image_folds if args.image_disjoint else category_folds
        result=function(read_rows(args.data),config.outer_folds,config.fold_seed,config.validation_category_fraction)
        write_json(args.output,result)
    elif args.command=="audit":
        from .decontamination import decontaminate
        from .checkpoints import file_hash
        from .data import write_jsonl
        review=json.loads(Path(args.review).read_text()) if args.review else None
        rows,record=decontaminate(read_rows(args.auxiliary),read_rows(args.vaquum),args.image_root,args.phash_threshold,review)
        write_jsonl(args.filtered,rows)
        record["filtered_manifest_sha256"]=file_hash(args.filtered)
        if args.backbone_provenance:
            provenance=json.loads(Path(args.backbone_provenance).read_text())
            if provenance.get("exposure_audit_status")!="audited" or not provenance.get("checkpoint_sha256"):
                raise ValueError("Supply a completed external backbone-pretraining exposure audit")
            record["backbone_exposure"]="audited"
            record["backbone_provenance"]=provenance
        write_json(args.output,record)
    elif args.command=="fixed-k-plan":
        from PIL import Image
        from .masking import FixedKPlan,read_instance_masks
        rows=read_rows(args.training_data)
        if any(r.get("partition")!="train" or r.get("role")!="expectation" for r in rows):
            raise ValueError("K and decoy shapes must be selected from Stage B training data only")
        masks=[]
        for r in rows:
            with Image.open(Path(args.image_root)/r["image_file"]) as image:
                masks.append(read_instance_masks(Path(args.image_root)/r["mask_file"],image.size))
        digest=hashlib.sha256(Path(args.training_data).read_bytes()).hexdigest()
        FixedKPlan.fit(masks,args.quantile,digest).save(args.output)
    elif args.command=="prepare":
        from .feature_interfaces import load_factory
        from .masking import FixedKPlan
        from .prepare import prepare_features
        from .data import write_jsonl
        encoder=load_factory(args.encoder_factory,json.loads(args.encoder_kwargs))
        painter=load_factory(args.inpaint_factory,json.loads(args.inpaint_kwargs)) if args.inpaint_factory else None
        plan=FixedKPlan.load(args.fixed_k_plan) if args.fixed_k_plan else None
        rows=prepare_features(read_rows(args.data),args.image_root,args.output_root,encoder,painter,args.variants,args.seed,plan)
        write_jsonl(args.manifest,rows)
    elif args.command=="train":
        from .training import fit_auxiliary,fit_decoder
        config=Config.load(args.config)
        if args.seed is not None:
            config=replace(config,seed=args.seed)
        if args.stage in {"A","B"}:
            fit_auxiliary(read_rows(args.data),args.data,args.feature_root,config,args.stage,args.output,args.audit_record)
        else:
            if not args.folds:
                raise ValueError("Stage C requires explicit category/image folds")
            fold=json.loads(Path(args.folds).read_text())[args.fold]
            fit_decoder(read_rows(args.data),args.data,args.feature_root,config,fold,args.output,
                        args.observation_checkpoint,args.expectation_checkpoint,args.gt_observation_diagnostic)
    elif args.command=="predict":
        from .inference import predict as run_predict,predict_expectations,predict_counts
        from .data import select_split,write_jsonl,validate_fold
        if Path(args.output).exists():
            raise FileExistsError("Use a new prediction path to preserve outputs")
        rows=read_rows(args.data)
        if args.folds:
            fold=json.loads(Path(args.folds).read_text())[args.fold]
            validate_fold(rows,fold)
            rows=select_split(rows,fold,"test")
        if args.expectation_only and args.counting_only:
            raise ValueError("Choose one distribution evaluation mode")
        result=(predict_counts(rows,args.feature_root,args.checkpoint,args.device,args.mask_operator) if args.counting_only else
                predict_expectations(rows,args.feature_root,args.checkpoint,args.device,args.mask_operator) if args.expectation_only else
                run_predict(rows,args.feature_root,args.checkpoint,args.device,args.gt_observation_diagnostic,args.mask_operator))
        write_jsonl(args.output,result)
    elif args.command=="evaluate":
        from .metrics import rating_report,image_bootstrap,paired_permutation,transfer_report,expected_count_report,count_report
        predictions,labels=read_rows(args.predictions),read_rows(args.labels)
        if args.paradigm=="rating":
            report=rating_report(predictions,labels)
            report["macro_r_ci95"]=image_bootstrap(predictions,labels,args.bootstrap_repeats)
            if args.comparison:
                report["paired_permutation"]=paired_permutation(predictions,read_rows(args.comparison),labels,args.permutation_repeats)
        elif args.paradigm=="expectation":
            report=expected_count_report(predictions,labels)
        elif args.paradigm=="counting":
            by_id={str(r["id"]):r for r in labels}
            report=count_report([{**r,"count":by_id[str(r["id"])]["count"]} for r in predictions])
        else:
            report=transfer_report(predictions,labels,args.paradigm)
        write_json(args.output,report)
    elif args.command=="merge-folds":
        from .data import write_jsonl
        rows=[r for path in args.predictions for r in read_rows(path)]
        if len({str(r["id"]) for r in rows})!=len(rows):
            raise ValueError("Out-of-fold files overlap in image-object items")
        if len({r["seed"] for r in rows})!=1 or len({r["config_hash"] for r in rows})!=1:
            raise ValueError("Merge five folds from the same seed and method configuration")
        write_jsonl(args.output,rows)
    else:
        reports=[json.loads(Path(p).read_text()) for p in args.reports]
        def summarize(values):
            if all(isinstance(v,dict) for v in values):
                return {k:summarize([v[k] for v in values]) for k in set.intersection(*(set(v) for v in values))}
            if all(isinstance(v,(int,float)) for v in values):
                return {"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1))}
            return values
        write_json(args.output,{"seeds":3,"statistics":summarize(reports)})

if __name__=="__main__":
    main()
