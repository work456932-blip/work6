"""Local feature/annotation interfaces; no downloader and no encoder fine-tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from . import QUANTIFIERS
from .support import count_bin

def read_jsonl(path):
    with open(path,encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [str(r["id"]) for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("Use one record per image-object pair; keep its judgments together")
    return rows

def write_jsonl(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")

def category_folds(rows,folds=5,seed=2026,validation_fraction=.2):
    categories = sorted({r["category"] for r in rows})
    if len(categories)<folds+2:
        raise ValueError("Need enough categories for distinct train/validation/test sets")
    rng = np.random.default_rng(seed)
    rng.shuffle(categories)
    test_sets = [set(a.tolist()) for a in np.array_split(np.array(categories),folds)]
    result = []
    for k,test in enumerate(test_sets):
        available = sorted(set(categories)-test)
        np.random.default_rng(seed+k+1).shuffle(available)
        nval = max(1,int(round(len(available)*validation_fraction)))
        if nval>=len(available):
            raise ValueError("Validation fraction leaves no training categories")
        validation = set(available[:nval])
        training = set(available[nval:])
        result.append({"fold":k,"train_categories":sorted(training),"validation_categories":sorted(validation),
                       "test_categories":sorted(test),"split_seed":seed})
    return result

def image_folds(rows,folds=5,seed=2026,validation_fraction=.2):
    # Secondary diagnostic only; all categories in an image share its split.
    proxy = [{"category":r["image_id"]} for r in rows]
    category = category_folds(proxy,folds,seed,validation_fraction)
    return [{k.replace("categories","images"):v for k,v in f.items()} for f in category]

def select_split(rows,fold,part):
    key = {"train":"train_categories","validation":"validation_categories","test":"test_categories"}[part]
    if key in fold:
        selected = set(fold[key])
        return [r for r in rows if r["category"] in selected]
    key = key.replace("categories","images")
    selected = set(fold[key])
    return [r for r in rows if r["image_id"] in selected]

def validate_fold(rows,fold):
    suffix = "categories" if "train_categories" in fold else "images"
    field = "category" if suffix=="categories" else "image_id"
    groups = [set(fold[p+"_"+suffix]) for p in ("train","validation","test")]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Train, validation and test partitions overlap")
    if set.union(*groups)!={r[field] for r in rows}:
        raise ValueError("Fold membership does not match the supplied data snapshot")

class FeatureDataset(Dataset):
    def __init__(self,rows,root,config,stage="C",training=False,supervised=False,gt_diagnostic=False,count_supervision=True):
        self.rows,self.root,self.config = rows,Path(root),config
        self.stage,self.training,self.supervised,self.gt_diagnostic = stage,training,supervised,gt_diagnostic
        self.count_supervision=count_supervision
        self.rng = np.random.default_rng(config.seed)
        allowed = {"A":"counting","B":"expectation","C":"vaquum","E":"human_expectation_validation"}[stage]
        if any(r.get("role")!=allowed for r in rows):
            raise ValueError("Stage "+stage+" requires role="+allowed+"; human expectation judgments never train this pipeline")
        if not rows:
            raise ValueError("Empty dataset split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        # Only named feature arrays enter the model; no pickled objects or metadata.
        with np.load(self.root/row["feature_file"],allow_pickle=False) as cache:
            item = {"id":str(row["id"]),"image_id":str(row["image_id"]),"category_name":row["category"]}
            def feature(name,dim,choose_variant=False):
                array = np.asarray(cache[name],dtype=np.float32)
                if choose_variant and array.ndim==2:
                    variant = self.rng.integers(len(array)) if self.training else 0
                    array = array[variant]
                if array.shape!=(dim,) or not np.isfinite(array).all():
                    raise ValueError("Invalid feature shape/value for "+name)
                return torch.from_numpy(array.copy())
            if self.stage=="A" or (self.stage=="C" and self.config.comparison!="frozen_mlp"):
                item["counting"] = feature("counting",self.config.counting_dim)
            if self.stage in {"B","E"} or (self.stage=="C" and self.config.comparison!="frozen_mlp"):
                operator = (self.rng.choice(self.config.operators,p=self.config.operator_probabilities)
                            if self.stage=="B" and self.training else self.config.inference_operator)
                name = "scene_validation" if self.stage=="E" and "scene_validation" in cache else "scene_"+self.config.feature_variant+"_"+operator
                item["scene"] = feature(name,self.config.scene_dim,True)
                item["category"] = feature("category",self.config.category_dim)
                item["mask_operator"] = operator
            if self.stage=="C" and self.config.comparison=="frozen_mlp":
                item["joint"] = feature("joint",self.config.joint_dim)
        if (self.stage in {"A","B"} and self.count_supervision) or self.gt_diagnostic:
            if isinstance(row["count"],bool) or int(row["count"])!=row["count"] or row["count"]<0:
                raise ValueError("Counts must be nonnegative integers")
            item["count_bin"] = count_bin(row["count"],self.config.nmax)
            item["count"] = float(row["count"])
        if self.stage=="C" and self.supervised:
            ratings = row["ratings"]
            item["ratings"] = []
            for quantifier in QUANTIFIERS:
                values = [float(v) for v in ratings.get(quantifier,[]) if v is not None]
                if any(not np.isfinite(v) or not 0<=v<=1 for v in values):
                    raise ValueError("Normalize participant ratings to [0,1]")
                item["ratings"].append(values)
            if not any(item["ratings"]):
                raise ValueError("No participant responses for this item")
        return item

def collate(items):
    batch = {}
    for key in ("counting","scene","category","joint"):
        if key in items[0]:
            batch[key] = torch.stack([r[key] for r in items])
    for key in ("id","image_id","category_name","mask_operator"):
        if key in items[0]:
            batch[key] = [r[key] for r in items]
    if "count_bin" in items[0]:
        batch["count_bin"] = torch.tensor([r["count_bin"] for r in items],dtype=torch.long)
        batch["count"] = torch.tensor([r["count"] for r in items],dtype=torch.float32)
    if "ratings" in items[0]:
        width = max(1,max(len(v) for r in items for v in r["ratings"]))
        ratings = torch.zeros(len(items),5,width)
        valid = torch.zeros_like(ratings,dtype=torch.bool)
        for i,r in enumerate(items):
            for q,values in enumerate(r["ratings"]):
                ratings[i,q,:len(values)] = torch.tensor(values)
                valid[i,q,:len(values)] = True
        batch.update(ratings=ratings,valid=valid)
    return batch

def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
