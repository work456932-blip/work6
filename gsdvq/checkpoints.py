from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch

def file_hash(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def save_checkpoint(path,module,config,stage,metadata):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":module.state_dict(),"config":asdict(config),"stage":stage,"metadata":metadata},path)
    return file_hash(path)

def load_checkpoint(path,module,device="cpu",expected_stage=None,strict_policy=False):
    payload = torch.load(path,map_location=device,weights_only=True)
    if expected_stage and payload["stage"]!=expected_stage:
        raise ValueError("Unexpected checkpoint stage")
    if strict_policy and payload.get("metadata",{}).get("exposure_policy")!="strict":
        raise ValueError("Primary pipeline cannot use an exposure-uncertain checkpoint")
    module.load_state_dict(payload["state_dict"],strict=True)
    return payload

def save_run_record(path,record):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
