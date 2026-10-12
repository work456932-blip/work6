"""Prepare features from supplied local images/masks; encoder adapters stay frozen."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .masking import read_instance_masks,boundary_perturbation,remove_targets,fixed_k_mask,mask_geometry
from .feature_interfaces import load_factory

def prepare_features(rows,image_root,output_root,encoder,inpaint=None,variants=3,seed=42,fixed_k_plan=None):
    root,output = Path(image_root),Path(output_root)
    output.mkdir(parents=True,exist_ok=True)
    manifests,geometry=[] ,[]
    for row in rows:
        identifier = str(row["id"])
        digest = hashlib.sha256(identifier.encode()).hexdigest()[:20]
        rng = np.random.default_rng(int(digest[:8],16)+seed)
        with Image.open(root/row["image_file"]) as im:
            image = im.convert("RGB")
        features={}
        if row["role"] in {"counting","vaquum"}:
            features["counting"] = np.asarray(encoder.counting(image,row["category"]),np.float32)
        if row["role"]=="vaquum":
            features["joint"] = np.asarray(encoder.joint(image,row["category"]),np.float32)
        if row["role"]=="human_expectation_validation" and row.get("image_is_target_removed",False):
            features["category"] = np.asarray(encoder.category(row["category"]),np.float32)
            features["scene_validation"] = np.asarray(encoder.scene(image),np.float32)
        elif row["role"] in {"expectation","vaquum","human_expectation_validation"}:
            masks = read_instance_masks(root/row["mask_file"],image.size)
            if row["role"]=="expectation" and int(row["count"])!=len(masks):
                raise ValueError("Expectation count must be the number of original target instances")
            target = masks.any(axis=0) if len(masks) else np.zeros((image.height,image.width),bool)
            features["category"] = np.asarray(encoder.category(row["category"]),np.float32)
            foreground = None
            if fixed_k_plan:
                foreground = np.load(root/row["foreground_mask_file"],allow_pickle=False).astype(bool)
            for operator in ("gray","blur","inpaint"):
                scene=[]
                for variant in range(variants):
                    mask = boundary_perturbation(target,rng)
                    if fixed_k_plan:
                        # Appendix E: true targets + decoys are JOINTLY inpainted.
                        if operator!="inpaint":
                            continue
                        mask = fixed_k_mask(mask,foreground,fixed_k_plan,rng)
                    painted = remove_targets(image,mask,operator,inpaint,seed+variant)
                    scene.append(np.asarray(encoder.scene(painted),np.float32))
                    geometry.append({"id":identifier,"operator":operator,"variant":variant,
                                     "feature_variant":"fixed_k" if fixed_k_plan else "primary",**mask_geometry(mask)})
                if scene:
                    features["scene_"+("fixed_k" if fixed_k_plan else "primary")+"_"+operator]=np.stack(scene)
        filename = digest+".npz"
        # Separate output directories keep primary and fixed-K caches immutable.
        path = output/filename
        if path.exists():
            raise FileExistsError("Use a fresh feature directory for each data/encoder snapshot")
        np.savez_compressed(path,**features)
        manifests.append({**row,"feature_file":filename})
    (output/"encoder_provenance.json").write_text(json.dumps(encoder.provenance,indent=2)+"\n")
    (output/"mask_geometry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in geometry))
    return manifests
