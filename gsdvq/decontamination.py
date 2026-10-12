"""Sequential source-ID / SHA256 / pHash filtering with mutually exclusive counts."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn

def canonical_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out,format="PNG")
        return im.copy(),out.getvalue()

def hashes(path):
    image,canonical = canonical_image(path)
    # Canonical resizing and low-frequency DCT provide an explicit pHash convention.
    small = np.asarray(image.convert("L").resize((32,32)),dtype=float)
    low = dctn(small,type=2,norm="ortho")[:8,:8].reshape(-1)
    threshold = np.median(low[1:])
    bits = low>threshold
    bits[0]=False
    phash = sum(int(v)<<i for i,v in enumerate(bits))
    return hashlib.sha256(canonical).hexdigest(),phash

def source_keys(row):
    # Namespace-qualified provenance IDs prevent accidental cross-source collisions.
    return set(str(v) for v in row.get("source_ids",[]))

def decontaminate(auxiliary,vaquum,image_root,threshold=6,review=None):
    root = Path(image_root)
    known_sources,known_sha,known_perceptual = set(),set(),[]
    for r in vaquum:
        known_sources |= source_keys(r)
        sha,phash = hashes(root/r["image_file"])
        known_sha.add(sha)
        known_perceptual.append((str(r["image_id"]),phash))
    # One filtering decision per auxiliary image, propagated to all category records.
    images = {}
    for row in auxiliary:
        key = (row.get("dataset",""),str(row["image_id"]))
        images.setdefault(key,[]).append(row)
    retained,decisions=[],[]
    for key,rows in images.items():
        source = set().union(*(source_keys(r) for r in rows))
        sha,phash = hashes(root/rows[0]["image_file"])
        decision = {"dataset":key[0],"image_id":key[1],"sha256":sha,"phash":str(phash),"record_ids":[str(r["id"]) for r in rows]}
        if source & known_sources:
            reason = "source_or_exact"
        elif sha in known_sha:
            reason = "source_or_exact"
        else:
            neighbors = [{"image_id":i,"hamming":(phash^p).bit_count()} for i,p in known_perceptual if (phash^p).bit_count()<=threshold]
            decision["near_candidates"] = neighbors
            # Conservative default; optional reviewed false-positive decisions are explicit.
            remove = bool(neighbors)
            if neighbors and review is not None:
                review_key = key[0]+":"+key[1]
                if review_key in review:
                    remove = bool(review[review_key]["duplicate"])
                    decision["review"] = review[review_key]
            reason = "near" if remove else "retained"
        decision["reason"] = reason
        decisions.append(decision)
        if reason=="retained":
            retained.extend(rows)
    summaries={}
    for d in decisions:
        s = summaries.setdefault(d["dataset"],{"initial_images":0,"source_or_exact":0,"near":0,"final_images":0})
        s["initial_images"]+=1
        s["final_images" if d["reason"]=="retained" else d["reason"]]+=1
    for s in summaries.values():
        assert s["final_images"]==s["initial_images"]-s["source_or_exact"]-s["near"]
    return retained,{"status":"completed","per_source":summaries,"decisions":decisions,
                     "retained_ids":[str(r["id"]) for r in retained],"phash_hamming_threshold":threshold,
                     "backbone_exposure":"unresolved"}
