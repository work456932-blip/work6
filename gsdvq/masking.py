"""Local target removal and Appendix E fixed-component geometry decoupling."""
from dataclasses import asdict,dataclass
import importlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from scipy.ndimage import binary_dilation,label

def read_instance_masks(path,size):
    masks = np.load(path,allow_pickle=False).astype(bool)
    if masks.ndim!=3 or masks.shape[1:]!=(size[1],size[0]):
        raise ValueError("Expected instance mask array [instances,height,width] matching RGB image")
    if any(not mask.any() for mask in masks):
        raise ValueError("Empty instance mask")
    return masks

def components(mask):
    return int(label(mask)[1])

def boundary_perturbation(mask,rng,max_dilation=3):
    # Outward perturbation retains full target coverage; erosion could expose targets.
    radius = int(rng.integers(max_dilation+1))
    return binary_dilation(mask,iterations=radius) if radius else mask.copy()

def remove_targets(image,mask,operator,inpaint=None,seed=0):
    rgb = np.asarray(image.convert("RGB")).copy()
    if mask.shape!=rgb.shape[:2]:
        raise ValueError("Mask/image shapes differ")
    if operator=="gray":
        rgb[mask]=(127,127,127)
    elif operator=="blur":
        blurred = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=15)))
        rgb[mask]=blurred[mask]
    elif operator=="inpaint":
        if inpaint is None:
            raise ValueError("Supply a real local inpainting adapter; blur is not relabeled inpainting")
        painted = np.asarray(inpaint(image.convert("RGB"),mask.copy(),seed=seed).convert("RGB"))
        if painted.shape!=rgb.shape:
            raise ValueError("Inpainting changed image shape")
        # Prevent the inpainting service from changing pixels outside authorized regions.
        rgb[mask]=painted[mask]
    else:
        raise ValueError("Unknown removal operator")
    return Image.fromarray(rgb)

@dataclass
class FixedKPlan:
    k: int
    areas: list[float]
    aspects: list[float]
    quantile: float = .99
    train_manifest_sha256: str = ""

    @classmethod
    def fit(cls,training_masks,quantile=.99,manifest_hash=""):
        counts,areas,aspects = [],[],[]
        for masks in training_masks:
            counts.append(len(masks))
            for mask in masks:
                ys,xs = np.nonzero(mask)
                if len(xs):
                    areas.append(float(mask.mean()))
                    aspects.append(float((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)))
        if not counts or not areas:
            raise ValueError("Fixed-K plan needs training-only real target-mask statistics")
        return cls(int(np.ceil(np.quantile(counts,quantile))),areas,aspects,quantile,manifest_hash)

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2)+"\n")

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))

def fixed_k_mask(target_mask,foreground_mask,plan,rng,max_attempts=5000):
    """Add separated decoys until the UNION has exactly K connected components.

    All target regions are kept. If an example is over K or lacks feasible
    background space, fail explicitly; never shrink/remove targets to meet K.
    """
    union = target_mask.copy()
    if foreground_mask is None:
        raise ValueError("Fixed-K requires foreground exclusion masks")
    if foreground_mask.shape!=union.shape:
        raise ValueError("Foreground and target masks must share image geometry")
    n = components(union)
    if n>plan.k:
        raise ValueError("True removal already exceeds frozen K; restrict all compared variants to the same eligible cohort")
    height,width = union.shape
    yy,xx = np.ogrid[:height,:width]
    attempts=0
    while n<plan.k and attempts<max_attempts:
        attempts+=1
        index = int(rng.integers(len(plan.areas)))
        area,aspect = plan.areas[index]*height*width,plan.aspects[index]
        rx = np.sqrt(area*aspect/np.pi)
        ry = np.sqrt(area/(aspect*np.pi))
        if rx>=width/2 or ry>=height/2 or rx<1 or ry<1:
            continue
        x,y = rng.uniform(rx,width-rx),rng.uniform(ry,height-ry)
        decoy = ((xx-x)/rx)**2+((yy-y)/ry)**2<=1
        exclusion = binary_dilation(union|foreground_mask,iterations=1)
        if np.any(decoy & exclusion) or components(decoy)!=1:
            continue
        candidate = union|decoy
        if components(candidate)!=n+1:
            continue
        union,n = candidate,n+1
    if n!=plan.k:
        raise ValueError("Unable to place exactly K disjoint, foreground-safe decoys")
    return union

def mask_geometry(mask):
    return {"components":components(mask),"area_fraction":float(mask.mean())}
