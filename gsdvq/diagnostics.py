"""Incomparability and artifact diagnostics; all numbers come from supplied inputs."""
import numpy as np
from PIL import Image
from .metrics import correlation,partial_spearman
from .masking import mask_geometry

def geometry_probe(train_masks,train_counts,test_masks,test_counts,ridge=1.):
    def features(masks):
        rows=[]
        for mask in masks:
            small=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((32,32)),float).reshape(-1)/255
            rows.append(small)
        return np.asarray(rows)
    x,z = features(train_masks),features(test_masks)
    y,truth = np.asarray(train_counts,float),np.asarray(test_counts,float)
    center,scale = x.mean(0),x.std(0)+1e-6
    x,z=(x-center)/scale,(z-center)/scale
    x,z=np.c_[np.ones(len(x)),x],np.c_[np.ones(len(z)),z]
    penalty=np.eye(x.shape[1])*ridge
    penalty[0,0]=0.
    weights=np.linalg.solve(x.T@x+penalty,x.T@y)
    predictions=z@weights
    geometry=[mask_geometry(m) for m in test_masks]
    return {"component_count_r":correlation([r["components"] for r in geometry],truth),
            "mask_area_r":correlation([r["area_fraction"] for r in geometry],truth),
            "mask_only_probe_r":correlation(predictions,truth),"mask_only_probe_rho":correlation(predictions,truth,True),
            "mask_only_probe_mae":float(np.mean(np.abs(predictions-truth)))}

def disagreement_report(predictions,labels,training_category_frequency,rating_bins=10):
    pred={str(r["id"]):r for r in predictions}
    labels={str(r["id"]):r for r in labels}
    if set(pred)!=set(labels):
        raise ValueError("Disagreement requires aligned image-object IDs")
    records=[]
    for identifier,p in pred.items():
        row=labels[identifier]
        variances,entropies,errors=[],[],[]
        for quantifier,values in row["ratings"].items():
            if not values:
                continue
            variances.append(float(np.var(values)))
            histogram,_=np.histogram(values,bins=rating_bins,range=(0,1))
            probabilities=histogram/histogram.sum()
            positive=probabilities[probabilities>0]
            entropies.append(float(-np.sum(positive*np.log(positive))))
            errors.append(abs(p["scores"][quantifier]-np.mean(values)))
        probabilities=np.asarray(p["pobs"])
        positive=probabilities[probabilities>0]
        records.append({"ainc":p["a_inc"],"variance":np.mean(variances),"entropy":np.mean(entropies),"mae":np.mean(errors),
            "controls":[abs(p["observed_mean"]-p["expected_mean"]),-np.sum(positive*np.log(positive)),
                        p["observed_mean"],training_category_frequency.get(row["category"],0)]})
    order=np.argsort([r["ainc"] for r in records],kind="stable")
    groups=[]
    for name,indices in zip(("low","middle","high"),np.array_split(order,3)):
        groups.append({"group":name,"n":len(indices),**{key:float(np.mean([records[i][key] for i in indices])) for key in ("variance","entropy","mae")}})
    controls=np.asarray([r["controls"] for r in records])
    partial={key:partial_spearman([r["ainc"] for r in records],[r[key] for r in records],controls) for key in ("variance","entropy","mae")}
    return {"groups":groups,"partial_spearman":partial,"rating_entropy_bins":rating_bins}
