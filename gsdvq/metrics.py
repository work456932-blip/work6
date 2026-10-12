"""Participant aggregation and benchmark statistics; no hard-coded paper results."""
import numpy as np
from scipy.stats import pearsonr,spearmanr,rankdata
from . import QUANTIFIERS
from .support import empirical_pmf,normalize_pmf,support_values,count_bin
from .numerics import crps_point,wasserstein,js_divergence

def correlation(a,b,rank=False):
    a,b = np.asarray(a,float),np.asarray(b,float)
    keep = np.isfinite(a)&np.isfinite(b)
    a,b = a[keep],b[keep]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float((spearmanr if rank else pearsonr)(a,b).statistic)

def rating_report(predictions,labels):
    pred = {str(r["id"]):r for r in predictions}
    gold = {str(r["id"]):r for r in labels}
    if len(pred)!=len(predictions) or len(gold)!=len(labels) or set(pred)!=set(gold):
        raise ValueError("Rating predictions and labels must match item IDs exactly")
    ids = sorted(pred)
    means = np.array([[np.mean(gold[i]["ratings"].get(q,[])) if gold[i]["ratings"].get(q) else np.nan for q in QUANTIFIERS] for i in ids])
    estimates = np.array([[pred[i]["scores"][q] for q in QUANTIFIERS] for i in ids])
    prs = {q:correlation(estimates[:,j],means[:,j]) for j,q in enumerate(QUANTIFIERS)}
    sps = {q:correlation(estimates[:,j],means[:,j],True) for j,q in enumerate(QUANTIFIERS)}
    return {"n_items":len(ids),"per_quantifier_pearson":prs,"per_quantifier_spearman":sps,
            "macro_r":float(np.mean(list(prs.values()))) if all(v is not None for v in prs.values()) else None,
            "macro_rho":float(np.mean(list(sps.values()))) if all(v is not None for v in sps.values()) else None,
            "mae":float(np.nanmean(np.abs(estimates-means)))}

def count_report(rows,overflow_value=101.):
    p = np.array([r["pmf"] for r in rows])
    p = normalize_pmf(p)
    y = np.array([r["count"] for r in rows],dtype=float)
    mean = p@support_values(overflow_value=overflow_value)
    nll = [-np.log(max(p[i,count_bin(n)],1e-12)) for i,n in enumerate(y)]
    return {"n":len(rows),"mae":float(np.mean(np.abs(mean-y))),"rmse":float(np.sqrt(np.mean((mean-y)**2))),
            "nll":float(np.mean(nll)),"crps":float(np.mean([crps_point(p[i],n) for i,n in enumerate(y)])),
            "overflow_value":overflow_value}

def expected_count_report(predictions,human_rows,overflow_value=101.):
    pred = {str(r["id"]):r for r in predictions}
    if set(pred)!={str(r["id"]) for r in human_rows}:
        raise ValueError("Human expectation validation needs matching item IDs")
    crps,w1,model_means,human_means = [],[],[],[]
    for r in human_rows:
        counts = r["expected_counts"]
        if not counts:
            raise ValueError("No independent human expected-count judgments")
        p = normalize_pmf(pred[str(r["id"])]["pexp"])
        human = empirical_pmf(counts)
        human_mean = float(np.mean(counts))
        if human_mean<=0:
            raise ValueError("Item-wise normalization is undefined for zero human mean; report these separately")
        crps.append(np.mean([crps_point(p,n) for n in counts])/human_mean)
        w1.append(float(wasserstein(p,human,overflow_value))/human_mean)
        model_means.append(float(p@support_values(overflow_value=overflow_value)))
        human_means.append(human_mean)
    return {"n":len(human_rows),"ncrps":float(np.mean(crps)),"nw1":float(np.mean(w1)),
            "r_mu":correlation(model_means,human_means),"rho_mu":correlation(model_means,human_means,True)}

def preference_distribution(scores,eligible=None):
    # Protocol-specific normalization is a declared reference choice.
    values = np.asarray(scores,float)
    if eligible is not None:
        values = values*np.asarray(eligible,float)
    total = values.sum(-1,keepdims=True)
    if (total<=0).any():
        raise ValueError("Empty eligible answer set")
    return values/total

def transfer_report(predictions,labels,paradigm="multiple_choice"):
    pred = {str(r["id"]):r for r in predictions}
    if set(pred)!={str(r["id"]) for r in labels}:
        raise ValueError("Transfer inputs require matching IDs")
    predicted,human = [],[]
    for r in labels:
        scores = [pred[str(r["id"])]["scores"][q] for q in QUANTIFIERS]
        predicted.append(preference_distribution(scores,r.get("eligible")))
        observed = [r[paradigm].get(q,0.) for q in QUANTIFIERS]
        human.append(preference_distribution(observed,r.get("eligible")))
    a,b = np.array(predicted),np.array(human)
    per = {q:correlation(a[:,j],b[:,j]) for j,q in enumerate(QUANTIFIERS)}
    return {"per_quantifier_r":per,"macro_r":float(np.mean(list(per.values()))) if all(v is not None for v in per.values()) else None,
            "preference_js":float(np.mean(js_divergence(a,b)))}

def image_bootstrap(predictions,labels,repeats=2000,seed=0):
    pred = {str(r["id"]):r for r in predictions}
    groups = {}
    for r in labels:
        groups.setdefault(str(r["image_id"]),[]).append(r)
    images = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        pp,ll = [],[]
        for draw,image in enumerate(rng.choice(images,len(images),replace=True)):
            for r in groups[image]:
                identifier = str(draw)+":"+str(r["id"])
                pp.append({**pred[str(r["id"])],"id":identifier})
                ll.append({**r,"id":identifier})
        score = rating_report(pp,ll)["macro_r"]
        if score is not None:
            draws.append(score)
    return [float(v) for v in np.quantile(draws,[.025,.975])] if draws else None

def paired_permutation(first,second,labels,repeats=5000,seed=0):
    a,b = {str(r["id"]):r for r in first},{str(r["id"]):r for r in second}
    ids = sorted(a)
    if set(a)!=set(b) or set(a)!={str(r["id"]) for r in labels}:
        raise ValueError("Paired permutation needs identical image-object IDs")
    observed = rating_report(first,labels)["macro_r"]-rating_report(second,labels)["macro_r"]
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for _ in range(repeats):
        swaps = rng.integers(0,2,len(ids))
        aa = [b[i] if swaps[j] else a[i] for j,i in enumerate(ids)]
        bb = [a[i] if swaps[j] else b[i] for j,i in enumerate(ids)]
        ra,rb = rating_report(aa,labels)["macro_r"],rating_report(bb,labels)["macro_r"]
        if ra is not None and rb is not None:
            valid += 1
            exceed += abs(ra-rb)>=abs(observed)
    return {"macro_r_difference":observed,"two_sided_p":(exceed+1)/(valid+1),"valid_permutations":valid}

def partial_spearman(x,y,controls):
    x,y,controls = np.asarray(x,float),np.asarray(y,float),np.asarray(controls,float)
    matrix = np.column_stack([np.ones(len(x)),np.apply_along_axis(rankdata,0,controls)])
    def residual(v):
        rank = rankdata(v)
        return rank-matrix@np.linalg.lstsq(matrix,rank,rcond=None)[0]
    return correlation(residual(x),residual(y))
