"""NumPy reference equations used by evaluation and offline invariant checks."""
import numpy as np
from .support import normalize_pmf,support_values,count_bin

def gsd_statistics(pobs,pexp):
    pobs,pexp = normalize_pmf(pobs),normalize_pmf(pexp)
    difference = np.cumsum(pexp,axis=-1)-np.cumsum(pobs,axis=-1)
    # Include the terminal overflow CDF, whose difference is mathematically zero.
    difference[...,-1] = 0.
    dplus = np.maximum(difference.max(axis=-1),0.)
    dminus = np.maximum((-difference).max(axis=-1),0.)
    return {"d_plus":dplus,"d_minus":dminus,"s_dom":dplus-dminus,"a_inc":np.minimum(dplus,dminus)}

def crps_point(p,count):
    p = normalize_pmf(p)
    y = count_bin(count)
    target = (np.arange(102)>=y).astype(float)
    return float(np.sum((np.cumsum(p)-target)**2))

def wasserstein(p,q,overflow_value=101.):
    p,q = normalize_pmf(p),normalize_pmf(q)
    widths = np.diff(support_values(overflow_value=overflow_value))
    return np.sum(np.abs(np.cumsum(p,axis=-1)[...,:-1]-np.cumsum(q,axis=-1)[...,:-1])*widths,axis=-1)

def js_divergence(p,q):
    p,q = np.asarray(p,float),np.asarray(q,float)
    p,q = p/p.sum(axis=-1,keepdims=True),q/q.sum(axis=-1,keepdims=True)
    m = (p+q)/2
    tiny = 1e-12
    return .5*np.sum(p*np.log(np.maximum(p,tiny)/np.maximum(m,tiny)),axis=-1)+.5*np.sum(q*np.log(np.maximum(q,tiny)/np.maximum(m,tiny)),axis=-1)
