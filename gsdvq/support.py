"""The overflow event is distinct from bin 100 (Appendix A.1)."""
import numpy as np

def count_bin(value,nmax=100):
    value = int(value)
    if value<0:
        raise ValueError("Counts must be nonnegative")
    return min(value,nmax+1)

def empirical_pmf(counts,nmax=100):
    if not counts:
        raise ValueError("At least one count judgment is required")
    bins = np.array([count_bin(v,nmax) for v in counts])
    return np.bincount(bins,minlength=nmax+2)/len(bins)

def support_values(nmax=100,overflow_value=101.):
    return np.r_[np.arange(nmax+1,dtype=float),float(overflow_value)]

def normalize_pmf(p):
    p = np.asarray(p,dtype=float)
    if p.shape[-1]!=102 or not np.isfinite(p).all() or (p<0).any() or (p.sum(axis=-1)<=0).any():
        raise ValueError("Expected finite PMFs on 0..100 plus overflow")
    return p/p.sum(axis=-1,keepdims=True)
