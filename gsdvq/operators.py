"""GSD and capacity-matched semantic operators, paper Eqs. 7--9."""
import torch
from torch import nn

class DistributionComparator(nn.Module):
    def __init__(self,kind="gsd",nmax=100,overflow_value=101.,remove_incomparability=False):
        super().__init__()
        self.kind,self.nmax,self.remove_incomparability = kind,nmax,remove_incomparability
        self.register_buffer("values",torch.cat([torch.arange(nmax+1,dtype=torch.float32),torch.tensor([overflow_value])]))

    def forward(self,pobs,pexp):
        fobs,fexp = pobs.cumsum(-1),pexp.cumsum(-1)
        difference = fexp-fobs
        # Terminal event guarantees nonnegative violations even under floating noise.
        difference = torch.cat([difference[...,:-1],torch.zeros_like(difference[...,-1:])],dim=-1)
        dplus,dminus = difference.amax(-1),(-difference).amax(-1)
        signed,inc = dplus-dminus,torch.minimum(dplus,dminus)
        mu_obs,mu_exp = (pobs*self.values).sum(-1),(pexp*self.values).sum(-1)
        if self.kind=="gsd":
            features = torch.stack([signed,torch.zeros_like(inc) if self.remove_incomparability else inc],-1)
        elif self.kind=="count":
            features = torch.stack([2*mu_obs/self.nmax-1,torch.zeros_like(mu_obs)],-1)
        elif self.kind=="context_cdf":
            point = mu_obs.round().long().clamp(0,self.nmax+1)
            percentile = torch.gather(fexp,-1,point.unsqueeze(-1)).squeeze(-1)
            features = torch.stack([2*percentile-1,torch.zeros_like(percentile)],-1)
        elif self.kind=="pos":
            # Strict Pr(Nobs > Nexp); ties are not counted as superiority.
            previous = torch.cat([torch.zeros_like(fexp[...,:1]),fexp[...,:-1]],-1)
            pos = (pobs*previous).sum(-1)
            features = torch.stack([2*pos-1,torch.zeros_like(pos)],-1)
        elif self.kind=="wasserstein":
            widths = self.values[1:]-self.values[:-1]
            distance = ((fobs-fexp)[...,:-1].abs()*widths).sum(-1)/self.nmax
            features = torch.stack([(mu_obs-mu_exp)/self.nmax,distance],-1)
        elif self.kind=="js":
            mean = (pobs+pexp)/2
            distance = .5*(pobs*(pobs.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
            distance += .5*(pexp*(pexp.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
            features = torch.stack([(mu_obs-mu_exp)/self.nmax,distance],-1)
        else:
            raise ValueError("Unknown distributional operator")
        return features,{"d_plus":dplus,"d_minus":dminus,"s_dom":signed,"a_inc":inc,
                         "observed_mean":mu_obs,"expected_mean":mu_exp}
