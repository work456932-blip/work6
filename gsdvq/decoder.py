"""Five independent acceptability scores; no ordinal chain or feature bypass."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class QuantifierDecoder(nn.Module):
    def __init__(self,constrained=True,initial_concentration=10.,min_concentration=1e-4):
        super().__init__()
        self.constrained,self.min_concentration = constrained,min_concentration
        self.bias = nn.Parameter(torch.zeros(5))
        self.raw_direction = nn.Parameter(torch.zeros(5))
        self.incomparability = nn.Parameter(torch.zeros(5))
        initial = math.log(math.expm1(initial_concentration-min_concentration))
        self.raw_concentration = nn.Parameter(torch.full((5,),initial))
        self.register_buffer("polarity",torch.tensor([-1.,-1.,0.,1.,1.]))

    def logits(self,g):
        if g.shape[-1]!=2:
            raise ValueError("The semantic decoder only accepts two comparator statistics")
        signed,inc = g[...,0:1],g[...,1:2]
        if self.constrained:
            weight = F.softplus(self.raw_direction)
            direction = signed*self.polarity*weight
            # Eq. 12: some has a central peak, independent of low/high terms.
            center = torch.stack([torch.zeros_like(signed[...,0]),torch.zeros_like(signed[...,0]),
                                  -signed[...,0].abs()*weight[2],torch.zeros_like(signed[...,0]),
                                  torch.zeros_like(signed[...,0])],dim=-1)
            return self.bias+direction+center+inc*self.incomparability
        return self.bias+signed*self.raw_direction+inc*self.incomparability

    def forward(self,g):
        mu = self.logits(g).sigmoid()
        kappa = F.softplus(self.raw_concentration)+self.min_concentration
        return mu,kappa

class FrozenFeatureMLP(nn.Module):
    """Supervised fairness baseline, not the primary GSD decoder."""
    def __init__(self,input_dim,hidden_dim=128,initial_concentration=10.,min_concentration=1e-4):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,5))
        self.raw_concentration = nn.Parameter(torch.full((5,),math.log(math.expm1(initial_concentration-min_concentration))))
        self.min_concentration = min_concentration

    def forward(self,x):
        return self.network(x).sigmoid(),F.softplus(self.raw_concentration)+self.min_concentration
