import torch
from torch import nn
import torch.nn.functional as F

class CategoricalHead(nn.Module):
    def __init__(self,input_dim,hidden_dim=128,bins=102):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,bins))

    def forward(self,x):
        return self.network(x)

class ObservationEnsemble(nn.Module):
    """Independent count heads on cached frozen CountSE evidence, Eq. 4."""
    def __init__(self,input_dim,hidden_dim=128,heads=3,bins=102):
        super().__init__()
        self.heads = nn.ModuleList([CategoricalHead(input_dim,hidden_dim,bins) for _ in range(heads)])

    def logits(self,x):
        return torch.stack([head(x) for head in self.heads],dim=1)

    def forward(self,x):
        # Average predictive probabilities, not logits (Appendix A.1).
        return self.logits(x).softmax(dim=-1).mean(dim=1)

class ExpectationHead(nn.Module):
    """Eq. 5--6: target-removed frozen scene features + category embedding."""
    def __init__(self,scene_dim,category_dim,hidden_dim=128,bins=102,use_scene=True,use_category=True):
        super().__init__()
        self.use_scene,self.use_category = use_scene,use_category
        self.head = CategoricalHead(scene_dim+category_dim,hidden_dim,bins)

    def logits(self,scene,category):
        if not self.use_scene:
            scene = torch.zeros_like(scene)
        if not self.use_category:
            category = torch.zeros_like(category)
        return self.head(torch.cat([scene,category],dim=-1))

    def forward(self,scene,category):
        return self.logits(scene,category).softmax(dim=-1)

def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module

def point_distribution(p,values):
    # Mean-count rounded to the shared support. Includes the overflow event.
    count = (p*values).sum(-1).round().long().clamp(0,p.shape[-1]-1)
    return F.one_hot(count,num_classes=p.shape[-1]).to(p.dtype)
