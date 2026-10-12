import torch
from torch import nn
from .distributions import ObservationEnsemble,ExpectationHead,freeze,point_distribution
from .operators import DistributionComparator
from .decoder import QuantifierDecoder,FrozenFeatureMLP

class GSDVQ(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.observation = ObservationEnsemble(config.counting_dim,config.head_hidden_dim,config.observation_heads,config.nmax+2)
        self.expectation = ExpectationHead(config.scene_dim,config.category_dim,config.head_hidden_dim,
                                           config.nmax+2,config.use_scene,config.use_category)
        self.comparator = DistributionComparator(config.comparison if config.comparison!="frozen_mlp" else "gsd",
            config.nmax,config.overflow_value,config.remove_incomparability)
        self.decoder = (FrozenFeatureMLP(config.joint_dim,config.head_hidden_dim,config.initial_concentration,config.min_concentration)
            if config.comparison=="frozen_mlp" else QuantifierDecoder(config.constrained_decoder,config.initial_concentration,config.min_concentration))

    def freeze_distributions(self):
        freeze(self.observation)
        freeze(self.expectation)

    def forward(self,batch,ground_truth_bins=None):
        # ground_truth_bins is opt-in ONLY for the declared GT operator diagnostic.
        if self.config.comparison=="frozen_mlp":
            mu,kappa = self.decoder(batch["joint"])
            return {"mu":mu,"kappa":kappa}
        with torch.no_grad():
            pobs = self.observation(batch["counting"])
            pexp = self.expectation(batch["scene"],batch["category"])
            if ground_truth_bins is not None:
                pobs = torch.nn.functional.one_hot(ground_truth_bins,num_classes=self.config.nmax+2).to(pobs.dtype)
            elif self.config.collapse_observation:
                pobs = point_distribution(pobs,self.comparator.values)
            if self.config.collapse_expectation:
                pexp = point_distribution(pexp,self.comparator.values)
            features,diagnostics = self.comparator(pobs,pexp)
        mu,kappa = self.decoder(features)
        return {"mu":mu,"kappa":kappa,"pobs":pobs,"pexp":pexp,"features":features,**diagnostics}
