import torch
import torch.nn.functional as F

def categorical_crps(logits,count_bins):
    p = logits.softmax(-1)
    cdf = p.cumsum(-1)
    support = torch.arange(p.shape[-1],device=p.device)
    target = (support>=count_bins.unsqueeze(-1)).to(p.dtype)
    return (cdf-target).square().sum(-1).mean()

def count_loss(logits,count_bins,crps_weight=.1):
    if logits.ndim==3:
        # Each ensemble head receives its own NLL+CRPS objective.
        losses = [count_loss(logits[:,h],count_bins,crps_weight) for h in range(logits.shape[1])]
        return torch.stack(losses).mean()
    return F.cross_entropy(logits,count_bins)+crps_weight*categorical_crps(logits,count_bins)

def participant_beta_nll(mu,kappa,ratings,valid,epsilon=1e-4):
    """Eq. 13, response-level likelihood with padded/missing ratings masked out."""
    if not valid.any():
        raise ValueError("Batch has no valid participant ratings")
    if ratings.shape!=valid.shape or ratings.shape[:2]!=mu.shape:
        raise ValueError("Expected ratings/valid [batch,5,participants] and means [batch,5]")
    y = ratings.clamp(epsilon,1-epsilon)
    alpha = (mu.clamp(epsilon,1-epsilon)*kappa).unsqueeze(-1)
    beta = ((1-mu.clamp(epsilon,1-epsilon))*kappa).unsqueeze(-1)
    log_norm = torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)
    nll = log_norm-(alpha-1)*y.log()-(beta-1)*torch.log1p(-y)
    return nll[valid].mean()
