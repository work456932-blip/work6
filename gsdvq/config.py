from dataclasses import asdict, dataclass, field
import hashlib
import json
import yaml

@dataclass
class Config:
    # Explicit manuscript specifications.
    nmax: int = 100
    observation_heads: int = 3
    outer_folds: int = 5
    seeds: list[int] = field(default_factory=lambda: [42,43,44])
    comparison: str = "gsd"
    use_scene: bool = True
    use_category: bool = True
    constrained_decoder: bool = True
    collapse_observation: bool = False
    collapse_expectation: bool = False
    remove_incomparability: bool = False
    # Reference choices where the paper does not specify a numerical value.
    counting_dim: int = 512
    scene_dim: int = 512
    category_dim: int = 512
    joint_dim: int = 512
    head_hidden_dim: int = 128
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    stage_a_epochs: int = 30
    stage_b_epochs: int = 30
    stage_c_epochs: int = 100
    patience: int = 10
    crps_weight: float = 0.1
    rating_epsilon: float = 0.0001
    initial_concentration: float = 10.0
    min_concentration: float = 0.0001
    overflow_value: float = 101.0
    fold_seed: int = 2026
    validation_category_fraction: float = 0.2
    operators: list[str] = field(default_factory=lambda: ["gray","blur","inpaint"])
    operator_probabilities: list[float] = field(default_factory=lambda: [1/3,1/3,1/3])
    seed: int = 42
    device: str = "cpu"
    feature_variant: str = "primary"
    inference_operator: str = "inpaint"
    checkpoint_policy: str = "strict"

    def __post_init__(self):
        if self.nmax != 100:
            raise ValueError("The supplied manuscript uses nmax=100 plus overflow")
        if self.overflow_value <= self.nmax:
            raise ValueError("Overflow event must follow the finite bins")
        if len(self.operators)!=len(self.operator_probabilities) or abs(sum(self.operator_probabilities)-1)>1e-6:
            raise ValueError("Invalid mask-operator mixture")
        if self.inference_operator not in self.operators:
            raise ValueError("Inference operator must be in the configured operator set")
        if self.comparison not in {"gsd","count","context_cdf","pos","wasserstein","js","frozen_mlp"}:
            raise ValueError("Unknown comparison")
        if self.checkpoint_policy not in {"strict","diagnostic"}:
            raise ValueError("Unknown checkpoint exposure policy")
        if self.outer_folds!=5 or len(self.seeds)!=3:
            raise ValueError("Primary protocol requires five folds and three seeds")
        if self.observation_heads not in {1,3}:
            raise ValueError("Use three heads or the stated single-head ablation")

    @classmethod
    def load(cls,path):
        with open(path,encoding="utf-8") as f:
            return cls(**(yaml.safe_load(f) or {}))

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()
