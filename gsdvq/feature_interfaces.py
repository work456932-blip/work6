"""External frozen CountSE / scene-text encoder boundary, intentionally local."""
from typing import Protocol
import importlib
import numpy as np

class FrozenEvidence(Protocol):
    provenance: dict
    def counting(self,image,category: str) -> np.ndarray: ...
    def scene(self,target_removed_image) -> np.ndarray: ...
    def category(self,category: str) -> np.ndarray: ...
    def joint(self,image,category: str) -> np.ndarray: ...

def load_factory(name,kwargs=None):
    module,attribute = name.split(":",1)
    return getattr(importlib.import_module(module),attribute)(**(kwargs or {}))
