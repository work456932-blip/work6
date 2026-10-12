"""Expectation prior controls fitted only to auxiliary expectation counts."""
from .support import empirical_pmf

class CountPriors:
    def __init__(self,training_rows):
        if any(r.get("role")!="expectation" or r.get("partition")!="train" for r in training_rows):
            raise ValueError("Priors use only Stage B training counts, never VAQUUM or human judgments")
        self.global_pmf=empirical_pmf([r["count"] for r in training_rows])
        categories={r["category"] for r in training_rows}
        self.categories={c:empirical_pmf([r["count"] for r in training_rows if r["category"]==c]) for c in categories}

    def predict(self,category,kind="category"):
        return (self.categories.get(category,self.global_pmf) if kind=="category" else self.global_pmf).copy()
