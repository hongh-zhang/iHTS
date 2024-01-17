from __future__ import annotations
import numpy as np
from typing import List, Tuple

from scipy import stats

class Scaffold():
    
    def __init__(self, 
                 smiles: str,
                 compound_ls: List[int], 
                 dist: Distribution
                ):
        
        self.smiles = smiles
        self.dist = dist
        self.unsampled = compound_ls
        self.sampled = {}
        self.length = len(self.unsampled)
        
    def sample(self) -> List[Tuple[float, int]]:
        # sample self distribution
        # for each compound
        # returns a list of tuple (probability, compound-index)
        return [(self.dist.sample(), comp) for comp in self.unsampled]
    
    def observe(self, compound_idx: int, hit: int) -> None:
        # observe whether a child compound is a hit
        # then update belief
        self.dist.update(hit, (1-hit))
        self.unsampled.remove(compound_idx)
        self.sampled[compound_idx] = hit
    
    def hit_rate(self) -> float:
        return sum(self.sampled.values()) / len(self.sampled.keys())
    
    def __len__(self) -> int:
        return self.length
    
    def __repr__(self):
        return f"{self.smiles} | {sum(self.sampled.values())} / {len(self.sampled.keys())}"
    
    
    
class Distribution():
    def __init__(self):
        pass
    
class Beta_dist(Distribution):
    def __init__(self, alpha: float = 1, beta: float = 1):
        self.alpha = alpha
        self.beta = beta
        
    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, n_success: int, n_failure:int) -> None:
        self.alpha += n_success
        self.beta += n_failure
                     
    def mean(self) -> float:
        return self.alpha / (self.alpha+self.beta)
                     
    def var(self) -> float:
        a = self.alpha
        b = self.beta
        return a*b / (a+b)**2 / (a+b+1)
    
    def std(self) -> float:
        return np.sqrt(self.var())
        
    def ppf(self, p: float = .25) -> float:
        return float(stats.beta.ppf(p,self.alpha,self.beta))