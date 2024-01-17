from __future__ import annotations

import numpy as np
from typing import List, Iterable, Tuple
from scipy import stats
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from .chem import random_pick


class Learner():
    # Template for iHTS learners
    
    def __init__():
        raise NotImplementedError
    
    def observe(self, idxs: List[int], activities: np.array) -> None:
        raise NotImplementedError
    
    def predict(self, idxs: List[int]) -> np.array:
        raise NotImplementedError
    
    def rank(self, idxs: List[int]) -> List[int]:
        probs = self.predict(idxs)
        ranked_idxs = np.argsort(probs)[::-1]
        return list(np.array(idxs)[ranked_idxs])
    
    def select(self, idxs: List[int], n: int) -> List[int]:
        raise NotImplementedError
        


class Supervised_learner(Learner):
    """Supervised learner for iHTS, wraps around a sklearn-style predictor"""
    
    def __init__(self, clf, p_explore, data):
        
        self.clf = clf
        self.p_explore = p_explore
        self.fp_ls = data["fp_ls"]
        self.X_master = np.concatenate([np.array(data.get("ds_ls")), np.array(self.fp_ls)], axis=1)
        self.X = None
        self.y = []
    
    def observe(self, idxs: List[int], activities: np.array) -> None:
        
        new_X = self.X_master[idxs]
        if self.X is None:
            self.X = new_X
        else:
            self.X = np.concatenate([self.X, new_X], axis=0)
        self.y = np.concatenate([self.y, activities], axis=0)
        
        self.clf.fit(self.X, self.y)
    
    def predict(self, idxs: List[int]) -> np.array:
        X = self.X_master[idxs]
        return self.clf.predict_proba(X)[:,1]  # probability of active
    
    def select(self, idxs: List[int], n: int) -> List[int]:
        n_exploit = int(n * (1-self.p_explore))
        n_explore = int(n * self.p_explore)
        
        exploits = self.rank(idxs)[:n_exploit]
        explores = set(idxs).difference(exploits)
        # explores = random_pick([self.fp_ls[i] for i in explores], n_explore)
        explores = naive_random_choice(idxs, n_explore)
        
        return list(exploits) + list(explores)
        
def naive_random_choice(idxs: List, n: int):
    return np.random.choice(idxs, size=n, replace=False)



"""Bayesian"""
class Distribution():
    def __init__(self):
        raise NotImplementedError
    
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
    
    
    
class Thompson_learner(Learner):
    """Thompson sampling learner for iHTS"""
    
    def __init__(self, data):
        
        scf_dict = data['scaffold_dict']
        self.scf_ls, self.idx_to_scf = initialize_scfs(scf_dict, Beta_dist())

        print(f"Initialized {len(self.scf_ls)} scaffolds for {len(self.idx_to_scf)} compounds")
    
    def observe(self, idxs: List[int], activities: np.array) -> None:
        for idx, y in zip(idxs, activities):
            self.idx_to_scf[idx].observe(idx, y)
    
    def predict(self, idxs: List[int]) -> np.array:
        return [self.idx_to_scf[idx].sample() for idx in idxs]
    
    def select(self, idxs: List[int], n: int) -> List[int]:
        return self.rank(idxs)[:n]
    




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
        
    def sample(self) -> float:
        # sample self distribution
        return self.dist.sample()
    
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


def initialize_scfs(scaffold_dict: dict, prior: Distribution) -> (List[Scaffold], List[int]):
    """Format scaffold dict ({scaffold: [compounds]} into scaffold object"""
    
    # initialize scaffolds
    scaffold_ls = []
    idx_to_scf = []
    for scf_smiles in scaffold_dict:
        scaffold_ls.append(Scaffold(scf_smiles, scaffold_dict[scf_smiles], prior))
        scf_obj = scaffold_ls[-1]
        idx_to_scf += [(scf_obj, comp) for comp in scf_obj.unsampled]

    # sort and prune list
    # resulting list will map compound index to corresponding scaffold object
    idx_to_scf = sorted(idx_to_scf, key=lambda x: x[1])
    idx_to_scf = list(zip(*idx_to_scf))[0]

    return scaffold_ls, idx_to_scf



# Helper class for managing indice during virtual iHTS
class Indice():
    def __init__(self, size: int) -> None:
        self.unsampled = list(range(size))
        self.sampled = []
        
    def add(self, idxs: Iterable[int]) -> None:
        self.sampled = list(set(self.sampled + list(idxs)))
        self.unsampled = list(set(self.unsampled) - set(self.sampled))