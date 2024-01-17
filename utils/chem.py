"""
Chemoinformatic/RDKit related functions
"""

import re
import pkgutil
import typing
from typing import List, Callable

import rdkit
from rdkit.Chem import AllChem

from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects




"""Chemical/Physical Descriptors"""
from rdkit.Chem import Descriptors

# prepare descriptor functions
# get names of required descriptors
DS = pkgutil.get_data(__name__, "descriptors.txt")
DS = re.findall("([\w]+)", str(DS))

# fetch corresponding functions
DS_FUNCS = [desc[1] for desc in Descriptors._descList if desc[0] in DS]
print(f"Loaded {len(DS_FUNCS)} descriptor functions")


def get_descriptors(mol_ls: List[rdkit.Chem.rdchem.Mol], ds_funcs: List[Callable] = DS_FUNCS) -> List[List[float]]:
    output_ls = []
    for mol in mol_ls:
        output_ls.append([f(mol) for f in ds_funcs])
    return output_ls

def get_descriptors_single(mol: rdkit.Chem.rdchem.Mol, ds_funcs: List[Callable] = DS_FUNCS):
    return [wrap_non_picklable_objects(f)(mol) for f in ds_funcs]

def get_descriptors_parallel(mol_ls: List[rdkit.Chem.rdchem.Mol], ds_funcs: List[Callable] = DS_FUNCS) -> List[List[float]]:
    return Parallel(n_jobs=8) \
        (delayed(get_descriptors_single)(mol) for mol in mol_ls)



"""Preprocess functions"""

def get_mol_ls(smiles_ls: List[str]) -> List[rdkit.Chem.rdchem.Mol]:
    return list(map(lambda x: rdkit.Chem.MolFromSmiles(x), smiles_ls))

def get_fingerprint_ls(mol_ls: List[rdkit.Chem.rdchem.Mol]):
    return [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits = 1024) for x in mol_ls]

# def get_descriptor_ls(mol_ls: List[rdkit.Chem.rdchem.Mol]):
#     return [get_descriptors_single(mol) for mol in mol_ls]
    
def get_descriptor_ls(mol_ls: List[rdkit.Chem.rdchem.Mol]):
    return get_descriptors_parallel(mol_ls)


"""Scaffolds"""
from collections import defaultdict
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric, GetScaffoldForMol

def get_scaffold(mol: rdkit.Chem.rdchem.Mol) -> str:
    return rdkit.Chem.MolToSmiles(MakeScaffoldGeneric(GetScaffoldForMol(mol)))

def get_scaffold_dict(mol_ls: List[rdkit.Chem.rdchem.Mol]) -> dict:
    scaffold_dict = defaultdict(list)
    for idx, mol in zip(range(len(mol_ls)), mol_ls):
        scaffold_dict[get_scaffold(mol)].append(idx)
    return scaffold_dict



"""Helpers for iteration"""
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
def random_pick(fp_ls: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect], 
                n: int, 
                seed: int = 42) -> List[int]:
    """Random diverse compound picking based on Rdkit MaxMinPicker"""
    picker = MaxMinPicker()
    return list(picker.LazyBitVectorPick(fp_ls, len(fp_ls), n, seed=seed))