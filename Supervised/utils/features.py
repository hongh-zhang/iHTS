import re
import pkgutil
import typing
from typing import List, Callable

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem as Chem

from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

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



def get_descriptors_parallel(mol_ls: List[rdkit.Chem.rdchem.Mol], ds_funcs: List[Callable] = DS_FUNCS) -> List[List[float]]:
    return Parallel(n_jobs=8) \
        (delayed(get_descriptors_single)(mol) for mol in mol_ls)


def get_descriptors_single(mol: rdkit.Chem.rdchem.Mol, ds_funcs: List[Callable] = DS_FUNCS):
    return [wrap_non_picklable_objects(f)(mol) for f in ds_funcs]