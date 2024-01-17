"""Functions for evaluating virtual iHTS results"""

def hit_rate(y):
    return sum(y) / len(y)

def hit_screened(y, sampled_idxs):
    return y[sampled_idxs].sum() / y.sum()



from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def locate_PAINs(mol_ls):
    pains_idxs = []
    
    # initialize filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    
    
    for i, mol in zip(range(len(mol_ls)), mol_ls):
        entry = catalog.GetFirstMatch(mol)  # Get the first matching PAINS
        if entry is not None:
            pains_idxs.append(i)