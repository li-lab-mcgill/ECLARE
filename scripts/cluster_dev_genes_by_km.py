import scanpy as sc
import anndata
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from esda.moran import Moran
from libpysal.weights import WSP

sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
from eclare import set_env_variables
set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')

## load EN-only data from PFC Zhu
pfc_zhu_rna_EN = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'rna', 'pfc_zhu_rna_EN_ordinal.h5ad'))

## create graph from ordinal latents
sc.pp.neighbors(pfc_zhu_rna_EN, use_rep='X_ordinal_latents', key_added='X_ordinal_latents_neighbors', n_neighbors=30, n_pcs=10)
sc.tl.leiden(pfc_zhu_rna_EN, resolution=0.6, random_state=0, neighbors_key='X_ordinal_latents_neighbors')
sc.tl.umap(pfc_zhu_rna_EN, neighbors_key='X_ordinal_latents_neighbors')
sc.pl.umap(pfc_zhu_rna_EN, color=['Cell type', 'ordinal_pseudotime', 'leiden'], wspace=0.3, ncols=3)


##
W = WSP(pfc_zhu_rna_EN.obsp['X_ordinal_latents_neighbors_connectivities'].tocsr())
W_dense = W.to_W()

moran_series_list = []
for g, gene in tqdm(enumerate(pfc_zhu_rna_EN.var_names), total=pfc_zhu_rna_EN.n_vars, desc='Calculating Moran\'s I for each gene'):
    x = pfc_zhu_rna_EN.X[:, g].toarray().flatten()
    moran = Moran(x, W_dense, two_tailed=True, permutations=1000)
    moran_series = pd.Series(moran.__dict__, name=gene).loc[['I', 'p_sim']]
    moran_series_list.append(moran_series)

## smooth gene expression by ordinal pseudotime
ordinal_sort = np.argsort(pfc_zhu_rna_EN.obs['ordinal_pseudotime'])
gene_expr = pfc_zhu_rna_EN.X.toarray()
gene_expr_sorted = gene_expr[ordinal_sort]
smooth_gene_expr_df = pd.DataFrame(gene_expr_sorted, columns=pfc_zhu_rna_EN.var_names).rolling(window=100, win_type='hamming', center=True, min_periods=1).mean()
smooth_gene_expr_df.index = pfc_zhu_rna_EN.obs['ordinal_pseudotime'].sort_values().values
smooth_gene_expr_df.plot(figsize=(10, 10))