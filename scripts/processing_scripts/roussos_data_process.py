from rds2py import read_rds, as_sparse_matrix
import os
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

datapath = "/home/dmannk/projects/def-liyue/dmannk/data/Roussos_lab"
atac_datapath = os.path.join(datapath, 'atac', 'GSE204682%5Fcount%5Fmatrix.RDS')
rna_datapath = os.path.join(datapath, 'rna', 'GSE204683%5Fcount%5Fmatrix.RDS')

atac_r = read_rds(atac_datapath)
rna_r  = read_rds(rna_datapath)

atac_sparse = csr_matrix(as_sparse_matrix(atac_r))
rna_sparse  = csr_matrix(as_sparse_matrix(rna_r))

atac_peak_labels = atac_r['attributes']['Dimnames']['data'][0]['data']
atac_cell_labels = atac_r['attributes']['Dimnames']['data'][1]['data']

rna_gene_labels = rna_r['attributes']['Dimnames']['data'][0]['data']
rna_cell_labels = rna_r['attributes']['Dimnames']['data'][1]['data']

print(f'Cell labels match: {(np.asarray(atac_cell_labels) == np.asarray(rna_cell_labels)).all()}')

atac_adata = AnnData(X=atac_sparse.T)
atac_adata.obs.index = atac_cell_labels
atac_adata.var.index = atac_peak_labels

rna_adata = AnnData(X=rna_sparse.T)
rna_adata.obs.index = rna_cell_labels
rna_adata.var.index = rna_gene_labels

atac_adata.write_h5ad(os.path.join(datapath, 'atac', 'roussos_atac.h5ad'))
rna_adata.write_h5ad(os.path.join(datapath, 'rna', 'roussos_rna.h5ad'))
