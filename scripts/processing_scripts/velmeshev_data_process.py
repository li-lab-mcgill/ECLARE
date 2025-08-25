import scanpy as sc
import pandas as pd
import os

datapath = "/home/dmannk/projects/def-liyue/dmannk/data/Cortex_Velmeshev"
atac_datapath = os.path.join(datapath, 'atac')
rna_datapath = os.path.join(datapath, 'rna')

atac_matrix = sc.read_mtx(os.path.join(atac_datapath, "matrix.mtx.gz"))
atac_meta = pd.read_csv(os.path.join(atac_datapath, "meta.tsv"), sep='\t')
atac_barcodes = pd.read_csv( os.path.join(atac_datapath, 'barcodes.tsv.gz'), compression='gzip', sep='\t', index_col=0, header=None)
atac_features = pd.read_csv( os.path.join(atac_datapath, 'features.tsv.gz'), compression='gzip', sep='\t', index_col=0, header=None)
atac_features['peak'] = atac_features.reset_index()[0].str.split('-').apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}').values
atac_features.index.name = 'peak_og'
assert (atac_meta['Cell_ID'] == atac_barcodes.index).all()

atac = atac_matrix.T
atac.obs = atac_meta.set_index('Cell_ID')
atac.var = atac_features

## Reset index to peak with proper formatting
atac.var = atac.var.reset_index().set_index('peak')

atac.write_h5ad(os.path.join(atac_datapath, "atac_unprocessed.h5ad"))