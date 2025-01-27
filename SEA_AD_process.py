from anndata import read_h5ad
import pandas as pd
import os

import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    datapath = "/home/dmannk/scratch"
    outpath = "/home/dmannk/projects/def-liyue/dmannk/data/SEA_AD"
    
elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    datapath = outpath = "/Users/dmannk/cisformer/workspace/SEA-AD/"

rna = read_h5ad(os.path.join(datapath, 'SEAAD_MTG_RNAseq_all-nuclei.2024-02-13.h5ad'), backed='r')
atac = read_h5ad(os.path.join(datapath, 'SEAAD_MTG_ATACseq_all-nuclei.2024-02-13.h5ad'), backed='r')

overlap_nuclei = rna.obs_names[rna.obs_names.isin(atac.obs_names)]

rna_filt = (rna.obs_names.isin(overlap_nuclei) * rna.obs['Overall AD neuropathological Change'].isin(['Not AD', 'Low']))
atac_filt = (atac.obs_names.isin(overlap_nuclei) * atac.obs['Overall AD neuropathological Change'].isin(['Not AD', 'Low']))

rna = rna[rna_filt].to_memory()
atac = atac[atac_filt].to_memory()

rna = rna[rna.obs_names.argsort()].copy()
atac = atac[atac.obs_names.argsort()].copy()

rna.write(os.path.join(outpath, 'rna_paired_not_and_low_AD.h5ad'))
atac.write(os.path.join(outpath, 'atac_paired_not_and_low_AD.h5ad'))

print('Done!')