from muon import read_10x_h5
from anndata import concat as anndata_concat
from anndata import read_h5ad
import pandas as pd
import os
from glob import glob
import GEOparse

import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    datapath = '/home/dmannk/projects/def-liyue/dmannk/data/PD_Adams_et_al'

elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    datapath = "/Users/dmannk/cisformer/workspace/PD_Adams_et_al/"

## Get (meta) data from GEO
gse = GEOparse.get_GEO(geo='GSE193688')
subjects_metadata = gse.gsms
#subjects_geo_accessions = subjects.keys()

## Create function to properly convert ATAC peak names
def convert_atac_var_name(var_name):
    parts = var_name.strip("()").split(", ")
    chrom = parts[0].strip("'")
    start = parts[1]
    end = parts[2]
    return f"{chrom}:{start}-{end}"

## Load ATAC data and remove peaks whose name starts with non-chromosomal prefix
atac = read_h5ad(os.path.join(datapath, 'atac_consensus.h5ad'))
atac.var_names = [convert_atac_var_name(var_name) for var_name in atac.var_names]
assert atac[:, atac.var_names.str.startswith('chr')].all()
atac_subject_strs_unique = list(atac.obs['sample'].unique())

## Get paths to all multimodal data
all_h5_files = glob(os.path.join(datapath, 'PD_all_files', 'GSM*filtered_feature_bc_matrix.h5'))

## Reorder h5 files to match ATAC order
subject_strs = [os.path.basename(h5_file).split('_')[0] for h5_file in all_h5_files]
subject_atac_order = [subject_strs.index(s) for s in atac_subject_strs_unique]
all_h5_files = [all_h5_files[i] for i in subject_atac_order]

rnas = []

for h5_file in all_h5_files:

    print(f'Loading {os.path.basename(h5_file)}...')
    subject_str = os.path.basename(h5_file).split('_')[0]

    ## Extract metadata for this subject
    metadata = subjects_metadata[subject_str].metadata['characteristics_ch1']
    metadata = [tuple(m.split(': ')) for m in metadata]
    metadata = pd.DataFrame(metadata, columns=['key', 'value']).set_index('key')

    ## Load data and split into RNA and ATAC
    data = read_10x_h5(h5_file)
    rna = data['rna']

    ## Fill-in obs variables
    rna.obs[metadata.index] = metadata.value
    rna.obs['subject'] = subject_str

    ## Relabel obs names
    rna.obs_names = [f'{obs_name}-{subject_str}' for obs_name in rna.obs_names]

    ## Make gene names unique - always the same 10 genes that are duplicated, so we can just make them unique and the genes will align across subjects
    rna.var_names_make_unique()

    rnas.append(rna)


## Concatenate data
rna = anndata_concat(rnas, axis=0)
assert (rna.obs_names == atac.obs_names).all()

## Set ATAC obs to RNA obs
atac.obs = rna.obs

## Extract control samples
rna_ctrl = rna[rna.obs['diagnosis']=='Unaffected Control']
atac_ctrl = atac[atac.obs['diagnosis']=='Unaffected Control']

print('Dimensions of RNA data for control subjects: ', rna_ctrl.shape)
print('Dimensions of ATAC data for control subjects: ', atac_ctrl.shape)

## Save data
print('Saving data from control subjects...')
rna_ctrl.write(os.path.join(datapath, 'rna_ctrl.h5ad'))
atac_ctrl.write(os.path.join(datapath, 'atac_ctrl.h5ad'))

print('Done!')