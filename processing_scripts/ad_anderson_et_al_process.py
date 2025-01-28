from muon import read_10x_h5
import pandas as pd
import os
from numpy import isin as np_isin

## set path to Alzheimer's data from Anderson et al.
datapath = '/Users/dmannk/cisformer/workspace/AD_Anderson_et_al/snMultiome'

## Load data and metadata
data = read_10x_h5(os.path.join(datapath, 'GSE214979_filtered_feature_bc_matrix.h5'))
cell_metadata = pd.read_csv(os.path.join(datapath, 'GSE214979_cell_metadata.csv.gz'), index_col=0)

## See where have repeated features
rep_features = pd.value_counts(data.var_names)>1
rep_features = rep_features[rep_features]
rep_features_data = data[:,data.var_names.isin(rep_features[rep_features].index)]
print(rep_features_data)

## Extract RNA data
rna = data['rna']
rna.obs = cell_metadata

## Compute average for repeated features (only RNA genes in this case)
factors, _ = pd.factorize(rna.var_names)
factors_rep  = pd.value_counts(factors)>1
factors_rep = factors_rep[factors_rep].index.values

for factor_rep in factors_rep:
    idx = np_isin(factors, factor_rep)
    rna.X[:,idx] = rna.X[:,idx].mean(axis=0)

## Remove duplicates
duplicated_mask = rna.var_names.duplicated(keep='first')
rna = rna[:,~duplicated_mask]

## Extract control samples
rna_ctrl = rna[rna.obs['Status']=='Ctrl']


## Extract ATAC data and filter for control samples
atac = data['atac']
atac.obs = cell_metadata

## Remove peaks whose name starts with non-chromosomal prefix
atac = atac[:, atac.var_names.str.startswith('chr')]

## Extract control samples
atac_ctrl = atac[atac.obs['Status']=='Ctrl']


## Save data
print('Saving data from control subjects...')
rna_ctrl.write(os.path.join(datapath, 'rna_ctrl.h5ad'))
atac_ctrl.write(os.path.join(datapath, 'atac_ctrl.h5ad'))

print('Done!')