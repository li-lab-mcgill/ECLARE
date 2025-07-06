#%% set env variables
import os
import sys

# Check if environment variables are already set
eclare_root = os.environ.get('ECLARE_ROOT')
outpath = os.environ.get('OUTPATH')
datapath = os.environ.get('DATAPATH')

# Print status of environment variables
if all([eclare_root, outpath, datapath]):
    print(f"Environment variables already set:")
    print(f"ECLARE_ROOT: {eclare_root}")
    print(f"OUTPATH: {outpath}")
    print(f"DATAPATH: {datapath}")
else:
    print(f"Missing environment variables")

    config_path = '../config'
    sys.path.insert(0, config_path)

    from export_env_variables import export_env_variables
    export_env_variables(config_path)

#%% import data

import pickle
import pandas as pd
import numpy as np

from eclare.post_hoc_utils import tree

## Create dict for methods and job_ids
methods_id_dict = {'eclare_mdd': ['16103846']}
base_output_dir = os.path.join(os.environ['OUTPATH'], f"enrichment_analyses_{methods_id_dict['eclare_mdd'][0]}")
output_dir = base_output_dir + '_41'

# Load all saved dictionaries
dicts_to_load = [
    'X_rna_dict',
    'X_atac_dict',
    'pydeseq2_results_dict',
    'significant_genes_dict',
    'overlapping_target_genes_dict',
    'overlapping_tfs_dict',
    'scompreg_loglikelihoods_dict',
    'std_errs_dict',
    'tg_expressions_dict',
    'tfrps_dict',
    'tfrp_predictions_dict',
    'slopes_dict',
    'intercepts_dict',
    'intercept_stderrs_dict',
    'enrs_dict',
    'magma_results_dict',
    'mean_grn_df_filtered_dict'
]

loaded_dicts = {}
for dict_name in dicts_to_load:
    dict_path = os.path.join(output_dir, f"{dict_name}.pkl")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            loaded_dicts[dict_name] = pickle.load(f)
        print(f"Loaded {dict_name}")
    else:
        print(f"Warning: {dict_path} not found")

# Unpack loaded dictionaries into individual variables
X_rna_dict = loaded_dicts.get('X_rna_dict', tree())
X_atac_dict = loaded_dicts.get('X_atac_dict', tree())
pydeseq2_results_dict = loaded_dicts.get('pydeseq2_results_dict', tree())
significant_genes_dict = loaded_dicts.get('significant_genes_dict', tree())
overlapping_target_genes_dict = loaded_dicts.get('overlapping_target_genes_dict', tree())
overlapping_tfs_dict = loaded_dicts.get('overlapping_tfs_dict', tree())
scompreg_loglikelihoods_dict = loaded_dicts.get('scompreg_loglikelihoods_dict', tree())
std_errs_dict = loaded_dicts.get('std_errs_dict', tree())
tg_expressions_dict = loaded_dicts.get('tg_expressions_dict', tree())
tfrps_dict = loaded_dicts.get('tfrps_dict', tree())
tfrp_predictions_dict = loaded_dicts.get('tfrp_predictions_dict', tree())
slopes_dict = loaded_dicts.get('slopes_dict', tree())
intercepts_dict = loaded_dicts.get('intercepts_dict', tree())
intercept_stderrs_dict = loaded_dicts.get('intercept_stderrs_dict', tree())
enrs_dict = loaded_dicts.get('enrs_dict', tree())
magma_results_dict = loaded_dicts.get('magma_results_dict', tree())
mean_grn_df_filtered_dict = loaded_dicts.get('mean_grn_df_filtered_dict', tree())
gene_set_scores_dict = loaded_dicts.get('gene_set_scores_dict', tree())

## Load CSV files and other file types
shared_TF_TG_pairs_df = pd.read_csv(os.path.join(output_dir, 'shared_TF_TG_pairs.csv'))

enrs_mdd_dn_hits_df = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_hits_df.csv'))
enrs_mdd_dn_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_tfs_multiple_hits.csv'))

all_sccompreg_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_tfs_multiple_hits.csv'))
all_sccompreg_hits_df = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_hits_df.csv'))

pydeseq2_match_length_genes_hits_df = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_hits_df.csv'))
pydeseq2_match_length_genes_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_tfs_multiple_hits.csv'))

#%%

unique_sexes = list(enrs_dict.keys())
unique_celltypes = list(enrs_dict[unique_sexes[0]].keys())

enrs_mdd_dn_hits_df = pd.DataFrame(columns=['ngenes', 'padj', 'mlog10_padj'], 
                                  index=pd.MultiIndex.from_product([unique_sexes, unique_celltypes], names=['sex', 'celltype']))

for sex in unique_sexes:
    for celltype in unique_celltypes:

        enrs = enrs_dict[sex][celltype]['All LR']
        enrs_mdd_dn = enrs[enrs.Term == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']

        if len(enrs_mdd_dn) == 0:
            continue

        ngenes = int(enrs_mdd_dn['Overlap'].str.split('/').item()[0])
        padj = enrs_mdd_dn['Adjusted P-value'].item()
        mlog10_padj = -np.log10(padj)

        enrs_mdd_dn_hits_df.loc[(sex, celltype), 'ngenes'] = ngenes
        enrs_mdd_dn_hits_df.loc[(sex, celltype), 'padj'] = padj
        enrs_mdd_dn_hits_df.loc[(sex, celltype), 'mlog10_padj'] = mlog10_padj


enrs_mdd_dn_hits_df.reset_index(inplace=True)
enrs_mdd_dn_hits_df['mlog10_padj'] = enrs_mdd_dn_hits_df['mlog10_padj'].fillna(0)
enrs_mdd_dn_hits_df['ngenes'] = enrs_mdd_dn_hits_df['ngenes'].fillna(0)
enrs_mdd_dn_hits_df['size_ngenes'] = enrs_mdd_dn_hits_df['ngenes'] * 10

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(enrs_mdd_dn_hits_df['celltype'], enrs_mdd_dn_hits_df['sex'], s=enrs_mdd_dn_hits_df['size_ngenes'], c=enrs_mdd_dn_hits_df['mlog10_padj'], cmap='viridis', alpha=0.5)

# Add grey crosses for points where mlog10_padj == 0
zero_data = enrs_mdd_dn_hits_df[enrs_mdd_dn_hits_df['mlog10_padj'] == 0]
if len(zero_data) > 0:
    ax.scatter(zero_data['celltype'], zero_data['sex'], 
                color='grey', alpha=0.7, 
                marker='+', linewidths=1)

ax.set_ylim(-0.5, 1.5)

## Bonferroni correction
bonferroni_threshold = 0.05 / len(enrs_mdd_dn_hits_df)
bonferroni_threshold_mlog10 = -np.log10(bonferroni_threshold)
bonferonni_mask = enrs_mdd_dn_hits_df['mlog10_padj'] > bonferroni_threshold_mlog10

enrs_mdd_dn_hits_df['Adjusted P-value (bonferroni)'] = 0
enrs_mdd_dn_hits_df['size_ngenes_bonferroni'] = 0
enrs_mdd_dn_hits_df.loc[bonferonni_mask, 'Adjusted P-value (bonferroni)'] = enrs_mdd_dn_hits_df['mlog10_padj']
enrs_mdd_dn_hits_df.loc[bonferonni_mask, 'size_ngenes_bonferroni'] = enrs_mdd_dn_hits_df['size_ngenes']

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(enrs_mdd_dn_hits_df['celltype'], enrs_mdd_dn_hits_df['sex'], s=enrs_mdd_dn_hits_df['size_ngenes_bonferroni'], c=enrs_mdd_dn_hits_df['Adjusted P-value (bonferroni)'], cmap='viridis', alpha=0.5)

zero_data = enrs_mdd_dn_hits_df[enrs_mdd_dn_hits_df['Adjusted P-value (bonferroni)'] == 0]
if len(zero_data) > 0:
    ax.scatter(zero_data['celltype'], zero_data['sex'], 
                color='grey', alpha=0.7, 
                marker='+', linewidths=1)

ax.set_ylim(-0.5, 1.5)






#%%





