import os
import pickle
import pandas as pd

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