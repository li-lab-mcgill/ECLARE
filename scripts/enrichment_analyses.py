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

#%%
import os
import pickle
import numpy as np
import pandas as pd
import pybedtools
import anndata
import torch
import seaborn as sns
from scanpy.tl import score_genes
from statsmodels.stats.weightstats import DescrStatsW
import json

# Set matplotlib to use a thread-safe backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add logging for better debugging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.environ.get('OUTPATH', '.'), 'enrichment_analyses.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from types import SimpleNamespace
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import threading

from eclare.post_hoc_utils import \
    extract_target_source_replicate, initialize_dicts, assign_to_dicts, perform_gene_set_enrichment, differential_grn_analysis, process_celltype, load_model_and_metadata, get_brain_gmt, magma_dicts_to_df, get_next_version_dir, compute_LR_grns, do_enrichr, find_hits_overlap, \
    set_env_variables, download_mlflow_runs,\
    tree

set_env_variables()

cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'

## Create dict for methods and job_ids
methods_id_dict = {
    'clip_mdd': '15093733',
    'eclare_mdd': ['16103846'], #16105437
}

## define search strings
search_strings = {
    'clip_mdd': 'CLIP' + '_' + methods_id_dict['clip_mdd'],
    'eclare_mdd': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare_mdd']]
}

## Create output directory with version counter
base_output_dir = os.path.join(os.environ['OUTPATH'], f"enrichment_analyses_{methods_id_dict['eclare_mdd'][0]}")
output_dir = get_next_version_dir(base_output_dir)
os.makedirs(output_dir, exist_ok=True)

#%% unpaired MDD data
experiment_name = f"clip_mdd_{methods_id_dict['clip_mdd']}"

if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
else:
    print(f"Downloading runs.csv for {experiment_name} from MLflow")
    all_metrics_csv_path = download_mlflow_runs(experiment_name)

mdd_metrics_df = pd.read_csv(all_metrics_csv_path)
ECLARE_mdd_header_idxs = np.where(mdd_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['eclare_mdd'])))[0]
ECLARE_mdd_run_id = mdd_metrics_df.iloc[ECLARE_mdd_header_idxs]['run_id']
ECLARE_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(ECLARE_mdd_run_id)]
ECLARE_mdd_metrics_df = extract_target_source_replicate(ECLARE_mdd_metrics_df, has_source=False)


#%% Load ECLARE student model

## Find path to best ECLARE model
best_eclare_mdd     = str(ECLARE_mdd_metrics_df['multimodal_ilisi'].argmax())
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_mdd_{methods_id_dict["eclare_mdd"][0]}', best_eclare_mdd, device)
eclare_student_model = eclare_student_model.train().to('cpu')

#%% load data

## define decimation factor
decimate_factor = 1

## define args for mdd_setup
args = SimpleNamespace(
    source_dataset='MDD',
    target_dataset=None,
    genes_by_peaks_str='17563_by_100000'
)

rna_datapath = atac_datapath = os.path.join(os.environ['DATAPATH'], 'mdd_data')

RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"

rna_fullpath = os.path.join(rna_datapath, RNA_file)
atac_fullpath = os.path.join(atac_datapath, ATAC_file)

atac = anndata.read_h5ad(atac_fullpath, backed='r')
rna  = anndata.read_h5ad(rna_fullpath, backed='r')

## define keys
rna_celltype_key='ClustersMapped'
atac_celltype_key='ClustersMapped'

rna_condition_key='Condition'
atac_condition_key='condition'

rna_subject_key='OriginalSub'
atac_subject_key='BrainID'

rna_sex_key = 'Sex'
atac_sex_key = 'sex'

rna_batch_key = 'Batch'
atac_batch_key = 'batch'

mdd_atac = atac[::decimate_factor].to_memory()
mdd_rna = rna[::decimate_factor].to_memory()

mdd_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(list(mdd_atac.var_names.str.split(':|-', expand=True)), columns=['chrom', 'start', 'end']))

## also load counts data for pyDESeq2 and full processed data for annotations
rna_full = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna.h5ad'), backed='r')
rna_scaled_with_counts = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna_scaled.h5ad'), backed='r')
rna_scaled_with_counts = rna_scaled_with_counts[::decimate_factor].to_memory()

ct_map_dict = dict({1: 'ExN', 0: 'InN', 4: 'Oli', 2: 'Ast', 3: 'OPC', 6: 'End', 5: 'Mix', 7: 'Mic'})
rna_scaled_with_counts.obs[rna_celltype_key] = rna_scaled_with_counts.obs['Broad'].map(ct_map_dict)

rna_counts_X = rna_scaled_with_counts.raw.X.astype(int).toarray()
rna_counts_obs = rna_scaled_with_counts.obs
rna_counts_var = rna_full.var
rna_counts_var['in_mdd'] = rna_counts_var.index.isin(mdd_rna.var_names)

mdd_rna_counts = anndata.AnnData(
    X=rna_counts_X,
    var=rna_counts_var,
    obs=rna_counts_obs,
)

#unique_celltypes = ['Ast', 'Oli']; logger.warning(f"Only using {unique_celltypes} celltypes for analysis.")
unique_celltypes = np.unique(np.concatenate([mdd_rna.obs[rna_celltype_key], mdd_atac.obs[atac_celltype_key]]))
unique_conditions = np.unique(np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]))
unique_sexes = np.unique(np.concatenate([mdd_rna.obs[rna_sex_key].str.lower(), mdd_atac.obs[atac_sex_key].str.lower()]))


## prepend subject label with 'case_' or 'control_'
mdd_rna.obs[rna_subject_key] = mdd_rna.obs[rna_subject_key].astype(str)
mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Case'] = mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Case'].apply(lambda x: f'case_{x}')
mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Control'] = mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Control'].apply(lambda x: f'control_{x}')

mdd_atac.obs[atac_subject_key] = mdd_atac.obs[atac_subject_key].astype(str)
mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Case'] = mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Case'].apply(lambda x: f'case_{x}')
mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Control'] = mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Control'].apply(lambda x: f'control_{x}')

subjects_by_condition_n_sex_df = pd.DataFrame({
    'subject': np.concatenate([mdd_rna.obs[rna_subject_key], mdd_atac.obs[atac_subject_key]]),
    'condition': np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]),
    'sex': np.concatenate([mdd_rna.obs[rna_sex_key].str.lower(), mdd_atac.obs[atac_sex_key].str.lower()])
})
overlapping_subjects = np.intersect1d(mdd_rna.obs[rna_subject_key], mdd_atac.obs[atac_subject_key])
subjects_by_condition_n_sex_df = subjects_by_condition_n_sex_df[subjects_by_condition_n_sex_df['subject'].isin(overlapping_subjects)]
subjects_by_condition_n_sex_df = subjects_by_condition_n_sex_df.groupby(['condition', 'sex'])['subject'].unique()

#%% setup before pyDESeq2 and sc-compReg

X_rna_dict, X_atac_dict, overlapping_target_genes_dict, overlapping_tfs_dict, genes_by_peaks_corrs_dict, genes_by_peaks_masks_dict, n_dict, scompreg_loglikelihoods_dict, std_errs_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, tfrps_dict, tfrp_predictions_dict \
    = initialize_dicts()

## initialize dicts
pydeseq2_results_dict = tree()
significant_genes_dict = tree()

enrs_dict = tree()
magma_results_dict = tree()
mean_grn_df_filtered_dict = tree()
mean_grn_df_filtered_pruned_dict = tree()

## Get BrainGMT and filter for cortical genes
brain_gmt_cortical, brain_gmt_cortical_wGO = get_brain_gmt()

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]

     
#%% pyDESeq2
for sex in unique_sexes:
    sex = sex.lower()
    
    for celltype in unique_celltypes:

        results = process_celltype(sex, celltype, rna_scaled_with_counts, mdd_rna.var, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key)
        pydeseq2_results_dict[sex][celltype] = results[1]
        significant_genes_dict[sex][celltype] = results[2]

# Save results with thread-safe file writing
dicts_to_save = {
    'pydeseq2_results_dict': pydeseq2_results_dict,
    'significant_genes_dict': significant_genes_dict,
}

for dict_name, dict_obj in dicts_to_save.items():
    with open(os.path.join(output_dir, f"{dict_name}.pkl"), "wb") as f:
        pickle.dump(dict_obj, f)
    print(f"Saved {dict_name}")


#%% sc-compReg analysis

def safe_differential_grn_analysis(condition, sex, celltype, *args, **kwargs):
    """Thread-safe wrapper for differential_grn_analysis"""
    try:
        logger.info(f"Starting differential GRN analysis for {condition}_{sex}_{celltype}")
        # Create unique subdirectory for each thread
        thread_id = threading.current_thread().ident
        unique_subdir = os.path.join(kwargs['subdir'], f'grn_thread_{thread_id}')
        os.makedirs(unique_subdir, exist_ok=True)
        kwargs['subdir'] = unique_subdir

        result = differential_grn_analysis(condition, sex, celltype, *args, **kwargs)
        logger.info(f"Completed differential GRN analysis for {condition}_{sex}_{celltype}")
        return result
    except Exception as e:
        logger.error(f"Error in thread {threading.current_thread().ident} for {condition}_{sex}_{celltype}: {e}")
        return None, None

for sex in unique_sexes:
    sex = sex.lower()
    
    for condition in unique_conditions:

        results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
                delayed(safe_differential_grn_analysis)(
                    condition, sex, celltype, mdd_rna, mdd_atac, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, rna_batch_key, atac_celltype_key, atac_condition_key, atac_sex_key, atac_subject_key, atac_batch_key, eclare_student_model, mean_grn_df, \
                    overlapping_subjects, subjects_by_condition_n_sex_df, cutoff=10000, ot_alignment_type='all', subdir=os.path.join(output_dir, f'{sex}_{celltype}', condition)
                )
            for celltype in unique_celltypes
        )
        for celltype, result in zip(unique_celltypes, results):
            assign_to_dicts(*result)

# Save results with thread-safe file writing
dicts_to_save = {
    'X_rna_dict': X_rna_dict,
    'X_atac_dict': X_atac_dict,
    'overlapping_target_genes_dict': overlapping_target_genes_dict,
    'overlapping_tfs_dict': overlapping_tfs_dict,
    'scompreg_loglikelihoods_dict': scompreg_loglikelihoods_dict,
    'std_errs_dict': std_errs_dict,
    'tg_expressions_dict': tg_expressions_dict,
    'tfrps_dict': tfrps_dict,
    'tfrp_predictions_dict': tfrp_predictions_dict,
    'slopes_dict': slopes_dict,
    'intercepts_dict': intercepts_dict,
    'intercept_stderrs_dict': intercept_stderrs_dict,
}

file_lock = threading.Lock()
for dict_name, dict_obj in dicts_to_save.items():
    with file_lock:
        with open(os.path.join(output_dir, f"{dict_name}.pkl"), "wb") as f:
            pickle.dump(dict_obj, f)
        print(f"Saved {dict_name}")

#%% gene set enrichment analyses

def safe_perform_gene_set_enrichment(sex, celltype, *args, **kwargs):
    """Thread-safe wrapper for perform_gene_set_enrichment"""
    try:
        logger.info(f"Starting enrichment for {sex}_{celltype}")
        # Create unique subdirectory for each thread
        thread_id = threading.current_thread().ident
        unique_subdir = os.path.join(kwargs['subdir'], f'enr_thread_{thread_id}')
        os.makedirs(unique_subdir, exist_ok=True)
        kwargs['subdir'] = unique_subdir
        
        result = perform_gene_set_enrichment(sex, celltype, *args, **kwargs)
        logger.info(f"Completed enrichment for {sex}_{celltype}")
        return result
    except Exception as e:
        logger.error(f"Error in thread {threading.current_thread().ident} for {sex}_{celltype}: {e}")
        return None, None

logger.info("Starting gene set enrichment analyses")
for sex in unique_sexes:
    sex = sex.lower()
    logger.info(f"Processing sex: {sex}")
    
    # Use a lock to prevent race conditions when creating directories
    dir_lock = threading.Lock()
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(safe_perform_gene_set_enrichment)(
            sex, celltype, scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, mean_grn_df, significant_genes_dict, mdd_rna.var_names, pydeseq2_results_dict, brain_gmt_cortical, slopes_dict, std_errs_dict, intercepts_dict, intercept_stderrs_dict, subdir=os.path.join(output_dir, f'{sex}_{celltype}')
        )
        for celltype in unique_celltypes
    )

    for celltype, result in zip(unique_celltypes, results):
        if result is not None and result[0] is not None:
            enrs_dict[sex][celltype] = result[0]
            magma_results_dict[sex][celltype] = result[1]
            mean_grn_df_filtered_dict[sex][celltype] = result[2]
        else:
            print(f"Warning: No results for {sex}_{celltype}")

# Save results with thread-safe file writing
dicts_to_save = {
    'enrs_dict': enrs_dict,
    'magma_results_dict': magma_results_dict,
    'mean_grn_df_filtered_dict': mean_grn_df_filtered_dict,
}

# Use a lock for file writing to prevent race conditions
file_lock = threading.Lock()
for dict_name, dict_obj in dicts_to_save.items():
    with file_lock:
        with open(os.path.join(output_dir, f"{dict_name}.pkl"), "wb") as f:
            pickle.dump(dict_obj, f)
        print(f"Saved {dict_name}")

## transform magma_results_dict to df
magma_results_df = magma_dicts_to_df(magma_results_dict)
magma_results_mlog10 = magma_results_df.apply(lambda x: -np.log10(x))

plt.figure(figsize=(10, 8))
ax = sns.heatmap(magma_results_mlog10, annot=False, cbar_kws={'label': '-log10(p-value)'})

# Add asterisks for significant values (p < 0.05, i.e., -log10 > -log10(0.05))
threshold = -np.log10(0.05)
for i in range(magma_results_mlog10.shape[0]):
    for j in range(magma_results_mlog10.shape[1]):
        if magma_results_mlog10.iloc[i, j] > threshold:
            ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='black')

plt.savefig(os.path.join(output_dir, "magma_heatmap.png"), bbox_inches='tight', dpi=150)
plt.close()


#%% obtain LR for TF-peak-TG combinations and find enriched TF-TG pairs

unique_TF_TG_combinations_dict = tree()
enriched_TF_TG_pairs_dict = dict()
shared_TF_TG_pairs = set()
all_TF_TG_pairs = set()

## compute LR for TF-peak-TG combinations
for sex in unique_sexes:
    sex = sex.lower()

    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(compute_LR_grns)(sex, celltype, mean_grn_df_filtered_dict, X_rna_dict, X_atac_dict, output_dir=output_dir)
        for celltype in unique_celltypes
    )

    for celltype, result in zip(unique_celltypes, results):
        mean_grn_df_filtered_pruned_dict[sex][celltype] = result

## save mean_grn_df_filtered_pruned_dict
with open(os.path.join(output_dir, 'mean_grn_df_filtered_pruned_dict.pkl'), 'wb') as f:
    pickle.dump(mean_grn_df_filtered_pruned_dict, f)


## save peak BED files for GREAT analysis (R script)
os.makedirs(os.path.join(output_dir, 'peak_bed_files'), exist_ok=True)

for sex in unique_sexes:
    sex = sex.lower()
    for celltype in unique_celltypes:
        #mean_grn_df_filtered_pruned = mean_grn_df_filtered_pruned_dict[sex][celltype]
        #unique_peaks_df = mean_grn_df_filtered_pruned.drop_duplicates('enhancer')

        mean_grn_df_filtered = mean_grn_df_filtered_dict[sex][celltype]
        unique_peaks_df = mean_grn_df_filtered.drop_duplicates('enhancer')

        peaks_bed = unique_peaks_df[['chrom','chromStart','chromEnd']]
        peaks_bed.to_csv(os.path.join(output_dir, 'peak_bed_files', f'{sex}_{celltype}_peaks.bed'), sep='\t', header=False, index=False)


## find shared TF-TG pairs across all celltypes and sexes
for sex in unique_sexes:
    sex = sex.lower()
    for celltype in unique_celltypes:
        unique_TF_TG_combinations_df = mean_grn_df_filtered_pruned_dict[sex][celltype][['TF', 'TG']].drop_duplicates()
        unique_TF_TG_combinations_str = unique_TF_TG_combinations_df.apply(lambda x: ' - '.join(x), axis=1).values
        unique_TF_TG_combinations_dict[sex][celltype] = unique_TF_TG_combinations_str

        pairs = set(unique_TF_TG_combinations_dict[sex][celltype])
        
        if not shared_TF_TG_pairs:
            shared_TF_TG_pairs = pairs
        else:
            shared_TF_TG_pairs = shared_TF_TG_pairs.intersection(pairs)

        all_TF_TG_pairs = all_TF_TG_pairs.union(pairs)

shared_TF_TG_pairs_df = pd.DataFrame(shared_TF_TG_pairs).iloc[:, 0].str.split(' - ', expand=True)
shared_TF_TG_pairs_df.columns = ['TF', 'TG']
shared_TF_TG_pairs_df.sort_values(by='TG', inplace=True)
shared_TF_TG_pairs_df.to_csv(os.path.join(output_dir, 'shared_TF_TG_pairs.csv'), index=False)

print(f'Shared TF-TG pairs (n={len(shared_TF_TG_pairs)} out of {len(all_TF_TG_pairs)}):')
print(shared_TF_TG_pairs_df)

shared_TF_TG_pairs_df_grouped = shared_TF_TG_pairs_df.groupby('TF').agg({
    'TG': [list, 'nunique'],
}).sort_values(by=('TG', 'nunique'), ascending=False)

## Perform enrichment analysis on TG regulons of TFs that are enriched in shared TF-TG pairs
shared_TF_TG_pairs_df_grouped_filtered = shared_TF_TG_pairs_df_grouped[shared_TF_TG_pairs_df_grouped['TG','nunique'] > 1] # not interesting to find TFs with only one regulon gene

for TF in shared_TF_TG_pairs_df_grouped_filtered.index:

    TF_TG_pairs = shared_TF_TG_pairs_df_grouped.loc[TF, ('TG', 'list')]
    TF_TG_pairs_series = pd.Series(TF, index=TF_TG_pairs)
    TF_TG_pairs_series.attrs = {'sex':'all', 'celltype':'all', 'type': 'TF-TG pairs'}

    enrichr_results_sig = do_enrichr(TF_TG_pairs_series, 'ChEA_2022', filter_var='P-value', outdir=None) # only looking for specific TF, so no need to correct for multiple testing

    if enrichr_results_sig is not None:

        #enrichr_results_sig = enrichr_results_sig[enrichr_results_sig['P-value'] < 0.05]
        enriched_tfs = enrichr_results_sig['Term'].str.split(' ').str[0]
        enriched_species = enrichr_results_sig['Term'].str.split(' ').str[-1]
        enriched_tfs_match_TF = np.isin(enriched_tfs, TF) & np.isin(enriched_species, 'Human')
    
        if enriched_tfs_match_TF.any():
            enriched_tfs_match_TF_list = enrichr_results_sig[enriched_tfs_match_TF]['Genes'].str.split(';').item()

            if len(enriched_tfs_match_TF_list) >= 2: # at least 2 TFs should be enriched for the same TG to study interesting TFs and their regulons
                enriched_TF_TG_pairs_dict[TF] = enriched_tfs_match_TF_list

## write enriched_TF_TG_pairs_dict to json
with open(os.path.join(output_dir, 'enriched_TF_TG_pairs_dict.json'), 'w') as f:
    json.dump(enriched_TF_TG_pairs_dict, f)


#%% investigate EGR1 and NR4A2

TFs_of_EGR1 = mean_grn_df[mean_grn_df['TG'] == 'EGR1']['TF'].to_list()

hit1 = 'NR4A2'
assert hit1 in TFs_of_EGR1

#NR4A2_targets_ExN_male = mean_grn_df_filtered_dict['male']['ExN'].loc[mean_grn_df_filtered_dict['male']['ExN']['TF'] == 'NR4A2']
grn_female_exn = mean_grn_df_filtered_pruned_dict['female']['ExN']
NR4A2_targets_ExN_female = grn_female_exn[grn_female_exn['TF']==hit1]
EGR1_targets_ExN_female = grn_female_exn[grn_female_exn['TF']=='EGR1']
SOX2_targets_ExN_female = grn_female_exn[grn_female_exn['TF']=='SOX2']

hit2 = 'ABHD17B'
assert hit2 in NR4A2_targets_ExN_female['TG'].to_list()

enriched_TFs = np.array(list(enriched_TF_TG_pairs_dict.keys()))
female_exn_TFs_of_ABHD17B = grn_female_exn[grn_female_exn['TG']==hit2]['TF'].values
female_exn_enriched_TFs_of_ABHD17B = enriched_TFs[np.isin(enriched_TFs, female_exn_TFs_of_ABHD17B)]

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import networkx as nx

G = nx.DiGraph()

edge_color_map = {
    'a priori': 'gray',
    'female_ExN': 'blue',
    'all': 'green'
}

G.add_edge('NR4A2', 'EGR1', interaction='a priori', color=edge_color_map['a priori'])
G.add_edge('NR4A2', 'ABHD17B', interaction='female_ExN', color=edge_color_map['female_ExN'])
G.add_edge('EGR1', 'ABHD17B', interaction='female_ExN', color=edge_color_map['female_ExN'])
G.add_edge('SOX2', 'ABHD17B', interaction='female_ExN', color=edge_color_map['female_ExN'])

all_targets = []
for tf in female_exn_enriched_TFs_of_ABHD17B:
    tf_targets = enriched_TF_TG_pairs_dict[tf]
    for tf_target in tf_targets:
        G.add_edge(tf, tf_target, interaction='all', color=edge_color_map['all'])
        all_targets.append(tf_target)

# Assign layer information to each node
for node in G.nodes():
    if node == 'NR4A2':
        G.nodes[node]['layer'] = 0
    elif node in ['EGR1', 'SOX2']:
        G.nodes[node]['layer'] = 1
    elif node == 'ABHD17B':
        G.nodes[node]['layer'] = 2
    elif node in all_targets:
        G.nodes[node]['layer'] = 3

pos = nx.multipartite_layout(G, subset_key='layer', scale=2)

for egr1_target in enriched_TF_TG_pairs_dict['EGR1']:
    pos[egr1_target] += np.array([0, 0.4])

colors = nx.get_edge_attributes(G, 'color').values()

fig, ax = plt.subplots(figsize=(4, 4))

nx.draw_networkx_nodes(G, pos,
    node_size=1200,
    node_color='lightgrey',
    edgecolors='k',
    ax=ax
)
nx.draw_networkx_edges(G, pos,
    arrowstyle='-|>',
    arrowsize=15,
    width=2,
    edge_color=list(nx.get_edge_attributes(G, 'color').values()),
    min_source_margin=0.05,
    min_target_margin=0.05,
    ax=ax
)

# 2) Draw all labels *except* the special one at your default size
labels = {n: n for n in G.nodes() if n != 'ABHD17B'}
nx.draw_networkx_labels(G, pos, labels,
    font_size=8,       # default for everyone else
    font_color='black',
    ax=ax
)

# 3) Finally, over‐draw the one with a custom fontsize:
x, y = pos['ABHD17B']
plt.text(
    x, y,
    'ABHD17B',
    fontsize=6,        # big label
    fontweight='bold',  # optional styling
    ha='center',
    va='center'
)

axins = inset_axes(ax,
                   width="40%",    # width = 20% of parent_bbox width
                   height="20%",   # height= 20%
                   loc='lower left',
                   borderpad=1)

H = nx.DiGraph()
H.add_edge('TF','TG')

pos2 = {'TF': (0.4, 0.5),
        'TG': (0.6, 0.5)}

nx.draw_networkx_nodes(H, pos2, node_size=800, node_color='lightgrey', edgecolors='k', ax=axins)
nx.draw_networkx_edges(H, pos2, arrowstyle='-|>', arrowsize=20, edge_color='k',min_source_margin=0.2, min_target_margin=0.2, ax=axins)
nx.draw_networkx_labels(H, pos2, font_size=10, ax=axins)

# Add legend
legend_handles = [
    mpatches.Patch(color=edge_color_map['a priori'], label='a priori'),
    mpatches.Patch(color=edge_color_map['female_ExN'], label='female_ExN'),
    mpatches.Patch(color=edge_color_map['all'], label='all')
]
ax.legend(handles=legend_handles, title='relevant group', loc='upper left')

axins.set_xlim(0.3, 0.7)
axins.set_ylim(0.3, 0.7)

axins.axis('off')
ax.axis('off')

plt.tight_layout()
plt.show()

#%% plot genome track around

import scglue

female_exn_grn = mean_grn_df_filtered_dict['female']['ExN']

gene_ad = anndata.AnnData(var=pd.DataFrame(index=female_exn_grn['TG'].drop_duplicates().to_list()))
scglue.data.get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')

genes = scglue.genomics.Bed(gene_ad.var)
tss = genes.strand_specific_start_site()

ABHD17B_grn_flag = (female_exn_grn['TG']=='ABHD17B') & female_exn_grn['TF'].isin(['NR4A2', 'EGR1', 'SOX2'])
ABHD17B_grn_flag = ABHD17B_grn_flag.values

ABHD17B_grn = female_exn_grn[ABHD17B_grn_flag]
tss_abhd17b = tss.loc[['ABHD17B']]
peaks_abhd17b = scglue.genomics.Bed(ABHD17B_grn.assign(name=ABHD17B_grn['enhancer']))

## create graph from mean_grn_df_filtered with lrCorr as edge weight
mean_grn_filtered_graph = nx.from_pandas_edgelist(
    ABHD17B_grn,
    source='TG',
    target='enhancer',
    edge_attr='lrCorr',
    create_using=nx.DiGraph())

peaks_abhd17b.write_bed(os.path.join(os.environ['OUTPATH'], 'peaks_abhd17b.bed'))

scglue.genomics.write_links(
    mean_grn_filtered_graph,
    tss_abhd17b,
    peaks_abhd17b,
    os.path.join(os.environ['OUTPATH'], 'gene2peak_lrCorr_abhd17b.links'),
    keep_attrs=["lrCorr"]
    )

gene = 'ABHD17B'
tg_grn = ABHD17B_grn[ABHD17B_grn['TG']==gene].sort_values('dist')[['enhancer','dist']].groupby('enhancer').mean()
tg_grn_bounds = np.stack(tg_grn.index.str.split(':|-')).flatten()
tg_grn_bounds = [int(bound) for bound in tg_grn_bounds if bound.isdigit()] + [genes.loc[gene, 'chromStart']] + [genes.loc[gene, 'chromEnd']]

print(tg_grn)
print(f'{genes.loc[gene, "chrom"]}:{min(tg_grn_bounds)}-{max(tg_grn_bounds)}')
print(genes.loc[gene,['chrom','chromStart','chromEnd','name']])

import shutil

# Copy tracks.ini to the output directory
shutil.copy(
    os.path.join(os.environ['OUTPATH'], "tracks.ini"),
    os.path.join(output_dir, "tracks_abhd17b.ini")
)

shutil.copy(
    os.path.join(os.environ['OUTPATH'], "tracks.ini"),
    os.path.join(os.environ['OUTPATH'], "tracks_abhd17b.ini")
)

#!pyGenomeTracks --tracks tracks_abhd17b.ini --region chr9:71700000-72000000 -o tracks_abhd17b.png


#%% module scores for enriched pathways

gene_set_scores_dict = tree()
ttest_comp_df_dict = tree()
enrs_mdd_dn_list = []

for sex in unique_sexes:
    sex = sex.lower()
    for celltype in unique_celltypes:

        enrs_scompreg = enrs_dict[sex][celltype]['All LR']

        ## extract MDD-DN pathway
        enrs_mdd_dn = enrs_scompreg[enrs_scompreg.Term == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']
        if len(enrs_mdd_dn) > 0:
            enrs_mdd_dn.index = [f'{sex} - {celltype}']
            enrs_mdd_dn = enrs_mdd_dn.assign(sex=sex, celltype=celltype)
            enrs_mdd_dn_list.append(enrs_mdd_dn)

        if len(enrs_scompreg) == 0:
            logger.warning(f"No enrichment results found for sex: {sex}, celltype: {celltype}")
            continue

        ## compute module scores for all pathways
        for condition in unique_conditions:
            adata = X_rna_dict[sex][celltype][condition]

            for gene_set in enrs_scompreg.itertuples():
                term = gene_set.Term
                genes = gene_set.Genes.split(';')
                term_scores = score_genes(adata, gene_list=genes, score_name=term, copy=True)
                #weighted_term_score = (term_scores.obs[term].values * term_scores.obs['proportion_of_cells']).sum()
                weighted_term_score = DescrStatsW(term_scores.obs[term].values, weights=term_scores.obs['proportion_of_cells'].values)
                gene_set_scores_dict[sex][celltype][condition][term] = weighted_term_score

        ## compute weighted t-tests for all pathways based on module scores
        ttest_comp_df_dict_list = []
        for term in enrs_scompreg.Term:
            d1 = gene_set_scores_dict[sex][celltype]['Case'][term]
            d2 = gene_set_scores_dict[sex][celltype]['Control'][term]

            ## case vs 0
            tstat, pval, df = d1.ttest_mean(0)

            ## case vs control
            cm = d1.get_compare(d2)
            tstat_comp, pvalue_comp, df_comp = cm.ttest_ind(usevar='unequal', alternative='two-sided')
            ttest_comp_df = pd.DataFrame({'tstat': tstat_comp, 'pvalue': pvalue_comp, 'df': df_comp}, index=[term])
            ttest_comp_df_dict_list.append(ttest_comp_df)

        ttest_comp_df_dict[sex][celltype] = pd.concat(ttest_comp_df_dict_list)


dicts_to_save = {
    'gene_set_scores_dict': gene_set_scores_dict,
    'ttest_comp_df_dict': ttest_comp_df_dict,
}

for dict_name, dict_obj in dicts_to_save.items():
    with open(os.path.join(output_dir, f"{dict_name}.pkl"), "wb") as f:
        pickle.dump(dict_obj, f)
    print(f"Saved {dict_name}")

#%% Aggregated EnrichR results

## EnrichR based on MDD-DN genes across all celltypes and sexes
enrs_mdd_dn_df = pd.concat(enrs_mdd_dn_list)
enrs_mdd_dn_genes = np.unique(np.hstack(enrs_mdd_dn_df['Genes'].apply(lambda x: x.split(';')).values))
enrs_mdd_dn_genes_series = pd.Series(enrs_mdd_dn_genes, index=enrs_mdd_dn_genes)
enrs_mdd_dn_genes_series.attrs = {'sex': 'all', 'celltype': 'all', 'type': 'MDD-DN_genes'}
print(f'Number of MDD-DN overlapping genes: {len(enrs_mdd_dn_genes_series)}')

enrs_mdd_dn_genes_enrichr = do_enrichr(enrs_mdd_dn_genes_series, brain_gmt_cortical, outdir=output_dir,
                                       remove_from_dotplot=['ASTON_MAJOR_DEPRESSIVE_DISORDER_DN'])

## search genes with largest overlap across enriched pathways
enrs_mdd_dn_genes_shared_TF_TG_pairs_df = shared_TF_TG_pairs_df[shared_TF_TG_pairs_df['TG'].isin(enrs_mdd_dn_genes)]
enrs_mdd_dn_genes_shared = enrs_mdd_dn_genes_shared_TF_TG_pairs_df['TG'].unique()

if len(enrs_mdd_dn_genes_shared) > 0:
    enrs_mdd_dn_hits_df, enrs_mdd_dn_tfs_multiple_hits = find_hits_overlap(enrs_mdd_dn_genes_shared, enrs_mdd_dn_genes_enrichr, shared_TF_TG_pairs_df)
    enrs_mdd_dn_hits_df.to_csv(os.path.join(output_dir, 'enrs_mdd_dn_hits_df.csv'), index=True, header=True)
    enrs_mdd_dn_tfs_multiple_hits.to_csv(os.path.join(output_dir, 'enrs_mdd_dn_tfs_multiple_hits.csv'), index=True, header=True)

## EnrichR based on all genes filtered using sc-compReg across all celltypes and sexes for which MDD-DN was significant
all_sccompreg_genes = np.unique(np.hstack([
    mean_grn_df_filtered_dict[sex][celltype]['TG'].unique() for sex in unique_sexes for celltype in unique_celltypes
    if enrs_mdd_dn_df['sex'].isin([sex]).any() and enrs_mdd_dn_df['celltype'].isin([celltype]).any()
    ]))

all_sccompreg_genes_series = pd.Series(all_sccompreg_genes, index=all_sccompreg_genes)
all_sccompreg_genes_series.attrs = {'sex': 'all', 'celltype': 'all', 'type': 'sc-compReg_filtered_genes'}

all_sccompreg_genes_enrichr = do_enrichr(all_sccompreg_genes_series, brain_gmt_cortical, outdir=output_dir, figsize=(3,9))

all_sccompreg_genes_shared_TF_TG_pairs_df = shared_TF_TG_pairs_df[shared_TF_TG_pairs_df['TG'].isin(all_sccompreg_genes)]
all_sccompreg_genes_shared = all_sccompreg_genes_shared_TF_TG_pairs_df['TG'].unique()

if len(all_sccompreg_genes_shared) > 0:
    all_sccompreg_hits_df, all_sccompreg_tfs_multiple_hits = find_hits_overlap(all_sccompreg_genes_shared, all_sccompreg_genes_enrichr, all_sccompreg_genes_shared_TF_TG_pairs_df)
    all_sccompreg_tfs_multiple_hits.to_csv(os.path.join(output_dir, 'all_sccompreg_tfs_multiple_hits.csv'), index=True, header=True)
    all_sccompreg_hits_df.to_csv(os.path.join(output_dir, 'all_sccompreg_hits_df.csv'), index=True, header=True)

## get all pyDESeq2 significant genes from significant_genes_dict
#all_significant_genes = np.unique(np.hstack([significant_genes_dict[sex][celltype].values for sex in unique_sexes for celltype in unique_celltypes]))
all_pydeseq2_padj = [pydeseq2_results_dict[sex][celltype]['padj'] for sex in unique_sexes for celltype in unique_celltypes]
all_pydeseq2_padj_series = pd.concat(all_pydeseq2_padj).dropna()
all_pydeseq2_padj_series.sort_values(inplace=True)
all_pydeseq2_padj_series.drop_duplicates(keep='first', inplace=True)

#pydeseq2_match_length_idxs = all_pydeseq2_padj_series.loc[all_pydeseq2_padj_series < 0.05]
#pydeseq2_match_length_idxs = all_pydeseq2_padj_series.iloc[:len(enrs_mdd_dn_genes)]
pydeseq2_match_length_idxs = all_pydeseq2_padj_series.iloc[:len(all_sccompreg_genes)]

pydeseq2_match_length_idxs = pydeseq2_match_length_idxs.index.values.astype(int)
pydeseq2_match_length_genes = mdd_rna.var_names[pydeseq2_match_length_idxs]

pydeseq2_match_length_genes_series = pd.Series(pydeseq2_match_length_genes, index=pydeseq2_match_length_genes)
pydeseq2_match_length_genes_series.attrs = {'sex': 'all', 'celltype': 'all', 'type': 'pyDESeq2_significant_genes'}

pydeseq2_match_length_genes_enrichr = do_enrichr(pydeseq2_match_length_genes_series, brain_gmt_cortical, outdir=output_dir)

pydeseq2_match_length_genes_shared_TF_TG_pairs_df = shared_TF_TG_pairs_df[shared_TF_TG_pairs_df['TG'].isin(pydeseq2_match_length_genes)]
pydeseq2_match_length_genes_shared = pydeseq2_match_length_genes_shared_TF_TG_pairs_df['TG'].unique()

if len(pydeseq2_match_length_genes_shared) > 0:
    pydeseq2_match_length_genes_hits_df, pydeseq2_match_length_genes_tfs_multiple_hits = find_hits_overlap(pydeseq2_match_length_genes_shared, pydeseq2_match_length_genes_enrichr, pydeseq2_match_length_genes_shared_TF_TG_pairs_df)
    pydeseq2_match_length_genes_hits_df.to_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_hits_df.csv'), index=True, header=True)
    pydeseq2_match_length_genes_tfs_multiple_hits.to_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_tfs_multiple_hits.csv'), index=True, header=True)

        
# %%
print ('Done!')