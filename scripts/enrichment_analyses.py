#%% set env variables
from eclare import set_env_variables
set_env_variables(config_path='../config')

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

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
import scanpy as sc
import gseapy as gp
import networkx as nx

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
from tqdm import tqdm

from eclare.post_hoc_utils import \
    extract_target_source_replicate, initialize_dicts, assign_to_dicts, perform_gene_set_enrichment, differential_grn_analysis, process_celltype, load_model_and_metadata, get_brain_gmt, magma_dicts_to_df, get_next_version_dir, compute_LR_grns, do_enrichr, find_hits_overlap, \
    download_mlflow_runs,\
    tree, get_latents

cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'

#%% set method IDs and output directory

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
decimate_factor = 5

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

#%% UMAP plots of RNA and ATAC data

latents_rna, latents_atac = get_latents(eclare_student_model, mdd_rna, mdd_atac, return_tensor=False)

rna_adata = anndata.AnnData(
    X=latents_rna,
    obs=mdd_rna.obs,
)
atac_adata = anndata.AnnData(
    X=latents_atac,
    obs=mdd_atac.obs,
)

rna_adata.obs = rna_adata.obs.assign(modality='RNA')
atac_adata.obs = atac_adata.obs.assign(modality='ATAC')
rna_atac_adata = anndata.concat([rna_adata, atac_adata], axis=0)
rna_atac_adata = rna_atac_adata[np.random.permutation(rna_atac_adata.shape[0])]
rna_atac_adata.obs.rename(columns={'ClustersMapped': 'cell type'}, inplace=True)

sc.pp.pca(rna_atac_adata)
sc.pp.neighbors(rna_atac_adata, n_neighbors=50, n_pcs=50)
sc.tl.umap(rna_atac_adata, min_dist=0.25)

sc.pl.umap(rna_atac_adata, color=['modality', 'cell type'], wspace=0.25, size=5)
#sc.pl.umap(rna_atac_adata[rna_atac_adata.obs['modality'].eq('RNA')], color=['ClustersMapped'], wspace=0.3, size=3)
#sc.pl.umap(rna_atac_adata[rna_atac_adata.obs['modality'].eq('ATAC')], color=['ClustersMapped'], wspace=0.3, size=3)

def mdd_umap_plot(rna_atac_adata, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'enrichment_analyses')):

    sc.settings._vector_friendly = True
    sc.settings.figdir = manuscript_figpath
    sc.pl.umap(rna_atac_adata, color=['modality', 'cell type'], wspace=0.25, size=5, save='_eclare_mdd.svg')

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

    # Create progress bar
    pbar = tqdm(total=len(unique_celltypes), desc=f"Computing LR GRNs ({sex})")
    
    # Wrapper function to update progress bar
    def compute_with_progress(*args, **kwargs):
        result = compute_LR_grns(*args, **kwargs)
        pbar.update(1)
        return result
    
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(compute_with_progress)(sex, celltype, mean_grn_df_filtered_dict, X_rna_dict, X_atac_dict, output_dir=output_dir)
        for celltype in unique_celltypes
    )
    
    pbar.close()

    for celltype, result in zip(unique_celltypes, results):
        mean_grn_df_filtered_pruned_dict[sex][celltype] = result

## save mean_grn_df_filtered_pruned_dict
with open(os.path.join(output_dir, 'mean_grn_df_filtered_pruned_dict.pkl'), 'wb') as f:
    pickle.dump(mean_grn_df_filtered_pruned_dict, f)

## save the different GRN results as separate tabs of an Excel file
keep_cols = ['TF', 'enhancer', 'TG', 'LR', 'LR_grns']
with pd.ExcelWriter(os.path.join(output_dir, 'grn_results.xlsx')) as writer:
    for sex in unique_sexes:
        sex = sex.lower()
        for celltype in unique_celltypes:
            mean_grn_df_filtered_pruned = mean_grn_df_filtered_pruned_dict[sex][celltype]
            mean_grn_df_filtered_pruned = mean_grn_df_filtered_pruned[keep_cols]
            mean_grn_df_filtered_pruned = mean_grn_df_filtered_pruned.reindex(columns=['TG', 'enhancer', 'TF', 'LR', 'LR_grns'])
            mean_grn_df_filtered_pruned.to_excel(writer, sheet_name=f'{sex}_{celltype}', index=False)
            print(f'Saved {sex}_{celltype} to Excel file')
            #writer.save()
print(f'Saved GRN results to Excel file to {os.path.join(output_dir, "grn_results.xlsx")}')


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


#%% find shared TF-TG pairs across all celltypes and sexes

'''
output_dir = os.path.join(os.environ['OUTPATH'], 'enrichment_analyses_16103846_41')
with open(os.path.join(output_dir, 'mean_grn_df_filtered_pruned_dict.pkl'), 'rb') as f:
    mean_grn_df_filtered_pruned_dict = pickle.load(f)
with open(os.path.join(output_dir, 'mean_grn_df_filtered_dict.pkl'), 'rb') as f:
    mean_grn_df_filtered_dict = pickle.load(f)
unique_sexes = ['female', 'male']
unique_celltypes = list(mean_grn_df_filtered_dict[unique_sexes[0]].keys())
'''

## flatten mean_grn_df_filtered_dict
mean_grn_df_filtered_dict_flattened = []
mean_grn_df_filtered_pruned_dict_flattened = []

for sex in unique_sexes:
    sex = sex.lower()
    for celltype in unique_celltypes:
        mean_grn_df_filtered_dict_flattened.append(
            mean_grn_df_filtered_dict[sex][celltype].assign(group=f'{sex}_{celltype}'))
        mean_grn_df_filtered_pruned_dict_flattened.append(
            mean_grn_df_filtered_pruned_dict[sex][celltype].assign(group=f'{sex}_{celltype}'))

mean_grn_df_filtered_dict_flattened = pd.concat(mean_grn_df_filtered_dict_flattened)
mean_grn_df_filtered_pruned_dict_flattened = pd.concat(mean_grn_df_filtered_pruned_dict_flattened)

## get shared TF-TG pairs across female ExN and Oli
#female_ExN_Oli = mean_grn_df_filtered_pruned_dict_flattened.loc[mean_grn_df_filtered_pruned_dict_flattened['group'].isin(['female_ExN', 'female_Oli'])]
female_ExN_Oli = mean_grn_df_filtered_dict_flattened.loc[mean_grn_df_filtered_dict_flattened['group'].isin(['female_ExN', 'female_Oli'])]
female_ExN_Oli_counts = female_ExN_Oli.groupby(['TF','TG'])['group'].nunique()
female_ExN_Oli_shared_TF_TG_pairs = female_ExN_Oli_counts.loc[female_ExN_Oli_counts.eq(2)].reset_index('TG').drop(columns='group')

female_ExN_Oli_shared_TF_TG_pairs_grouped = female_ExN_Oli_shared_TF_TG_pairs.reset_index().groupby('TF').agg({
    'TG': [list, 'nunique']
}).sort_values(by=('TG', 'nunique'), ascending=False)

female_ExN_Oli_EGR1_SOX2 = np.intersect1d(female_ExN_Oli_shared_TF_TG_pairs_grouped.loc['EGR1', ('TG', 'list')], female_ExN_Oli_shared_TF_TG_pairs_grouped.loc['SOX2', ('TG', 'list')])
female_ExN_Oli_EGR1_SOX2_NR4A2 = np.intersect1d(female_ExN_Oli_EGR1_SOX2, female_ExN_Oli_shared_TF_TG_pairs_grouped.loc['NR4A2', ('TG', 'list')])

## get shared TF-TG pairs across male ExN and Oli
male_ExN_Oli = mean_grn_df_filtered_dict_flattened.loc[mean_grn_df_filtered_dict_flattened['group'].isin(['male_ExN', 'male_Oli'])]
male_ExN_Oli_counts = male_ExN_Oli.groupby(['TF','TG'])['group'].nunique()
male_ExN_Oli_shared_TF_TG_pairs = male_ExN_Oli_counts.loc[male_ExN_Oli_counts.eq(2)].reset_index('TG').drop(columns='group')

male_ExN_Oli_shared_TF_TG_pairs_grouped = male_ExN_Oli_shared_TF_TG_pairs.reset_index().groupby('TF').agg({
    'TG': [list, 'nunique']
}).sort_values(by=('TG', 'nunique'), ascending=False)

male_ExN_Oli_EGR1_SOX2 = np.intersect1d(male_ExN_Oli_shared_TF_TG_pairs_grouped.loc['EGR1', ('TG', 'list')], male_ExN_Oli_shared_TF_TG_pairs_grouped.loc['SOX2', ('TG', 'list')])
male_ExN_Oli_EGR1_SOX2_NR4A2 = np.intersect1d(male_ExN_Oli_EGR1_SOX2, male_ExN_Oli_shared_TF_TG_pairs_grouped.loc['NR4A2', ('TG', 'list')])

## get shared TF-TG pairs across ExN and Oli across sexes
ExN_Oli_EGR1_SOX2 = np.intersect1d(female_ExN_Oli_EGR1_SOX2, male_ExN_Oli_EGR1_SOX2)
ExN_Oli_EGR1_SOX2_NR4A2 = np.intersect1d(ExN_Oli_EGR1_SOX2, female_ExN_Oli_EGR1_SOX2_NR4A2, male_ExN_Oli_EGR1_SOX2_NR4A2)

'''
with open(os.path.join(output_dir, 'broad_gene_series_dict.pkl'), 'rb') as f: res_dict = pickle.load(f)
enrs_mdd_dn_genes_series = res_dict['enrs_mdd_dn_genes_series']

nr4a2_exn_hits_df = mean_grn_df_filtered_pruned_dict_flattened.loc[
    mean_grn_df_filtered_pruned_dict_flattened['TF'].isin(['NR4A2']) &
    mean_grn_df_filtered_pruned_dict_flattened['group'].str.contains('ExN')
    ].loc[:,['TF','TG','group']].set_index('group')

assert (nr4a2_exn_hits_df['TG'].unique().item() == 'ABHD17B') # show that only ABHD17B as target for NR4A2 in ExN
assert (nr4a2_exn_hits_df.index.unique().item() == 'female_ExN') # show that only female ExN

abhd17b_exn_hits_df = mean_grn_df_filtered_pruned_dict_flattened.loc[
    mean_grn_df_filtered_pruned_dict_flattened['TG'].isin(['ABHD17B']) &
    mean_grn_df_filtered_pruned_dict_flattened['group'].eq('female_ExN')
    ].loc[:,['TF','TG','group','enhancer']].set_index('group')

abhd17b_exn_skip1_hits_df = mean_grn_df_filtered_pruned_dict_flattened.loc[
    mean_grn_df_filtered_pruned_dict_flattened['TF'].isin(abhd17b_exn_hits_df['TF']) &
    mean_grn_df_filtered_pruned_dict_flattened['group'].eq('female_ExN')
    ].loc[:,['TF','TG','group','enhancer']].set_index('group')

adjacency_matrix = pd.get_dummies(abhd17b_exn_skip1_hits_df.set_index('TF')['TG']).astype(int)
adjacency_matrix = adjacency_matrix.groupby(adjacency_matrix.index).sum()
adjacency_matrix_binary = adjacency_matrix.applymap(lambda x: 1 if x > 0 else 0)

#adjacency_matrix_binary_trunc = adjacency_matrix_binary.loc[:, adjacency_matrix_binary.columns.isin(['ABHD17B'] + mdd_genes)]
adjacency_matrix_binary_trunc = adjacency_matrix_binary.copy()
adjacency_matrix_binary_trunc = adjacency_matrix_binary_trunc.loc[adjacency_matrix_binary_trunc.sum(1).ge(2) | adjacency_matrix_binary_trunc.index.isin(['NR3C1','NR4A2'])]
adjacency_matrix_binary_trunc = adjacency_matrix_binary_trunc.loc[:,adjacency_matrix_binary_trunc.sum(0).ge(2)]

def protected_k_core(G, k, whitelist):
    """
    Standard k-core algorithm that refuses to delete nodes in the whitelist.
    """
    H = G.copy()
    whitelist = set(whitelist)
    
    while True:
        # Find nodes to remove: degree < k AND not in whitelist
        nodes_to_remove = [
            n for n, d in H.degree() 
            if d < k and n not in whitelist
        ]
        
        if not nodes_to_remove:
            break
            
        H.remove_nodes_from(nodes_to_remove)
        
    return H

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralCoclustering
from networkx.algorithms import bipartite
from scipy.sparse.linalg import svds

def cluster_bipartite_graph(G, n_clusters=8, degree_threshold=0.9):
    # 1. Get sets and matrix (original, full)
    sets = bipartite.sets(G)
    top_nodes, bottom_nodes = list(sets[0]), list(sets[1])
    matrix = bipartite.biadjacency_matrix(
        G, row_order=top_nodes, column_order=bottom_nodes
    ).toarray()

    n_rows, n_cols = matrix.shape

    # 2. Identify dense columns (a.k.a. "full vertical bars")
    col_densities = matrix.sum(axis=0) / n_rows
    dense_cols_mask = col_densities > degree_threshold

    dense_col_idx = np.flatnonzero(dense_cols_mask)       # indices in ORIGINAL matrix
    keep_col_idx  = np.flatnonzero(~dense_cols_mask)

    # Filter columns (for fitting only)
    matrix_filtered = matrix[:, keep_col_idx]

    # Rows that become empty after removing dense columns ("singleton rows")
    row_sums = matrix_filtered.sum(axis=1)
    valid_rows_mask = row_sums > 0

    singleton_row_idx = np.flatnonzero(~valid_rows_mask)  # indices in ORIGINAL matrix
    keep_row_idx      = np.flatnonzero(valid_rows_mask)

    # Final matrix for fitting (no dense cols, no empty rows)
    matrix_final = matrix[np.ix_(keep_row_idx, keep_col_idx)]

    print(f"Removed {dense_col_idx.size} dense columns.")
    print(f"Removed {singleton_row_idx.size} rows that became empty.")

    # plot singular values
    u, s, vt = svds(matrix_final.astype(float), k=min(matrix_final.shape) - 1)
    plt.plot(sorted(s, reverse=True), "o-")
    plt.title("Singular Values (Look for the Elbow)")
    plt.show()

    # 3. Perform Co-Clustering (on filtered matrix)
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(matrix_final)

    # Build node->cluster maps in ORIGINAL node space
    # cluster id -1 means "special": singleton rows or dense columns (excluded from fit)
    top_cluster = np.full(len(top_nodes), -1, dtype=int)
    bottom_cluster = np.full(len(bottom_nodes), -1, dtype=int)

    top_cluster[keep_row_idx] = model.row_labels_.astype(int)
    bottom_cluster[keep_col_idx] = model.column_labels_.astype(int)

    tf_cluster_map = {top_nodes[i]: int(top_cluster[i]) for i in range(len(top_nodes))}
    tg_cluster_map = {bottom_nodes[j]: int(bottom_cluster[j]) for j in range(len(bottom_nodes))}

    # 4. Cluster-based ordering (ONLY for the kept rows/cols)
    ordered_keep_rows = keep_row_idx[np.argsort(model.row_labels_)]
    ordered_keep_cols = keep_col_idx[np.argsort(model.column_labels_)]

    # Optional: order dense columns by density (most dense first). Otherwise keep original order.
    dense_col_idx = dense_col_idx[np.argsort(-col_densities[dense_col_idx])]

    # 5. Reintegrate: build FULL row/col order in ORIGINAL index space
    row_order_full = np.concatenate([singleton_row_idx, ordered_keep_rows])
    col_order_full = np.concatenate([dense_col_idx, ordered_keep_cols])

    final_matrix = matrix[np.ix_(row_order_full, col_order_full)]

    # Node labels in the new order
    final_row_nodes = [top_nodes[i] for i in row_order_full]
    final_col_nodes = [bottom_nodes[j] for j in col_order_full]

    # 6. Plot (original vs reintegrated+reordered)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.spy(matrix, aspect="auto")
    ax1.set_title(f"Original Matrix\n({matrix.shape[0]}x{matrix.shape[1]})")
    ax2.spy(final_matrix, aspect="auto")
    ax2.set_title("Final Matrix (Singleton rows first, Dense cols first,\nthen co-clustered remainder)")
    plt.tight_layout()
    plt.show()

    # cast adjacency matrix to DataFrame
    final_matrix = pd.DataFrame(final_matrix, index=final_row_nodes, columns=final_col_nodes)

    # cast as NetworkX graph (and ATTACH cluster labels to nodes)
    G2 = nx.DiGraph()
    for tf in final_matrix.index:
        G2.add_node(tf, node_type="TF", cluster=tf_cluster_map.get(tf, -1))
    for tg in final_matrix.columns:
        G2.add_node(tg, node_type="TG", cluster=tg_cluster_map.get(tg, -1))

    for tf in final_matrix.index:
        for tg in final_matrix.columns:
            if final_matrix.loc[tf, tg] == 1:
                G2.add_edge(tf, tg)

    return G2

# -----------------------------
# Build initial graph from adjacency matrices
# -----------------------------
G = nx.DiGraph()
for tf in adjacency_matrix_binary_trunc.index:
    G.add_node(tf, node_type="TF")
for tg in adjacency_matrix_binary_trunc.columns:
    G.add_node(tg, node_type="TG")
for tf in adjacency_matrix_binary_trunc.index:
    for tg in adjacency_matrix_binary_trunc.columns:
        if adjacency_matrix_binary.loc[tf, tg] == 1:
            G.add_edge(tf, tg)

# Shrink graph
print(f"Original: {G.number_of_nodes()} nodes")
G = protected_k_core(G, k=5, whitelist=["NR3C1", "NR4A2"])
print(f"Shrunken: {G.number_of_nodes()} nodes")

# Cluster bipartite graph (clusters stored as node attribute "cluster")
n_clusters = 3
G = cluster_bipartite_graph(G, n_clusters=n_clusters)

# Separate nodes by type (for MARKER SHAPES)
tf_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "TF"]
tg_nodes_all = [n for n, d in G.nodes(data=True) if d.get("node_type") == "TG"]

# Exception node
special_tg = "ABHD17B"
tg_special = [special_tg] if special_tg in tg_nodes_all else []
tg_nodes = [n for n in tg_nodes_all if n != special_tg]

# Layout
pos = nx.shell_layout(G, nlist=[[special_tg], tf_nodes, tg_nodes], rotate=24.5)

# -----------------------------
# Node colors by cluster assignment
# -----------------------------
all_clusters = [d.get("cluster", -1) for _, d in G.nodes(data=True)]
k_eff = max([c for c in all_clusters if c is not None] + [-1]) + 1  # number of non-negative clusters
k_eff = max(k_eff, 1)

cmap = plt.cm.get_cmap("tab20", k_eff)

def node_color(n):
    c = G.nodes[n].get("cluster", -1)
    if c is None or c < 0:
        return (0.7, 0.7, 0.7, 0.9)  # special / excluded nodes
    return cmap(int(c))

tf_colors = [node_color(n) for n in tf_nodes]
tg_colors = [node_color(n) for n in tg_nodes]
tg_special_colors = [node_color(n) for n in tg_special]

# -----------------------------
# Edge colors by cocluster membership (within-bicluster edges colored; others gray)
# -----------------------------
edges = list(G.edges())
edge_colors = []
for u, v in edges:
    cu = G.nodes[u].get("cluster", -1)
    cv = G.nodes[v].get("cluster", -1)
    if cu is not None and cv is not None and cu >= 0 and cu == cv:
        edge_colors.append(cmap(int(cu)))
    else:
        edge_colors.append((0.6, 0.6, 0.6, 0.25))  # inter-cluster / special

# -----------------------------
# Plot
# Shapes: TF=diamond, TG=circle, ABHD17B=square
# Colors: cluster-based
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 10))

nx.draw_networkx_nodes(
    G, pos,
    nodelist=tf_nodes,
    node_color=tf_colors,
    node_shape="D",          # diamond
    node_size=900,
    alpha=0.95,
    linewidths=1.0,
    ax=ax,
    label="TF (diamond)"
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=tg_nodes,
    node_color=tg_colors,
    node_shape="o",          # circle
    node_size=800,
    alpha=0.95,
    linewidths=1.0,
    ax=ax,
    label="TG (circle)"
)

if tg_special:
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=tg_special,
        node_color=tg_special_colors,
        node_shape="s",      # square
        node_size=900,
        alpha=0.98,
        linewidths=1.2,
        ax=ax,
        label="ABHD17B (square)"
    )

nx.draw_networkx_edges(
    G, pos,
    edgelist=edges,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=10,
    connectionstyle="arc3,rad=-0.2",
    alpha=0.7,
    ax=ax
)

nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

ax.legend(loc="best")
ax.set_title("TF–TG Network Graph (node shape by type, node/edge color by cocluster)", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------
# Save graph data for HoloViews visualization
# -----------------------------
# Save all needed objects for the holoviews_grn_graph.py script
graph_data = {
    'G': G,
    'pos': pos,
    'tf_nodes': tf_nodes,
    'tg_nodes': tg_nodes,
    'tg_special': tg_special,
    'special_tg': special_tg
}
graph_pickle_path = os.path.join(os.environ['OUTPATH'], 'grn_graph_data.pkl')
with open(graph_pickle_path, 'wb') as f:
    pickle.dump(graph_data, f)
print(f"Saved graph data for HoloViews to: {graph_pickle_path}")
print(f"Run: python holoviews_grn_graph.py --input {graph_pickle_path}")

## get hits by enhancer for NR4A2-ABHD17B
hits_by_enhancer = (
    abhd17b_exn_skip1_hits_df
    .assign(TF_TG=lambda d: d['TF'].astype(str) + '-' + d['TG'].astype(str))
    .groupby('enhancer').agg({
        'TF_TG': ['unique', 'nunique']
    })
).sort_values(('TF_TG','nunique'), ascending=False)

hits_nr4a2_abhd17b_enhancer = \
hits_by_enhancer.loc[
    abhd17b_exn_skip1_hits_df.set_index('TF').loc['NR4A2'].iloc[1]
].loc[('TF_TG',  'unique')]
'''

'''
dat = mean_grn_df_filtered_dict_flattened.copy()
candidate_groups_canonical = ['female_ExN', 'female_Oli', 'male_ExN', 'male_Oli']
candidate_groups_mdd_hits = candidate_groups_canonical + ['female_End', 'male_End', 'female_InN']
candidate_groups_mdd_nominal = candidate_groups_mdd_hits + ['female_Ast', 'male_InN', 'male_OPC']

dat_candidate_groups = dat[dat['group'].isin(candidate_groups_mdd_nominal)]
tf_tg_group = dat_candidate_groups.groupby(['TF','TG']).agg({'group': ['unique', 'nunique']})
tf_tg_group_sorted = tf_tg_group.sort_values(by=('group','nunique'), ascending=False)
#tf_tg_group_sorted_candidate_groups = tf_tg_group_sorted.loc[tf_tg_group_sorted['group','unique'].apply(lambda x: any(np.isin(x, candidate_groups)))]
#tf_tg_group_sorted_candidate_groups.index.get_level_values(0).value_counts().head(20)

dat_candidate_groups_genes = dat[dat['group'].isin(candidate_groups_mdd_nominal) & dat['TG'].isin(enrs_mdd_dn_genes_series)]
dat_candidate_groups_genes_grouped = dat_candidate_groups_genes.groupby(['TF','TG']).agg({'group': ['unique', 'nunique']})
dat_candidate_groups_genes_grouped_sorted = dat_candidate_groups_genes_grouped.sort_values(by=('group','nunique'), ascending=False)
dat_candidate_groups_genes_grouped_sorted.index.get_level_values(0).value_counts().head(20)

mdd_genes = brain_gmt_cortical['ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']
dat_mdd = dat[dat['TG'].isin(mdd_genes)]
dat_mdd_hits = dat_mdd[dat_mdd['group'].isin(candidate_groups_mdd_hits)]
dat_mdd_grouped = dat_mdd_hits.groupby(['TF','TG']).agg({'group': ['unique', 'nunique']})
dat_mdd_grouped_sorted = dat_mdd_grouped.sort_values(by=('group','nunique'), ascending=False)
dat_mdd_grouped_sorted.index.get_level_values(0).value_counts().head(20)


tf_tg_group_filtered = mean_grn_df_filtered_dict_flattened.groupby(['TF','TG']).agg({'group': ['unique', 'nunique']})
tf_tg_group_sorted_filtered = tf_tg_group_filtered.sort_values(by=('group','nunique'), ascending=False)
tf_tg_group_sorted_filtered_candidate_groups = tf_tg_group_sorted_filtered.loc[tf_tg_group_sorted_filtered[('group','unique')].apply(lambda x: all(np.isin(candidate_groups_canonical, x)))]
#tf_tg_group_sorted_filtered_tops = tf_tg_group_sorted_filtered.loc[tf_tg_group_sorted_filtered['group','nunique'].eq(tf_tg_group_sorted_filtered['group','nunique'].max())]

tmp1 = tf_tg_group_sorted_filtered_candidate_groups.index.get_level_values(0).value_counts(); tmp1.name = 'num'
tmp2 = mean_grn_df['TF'].value_counts(); tmp2.name = 'denom'
tmp = pd.merge(tmp1, tmp2, left_index=True, right_index=True)
tf_rank = (tmp['num'] / np.log1p(tmp['denom'])).sort_values(ascending=False).to_frame()
tf_rank['rank'] = np.arange(len(tf_rank))
#LAPLACE: ((tmp['num'] + 1) / (len(mean_grn_df_filtered_dict_flattened) + mean_grn_df_filtered_dict_flattened['TF'].nunique())).sort_values(ascending=False).to_frame().head(12)

tf_tg_group_filtered_pruned = mean_grn_df_filtered_pruned_dict_flattened.groupby(['TF','TG']).agg({'group': ['unique', 'nunique']})
tf_tg_group_sorted_filtered_pruned = tf_tg_group_filtered_pruned.sort_values(by=('group','nunique'), ascending=False)
tf_tg_group_sorted_filtered_pruned_candidate_groups = tf_tg_group_sorted_filtered_pruned.loc[tf_tg_group_sorted_filtered_pruned[('group','unique')].apply(lambda x: any(np.isin(candidate_groups_canonical, x)))]

westside_hits = tf_tg_group_sorted_filtered_candidate_groups.loc[
    tf_tg_group_sorted_filtered_candidate_groups.index.isin(tf_tg_group_sorted_filtered_pruned_candidate_groups.index)
]
westside_hits_wtf_not = westside_hits.reset_index().droplevel(1, axis=1).loc[:,['TF','TG']].groupby('TG').agg({
    'TF': ('unique', 'nunique')
}).sort_values(by=('TF', 'nunique'), ascending=False)

# Convert to adjacency matrix (TF x TG)
tf_unique_col = westside_hits_wtf_not[('TF', 'unique')]
all_tfs = sorted(set([tf for tf_list in tf_unique_col for tf in tf_list]))
all_tgs = westside_hits_wtf_not.index.tolist()

adjacency_matrix = pd.DataFrame(0, index=all_tfs, columns=all_tgs)
for tg in all_tgs:
    tfs = tf_unique_col[tg]
    adjacency_matrix.loc[tfs, tg] = 1



# Plot adjacency matrix as network graph
G = nx.DiGraph()
# Add nodes with node type attribute
for tf in all_tfs:
    G.add_node(tf, node_type='TF')
for tg in all_tgs:
    G.add_node(tg, node_type='TG')
# Add edges from adjacency matrix
for tf in all_tfs:
    for tg in all_tgs:
        if adjacency_matrix.loc[tf, tg] == 1:
            G.add_edge(tf, tg)

# Weakly connected components (treats graph as undirected for connectivity)
components = list(nx.weakly_connected_components(G))

# Keep only the largest component
largest = max(components, key=len)
G = G.subgraph(largest).copy()

# Create layout
pos = nx.fruchterman_reingold_layout(G, k=1, iterations=50, seed=42)
#pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")

# Plot network
fig, ax = plt.subplots(figsize=(16, 12))
# Draw nodes with different colors for TFs and TGs
tf_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'TF']
tg_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'TG']
nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_color='lightblue', 
                       node_size=500, alpha=0.9, ax=ax, label='TF')
nx.draw_networkx_nodes(G, pos, nodelist=tg_nodes, node_color='lightcoral', 
                       node_size=500, alpha=0.9, ax=ax, label='TG')
# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, 
                       edge_color='gray', ax=ax)
# Draw labels (optional - can be commented out if too cluttered)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
ax.legend()
ax.set_title('TF-TG Network Graph', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()

plt.savefig(os.path.join(os.environ.get('OUTPATH', '.'), 'westside_hits_network.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
'''

''' FLAWED
for sex in unique_sexes:
    sex = sex.lower()
    for celltype in unique_celltypes:
        unique_TF_TG_combinations_df = mean_grn_df_filtered_pruned_dict[sex][celltype][['TF', 'TG']].drop_duplicates()
        unique_TF_TG_combinations_str = unique_TF_TG_combinations_df.apply(lambda x: ' - '.join(x), axis=1).values
        unique_TF_TG_combinations_dict[sex][celltype] = unique_TF_TG_combinations_str

        pairs = set(unique_TF_TG_combinations_dict[sex][celltype])
        
        if not shared_TF_TG_pairs: # ...if empty set, initialize with current pairs
            shared_TF_TG_pairs = pairs.copy()
        else:
            shared_TF_TG_pairs = shared_TF_TG_pairs.intersection(pairs) # returns empty set if no intersection, then hits conditional above...
            
        print('Current size of shared TF-TG pairs:', len(shared_TF_TG_pairs))
        all_TF_TG_pairs = all_TF_TG_pairs.union(pairs)

shared_TF_TG_pairs_df = pd.DataFrame(shared_TF_TG_pairs).iloc[:, 0].str.split(' - ', expand=True)
shared_TF_TG_pairs_df.columns = ['TF', 'TG']
shared_TF_TG_pairs_df.sort_values(by='TG', inplace=True)
#shared_TF_TG_pairs_df.to_csv(os.path.join(output_dir, 'shared_TF_TG_pairs.csv'), index=False)

print(f'Shared TF-TG pairs (n={len(shared_TF_TG_pairs)} out of {len(all_TF_TG_pairs)}):')
print(shared_TF_TG_pairs_df)

shared_TF_TG_pairs_df_grouped = shared_TF_TG_pairs_df.groupby('TF').agg({
    'TG': [list, 'nunique'],
}).sort_values(by=('TG', 'nunique'), ascending=False)
'''

def chea_2022_enrichment(shared_TF_TG_pairs_df_grouped):

    enriched_TF_TG_pairs_dict = dict()

    shared_TF_TG_pairs_df_grouped_filtered = shared_TF_TG_pairs_df_grouped[shared_TF_TG_pairs_df_grouped['TG','nunique'] > 1] # not interesting to find TFs with only one regulon gene

    for TF in shared_TF_TG_pairs_df_grouped_filtered.index:

        TF_TG_pairs = shared_TF_TG_pairs_df_grouped.loc[TF, ('TG', 'list')]
        TF_TG_pairs_series = pd.Series(TF, index=TF_TG_pairs)
        TF_TG_pairs_series.attrs = {'sex':'all', 'celltype':'all', 'type': 'TF-TG pairs'}

        try:
            enr = gp.enrichr(TF_TG_pairs_series.index.to_list(), gene_sets='ChEA_2022', outdir=None)
            enr.res2d['-log10(fdr)'] = -np.log10(enr.res2d['Adjusted P-value'])
            enrichr_results_sig = enr.res2d
        except Exception as e:
            print(f"Error performing enrichment analysis for {TF}: {e}")
            continue

        plt.close('all')

        if enrichr_results_sig is not None:

            #enrichr_results_sig = enrichr_results_sig[enrichr_results_sig['P-value'] < 0.05]
            enriched_tfs = enrichr_results_sig['Term'].str.split(' ').str[0]
            enriched_species = enrichr_results_sig['Term'].str.split(' ').str[-1]
            enriched_tfs_match_TF = np.isin(enriched_tfs, TF) & np.isin(enriched_species, 'Human')
        
            if enriched_tfs_match_TF.any():
                genes_of_TF = enrichr_results_sig[enriched_tfs_match_TF]['Genes']
                if len(genes_of_TF) == 1:
                    enriched_tfs_match_TF_list = genes_of_TF.str.split(';').item()
                elif len(genes_of_TF) > 1:
                    enriched_tfs_match_TF_list = genes_of_TF.str.split(';').explode().tolist()

                if len(enriched_tfs_match_TF_list) >= 2: # at least 2 TFs should be enriched for the same TG to study interesting TFs and their regulons
                    enriched_TF_TG_pairs_dict[TF] = enriched_tfs_match_TF_list.copy()

    return enriched_TF_TG_pairs_dict

## Perform enrichment analysis on TG regulons of TFs that are enriched in shared TF-TG pairs
female_enriched_TF_TG_pairs_dict = chea_2022_enrichment(female_ExN_Oli_shared_TF_TG_pairs_grouped)
male_enriched_TF_TG_pairs_dict = chea_2022_enrichment(male_ExN_Oli_shared_TF_TG_pairs_grouped)

np.intersect1d(list(female_enriched_TF_TG_pairs_dict.keys()), list(male_enriched_TF_TG_pairs_dict.keys()))

## write enriched_TF_TG_pairs_dict to json
with open(os.path.join(output_dir, 'enriched_TF_TG_pairs_dict.json'), 'w') as f:
    json.dump(enriched_TF_TG_pairs_dict, f)

#%% plot genome track around

import scglue

#with open(os.path.join(output_dir, f"mean_grn_df_filtered_dict.pkl"), "rb") as f:
#    mean_grn_df_filtered_dict = pickle.load(f)

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

## save filtered results
with open(os.path.join(output_dir, 'broad_gene_series_dict.pkl'), 'wb') as f:
    pickle.dump({
        'enrs_mdd_dn_genes_series': enrs_mdd_dn_genes_series,
        'all_sccompreg_genes_series': all_sccompreg_genes_series,
        'pydeseq2_match_length_genes_series': pydeseq2_match_length_genes_series
    }, f)


#%% differential accessibility

from pybedtools import BedTool

tss_abhd17b

def get_dap_df(sex):

    atac_case = X_atac_dict[sex]['ExN']['Case']
    atac_control = X_atac_dict[sex]['ExN']['Control']
    assert (atac_case.var_names == atac_control.var_names).all()

    ## get peaks from GRNs
    peaks_df = pd.DataFrame(index=atac_case.var_names.str.split(':|-', expand=True)).reset_index()
    peaks_bedtool = BedTool.from_dataframe(peaks_df)

    unique_peaks = pd.Series(ABHD17B_grn['enhancer'].unique())
    grns_peaks_df = pd.DataFrame(unique_peaks.str.split(':|-', expand=True))
    grns_peaks_bedtool = BedTool.from_dataframe(grns_peaks_df)

    # Get indices of ATAC peaks that overlap with GRN peaks
    grn_peaks_in_data = peaks_bedtool.intersect(grns_peaks_bedtool, wa=True, wb=True)
    grn_peaks_in_data_df = grn_peaks_in_data.to_dataframe()

    # Get names of peaks to create mapper
    atac_peak_names = grn_peaks_in_data_df.iloc[:,:3].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
    grn_peak_names = grn_peaks_in_data_df.iloc[:,3:].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
    peaks_names_mapper = dict(zip(atac_peak_names, grn_peak_names)) # not necessarly one-to-one map, can have many-to-one

    dap_df = pd.DataFrame(index=atac_peak_names, columns=['tstat', 'pvalue', 'mlog10_pvalue', 'df', 'brainscope_coords'])

    for enhancer_in_data in atac_peak_names:

        idx_in_data = np.where(atac_case.var_names == enhancer_in_data)[0].item()

        atac_case_dat = atac_case.X[:, idx_in_data].toarray().flatten()
        atac_control_dat = atac_control.X[:, idx_in_data].toarray().flatten()

        d1 = DescrStatsW(atac_case_dat, weights=atac_case.obs['proportion_of_cells'].values)
        d2 = DescrStatsW(atac_control_dat, weights=atac_control.obs['proportion_of_cells'].values)

        cm = d1.get_compare(d2)
        tstat_comp, pvalue_comp, df_comp = cm.ttest_ind(usevar='unequal', alternative='two-sided')

        dap_df.loc[enhancer_in_data, 'tstat'] = tstat_comp
        dap_df.loc[enhancer_in_data, 'pvalue'] = pvalue_comp
        dap_df.loc[enhancer_in_data, 'mlog10_pvalue'] = -np.log10(pvalue_comp)
        dap_df.loc[enhancer_in_data, 'df'] = df_comp

        dap_df.loc[enhancer_in_data, 'brainscope_coords'] = peaks_names_mapper[enhancer_in_data]

    return dap_df

dap_female_df = get_dap_df('female')
dap_male_df = get_dap_df('male')




# %%
print ('Done!')