#%%
import os
import pickle
import numpy as np
import pandas as pd
import pybedtools
import anndata
import torch
import seaborn as sns

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
    extract_target_source_replicate, initialize_dicts, assign_to_dicts, perform_gene_set_enrichment, differential_grn_analysis, process_celltype, load_model_and_metadata, get_brain_gmt, magma_dicts_to_df, get_next_version_dir, \
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

## TEMPORARY - restrict unique celltypes
unique_celltypes = ['Ast', 'Mic']

## Get BrainGMT and filter for cortical genes
brain_gmt_cortical, brain_gmt_cortical_wGO = get_brain_gmt()

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]
     
#%% pyDESeq2
for sex in unique_sexes:
    sex = sex.lower()
    
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(process_celltype)(
            sex, celltype, rna_scaled_with_counts, mdd_rna.var,
            rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key
        )
        for celltype in unique_celltypes
    )
    
    # Process results
    for celltype, (mdd_subjects_counts_adata, pydeseq2_results, significant_genes) in zip(unique_celltypes, results):
        pydeseq2_results_dict[sex][celltype] = pydeseq2_results
        significant_genes_dict[sex][celltype] = significant_genes

# Save results with thread-safe file writing
dicts_to_save = {
    'pydeseq2_results_dict': pydeseq2_results_dict,
    'significant_genes_dict': significant_genes_dict,
}

file_lock = threading.Lock()
for dict_name, dict_obj in dicts_to_save.items():
    with file_lock:
        with open(os.path.join(output_dir, f"{dict_name}.pkl"), "wb") as f:
            pickle.dump(dict_obj, f)
        print(f"Saved {dict_name}")

#%% sc-compReg analysis
for sex in unique_sexes:
    sex = sex.lower()
    
    for condition in unique_conditions:

        results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
                delayed(differential_grn_analysis)(
                    condition, sex, celltype, mdd_rna, mdd_atac, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, atac_celltype_key, atac_condition_key, atac_sex_key, atac_subject_key, eclare_student_model, mean_grn_df, \
                    overlapping_subjects, subjects_by_condition_n_sex_df, cutoff=5025, ot_alignment_type='all'
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
# Note: For CPU-intensive tasks like MAGMA analysis, consider using multiprocessing instead of threading:
# backend='multiprocessing' instead of backend='threading'
logger.info("Starting gene set enrichment analyses")
for sex in unique_sexes:
    sex = sex.lower()
    logger.info(f"Processing sex: {sex}")
    
    # Use a lock to prevent race conditions when creating directories
    dir_lock = threading.Lock()
    
    def safe_perform_gene_set_enrichment(sex, celltype, *args, **kwargs):
        """Thread-safe wrapper for perform_gene_set_enrichment"""
        try:
            logger.info(f"Starting enrichment for {sex}_{celltype}")
            # Create unique subdirectory for each thread
            thread_id = threading.current_thread().ident
            unique_output_dir = os.path.join(kwargs['output_dir'], f'thread_{thread_id}')
            os.makedirs(unique_output_dir, exist_ok=True)
            kwargs['output_dir'] = unique_output_dir
            
            result = perform_gene_set_enrichment(sex, celltype, *args, **kwargs)
            logger.info(f"Completed enrichment for {sex}_{celltype}")
            return result
        except Exception as e:
            logger.error(f"Error in thread {threading.current_thread().ident} for {sex}_{celltype}: {e}")
            return None, None
    
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='multiprocessing')(
        delayed(safe_perform_gene_set_enrichment)(
            sex, celltype, scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, mean_grn_df, significant_genes_dict, mdd_rna.var_names, pydeseq2_results_dict, brain_gmt_cortical, slopes_dict, std_errs_dict, intercepts_dict, intercept_stderrs_dict, output_dir=output_dir
        )
        for celltype in unique_celltypes
    )

    for celltype, result in zip(unique_celltypes, results):
        if result is not None and result[0] is not None:
            enrs_dict[sex][celltype] = result[0]
            magma_results_dict[sex][celltype] = result[1]
        else:
            print(f"Warning: No results for {sex}_{celltype}")

# Save results with thread-safe file writing
dicts_to_save = {
    'enrs_dict': enrs_dict,
    'magma_results_dict': magma_results_dict,
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
sns.heatmap(magma_results_df.apply(lambda x: -np.log10(x)))

#%%
# Load all saved dictionaries
dicts_to_load = [
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
    'magma_results_dict'
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

# %%
print ('Done!')