import os
import pickle
import numpy as np
import pandas as pd
import pybedtools
import anndata
import torch

from types import SimpleNamespace
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from eclare.post_hoc_utils import \
    extract_target_source_replicate, initialize_dicts, assign_to_dicts, perform_gene_set_enrichment, differential_grn_analysis, process_celltype, load_model_and_metadata, get_brain_gmt, metric_boxplots, \
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

#%% differential expression analysis

X_rna_dict, X_atac_dict, overlapping_target_genes_dict, overlapping_tfs_dict, genes_by_peaks_corrs_dict, genes_by_peaks_masks_dict, n_dict, scompreg_loglikelihoods_dict, std_errs_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, tfrps_dict, tfrp_predictions_dict \
    = initialize_dicts()

## initialize dicts
pydeseq2_results_dict = tree()
significant_genes_dict = tree()

enrs_dict = tree()
magma_results_dict = tree()

## TEMPORARY - restrict unique celltypes
unique_celltypes = ['Ast', 'End', 'InN', 'Mic', 'OPC', 'Oli']

## Get BrainGMT and filter for cortical genes
brain_gmt_cortical, brain_gmt_cortical_wGO = get_brain_gmt()

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]
     
## loop through sex and celltype
for sex in unique_sexes:
    # Use joblib for parallel processing
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(process_celltype)(
            sex, celltype, rna_scaled_with_counts, mdd_rna.var,
            rna_celltype_key, rna_condition_key, rna_sex_key
        )
        for celltype in unique_celltypes
    )
    
    # Process results
    for celltype, (mdd_subjects_counts_adata, pydeseq2_results, significant_genes) in zip(unique_celltypes, results):
        pydeseq2_results_dict[sex][celltype] = pydeseq2_results
        significant_genes_dict[sex][celltype] = significant_genes

## set cutoff for number of cells to keep for SEACells representation. see Bilous et al. 2024, Liu & Li 2024 (mcRigor) or Li et al. 2025 (MetaQ) for benchmarking experiments
for sex in unique_sexes:
    for condition in unique_conditions:

        results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
                delayed(differential_grn_analysis)(
                    condition, sex, celltype, mdd_rna, mdd_atac, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, atac_celltype_key, atac_condition_key, atac_sex_key, atac_subject_key, eclare_student_model, mean_grn_df, cutoff=5025, ot_alignment_type='all'
                )
            for celltype in unique_celltypes
        )

        for celltype, result in zip(unique_celltypes, results):
            assign_to_dicts(*result)


for sex in unique_sexes:
    
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(perform_gene_set_enrichment)(
            sex, celltype, scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, mean_grn_df, significant_genes_dict, mdd_rna.var_names
        )
        for celltype in unique_celltypes
    )

    for celltype, result in zip(unique_celltypes, results):
        enrs_dict[sex][celltype] = result[0]
        magma_results_dict[sex][celltype] = result[1]