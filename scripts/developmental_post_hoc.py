#%%
from eclare import set_env_variables
set_env_variables(config_path='../config')

import torch
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from glob import glob
from sklearn.model_selection import StratifiedKFold
import scanpy as sc
import seaborn as sns
import warnings
import anndata
import magic
import graphtools as gt

from eclare.post_hoc_utils import metric_boxplots, download_mlflow_runs, extract_target_source_replicate
from eclare.post_hoc_utils import load_model_and_metadata
from eclare.setup_utils import return_setup_func_from_dataset

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='scanpy')
warnings.filterwarnings('ignore', category=UserWarning, module='scanpy')

cuda_available = torch.cuda.is_available()
n_cudas = torch.cuda.device_count()
#device = torch.device(f'cuda:{n_cudas - 1}') if cuda_available else 'cpu'
device = 'cpu'

## Define target and source datasets
'''
target_dataset = 'Cortex_Velmeshev'
genes_by_peaks_str = '9584_by_66620'
source_datasets = ['PFC_V1_Wang', 'PFC_Zhu']
subsample = 5000
methods_id_dict = {
    'clip': '25165730',
    'kd_clip': '25173640',
    'eclare': ['04164533'],
    'ordinal': '27204131',
}
'''
'''
target_dataset = 'MDD'
genes_by_peaks_str = '6816_by_55284'
source_datasets = ['PFC_Zhu']
subsample = -1
methods_id_dict = {
    'clip': '21164436',
    'kd_clip': '24112954',
    'eclare': ['22105844'],
    'ordinal': '22130216',
}
'''
target_dataset = 'MDD'
genes_by_peaks_str = '17279_by_66623'
source_datasets = ['PFC_V1_Wang']
subsample = -1
methods_id_dict = {
    'clip': '04163347',
    'kd_clip': '07185908',
    'eclare': ['04203819'],
    'ordinal': '22094649',
}

## define search strings
search_strings = {
    'clip': 'CLIP' + '_' + methods_id_dict['clip'],
    'kd_clip': ['KD_CLIP' + '_' + job_id for job_id in methods_id_dict['kd_clip']],
    'eclare': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare']]
}

## for ECLARE, map search_strings to 'dataset' column
dataset_column = [
    'eclare',
    ]
search_strings_to_dataset = {
    'ECLARE' + '_' + job_id: dataset_column[j] for j, job_id in enumerate(methods_id_dict['eclare'])
}

#%% load metrics dataframes

experiment_name = f"clip_{target_dataset.lower()}_{methods_id_dict['clip']}"

#if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
#    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
#    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
#else:
print(f"Downloading runs.csv for {experiment_name} from MLflow")
all_metrics_csv_path = download_mlflow_runs(experiment_name)

all_metrics_df = pd.read_csv(all_metrics_csv_path)

CLIP_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['clip']))[0]
KD_CLIP_header_idx = np.where(all_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['kd_clip'])))[0]
ECLARE_header_idx = np.where(all_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['eclare'])))[0]

CLIP_run_id = all_metrics_df.iloc[CLIP_header_idx]['run_id']
KD_CLIP_run_id = all_metrics_df.iloc[KD_CLIP_header_idx]['run_id']
ECLARE_run_id = all_metrics_df.iloc[ECLARE_header_idx]['run_id']

CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(CLIP_run_id)]
KD_CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(KD_CLIP_run_id)]
ECLARE_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(ECLARE_run_id)]

CLIP_metrics_df = extract_target_source_replicate(CLIP_metrics_df)
KD_CLIP_metrics_df = extract_target_source_replicate(KD_CLIP_metrics_df)
ECLARE_metrics_df = extract_target_source_replicate(ECLARE_metrics_df, has_source=False)

## add dataset column
CLIP_metrics_df.loc[:, 'dataset']      = 'clip'
KD_CLIP_metrics_df.loc[:, 'dataset']   = 'kd_clip'
ECLARE_metrics_df.loc[:, 'dataset']    = 'eclare'

combined_metrics_df = pd.concat([
    ECLARE_metrics_df,
    KD_CLIP_metrics_df,
    CLIP_metrics_df
    ]) # determines order in which metrics are plotted
    
## if source and/or target contain 'multiome', convert to '10x'
combined_metrics_df.loc[:, 'source'] = combined_metrics_df['source'].str.replace('multiome', '10x')
combined_metrics_df.loc[:, 'target'] = combined_metrics_df['target'].str.replace('multiome', '10x')

## only keep runs with 'FINISHED' status
#combined_metrics_df = combined_metrics_df[combined_metrics_df['status'] == 'FINISHED']

## plot boxplots
metric_boxplots(
    combined_metrics_df, target_source_combinations=True, include_paired=False
    )

#%% Load ECLARE students
import pickle

## Find path to best ECLARE model
best_eclare     = str(ECLARE_metrics_df['compound_metric'].argmax())
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_{target_dataset.lower()}_{methods_id_dict["eclare"][0]}', best_eclare, device, target_dataset=target_dataset, set_train=True if target_dataset == 'MDD' else True)
eclare_student_model = eclare_student_model.to(device=device)

## Load KD_CLIP student models
best_kd_clip = '0'
kd_clip_student_models = {}

for source_dataset in source_datasets:
    kd_clip_student_model, kd_clip_student_model_metadata     = load_model_and_metadata(f'kd_clip_{target_dataset.lower()}_{methods_id_dict["kd_clip"]}', best_kd_clip, device, target_dataset=os.path.join(target_dataset, source_dataset), set_train=True if target_dataset == 'MDD' else True)
    kd_clip_student_models[source_dataset] = kd_clip_student_model

#%% Get ECLARE student data

student_setup_func = return_setup_func_from_dataset(target_dataset)

args = SimpleNamespace(
    source_dataset= source_datasets[0] if target_dataset == 'MDD' else target_dataset,
    target_dataset= target_dataset if target_dataset == 'MDD' else None,
    genes_by_peaks_str=genes_by_peaks_str,
    ignore_sources=[None],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)

student_rna, student_atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = student_setup_func(args, return_type='data')
#student_rna_keep = student_rna
#student_atac_keep = student_atac

if target_dataset == 'MDD':
    dev_group_key = 'Age'
    dev_stages = pd.Categorical(student_rna.obs[dev_group_key].sort_values().unique().tolist(), ordered=True)

    rna_clusters_metadata = pd.read_excel(os.path.join(os.environ['DATAPATH'], 'maitra_2023_SourceDataFigures', 'Fig1b_Fig1c_SuppFig1a_SuppFig1b_SuppFig1c_SuppFig1d.xlsx'))
    subclusters = rna_clusters_metadata[['Cell', 'Cluster']].set_index('Cell').rename(columns={'Cluster': 'SubClusters'})
    student_rna.obs = student_rna.obs.merge(subclusters, left_index=True, right_index=True, how='left')
    #student_rna.obs = student_rna.obs.rename(columns={'RNA_snn_res.0.7': 'SubClusters'}).astype(str) # don't have labels for subclusters for RNA, so use one of integer-based cluster annotations

elif target_dataset == 'Cortex_Velmeshev':
    dev_group_key = 'Age_Range'
    dev_stages = student_rna.obs[dev_group_key].cat.categories.tolist()
else:
    raise ValueError(f'Target dataset {target_dataset} not supported')


## Load validation cell IDs
validation_cell_ids_path = os.path.join(os.environ['OUTPATH'], f'eclare_{target_dataset.lower()}_{methods_id_dict["eclare"][0]}', target_dataset, best_eclare, 'valid_cell_ids.pkl')
with open(validation_cell_ids_path, 'rb') as f:
    validation_cell_ids = pickle.load(f)
    valid_rna_ids = validation_cell_ids['valid_cell_ids_rna']
    valid_atac_ids = validation_cell_ids['valid_cell_ids_atac']

'''
student_rna, student_atac, cell_group, dev_group_key, dev_stages = student_setup_func(args, return_backed=True)

#macroglia_cell_types = ['OL', 'OPC', 'GLIALPROG', 'AST']
#keep_rna = student_rna.obs[cell_group].isin(macroglia_cell_types) & student_rna.obs[dev_group_key].str.contains('3rd trimester')
#keep_atac = student_atac.obs[cell_group].isin(macroglia_cell_types)

keep_rna = student_rna.obs[cell_group].str.contains('ExNeu')
keep_atac = student_atac.obs[cell_group].str.contains('ExNeu')

student_rna_keep = student_rna[keep_rna].to_memory()
student_atac_keep = student_atac[keep_atac].to_memory()
'''

#%% Teacher models and data
from eclare.setup_utils import teachers_setup

## Load CLIP teacher models
replicate_idx = '0'

model_uri_paths_str = f'clip_{target_dataset.lower()}_*{methods_id_dict["clip"]}/{target_dataset}/**/{replicate_idx}/model_uri.txt'
model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], model_uri_paths_str))

args = SimpleNamespace(
    target_dataset=target_dataset,
    ignore_sources=[None],
    source_dataset_embedder=None,
)

## Setup teachers
datasets, clip_models, teacher_rnas, teacher_atacs = teachers_setup(model_uri_paths, args, device, return_type='data')

#for source_dataset in source_datasets:
#    teacher_rna = teacher_rnas[source_dataset][valid_rna_ids]
#    teacher_atac = teacher_atacs[source_dataset]

#%% load ordinal model
import mlflow
from mlflow.models import Model

ordinal_model_uri_paths_str = f"ordinal_*{methods_id_dict['ordinal']}/model_uri.txt"
ordinal_model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], ordinal_model_uri_paths_str))
assert len(ordinal_model_uri_paths) > 0, f'Model URI path not found @ {ordinal_model_uri_paths_str}'

with open(ordinal_model_uri_paths[0], 'r') as f:
    model_uris = f.read().strip().splitlines()
    model_uri = model_uris[0]

mlflow.set_tracking_uri('file:///home/mcb/users/dmannk/scMultiCLIP/ECLARE/mlruns')
ordinal_model = mlflow.pytorch.load_model(model_uri, device=device)

ordinal_model_metadata = Model.load(model_uri)
ordinal_source_dataset = ordinal_model_metadata.metadata['source_dataset']


#%% extract student latents for analysis
from eclare.post_hoc_utils import create_celltype_palette
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from skimage.exposure import match_histograms
#from sklearn.preprocessing import quantile_transform

def subsample_adata(adata, subsample, combinations_keys, subsample_type='balanced'):

    combinations = adata.obs[combinations_keys].astype(str).apply(lambda x: ' - '.join(x), axis=1).values

    if subsample_type == 'stratified':
        n_cells = adata.n_obs
        n_splits = np.ceil(n_cells / subsample).astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        _, valid_idx = next(skf.split(np.zeros_like(combinations), combinations))

    elif subsample_type == 'balanced':
        if subsample > 0:
            classes = np.unique(combinations)
            k = len(classes)
            per_class = subsample // k
            counts = Counter(combinations)
            strategy = {c: min(counts[c], per_class) for c in classes}
            rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
        elif subsample == -1:
            rus = RandomUnderSampler(random_state=42, sampling_strategy='all')
        valid_idx, valid_devs = rus.fit_resample(np.arange(len(combinations)).reshape(-1, 1), combinations)

    return adata[valid_idx.flatten()].copy()

#student_model = deepcopy(eclare_student_model)
#print('Using KD-CLIP student model'); student_model = deepcopy(kd_clip_student_models['PFC_V1_Wang'])

## subsample validation data
if cell_group == 'Lineage':
    lineages = ['ExNeu', 'IN']
    if student_rna.obs_names.isin(valid_rna_ids).any():
        rna_idxs = np.where(student_rna.obs_names.isin(valid_rna_ids) & student_rna.obs['Lineage'].isin(lineages))[0]
        atac_idxs = np.where(student_atac.obs_names.isin(valid_atac_ids) & student_atac.obs['Lineage'].isin(lineages))[0]
    else:
        print('WARNING: obs_names not in valid_rna_ids, resorting to reset_index().index.astype(str)')
        rna_idxs = np.where(student_rna.obs.reset_index().index.astype(str).isin(valid_rna_ids) & student_rna.obs['Lineage'].isin(lineages))[0]
        atac_idxs = np.where(student_atac.obs.reset_index().index.astype(str).isin(valid_atac_ids) & student_atac.obs['Lineage'].isin(lineages))[0]

elif cell_group == 'ClustersMapped':
    #clusters = student_rna.obs['ClustersMapped'].cat.categories.tolist()
    #target_clusters = ['ExN']
    target_clusters = ['RG-vRG','IPC-EN','EN-newborn','EN-IT-immature','EN-non-IT-immature','ExN1','ExN1_L23','ExN2','ExN2_L23','EN-L2_3-IT'] # contains source and target clusters
    if student_rna.obs_names.isin(valid_rna_ids).any():
        rna_idxs = np.where(~student_rna.obs_names.isin(valid_rna_ids) & student_rna.obs['ClustersMapped'].isin(target_clusters))[0]
        atac_idxs = np.where(~student_atac.obs_names.isin(valid_atac_ids) & student_atac.obs['ClustersMapped'].isin(target_clusters))[0]
    else:
        print('WARNING: obs_names not in valid_rna_ids, resorting to reset_index().index.astype(str)')
        rna_idxs = np.where(student_rna.obs.reset_index().index.astype(str).isin(valid_rna_ids) & student_rna.obs['ClustersMapped'].isin(target_clusters))[0]
        atac_idxs = np.where(student_atac.obs.reset_index().index.astype(str).isin(valid_atac_ids) & student_atac.obs['ClustersMapped'].isin(target_clusters))[0]

if target_dataset == 'MDD':

    student_rna.obs['modality'] = 'RNA'
    student_atac.obs['modality'] = 'ATAC'
    student_rna_atac = anndata.concat([student_rna[rna_idxs], student_atac[atac_idxs]], axis=0) # no features, since features don't align
    student_rna_atac.obs['Age_bins'] = pd.qcut(student_rna_atac.obs[dev_group_key], q=17, labels=None)

    max_subsample = len(rna_idxs) + len(atac_idxs)
    if subsample > max_subsample:
        subsample = max_subsample

    student_rna_atac_sub    = subsample_adata(student_rna_atac, subsample, ['Age_bins', 'Condition', 'modality'], subsample_type='balanced')
    sns.barplot(student_rna_atac_sub.obs, x='modality', y='Age', hue='Condition', errorbar='se'); plt.ylim([30,50])

    student_rna_sub         = student_rna[student_rna.obs_names.isin(student_rna_atac_sub.obs_names)]
    student_atac_sub        = student_atac[student_atac.obs_names.isin(student_rna_atac_sub.obs_names)]

    student_rna_sub.obs = student_rna_sub.obs.merge(student_rna_atac_sub.obs['Age_bins'], left_index=True, right_index=True, how='left')
    student_atac_sub.obs = student_atac_sub.obs.merge(student_rna_atac_sub.obs['Age_bins'], left_index=True, right_index=True, how='left')

elif target_dataset == 'Cortex_Velmeshev':

    student_rna_sub = subsample_adata(student_rna[rna_idxs], subsample, [cell_group, dev_group_key], subsample_type='balanced')
    student_atac_sub = subsample_adata(student_atac[atac_idxs], subsample, [cell_group, dev_group_key], subsample_type='balanced')

student_rna_cells_sub = torch.from_numpy(student_rna_sub.X.toarray().astype(np.float32))
student_atac_cells_sub = torch.from_numpy(student_atac_sub.X.toarray().astype(np.float32))

## create obs
obs = pd.concat([
    student_rna_sub.obs.assign(modality='RNA'),
    student_atac_sub.obs.assign(modality='ATAC')
    ])
obs[dev_group_key] = pd.Categorical(obs[dev_group_key], categories=dev_stages, ordered=True)

## get subset data latents
n_genes_eclare = next(eclare_student_model.rna_to_core.parameters()).shape[1]
n_peaks_eclare = next(eclare_student_model.atac_to_core.parameters()).shape[1]

if (n_genes_eclare == student_rna_cells_sub.shape[1]) and (n_peaks_eclare == student_atac_cells_sub.shape[1]):
    student_rna_latents_sub, _ = eclare_student_model.to('cpu')(student_rna_cells_sub, modality=0)
    student_atac_latents_sub, _ = eclare_student_model.to('cpu')(student_atac_cells_sub, modality=1)

    ## create ECLARE adata for subsampled data
    X = np.vstack([
        student_rna_latents_sub.detach().cpu().numpy(),
        student_atac_latents_sub.detach().cpu().numpy()
        ])

    subsampled_eclare_adata = sc.AnnData(
        X=X,
        obs=obs,
    )

else:
    print(f'WARNING: ECLARE model has {n_genes_eclare} genes and {n_peaks_eclare} peaks, but student data has {student_rna_cells_sub.shape[1]} genes and {student_atac_cells_sub.shape[1]} peaks')


## create KD-CLIP adatas based on latents from KD-CLIP student model
subsampled_kd_clip_adatas = {}
for source_dataset in source_datasets:

    ## KD-CLIP student model
    kd_student_rna_latents_sub, _ = kd_clip_student_models[source_dataset].to('cpu')(student_rna_cells_sub, modality=0)
    kd_student_atac_latents_sub, _ = kd_clip_student_models[source_dataset].to('cpu')(student_atac_cells_sub, modality=1)

    kd_student_rna_latents_sub = kd_student_rna_latents_sub.detach().cpu().numpy()
    kd_student_atac_latents_sub = kd_student_atac_latents_sub.detach().cpu().numpy()

    kd_student_latents_sub = np.vstack([kd_student_rna_latents_sub, kd_student_atac_latents_sub])
    subsampled_kd_clip_adatas[source_dataset] = sc.AnnData(
        X=kd_student_latents_sub,
        obs=obs,
    )

#%% create adatas for teacher data

valid_ids = obs.index # will get overwritten below

subsampled_clip_adatas = {}
for source_dataset in source_datasets:

    ## load teacher data
    teacher_rna = teacher_rnas[source_dataset]
    teacher_atac = teacher_atacs[source_dataset]

    ## keep only overlapping cells
    teacher_rna_sub = teacher_rna[teacher_rna.obs_names.isin(valid_ids)]
    teacher_atac_sub = teacher_atac[teacher_atac.obs_names.isin(valid_ids)]

    ## project data through teacher model
    teacher_rna = torch.from_numpy(teacher_rna_sub.to_memory().X.toarray().astype(np.float32))
    teacher_atac = torch.from_numpy(teacher_atac_sub.to_memory().X.toarray().astype(np.float32))

    teacher_rna_latents, _ = clip_models[source_dataset].to('cpu')(teacher_rna, modality=0)
    teacher_atac_latents, _ = clip_models[source_dataset].to('cpu')(teacher_atac, modality=1)

    teacher_rna_latents = teacher_rna_latents.detach().cpu().numpy()
    teacher_atac_latents = teacher_atac_latents.detach().cpu().numpy()

    ## create adata
    X = np.vstack([teacher_rna_latents, teacher_atac_latents])

    obs = pd.concat([
        teacher_rna_sub.obs.assign(modality='RNA'),
        teacher_atac_sub.obs.assign(modality='ATAC'),
    ])
    obs[dev_group_key] = pd.Categorical(obs[dev_group_key], categories=dev_stages, ordered=True)

    adata = sc.AnnData(
        X=X,
        obs=obs,
    )

    if ordinal_source_dataset == source_dataset:

        print(f'Projecting data through ordinal model for {source_dataset}')

        ordinal_rna_logits_sub, ordinal_rna_probas_sub, ordinal_rna_latents_sub = ordinal_model.to('cpu')(teacher_rna, modality=0)
        ordinal_atac_logits_sub, ordinal_atac_probas_sub, ordinal_atac_latents_sub = ordinal_model.to('cpu')(teacher_atac, modality=1)

        ordinal_rna_prebias_sub = ordinal_model.ordinal_layer_rna.coral_weights(ordinal_rna_latents_sub)
        ordinal_atac_prebias_sub = ordinal_model.ordinal_layer_atac.coral_weights(ordinal_atac_latents_sub)

        ordinal_rna_pt_sub = torch.from_numpy(match_histograms(ordinal_rna_prebias_sub.detach().cpu().numpy().flatten(), teacher_rna_sub.obs[dev_group_key].values))
        ordinal_atac_pt_sub = torch.from_numpy(match_histograms(ordinal_atac_prebias_sub.detach().cpu().numpy().flatten(), teacher_atac_sub.obs[dev_group_key].values))

        #ordinal_rna_pt_sub = torch.sigmoid(ordinal_rna_prebias_sub / ordinal_rna_logits_sub.var().pow(0.5)).flatten().detach().cpu().numpy()
        #ordinal_atac_pt_sub = torch.sigmoid(ordinal_atac_prebias_sub / ordinal_atac_logits_sub.var().pow(0.5)).flatten().detach().cpu().numpy()

        #ordinal_rna_pt_sub = torch.from_numpy(quantile_transform(ordinal_rna_prebias_sub.detach().cpu().numpy(), output_distribution='normal'))
        #ordinal_atac_pt_sub = torch.from_numpy(quantile_transform(ordinal_atac_prebias_sub.detach().cpu().numpy(), output_distribution='normal'))


    ## add to dictionary
    subsampled_clip_adatas[source_dataset] = adata

#%% project data through ordinal model and assign to adata

## if ordinal model not already associated with source datasets, then boils down to student data
if ordinal_source_dataset == target_dataset:

    student_rna_cells_sub = torch.from_numpy(student_rna_sub.to_memory().X.toarray().astype(np.float32))
    student_atac_cells_sub = torch.from_numpy(student_atac_sub.to_memory().X.toarray().astype(np.float32))

    ordinal_rna_logits_sub, ordinal_rna_probas_sub, ordinal_rna_latents_sub = ordinal_model.to('cpu')(student_rna_cells_sub, modality=0)
    ordinal_atac_logits_sub, ordinal_atac_probas_sub, ordinal_atac_latents_sub = ordinal_model.to('cpu')(student_atac_cells_sub, modality=1)

    ordinal_rna_prebias_sub = ordinal_model.ordinal_layer_rna.coral_weights(ordinal_rna_latents_sub)
    ordinal_atac_prebias_sub = ordinal_model.ordinal_layer_atac.coral_weights(ordinal_atac_latents_sub)

    if target_dataset == 'MDD':
        ordinal_rna_pt_sub = torch.from_numpy(match_histograms(ordinal_rna_prebias_sub.detach().cpu().numpy().flatten(), student_rna_sub.obs[dev_group_key].values))
        ordinal_atac_pt_sub = torch.from_numpy(match_histograms(ordinal_atac_prebias_sub.detach().cpu().numpy().flatten(), student_atac_sub.obs[dev_group_key].values))

    elif target_dataset == 'Cortex_Velmeshev':
        ordinal_rna_pt_sub = torch.sigmoid(ordinal_rna_prebias_sub / ordinal_rna_logits_sub.var().pow(0.5)).flatten().detach().cpu().numpy()
        ordinal_atac_pt_sub = torch.sigmoid(ordinal_atac_prebias_sub / ordinal_atac_logits_sub.var().pow(0.5)).flatten().detach().cpu().numpy()

    else:
        raise ValueError(f'Target dataset {target_dataset} not supported')

    #ordinal_rna_pt_sub = torch.from_numpy(quantile_transform(ordinal_rna_prebias_sub.detach().cpu().numpy(), output_distribution='uniform'))
    #ordinal_atac_pt_sub = torch.from_numpy(quantile_transform(ordinal_atac_prebias_sub.detach().cpu().numpy(), output_distribution='uniform'))

## add to adata
ordinal_pseudotimes = np.concatenate([ordinal_rna_pt_sub, ordinal_atac_pt_sub], axis=0)

try:
    subsampled_eclare_adata.obs['ordinal_pseudotime'] = ordinal_pseudotimes

    ## create ordinal_pseudotime Series and merge with sub_cell_type
    #ordinal_pseudotime_adata = pd.concat([student_rna_sub.obs['ordinal_pseudotime'], student_atac_sub.obs['ordinal_pseudotime']])
    obs_df = pd.merge(subsampled_eclare_adata.obs['ordinal_pseudotime'], subsampled_eclare_adata.obs['sub_cell_type'], left_index=True, right_index=True, how='left')

except:
    print('WARNING: ECLARE adata not created')

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset].obs['ordinal_pseudotime'] = ordinal_pseudotimes
    subsampled_clip_adatas[source_dataset].obs['ordinal_pseudotime'] = ordinal_pseudotimes


#%% Import latents from other models

## specify job IDs
scJoint_job_id  = '20250905_125142'
glue_job_id     = '20250906_003832'

## add to dictionary
methods_id_dict['scJoint'] = scJoint_job_id
methods_id_dict['scGLUE'] = glue_job_id

## scJoint
scJoint_path = os.path.join(os.environ['OUTPATH'], 'scJoint_data_tmp', scJoint_job_id, 'scJoint_latents.h5ad')
scJoint_adata = sc.read_h5ad(scJoint_path)
scJoint_adata.obs = scJoint_adata.obs.merge(obs_df, left_index=True, right_index=True, how='left')

## scGLUE
glue_path = os.path.join(os.environ['OUTPATH'], 'glue', glue_job_id, 'glue_latents.h5ad')
glue_adata = sc.read_h5ad(glue_path)
glue_adata.obs = glue_adata.obs.merge(obs_df, left_index=True, right_index=True, how='left')

## find overlapping cell IDs between ECLARE and scJoint/scGLUE
valid_ids = set(subsampled_eclare_adata.obs_names) & set(glue_adata.obs_names) & set(scJoint_adata.obs_names)

## keep only overlapping cells
subsampled_eclare_adata = subsampled_eclare_adata[subsampled_eclare_adata.obs_names.isin(valid_ids)]

scJoint_adata = scJoint_adata[scJoint_adata.obs_names.isin(valid_ids)]
glue_adata = glue_adata[glue_adata.obs_names.isin(valid_ids)]

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset] = subsampled_kd_clip_adatas[source_dataset][subsampled_kd_clip_adatas[source_dataset].obs_names.isin(valid_ids)]


#%% PAGA
## install scib from github, see scib documentation for details
from scib.metrics.lisi import clisi_graph, ilisi_graph
from scib.metrics import nmi, ari
from scipy.stats import pearsonr, spearmanr, kendalltau
import random

def paga_analysis(adata, dev_group_key='dev_stage', cell_group_key='Lineage', correct_imbalance=False, random_seed=0):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    try:
        torch.manual_seed(random_seed)
    except ImportError:
        pass

    ## graph construction
    if cell_group_key == 'Lineage':
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, random_state=random_seed)
        sc.tl.leiden(adata, resolution=1, random_state=random_seed)
        entropy_threshold = 0.25

    elif cell_group_key == 'ClustersMapped':
        sc.pp.pca(adata, n_comps=10)
        sc.pp.neighbors(adata, n_neighbors=100, n_pcs=10, use_rep='X_pca', random_state=random_seed)
        sc.tl.leiden(adata, resolution=0.25, random_state=random_seed)
        entropy_threshold = 0.60


    if correct_imbalance:

        ## UMAP before correction
        sc.tl.umap(adata, random_state=random_seed)
        sc.pl.umap(adata, color=['modality','leiden'])

        ## check for imbalanced clusters
        '''
        imb = adata.obs.groupby('leiden')[['modality', 'source_or_target']].agg([
            ('proportion', lambda x: x.value_counts(normalize=True).max()),
            ('n', lambda x: x.value_counts().max()),
        ])
        imb.columns = imb.columns.map('_'.join)

        Z = (imb['proportion']-0.5) / np.sqrt(0.25/imb['n'])
        imb['p'] = norm.sf(np.abs(Z))
        imb['entropy'] = -( (imb['proportion']*np.log(imb['proportion'])) + ((1-imb['proportion'])*np.log(1-imb['proportion'])) )
        '''

        imb = adata.obs.groupby('leiden')[['modality', 'source_or_target']].value_counts(normalize=True).reset_index().pivot_table(index='leiden', columns=['modality', 'source_or_target'], values='proportion')
        k = imb.shape[-1]
        max_entropy = -( 1/k * np.log(1/k) ) * k
        imb['entropy'] = -( imb*np.log(imb) ).sum(axis=1)
        imb['entropy_ratio'] = imb['entropy'] / max_entropy

        keep_leidens = imb[imb['entropy_ratio'] > entropy_threshold].index.tolist()
        adata = adata[adata.obs['leiden'].isin(keep_leidens)]

        sc.pp.pca(adata, n_comps=adata.obsm['X_pca'].shape[-1])
        sc.pp.neighbors(adata, n_neighbors=adata.uns['neighbors']['params']['n_neighbors'], n_pcs=adata.obsm['X_pca'].shape[-1], use_rep='X_pca', random_state=random_seed)
        sc.tl.leiden(adata, resolution=adata.uns['leiden']['params']['resolution'], random_state=random_seed)

    ## UMAP
    sc.tl.umap(adata, random_state=random_seed)
    #sc.pl.umap(adata, color='leiden')

    color = ['modality', cell_group_key, 'ordinal_pseudotime', dev_group_key]

    if ('sub_cell_type' in adata.obs.columns) and ('velmeshev_pseudotime' in adata.obs.columns):
        color += ['sub_cell_type', 'velmeshev_pseudotime']
    #if ('Age' in adata.obs.columns) and (dev_group_key != 'Age'):
    #    color += ['Age']
    #    adata.obs['Age'] = adata.obs['Age'].astype(float)
    #elif dev_group_key != 'Age':
    #    adata.obs['Age'] = adata.obs['Age'].astype(float)

    colors_uns_keys = [key for key in list(adata.uns.keys()) if '_colors' in key]
    for key in colors_uns_keys: del adata.uns[key]

    sc.pl.umap(adata, color=color, ncols=3, wspace=0.5)

    ## PAGA
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, color=["leiden", "modality", cell_group_key, "ordinal_pseudotime"])

    ## Graph based on PAGA
    sc.tl.draw_graph(adata, init_pos='paga', random_state=random_seed)

    ## Pseudotime with DPT

    ## find centroid of 2nd trimester cells
    if '2nd trimester' in adata.obs[dev_group_key].values:

        mask = adata.obs[dev_group_key] == '2nd trimester'
        X = adata.X if not hasattr(adata, 'obsm') or 'X_pca' not in adata.obsm else adata.obsm['X_pca']
        if hasattr(X, 'toarray'):  # handle sparse matrices
            X = X.toarray()
        centroid = X[mask.values].mean(axis=0)
        dists = np.linalg.norm(X - centroid, axis=1)

        ## Set iroot as the cell closest to the centroid of 2nd trimester cells
        adata.uns['iroot'] = np.argmin(np.where(mask.values, dists, np.inf))

    elif cell_group_key in adata.obs.columns: # should be True in all cases

        root_cell_group = adata.obs.groupby('leiden')['ordinal_pseudotime'].mean().idxmin()
        print(f'Root cell group: {root_cell_group}')

        mask = adata.obs['leiden'] == root_cell_group
        X = adata.X if not hasattr(adata, 'obsm') or 'X_pca' not in adata.obsm else adata.obsm['X_pca']
        if hasattr(X, 'toarray'):  # handle sparse matrices
            X = X.toarray()
        centroid = X[mask.values].mean(axis=0)
        dists = np.linalg.norm(X - centroid, axis=1)
        adata.uns['iroot'] = np.argmin(np.where(mask.values, dists, np.inf))

        #median_root_pseudotime = adata.obs.loc[adata.obs['leiden'] == root_cell_group, 'ordinal_pseudotime'].median()
        #iroot_cell_id = (adata.obs.loc[adata.obs['leiden'] == root_cell_group, 'ordinal_pseudotime'] == median_root_pseudotime).idxmax()
        #adata.uns['iroot'] = adata.obs_names.tolist().index(iroot_cell_id)

        #p = adata.obs.groupby('leiden')[cell_group_key].value_counts(normalize=True)
        #entropy_per_leiden = (-p*np.log(p)).groupby('leiden').mean().sort_values()
        #max_entropy_leiden = entropy_per_leiden.index[-1] # last leiden with highest entropy across cell groups
        #iroot_cell_id = adata.obs.loc[adata.obs['leiden'] == max_entropy_leiden, 'ordinal_pseudotime'].idxmin()
        #adata.uns['iroot'] = adata.obs_names.tolist().index(iroot_cell_id)

    elif 'ordinal_pseudotime' in adata.obs.columns:
        adata.uns['iroot'] = adata.obs['ordinal_pseudotime'].argmin()

    ## DPT
    sc.tl.dpt(adata)
    
    if 'sub_cell_type' in adata.obs.columns:
        sc.pl.draw_graph(adata, color=['modality', cell_group_key, 'sub_cell_type', 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)
    else:
        sc.pl.draw_graph(adata, color=['modality', cell_group_key, 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)

    return adata

def trajectory_metrics(adata, modality=None):

    if modality:
        adata = adata[adata.obs['modality'].str.upper() == modality.upper()]

    pt_adata = adata.obs[[dev_group_key, 'ordinal_pseudotime', 'dpt_pseudotime']]
    pt_adata[dev_group_key] = pt_adata[dev_group_key].cat.codes.to_numpy()

    pearson_corr_matrix     = pt_adata.corr(method=lambda x, y: pearsonr(x, y)[0])
    spearman_corr_matrix    = pt_adata.corr(method=lambda x, y: spearmanr(x, y)[0])
    kendall_corr_matrix     = pt_adata.corr(method=lambda x, y: kendalltau(x, y)[0])

    metrics_adata = pd.concat([
        pearson_corr_matrix['dpt_pseudotime'].iloc[:-1],
        spearman_corr_matrix['dpt_pseudotime'].iloc[:-1],
        kendall_corr_matrix['dpt_pseudotime'].iloc[:-1],
    ], axis=1)
    metrics_adata.columns = ['pearson', 'spearman', 'kendall']

    ## perform Leiden again, weith lower resolution
    res = 0.25
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')

    ## integration metrics
    lineage_clisi = clisi_graph(adata, label_key='Lineage', type_='knn', scale=True, n_cores=1)
    modality_ilisi = ilisi_graph(adata, batch_key='modality', type_='knn', scale=True, n_cores=1)
    lineage_nmi = nmi(adata, cluster_key=f'leiden_res{res}', label_key='sub_cell_type')
    lineage_ari = ari(adata, cluster_key=f'leiden_res{res}', label_key='sub_cell_type')
    age_range_nmi = nmi(adata, cluster_key=f'leiden_res{res}', label_key='Age_Range')
    age_range_ari = ari(adata, cluster_key=f'leiden_res{res}', label_key='Age_Range')
    integration_adata = pd.DataFrame(
        np.stack([lineage_clisi, modality_ilisi, lineage_nmi, lineage_ari, age_range_nmi, age_range_ari])[None],
        columns=['lineage_clisi', 'modality_ilisi', 'ct_nmi', 'ct_ari', 'age_range_nmi', 'age_range_ari'])

    return metrics_adata, integration_adata

def trajectory_metrics_all(metrics_dfs, methods_list, suptitle=None, drop_metrics=['pearson', 'kendall']):

    metrics_df = pd.concat([df.T.stack().to_frame() for df in metrics_dfs], axis=1)
    metrics_df.columns = methods_list

    if drop_metrics:
        metrics_df = metrics_df.drop(index=drop_metrics)

    metrics_df_melted = metrics_df.reset_index().melt(id_vars=['level_0', 'level_1'], var_name='method', value_name='correlation')
    metrics_df_melted = metrics_df_melted.rename(columns={'level_0': 'metric', 'level_1': 'reference'})

    ## rename reference variables
    metrics_df_melted['reference'] = metrics_df_melted['reference'].replace({
        'ordinal_pseudotime': 'ordinal pseud.',
        'Age_Range': 'age range',
    })

    ## order reference variables such that appear in proper order in plot
    metrics_df_melted['reference'] = pd.Categorical(metrics_df_melted['reference'], categories=['ordinal pseud.', 'age range'], ordered=True)

    if metrics_df_melted['metric'].nunique() > 1:

        g = sns.catplot(
            data=metrics_df_melted,
            x='reference',
            y='correlation',
            col='metric',
            hue='method',
            kind='bar',
            height=5,
            aspect=.75
        )
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(30)

    elif metrics_df_melted['metric'].nunique() == 1:

        corr_type = metrics_df_melted['metric'].unique().item()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax = sns.barplot(
            data=metrics_df_melted,
            x='reference',
            y='correlation',
            hue='method',
            palette='Set2',
            linewidth=2, edgecolor='.5',
            ax=ax
        )
        plt.xticks(rotation=20)
        plt.xlabel('reference variable')
        plt.ylabel(f'{corr_type} corr.')

        # Move legend to the right of the figure
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Method', frameon=True, edgecolor='black')
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend

    if suptitle:
        plt.suptitle(suptitle)

    #plt.tight_layout()

    return metrics_df, fig

def integration_metrics_all(integration_dfs, methods_list, suptitle=None, drop_columns=None):

    integration_df = pd.concat(integration_dfs, axis=0)
    integration_df.index = methods_list

    if drop_columns:
        integration_df = integration_df.drop(columns=drop_columns)

    integration_df_melted = integration_df.reset_index().melt(id_vars=['index'], var_name='method', value_name='score')
    integration_df_melted.rename(columns={'index': 'method', 'method': 'metric'}, inplace=True)

    integration_df_melted.replace({
        'lineage_clisi': 'cLISI - lineage',
        'modality_ilisi': 'iLISI - modality',
        'ct_nmi': 'NMI - cell type',
        'ct_ari': 'ARI - cell type',
        'age_range_nmi': 'NMI - age range',
        'age_range_ari': 'ARI - age range',
        }, inplace=True)

    # Create catplot which returns a FacetGrid
    g = sns.catplot(
        data=integration_df_melted,
        palette='Set2',
        aspect=0.6,
        edgecolor='.5',
        y='score',
        hue='method',
        col='metric',
        kind='bar',
        sharex=False,
        sharey=False
    )

    # Set the same number of evenly spaced yticklabels for each subplot,
    # and ensure the last yticklabel is a multiple of 0.10 for each subplot
    n_yticks = 5  # or choose another number as appropriate
    for ax in g.axes.flat:
        if ax is not None:
            y_min, y_max = ax.get_ylim()
            # Find the next highest multiple of 0.10 for the top ytick
            y_max_mult010 = np.ceil(y_max * 10) / 10
            if y_max_mult010 < y_max:
                y_max_mult010 += 0.10
            # Now generate evenly spaced ticks from y_min to y_max_mult010
            yticks = np.linspace(y_min, y_max_mult010, n_yticks)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.2f}" for y in yticks])
            # Get the metric name from the subplot title or column
            metric_name = ax.get_title().split(' = ')[-1] if ' = ' in ax.get_title() else ax.get_title()
            ax.set_xticklabels([metric_name])
            ax.set_xlabel('')  # Remove the default x-label
            ax.set_title('')

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend

    # Add box around legend, like for corrs plots
    legend = g._legend
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)

    return integration_df, g.fig

from scipy import sparse

def gene_obs_pearson(adata, obs_key, layer=None, mask=None):
    """
    Pearson r for each gene vs adata.obs[obs_key], using sparse algebra.
    Returns a pandas Series indexed by adata.var_names.
    """
    # pick matrix
    X = adata.layers[layer] if layer is not None else adata.X
    if not sparse.isspmatrix(X):
        X = sparse.csr_matrix(X)  # ok if already dense, but we keep API consistent

    # cells to use
    y = adata.obs[obs_key].to_numpy()
    valid = np.isfinite(y)
    if mask is not None:
        valid &= mask
    X = X[valid]
    y = y[valid].astype(float)

    n = y.shape[0]
    if n < 2:
        raise ValueError("Not enough valid cells.")

    # center y
    y = y - y.mean()
    # precompute ||y_centered||
    y_norm = np.sqrt(np.dot(y, y))
    if y_norm == 0:
        # y is constant → correlations undefined
        return pd.Series(np.nan, index=adata.var_names, name=f"corr({obs_key})")

    # sparse algebra:
    # num_j = (x_j · y_centered)
    # den_j = sqrt( sum(x_j^2) - (sum(x_j)^2)/n ) * ||y_centered||
    # r_j = num_j / den_j
    # All of this is vectorized across genes.

    # sums per gene (1×G)
    s1 = np.asarray(X.sum(axis=0)).ravel()                      # sum x_j
    s2 = np.asarray(X.power(2).sum(axis=0)).ravel()             # sum x_j^2
    # dot with centered y (G,)
    num = X.T.dot(y)                                            # x_j · y

    varx = s2 - (s1**2) / n                                     # n * Var(x_j) with population defn
    den = np.sqrt(varx) * y_norm

    # avoid divide-by-zero for constant genes
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
        r[~np.isfinite(r)] = np.nan

    return pd.Series(r, index=adata.var_names, name=f"corr({obs_key})")

def cross_batch_graph_neighbors_with_summary(
    adata,
    from_value="source",
    to_value="target",
    batch_key="source_or_target",
    cluster_key="ClustersMapped",
    k=5,
    use="distances",  # or "connectivities"
    agg_cols=("ordinal_pseudotime", "dpt_pseudotime"),
):
    """
    For each cell in `from_value`, find its k nearest neighbors from `to_value`
    (using the precomputed neighbors graph in `adata.obsp[use]`), report the most
    common neighbor cluster, and compute unweighted means of selected obs columns
    across those neighbors.

    Returns a tidy DataFrame with one row per source cell.
    """
    if use not in adata.obsp:
        raise KeyError(f"`{use}` not found in adata.obsp. Did you run sc.pp.neighbors?")

    G = adata.obsp[use].tocsr()

    from_mask = (adata.obs[batch_key].values == from_value)
    to_mask   = (adata.obs[batch_key].values == to_value)

    # Ensure agg_cols exist
    agg_cols = tuple(agg_cols) if agg_cols is not None else tuple()
    missing = [c for c in agg_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Columns missing in adata.obs: {missing}")

    rows = []
    for i in np.where(from_mask)[0]:
        idx = G[i].indices
        vals = G[i].data

        # keep only neighbors in the target dataset
        keep = to_mask[idx]
        idx, vals = idx[keep], vals[keep]

        if idx.size == 0:
            # no cross-dataset neighbors → NaNs
            row = {"cell": adata.obs_names[i],
                   "most_common_cluster": np.nan,
                   "cluster_count": 0}
            for c in agg_cols:
                row[f"mean_neighbors_{c}"] = np.nan
            rows.append(row)
            continue

        # choose top-k neighbors
        order = np.argsort(vals if use == "distances" else -vals)[:k]
        n_idx = idx[order]

        neighbor_clusters = adata.obs[cluster_key].iloc[n_idx].to_numpy()
        counts = Counter(neighbor_clusters)
        top_cluster, top_count = counts.most_common(1)[0]

        row = {
            "cell": adata.obs_names[i],
            "most_common_cluster": top_cluster,
            "cluster_count": int(top_count),
        }

        # compute plain means for each column
        for c in agg_cols:
            vals_c = adata.obs[c].iloc[n_idx].to_numpy()
            row[f"mean_neighbors_{c}"] = np.nanmean(vals_c)

        rows.append(row)

    return pd.DataFrame(rows)


def magic_diffusion(adata, var, split_var=None, t=3):

    if f'{var}_magic' in adata.obs.columns:
        adata.obs.drop(columns=[f'{var}_magic'], inplace=True)

    if split_var is None:
        W = adata.obsp["connectivities"]                  # csr_matrix (n_cells × n_cells)
        G = gt.Graph(W, precomputed="adjacency")
        op = magic.MAGIC(t=1, solver='exact').fit(adata.X, graph=G)    # compute operator only
        D = op.diff_op                            # (n_cells × n_cells) diffusion operator
        
        x = adata.obs[var].to_numpy().reshape(-1, 1)
        x_t = x.copy()
        for _ in range(t): x_t = D @ x_t
        x_smooth = x_t.ravel()
        adata.obs[f'{var}_magic'] = x_smooth

    elif split_var is not None:

        adata_splits_var = []
        for split_value in adata.obs[split_var].unique().dropna():

            adata_split = adata[adata.obs[split_var]==split_value].copy()
            W = adata_split.obsp["connectivities"]                  # csr_matrix (n_cells × n_cells)
            G = gt.Graph(W, precomputed="adjacency")
            op = magic.MAGIC(t=1, solver='exact').fit(adata_split.X, graph=G)    # compute operator only
            D = op.diff_op                            # (n_cells × n_cells) diffusion operator
            
            x = adata_split.obs[var].to_numpy().reshape(-1, 1)
            x_t = x.copy()
            for _ in range(t): x_t = D @ x_t
            x_smooth = x_t.ravel()
            adata_split.obs[f'{var}_magic'] = x_smooth
            adata_splits_var.append(adata_split.obs[f'{var}_magic'])
        
        adata_splits_var = pd.concat(adata_splits_var, axis=0)
        adata.obs = adata.obs.merge(adata_splits_var, left_index=True, right_index=True, how='left')

    return adata

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

def hac_weighted_mean_test(df, path, direction='case_more_control'):

    # per-(leiden, condition) summaries
    g = (df.groupby(['leiden', 'Condition'])['ordinal_pseudotime']
        .agg(mean='mean', sd='std', n='size')
        .reset_index())
    g['se'] = g['sd'] / np.sqrt(g['n'])

    # wide table
    wide = g.pivot(index='leiden', columns='Condition')

    # enforce the intended Leiden order
    order = list(path)
    wide = wide.reindex(order)

    # align means/SEs and compute difference + its SE
    m_case  = wide['mean']['case']
    m_ctrl  = wide['mean']['control']
    se_case = wide['se']['case']
    se_ctrl = wide['se']['control']

    valid = m_case.notna() & m_ctrl.notna() & se_case.notna() & se_ctrl.notna()
    m_case, m_ctrl, se_case, se_ctrl = m_case[valid], m_ctrl[valid], se_case[valid], se_ctrl[valid]

    if direction == 'case_more_control':
        d   = (m_case - m_ctrl)
    elif direction == 'control_more_case':
        d   = (m_ctrl - m_case)
    elif direction == 'all':
        d   = (m_case - m_ctrl)

    seD = np.sqrt(se_case**2 + se_ctrl**2)
    d_df = pd.concat([d.to_frame().rename(columns={0:"d"}), seD.to_frame().rename(columns={0:"seD"})], axis=1)

    # inverse-variance weights (clip zeros to avoid inf)
    eps = 1e-12
    w = 1.0 / np.maximum(seD**2, eps)

    # intercept-only WLS with HAC (Newey–West) SEs
    X = np.ones((len(d), 1))
    ols = sm.WLS(d.values, X, weights=w.values).fit()

    # choose maxlags (e.g., 2 or 3; or ~= n^(1/3))
    maxlags = max(1, int(round(len(d) ** (1/3))))
    hac = ols.get_robustcov_results(cov_type='HAC', maxlags=maxlags, use_correction=True)

    beta = hac.params[0]
    se   = hac.bse[0]
    z    = beta / se

    # one-sided p for H1: mean(diff) > 0
    p_one_sided = norm.sf(z)   # == 1 - norm.cdf(z)

    print(f"Leiden K = {len(d)}, HAC maxlags = {maxlags}")
    print(f"Weighted mean diff = {beta:.3f} (HAC SE {se:.3f}), one-sided p = {p_one_sided:.4g}")

    return d_df, p_one_sided

#%% run PAGA
print('ECLARE'); subsampled_eclare_adata = paga_analysis(subsampled_eclare_adata, dev_group_key=dev_group_key, cell_group_key=cell_group, correct_imbalance=True)
print('scJoint'); scJoint_adata = paga_analysis(scJoint_adata, dev_group_key=dev_group_key, cell_group_key=cell_group)
print('scGLUE'); glue_adata = paga_analysis(glue_adata, dev_group_key=dev_group_key, cell_group_key=cell_group)

for source_dataset in source_datasets:
    print(f'KD-CLIP {source_dataset}'); subsampled_kd_clip_adatas[source_dataset] = paga_analysis(subsampled_kd_clip_adatas[source_dataset], dev_group_key=dev_group_key, cell_group_key=cell_group, correct_imbalance=True)
    print(f'CLIP {source_dataset}'); subsampled_clip_adatas[source_dataset] = paga_analysis(subsampled_clip_adatas[source_dataset], dev_group_key=dev_group_key, cell_group_key=cell_group, correct_imbalance=True)

'''
subsampled_eclare_adata_rna = paga_analysis(subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'])
subsampled_eclare_adata_atac = paga_analysis(subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'])

scJoint_adata_rna = paga_analysis(scJoint_adata[scJoint_adata.obs['modality'].str.upper() == 'RNA'])
scJoint_adata_atac = paga_analysis(scJoint_adata[scJoint_adata.obs['modality'].str.upper() == 'ATAC'])

glue_adata_rna = paga_analysis(glue_adata[glue_adata.obs['modality'].str.upper() == 'RNA'])
glue_adata_atac = paga_analysis(glue_adata[glue_adata.obs['modality'].str.upper() == 'ATAC'])

student_rna.obs['modality'] = 'RNA'
student_atac.obs['modality'] = 'ATAC'

student_rna = paga_analysis(student_rna)
student_atac = paga_analysis(student_atac)
'''

#%% import source datasets

source_clusters_dict = {
    'PFC_Zhu': ['RG', 'IPC', 'EN-fetal-early', 'EN-fetal-late', 'EN'], # all: ['EN-fetal-late', 'IN-fetal', 'VSMC', 'Endothelial', 'RG', 'OPC', 'IN-CGE', 'Microglia', 'IN-MGE', 'EN-fetal-early', 'Astrocytes', 'IPC', 'Pericytes', 'EN', 'Oligodendrocytes']
    'PFC_V1_Wang': ["RG-vRG", "IPC-EN", "EN-newborn", "EN-IT-immature", "EN-non-IT-immature", "EN-L2_3-IT", "EN-L4-IT-V1", "EN-L4-IT", "EN-L5-IT", "EN-L6-IT", "EN-L5_6-NP", "EN-L5-ET", "EN-L6-CT", "EN-L6b"],
}

source_adatas = {}

for source_dataset in source_datasets:

    args = SimpleNamespace(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        genes_by_peaks_str=genes_by_peaks_str,
        ignore_sources=[None],
        source_dataset_embedder=None,
    )

    source_setup_func = return_setup_func_from_dataset(source_dataset)
    source_rna, source_atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = source_setup_func(args, return_type='data')

    source_clusters = source_clusters_dict[source_dataset]

    source_rna.obs['modality'] = 'RNA'
    source_atac.obs['modality'] = 'ATAC'

    source_rna_atac = anndata.concat([source_rna[source_rna.obs[cell_group].isin(source_clusters)], source_atac[source_atac.obs[cell_group].isin(source_clusters)]], axis=0)
    source_rna_atac_sub = subsample_adata(source_rna_atac, subsample, ['modality', cell_group], subsample_type='balanced')

    source_rna_sub = source_rna[source_rna.obs_names.isin(source_rna_atac_sub.obs_names)]
    source_atac_sub = source_atac[source_atac.obs_names.isin(source_rna_atac_sub.obs_names)]

    #source_rna_sub = subsample_adata(source_rna[source_rna.obs[cell_group].isin(source_clusters)], subsample, [cell_group], subsample_type='stratified')
    #source_atac_sub = subsample_adata(source_atac[source_atac.obs[cell_group].isin(source_clusters)], subsample, [cell_group], subsample_type='stratified')

    source_rna_cells_sub = torch.from_numpy(source_rna_sub.X.toarray().astype(np.float32))
    source_atac_cells_sub = torch.from_numpy(source_atac_sub.X.toarray().astype(np.float32))
    
    ## KD-CLIP embeddings
    source_rna_latents_sub, _ = kd_clip_student_models[source_dataset](source_rna_cells_sub, modality=0)
    source_atac_latents_sub, _ = kd_clip_student_models[source_dataset](source_atac_cells_sub, modality=1)

    source_rna_latents_sub = source_rna_latents_sub.detach().cpu().numpy()
    source_atac_latents_sub = source_atac_latents_sub.detach().cpu().numpy()

    source_latents_sub = np.vstack([source_rna_latents_sub, source_atac_latents_sub])
    source_adatas[source_dataset] = sc.AnnData(
        X=source_latents_sub,
        obs=pd.concat([source_rna_sub.obs.assign(modality='RNA'), source_atac_sub.obs.assign(modality='ATAC')], axis=0),
    )

    ## CLIP embeddings
    source_rna_latents_sub, _ = clip_models[source_dataset](source_rna_cells_sub, modality=0)
    source_atac_latents_sub, _ = clip_models[source_dataset](source_atac_cells_sub, modality=1)

    source_rna_latents_sub = source_rna_latents_sub.detach().cpu().numpy()
    source_atac_latents_sub = source_atac_latents_sub.detach().cpu().numpy()

    source_latents_sub = np.vstack([source_rna_latents_sub, source_atac_latents_sub])
    source_adatas[source_dataset + '_clip'] = sc.AnnData(
        X=source_latents_sub,
        obs=pd.concat([source_rna_sub.obs.assign(modality='RNA'), source_atac_sub.obs.assign(modality='ATAC')], axis=0),
    )

    if ordinal_source_dataset == source_dataset:

        ordinal_rna_logits_sub, ordinal_rna_probas_sub, ordinal_rna_latents_sub = ordinal_model.to('cpu')(source_rna_cells_sub, modality=0)
        ordinal_atac_logits_sub, ordinal_atac_probas_sub, ordinal_atac_latents_sub = ordinal_model.to('cpu')(source_atac_cells_sub, modality=1)

        ordinal_rna_prebias_sub = ordinal_model.ordinal_layer_rna.coral_weights(ordinal_rna_latents_sub)
        ordinal_atac_prebias_sub = ordinal_model.ordinal_layer_atac.coral_weights(ordinal_atac_latents_sub)

        ordinal_rna_pt_sub = torch.from_numpy(match_histograms(ordinal_rna_prebias_sub.detach().cpu().numpy().flatten(), subsampled_kd_clip_adatas[source_dataset].obs['ordinal_pseudotime'].values))
        ordinal_atac_pt_sub = torch.from_numpy(match_histograms(ordinal_atac_prebias_sub.detach().cpu().numpy().flatten(), subsampled_kd_clip_adatas[source_dataset].obs['ordinal_pseudotime'].values))

        source_adatas[source_dataset].obs['ordinal_pseudotime'] = torch.cat([ordinal_rna_pt_sub, ordinal_atac_pt_sub], axis=0)
        source_adatas[source_dataset + '_clip'].obs['ordinal_pseudotime'] = torch.cat([ordinal_rna_pt_sub, ordinal_atac_pt_sub], axis=0)


#for source_dataset in source_datasets:
#    print(f'KD-CLIP {source_dataset}'); source_adatas[source_dataset] = paga_analysis(source_adatas[source_dataset], dev_group_key='dev_stage', cell_group_key=cell_group)

#%% Combine source and target data

if source_dataset == 'PFC_V1_Wang':
    source_adatas[source_dataset].obs.rename(columns={'type': 'ClustersMapped', 'Group': 'Age'}, inplace=True)
elif source_dataset == 'PFC_Zhu':
    source_adatas[source_dataset].obs.rename(columns={'Cell type': 'ClustersMapped', 'dev_stage': 'Age'}, inplace=True)

source_adatas[source_dataset].obs['SubClusters'] = source_adatas[source_dataset].obs['ClustersMapped'].copy()
source_adatas[source_dataset].obs['Age_bins'] = source_adatas[source_dataset].obs['Age'].copy()

source_adatas[source_dataset].obs['source_or_target'] = 'source'
source_adatas[source_dataset].obs['dataset_name'] = source_dataset

subsampled_kd_clip_adatas[source_dataset].obs['source_or_target'] = 'target'
subsampled_kd_clip_adatas[source_dataset].obs['dataset_name'] = target_dataset

source_target_adata = anndata.concat([ source_adatas[source_dataset], subsampled_kd_clip_adatas[source_dataset] ], axis=0)
source_target_adata.obs = source_target_adata.obs.merge(subsampled_kd_clip_adatas[source_dataset].obs['Condition'], left_index=True, right_index=True, how='left')
source_target_adata.obs['Age'] = pd.Categorical(source_target_adata.obs['Age'])

#source_target_adata = source_target_adata[source_target_adata.obs['ClustersMapped'].isin(['ExN', 'EN-fetal-early', 'EN-fetal-late', 'EN'])]
#source_target_adata = source_target_adata[source_target_adata.obs['ClustersMapped'].isin(['ExN', 'RG-vRG', 'IPC-EN', 'EN-newborn', 'EN-IT-immature', 'EN-L2_3-IT'])]

source_target_adata = source_target_adata[source_target_adata.obs['SubClusters'].isin(
    ['RG-vRG', 'IPC-EN', 'EN-newborn', 'EN-IT-immature','EN-L2_3-IT'] + \
    ['ExN1','ExN2','ExN1_L23','ExN2_L23'] + \
    ['ExN6', 'ExN2_L23', 'ExN9_L23']
        )]

source_target_adata = paga_analysis(source_target_adata, dev_group_key='Age', cell_group_key='ClustersMapped', correct_imbalance=True)
#source_target_adata.write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'source_target_adata_{source_dataset}_{target_dataset}.h5ad'))

## plot FA embeddings comparing source and target datasets
colors = ['source_or_target', 'modality', 'dpt_pseudotime', 'ordinal_pseudotime', 'ClustersMapped']
f1 = sc.pl.draw_graph(source_target_adata, color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f2 = sc.pl.draw_graph(source_target_adata[source_target_adata.obs['source_or_target']=='target'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f3 = sc.pl.draw_graph(source_target_adata[source_target_adata.obs['source_or_target']=='source'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f1.show(); f2.show(); f3.show()

source_target_adata = magic_diffusion(source_target_adata, 'ordinal_pseudotime', split_var='Condition', t=3)

target_adata = source_target_adata[source_target_adata.obs['source_or_target']=='target'].copy()

colors = ['Condition', 'modality', 'ordinal_pseudotime_magic']
f1 = sc.pl.draw_graph(target_adata, color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f2 = sc.pl.draw_graph(target_adata[target_adata.obs['Condition']=='case'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f3 = sc.pl.draw_graph(target_adata[target_adata.obs['Condition']=='control'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f1.show(); f2.show(); f3.show()

#%% perform label transfer between source and target datasets

df = cross_batch_graph_neighbors_with_summary(
    source_target_adata,
    from_value="target",
    to_value="source",
    batch_key="source_or_target",
    cluster_key="ClustersMapped",
    k=10,
    use="distances",
    agg_cols=("ordinal_pseudotime_magic", "ordinal_pseudotime", "dpt_pseudotime")
)
df.head()

df['most_common_cluster'] = pd.Categorical(df['most_common_cluster'], categories=np.unique(source_clusters + target_clusters), ordered=True)

count_in_source = source_target_adata.obs.loc[source_target_adata.obs['source_or_target'] == 'source', 'ClustersMapped'].value_counts()
count_in_source.name = 'count_in_source'

count_in_match = df['most_common_cluster'].value_counts()
count_in_match.name = 'count_in_match'

df = df[df['most_common_cluster'].isin(count_in_match[count_in_match > 25].index)]
df['most_common_cluster'] = df['most_common_cluster'].cat.remove_unused_categories()

df = df.merge(source_target_adata.obs.loc[source_target_adata.obs['source_or_target']=='target', ['ordinal_pseudotime_magic', 'ordinal_pseudotime', 'dpt_pseudotime']], left_on='cell', right_index=True, how='left')
df = df.merge(subsampled_kd_clip_adatas[source_dataset].obs['Condition'], left_on='cell', right_index=True, how='left')

## merge most_common_cluster onto adata.obs
source_target_adata.obs = source_target_adata.obs.merge(df.set_index('cell')['most_common_cluster'], left_index=True, right_index=True, how='left')


fig, ax = plt.subplots(1, 2, figsize=[df['most_common_cluster'].nunique() + 8, 7], sharex=True)
#sns.boxplot(df, x='most_common_cluster', y='mean_neighbors_ordinal_pseudotime', ax=ax[0])
#sns.boxplot(df, x='most_common_cluster', y='mean_neighbors_dpt_pseudotime', ax=ax[1])
sns.barplot(df, x='most_common_cluster', y='mean_neighbors_ordinal_pseudotime', hue='Condition', errorbar='se', ax=ax[0])
sns.barplot(df, x='most_common_cluster', y='mean_neighbors_dpt_pseudotime', hue='Condition', errorbar='se', ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30); ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30)
ax[0].set_ylabel('Ordinal Pseudotime'); ax[1].set_ylabel('DPT Pseudotime')
ax[0].set_xlabel(''); ax[1].set_xlabel('')
plt.tight_layout(); plt.show()

def devmdd_fig1(df, source_dataset, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'devmdd_fig1.svg')):

    if source_dataset == 'PFC_Zhu':
        fig, ax = plt.subplots(1, 1, figsize=[5, 6])

        # Barplot
        pastel2_colors = sns.color_palette('Pastel2')
        # Flip the order of the first two colors
        pastel2_colors = [pastel2_colors[1], pastel2_colors[0]] + list(pastel2_colors[2:])
        sns.barplot(
            df,
            x='most_common_cluster',
            y='mean_neighbors_ordinal_pseudotime',
            hue='Condition',
            errorbar='se',
            palette='Pastel1',
            linewidth=2,
            edgecolor='.5',
            ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_ylabel('Ordinal Pseudotime')
        ax.set_xlabel('')
        ax.set_ylim(30, 62)

    elif source_dataset == 'PFC_V1_Wang':

        df['most_common_cluster'] = df['most_common_cluster'].replace({'EN-L2_3-IT': 'EN-ITs', 'EN-L4-IT': 'EN-ITs', 'EN-L5-IT': 'EN-ITs', 'EN-L6-IT': 'EN-ITs'})

        fig, ax = plt.subplots(1, 1, figsize=[5, 3])

        # Barplot
        pastel2_colors = sns.color_palette('Pastel2')
        # Flip the order of the first two colors
        pastel2_colors = [pastel2_colors[1], pastel2_colors[0]] + list(pastel2_colors[2:])
        sns.barplot(
            df,
            x='mean_neighbors_ordinal_pseudotime',
            y='most_common_cluster',
            hue='Condition',
            errorbar='se',
            palette='Pastel1',
            linewidth=2,
            edgecolor='.5',
            ax=ax)

        ax.set_xlabel('Ordinal Pseudotime')
        ax.set_ylabel('')
        ax.set_xlim(30, 70)

    plt.tight_layout()
    fig.savefig(manuscript_figpath, bbox_inches='tight', dpi=300)
    plt.close()

def devmdd_fig2(target_adata, source_adata, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results')):
    #pos = adata.uns['paga']['pos']

    sc.settings._vector_friendly = True
    sc.settings.figdir = manuscript_figpath

    ## target data
    adata = target_adata.copy()
    sc.pl.umap(adata, color='most_common_cluster', title='target data (imputed)', save='_devmdd_fig3.svg')

    perm_idxs = np.flip(np.random.permutation(len(adata)))
    size_0_for_na = np.where(adata.obs['most_common_cluster'].isna(), 0, 25)

    sc.tl.paga(adata, groups='leiden')
    sc.pl.paga_compare(adata[perm_idxs])
    sc.pl.paga_compare(adata[perm_idxs], color='most_common_cluster', legend_loc='right margin', right_margin=0.9, title='', size=size_0_for_na[perm_idxs], node_size_scale=3, frameon=True, save='_devmdd_fig2.svg') # will prepend "paga_compare" to the filename

    #sc.pl.paga_compare(adata[perm_idxs], color='ordinal_pseudotime_magic', size=size_0_for_na[perm_idxs], right_margin=0.4)

    sc.pl.draw_graph(adata[perm_idxs], color='leiden', size=size_0_for_na[perm_idxs], na_in_legend=False, cmap='plasma', title='', save='_devmdd_fig5.svg')

    #sc.pl.draw_graph(adata[perm_idxs], color='most_common_cluster', size=size_0_for_na[perm_idxs], na_in_legend=False, title='')
    #sc.pl.paga(adata)

    ## source data
    adata = source_adata.copy()
    adata.obs['ClustersMapped'] = pd.Categorical(adata.obs['ClustersMapped'], categories=target_adata.obs['most_common_cluster'].cat.categories, ordered=True)
    sc.pl.umap(adata, color='ClustersMapped', title='source data', legend_loc=None, save='_devmdd_fig4.svg')



target_adata = source_target_adata[source_target_adata.obs['source_or_target']=='target'].copy()
colors_uns_keys = [key for key in list(target_adata.uns.keys()) if '_colors' in key]
for key in colors_uns_keys: del target_adata.uns[key]

## case vs control ordinal pseudotime for each leiden cluster
leiden_sorted = target_adata.obs.groupby('leiden')['ordinal_pseudotime'].mean().sort_values().index.tolist()
target_adata.obs['leiden'] = pd.Categorical(target_adata.obs['leiden'], categories=leiden_sorted, ordered=True)

leiden_diff = (target_adata.obs.groupby(['leiden','Condition'])['ordinal_pseudotime'].mean()
          .unstack()
          .pipe(lambda m: m['case'] - m['control']))

leiden_sorted_case_more_control = leiden_diff[leiden_diff > 0].index.tolist()
leiden_sorted_control_more_case = leiden_diff[leiden_diff < 0].index.tolist()

sc.pl.embedding(target_adata, color='leiden', basis='X_draw_graph_fa', na_in_legend=False, palette='plasma')
#sc.pl.paga_compare(target_adata, color="ordinal_pseudotime", threshold=0.25, right_margin=0.5)


paths = [
    ("MDD_ExN_all", leiden_sorted),
    ("MDD_ExN_case_more_control", leiden_sorted_case_more_control),
    ("MDD_ExN_control_more_case", leiden_sorted_control_more_case),
]

## plot rolling mean and standard error of ordinal pseudotime for each leiden cluster
window_len = 3

means = target_adata.obs.groupby(['leiden','modality','Condition'])['ordinal_pseudotime'].mean()
means_pivot = means.reset_index().pivot_table(index='leiden', columns=['modality','Condition'], values='ordinal_pseudotime')
means_pivot = means_pivot.rolling(window_len, min_periods=0).mean()
means_pivot = means_pivot.melt(ignore_index=False).rename(columns={'value': 'ordinal_pseudotime'})
means_pivot['Condition_modality'] = means_pivot[['Condition','modality']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)

std_errs = target_adata.obs.groupby(['leiden','modality','Condition'])['ordinal_pseudotime'].std() / np.sqrt(target_adata.obs.groupby(['leiden','modality','Condition'])['ordinal_pseudotime'].count())
std_errs_pivot = std_errs.reset_index().pivot_table(index='leiden', columns=['modality','Condition'], values='ordinal_pseudotime')
std_errs_pivot = std_errs_pivot.rolling(window_len, min_periods=0).mean()
std_errs_pivot = std_errs_pivot.melt(ignore_index=False).rename(columns={'value': 'ordinal_pseudotime'})
std_errs_pivot['Condition_modality'] = std_errs_pivot[['Condition','modality']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)

## resort Leiden clusters
leiden_sorted = means_pivot.groupby('leiden')['ordinal_pseudotime'].mean().sort_values().index.tolist()
means_pivot.index = pd.CategoricalIndex(means_pivot.index, categories=leiden_sorted, ordered=True)
std_errs_pivot.index = pd.CategoricalIndex(std_errs_pivot.index, categories=leiden_sorted, ordered=True)
means_pivot.sort_index(inplace=True)
std_errs_pivot.sort_index(inplace=True)

for descr, path in [paths[0]]:

    means_pivot_path = means_pivot[means_pivot.index.isin(path)].reset_index()
    std_errs_pivot_path = std_errs_pivot[std_errs_pivot.index.isin(path)].reset_index()

    #means_pivot_path['leiden'] = pd.Categorical(means_pivot_path['leiden'], categories=path, ordered=True)
    #std_errs_pivot_path['leiden'] = pd.Categorical(std_errs_pivot_path['leiden'], categories=path, ordered=True)

    fig, ax = plt.subplots(1,3, figsize=[15,8], sharex=True, sharey=True)
    sns.lineplot(data=means_pivot_path, x='leiden', y='ordinal_pseudotime', color='grey', errorbar=None, marker='.', markersize=20, linewidth=2, ax=ax[0])
    sns.lineplot(data=means_pivot_path[means_pivot_path['modality']=='RNA'], x='leiden', y='ordinal_pseudotime', palette='Pastel1', hue='Condition_modality', hue_order=['case_RNA', 'control_RNA'], errorbar=None, marker='.', markersize=20, linewidth=2, ax=ax[1])
    sns.lineplot(data=means_pivot_path[means_pivot_path['modality']=='ATAC'], x='leiden', y='ordinal_pseudotime', palette='Pastel1', hue='Condition_modality', hue_order=['case_ATAC', 'control_ATAC'], errorbar=None, marker='.', markersize=20, linewidth=2, linestyle='--', ax=ax[2])

    x = means_pivot_path['leiden']; low = (means_pivot_path['ordinal_pseudotime'] - std_errs_pivot_path['ordinal_pseudotime']).values; high = (means_pivot_path['ordinal_pseudotime'] + std_errs_pivot_path['ordinal_pseudotime']).values
    high_low = x.to_frame().assign(low=low, high=high)
    ax[0].fill_between(x.unique().tolist(), high_low.groupby('leiden')['low'].mean(), high_low.groupby('leiden')['high'].mean(), alpha=0.15, color='grey')
    ax[1].fill_between(x[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], low[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], high[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], alpha=0.3, color=sns.color_palette('Pastel1')[0])
    ax[1].fill_between(x[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], low[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], high[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], alpha=0.3, color=sns.color_palette('Pastel1')[1])
    ax[2].fill_between(x[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], low[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], high[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], alpha=0.3, color=sns.color_palette('Pastel1')[0])
    ax[2].fill_between(x[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], low[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], high[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], alpha=0.3, color=sns.color_palette('Pastel1')[1])

    ax[0].set_title('all nuclei'); ax[1].set_title('RNA'); ax[2].set_title('ATAC')
    plt.suptitle(f'{descr}')
    plt.tight_layout(); plt.show()

def devmdd_fig4(lineplots_fig, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'devmdd_fig4.svg')):

    lineplots_fig.suptitle('')
    lineplots_fig.savefig(manuscript_figpath, bbox_inches='tight', dpi=300)
    plt.close()

def dev_fig4(adata, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'dev_fig4.svg')):
    sc.settings._vector_friendly = True
    #adata = source_target_adata.copy()
    permute_idxs = np.random.permutation(len(adata))
    adata.obs = adata.obs.rename(columns={
        'dpt_pseudotime': 'DPT pseudotime',
        'ordinal_pseudotime': 'ordinal pseudotime',
        'ClustersMapped': 'developmental cell type'
        })
    adata.uns['lineage_colors'] = ['purple','orange']
    colors = ['modality','developmental cell type', 'ordinal pseudotime', 'leiden']

    adata.obs['developmental cell type'] = pd.Categorical(adata.obs['developmental cell type'], categories=['EN-fetal-early','EN-fetal-late','EN','ExN'], ordered=True)
    adata.obs['developmental cell type'] = adata.obs['developmental cell type'].replace({'ExN':np.nan})
    dev_fig4 = sc.pl.draw_graph(adata[permute_idxs], color=colors, wspace=0.5, ncols=2, return_fig=True, na_in_legend=False)

    print(f'Saving figure to {manuscript_figpath}')
    dev_fig4.savefig(manuscript_figpath, bbox_inches='tight', dpi=300)

## plot embedding density for each cell type
source_target_adata.obs['ClustersMapped'] = pd.Categorical(source_target_adata.obs['ClustersMapped'], categories=source_target_adata.obs.groupby('ClustersMapped')['ordinal_pseudotime'].mean().sort_values().index.tolist(), ordered=True).remove_unused_categories()
sc.tl.embedding_density(source_target_adata, basis='draw_graph_fa', groupby='ClustersMapped')
sc.pl.embedding_density(source_target_adata, basis='draw_graph_fa', key='draw_graph_fa_density_ClustersMapped')

target_adata.obs['SubClusters'] = pd.Categorical(target_adata.obs['SubClusters'], categories=target_adata.obs.groupby('SubClusters')['ordinal_pseudotime'].mean().sort_values().index.tolist(), ordered=True).remove_unused_categories()
sc.tl.embedding_density(target_adata, basis='draw_graph_fa', groupby='SubClusters')
sc.pl.embedding_density(target_adata[target_adata.obs['modality']=='ATAC'], basis='draw_graph_fa', key='draw_graph_fa_density_SubClusters')
sc.pl.embedding_density(target_adata[target_adata.obs['modality']=='RNA'], basis='draw_graph_fa', key='draw_graph_fa_density_SubClusters')


subsampled_eclare_adata = target_adata.copy()
'''
## case vs control leiden cluster proportions
plt.figure(figsize=[8,6])
leiden_proportions = target_adata.obs.groupby('Condition')['leiden'].value_counts(normalize=True).swaplevel().sort_index()
sns.barplot(data=leiden_proportions.reset_index(), x='leiden', y='proportion', hue='Condition')

## case vs control most common cluster proportions
most_common_cluster_proportions = source_target_adata[source_target_adata.obs['source_or_target']=='target'].obs.groupby('Condition')['most_common_cluster'].value_counts(normalize=True).swaplevel().sort_index()
plt.figure(figsize=[5,6])
sns.barplot(data=most_common_cluster_proportions.reset_index(), x='most_common_cluster', y='proportion', hue='Condition')
plt.xticks(rotation=30)

fig, ax = plt.subplots(1, 2, figsize=[10, 7], sharex=True)
#sns.boxplot(df, x='most_common_cluster', y='ordinal_pseudotime', ax=ax[0])
#sns.boxplot(df, x='most_common_cluster', y='dpt_pseudotime', ax=ax[1])
sns.barplot(df, x='most_common_cluster', y='ordinal_pseudotime_magic', hue='Condition', errorbar='se', ax=ax[0])
sns.barplot(df, x='most_common_cluster', y='dpt_pseudotime', hue='Condition', errorbar='se', ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30); ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30)
ax[0].set_ylabel('Ordinal Pseudotime'); ax[1].set_ylabel('DPT Pseudotime')
ax[0].set_xlabel(''); ax[1].set_xlabel('')
ax[0].set_ylim(20, 55)
plt.tight_layout(); plt.show()
'''

'''
pre_EN_mapper = {
    'EN': 'EN',
    'EN-fetal-late': 'pre-EN',
    'EN-fetal-early': 'pre-EN',
    'IPC': 'pre-EN',
    'RG': 'pre-EN'
}

fig, ax = plt.subplots(1, 2, figsize=[10, 7], sharex=True)
sns.barplot(df.assign(most_common_cluster=df['most_common_cluster'].map(pre_EN_mapper)), x='most_common_cluster', y='mean_neighbors_ordinal_pseudotime', hue='Condition', errorbar='se', ax=ax[0])
sns.barplot(df.assign(most_common_cluster=df['most_common_cluster'].map(pre_EN_mapper)), x='most_common_cluster', y='mean_neighbors_dpt_pseudotime', hue='Condition', errorbar='se', ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30); ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30)
ax[0].set_ylabel('Ordinal Pseudotime'); ax[1].set_ylabel('DPT Pseudotime')
ax[0].set_xlabel(''); ax[1].set_xlabel('')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=[10, 7], sharex=True)
sns.barplot(df.assign(most_common_cluster=df['most_common_cluster'].map(pre_EN_mapper)), x='most_common_cluster', y='ordinal_pseudotime', hue='Condition', errorbar='se', ax=ax[0])
sns.barplot(df.assign(most_common_cluster=df['most_common_cluster'].map(pre_EN_mapper)), x='most_common_cluster', y='dpt_pseudotime', hue='Condition', errorbar='se', ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30); ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30)
ax[0].set_ylabel('Ordinal Pseudotime'); ax[1].set_ylabel('DPT Pseudotime')
ax[0].set_xlabel(''); ax[1].set_xlabel('')
ax[0].set_ylim(30, 50)
plt.tight_layout(); plt.show()
'''

#%% Perform t-test between Condition for each ClustersMapped
import scipy.stats as stats
# Assumes 'subsampled_eclare_adata' is the AnnData object of interest

clusters = subsampled_eclare_adata.obs['ClustersMapped'].unique()
ttest_results = []

for cluster in clusters:
    cluster_mask = subsampled_eclare_adata.obs['ClustersMapped'] == cluster
    cluster_df = subsampled_eclare_adata.obs[cluster_mask]
    conds = cluster_df['Condition'].unique()
    if len(conds) == 2:
        group1 = cluster_df[cluster_df['Condition'] == conds[0]]['ordinal_pseudotime'].dropna()
        group2 = cluster_df[cluster_df['Condition'] == conds[1]]['ordinal_pseudotime'].dropna()
        # Only perform t-test if both groups have at least 2 samples
        if len(group1) > 1 and len(group2) > 1:
            tstat, pval = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            tstat, pval = float('nan'), float('nan')
        ttest_results.append({'ClustersMapped': cluster, 
                              'Condition1': conds[0], 
                              'Condition2': conds[1], 
                              't-stat': tstat, 
                              'p-value': pval,
                              'n1': len(group1),
                              'n2': len(group2)})
    else:
        ttest_results.append({'ClustersMapped': cluster, 
                              'Condition1': None, 
                              'Condition2': None, 
                              't-stat': float('nan'), 
                              'p-value': float('nan'),
                              'n1': 0,
                              'n2': 0})

ttest_df = pd.DataFrame(ttest_results)
print("T-test results between Condition for each ClustersMapped:")
print(ttest_df)

# Optional: visualize the barplot as described
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.barplot(data=subsampled_eclare_adata.obs, x='ClustersMapped', y='ordinal_pseudotime', hue='Condition', errorbar='se')
plt.title('Ordinal Pseudotime by ClustersMapped and Condition')
plt.tight_layout()
plt.show()

#%% 3D embeddings to establish PAGA paths

#sc.tl.umap(adata, n_components=3)
#sc.pl.umap(adata, color='leiden', projection='3d')

paths = [
    ('MDD_ExN_3510', ['3', '5', '1', '0']),
    ('MDD_ExN_2640', ['2', '6', '4', '0']),
         ]
paths_leiden_clusters = np.unique(np.hstack([path[1] for path in paths]))

adata_ExN = adata[adata.obs['ClustersMapped'] == 'ExN'].copy()
adata_in_paths = adata_ExN[adata_ExN.obs['leiden'].isin(paths_leiden_clusters)].copy()
adata_target = adata_in_paths[adata_in_paths.obs['source_or_target'] == 'target'].copy()
adata_target_case = adata_target[adata_target.obs['Condition'] == 'Case'].copy()
adata_target_control = adata_target[adata_target.obs['Condition'] == 'Control'].copy()

r_case = gene_obs_pearson(adata_target_case, "dpt_pseudotime")
r_control = gene_obs_pearson(adata_target_control, "dpt_pseudotime")
r = r_case - r_control
r_tops = pd.concat([r.sort_values(ascending=False).head(10), r.sort_values(ascending=True).head(10)[::-1]], axis=0)
r_tops_index = r_tops.index.tolist()

_, axs = plt.subplots(ncols=len(paths), figsize=(10, 10), gridspec_kw={
                     'wspace': 0.05, 'left': 0.12})
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)

for ipath, (descr, path) in enumerate(paths):
    sc.pl.paga_path(
        adata=adata_target, 
        nodes=path, 
        keys=r_tops_index,
        annotations=['dpt_pseudotime'],
        groups_key='leiden',
        show_node_names=False,
        ax=axs[ipath],
        ytick_fontsize=12,
        left_margin=0.15,
        n_avg=50,
        show_yticks=True if ipath == 0 else False,
        show_colorbar=False,
        color_map='Greys',
        color_maps_annotations={'distance': 'viridis'},
        title='{} path'.format(descr),
        return_data=True,
        use_raw=False,
        show=False)

plt.show()

#%% add new columns to MDD obs data

student_rna_sub.obs = student_rna_sub.obs.merge(adata.obs.set_index('cell')[['most_common_cluster', 'dpt_pseudotime', 'leiden']], left_index=True, right_index=True, how='left')
rna_tmp = student_rna_sub[student_rna_sub.obs['ClustersMapped'] == 'ExN']

student_atac_sub.obs = student_atac_sub.obs.merge(adata.obs.set_index('cell')[['most_common_cluster', 'dpt_pseudotime', 'leiden']], left_index=True, right_index=True, how='left')
atac_tmp = student_atac_sub[student_atac_sub.obs['ClustersMapped'] == 'ExN']

r_rna = gene_obs_pearson(rna_tmp, "dpt_pseudotime")
r_tops_rna = pd.concat([r_rna.sort_values(ascending=False).head(8), r_rna.sort_values(ascending=True).head(8)[::-1]], axis=0)
r_tops_index_rna = r_tops_rna.index.tolist()

_, axs = plt.subplots(ncols=len(paths), figsize=(10, 10), gridspec_kw={
                     'wspace': 0.05, 'left': 0.12})
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)

for ipath, (descr, path) in enumerate(paths):
    sc.pl.paga_path(
        adata=tmp, 
        nodes=path, 
        keys=r_tops_index_rna,
        annotations=['dpt_pseudotime'],
        groups_key='leiden',
        show_node_names=False,
        ax=axs[ipath],
        ytick_fontsize=12,
        left_margin=0.15,
        n_avg=50,
        show_yticks=True if ipath == 0 else False,
        show_colorbar=False,
        color_map='Greys',
        color_maps_annotations={'distance': 'viridis'},
        title='{} path'.format(descr),
        return_data=True,
        use_raw=False,
        show=False)

plt.show()

r_atac = gene_obs_pearson(atac_tmp, "dpt_pseudotime")
r_tops_atac = pd.concat([r_atac.sort_values(ascending=False).head(8), r_atac.sort_values(ascending=True).head(8)[::-1]], axis=0)
r_tops_index_atac = r_tops_atac.index.tolist()

_, axs = plt.subplots(ncols=len(paths), figsize=(8, 10), gridspec_kw={
    'wspace': 0.05, 'left': 0.12})
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)

for ipath, (descr, path) in enumerate(paths):
    sc.pl.paga_path(
        adata=atac_tmp, 
        nodes=path, 
        keys=r_tops_index_atac,
        annotations=['dpt_pseudotime'],
        groups_key='leiden',
        show_node_names=False,
        ax=axs[ipath],
        ytick_fontsize=12,
        left_margin=0.15,
        n_avg=50,
        show_yticks=True if ipath == 0 else False,
        show_colorbar=False,
        color_map='Greys',
        color_maps_annotations={'distance': 'viridis'},
        title='{} path'.format(descr),
        return_data=True,
        use_raw=False,
        show=False)

plt.show()

#%% draw graphs for all methods

permute_idxs = np.random.permutation(len(subsampled_eclare_adata))
colors = ['modality','Lineage', 'sub_cell_type', 'dpt_pseudotime', 'ordinal_pseudotime','Age_Range']

sc.pl.draw_graph(subsampled_eclare_adata[permute_idxs], color=colors, wspace=0.5, ncols=len(colors)); print('↑ subsampled_eclare_adata ↑')
sc.pl.draw_graph(glue_adata[permute_idxs], color=colors, wspace=0.5, ncols=len(colors)); print('↑ glue_adata ↑')
#sc.pl.draw_graph(scJoint_adata[permute_idxs], color=colors, wspace=0.5); print('↑ scJoint_adata ↑')

for source_dataset in source_datasets:
    sc.pl.draw_graph(subsampled_kd_clip_adatas[source_dataset][permute_idxs], color=colors, wspace=0.5, ncols=len(colors)); print(f'↑ subsampled_kd_clip_adatas[{source_dataset}] ↑')
    sc.pl.draw_graph(subsampled_clip_adatas[source_dataset][permute_idxs], color=colors, wspace=0.5, ncols=len(colors)); print(f'↑ subsampled_clip_adatas[{source_dataset}] ↑')

def dev_fig1(eclare_adata, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'dev_fig1.svg')):
    #eclare_adata = subsampled_eclare_adata.copy()
    permute_idxs = np.random.permutation(len(eclare_adata))
    eclare_adata.obs = eclare_adata.obs.rename(columns={
        'Lineage': 'lineage',
        'dpt_pseudotime': 'DPT pseudotime',
        'ordinal_pseudotime': 'ordinal pseudotime',
        'dev_stage': 'age range',
        'sub_cell_type': 'cell type'
        })
    eclare_adata.uns['lineage_colors'] = ['purple','orange']
    colors = ['modality','lineage', 'cell type', 'DPT pseudotime', 'ordinal pseudotime','age range']
    dev_fgi1 = sc.pl.draw_graph(eclare_adata[permute_idxs], color=colors, wspace=0.5, ncols=3, return_fig=True)

    print(f'Saving figure to {manuscript_figpath}')
    dev_fgi1.savefig(manuscript_figpath, bbox_inches='tight', dpi=300)

#%% trajectory analysis metrics

methods_list = ['ECLARE', 'scJoint', 'scGLUE']

## multi-modal
sub_eclare_metrics, sub_eclare_integration = trajectory_metrics(subsampled_eclare_adata)

scJoint_metrics, scJoint_integration = trajectory_metrics(scJoint_adata)
glue_metrics, glue_integration = trajectory_metrics(glue_adata)

sub_eclare_metrics_fig, sub_eclare_metrics_fig = trajectory_metrics_all([sub_eclare_metrics, scJoint_metrics, glue_metrics], methods_list, suptitle=None)
_, integration_fig = integration_metrics_all([sub_eclare_integration, scJoint_integration, glue_integration], methods_list, suptitle=None, drop_columns=['ct_ari', 'age_range_ari'])

## multimodal - ECLARE vs KD-CLIP vs CLIP
methods_list = ['ECLARE'] + [f'KD-CLIP_{source_dataset}' for source_dataset in source_datasets] + [f'CLIP_{source_dataset}' for source_dataset in source_datasets]  

kd_clip_metrics = {}
kd_clip_integration = {}
clip_metrics = {}
clip_integration = {}

for source_dataset in source_datasets:

    subsampled_kd_clip_metrics, subsampled_kd_clip_integration = trajectory_metrics(subsampled_kd_clip_adatas[source_dataset])
    subsampled_clip_metrics, subsampled_clip_integration = trajectory_metrics(subsampled_clip_adatas[source_dataset])

    kd_clip_metrics[source_dataset] = subsampled_kd_clip_metrics
    kd_clip_integration[source_dataset] = subsampled_kd_clip_integration
    clip_metrics[source_dataset] = subsampled_clip_metrics
    clip_integration[source_dataset] = subsampled_clip_integration

_ = trajectory_metrics_all([sub_eclare_metrics, *kd_clip_metrics.values(), *clip_metrics.values()], methods_list, suptitle=None)
_ = integration_metrics_all([sub_eclare_integration, *kd_clip_integration.values(), *clip_integration.values()], methods_list, suptitle='Multi-modal')

def dev_fig2(corrs_fig, scib_fig, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results')):
    print(f'Saving figure to {manuscript_figpath}')
    corrs_fig.savefig(os.path.join(manuscript_figpath, 'dev_fig2_corrs.svg'), bbox_inches='tight', dpi=300)
    scib_fig.savefig(os.path.join(manuscript_figpath, 'dev_fig2_scib.svg'), bbox_inches='tight', dpi=300)

'''
## RNA
sub_eclare_metrics_rna, sub_eclare_integration_rna = trajectory_metrics(subsampled_eclare_adata_rna)
scJoint_metrics_rna, scJoint_integration_rna = trajectory_metrics(scJoint_adata_rna)
glue_metrics_rna, glue_integration_rna = trajectory_metrics(glue_adata_rna)
trajectory_metrics_all([sub_eclare_metrics_rna, scJoint_metrics_rna, glue_metrics_rna], methods_list, suptitle='RNA')
integration_metrics_all([sub_eclare_integration_rna, scJoint_integration_rna, glue_integration_rna], methods_list, suptitle='RNA')

## ATAC
sub_eclare_metrics_atac, sub_eclare_integration_atac = trajectory_metrics(subsampled_eclare_adata_atac)
scJoint_metrics_atac, scJoint_integration_atac = trajectory_metrics(scJoint_adata_atac)
glue_metrics_atac, glue_integration_atac = trajectory_metrics(glue_adata_atac)
trajectory_metrics_all([sub_eclare_metrics_atac, scJoint_metrics_atac, glue_metrics_atac], methods_list, suptitle='ATAC')
integration_metrics_all([sub_eclare_integration_atac, scJoint_integration_atac, glue_integration_atac], methods_list, suptitle='ATAC')

## RNA, sampled from multimodal data
sub_eclare_metrics_rna, sub_eclare_integration_rna = trajectory_metrics(subsampled_eclare_adata, modality='RNA')
scJoint_metrics_rna, scJoint_integration_rna = trajectory_metrics(scJoint_adata, modality='RNA')
trajectory_metrics_all([sub_eclare_metrics_rna, scJoint_metrics_rna], methods_list, suptitle='RNA, sampled from multimodal data')
integration_metrics_all([sub_eclare_integration_rna, scJoint_integration_rna], methods_list, suptitle='RNA, sampled from multimodal data')

## ATAC, sampled from multimodal data
sub_eclare_metrics_atac, sub_eclare_integration_atac = trajectory_metrics(subsampled_eclare_adata, modality='ATAC')
scJoint_metrics_atac, scJoint_integration_atac = trajectory_metrics(scJoint_adata, modality='ATAC')
trajectory_metrics_all([sub_eclare_metrics_atac, scJoint_metrics_atac], methods_list, suptitle='ATAC, sampled from multimodal data')
integration_metrics_all([sub_eclare_integration_atac, scJoint_integration_atac], methods_list, suptitle='ATAC, sampled from multimodal data')
'''

#%% OT-based pairing
import ot
import ot.plot
from sklearn.metrics import pairwise_distances

def diffusion_distances(adata, t=1.0):
    X_diffmap = adata.obsm["X_diffmap"]
    evals = adata.uns["diffmap_evals"]

    evals = np.asarray(evals)
    mask = evals > 1e-12
    Xw = X_diffmap[:, mask] * (evals[mask] ** t)[None, :]
    # Euclidean on the scaled coords
    return pairwise_distances(Xw, metric="euclidean")

X_key = 'X_draw_graph_fa'
plot_key = 'X_draw_graph_fa'

X_rna = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obsm[X_key]
X_atac = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obsm[X_key]

plot_rna = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obsm[plot_key]
plot_atac = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obsm[plot_key]

a = np.ones(len(X_rna)) / len(X_rna)
b = np.ones(len(X_atac)) / len(X_atac)

if X_key == 'X_diffmap':
    D = diffusion_distances(subsampled_eclare_adata, t=2.0)
    M = D[:len(X_rna), len(X_rna):]
else:
    M = ot.dist(X_rna, X_atac, metric='sqeuclidean')
    #M_full = subsampled_eclare_adata.obsp['connectivities']
    #M = M_full[:len(X_rna), len(X_rna):].toarray()
    #M = 1 - M

reg = M[~np.isnan(M)].mean() / 100

#G = ot.sinkhorn(a, b, M, 1e-3)
#G = ot.emd(a, b, M)
G = ot.partial.partial_wasserstein(a, b, M, m=0.75)
#res = ot.solve(M, reg=None)
#G = res.plan

## get matching
atac2rna_matching = np.argmax(G, axis=0) #RNA IDs, length of X_atac (i.e. one RNA ID per ATAC cell)
atac2rna_matching_weights = np.max(G, axis=0)
assert len(atac2rna_matching) == len(X_atac), 'Number of ATAC-to-RNA matches does not match number of RNA cells'

atac_ids = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obs_names
rna_ids = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obs_names
atac2rna_ids = rna_ids[atac2rna_matching]

G_plot = np.zeros_like(G)
G_plot[atac2rna_matching, np.arange(len(atac2rna_matching))] = G[atac2rna_matching, np.arange(len(atac2rna_matching))]

# Plot for all cells
plt.figure(figsize=[6,6])
ot.plot.plot2D_samples_mat(plot_rna, plot_atac, G_plot, thr=0.02, alpha=0.05)
plt.plot(plot_rna[:,0], plot_rna[:,1], ".b", markersize=1)
plt.plot(plot_atac[:,0], plot_atac[:,1], ".r", markersize=1)
plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.axis('on')

## remove cells with no matching weight
rna_ids_dict = {rna_id: i for i, rna_id in enumerate(rna_ids)}
atac_ids_dict = {atac_id: i for i, atac_id in enumerate(atac_ids)}

remaining_rna_ids = atac2rna_ids[atac2rna_matching_weights>0]
remaining_atac_ids = atac_ids[atac2rna_matching_weights>0]

remaining_rna_idxs = [rna_ids_dict[rna_id] for rna_id in remaining_rna_ids]
remaining_atac_idxs = [atac_ids_dict[atac_id] for atac_id in remaining_atac_ids]

subsampled_eclare_adata = subsampled_eclare_adata[subsampled_eclare_adata.obs_names.isin(list(remaining_rna_ids) + list(remaining_atac_ids))].copy()
subsampled_eclare_adata.obs['Cell_ID_OT'] = subsampled_eclare_adata.obs_names.to_list()
subsampled_eclare_adata.obs.loc[subsampled_eclare_adata.obs['modality'] == 'ATAC', 'Cell_ID_OT'] = remaining_rna_ids.to_list() # transfered labels from RNA to ATAC

#G_plot_trim = np.zeros_like(G_plot)
#G_plot_trim[remaining_rna_idxs, remaining_atac_idxs] = G_plot[remaining_rna_idxs, remaining_atac_idxs]
plot_rna_trim = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obsm[plot_key]
plot_atac_trim = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obsm[plot_key]

plt.figure(figsize=[6,6])
ot.plot.plot2D_samples_mat(plot_rna, plot_atac, G_plot, thr=0.02, alpha=0.05)
plt.plot(plot_rna_trim[:,0], plot_rna_trim[:,1], ".b", markersize=1)
plt.plot(plot_atac_trim[:,0], plot_atac_trim[:,1], ".r", markersize=1)
plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.axis('on')

plt.figure(figsize=[6,6])
ot.plot.plot2D_samples_mat(plot_rna, plot_atac, G_plot, thr=0.02, alpha=0.3)
plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.axis('on')

## density plots
sc.tl.embedding_density(subsampled_eclare_adata, basis='draw_graph_fa', groupby='modality')
sc.pl.embedding_density(subsampled_eclare_adata, basis='draw_graph_fa', key='draw_graph_fa_density_modality')

sc.tl.embedding_density(subsampled_eclare_adata, basis='draw_graph_fa', groupby='Lineage')
sc.pl.embedding_density(subsampled_eclare_adata, basis='draw_graph_fa', key='draw_graph_fa_density_Lineage')

#%% save results

## add JOB ID to adatas
subsampled_eclare_adata.uns['job_id'] = methods_id_dict['eclare']
scJoint_adata.uns['job_id'] = methods_id_dict['scJoint']
glue_adata.uns['job_id'] = methods_id_dict['scGLUE']

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset].uns['job_id'] = methods_id_dict[f'kd_clip']
    subsampled_clip_adatas[source_dataset].uns['job_id'] = methods_id_dict[f'clip']

##  add metrics to respective adatas
subsampled_eclare_adata.uns['metrics'] = sub_eclare_metrics
scJoint_adata.uns['metrics'] = scJoint_metrics
glue_adata.uns['metrics'] = glue_metrics

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset].uns['metrics'] = kd_clip_metrics[source_dataset]
    subsampled_clip_adatas[source_dataset].uns['metrics'] = clip_metrics[source_dataset]

def cast_object_columns_to_str(adata):
    """
    Cast all object dtype columns in adata.obs to str type, if possible.
    """
    for col in adata.obs.select_dtypes(include=['object']).columns:
        try:
            adata.obs[col] = adata.obs[col].astype(str)
        except Exception as e:
            print(f"Could not cast column {col} to str: {e}")

## save adatas
os.makedirs(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results'), exist_ok=True)

# Cast object columns to str before saving
cast_object_columns_to_str(subsampled_eclare_adata)
subsampled_eclare_adata.write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_eclare_adata_{target_dataset}.h5ad'))

cast_object_columns_to_str(scJoint_adata)
scJoint_adata.write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'scJoint_adata_{target_dataset}.h5ad'))

cast_object_columns_to_str(glue_adata)
glue_adata.write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'glue_adata_{target_dataset}.h5ad'))

for source_dataset in source_datasets:
    cast_object_columns_to_str(subsampled_kd_clip_adatas[source_dataset])
    subsampled_kd_clip_adatas[source_dataset].write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_kd_clip_adatas_{source_dataset}_{target_dataset}.h5ad'))

    cast_object_columns_to_str(subsampled_clip_adatas[source_dataset])
    subsampled_clip_adatas[source_dataset].write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_clip_adatas_{source_dataset}_{target_dataset}.h5ad'))

#%% load adatas

subsampled_eclare_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'subsampled_eclare_adata.h5ad'))
scJoint_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'scJoint_adata.h5ad'))
glue_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'glue_adata.h5ad'))

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset] = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_kd_clip_adatas_{source_dataset}.h5ad'))
    subsampled_clip_adatas[source_dataset] = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_clip_adatas_{source_dataset}.h5ad'))

#%% create PHATE embeddings
import phate
import matplotlib.pyplot as plt

phate_op = phate.PHATE(
    k=50,                # denser graph
    n_pca=10,            # moderate denoising
    verbose=True
)

phate_atac = phate_op.fit_transform(student_atac_latents_sub.detach().cpu().numpy())

dev_labels = student_atac_sub.obs[dev_group_key]
cmap_colors = plt.get_cmap('plasma', len(dev_labels.unique()))
cmap = {dev_labels.unique()[i]: cmap_colors(i) for i in range(len(dev_labels.unique()))}
phate.plot.scatter2d(phate_op, c=dev_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

phate_rna = phate_op.fit_transform(student_rna_latents_sub.detach().cpu().numpy())

dev_labels = student_rna_sub.obs[dev_group_key]
cmap_colors = plt.get_cmap('plasma', len(dev_labels.unique()))
cmap = {dev_labels.unique()[i]: cmap_colors(i) for i in range(len(dev_labels.unique()))}
phate.plot.scatter2d(phate_op, c=dev_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)


#%% train UMAP on subsampled data
import seaborn as sns
from eclare.post_hoc_utils import plot_umap_embeddings

## define color palettes
cmap_dev = plt.get_cmap('plasma', len(dev_stages))
cmap_dev = {dev_stages[i]: cmap_dev(i) for i in range(len(dev_stages))}
cmap_ct = create_celltype_palette(student_rna_sub.obs[cell_group].values, student_atac_sub.obs[cell_group].values, plot_color_palette=False)


use_sub_data = True

if use_sub_data:
    rna_latents = student_rna_latents_sub.detach().cpu().numpy()
    atac_latents = student_atac_latents_sub.detach().cpu().numpy()
    rna_celltypes = student_rna_sub.obs[cell_group].values
    atac_celltypes = student_atac_sub.obs[cell_group].values
    rna_dev_stages = student_rna_sub.obs[dev_group_key].values
    atac_dev_stages = student_atac_sub.obs[dev_group_key].values
    rna_condition = ['nan'] * len(rna_celltypes)
    atac_condition = ['nan'] * len(atac_celltypes)

else:
    rna_latents = student_rna_latents.detach().cpu().numpy()
    atac_latents = student_atac_latents.detach().cpu().numpy()
    rna_celltypes = student_rna_keep.obs[cell_group].values
    atac_celltypes = student_atac_keep.obs[cell_group].values
    rna_dev_stages = student_rna_keep.obs[dev_group_key].values
    atac_dev_stages = student_atac_keep.obs[dev_group_key].values
    rna_condition = ['nan'] * len(rna_celltypes)
    atac_condition = ['nan'] * len(atac_celltypes)

#umap_embedder, umap_fig, rna_atac_df_umap = plot_umap_embeddings(rna_latents, atac_latents, rna_dev_stages, atac_dev_stages, rna_condition, atac_condition, cmap_dev)
umap_embedder, umap_fig, rna_atac_df_umap = plot_umap_embeddings(rna_latents, atac_latents, rna_celltypes, atac_celltypes, rna_condition, atac_condition, cmap_ct)

#%% Retrain UMAP based on original RNA UMAP coordinates
from umap import UMAP
import scanpy as sc

def retrain_umap(student_rna_sub):
    # Use the exact representation used to build neighbors
    n_pcs = student_rna_sub.obsm['X_pca'].shape[1]
    X_rep = student_rna_sub.obsm['X_pca'][:, :n_pcs]
    init_coords = student_rna_sub.obsm['X_umap']                 # from sc.tl.umap

    neigh = student_rna_sub.uns.get('neighbors', {}).get('params', {})
    umap_uns = student_rna_sub.uns.get('umap', {}).get('params', {})

    mapper = UMAP(
        n_neighbors=int(neigh.get('n_neighbors', 15)),
        metric=neigh.get('metric', 'euclidean'),
        min_dist=float(umap_uns.get('min_dist', 0.5)),
        spread=float(umap_uns.get('spread', 1.0)),
        n_components=int(umap_uns.get('n_components', 2)),
        random_state=umap_uns.get('random_state', 0),
        init=init_coords,        # “pin” the starting layout to your Scanpy embedding
    )

    mapper.fit(X_rep)            # now you have a reusable transformer
    student_rna_sub.obsm['X_umap_retrained'] = mapper.embedding_
    return mapper

    # For a new dataset:
    # 1) apply identical preprocessing (same genes/order, normalization, scaling, PCA)
    # 2) then transform:

mapper = retrain_umap(student_rna_sub)

##
sc.pp.pca(student_atac_sub, n_comps=n_pcs)
new_X_rep = student_atac_sub.obsm['X_pca'][:, :n_pcs]
student_atac_sub.obsm['X_umap_retrained'] = mapper.transform(new_X_rep)

#sc.pl.umap(student_rna_sub, color=cell_group)

## plot RNA umaps
sc.pl.embedding(student_rna_sub, basis='X_umap', color=cell_group)
sc.pl.embedding(student_rna_sub, basis='X_umap_retrained', color=cell_group)

student_rna_sub.obs[dev_group_key] = student_rna_sub.obs[dev_group_key].cat.reorder_categories(dev_stages, ordered=True)
sc.pl.embedding(student_rna_sub, basis='X_umap', color=dev_group_key, palette=cmap_dev, sort_order=True)
sc.pl.embedding(student_rna_sub, basis='X_umap_retrained', color=dev_group_key, palette=cmap_dev, sort_order=True)

## plot ATAC umaps
sc.pl.embedding(student_atac_sub, basis='X_umap_retrained', color=cell_group)
dev_stages_atac = [dev_stage for dev_stage in dev_stages if dev_stage in student_atac_sub.obs[dev_group_key].unique()]
student_atac_sub.obs[dev_group_key] = student_atac_sub.obs[dev_group_key].cat.reorder_categories(dev_stages_atac, ordered=True)
sc.pl.embedding(student_atac_sub, basis='X_umap_retrained', color=dev_group_key, palette=cmap_dev, sort_order=True)

