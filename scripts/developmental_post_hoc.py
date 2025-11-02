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
genes_by_peaks_str = '6816_by_55284'#'17279_by_66623'
source_datasets = ['PFC_Zhu', 'PFC_V1_Wang']
subsample = -1
methods_id_dict = {
    'clip': '04163347',
    'kd_clip': '07185908',
    'eclare': ['04203819'],
    'ordinal': '14112531',
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

    student_rna.obs['SubClusters'] = student_rna.obs['SubClusters'].astype(str) + '_RNA'
    student_atac.obs['SubClusters'] = student_atac.obs['SubClusters'].astype(str) + '_ATAC'
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
    target_clusters_key = 'ClustersMapped'; q=17; target_clusters = ['ExN']
    #target_clusters_key = 'SubClusters'; q=17; target_clusters = student_rna.obs.loc[student_rna.obs[target_clusters_key].str.contains('L23|L46', regex=True), target_clusters_key].unique().tolist()
    if student_rna.obs_names.isin(valid_rna_ids).any():
        rna_idxs = np.where(~student_rna.obs_names.isin(valid_rna_ids) & student_rna.obs[target_clusters_key].isin(target_clusters))[0]
        atac_idxs = np.where(~student_atac.obs_names.isin(valid_atac_ids) & student_atac.obs[target_clusters_key].isin(target_clusters))[0]
    else:
        print('WARNING: obs_names not in valid_rna_ids, resorting to reset_index().index.astype(str)')
        rna_idxs = np.where(student_rna.obs.reset_index().index.astype(str).isin(valid_rna_ids) & student_rna.obs['ClustersMapped'].isin(target_clusters))[0]
        atac_idxs = np.where(student_atac.obs.reset_index().index.astype(str).isin(valid_atac_ids) & student_atac.obs['ClustersMapped'].isin(target_clusters))[0]

if target_dataset == 'MDD':

    student_rna.obs['modality'] = 'RNA'
    student_atac.obs['modality'] = 'ATAC'
    student_rna_atac = anndata.concat([student_rna[rna_idxs], student_atac[atac_idxs]], axis=0) # no features, since features don't align
    student_rna_atac.obs['Age_bins'] = pd.qcut(student_rna_atac.obs[dev_group_key], q=q, labels=None)

    max_subsample = len(rna_idxs) + len(atac_idxs)
    if subsample > max_subsample:
        subsample = max_subsample

    student_rna_atac_sub    = subsample_adata(student_rna_atac, subsample, ['Age_bins', 'Condition', 'modality'], subsample_type='balanced')
    sns.barplot(student_rna_atac_sub.obs, x='modality', y='Age', hue='Condition', errorbar='se'); plt.ylim([30,50])
    sns.catplot(data=student_rna_atac_sub.obs, x='modality', y='Age', hue='Condition', col='SubClusters', kind='bar', height=5, aspect=.75)
    sns.catplot(data=student_rna_atac_sub.obs, x='modality', hue='Condition', col='SubClusters', kind='count')

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

    if target_dataset == 'MDD':
        print(f'Skipping other source dataset(s) given MDD target dataset')
        break

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

    #kd_teacher_rna_latents, _ = kd_clip_teacher_models[source_dataset].to('cpu')(teacher_rna, modality=0)
    #kd_teacher_atac_latents, _ = kd_clip_teacher_models[source_dataset].to('cpu')(teacher_atac, modality=1)

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

    subsampled_clip_adatas[source_dataset].obs['ordinal_pseudotime'] = ordinal_pseudotimes
    assert subsampled_clip_adatas[source_dataset].obs_names.tolist() == (student_rna_sub.obs_names.tolist() + student_atac_sub.obs_names.tolist())

    try:
        subsampled_kd_clip_adatas[source_dataset].obs['ordinal_pseudotime'] = ordinal_pseudotimes
        assert subsampled_kd_clip_adatas[source_dataset].obs_names.tolist() == (student_rna_sub.obs_names.tolist() + student_atac_sub.obs_names.tolist())
    except:
        print(f'WARNING: KD-CLIP adata not created for {source_dataset}')


#%% import source datasets

source_clusters_dict = {
    'PFC_Zhu': ['RG', 'IPC', 'EN-fetal-early', 'EN-fetal-late', 'EN'], # all: ['EN-fetal-late', 'IN-fetal', 'VSMC', 'Endothelial', 'RG', 'OPC', 'IN-CGE', 'Microglia', 'IN-MGE', 'EN-fetal-early', 'Astrocytes', 'IPC', 'Pericytes', 'EN', 'Oligodendrocytes']
    'PFC_V1_Wang': ["EN-newborn", "EN-IT-immature", "EN-L2_3-IT"],
    #'PFC_V1_Wang': ["RG-vRG", "IPC-EN", "EN-newborn", "EN-IT-immature", "EN-non-IT-immature", "EN-L2_3-IT", "EN-L4-IT-V1", "EN-L4-IT", "EN-L5-IT", "EN-L6-IT", "EN-L5_6-NP", "EN-L5-ET", "EN-L6-CT", "EN-L6b"],
}

source_adatas = {}

for source_dataset in [source_datasets[0]]:

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
    source_rna_atac_sub = subsample_adata(source_rna_atac, len(subsampled_kd_clip_adatas[source_dataset])//2, ['modality', 'type' if source_dataset == 'PFC_V1_Wang' else 'Cell type'], subsample_type='balanced')

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

    else:

        proxy = source_adatas[source_dataset].obs['dev_stage' if source_dataset == 'PFC_V1_Wang' else 'type']
        proxy = proxy.cat.codes

        source_adatas[source_dataset].obs['ordinal_pseudotime'] = proxy
        source_adatas[source_dataset + '_clip'].obs['ordinal_pseudotime'] = proxy


#for source_dataset in source_datasets:
#    print(f'KD-CLIP {source_dataset}'); source_adatas[source_dataset] = paga_analysis(source_adatas[source_dataset], dev_group_key='dev_stage', cell_group_key=cell_group)

#%% PAGA
## install scib from github, see scib documentation for details
from scib.metrics.lisi import clisi_graph, ilisi_graph
from scib.metrics import nmi, ari
from scipy.stats import pearsonr, spearmanr, kendalltau
import random

def paga_analysis(adata, dev_group_key='dev_stage', cell_group_key='Lineage', correct_imbalance=False, do_fa=True, random_seed=0):
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
        sc.pp.pca(adata, n_comps=30)
        n_pcs = np.abs(adata.uns['pca']['variance_ratio'].cumsum() - 0.66).argmin() # get number of PCs to explain 60% of variance
        sc.pp.neighbors(adata, n_neighbors=50, n_pcs=n_pcs, use_rep='X_pca', method='umap', random_state=random_seed) # use gaussian kernel for correspondance between DPT and UMAP/FA, does not work if keep all IT EN cells
        sc.tl.leiden(adata, resolution=0.6, random_state=random_seed)
        entropy_threshold = 0.8


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
        sc.pp.neighbors(adata, n_neighbors=adata.uns['neighbors']['params']['n_neighbors'], n_pcs=n_pcs, use_rep='X_pca', random_state=random_seed)
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

    if do_fa:
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
    sc.tl.diffmap(adata, n_comps=20) # X_diffmap components do not seem orthogonal to each other
    # diffmap_corrs(adata)
    # adata.obsm['X_diffmap'] = adata.obsm['X_diffmap'][:,1:7]
    # adata.uns['diffmap_evals'] = adata.uns['diffmap_evals'][1:7]
    sc.tl.dpt(adata, n_dcs=20)
    # sc.pl.umap(adata, color=['leiden', 'dpt_pseudotime', 'ordinal_pseudotime'], ncols=3, wspace=0.5)
    
    if do_fa:            
        if 'sub_cell_type' in adata.obs.columns:
            sc.pl.draw_graph(adata, color=['modality', cell_group_key, 'sub_cell_type', 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)
        else:
            sc.pl.draw_graph(adata, color=['modality', cell_group_key, 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)
    elif not do_fa:
        if 'sub_cell_type' in adata.obs.columns:
            sc.pl.umap(adata, color=['modality', cell_group_key, 'sub_cell_type', 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)
        else:
            sc.pl.umap(adata, color=['modality', cell_group_key, 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key], ncols=3, wspace=0.5)

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
        adata_splits_var = adata_splits_var[~adata_splits_var.index.duplicated(keep='first')]
        adata.obs = adata.obs.merge(adata_splits_var, left_index=True, right_index=True, how='left')

    return adata

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

def hac_weighted_mean_test(df, path, window_len, direction='all', pseudotime_key='ordinal_pseudotime', two_sided=False):

    # per-(leiden, condition) summaries
    g = (df.groupby(['leiden', 'Condition'])[pseudotime_key]
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

    # choose maxlags (e.g., window_len; or ~= n^(1/3))
    maxlags = window_len if window_len is not None else max(1, int(round(len(d) ** (1/3))))
    hac = ols.get_robustcov_results(cov_type='HAC', maxlags=maxlags, use_correction=True)

    beta = hac.params[0]
    se   = hac.bse[0]
    z    = beta / se

    # z for Leiden-level differences
    d_df['z'] = d_df['d'] / d_df['seD']

    if two_sided:
        # p for H1: mean(diff) != 0
        p = 2 * norm.sf(np.abs(z))
        d_df['p'] = 2 * norm.sf(abs(d_df['z']))
    else:
        # p for H1: mean(diff) > 0
        p = norm.sf(z)
        d_df['p'] = norm.sf(d_df['z'])


    print(f"Leiden K = {len(d)}, HAC maxlags = {maxlags}")
    print(f"Weighted mean diff = {beta:.3f} (HAC SE {se:.3f}), one-sided p = {p:.4g}")

    return d_df, p


#%% Combine source and target data

source_dataset = source_datasets[0]

if source_dataset == 'PFC_V1_Wang':
    source_adatas[source_dataset].obs.rename(columns={'type': 'ClustersMapped', 'Group': 'Age'}, inplace=True)
elif source_dataset == 'PFC_Zhu':
    source_adatas[source_dataset].obs.rename(columns={'Cell type': 'ClustersMapped', 'dev_stage': 'Age', 'Donor ID': 'Sex'}, inplace=True)

source_adatas[source_dataset].obs['SubClusters'] = source_adatas[source_dataset].obs['ClustersMapped'].copy()
source_adatas[source_dataset].obs['Age_bins'] = source_adatas[source_dataset].obs['Age'].copy()

source_adatas[source_dataset].obs['source_or_target'] = 'source'
source_adatas[source_dataset].obs['dataset_name'] = source_dataset

subsampled_kd_clip_adatas[source_dataset].obs['source_or_target'] = 'target'
subsampled_kd_clip_adatas[source_dataset].obs['dataset_name'] = target_dataset

source_adata = source_adatas[source_dataset].copy()
target_adata = subsampled_kd_clip_adatas[source_dataset].copy()
source_target_adata = anndata.concat([ source_adata, target_adata ], axis=0)

source_target_adata.obs = source_target_adata.obs.merge(target_adata.obs['Condition'], left_index=True, right_index=True, how='left')
source_target_adata.obs['Age'] = pd.Categorical(source_target_adata.obs['Age'])

#source_target_adata = source_target_adata[source_target_adata.obs['ClustersMapped'].isin(['ExN', 'EN-fetal-early', 'EN-fetal-late', 'EN'])]
#source_target_adata = source_target_adata[source_target_adata.obs['ClustersMapped'].isin(['ExN', 'RG-vRG', 'IPC-EN', 'EN-newborn', 'EN-IT-immature', 'EN-L2_3-IT'])]

if source_dataset == 'PFC_V1_Wang':
    source_target_adata = source_target_adata[source_target_adata.obs['SubClusters'].isin(
        ['EN-newborn', 'EN-IT-immature','EN-L2_3-IT'] + \
            ['ExN1_L23_ATAC', 'ExN2_L23_ATAC'] + \
            ['ExN9_L23_RNA', 'ExN2_L23_RNA']
            )]
elif source_dataset == 'PFC_Zhu':
    print('WARNING: not filtering SubClusters for PFC_Zhu')

    #subclusters_props = target_adata.obs['SubClusters'].value_counts(normalize=True)
    #subclusters_filtered = subclusters_props.loc[subclusters_props > 0.01].index.tolist() + source_clusters
    #source_target_adata = source_target_adata[source_target_adata.obs['SubClusters'].isin(subclusters_filtered)]


def cast_object_columns_to_str(adata):
    """
    Cast all object dtype columns in adata.obs to str type, if possible.
    """
    for col in adata.obs.select_dtypes(include=['object','category']).columns:
        try:
            adata.obs[col] = adata.obs[col].astype(str)
        except Exception as e:
            print(f"Could not cast column {col} to str: {e}")

#%% Plot distribution of age and ordinal pseudotime

adata = source_target_adata.copy()
adata.obs_names_make_unique()
adata.obs[['Age', 'ordinal_pseudotime']] = adata.obs[['Age', 'ordinal_pseudotime']].astype(float)
#adata.obs['SubClusters'] = pd.Categorical(adata.obs['SubClusters'], categories=source_clusters + target_clusters, ordered=True)

fig, ax = plt.subplots(1, 2, figsize=[12, 8])
sns.barplot(adata[adata.obs['source_or_target']=='source'].obs, x='modality', y='Age', hue='ClustersMapped', errorbar='se', ax=ax[0])
sns.barplot(adata[adata.obs['source_or_target']=='source'].obs, x='modality', y='ordinal_pseudotime', hue='ClustersMapped', errorbar='se', ax=ax[1])
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=[6, 4])
sns.barplot(adata[adata.obs['source_or_target']=='target'].obs, x='modality', y='Age', hue='Condition', errorbar='se', ax=ax[0])
sns.barplot(adata[adata.obs['source_or_target']=='target'].obs, x='modality', y='ordinal_pseudotime', hue='Condition', errorbar='se', ax=ax[1])
plt.tight_layout(); plt.show()

#%% Run PAGA on combined source & target data

do_fa = False
correct_imbalance = False

source_target_adata = paga_analysis(source_target_adata, dev_group_key='Age', cell_group_key='ClustersMapped', do_fa=do_fa,
    correct_imbalance=correct_imbalance)
#source_target_adata.write(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'source_target_adata_{source_dataset}_{target_dataset}.h5ad'))

## plot FA embeddings comparing source and target datasets
source_target_adata = magic_diffusion(source_target_adata, 'ordinal_pseudotime', split_var='Condition', t=3)
target_adata = source_target_adata[source_target_adata.obs['source_or_target']=='target'].copy()
source_adata = source_target_adata[source_target_adata.obs['source_or_target']=='source'].copy()

if do_fa:
    plot_func = sc.pl.draw_graph
    basis = 'draw_graph_fa'
elif not do_fa:
    plot_func = sc.pl.umap
    basis = 'umap'

colors = ['source_or_target', 'modality', 'dpt_pseudotime', 'ordinal_pseudotime', 'ClustersMapped']
f1 = plot_func(source_target_adata, color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f2 = plot_func(source_target_adata[source_target_adata.obs['source_or_target']=='target'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f3 = plot_func(source_target_adata[source_target_adata.obs['source_or_target']=='source'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f1.show(); f2.show(); f3.show()

colors = ['Condition', 'modality', 'ordinal_pseudotime_magic']
f1 = plot_func(target_adata, color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f2 = plot_func(target_adata[target_adata.obs['Condition']=='case'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f3 = plot_func(target_adata[target_adata.obs['Condition']=='control'], color=colors, ncols=len(colors), wspace=0.5, return_fig=True)
f1.show(); f2.show(); f3.show()


#%% correlate diffusion components with ordinal pseudotime
'''

def diffmap_corrs(adata, pseudotime_key='ordinal_pseudotime'):
    n_diffmaps = len(adata.uns['diffmap_evals'])
    diffmap_df = pd.DataFrame(adata.obsm['X_diffmap'], columns=[f'dm{idx}' for idx in range(n_diffmaps)], index=adata.obs_names)
    diffmap_corrs_df = pd.concat([adata.obs[pseudotime_key], diffmap_df], axis=1).corr()
    diffmap_corrs_series = diffmap_corrs_df.loc[pseudotime_key,:].iloc[1:]
    return diffmap_corrs_series

pseudotime_key = 'dpt_pseudotime'

diffmap_corrs_df = pd.concat([
    diffmap_corrs(source_target_adata, pseudotime_key=pseudotime_key).to_frame().assign(source_or_target='all'),
    diffmap_corrs(source_target_adata[source_target_adata.obs['source_or_target']=='target'], pseudotime_key=pseudotime_key).to_frame().assign(source_or_target='target'),
    diffmap_corrs(source_target_adata[source_target_adata.obs['source_or_target']=='source'], pseudotime_key=pseudotime_key).to_frame().assign(source_or_target='source')
    ], axis=0).reset_index()

fig, ax = plt.subplots(1, 1, figsize=[12, 8])
sns.barplot(diffmap_corrs_df, x='index', y=pseudotime_key, hue='source_or_target', ax=ax)
plt.xticks(rotation=30)
plt.xlabel('')
plt.ylabel('Pearson correlation')
plt.title(f'Diffusion components vs. {pseudotime_key}')
plt.tight_layout(); plt.show()

'''

from sklearn.cross_decomposition import PLSRegression, PLSSVD, CCA

def pls_analysis(adata):

    x = adata.obsm['X_diffmap'][:,1:]

    ## PLS for ordinal pseudotime
    pseudotime_key = 'ordinal_pseudotime'
    pls_ordinal = PLSRegression(n_components=1)
    y_ordinal = adata.obs[pseudotime_key]
    pls_ordinal.fit(x, y_ordinal)
    adata.obs[f'{pseudotime_key}_pls'] = pls_ordinal.predict(x).squeeze()

    ## PLS for dpt pseudotime
    pseudotime_key = 'dpt_pseudotime'
    pls_dpt = PLSRegression(n_components=1)
    y_dpt = adata.obs[pseudotime_key]
    pls_dpt.fit(x, y_dpt)
    adata.obs[f'{pseudotime_key}_pls'] = pls_dpt.predict(x).squeeze()

    ## combine PLS coefficients and intercepts
    pls_coeffs_df = pd.DataFrame({
        'ordinal': pls_ordinal.coef_.squeeze(),
        'dpt': pls_dpt.coef_.squeeze()
    })

    pls_intercepts_df = pd.Series({
        'ordinal': pls_ordinal.intercept_.item(),
        'dpt': pls_dpt.intercept_.item()
    }, name='intercept')

    pls_coeffs_df.plot(kind='bar', subplots=True, layout=(1, 2), legend=False, figsize=[10, 5])
    plt.tight_layout(); plt.show()

    ## extract positive and negative coefficients
    poscoefs = pls_coeffs_df['ordinal'].copy()
    pos_coefs_idx = (pls_coeffs_df > 0).all(axis=1)
    poscoefs[~pos_coefs_idx] = 0

    negcoefs = pls_coeffs_df['ordinal'].copy()
    neg_coefs_idx = (pls_coeffs_df < 0).all(axis=1)
    negcoefs[~neg_coefs_idx] = 0

    coefs_delta = negcoefs - poscoefs

    ## apply PLS coefficients to target data
    adata.obs['ordinal_pos'] = np.matmul(x, poscoefs)# + pls_intercepts_df['ordinal']
    adata.obs['ordinal_delta'] = np.matmul(x, coefs_delta)

    adata.obs['delta_bins'] = pd.qcut(adata.obs['ordinal_delta'], q=4)
    sns.histplot(adata.obs, x='ordinal_delta', hue='delta_bins', bins=50, legend=False)

    last_bin = adata.obs['delta_bins'].cat.categories[-1]
    adata_clean = adata[adata.obs['ordinal_delta'] <= 0].copy()

    sc.pl.umap(adata, color=['ordinal_pos', 'ordinal_delta', 'ordinal_pseudotime_pls', 'ordinal_pseudotime'])
    sc.pl.umap(adata_clean, color=['ordinal_pos', 'ordinal_delta', 'ordinal_pseudotime_pls', 'ordinal_pseudotime'])

def pls_analysis_2(adata):

    x = adata.obsm['X_diffmap'][:,1:]
    pls_ordinal = PLSRegression(n_components=2)

    y_ordinal = adata.obs['ordinal_pseudotime']
    y_dpt = adata.obs['dpt_pseudotime']
    y_both = np.column_stack([y_ordinal, y_dpt])

    pls_ordinal.fit(x, y_both)

    T = pls_ordinal.transform(x)                 # (n_samples, n_components)
    Q = pls_ordinal.y_loadings_                  # (n_targets, n_components)

    Y_pred_components = np.einsum("ij,kj->ijk", T, Q)

    all_cells_hits = []
    for comp in range(Y_pred_components.shape[1]):

        ordinal_pred = Y_pred_components[:,comp,0]
        corr, pval = pearsonr(ordinal_pred, y_dpt) # correlation between ordinal prediction and actual dpt pseudotime
        
        print(f'Component {comp+1}: Pearson r = {corr:.4f}, p = {pval:.4e}')

        if (r > 0) and (pval < 1e-5):
            hits = pls_ordinal.x_scores_[:,comp] > 0
            cells_hits = adata.obs_names[hits]
            all_cells_hits.extend(cells_hits)

    all_cells_hits = list(set(all_cells_hits))
    adata_hits = adata[adata.obs_names.isin(all_cells_hits)].copy()
    sc.pl.umap(adata_hits, color=['ordinal_pseudotime', 'dpt_pseudotime'])

    sc.tl.paga(adata_hits)
    sc.pl.paga_compare(adata_hits)
    sc.pl.paga_compare(adata_hits, color=['ordinal_pseudotime'])

def pls_analysis_3(adata):

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=10, n_neighbors=50)

    x = adata.obsm['X_pca']
    y = adata.obs['dpt_pseudotime']

    pls = PLSRegression(n_components=30)
    pls.fit(x,y_both)

    adata.obsm['X_dpt_pseudotime_pls'] = pls.x_scores_
    sc.pp.neighbors(adata, use_rep='X_dpt_pseudotime_pls')

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['dpt_pseudotime', 'ordinal_pseudotime'])
    sc.pl.umap(adata[adata.obs['source_or_target']=='target'], color=['dpt_pseudotime', 'dpt_pseudotime_pls', 'ordinal_pseudotime'])

def pls_analysis_3(target_adata):

    x = adata.obsm['X_pca']
    pls = PLSRegression(n_components=5)

    y_ordinal = adata.obs['ordinal_pseudotime']
    y_diffmaps = adata.obsm['X_diffmap'][:,1:5]
    y_both = np.column_stack([y_ordinal, y_diffmaps])

    pls.fit(x, y_both)
    fig, ax = plt.subplots(2, 1, figsize=[10, 10])
    pd.DataFrame(pls.x_weights_.T).plot(kind='bar', legend=False, ax=ax[0])
    pd.DataFrame(pls.y_weights_.T).plot(kind='bar', legend=False, ax=ax[1])
    plt.tight_layout(); plt.show()

    # Find PLS component where weight for ordinal_pseudotime is most similar to that for first diffmap component
    ordinal_idx = 0  # index of ordinal_pseudotime in y_both
    diffmap1_idx = 1  # index of first diffmap component in y_both

    # Find indices of PLS components where both ordinal_pseudotime and first diffmap component have the same sign
    ordinal_weights = pls.y_weights_[ordinal_idx]
    diffmap1_weights = pls.y_weights_[diffmap1_idx]
    same_sign_components = np.where(np.sign(ordinal_weights) == np.sign(diffmap1_weights))[0]
    print(f"PLS components where weights for ordinal_pseudotime and first diffmap have same sign: {same_sign_components}")

    X_new_pca = pls.x_scores_[:, same_sign_components]
    adata.obsm['X_dpt_pseudotime_pls'] = X_new_pca

    sc.pp.neighbors(adata, use_rep='X_dpt_pseudotime_pls', n_pcs=len(same_sign_components), n_neighbors=50)
    sc.tl.leiden(adata, resolution=0.6, random_state=0, neighbors_key='X_dpt_pseudotime_pls_neighbors')

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['dpt_pseudotime', 'ordinal_pseudotime', 'leiden'])


def diffmap_match_geom(adata):

    if 'X_pca' not in adata.obsm.keys():
        sc.pp.pca(adata)
    if 'neighbors' not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=10, n_neighbors=200)
    if 'X_umap' not in adata.obsm.keys():
        sc.tl.umap(adata)
    if 'leiden' not in adata.obs.columns:
        sc.tl.leiden(adata, resolution=0.6)
    if 'paga' not in adata.uns.keys():
        sc.tl.paga(adata, groups='leiden')
    if 'X_diffmap' not in adata.obsm.keys():
        sc.tl.diffmap(adata)

    # DC1, DC2 (skip DC0)
    dm = adata.obsm['X_diffmap'][:, 1:3]

    # row-normalize connectivities (remove self loops if any)
    W = adata.obsp[adata.uns['neighbors']['connectivities_key']].tocsr().copy()
    W.setdiag(0); W.eliminate_zeros()
    rsum = np.asarray(W.sum(1)).ravel(); rsum[rsum==0] = 1.0
    Wn = W.multiply(1.0/rsum[:,None])

    # local variance per DC: Var(X) = E[X^2] - E[X]^2, with E computed via Wn
    Ex   = Wn @ dm                     # (n×2)
    Ex2  = Wn @ (dm**2)                # (n×2)
    Var  = Ex2 - Ex**2                 # (n×2)

    share1 = Var[:,0] / (Var.sum(1) + 1e-12)
    label = np.where(share1 > 0.55, "diffmap_1",
            np.where(share1 < 0.45, "diffmap_2", "ambiguous"))

    adata.obs["dm_match_geom"]  = pd.Categorical(label)
    adata.obs["dm_share_dm1"]   = share1
    
    diffmap_adata = anndata.AnnData(adata.obsm['X_diffmap'], obsm={'X_umap': adata.obsm['X_umap']}, obs=adata.obs.copy())
    diffmap_adata.var_names = pd.Index([f'dm{i}' for i in range(adata.obsm['X_diffmap'].shape[1])])
    sc.pl.umap(diffmap_adata, color=['dm1', 'dm2', 'ordinal_pseudotime'], wspace=0.4)

    sc.pl.umap(adata, color='dm_share_dm1')
    sc.pl.paga_compare(adata, show=False); plt.close()
    sc.pl.paga_compare(adata, color='dm_match_geom', legend_loc='right margin', right_margin=0.7, node_size_scale=5)

    adata_dm1 = adata[adata.obs['dm_match_geom'].isin(['diffmap_1', 'ambiguous'])].copy()
    sc.tl.paga(adata_dm1, groups='leiden')
    sc.pl.paga_compare(adata_dm1, show=False); plt.close()
    sc.pl.paga_compare(adata_dm1, color='ordinal_pseudotime', legend_loc='right margin', right_margin=0.7, node_size_scale=5)

    return adata_dm1

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

df['most_common_cluster'] = pd.Categorical(df['most_common_cluster'], categories=source_clusters, ordered=True)

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
        fig, ax = plt.subplots(1, 1, figsize=[6, 5])

        # Barplot
        pastel2_colors = sns.color_palette('Pastel2')
        # Flip the order of the first two colors
        pastel2_colors = [pastel2_colors[1], pastel2_colors[0]] + list(pastel2_colors[2:])
        sns.barplot(
            df,
            y='most_common_cluster',
            x='mean_neighbors_ordinal_pseudotime',
            hue='Condition',
            errorbar='se',
            palette='Pastel1',
            linewidth=2,
            edgecolor='.5',
            ax=ax)

        #ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_xlabel('Ordinal Pseudotime')
        ax.set_ylabel('')
        ax.set_xlim(30, 62)

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
    print(f'Saving figure to {manuscript_figpath}')
    plt.close()

def devmdd_fig2(target_adata, source_adata, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results')):
    #pos = adata.uns['paga']['pos']

    sc.settings._vector_friendly = True
    sc.settings.figdir = manuscript_figpath

    ## target data
    adata = target_adata.copy()
    sc.pl.umap(adata, color='most_common_cluster', title='MDD data (imputed)', save='_devmdd_fig3.svg')

    perm_idxs = np.flip(np.random.permutation(len(adata)))
    size_0_for_na = np.where(adata.obs['most_common_cluster'].isna(), 0, 50)

    sc.tl.paga(adata, groups='leiden')
    sc.pl.paga_compare(adata[perm_idxs])
    sc.pl.paga_compare(adata[perm_idxs], color='most_common_cluster', legend_loc='right margin', right_margin=0.9, title='', size=size_0_for_na[perm_idxs], fontsize=0, node_size_scale=3, frameon=True, save='_devmdd_fig2.svg') # will prepend "paga_compare" to the filename

    #sc.pl.paga_compare(adata[perm_idxs], color='ordinal_pseudotime_magic', size=size_0_for_na[perm_idxs], right_margin=0.4)

    if 'draw_graph' in adata.uns.keys():
        sc.pl.draw_graph(adata[perm_idxs], color='leiden', size=size_0_for_na[perm_idxs], na_in_legend=False, cmap='plasma', title='', save='_devmdd_fig5.svg')
    else:
        sc.pl.umap(adata[perm_idxs], color='leiden', size=size_0_for_na[perm_idxs], na_in_legend=False, cmap='plasma', legend_loc='on data', legend_fontsize=8, title='', save='_devmdd_fig5.svg', groups=None)

    #sc.pl.draw_graph(adata[perm_idxs], color='most_common_cluster', size=size_0_for_na[perm_idxs], na_in_legend=False, title='')
    #sc.pl.paga(adata)

    ## source data
    adata = source_adata.copy()
    adata.obs['ClustersMapped'] = pd.Categorical(adata.obs['ClustersMapped'], categories=target_adata.obs['most_common_cluster'].cat.categories, ordered=True)
    sc.pl.umap(adata, color='ClustersMapped', title='developmental data', legend_loc=None, save='_devmdd_fig4.svg')
    print(f'Saving figure to {manuscript_figpath}')



#%% UMAP of leiden clusters to be able to set paths

# For each SubCluster, compute the correlation matrix between 'dpt_pseudotime' and 'ordinal_pseudotime',
corrs = target_adata.obs.groupby('SubClusters')[['dpt_pseudotime', 'ordinal_pseudotime']].corr(lambda x, y: spearmanr(x, y)[0]).unstack().loc[:, ('dpt_pseudotime', 'ordinal_pseudotime')]


'''
## confirm correspondence between Age bins and ordinal pseudotime for MDD data
plt.figure(figsize=[5, 5])
target_adata.obs['Age_bins_small'] = pd.qcut(target_adata.obs['Age'], q=6, labels=None)
target_adata.obs['Age_bins_small'] = pd.Categorical(target_adata.obs['Age_bins_small'].astype(str), categories=target_adata.obs['Age_bins_small'].cat.categories.astype(str), ordered=True)
sns.lineplot(target_adata.obs, x='Age_bins_small', y='ordinal_pseudotime', hue='Condition', marker='o')
plt.xticks(rotation=30)
plt.gca().set_xticklabels([])
'''

if 'Sex' not in target_adata.obs.columns:
    target_adata.obs = target_adata.obs.merge(student_rna_atac_sub.obs['Sex'], left_index=True, right_index=True, how='left')

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

#sc.pl.embedding(target_adata, color='leiden', basis=basis, na_in_legend=False, palette='plasma')
plot_func(target_adata, color='leiden', legend_loc='on data', alpha=0.2)
#sc.pl.paga_compare(target_adata, color="ordinal_pseudotime", node_size_scale=3, legend_loc='right margin', right_margin=0.5)

'''
paths = [
    ("MDD_ExN_all", leiden_sorted),
    ('MDD_ExN_5_1_6_4_7_0', ['5', '1', '6', '4', '7', '0']),
    ('MDD_ExN_5_3_11_10_9_8', ['5', '3', '11', '10', '9', '8']),
]
'''
#%% set paths

'''
paths = [
    ("MDD_ExN_all", leiden_sorted),
    ('MDD_ExN_5_1_6_4_7_0', ['5', '1', '6', '4', '7', '0']),
    ('MDD_ExN_5_3_11_10_9_8', ['5', '3', '11', '10', '9', '8']),
]
'''
paths = [
    ("MDD_ExN_all", leiden_sorted),
    ('MDD_ExN_0_10_3_4_1_7_6_9', ['0', '10', '3', '4', '1', '7', '6', '9']),
    ('MDD_ExN_0_2_12_11_13_8_14', ['0', '2', '12', '11', '13', '8', '14']),
]


for path in paths[1:]:
    last_leiden = path[-1][-1]
    last_leiden_top5 = target_adata.obs.loc[target_adata.obs['leiden'] == last_leiden, 'SubClusters'].value_counts().head(5).index.tolist()

    print(path[0])
    print(last_leiden_top5)
    print('--------------------------------')

#%% Rename paths

## create name mappers for paths and leiden clusters
path_names_mapper = {
    paths[0][0]: 'all',
    paths[1][0]: 'L46',
    paths[2][0]: 'L23',
}
print(path_names_mapper)

path_leidens_mapper = {}

for descr, path in paths[1:]:
    path_leidens = {}
    # Only append path_names_mapper[descr] if leiden is unique within the path
    seen_leidens = set()
    for l, leiden in enumerate(path):
        # Check if leiden cluster is unique among all defined paths (not just this path)
        leiden_is_unique = sum([p.count(leiden) for _, p in paths[1:]]) == 1
        if leiden_is_unique:
            path_leidens[leiden] = f'{l}_{path_names_mapper[descr]}'
        else:
            path_leidens[leiden] = str(l)
    path_leidens_mapper.update(path_leidens)

## add missing leiden clusters
for leiden in target_adata.obs['leiden'].unique().tolist():
    if leiden not in path_leidens_mapper.keys():
        print(f'{leiden} not in paths')
        path_leidens_mapper[leiden] = leiden + '_na'

## apply name mappers to paths
paths = [(path_names_mapper[descr], [path_leidens_mapper[leiden] for leiden in path]) for descr, path in paths]

## apply name mappers to target_adata
target_adata.obs['leiden_og'] = target_adata.obs['leiden']
target_adata.obs['leiden'] = target_adata.obs['leiden'].map(path_leidens_mapper)

#sc.tl.umap(target_adata)
#sc.pl.umap(target_adata, color='leiden', legend_loc='on data', alpha=0.2)
#sc.tl.paga(target_adata, groups="leiden_renamed")

sc.tl.paga(target_adata, groups="leiden")
sc.pl.paga_compare(target_adata)
sc.pl.paga_compare(target_adata, color="leiden", node_size_scale=5, size=25, legend_loc='on data', legend_fontsize=8, fontsize=8, right_margin=0.1)
sc.pl.paga_compare(target_adata, color="ordinal_pseudotime", node_size_scale=5, size=25, legend_loc='on data', legend_fontsize=8, fontsize=8, right_margin=0.1)

#devmdd_fig2(target_adata, source_adata)

#%% load full datasets
from eclare.data_utils import gene_activity_score_glue

## MDD data
rna_filename = f"rna_16448_by_169411_aligned_source_PFC_Zhu.h5ad"
atac_filename = f"atac_16448_by_169411_aligned_source_PFC_Zhu.h5ad"
mdd_rna_full = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', rna_filename), backed='r')
mdd_atac_full = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', atac_filename), backed='r')
mdd_rna_full_sub = mdd_rna_full[mdd_rna_full.obs_names.isin(student_rna_sub.obs_names)].to_memory()
mdd_atac_full_sub = mdd_atac_full[mdd_atac_full.obs_names.isin(student_atac_sub.obs_names)].to_memory()

## PFC Zhu data
rna_filename = "rna_16448_by_169411_aligned_target_MDD.h5ad" #f"rna_9832_by_70751.h5ad"
atac_filename = "atac_16448_by_169411_aligned_target_MDD.h5ad" #f"atac_9832_by_70751.h5ad"
pfc_zhu_rna_full = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'rna', rna_filename), backed='r')
pfc_zhu_atac_full = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'atac', atac_filename), backed='r')
pfc_zhu_rna_full_sub = pfc_zhu_rna_full[pfc_zhu_rna_full.obs_names.isin(source_rna_sub.obs_names)].to_memory()
pfc_zhu_atac_full_sub = pfc_zhu_atac_full[pfc_zhu_atac_full.obs_names.isin(source_atac_sub.obs_names)].to_memory()

## add modality column to obs
mdd_rna_full_sub.obs['modality'] = 'RNA'
mdd_atac_full_sub.obs['modality'] = 'ATAC'
pfc_zhu_rna_full_sub.obs['modality'] = 'RNA'
pfc_zhu_atac_full_sub.obs['modality'] = 'ATAC'


#%% save data in preparation for pyDESeq2

## add ordinal latents and pseudotime to PFC Zhu data
pfc_zhu_rna_EN = pfc_zhu_rna_full[pfc_zhu_rna_full.obs['Cell type'].isin(source_clusters)].to_memory()
ordinal_rna_logits, ordinal_rna_probas, ordinal_rna_latents = ordinal_model(torch.tensor(source_rna.X.toarray(), dtype=torch.float32), modality=0, normalize=0)
ordinal_rna_prebias = ordinal_model.ordinal_layer_rna.coral_weights(ordinal_rna_latents)
ordinal_rna_pt = torch.sigmoid(ordinal_rna_prebias).flatten().detach().cpu().numpy()
pfc_zhu_rna_EN.obs['ordinal_pseudotime'] = ordinal_rna_pt
pfc_zhu_rna_EN.obsm['X_ordinal_latents'] = ordinal_rna_latents.detach().cpu().numpy()
pfc_zhu_rna_EN.write_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'rna', 'pfc_zhu_rna_EN_ordinal.h5ad'))

# Load data
mdd_rna_scaled = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_rna_scaled.h5ad'), backed='r')
mdd_rna_scaled_sub = mdd_rna_scaled[mdd_rna_scaled.obs_names.isin(student_rna_sub.obs_names)].to_memory()

mdd_rna_scaled_sub.obs = mdd_rna_scaled_sub.obs.merge(source_target_adata.obs, left_index=True, right_index=True, suffixes=('', '_right'))
mdd_rna_scaled_sub.obs = mdd_rna_scaled_sub.obs.loc[:, ~mdd_rna_scaled_sub.obs.columns.str.endswith('_right')]

## add ordinal latents to mdd_rna_scaled_sub
mdd_rna_scaled_sub.obsm['X_ordinal_latents'] = ordinal_rna_latents.detach().cpu().numpy()
sc.pp.neighbors(mdd_rna_scaled_sub, use_rep='X_ordinal_latents', key_added='X_ordinal_latents_neighbors', n_neighbors=30, n_pcs=10)

## find genes in full data
mdd_rna_scaled_sub.raw.var.set_index('_index', inplace=True)
genes_in_full_data_bool = mdd_rna_scaled_sub.raw.var.index.isin(mdd_rna_full.var_names)

## filter raw data to only include genes in full data
filtered_raw_X = mdd_rna_scaled_sub.raw.X[:, genes_in_full_data_bool]
filtered_raw_var = mdd_rna_scaled_sub.raw.var.loc[genes_in_full_data_bool].copy()
raw_adata = anndata.AnnData(X=filtered_raw_X, var=filtered_raw_var)

mdd_rna_scaled_sub.raw = raw_adata

## save mdd_rna_scaled_sub to mdd_data directory
cast_object_columns_to_str(mdd_rna_scaled_sub)
mdd_rna_scaled_sub.write_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_rna_scaled_sub.h5ad'))

## ATAC
def rename_obs_columns(adata):
    mapping = {
        'sex': 'Sex',
        'Subject': 'OriginalSub',
        'condition': 'Condition',
        'batch': 'Batch',
    }
    mapping = {key: val for key, val in mapping.items() if val not in adata.obs.columns}
    adata.obs.rename(columns=mapping, inplace=True)
    return adata

## add gene activity score to ATAC
mdd_atac_broad = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_broad.h5ad'), backed='r')
mdd_atac_broad = rename_obs_columns(mdd_atac_broad)
mdd_atac_broad.var_names = mdd_atac_broad.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values

mdd_atac_broad_sub = mdd_atac_broad[
    mdd_atac_broad.obs_names.isin(student_atac_sub.obs_names),
    mdd_atac_broad.var_names.isin(mdd_atac_full.var_names)
].to_memory()

mdd_atac_broad_sub.obs = mdd_atac_broad_sub.obs.merge(source_target_adata.obs, left_index=True, right_index=True, suffixes=('', '_right'))
mdd_atac_broad_sub.obs = mdd_atac_broad_sub.obs.loc[:, ~mdd_atac_broad_sub.obs.columns.str.endswith('_right')]

## compute gene activity score on processed data
mdd_atac_broad_sub.var['interval'] = mdd_atac_broad_sub.var_names
mdd_atac_gas_broad_sub, _ = gene_activity_score_glue(mdd_atac_broad_sub, mdd_rna_full_sub)

## create raw data object
mdd_atac_broad_raw = anndata.AnnData(
    X=mdd_atac_broad.raw.X,
    var=mdd_atac_broad.raw.var,
    obs=mdd_atac_broad.obs
)
mdd_atac_broad_raw.var_names = mdd_atac_broad_raw.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values

## subset to student data
mdd_atac_broad_raw_sub = mdd_atac_broad_raw[
    mdd_atac_broad_raw.obs_names.isin(student_atac_sub.obs_names),
    mdd_atac_broad_raw.var_names.isin(mdd_atac_full.var_names)
].to_memory()

## compute gene activity score on raw data
mdd_atac_broad_raw_sub.var['interval'] = mdd_atac_broad_raw_sub.var_names
mdd_atac_gas_broad_raw_sub, _ = gene_activity_score_glue(mdd_atac_broad_raw_sub, mdd_rna_full_sub)

## check for gene activity score with zero expression in all cells
mdd_atac_gas_broad_sub = mdd_atac_gas_broad_sub[:, mdd_atac_gas_broad_sub.X.toarray().sum(axis=0) > 0].copy()
mdd_atac_gas_broad_raw_sub = mdd_atac_gas_broad_raw_sub[:, mdd_atac_gas_broad_raw_sub.X.toarray().sum(axis=0) > 0].copy()

## rename obs columns
mdd_atac_gas_broad_sub = rename_obs_columns(mdd_atac_gas_broad_sub)
mdd_atac_gas_broad_raw_sub = rename_obs_columns(mdd_atac_gas_broad_raw_sub)

## set raw data and save
mdd_atac_gas_broad_sub.raw = mdd_atac_gas_broad_raw_sub
cast_object_columns_to_str(mdd_atac_gas_broad_sub)
mdd_atac_gas_broad_sub.write_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_gas_broad_sub.h5ad'))

## filterered mean GRN from brainSCOPE
from eclare.post_hoc_utils import tree
with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1] # female mean GRN

from eclare.data_utils import filter_mean_grn
mean_grn_df_filtered, mdd_rna_scaled_sub_filtered, mdd_atac_broad_sub_filtered = filter_mean_grn(mean_grn_df, mdd_rna_scaled_sub, mdd_atac_broad_sub, deg_genes=None)
peak_names_mapper = mdd_atac_broad_sub_filtered.var['GRN_peak_interval'].to_dict()

## save to OUTPATH
mean_grn_df_filtered.to_csv(os.path.join(os.environ['OUTPATH'], 'mean_grn_df_filtered.csv'), index=False)

with open(os.path.join(os.environ['OUTPATH'], 'peak_names_mapper.pkl'), 'wb') as file:
    pickle.dump(peak_names_mapper, file)


#%% import information from Zhu et al. Supplementary Tables S12 and S7
zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
gwas_hits = pd.read_excel(zhu_supp_tables, sheet_name='Table S12', header=2)
peak_gene_links = pd.read_excel(zhu_supp_tables, sheet_name='Table S7', header=2)

## filter for MDD hits and get gene-peak links
mdd_hits = gwas_hits[gwas_hits['Trait'].eq('MDD')]
mdd_hits_gene_peaks = peak_gene_links[peak_gene_links['gene'].isin(mdd_hits['Target gene name'].unique())]
mdd_hits = mdd_hits.merge(mdd_hits_gene_peaks, left_on='Target gene name', right_on='gene', how='left')
mdd_hits.rename(columns={'gene': 'gene_linked', 'peak': 'peak_linked', 'Peak coordinates (hg38)': 'peak_ld_buddy'}, inplace=True)

## group by gene and get list of peaks
genes_peaks_ld_buddy_dict   = mdd_hits.dropna(subset='peak_ld_buddy').groupby('Target gene name')['peak_ld_buddy'].apply(list).to_dict()
genes_peaks_linked_dict     = mdd_hits.dropna(subset='peak_linked').groupby('Target gene name')['peak_linked'].apply(list).to_dict()
print('Number of genes with LD buddies: ', len(genes_peaks_ld_buddy_dict))
print('Number of genes with linked peaks: ', len(genes_peaks_linked_dict))

genes_km_clusters_dict = gwas_hits.dropna(subset='km').groupby('Target gene name')['km'].unique().to_dict()

#%% get gene expression gene expression of select genes

# Find overlap between student_atac_sub.var_names and peaks_list using BedTool
from pybedtools import BedTool
import re
# Convert peak strings to BedTool format
def peaks_to_bedtool(peaks):
    # Parse: "chr9:123999023-124000462" -> chr9 123999023 124000462
    bed_lines = []
    for p in peaks:
        chrom, start, end = re.split("[:|-]", p)
        start_pos = int(start)
        end_pos = int(end)
        
        # Skip invalid coordinates where start >= end
        if start_pos >= end_pos:
            print(f"Warning: Skipping invalid peak coordinates {p} (start >= end)")
            continue
            
        bed_lines.append(f"{chrom}\t{start}\t{end}")
    return BedTool("\n".join(bed_lines), from_string=True)


def get_gene_expression_and_chromatin_accessibility(student_rna_sub, source_rna_sub, student_atac_sub, source_atac_sub, genes_peaks_dict):

    ## chromatin accessibility
    hit = list(genes_peaks_dict.keys())
    peaks_list = list(genes_peaks_dict.values())[0]
    peaks_bedtool_list = peaks_to_bedtool(peaks_list)#.slop(b=250, genome='hg38')
    
    ## gene expression
    if student_rna_sub.var_names.isin(hit).any():
        target_gene_expr = student_rna_sub[:,student_rna_sub.var_names.isin(hit)].X.toarray().squeeze()
        source_gene_expr = source_rna_sub[:,source_rna_sub.var_names.isin(hit)].X.toarray().squeeze()
    else:
        print(' - Target gene not found in student RNA data')
        return None

    student_atac_var_names = list(student_atac_sub.var_names)
    student_atac_bedtool = peaks_to_bedtool(student_atac_var_names)#.slop(b=250, genome='hg38')

    # Get the list of indices in var_names corresponding to the overlapping peaks
    overlap = student_atac_bedtool.intersect(peaks_bedtool_list, u=True, f=0.01, F=0.01)
    
    # Map intersected results back to original peak coordinates
    # Find which original peaks correspond to the overlapping slopped peaks
    original_student_peaks_bedtool = peaks_to_bedtool(student_atac_var_names)
    original_student_overlap = original_student_peaks_bedtool.intersect(overlap, u=True)

    original_source_peaks_bedtool = peaks_to_bedtool(source_atac_sub.var_names)
    original_source_overlap = original_source_peaks_bedtool.intersect(peaks_bedtool_list, u=True) # same peak set as peaks_bedtool_list

    peaks_overlap_target = [f"{feature.chrom}:{feature.start}-{feature.end}" for feature in original_student_overlap]
    peaks_overlap_source = [f"{feature.chrom}:{feature.start}-{feature.end}" for feature in original_source_overlap]

    if len(peaks_overlap_target) == 0:
        print(' - No overlapping peaks found')
        return None

    # Get indices for those peaks in the student and source atac data
    target_indices = [i for i, v in enumerate(student_atac_sub.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values) if v in peaks_overlap_target]
    source_indices = [i for i, v in enumerate(source_atac_sub.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values) if v in peaks_overlap_source]
    if student_atac_sub.var_names.tolist() == source_atac_sub.var_names.tolist():
        assert (target_indices == source_indices)
    else:
        print(' - Different number of peaks found in MDD and developmental data')

    # Index by integer index for consistent results
    target_chromatin_accessibility = student_atac_sub.X[:,target_indices].toarray().squeeze()
    source_chromatin_accessibility = source_atac_sub.X[:,source_indices].toarray().squeeze()

    ## create gene expression and chromatin accessibility dataframes
    source_target_gene_expr = pd.Series(
        np.concatenate([target_gene_expr, source_gene_expr], axis=0),
        index=np.concatenate([student_rna_sub.obs_names, source_rna_sub.obs_names]),
        name=hit[0]
    )
    
    #import pdb; pdb.set_trace()
    if len(target_chromatin_accessibility.shape) == 2:
        target_chromatin_accessibility = target_chromatin_accessibility.mean(1)
    if len(source_chromatin_accessibility.shape) == 2:
        source_chromatin_accessibility = source_chromatin_accessibility.mean(1)

    source_target_chromatin_accessibility = pd.Series(
        np.concatenate([target_chromatin_accessibility, source_chromatin_accessibility], axis=0),
        index=np.concatenate([student_atac_sub.obs_names, source_atac_sub.obs_names]),
        name=hit[0]
    )

    ## merge gene expression and chromatin accessibility
    source_target_hit = pd.concat([source_target_gene_expr, source_target_chromatin_accessibility], axis=0)
    #source_target_hit = source_target_hit[~source_target_hit.index.duplicated(keep='first')]

    return source_target_hit

## get gene expression and chromatin accessibility for LD buddies and linked peaks
def get_gene_expression_and_chromatin_accessibility_full(mdd_rna, dev_rna, mdd_atac, dev_atac, genes_peaks_ld_buddy_dict, genes_peaks_linked_dict):
    ld_buddy_hits = []
    for hit in genes_peaks_ld_buddy_dict.keys():
        print(hit)
        genes_peaks_ld_buddy_dict_hit = {hit: genes_peaks_ld_buddy_dict[hit]}
        ld_buddy_hit = get_gene_expression_and_chromatin_accessibility(mdd_rna, dev_rna, mdd_atac, dev_atac, genes_peaks_ld_buddy_dict_hit)
        if ld_buddy_hit is not None:
            ld_buddy_hits.append(ld_buddy_hit)

    linked_hits = []
    for hit in genes_peaks_linked_dict.keys():
        print(hit)
        genes_peaks_linked_dict_hit = {hit: genes_peaks_linked_dict[hit]}
        linked_hit = get_gene_expression_and_chromatin_accessibility(mdd_rna, dev_rna, mdd_atac, dev_atac, genes_peaks_linked_dict_hit)
        if linked_hit is not None:
            linked_hits.append(linked_hit)

    modalities = mdd_rna.obs['modality'].tolist() + dev_rna.obs['modality'].tolist() + mdd_atac.obs['modality'].tolist() + dev_atac.obs['modality'].tolist()

    return ld_buddy_hits, linked_hits, modalities

def plot_gene_expression_and_chromatin_accessibility(ld_buddy_hits_df, linked_hits_df, obs, paths, plot_difference=False, separate_modalities=False):
    """
    Plot gene expression and chromatin accessibility data.
    
    Parameters:
    -----------
    ld_buddy_hits_df : DataFrame
        DataFrame containing LD buddy hits data
    linked_hits_df : DataFrame  
        DataFrame containing linked hits data
    obs : DataFrame
        Observations DataFrame
    paths : dict
        Dictionary containing path information
    plot_difference : bool, default False
        If True, plot ATAC-RNA difference instead of individual modalities.
        If False, plot individual RNA and ATAC modalities separately.
    separate_modalities : bool, default False
        If True, create separate figures for RNA and ATAC modalities.
        If False, plot both modalities on the same figure.
    """

    if (paths is not None):
        obs_keep = obs['leiden'].isin(dict(paths)['L46'])
        ld_buddy_hits_df = ld_buddy_hits_df.loc[obs_keep.values]
        linked_hits_df = linked_hits_df.loc[obs_keep.values]
        obs = obs[obs_keep].copy()
        obs['leiden'] = pd.Categorical(obs['leiden'], categories=dict(paths)['L46'], ordered=True)

    obs_sort = obs['orig_index'].tolist()
    obs.set_index('index', inplace=True)
    ld_buddy_hits_df.drop(columns=['modality'], inplace=True)
    linked_hits_df.drop(columns=['modality'], inplace=True)

    ## create anndata object for LD buddies
    ld_buddy_adata = anndata.AnnData(
        ld_buddy_hits_df, obs=obs,
        obsm={obsm_key: source_target_adata.obsm[obsm_key][obs_sort] for obsm_key in source_target_adata.obsm.keys()})

    linked_adata = anndata.AnnData(
        linked_hits_df, obs=obs,
        obsm={obsm_key: source_target_adata.obsm[obsm_key][obs_sort] for obsm_key in source_target_adata.obsm.keys()})

    #sc.pl.umap(source_target_adata, color='leiden', legend_loc='on data', alpha=0.3)
    #sc.pl.umap(ld_buddy_adata[ld_buddy_adata.obs['source_or_target'].eq('target')], color=ld_buddy_adata.var_names, legend_loc='on data', size=50, ncols=len(ld_buddy_adata.var_names)//2)
    #sc.pl.umap(linked_adata[linked_adata.obs['source_or_target'].eq('target')], color=linked_adata.var_names, legend_loc='on data', size=50)

    pseudotime_keys_continuous = ['ordinal_pseudotime', 'dpt_pseudotime']
    pseudotime_keys_ordinal = ['leiden', 'Age_bins', 'Age_bins_smaller', 'most_common_cluster']
    pseudotime_keys = pseudotime_keys_continuous# + pseudotime_keys_ordinal

    for pseudotime_key in pseudotime_keys:

        if pseudotime_key in pseudotime_keys_continuous:
            n_stages = source_rna_atac_sub.obs['dev_stage'].nunique()
            window_len = int(len(obs) // n_stages * 2)
            window_len = int(window_len / 2)

        for hits_df, hits_adata, hits_name in [(ld_buddy_hits_df, ld_buddy_adata, 'LD buddies'), (linked_hits_df, linked_adata, 'linked peaks')]:

            n_genes = hits_df.shape[1]

            # COMBINED ANALYSIS (source + target with conditions)
            # Prepare target data (with Condition)
            target_indices = hits_adata.obs.loc[hits_adata.obs['source_or_target'].eq('target')].index.tolist()
            tmp_target = hits_df.loc[target_indices].copy()
            tmp_target = tmp_target.assign(
                **{pseudotime_key: hits_adata.obs.loc[target_indices][pseudotime_key], 'Condition': hits_adata.obs.loc[target_indices]['Condition'], 'modality': hits_adata.obs.loc[target_indices]['modality']},
            ).sort_values(pseudotime_key)

            # Prepare source data (no Condition, but we'll add a 'source' label)
            source_indices = hits_adata.obs.loc[hits_adata.obs['source_or_target'].eq('source')].index.tolist()
            tmp_source = hits_df.loc[source_indices].copy()
            tmp_source = tmp_source.assign(
                **{pseudotime_key: hits_adata.obs.loc[source_indices][pseudotime_key], 'modality': hits_adata.obs.loc[source_indices]['modality'], 'Condition': 'developmental'},
            ).sort_values(pseudotime_key)

            if separate_modalities and not plot_difference:
                # Create separate figures for each modality
                modalities_to_plot = ['RNA', 'ATAC']
            else:
                # Create single figure (original behavior or difference plotting)
                modalities_to_plot = [None]  # Single iteration for combined plotting
                
            for modality_plot in modalities_to_plot:
                fig, ax = plt.subplots(int(np.ceil(n_genes/2)), 2, figsize=[14, 10], sharex=True, squeeze=False)
                
                if separate_modalities and not plot_difference:
                    title_suffix = f' ({modality_plot})'
                elif plot_difference:
                    title_suffix = ' (ATAC-RNA difference)'
                else:
                    title_suffix = ' (source + target by Condition)'
                    
                plt.suptitle(f'{hits_name} - {pseudotime_key}{title_suffix}')
                minx = max(tmp_target[pseudotime_key].min(), tmp_source[pseudotime_key].min())
                maxx = min(tmp_target[pseudotime_key].max(), tmp_source[pseudotime_key].max())
                ax[0, 0].set_xlim(minx, maxx)

                for i, gene in enumerate(hits_df.columns):
                    # Set up the subplot for this gene
                    current_ax = ax[i // 2, i % 2]
                    current_ax.set_title(f'{gene} - {genes_km_clusters_dict[gene].item()}' if gene in genes_km_clusters_dict else gene)
                    
                    # Create secondary y-axis for source data (once per gene)
                    ax2 = current_ax.twinx()
                    
                    if plot_difference:
                        # Compute ATAC - RNA difference for each condition
                        for cond in ['case', 'control']:
                            # Get ATAC and RNA data for this condition
                            atac_data = tmp_target[(tmp_target['modality'] == 'ATAC') & (tmp_target['Condition'] == cond)].drop(columns=['modality', 'Condition']).set_index([pseudotime_key])
                            rna_data = tmp_target[(tmp_target['modality'] == 'RNA') & (tmp_target['Condition'] == cond)].drop(columns=['modality', 'Condition']).set_index([pseudotime_key])
                            
                            # Align the dataframes on pseudotime
                            common_idx = atac_data.index.intersection(rna_data.index)
                            if len(common_idx) > 0:
                                atac_aligned = atac_data.loc[common_idx]
                                rna_aligned = rna_data.loc[common_idx]
                                
                                # Compute difference
                                diff_data = atac_aligned - rna_aligned
                                diff_data = diff_data[diff_data.index.notna()]
                                
                                if pseudotime_key in pseudotime_keys_continuous:
                                    diff_data = diff_data.rolling(window=window_len, win_type='hamming', center=True, min_periods=1).mean()
                                else:
                                    diff_data = diff_data.reset_index().groupby([pseudotime_key]).mean()
                                
                                diff_data = diff_data[~diff_data.index.isna()]
                                
                                # Plot difference
                                linestyle = '-' if cond == 'case' else '--'
                                current_ax.plot(diff_data.index, diff_data[gene], 
                                            label=f'{cond} (ATAC-RNA)', 
                                            color='purple', 
                                            linestyle=linestyle,
                                            alpha=0.7)
                    
                    # Compute source difference (ATAC - RNA)
                    atac_source = tmp_source[tmp_source['modality'] == 'ATAC'].drop(columns=['modality', 'Condition']).set_index([pseudotime_key])
                    rna_source = tmp_source[tmp_source['modality'] == 'RNA'].drop(columns=['modality', 'Condition']).set_index([pseudotime_key])
                    
                    common_idx = atac_source.index.intersection(rna_source.index)
                    if len(common_idx) > 0:
                        atac_source_aligned = atac_source.loc[common_idx]
                        rna_source_aligned = rna_source.loc[common_idx]
                        
                        source_diff = atac_source_aligned - rna_source_aligned
                        source_diff = source_diff[source_diff.index.notna()]
                        
                        if pseudotime_key in pseudotime_keys_continuous:
                            source_diff = source_diff.rolling(window=window_len, win_type='hamming', center=True, min_periods=1).mean()
                        else:
                            source_diff = source_diff.reset_index().groupby([pseudotime_key]).mean()
                        
                        source_diff = source_diff[~source_diff.index.isna()]
                        
                        ax2.plot(source_diff.index, source_diff[gene], 
                                label='source (ATAC-RNA)', 
                                color='lightcoral', 
                                linestyle=':', 
                                alpha=0.6)
                
                    else:
                        # Original plotting logic for individual modalities
                        modalities_to_iterate = [modality_plot] if separate_modalities and modality_plot is not None else ['RNA', 'ATAC']
                        
                        for modality in modalities_to_iterate:
                            
                            # Get target data for this modality
                            tmp_target_modality = tmp_target[tmp_target['modality'].eq(modality)].drop(columns=['modality'])
                            # Get source data for this modality
                            tmp_source_modality = tmp_source[tmp_source['modality'].eq(modality)].drop(columns=['modality'])

                            # Plot target conditions (case, control) on primary y-axis
                            cond_data_list = {}
                            for cond in tmp_target_modality['Condition'].unique():
                                cond_data = tmp_target_modality[tmp_target_modality['Condition'].eq(cond)].drop(columns=['Condition']).set_index([pseudotime_key])
                                cond_data = cond_data[cond_data.index.notna()]
                                cond_data_list[cond] = cond_data

                                if pseudotime_key in pseudotime_keys_continuous:
                                    cond_data = cond_data.rolling(window=window_len, win_type='hamming', center=True, min_periods=1).mean().reset_index().set_index(pseudotime_key)
                                else:
                                    cond_data = cond_data.reset_index().groupby([pseudotime_key]).mean()
                                    cond_data.index = pd.Categorical(cond_data.index.astype(str), categories=cond_data.index.astype(str), ordered=True)

                                # Plot with condition-specific linestyles and modality-specific colors
                                color = 'blue' if modality == 'RNA' else 'red'
                                linestyle = '-' if cond == 'case' else '--'
                                current_ax.plot(cond_data.index, cond_data[gene], 
                                              label=f'{cond} ({modality})', 
                                              color=color, 
                                              linestyle=linestyle,
                                              alpha=0.7)

                            # Plot source data on secondary y-axis
                            source_data = tmp_source_modality.drop(columns=['Condition'])

                            if source_data[pseudotime_key].notna().all():
                                source_data = source_data.set_index(pseudotime_key)

                                if pseudotime_key in pseudotime_keys_continuous:
                                    source_data = source_data.rolling(window=window_len, win_type='hamming', center=True, min_periods=1).mean()
                                else:
                                    source_data = source_data.reset_index().groupby([pseudotime_key]).mean()

                                source_data = source_data[~source_data.index.isna()]
                                
                                # Plot source data with modality-specific colors and source-specific linestyle
                                color = 'lightblue' if modality == 'RNA' else 'lightcoral'
                                ax2.plot(source_data.index, source_data[gene], 
                                        label=f'source ({modality})', 
                                        color=color, 
                                        linestyle=':', 
                                        alpha=0.6)
                            else:
                                print(f' - {pseudotime_key} is not available for source data ({modality})')

                    # Set axis labels and styling
                    current_ax.set_xlabel(pseudotime_key)
                    current_ax.tick_params(axis='y', labelcolor='black')
                    ax2.tick_params(axis='y', labelcolor='gray')
                    
                    # Add legends
                    current_ax.legend(loc='upper left', fontsize=8)
                    ax2.legend(loc='upper right', fontsize=8)

                # Set x-axis labels for bottom row
                ax[-1, -1].set_xlabel(pseudotime_key)
                ax[-1, 0].set_xlabel(pseudotime_key)
                
                plt.tight_layout()
                plt.show()



data_type = 'leiden_filtered'
assert data_type in ['aligned','full','leiden_filtered']

if data_type == 'aligned':
    ld_buddy_hits, linked_hits, modalities = \
        get_gene_expression_and_chromatin_accessibility_full(student_rna_sub, source_rna_sub, student_atac_sub, source_atac_sub, genes_peaks_ld_buddy_dict, genes_peaks_linked_dict)

elif data_type == 'leiden_filtered':
    keep_nuclei = source_target_adata[source_target_adata.obs['leiden'].isin(dict(paths)['L46'])].obs_names.tolist()
    ld_buddy_hits, linked_hits, modalities = \
        get_gene_expression_and_chromatin_accessibility_full(student_rna_sub[student_rna_sub.obs_names.isin(keep_nuclei)], source_rna_sub[source_rna_sub.obs_names.isin(keep_nuclei)], student_atac_sub[student_atac_sub.obs_names.isin(keep_nuclei)], source_atac_sub[source_atac_sub.obs_names.isin(keep_nuclei)],
        genes_peaks_ld_buddy_dict, genes_peaks_linked_dict)

elif data_type == 'full':
    ld_buddy_hits, linked_hits, modalities = \
        get_gene_expression_and_chromatin_accessibility_full(mdd_rna_full_sub, pfc_zhu_rna_full_sub, mdd_atac_full_sub, pfc_zhu_atac_full_sub, genes_peaks_ld_buddy_dict, genes_peaks_linked_dict)


## concat hits dataframes
ld_buddy_hits_df = pd.concat(ld_buddy_hits, axis=1)
linked_hits_df = pd.concat(linked_hits, axis=1)

## add modalities to hits dataframes to ensure that corresponding cells are identifiable
ld_buddy_hits_df['modality'] = modalities
linked_hits_df['modality'] = modalities

## create obs object with only unique cells
obs = pd.merge(ld_buddy_hits_df['modality'].reset_index(), source_target_adata.obs.reset_index().assign(orig_index=np.arange(len(source_target_adata.obs))), left_on=['index','modality'], right_on=['index','modality'], how='left')
obs['leiden'] = obs['leiden'].map(path_leidens_mapper)

## create smaller age bins than Age_bins
target_ages = obs.loc[obs['source_or_target'].eq('target'), 'Age']
target_ages = target_ages.cat.remove_unused_categories().astype(float)
if 'Age_bins_smaller' in obs.columns:
    obs.drop(columns=['Age_bins_smaller'], inplace=True)
obs.loc[obs['source_or_target'].eq('target'), 'Age_bins_smaller'] = pd.qcut(target_ages, q=5)

## set most_common_cluster to ClustersMapped for source data
obs.loc[obs['source_or_target'].eq('source'), 'most_common_cluster'] = pd.Categorical(obs.loc[obs['source_or_target'].eq('source'), 'ClustersMapped'], categories=source_clusters, ordered=True)

assert obs['index'].tolist() == ld_buddy_hits_df.index.tolist()
#assert obs['orig_index'].sort_values().values.tolist() == np.arange(len(obs)).tolist()


plot_gene_expression_and_chromatin_accessibility(ld_buddy_hits_df, linked_hits_df, obs, paths, separate_modalities=True)


#%% see which are compliant with diffmap geometry

target_adata_tmp = target_adata.copy()
target_adata = diffmap_match_geom(target_adata_tmp)


#%% keep cells that comply with first diffmap component

remove_leidens = ['3_L23', '4_L23', '5_L23']
target_adata = target_adata[~target_adata.obs['leiden'].isin(remove_leidens)].copy()
sc.pl.umap(target_adata, color='leiden', legend_loc='on data', alpha=0.2)

# Get the unique current leiden clusters in target_adata
current_leidens = set(target_adata.obs['leiden'].unique().tolist())

# Filter paths to only include leiden clusters still present
paths = [
    (descr, [leiden for leiden in path if leiden in current_leidens])
    for descr, path in paths
]


#%% plot rolling mean and standard error of ordinal pseudotime for each leiden cluster

def rolling_mean_and_std_err(target_adata, window_len, paths, d_df_dict_, p_one_sided_dict_, pseudotime_key='ordinal_pseudotime', plot_all=False, split_by_modality=True):
    '''
    Plot rolling mean and standard error of ordinal pseudotime for each leiden cluster.
    
    Parameters:
    -----------
    target_adata : AnnData
        Annotated data object
    window_len : int
        Window length for rolling mean
    paths : list
        List of paths to plot
    d_df_dict_ : dict
        Dictionary containing statistical test results
    p_one_sided_dict_ : dict
        Dictionary containing p-values for one-sided tests
    pseudotime_key : str, default 'ordinal_pseudotime'
        Key for pseudotime values in obs
    plot_all : bool, default False
        Whether to plot all nuclei
    split_by_modality : bool, default True
        If True, create separate plots for RNA and ATAC modalities.
        If False, combine RNA and ATAC data without modality distinction, showing only case vs control.
    
    '''

    # Create a single figure with all paths
    n_paths = len(paths)
    
    if split_by_modality:

        ## rolling mean
        means = target_adata.obs.groupby(['leiden','modality','Condition'])[pseudotime_key].mean()
        means_pivot = means.reset_index().pivot_table(index='leiden', columns=['modality','Condition'], values=pseudotime_key)
        means_pivot = means_pivot.rolling(window_len, min_periods=0).mean()
        means_pivot = means_pivot.melt(ignore_index=False).rename(columns={'value': pseudotime_key})
        means_pivot['Condition_modality'] = means_pivot[['Condition','modality']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)

        ## rolling standard error
        std_errs = target_adata.obs.groupby(['leiden','modality','Condition'])[pseudotime_key].std() / np.sqrt(target_adata.obs.groupby(['leiden','modality','Condition'])[pseudotime_key].count())
        std_errs_pivot = std_errs.reset_index().pivot_table(index='leiden', columns=['modality','Condition'], values=pseudotime_key)
        std_errs_pivot = std_errs_pivot.rolling(window_len, min_periods=0).mean()
        std_errs_pivot = std_errs_pivot.melt(ignore_index=False).rename(columns={'value': pseudotime_key})
        std_errs_pivot['Condition_modality'] = std_errs_pivot[['Condition','modality']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)

        ## resort Leiden clusters
        leiden_sorted = means_pivot.groupby('leiden')[pseudotime_key].mean().sort_values().index.tolist()
        means_pivot.index = pd.CategoricalIndex(means_pivot.index, categories=leiden_sorted, ordered=True)
        std_errs_pivot.index = pd.CategoricalIndex(std_errs_pivot.index, categories=leiden_sorted, ordered=True)
        means_pivot.sort_index(inplace=True)
        std_errs_pivot.sort_index(inplace=True)

        n_modalities = 3 if plot_all else 2  # RNA, ATAC, and optionally all nuclei
        # Create subplots: rows = modalities, columns = paths
        fig, axes = plt.subplots(n_modalities, n_paths, 
                                figsize=[3*n_paths, 3*n_modalities], 
                                sharex='col', sharey=False)
        
    else:
        # Use the simple plotting approach when not splitting by modality
        n_modalities = 1
        fig, ax = plt.subplots(1, len(paths), figsize=(3*len(paths), 3), sharex='col', sharey=False)
        if len(paths) == 1:
            ax = [ax]
        
        for i, (descr, path) in enumerate(paths):
            path_data = target_adata[target_adata.obs['leiden'].isin(path)]
            path_data.obs['leiden'] = pd.Categorical(path_data.obs['leiden'], categories=path, ordered=True)
            ax[i].set_title(descr)
            sns.lineplot(path_data.obs[['leiden','Condition',pseudotime_key]], 
                        x='leiden', y=pseudotime_key, hue='Condition', 
                        palette='Pastel1', errorbar='se', marker='o', 
                        legend=True if i==0 else False, ax=ax[i])

        for a in ax:
            for label in a.get_xticklabels():
                label.set_rotation(30)
        plt.tight_layout()

    # Handle case where there's only one path (axes will be 1D)
    if n_paths == 1:
        axes = axes.reshape(-1, 1)
    if n_modalities == 1:
        axes = ax.reshape(1, -1)

    # Process each path
    for col_idx, (descr, path) in enumerate(paths):
        means_pivot_path = means_pivot[means_pivot.index.isin(path)].reset_index()
        std_errs_pivot_path = std_errs_pivot[std_errs_pivot.index.isin(path)].reset_index()

        means_pivot_path['leiden'] = pd.Categorical(means_pivot_path['leiden'], categories=path, ordered=True)
        std_errs_pivot_path['leiden'] = pd.Categorical(std_errs_pivot_path['leiden'], categories=path, ordered=True)

        # Get significance levels for this path
        p_one_sided_path_rna = p_one_sided_dict_['RNA'][descr]
        p_one_sided_path_atac = p_one_sided_dict_['ATAC'][descr]
        
        #plt.rcParams['mathtext.fontset'] = 'stix'
        from matplotlib.font_manager import FontProperties
        #mono = FontProperties(family="monospace", math_fontfamily='stixsans')
        mono = FontProperties()

        sig_levels = pd.Series({1.0: 'ns', 0.05: r"$\boldsymbol{\ast}$", 0.01: r"$\boldsymbol{\ast\ast}$", 0.001: r"$\boldsymbol{\ast\ast\!\ast}$"})
        sig_level_rna = sig_levels.loc[p_one_sided_path_rna <= sig_levels.index].iloc[-1]
        sig_level_atac = sig_levels.loc[p_one_sided_path_atac <= sig_levels.index].iloc[-1]

        # Plot RNA (row 0)
        ax_rna = axes[0, col_idx]
        sns.lineplot(
            data=means_pivot_path[means_pivot_path['modality']=='RNA'],
            x='leiden', y=pseudotime_key, palette='Pastel1',
            hue='Condition_modality', hue_order=['case_RNA', 'control_RNA'],
            errorbar=None, marker='.', markersize=20, linewidth=2, ax=ax_rna
        )
        
        # Plot ATAC (row 1)
        ax_atac = axes[1, col_idx]
        sns.lineplot(
            data=means_pivot_path[means_pivot_path['modality']=='ATAC'],
            x='leiden', y=pseudotime_key, palette='Pastel1',
            hue='Condition_modality', hue_order=['case_ATAC', 'control_ATAC'],
            errorbar=None, marker='.', markersize=20, linewidth=2, linestyle='--', ax=ax_atac
        )

        # Fill between values for error bars
        x = means_pivot_path['leiden']
        low = (means_pivot_path[pseudotime_key] - std_errs_pivot_path[pseudotime_key]).values
        high = (means_pivot_path[pseudotime_key] + std_errs_pivot_path[pseudotime_key]).values
        high_low = x.to_frame().assign(low=low, high=high)
        
        x = x.sort_values()
        high_low = high_low.sort_values('leiden')
        high = high_low['high']
        low = high_low['low']

        # Fill between for RNA
        ax_rna.fill_between(x[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], 
                           low[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], 
                           high[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='case')], 
                           alpha=0.3, color=sns.color_palette('Pastel1')[0])
        ax_rna.fill_between(x[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], 
                           low[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], 
                           high[(means_pivot_path['modality']=='RNA') & (means_pivot_path['Condition']=='control')], 
                           alpha=0.3, color=sns.color_palette('Pastel1')[1])
        
        # Fill between for ATAC
        ax_atac.fill_between(x[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], 
                            low[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], 
                            high[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='case')], 
                            alpha=0.3, color=sns.color_palette('Pastel1')[0])
        ax_atac.fill_between(x[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], 
                            low[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], 
                            high[(means_pivot_path['modality']=='ATAC') & (means_pivot_path['Condition']=='control')], 
                            alpha=0.3, color=sns.color_palette('Pastel1')[1])

        # Set labels and titles
        ax_rna.set_ylabel(pseudotime_key if col_idx == 0 else '', color='grey')
        ax_rna.set_title(f'RNA [{sig_level_rna}]', fontproperties=mono)
        ax_rna.tick_params(colors='grey')
        ax_rna.spines['top'].set_color('grey')
        ax_rna.spines['right'].set_color('grey')
        ax_rna.spines['bottom'].set_color('grey')
        ax_rna.spines['left'].set_color('grey')
        ax_rna.set_xticklabels(ax_rna.get_xticklabels(), rotation=30)
        ax_rna.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        
        ax_atac.set_ylabel(pseudotime_key if col_idx == 0 else '', color='grey')
        ax_atac.set_title(f'ATAC [{sig_level_atac}]', fontproperties=mono)
        ax_atac.tick_params(colors='grey')
        ax_atac.spines['top'].set_color('grey')
        ax_atac.spines['right'].set_color('grey')
        ax_atac.spines['bottom'].set_color('grey')
        ax_atac.spines['left'].set_color('grey')
        ax_atac.set_xticklabels(ax_atac.get_xticklabels(), rotation=30)
        ax_atac.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

        # Plot all nuclei if requested (row 2)
        if plot_all:
            ax_all = axes[2, col_idx]
            sns.lineplot(data=means_pivot_path, x='leiden', y=pseudotime_key, color='grey', 
                        errorbar=None, marker='.', markersize=20, linewidth=2, ax=ax_all)
            ax_all.fill_between(x.unique().tolist(), 
                               high_low.groupby('leiden')['low'].mean().dropna(), 
                               high_low.groupby('leiden')['high'].mean().dropna(), 
                               alpha=0.15, color='grey')
            ax_all.set_title('all nuclei')
            ax_all.set_ylabel(pseudotime_key if col_idx == 0 else '', color='grey')
            ax_all.tick_params(colors='grey')
            ax_all.spines['top'].set_color('grey')
            ax_all.spines['right'].set_color('grey')
            ax_all.spines['bottom'].set_color('grey')
            ax_all.spines['left'].set_color('grey')
            ax_all.set_xticklabels(ax_all.get_xticklabels(), rotation=30)
            ax_all.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

        # Add daggers for significant differences
        y_shift = 0.3
        
        for l, leiden in enumerate(path):
            p_one_sided_path_rna_leiden = d_df_dict_['RNA'][descr].loc[leiden].loc['p']
            p_one_sided_path_atac_leiden = d_df_dict_['ATAC'][descr].loc[leiden].loc['p']

            # Get the actual x-coordinate for this leiden cluster (it's just the index l)
            x_coord = l
            
            # Find the maximum y-value (including error bars) for this leiden cluster
            leiden_mask_rna = means_pivot_path['leiden'] == leiden
            leiden_mask_atac = means_pivot_path['leiden'] == leiden
            
            if p_one_sided_path_rna_leiden < 0.05 and leiden_mask_rna.any():
                # Get the maximum y-value for this leiden cluster in RNA
                rna_data = means_pivot_path[leiden_mask_rna & (means_pivot_path['modality']=='RNA')]
                if not rna_data.empty:
                    max_y_rna = rna_data[pseudotime_key].max()
                    # Add error bar height
                    rna_std_data = std_errs_pivot_path[leiden_mask_rna & (std_errs_pivot_path['modality']=='RNA')]
                    if not rna_std_data.empty:
                        max_y_rna += rna_std_data[pseudotime_key].max()
                    ax_rna.text(x_coord, max_y_rna + y_shift, '†', fontsize=14, color='grey', 
                              ha='center', va='bottom')
            
            if p_one_sided_path_atac_leiden < 0.05 and leiden_mask_atac.any():
                # Get the maximum y-value for this leiden cluster in ATAC
                atac_data = means_pivot_path[leiden_mask_atac & (means_pivot_path['modality']=='ATAC')]
                if not atac_data.empty:
                    max_y_atac = atac_data[pseudotime_key].max()
                    # Add error bar height
                    atac_std_data = std_errs_pivot_path[leiden_mask_atac & (std_errs_pivot_path['modality']=='ATAC')]
                    if not atac_std_data.empty:
                        max_y_atac += atac_std_data[pseudotime_key].max()
                    ax_atac.text(x_coord, max_y_atac + y_shift, '†', fontsize=14, color='grey', 
                               ha='center', va='bottom')

        # Set column titles (path names)
        if n_modalities > 1:
            axes[0, col_idx].text(0.5, 1.15, descr, transform=axes[0, col_idx].transAxes, 
                                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

    # Create a single legend for the entire figure
    handles_rna, labels_rna = axes[0, 0].get_legend_handles_labels()
    handles_atac, labels_atac = axes[1, 0].get_legend_handles_labels()
    labels_rna = [label.replace('_', ' - ') for label in labels_rna]
    labels_atac = [label.replace('_', ' - ') for label in labels_atac]

    # Add single legend
    fig.legend(handles_rna + handles_atac, labels_rna + labels_atac, loc='center left', bbox_to_anchor=(1.0, 0.5))
    axes[0,0].set_ylabel('ordinal pseudotime')
    axes[1,0].set_ylabel('ordinal pseudotime')
    
    # Remove legends from all subplots
    for i in range(n_modalities):
        for j in range(n_paths):
            if axes[i, j].get_legend() is not None:
                axes[i, j].get_legend().remove()
    
    fig.tight_layout()
    fig.show()
    
    return fig

def scenic_plus_lineplots(target_adata, paths, d_df_dict, p_one_sided_dict, pseudotime_key='ordinal_pseudotime'):
    
    fig, ax = plt.subplots(1, len(paths), figsize=(4*len(paths), 4.5), sharex='col', sharey=False)
    if len(paths) == 1:
        ax = [ax]
    
    # Set up font properties for consistency
    from matplotlib.font_manager import FontProperties
    mono = FontProperties()
    
    # Define significance levels
    sig_levels = pd.Series({1.0: 'ns', 0.05: r"$\boldsymbol{\ast}$", 0.01: r"$\boldsymbol{\ast\ast}$", 0.001: r"$\boldsymbol{\ast\ast\!\ast}$"})
    
    for i, (descr, path) in enumerate(paths):
        path_data = target_adata[target_adata.obs['leiden'].isin(path)]
        path_data.obs['leiden'] = pd.Categorical(path_data.obs['leiden'], categories=path, ordered=True)
        
        # Get significance level for this path
        p_one_sided_path = p_one_sided_dict[descr]
        sig_level = sig_levels.loc[p_one_sided_path <= sig_levels.index].iloc[-1]
        
        # Set title with significance annotation
        ax[i].set_title(f'{descr} [{sig_level}]', fontproperties=mono)
        
        # Create lineplot with consistent styling
        sns.lineplot(path_data.obs[['leiden','Condition',pseudotime_key]], 
                    x='leiden', y=pseudotime_key, hue='Condition', 
                    palette='Pastel1', errorbar='se', marker='o', 
                    legend=True if i==0 else False, ax=ax[i])
        
        # Apply consistent styling to match rolling_mean_and_std_err
        ax[i].tick_params(colors='grey')
        ax[i].spines['top'].set_color('grey')
        ax[i].spines['right'].set_color('grey')
        ax[i].spines['bottom'].set_color('grey')
        ax[i].spines['left'].set_color('grey')
        #ax[i].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax[i].set_yticklabels('')
        
        # Add daggers for significant differences at individual leiden clusters
        y_shift = 0.03
        for l, leiden in enumerate(path):
            if leiden in d_df_dict[descr].index:
                p_one_sided_leiden = d_df_dict[descr].loc[leiden, 'p']
                if p_one_sided_leiden < 0.05:
                    # Get the maximum y-value for this leiden cluster
                    leiden_data = path_data.obs[path_data.obs['leiden'] == leiden]
                    if not leiden_data.empty:
                        max_y = leiden_data.groupby('Condition')[pseudotime_key].mean().max()
                        # Add some buffer for the dagger
                        ax[i].text(l, max_y + y_shift, '†', fontsize=14, color='grey', 
                                 ha='center', va='bottom')

    # Apply consistent styling to all axes
    for a in ax:
        for label in a.get_xticklabels():
            label.set_rotation(30)
        a.set_ylabel(pseudotime_key, color='grey')
    
    # Create a single legend for the entire figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Remove legends from all subplots
    for a in ax:
        if a.get_legend() is not None:
            a.get_legend().remove()
    
    plt.tight_layout()
    fig.show()
    
    return fig

#%% perform HAC weighted mean tests for each path and plot rolling mean and standard error

## focus on paths L23 and L46 (in that order)
paths = [('L23', dict(paths)['L23']), ('L46', dict(paths)['L46'])]

from eclare.post_hoc_utils import tree

window_len = 1
pseudotime_key = 'ordinal_pseudotime'

d_df_dict = tree()
p_one_sided_dict = tree()

for sex in ['both','female','male']:
    for modality in ['RNA','ATAC']:
        for descr, path in paths:

            print(f'{sex} {modality} {descr}')

            if sex == 'both':
                df = target_adata.obs[
                    (target_adata.obs['modality'] == modality)
                ]
            else:
                df = target_adata.obs[
                    (target_adata.obs['modality'] == modality) & \
                    (target_adata.obs['Sex'] == sex)
                ]

            d_df, p_one_sided = hac_weighted_mean_test(df, path, window_len, direction='all', pseudotime_key=pseudotime_key)
            d_df_dict[sex][modality][descr] = d_df
            p_one_sided_dict[sex][modality][descr] = p_one_sided

            print('')

## lineplots
figs_both = rolling_mean_and_std_err(target_adata, window_len, paths, d_df_dict['both'], p_one_sided_dict['both'], pseudotime_key=pseudotime_key)
figs_female = rolling_mean_and_std_err(target_adata[target_adata.obs['Sex']=='female'], window_len, paths, d_df_dict['female'], p_one_sided_dict['female'], pseudotime_key=pseudotime_key)
figs_male = rolling_mean_and_std_err(target_adata[target_adata.obs['Sex']=='male'], window_len, paths, d_df_dict['male'], p_one_sided_dict['male'], pseudotime_key=pseudotime_key)

def devmdd_fig4(lineplots_fig, suffix='', manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'devmdd_fig4.svg')):

    lineplots_fig.suptitle(lineplots_fig._suptitle.get_text() + ' - ' + str(suffix) if lineplots_fig._suptitle is not None else str(suffix))
    lineplots_fig.savefig(manuscript_figpath.replace('.svg', f'_{descr}_{suffix}.svg'), bbox_inches='tight', dpi=300)
    print(f"Saving figure to {manuscript_figpath.replace('.svg', f'_{descr}_{suffix}.svg')}")
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

#%%
## plot embedding density for each cell type
sc.tl.embedding_density(source_target_adata, basis=basis, groupby='source_or_target')
sc.pl.embedding_density(source_target_adata, basis=basis, key=f'{basis}_density_source_or_target', ncols=source_target_adata.obs['source_or_target'].nunique())

source_adata.obs['SubClusters'] = pd.Categorical(source_adata.obs['SubClusters'], categories=source_clusters, ordered=True).remove_unused_categories()
sc.tl.embedding_density(source_adata, basis=basis, groupby='SubClusters')
sc.pl.embedding_density(source_adata, basis=basis, key=f'{basis}_density_SubClusters', ncols=source_adata.obs['SubClusters'].nunique())
#sc.pl.embedding_density(source_adata[source_adata.obs['modality']=='ATAC'], basis=basis, key=f'{basis}_density_SubClusters')
#sc.pl.embedding_density(source_adata[source_adata.obs['modality']=='RNA'], basis=basis, key=f'{basis}_density_SubClusters')

target_adata.obs['SubClusters'] = pd.Categorical(target_adata.obs['SubClusters'], categories=target_adata.obs.groupby('SubClusters')['ordinal_pseudotime'].mean().sort_values().index.tolist(), ordered=True).remove_unused_categories()
sc.tl.embedding_density(target_adata, basis=basis, groupby='SubClusters')
sc.pl.embedding_density(target_adata, basis=basis, key=f'{basis}_density_SubClusters')
#sc.pl.embedding_density(target_adata[target_adata.obs['modality']=='ATAC'], basis=basis, key=f'{basis}_density_SubClusters')
#sc.pl.embedding_density(target_adata[target_adata.obs['modality']=='RNA'], basis=basis, key=f'{basis}_density_SubClusters')

## create subsampled eclare adata, even if in fact is KD-CLIP data
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

adata = source_target_adata.copy()

#sc.tl.umap(adata, n_components=3)
#sc.pl.umap(adata, color='leiden', projection='3d')

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

#%% PAGA paths

#student_rna_sub.obs = student_rna_sub.obs.merge(adata.obs.set_index('cell')[['most_common_cluster', 'dpt_pseudotime', 'leiden']], left_index=True, right_index=True, how='left')
student_rna_sub.obs = student_rna_sub.obs.merge(source_target_adata.obs[['most_common_cluster', 'dpt_pseudotime', 'leiden']], left_index=True, right_index=True, how='left')
rna_tmp = student_rna_sub[student_rna_sub.obs['ClustersMapped'] == 'ExN']

student_atac_sub.obs = student_atac_sub.obs.merge(source_target_adata.obs[['most_common_cluster', 'dpt_pseudotime', 'leiden']], left_index=True, right_index=True, how='left')
atac_tmp = student_atac_sub[student_atac_sub.obs['ClustersMapped'] == 'ExN']

r_rna = gene_obs_pearson(rna_tmp, "dpt_pseudotime")
r_tops_rna = pd.concat([r_rna.sort_values(ascending=False).head(8), r_rna.sort_values(ascending=True).head(8)[::-1]], axis=0)
r_tops_index_rna = r_tops_rna.index.tolist()

_, axs = plt.subplots(ncols=len(paths), figsize=(10, 10), gridspec_kw={                     'wspace': 0.05, 'left': 0.12})
plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)

for ipath, (descr, path) in enumerate(paths):
    sc.pl.paga_path(
        adata=rna_tmp, 
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

X_key = f'X_{basis}'
plot_key = f'X_{basis}'

X_rna = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obsm[X_key]
X_atac = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obsm[X_key]

plot_rna = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obsm[plot_key]
plot_atac = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obsm[plot_key]

groups_rna = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'RNA'].obs[['Condition','Sex']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1).values
groups_atac = subsampled_eclare_adata[subsampled_eclare_adata.obs['modality'] == 'ATAC'].obs[['Condition','Sex']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1).values

groups_mask = (groups_rna[:,None] == groups_atac[None,:])

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

## apply groups mask to M
M[~groups_mask] = 1e12

## compute regularization value
reg = np.median(M[~np.isnan(M)]) / 100

#G = ot.sinkhorn(a, b, M, 1e-3)
#G = ot.emd(a, b, M)
m = len(b) / len(a)
print(f'm: {m:.2f}')
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

groups = subsampled_eclare_adata.obs.set_index('Cell_ID_OT')[['Condition','Sex']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)
max_ngroups_per_cellidot = groups.groupby('Cell_ID_OT').apply(lambda x: x.value_counts().max())
prop_ingroup_pairs = max_ngroups_per_cellidot.value_counts(normalize=True).loc[2]
print(f'Proportion of in-group pairs: {prop_ingroup_pairs:.2f}')

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
sc.tl.embedding_density(subsampled_eclare_adata, basis=basis, groupby='modality')
sc.pl.embedding_density(subsampled_eclare_adata, basis=basis, key=f'{basis}_density_modality')

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

## save adatas
os.makedirs(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results'), exist_ok=True)

# Cast object columns to str before saving
cast_object_columns_to_str(subsampled_eclare_adata)
subsampled_eclare_adata.write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_eclare_adata_{target_dataset}.h5ad'))

cast_object_columns_to_str(scJoint_adata)
scJoint_adata.write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'scJoint_adata_{target_dataset}.h5ad'))

cast_object_columns_to_str(glue_adata)
glue_adata.write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'glue_adata_{target_dataset}.h5ad'))

for source_dataset in source_datasets:
    cast_object_columns_to_str(subsampled_kd_clip_adatas[source_dataset])
    subsampled_kd_clip_adatas[source_dataset].write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_kd_clip_adatas_{source_dataset}_{target_dataset}.h5ad'))

    cast_object_columns_to_str(subsampled_clip_adatas[source_dataset])
    subsampled_clip_adatas[source_dataset].write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_clip_adatas_{source_dataset}_{target_dataset}.h5ad'))

#%% load adatas

subsampled_eclare_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_eclare_adata_{target_dataset}.h5ad'))
scJoint_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'scJoint_adata.h5ad'))
glue_adata = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'glue_adata.h5ad'))

for source_dataset in source_datasets:
    subsampled_kd_clip_adatas[source_dataset] = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_kd_clip_adatas_{source_dataset}.h5ad'))
    subsampled_clip_adatas[source_dataset] = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_clip_adatas_{source_dataset}.h5ad'))


#%% plot eRegulon scores

signatures = sc.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'dev_fig3_signatures.h5ad'))

if 'Sex' not in signatures.obs.columns:
    signatures.obs = signatures.obs.merge(subsampled_eclare_adata.obs['Sex'], left_index=True, right_index=True, how='left')
if 'modality' not in signatures.obs.columns:
    signatures.obs = signatures.obs.merge(subsampled_eclare_adata.obs['modality'], left_index=True, right_index=True, how='left')

eRegulons_all = pd.Series(np.hstack(list({
    'Velmeshev_L23': ['ZNF184', 'NFIX'],
    'Velmeshev_L5': ['BACH2', 'SOX5'],
    'Velmeshev_sexes_L6': ['CPLX3', 'JAM2', 'NXPH3'],
    'Velmeshev_sexes_L23': ['RORA', 'CNIH3', 'CNTNAP3B'],
    'sc-compReg': ['EGR1', 'SOX2', 'NR4A2'],
    'Wang_MDD': ['MEF2C', 'SATB2', 'FOXP1', 'POU3F1', 'PKNOX2', 'CUX2', 'THRB', 'POU6F2', 'RORB', 'ZBTB18']
    }.values())))
eRegulons = eRegulons_all[eRegulons_all.isin(signatures.obs.columns)].tolist()

d_df_signatures_dict = tree()
p_one_sided_signatures_dict = tree()

for eRegulon in eRegulons:
    for sex in ['both','female','male']:
        for descr, path in paths:

            if sex == 'both':
                df = signatures.obs.copy()
            else:
                df = signatures.obs[signatures.obs['Sex']==sex].copy()

            d_df, p_one_sided = hac_weighted_mean_test(df, path, window_len, pseudotime_key=eRegulon, two_sided=True)
            d_df_signatures_dict[eRegulon][sex][descr] = d_df
            p_one_sided_signatures_dict[eRegulon][sex][descr] = p_one_sided

figs_signatures_both    = scenic_plus_lineplots(signatures, paths, d_df_signatures_dict['EGR1']['both'], p_one_sided_signatures_dict['EGR1']['both'], pseudotime_key='EGR1')
figs_signatures_female  = scenic_plus_lineplots(signatures[signatures.obs['Sex']=='female'], paths, d_df_signatures_dict['EGR1']['female'], p_one_sided_signatures_dict['EGR1']['female'], pseudotime_key='EGR1')
figs_signatures_male    = scenic_plus_lineplots(signatures[signatures.obs['Sex']=='male'], paths, d_df_signatures_dict['EGR1']['male'], p_one_sided_signatures_dict['EGR1']['male'], pseudotime_key='EGR1')

'''
fig, ax = plt.subplots(len(eRegulons), len(paths), figsize=(6, 10), sharex='col', sharey=False)
for i, (descr, path) in enumerate(paths):
    path_signatures = signatures[signatures.obs['leiden'].isin(path)]
    path_signatures.obs['leiden'] = pd.Categorical(path_signatures.obs['leiden'], categories=path, ordered=True)
    ax[0,i].set_title(descr)
    for j, eRegulon in enumerate(eRegulons):
        sns.lineplot(path_signatures.obs[['leiden','Condition',eRegulon]], x='leiden', y=eRegulon, hue='Condition', palette='Pastel1', errorbar='se', marker='o', legend=True if (i,j)==(0,0) else False, ax=ax[j,i])

for a in ax.flat:
    for label in a.get_xticklabels():
        label.set_rotation(30)
plt.tight_layout()
'''

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

