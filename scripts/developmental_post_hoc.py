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

from eclare.post_hoc_utils import metric_boxplots, download_mlflow_runs, extract_target_source_replicate
from eclare.post_hoc_utils import load_model_and_metadata
from eclare.setup_utils import return_setup_func_from_dataset

cuda_available = torch.cuda.is_available()
n_cudas = torch.cuda.device_count()
#device = torch.device(f'cuda:{n_cudas - 1}') if cuda_available else 'cpu'
device = 'cpu'

## Define target and source datasets
target_dataset = 'Cortex_Velmeshev'
source_datasets = ['PFC_V1_Wang', 'PFC_Zhu']
subsample = 5000

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '25165730',
    'kd_clip': '25173640',
    'eclare': ['25182625'],
    'ordinal': '27204131',
}

## define search strings
search_strings = {
    'clip': 'CLIP' + '_' + methods_id_dict['clip'],
    'kd_clip': 'KD_CLIP' + '_' + methods_id_dict['kd_clip'],
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

if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
else:
    print(f"Downloading runs.csv for {experiment_name} from MLflow")
    all_metrics_csv_path = download_mlflow_runs(experiment_name)

all_metrics_df = pd.read_csv(all_metrics_csv_path)

CLIP_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['clip']))[0]
KD_CLIP_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['kd_clip']))[0]
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

## Find path to best ECLARE model
best_eclare     = str(ECLARE_metrics_df['compound_metric'].argmax())
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_{target_dataset.lower()}_{methods_id_dict["eclare"][0]}', best_eclare, device, target_dataset=target_dataset)
eclare_student_model = eclare_student_model.eval().to(device=device)

## Load KD_CLIP student model
best_kd_clip = '0'
kd_clip_student_models = {}

for source_dataset in source_datasets:
    kd_clip_student_model, kd_clip_student_model_metadata     = load_model_and_metadata(f'kd_clip_{target_dataset.lower()}_{methods_id_dict["kd_clip"]}', best_kd_clip, device, target_dataset=os.path.join(target_dataset, source_dataset))
    kd_clip_student_models[source_dataset] = kd_clip_student_model

#%% Get student loaders

student_setup_func = return_setup_func_from_dataset(target_dataset)

args = SimpleNamespace(
    source_dataset=target_dataset,
    target_dataset=None,
    genes_by_peaks_str='9584_by_66620',
    ignore_sources=[None],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)

student_rna, student_atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = student_setup_func(args, return_type='data')
#student_rna_keep = student_rna
#student_atac_keep = student_atac

dev_group_key = 'Age_Range'
dev_stages = ['2nd trimester', '3rd trimester', '0-1 years', '1-2 years', '2-4 years', '4-10 years', '10-20 years', 'Adult']

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

#%% load ordinal model
import mlflow

ordinal_model_uri_paths_str = f"ordinal_*{methods_id_dict['ordinal']}/model_uri.txt"
ordinal_model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], ordinal_model_uri_paths_str))
assert len(ordinal_model_uri_paths) > 0, f'Model URI path not found @ {ordinal_model_uri_paths_str}'

with open(ordinal_model_uri_paths[0], 'r') as f:
    model_uris = f.read().strip().splitlines()
    model_uri = model_uris[0]

mlflow.set_tracking_uri('file:///home/mcb/users/dmannk/scMultiCLIP/ECLARE/mlruns')
ordinal_model = mlflow.pytorch.load_model(model_uri, device=device)

#%% extract student latents for analysis
from copy import deepcopy
from eclare.post_hoc_utils import create_celltype_palette
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def subsample_adata(adata, cell_group, dev_group_key, subsample, subsample_type='balanced'):
    combinations = adata.obs[[cell_group, dev_group_key]].apply(lambda x: ' - '.join(x), axis=1).values

    if subsample_type == 'stratified':
        n_cells = adata.n_obs
        n_splits = np.ceil(n_cells / subsample).astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        _, valid_idx = next(skf.split(np.zeros_like(combinations), combinations))

    elif subsample_type == 'balanced':
        classes = np.unique(combinations)
        k = len(classes)
        per_class = subsample // k
        counts = Counter(combinations)
        strategy = {c: min(counts[c], per_class) for c in classes}
        rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
        valid_idx, valid_devs = rus.fit_resample(np.arange(len(combinations)).reshape(-1, 1), combinations)

    return adata[valid_idx.flatten()].copy()

student_model = deepcopy(eclare_student_model)
#print('Using KD-CLIP student model'); student_model = deepcopy(kd_clip_student_models['PFC_V1_Wang'])

## project data through student model
student_rna_cells = torch.from_numpy(student_rna.X.toarray().astype(np.float32))
student_atac_cells = torch.from_numpy(student_atac.X.toarray().astype(np.float32))

student_rna_latents, _ = student_model.to('cpu')(student_rna_cells, modality=0)
student_atac_latents, _ = student_model.to('cpu')(student_atac_cells, modality=1)

## project data through ordinal model and assign to adata
ordinal_rna_logits, ordinal_rna_probas, ordinal_rna_latents = ordinal_model.to('cpu')(student_rna_cells, modality=0)
ordinal_atac_logits, ordinal_atac_probas, ordinal_atac_latents = ordinal_model.to('cpu')(student_atac_cells, modality=1)

ordinal_rna_prebias = ordinal_model.ordinal_layer_rna.coral_weights(ordinal_rna_latents)
ordinal_atac_prebias = ordinal_model.ordinal_layer_atac.coral_weights(ordinal_atac_latents)

ordinal_rna_pt = torch.sigmoid(ordinal_rna_prebias / ordinal_rna_logits.var().pow(0.5)).flatten().detach().cpu().numpy()
ordinal_atac_pt = torch.sigmoid(ordinal_atac_prebias / ordinal_atac_logits.var().pow(0.5)).flatten().detach().cpu().numpy()

student_rna.obs['ordinal_pseudotime'] = ordinal_rna_pt
student_atac.obs['ordinal_pseudotime'] = ordinal_atac_pt

## subsample data
student_rna_sub = subsample_adata(student_rna, cell_group, dev_group_key, subsample, subsample_type='balanced')
student_atac_sub = subsample_adata(student_atac, cell_group, dev_group_key, subsample, subsample_type='balanced')

## get subset data latents for retraining UMAP
student_rna_cells_sub = torch.from_numpy(student_rna_sub.X.toarray().astype(np.float32))
student_atac_cells_sub = torch.from_numpy(student_atac_sub.X.toarray().astype(np.float32))

student_rna_latents_sub, _ = student_model.to('cpu')(student_rna_cells_sub, modality=0)
student_atac_latents_sub, _ = student_model.to('cpu')(student_atac_cells_sub, modality=1)

## define color palettes
cmap_dev = plt.get_cmap('plasma', len(dev_stages))
cmap_dev = {dev_stages[i]: cmap_dev(i) for i in range(len(dev_stages))}
cmap_ct = create_celltype_palette(student_rna_sub.obs[cell_group].values, student_atac_sub.obs[cell_group].values, plot_color_palette=False)


#%% Create ECLARE adata
import scanpy as sc

## create ECLARE adata for subsampled data
X = np.vstack([
    student_rna_latents_sub.detach().cpu().numpy(),
    student_atac_latents_sub.detach().cpu().numpy()
    ])

obs = pd.concat([
    student_rna_sub.obs.assign(modality='RNA'),
    student_atac_sub.obs.assign(modality='ATAC')
    ])
obs[dev_group_key] = pd.Categorical(obs[dev_group_key], categories=dev_stages, ordered=True)

subsampled_eclare_adata = sc.AnnData(
    X=X,
    obs=obs,
)

## create ECLARE adata for full data
X = np.vstack([
    student_rna_latents.detach().cpu().numpy(),
    student_atac_latents.detach().cpu().numpy()
    ])

obs = pd.concat([
    student_rna.obs.assign(modality='RNA'),
    student_atac.obs.assign(modality='ATAC')
    ])

eclare_adata = sc.AnnData(
    X=X,
    obs=obs,
)
eclare_adata.obs[dev_group_key] = eclare_adata.obs[dev_group_key].cat.reorder_categories(dev_stages, ordered=True)

## create unimodal ECLARE adatas
rna_adata = sc.AnnData(
    X=student_rna_latents.detach().cpu().numpy(),
    obs=student_rna.obs.assign(modality='RNA'),
)

atac_adata = sc.AnnData(
    X=student_atac_latents.detach().cpu().numpy(),
    obs=student_atac.obs.assign(modality='ATAC'),
)

#%% Import latents from other models

## create ordinal_pseudotime Series
ordinal_pseudotime_adata = pd.concat([student_rna.obs['ordinal_pseudotime'], student_atac.obs['ordinal_pseudotime']])

## scJoint
scJoint_path = os.path.join(os.environ['OUTPATH'], 'scJoint_data_tmp', '20250903_190439', 'scJoint_latents.h5ad')
scJoint_adata = sc.read_h5ad(scJoint_path)
scJoint_adata.obs = scJoint_adata.obs.merge(ordinal_pseudotime_adata, left_index=True, right_index=True)

## scGLUE
glue_path = os.path.join(os.environ['OUTPATH'], 'glue', '20250904_110231', 'glue_latents.h5ad')
glue_adata = sc.read_h5ad(glue_path)
glue_adata.obs = glue_adata.obs.merge(ordinal_pseudotime_adata, left_index=True, right_index=True)


#%% PAGA
from scipy.stats import pearsonr, spearmanr, kendalltau
from scib.metrics.lisi import clisi_graph, ilisi_graph, nmi # install scib from github, see scib documentation for details
from scib.metrics import nmi, ari

def paga_analysis(adata):

    ## graph construction
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata)

    ## UMAP
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
    sc.pl.umap(adata, color=['leiden', 'modality', cell_group, 'ordinal_pseudotime', dev_group_key])

    ## PAGA
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, color=["leiden", "modality", "Lineage", "ordinal_pseudotime"])

    ## Graph based on PAGA
    sc.tl.draw_graph(adata, init_pos='paga')
    #sc.pl.draw_graph(adata, color=['modality', 'Lineage', 'ordinal_pseudotime', dev_group_key]) # plot generated later too

    ## Pseudotime with DPT

    # Set iroot as the cell closest to the centroid of 2nd trimester cells
    mask = adata.obs[dev_group_key] == '2nd trimester'
    X = adata.X if not hasattr(adata, 'obsm') or 'X_pca' not in adata.obsm else adata.obsm['X_pca']
    if hasattr(X, 'toarray'):  # handle sparse matrices
        X = X.toarray()
    centroid = X[mask.values].mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    adata.uns['iroot'] = np.argmin(np.where(mask.values, dists, np.inf))

    # DPT
    sc.tl.dpt(adata)
    sc.pl.draw_graph(adata, color=['modality', cell_group, 'dpt_pseudotime', 'ordinal_pseudotime', dev_group_key])

    return adata

def trajectory_metrics(adata, modality=None):

    if modality:
        adata = adata[adata.obs['modality'].str.upper() == modality.upper()]

    pt_adata = adata.obs[[dev_group_key, 'ordinal_pseudotime', 'dpt_pseudotime']]
    pt_adata[dev_group_key] = pt_adata[dev_group_key].cat.codes.to_numpy()

    pearson_corr_matrix = pt_adata.corr(method=lambda x, y: pearsonr(x, y)[0])
    spearman_corr_matrix = pt_adata.corr(method=lambda x, y: spearmanr(x, y)[0])
    kendall_corr_matrix = pt_adata.corr(method=lambda x, y: kendalltau(x, y)[0])

    metrics_adata = pd.concat([
        pearson_corr_matrix['dpt_pseudotime'].iloc[:-1],
        spearman_corr_matrix['dpt_pseudotime'].iloc[:-1],
        kendall_corr_matrix['dpt_pseudotime'].iloc[:-1],
    ], axis=1)
    metrics_adata.columns = ['pearson', 'spearman', 'kendall']

    ## integration metrics
    lineage_clisi = clisi_graph(adata, label_key='Lineage', type_='knn', scale=True, n_cores=1)
    modality_ilisi = ilisi_graph(adata, batch_key='modality', type_='knn', scale=True, n_cores=1)
    lineage_nmi = nmi(adata, cluster_key='leiden', label_key='Lineage')
    lineage_ari = ari(adata, cluster_key='leiden', label_key='Lineage')
    age_range_nmi = nmi(adata, cluster_key='leiden', label_key='Age_Range')
    age_range_ari = ari(adata, cluster_key='leiden', label_key='Age_Range')
    integration_adata = pd.DataFrame(
        np.stack([lineage_clisi, modality_ilisi, lineage_nmi, lineage_ari, age_range_nmi, age_range_ari])[None],
        columns=['lineage_clisi', 'modality_ilisi', 'lineage_nmi', 'lineage_ari', 'age_range_nmi', 'age_range_ari'])

    return metrics_adata, integration_adata

def trajectory_metrics_all(metrics_dfs, methods_list, suptitle=None):

    metrics_df = pd.concat([df.T.stack().to_frame() for df in metrics_dfs], axis=1)
    metrics_df.columns = methods_list

    metrics_df_melted = metrics_df.reset_index().melt(id_vars=['level_0', 'level_1'], var_name='method', value_name='correlation')
    metrics_df_melted = metrics_df_melted.rename(columns={'level_0': 'metric', 'level_1': 'reference'})

    g = sns.catplot(
        data=metrics_df_melted,
        x='reference',
        y='correlation',
        col='metric',
        hue='method',
        kind='bar',
        height=5,
        aspect=.25
    )
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)

    if suptitle:
        plt.suptitle(suptitle)

    plt.tight_layout()

    return metrics_df

def integration_metrics_all(integration_dfs, methods_list, suptitle=None):

    integration_df = pd.concat(integration_dfs, axis=0)
    integration_df.index = methods_list

    integration_df.drop(columns=['lineage_clisi'], inplace=True)

    integration_df_melted = integration_df.reset_index().melt(id_vars=['index'], var_name='method', value_name='score')
    integration_df_melted.rename(columns={'index': 'method', 'method': 'metric'}, inplace=True)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=integration_df_melted, x='metric', y='score', hue='method')
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
    # Place legend outside the main figure
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Method')

    return integration_df


#eclare_adata = paga_analysis(eclare_adata)
subsampled_eclare_adata = paga_analysis(subsampled_eclare_adata)
scJoint_adata = paga_analysis(scJoint_adata)
glue_adata = paga_analysis(glue_adata)

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

#%% draw graphs for all methods
sc.pl.draw_graph(subsampled_eclare_adata[np.random.permutation(len(subsampled_eclare_adata))], color=['modality','Lineage','Age_Range'], wspace=0.5)
sc.pl.draw_graph(scJoint_adata[np.random.permutation(len(scJoint_adata))], color=['modality','Lineage','Age_Range'], wspace=0.5)
sc.pl.draw_graph(glue_adata[np.random.permutation(len(glue_adata))], color=['modality','Lineage','Age_Range'], wspace=0.5)

#%% trajectory analysis metrics

methods_list = ['ECLARE', 'scJoint', 'scGLUE']

## multi-modal
sub_eclare_metrics, sub_eclare_integration = trajectory_metrics(subsampled_eclare_adata)
scJoint_metrics, scJoint_integration = trajectory_metrics(scJoint_adata)
glue_metrics, glue_integration = trajectory_metrics(glue_adata)
trajectory_metrics_all([sub_eclare_metrics, scJoint_metrics, glue_metrics], methods_list, suptitle='Multi-modal')
integration_metrics_all([sub_eclare_integration, scJoint_integration, glue_integration], methods_list, suptitle='Multi-modal')

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




#%% create PHATE embeddings
import phate

phate_op = phate.PHATE(
    k=50,                # denser graph
    n_pca=10,            # moderate denoising
    verbose=True
)

phate_rna = phate_op.fit_transform(student_rna_latents.detach().cpu().numpy())
phate_atac = phate_op.fit_transform(student_atac_latents.detach().cpu().numpy())

#%% plot PHATE embeddings
import matplotlib.pyplot as plt

dev_labels = student_atac.obs[dev_group_key]
cmap_colors = plt.get_cmap('plasma', len(dev_labels.unique()))
cmap = {dev_labels.unique()[i]: cmap_colors(i) for i in range(len(dev_labels.unique()))}
phate.plot.scatter2d(phate_op, c=dev_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

#%% train UMAP on subsampled data
import seaborn as sns
from eclare.post_hoc_utils import plot_umap_embeddings

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

#%% project select ATAC and RNA latents through UMAP

student_atac_latents_umap = umap_embedder.transform(student_atac_latents.detach().cpu().numpy())
student_rna_latents_umap = umap_embedder.transform(student_rna_latents.detach().cpu().numpy())

dev_rna = student_rna_keep.obs[dev_group_key].values
dev_atac = student_atac_keep.obs[dev_group_key].values

ct_rna = student_rna_keep.obs[cell_group].values
ct_atac = student_atac_keep.obs[cell_group].values

mod_atac = ['ATAC'] * len(student_atac_latents)

student_latents_umap = np.concatenate([student_rna_latents_umap, student_atac_latents_umap], axis=0)
dev = np.concatenate([dev_rna, dev_atac])
ct = np.concatenate([ct_rna, ct_atac])
modality = np.concatenate([['RNA'] * len(student_rna_latents), ['ATAC'] * len(student_atac_latents)])
cmap_mod = { 'RNA': 'tab:blue', 'ATAC': 'tab:orange' }

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

sns.scatterplot(x=student_atac_latents_umap[:, 0], y=student_atac_latents_umap[:, 1], hue=dev_atac, palette=cmap_dev, marker='.', hue_order=dev_stages, ax=ax[0,0])
sns.scatterplot(x=student_latents_umap[:, 0], y=student_latents_umap[:, 1], hue=dev, palette=cmap_dev, marker='.', hue_order=dev_stages, ax=ax[1,0])
ax[0,0].set_title('ATAC'); ax[1,0].set_title('ATAC & RNA')
ax[0,0].xaxis.set_visible(False); ax[0,0].yaxis.set_visible(False); ax[1,0].xaxis.set_visible(False); ax[1,0].yaxis.set_visible(False)

sns.scatterplot(x=student_atac_latents_umap[:, 0], y=student_atac_latents_umap[:, 1], hue=ct_atac, palette=cmap_ct, marker='.', ax=ax[0,1])
sns.scatterplot(x=student_latents_umap[:, 0], y=student_latents_umap[:, 1], hue=ct, palette=cmap_ct, marker='.', ax=ax[1,1])
ax[0,1].set_title('ATAC'); ax[1,1].set_title('ATAC & RNA')
ax[0,1].xaxis.set_visible(False); ax[0,1].yaxis.set_visible(False); ax[1,1].xaxis.set_visible(False); ax[1,1].yaxis.set_visible(False)

sns.scatterplot(x=student_atac_latents_umap[:, 0], y=student_atac_latents_umap[:, 1], hue=mod_atac, palette=cmap_mod, marker='.', ax=ax[0,2])
sns.scatterplot(x=student_latents_umap[:, 0], y=student_latents_umap[:, 1], hue=modality, palette=cmap_mod, marker='.', ax=ax[1,2])
ax[0,2].set_title('ATAC'); ax[1,2].set_title('ATAC & RNA')
ax[0,2].xaxis.set_visible(False); ax[0,2].yaxis.set_visible(False); ax[1,2].xaxis.set_visible(False); ax[1,2].yaxis.set_visible(False)
