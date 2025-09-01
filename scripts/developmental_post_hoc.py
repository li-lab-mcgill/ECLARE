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
device = torch.device(f'cuda:{n_cudas - 1}') if cuda_available else 'cpu'

## Define target and source datasets
target_dataset = 'Cortex_Velmeshev'
source_datasets = ['PFC_V1_Wang', 'PFC_Zhu']
subsample = 2000

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '25165730',
    'kd_clip': '25173640',
    'eclare': ['25182625'],
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

student_rna, student_atac, cell_group, dev_group_key, dev_stages = student_setup_func(args, return_backed=True)

#macroglia_cell_types = ['OL', 'OPC', 'GLIALPROG', 'AST']
#keep_rna = student_rna.obs[cell_group].isin(macroglia_cell_types) & student_rna.obs[dev_group_key].str.contains('3rd trimester')
#keep_atac = student_atac.obs[cell_group].isin(macroglia_cell_types)

keep_rna = student_rna.obs[cell_group].str.contains('ExNeu')
keep_atac = student_atac.obs[cell_group].str.contains('ExNeu')

student_rna_keep = student_rna[keep_rna].to_memory()
student_atac_keep = student_atac[keep_atac].to_memory()

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

#student_model = deepcopy(eclare_student_model)
student_model = deepcopy(kd_clip_student_models['PFC_V1_Wang'])

## project data through student model
student_rna_cells = torch.from_numpy(student_rna_keep.X.toarray().astype(np.float32))
student_atac_cells = torch.from_numpy(student_atac_keep.X.toarray().astype(np.float32))

student_rna_latents, _ = student_model.to('cpu')(student_rna_cells, modality=0)
student_atac_latents, _ = student_model.to('cpu')(student_atac_cells, modality=1)

## subsample data
student_rna_sub = subsample_adata(student_rna, cell_group, dev_group_key, subsample, subsample_type='stratified')
student_atac_sub = subsample_adata(student_atac, cell_group, dev_group_key, subsample, subsample_type='stratified')

## get subset data latents for retraining UMAP
student_rna_cells_sub = torch.from_numpy(student_rna_sub.X.toarray().astype(np.float32))
student_atac_cells_sub = torch.from_numpy(student_atac_sub.X.toarray().astype(np.float32))

student_rna_latents_sub, _ = student_model.to('cpu')(student_rna_cells_sub, modality=0)
student_atac_latents_sub, _ = student_model.to('cpu')(student_atac_cells_sub, modality=1)

cmap_dev = plt.get_cmap('plasma', len(dev_stages))
cmap_dev = {dev_stages[i]: cmap_dev(i) for i in range(len(dev_stages))}
cmap_ct = create_celltype_palette(student_rna_sub.obs[cell_group].values, student_atac_sub.obs[cell_group].values, plot_color_palette=False)


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

#%% Create ECLARE adata
import scanpy as sc

X = np.vstack([
    student_rna_latents_sub.detach().cpu().numpy(),
    student_atac_latents_sub.detach().cpu().numpy()
    ])

obs = pd.concat([
    student_rna_sub.obs.assign(modality='RNA'),
    student_atac_sub.obs.assign(modality='ATAC')
    ])

eclare_adata = sc.AnnData(
    X=X,
    obs=obs,
)

## create unimodal ECLARE adatas
rna_adata = sc.AnnData(
    X=student_rna_latents_sub.detach().cpu().numpy(),
    obs=student_rna_sub.obs.assign(modality='RNA'),
)

atac_adata = sc.AnnData(
    X=student_atac_latents_sub.detach().cpu().numpy(),
    obs=student_atac_sub.obs.assign(modality='ATAC'),
)

#%% PAGA

def paga_analysis(adata):

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=1.5)

    ## UMAP
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[cell_group, 'modality', dev_group_key])

    #sc.tl.draw_graph(eclare_adata)
    #sc.pl.draw_graph(eclare_adata, color=[cell_group, 'modality', dev_group_key])

    ## PAGA
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, color=["leiden", "modality", "Lineage"])

    ## Graph based on PAGA
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color=['Lineage', 'modality', dev_group_key])

    ## Pseudotime with DPT
    adata.uns['iroot'] = np.flatnonzero(adata.obs[dev_group_key] == '2nd trimester')[0]
    sc.tl.dpt(adata)
    sc.pl.draw_graph(adata, color=['dpt_pseudotime', dev_group_key, cell_group, 'modality'])

paga_analysis(eclare_adata)

paga_analysis(rna_adata)
paga_analysis(atac_adata)

student_rna_sub.obs['modality'] = 'RNA'
student_atac_sub.obs['modality'] = 'ATAC'

paga_analysis(student_rna_sub)
paga_analysis(student_atac_sub)


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

dev_labels = student_atac_keep.obs[dev_group_key]
cmap_colors = plt.get_cmap('plasma', len(dev_labels.unique()))
cmap = {dev_labels.unique()[i]: cmap_colors(i) for i in range(len(dev_labels.unique()))}
phate.plot.scatter2d(phate_op, c=dev_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

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

