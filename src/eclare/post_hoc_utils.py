import os
import sys
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from umap import UMAP
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import ot
from ot import solve as ot_solve
import scanpy as sc
import scglue
import SEACells
import mlflow
import mlflow.pytorch
from mlflow.models import Model
import anndata
from mlflow.tracking import MlflowClient
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from collections import defaultdict
from IPython.display import display
from tqdm import tqdm
from scipy.stats import linregress, norm, gamma
from scipy.optimize import minimize
import networkx as nx
import gseapy as gp
import mygene
from venn import venn
from datetime import datetime
import subprocess

from eclare.models import load_CLIP_model
from eclare.setup_utils import return_setup_func_from_dataset, mdd_setup
from eclare.eval_utils import unpaired_metrics
from eclare.data_utils import get_scompreg_loglikelihood, filter_mean_grn, scompreg_likelihood_ratio, get_scompreg_loglikelihood_full


def get_model_and_data(model_path, load_mdd=False):

    ## Load the model
    model, model_args_dict = load_CLIP_model(model_path, device='cpu')

    ## Determine the dataset
    dataset = model_args_dict['args'].dataset
    setup_func = return_setup_func_from_dataset(dataset)

    ## Load the dataset
    source_rna, source_atac, source_cell_group, _, _, source_atac_fullpath, source_rna_fullpath = setup_func(model.args, pretrain=None, return_type='data')

    if load_mdd:
        mdd_rna, mdd_atac, mdd_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, _, _ = mdd_setup(model.args, pretrain=None, return_type='data',\
            overlapping_subjects_only=False)
        return model, source_rna, source_atac, mdd_rna, mdd_atac
    
    else:
        return model, source_rna, source_atac


def sample_proportional_celltypes_and_condition(rna, atac, \
    rna_celltype_key='ClustersMapped', atac_celltype_key='ClustersMapped', \
    rna_condition_key='Condition', atac_condition_key='condition', \
    batch_size=500, paired=False):

    rna_celltypes = rna.obs[rna_celltype_key].values
    atac_celltypes = atac.obs[atac_celltype_key].values

    if rna_condition_key is None or atac_condition_key is None:
        rna_condition = [''] * len(rna)
        atac_condition = [''] * len(atac)
    else:
        rna_condition = rna.obs[rna_condition_key].values
        atac_condition = atac.obs[atac_condition_key].values

    rna_celltypes_and_condition = [ rna_celltype + '_' + rna_condition for rna_celltype, rna_condition in zip(rna_celltypes, rna_condition) ]
    atac_celltypes_and_condition = [ atac_celltype + '_' + atac_condition for atac_celltype, atac_condition in zip(atac_celltypes, atac_condition) ]

    ## sample data proportional to cell types and condition
    if paired:
        skf = StratifiedShuffleSplit(n_splits=1, test_size=batch_size)
        _, indices = next(skf.split(rna.X, rna_celltypes_and_condition))
        rna_indices = atac_indices = indices
    else:
        rna_skf = StratifiedShuffleSplit(n_splits=1, test_size=batch_size)
        _, rna_indices = next(rna_skf.split(rna.X, rna_celltypes_and_condition))
        
        atac_skf = StratifiedShuffleSplit(n_splits=1, test_size=batch_size)
        _, atac_indices = next(atac_skf.split(atac.X, atac_celltypes_and_condition))

    rna_sampled = rna[rna_indices].copy()
    atac_sampled = atac[atac_indices].copy()

    rna_celltypes = np.asarray(rna_celltypes)[rna_indices]
    atac_celltypes = np.asarray(atac_celltypes)[atac_indices]

    rna_condition = np.asarray(rna_condition)[rna_indices]
    atac_condition = np.asarray(atac_condition)[atac_indices]

    return rna_sampled, atac_sampled, rna_celltypes, atac_celltypes, rna_condition, atac_condition


def get_latents(model, rna, atac, return_tensor=False):

    device = next(model.parameters()).device

    ## convert to torch
    rna = torch.from_numpy(rna.X.toarray()).float().to(device)
    atac = torch.from_numpy(atac.X.toarray()).float().to(device)

    #with torch.inference_mode():
    rna_latent = model(rna, modality=0)[0].detach()
    atac_latent = model(atac, modality=1)[0].detach()

    ## normalize
    #rna = torch.nn.functional.normalize(rna, p=2, dim=1)
    #atac = torch.nn.functional.normalize(atac, p=2, dim=1)

    if return_tensor:
        rna_latent, atac_latent = rna_latent.detach(), atac_latent.detach()
    else:
        rna_latent, atac_latent = rna_latent.detach().cpu().numpy(), atac_latent.detach().cpu().numpy()

    return rna_latent, atac_latent


def create_celltype_palette(all_rna_celltypes, all_atac_celltypes, plot_color_palette=True):
    all_cell_types = np.unique(np.hstack([all_rna_celltypes, all_atac_celltypes]))

    if len(all_cell_types) >= 12:
        palette = sns.color_palette("tab20", len(all_cell_types))
    elif (len(all_cell_types) == 11):
        palette = sns.color_palette("bright", len(all_cell_types))
        #palette[-1] = (0.3333333333333333, 0.4196078431372549, 0.1843137254901961) # change last color to dark olive green using rgb scheme
        palette = sns.color_palette("tab20", len(all_cell_types))
    elif (len(all_cell_types) < 11) and (len(all_cell_types) >= 9):
        palette = sns.color_palette("Set1", len(all_cell_types))
    elif (len(all_cell_types) <= 8):
        palette = sns.color_palette("Dark2", len(all_cell_types))

    color_map = dict(zip(all_cell_types, palette))

    if plot_color_palette:

        fig, ax = plt.subplots(figsize=(2, 8))
        for i, (name, color) in enumerate(color_map.items()):
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
            ax.text(1.1, i + 0.5, name, va='center', ha='left', fontsize=12)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, len(color_map))
        ax.axis('off')
        plt.show()

    return color_map


def plot_umap_embeddings(rna_latents, atac_latents, rna_celltypes, atac_celltypes, rna_condition, atac_condition, color_map_ct, title=None, marker='.', umap_embedding=None, save_path=None):

    ## train UMAP embedding on project RNA and ATAC latents separately
    if umap_embedding is None:
        rna_atac_latents = np.concatenate([rna_latents, atac_latents], axis=0)
        umap_embedding = UMAP(n_neighbors=50, min_dist=0.5, n_components=2, metric='cosine', random_state=42)
        umap_embedding.fit(rna_atac_latents)

    rna_umap = umap_embedding.transform(rna_latents)
    atac_umap = umap_embedding.transform(atac_latents)

    ## save UMAP embeddings to dataframe
    rna_df_umap = pd.DataFrame(data={'umap_1': rna_umap[:, 0], 'umap_2': rna_umap[:, 1], 'celltypes': rna_celltypes, 'condition': rna_condition, 'modality': 'RNA'})
    atac_df_umap = pd.DataFrame(data={'umap_1': atac_umap[:, 0], 'umap_2': atac_umap[:, 1], 'celltypes': atac_celltypes, 'condition': atac_condition, 'modality': 'ATAC'})
    rna_atac_df_umap = pd.concat([rna_df_umap, atac_df_umap], axis=0).sample(frac=1) # shuffle

    ## check if multiple conditions are present, in which case we will plot the condition on the fourth subplot
    multiple_conditions = (len(np.unique(rna_condition)) > 1) or (len(np.unique(atac_condition)) > 1)

    if multiple_conditions:
        hue_order = np.unique(np.concatenate([rna_condition, atac_condition]))

        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='condition', hue_order=hue_order, alpha=0.5, ax=ax[3], legend=True, marker=marker)
        ax[3].set_xticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xlabel(''); ax[3].set_ylabel('')
    else:
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ## check if multiple cell types are present
    multiple_celltypes = (len(color_map_ct) > 1)

    if multiple_celltypes:
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='celltypes', palette=color_map_ct, alpha=0.8, ax=ax[1], legend=False, marker=marker)
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, ax=ax[2], legend=True, marker=marker)

        ax[1].set_xticklabels([]); ax[1].set_yticklabels([]); ax[2].set_xticklabels([]); ax[2].set_yticklabels([])
        ax[1].set_xlabel(''); ax[1].set_ylabel(''); ax[2].set_xlabel(''); ax[2].set_ylabel('')

        ## add cell-type legend
        ax[0].set_xlim(0, 3)
        ax[0].set_ylim(-0.5, len(color_map_ct) - 0.5)
        for i, (name, color) in enumerate(reversed(color_map_ct.items())):  # Reverse order to match plot
            rect_height = 0.8
            rect_y = i - rect_height/2  # Center the rectangle vertically
            ax[0].add_patch(plt.Rectangle((0, rect_y), 0.5, rect_height, color=color))
            ax[0].text(0.7, i, name, va='center', ha='left', fontsize=10)
        ax[0].axis('off')

        ax[0].set_title('legend: cell types')
        ax[1].set_title('labelled by cell type')
        ax[2].set_title('labelled by modality')

    elif not multiple_celltypes: # plot ATAC and RNA separately, then plot both in overlay
        sns.scatterplot(data=atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], ax=ax[0], marker=marker, legend=False)
        sns.scatterplot(data=rna_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], ax=ax[1], marker=marker, legend=False)
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, ax=ax[2], legend=True, marker=marker)

        ax[0].set_xticklabels([]); ax[0].set_yticklabels([]); ax[1].set_xticklabels([]); ax[1].set_yticklabels([])
        ax[0].set_xlabel(''); ax[0].set_ylabel(''); ax[1].set_xlabel(''); ax[1].set_ylabel('')
        ax[2].set_xticklabels([]); ax[2].set_yticklabels([]); ax[2].set_xlabel(''); ax[2].set_ylabel('')
        ax[0].set_title('ATAC'); ax[1].set_title('RNA'); ax[2].set_title('ATAC & RNA')

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return umap_embedding, fig, rna_atac_df_umap


def ot_pairing(rna_latent, atac_latent):

    rna_latent_tensor = torch.from_numpy(rna_latent).float()
    atac_latent_tensor = torch.from_numpy(atac_latent).float()

    rna_latent_norm = torch.nn.functional.normalize(rna_latent_tensor, p=2, dim=1)
    atac_latent_norm = torch.nn.functional.normalize(atac_latent_tensor, p=2, dim=1)

    a = np.ones((atac_latent.shape[0],)) / atac_latent.shape[0]
    b = np.ones((rna_latent.shape[0],)) / rna_latent.shape[0]
    distance = torch.cdist(atac_latent_norm, rna_latent_norm, p=2).detach().cpu().numpy()
    assert (np.isnan(distance).sum() == 0), 'Distance matrix contains NaN values'

    transport_matrix = ot.emd(a, b, distance)
    transport_matrix = transport_matrix.astype(rna_latent.dtype)
    atac_latent_transported = np.dot( (len(a) * transport_matrix) , rna_latent)

    return atac_latent_transported, transport_matrix, distance



def match_analyses(similarities, datasets):

    # similarities = cosine_mdds

    plans = [ot.solve(similarity).plan for similarity in similarities]
    similarities_masked = [similarity[plan.T>0] for similarity, plan in zip(similarities, plans)]
    similarities_masked = torch.stack(similarities_masked).T

    similarities_masked_idxs = [np.where(plan.T>0) for plan in plans]
    similarities_masked_idxs = [list(zip(idx[0].tolist(), idx[1].tolist())) for idx in similarities_masked_idxs]
    similarities_masked_idxs = [item for idxs in similarities_masked_idxs for item in idxs]

    similarities_row_softmax = [torch.nn.functional.softmax(similarity, dim=1) for similarity in similarities]
    similarities_row_softmax_masked = [similarity[plan.T>0] for similarity, plan in zip(similarities_row_softmax, plans)]
    similarities_row_softmax_masked = torch.stack(similarities_row_softmax_masked).T

    similarities_col_softmax = [torch.nn.functional.softmax(similarity, dim=0) for similarity in similarities]
    similarities_col_softmax_masked = [similarity[plan.T>0] for similarity, plan in zip(similarities_col_softmax, plans)]
    similarities_col_softmax_masked = torch.stack(similarities_col_softmax_masked).T

    pd.DataFrame(similarities_masked, columns=datasets).hist(sharex=True, sharey=True, figsize=[5,5])
    plt.suptitle('Cosine similarities between MDD RNA and ATAC latent nuclei \n masked by optimal transport plan')
    plt.tight_layout()

    pd.DataFrame(similarities_row_softmax_masked, columns=datasets).hist(sharex=True, sharey=True, figsize=[6,6])
    plt.suptitle('Row-softmax cosine similarities between MDD RNA and ATAC latent nuclei \n masked by optimal transport plan')
    plt.tight_layout()

    pd.DataFrame(similarities_col_softmax_masked, columns=datasets).hist(sharex=True, sharey=True, figsize=[5,5])
    plt.suptitle('Column-softmax cosine similarities between MDD RNA and ATAC latent nuclei \n masked by optimal transport plan')
    plt.tight_layout()

    sort_indices_1d = torch.flip(similarities_masked.flatten().argsort(stable=True), dims=(0,))
    similarities_masked_idxs_sorted = [similarities_masked_idxs[i] for i in sort_indices_1d]

    ## OT on mean cosine similarities across datasets
    similarities_mean = torch.stack(similarities).mean(0)
    plan_mean = ot.solve(similarities_mean).plan

    # Sets to track which row and column indices have already been seen
    seen_rows = set()
    seen_cols = set()

    # List to store the selected indices
    trimmed_indices = []

    # Iterate through the sorted indices
    for row, col in similarities_masked_idxs_sorted:
        # If the row and column haven't been seen, add the pair to trimmed_indices
        if row not in seen_rows and col not in seen_cols:
            trimmed_indices.append([row, col])
            seen_rows.add(row)
            seen_cols.add(col)

    assert len(trimmed_indices) == len(plans[0]), 'Number of selected indices does not match number of nuclei in each modality'


def ECLARE_loss_and_metrics_plot(clip_job_id, ignore_losses=['cLISI'], remove_MDD=True):

    data = pd.read_csv(os.path.join(os.environ['OUTPATH'], f'multisource_align_{clip_job_id}', 'training_log.csv', index_col=0))
    data = data.loc[:,~data.isna().all()]
    data = data.loc[:, ~data.columns.str.contains('|'.join(ignore_losses))] if len(ignore_losses) > 0 else data
    data = data.ffill()

    if remove_MDD:
        print('Removing MDD losses (but keeping mdd)')
        data = data.loc[:, ~data.columns.str.contains('MDD')]

    kws = pd.unique([strings[0] for strings in data.columns.str.split('-')])
    kws.sort()
    #kws = [ kw for kw in kws if not (np.isin(kw, ['Loss_MMD', 'Loss_CLIP']).item()) ]

    if len(kws) <= 3:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    elif len(kws) == 4:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    elif len(kws) > 4 and len(kws) <= 6:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    elif len(kws) > 6 and len(kws) <= 9:
        fig, ax = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    elif len(kws) > 9 and len(kws) <= 12:
        fig, ax = plt.subplots(3, 4, figsize=(15, 12), sharex=True)

    for i, kw in enumerate(kws):
        data_kw = data[[col for col in data.columns if col.startswith(f'{kw}-')]]

        row, col = i // np.max(ax.shape), i % np.max(ax.shape)
        data_kw.plot(ax=ax[row, col], title=kw, legend=False, marker='.')

        if np.isin('Loss',kws).item() and (kw == 'Loss'):
            ax[row, col].legend()
        elif (row == 0) and (col == 0):
            ax[row, col].legend()

    fig.tight_layout()
    fig.show()


def CLIP_loss_and_metrics_plot(clip_job_id):

    metrics_paths = glob(os.path.join(os.environ['OUTPATH'], f'clip_{clip_job_id}/**/**/**/metrics.csv'))

    for metrics_path in metrics_paths:
        metrics = pd.read_csv(metrics_path)

        fig, ax = plt.subplots(1,4, figsize=[16,5])
        metrics[['clip_loss', 'clip_loss_censored']].plot(marker='.', ax=ax[0])
        metrics[['recon_loss_rna','recon_loss_atac']].plot(marker='.', legend=True, ax=ax[1])
        metrics[['acc','acc_top5']].plot(marker='.', legend=True, ax=ax[2])
        metrics['foscttm'].plot(marker='.', legend=True, ax=ax[3])
        plt.suptitle(f'CLIP job id: {clip_job_id}')
        plt.tight_layout(); plt.show()

def get_metrics(method, job_id, target_only=False):

    method_job_id = f'{method}_{job_id}'
    paths_root = os.path.join(os.environ['OUTPATH'], method_job_id)

    ## retain leaf directories only
    paths, all_data_source, all_data_target = [], [], []
    paths = glob(os.path.join(paths_root, '**', '**', '**', f'*_metrics_target_valid.csv'))

    ## For scMulticlip, will not find metrics with previous command
    if paths == []:
        paths = glob(os.path.join(paths_root, '**', '**', f'*_metrics_target_valid.csv'))
    #for dirpath, dirnames, filenames in os.walk(paths_root): paths.append(dirpath) if not dirnames else None

    paths = sorted(paths)

    for path in paths:

        path = os.path.dirname(path)

        ## Get source and target dataset names
        path_split = path.split('/')
        method_job_id_idx = np.where([split==method_job_id for split in path_split])[0][0]

        target = path_split[method_job_id_idx + 1]
        source = path_split[method_job_id_idx + 2] if len(path_split) > method_job_id_idx + 2 else None
        if source in ['0', '1', '2']: source = None

        ## Read metrics
        try:
            metrics_target_valid = pd.read_csv(glob(os.path.join(path, f'*_metrics_target_valid.csv'))[0], index_col=0)
            metrics_source_valid = None if target_only else pd.read_csv(glob(os.path.join(path, f'*_metrics_source_valid.csv'))[0], index_col=0)
        except:
            print(f'Error reading {path}')
            continue

        ## Drop foscttm_score_ct & rank_score
        if 'foscttm_score_ct' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['foscttm_score_ct'])
            if ('foscttm_score_ct' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['foscttm_score_ct'])

        if 'rank_score' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['rank_score'])
            if ('rank_score' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['rank_score'])

        ## Transpose, such that metrics as columns rather than indices
        metrics_target_valid = metrics_target_valid.T
        if not target_only:
            metrics_source_valid = metrics_source_valid.T

        ## Add target dataset names, then append to all_data
        metrics_target_valid['target'] = target
        metrics_target_valid['source'] = source
        all_data_target.append(metrics_target_valid)

        if not target_only:
            metrics_source_valid['target'] = target
            metrics_source_valid['source'] = source
            all_data_source.append(metrics_source_valid)

    ## Concatenate all_data and set multi-index
    target_metrics_valid_df = pd.concat(all_data_target).set_index(['source', 'target'])
    target_metrics_valid_df = target_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in target_metrics_valid_df.columns:
        target_metrics_valid_df['1 - foscttm_score'] = 1 - target_metrics_valid_df['foscttm_score']
        target_metrics_valid_df = target_metrics_valid_df.drop(columns=['foscttm_score'])

    if target_only:
        return target_metrics_valid_df

    ## --- SOURCE --- ##
    source_metrics_valid_df = pd.concat(all_data_source).set_index(['source', 'target'])
    source_metrics_valid_df = source_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in source_metrics_valid_df.columns:
        source_metrics_valid_df['1 - foscttm_score'] = 1 - source_metrics_valid_df['foscttm_score']
        source_metrics_valid_df = source_metrics_valid_df.drop(columns=['foscttm_score'])

    ## Create source-only metrics dataframe, and sort according to existing order of target in original df
    source_only_metrics_valid_df = source_metrics_valid_df.reset_index(level='target').drop(columns='target')
    source_only_metrics_valid_df = source_only_metrics_valid_df.loc[source_metrics_valid_df.index.get_level_values(0).unique().values]

    return source_metrics_valid_df, target_metrics_valid_df, source_only_metrics_valid_df

def compare_dataframes_target(dataframes, value_column, dataset_labels, target_source_combinations, ax=None):

    dataset_label_mapper = {
        'AD_Anderson_et_al': 'DLPFC_Anderson',
        'PD_Adams_et_al': 'Midbrain_Adams',
        'human_dlpfc': 'DLPFC_Ma',
        'roussos': 'PFC_Zhu',
        'mdd': 'MDD'
    }

    hue_order = list(dataset_label_mapper.values())

    combined_df = pd.concat(
        [df.assign(dataset=label).rename(index=dataset_label_mapper)
         for df, label in zip(dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(combined_df["target"].unique())

    colors = sns.color_palette("Dark2", 7)
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    #target_to_color = {target: colors[i] if len(unique_targets)>1 else 'black' for i, target in enumerate(unique_targets)}
    if len(unique_targets) > 1:
        target_to_color = {hue_order[i]: colors[i] if len(unique_targets)>1 else 'black' for i in range(len(unique_targets))}
    elif unique_targets[0] == 'MDD':
        target_to_color = {unique_targets[0]: colors[6]}

    if target_source_combinations:
        source_to_marker = {source: markers[i % len(markers)] for i, source in enumerate(unique_sources)}
    else:
        source_to_marker = {source: 'o' for i, source in enumerate(unique_sources)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    ).tick_params(axis='x', rotation=30)
    ax.set_xlabel("method")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for target in unique_targets:
        for dataset in dataset_positions:
            for source in unique_sources:

                if pd.isna(source):
                    subset = combined_df[(combined_df["target"] == target) &
                                        (combined_df["source"].isna()) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker='o',
                        edgecolor=None,
                        s=50,
                        zorder=10,
                        alpha=0.4
                    )

                else:
                    subset = combined_df[(combined_df["source"] == source) &
                                        (combined_df["target"] == target) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker=source_to_marker[source],
                        edgecolor=None,
                        s=50 if source_to_marker[source] == '*' else 50,
                        zorder=10,
                        alpha=0.4
                    )

    return source_to_marker, target_to_color
    
def combined_plot(target_dataframes, value_columns, dataset_labels, target_source_combinations=False, source_only_dataframes=None, figsize=(6,6)):

    # Create a figure with two subplots
    if source_only_dataframes is None:
        fig, axs = plt.subplots(1, len(value_columns), figsize=figsize, sharex=True, squeeze=False)
    else:
        fig, axs = plt.subplots(2, len(value_columns), figsize=figsize, sharey='col')

    for value_column, ax in zip(value_columns, axs[-1]):
        # Pass axes to the individual plotting functions
        source_to_marker, target_to_color = compare_dataframes_target(target_dataframes, value_column, dataset_labels, target_source_combinations, ax=ax)

    # Create handles for color-coding by target
    color_handles = [
        plt.Line2D([0], [0], marker='D' if target_source_combinations else 'o', color=color, label=f"{target}", linestyle='None', markersize=10)
        for target, color in target_to_color.items()
    ]

    # Create handles for marker-coding by source
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', label=f"{source}", linestyle='None', markersize=10)
        if not pd.isna(source) else
        plt.Line2D([0], [0], marker='o', color='black', label=f"all sources (ensemble)", linestyle='None', markersize=10)
        for source, marker in source_to_marker.items()
    ]


    # Add labels for color and marker sections
    if ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) > 1) ):

        # Add a blank entry and title for marker section
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles +
            [plt.Line2D([0], [0], color='none', label="")] +  # Blank separator
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +  # Title for the second block
            marker_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()] +
            [''] +  # Blank separator
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )

    elif ( (len(set(source_to_marker.values())) == 1) and ( len(set(target_to_color.keys())) > 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()]
        )
    elif ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) == 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +
            marker_handles
        )
        combined_labels = (
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )
    else:
        combined_handles = None
        combined_labels = None

    axs[-1,-1].legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    # Set a title for the entire figure
    for value_column, ax, letter in zip(value_columns, axs[0], ascii_uppercase):

        ax.set_title(f"{letter}) {value_column}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        if source_only_dataframes is not None:
            source_to_target = compare_dataframes_source_only(source_only_dataframes, value_column, dataset_labels, ax=ax)

            axs[0,0].set_ylabel(f"source to source")
            axs[-1,0].set_ylabel(f"source to target")

            combined_handles = (
                [plt.Line2D([0], [0], color='none', label="Source, by color:")] +
                color_handles
            )
            combined_labels = (
                ['Source, by color:'] +
                [f"{target}" for target in target_to_color.keys()]
            )

            axs[0,-1].legend(
                handles=combined_handles,
                labels=combined_labels,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=False
            )

    plt.tight_layout()
    plt.show()

    return fig


def cell_gap_ot(student_logits, atac_latents, rna_latents, mdd_atac_sampled_group, mdd_rna_sampled_group, cells_gap, type='emd'):

    if type == 'emd':
        res = ot_solve(1 - student_logits)
        plan = res.plan
        value = res.value_linear

        '''
        plan_df = pd.DataFrame(plan.cpu().numpy(), index=mdd_atac_sampled_group.obs[atac_subject_key], columns=mdd_rna_sampled_group.obs[rna_subject_key])
        row_order = plan_df.index.argsort()
        col_order = plan_df.columns.argsort()
        plan_df = plan_df.iloc[row_order, col_order]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(plan_df, ax=ax)
        plt.show()
        '''

    elif type == 'partial':
        a, b = torch.ones((len(student_logits.shape[0]),)) / len(student_logits.shape[0]), torch.ones((len(student_logits.shape[1]),)) / len(student_logits.shape[1])
        mass = 1 - (a[0] * cells_gap).item()
        plan = ot.partial.partial_wasserstein(a, b, 1-student_logits, m=mass)

    if student_logits.shape[0] > student_logits.shape[1]:
        keep_atac_cells = plan.max(1).values.argsort(stable=True)[cells_gap:].sort().values.detach().cpu().numpy()
        atac_latents = atac_latents[keep_atac_cells]
        mdd_atac_sampled_group = mdd_atac_sampled_group[keep_atac_cells]
    else:
        keep_rna_cells = plan.max(0).values.argsort(stable=True)[cells_gap:].sort().values.detach().cpu().numpy()
        rna_latents = rna_latents[keep_rna_cells]
        mdd_rna_sampled_group = mdd_rna_sampled_group[keep_rna_cells]

    student_logits = torch.matmul(atac_latents, rna_latents.T)

    return mdd_atac_sampled_group, mdd_rna_sampled_group, student_logits

def run_SEACells(adata_train, adata_apply, build_kernel_on, redo_umap=False, key='X_umap', n_SEACells=None, save_dir=None):

    # Copy the counts to ".raw" attribute of the anndata since it is necessary for downstream analysis
    # This step should be performed after filtering 
    raw_ad = sc.AnnData(adata_train.X).copy()
    raw_ad.obs_names, raw_ad.var_names = adata_train.obs_names, adata_train.var_names
    adata_train.raw = raw_ad

    raw_ad = sc.AnnData(adata_apply.X).copy()
    raw_ad.obs_names, raw_ad.var_names = adata_apply.obs_names, adata_apply.var_names
    adata_apply.raw = raw_ad

    # Normalize cells, log transform and compute highly variable genes
    #sc.pp.normalize_per_cell(adata_train)
    #sc.pp.log1p(adata_train)
    #sc.pp.highly_variable_genes(adata_train, n_top_genes=1500)

    ## User defined parameters

    ## Core parameters 
    n_SEACells = n_SEACells if n_SEACells is not None else max(adata_train.n_obs // 100, 15) # 75 cells per SEACell, for a minimum of 15 SEACells
    print(f'Number of SEACells: {n_SEACells}')
    n_waypoint_eigs = 15 # Number of eigenvalues to consider when initializing metacells

    ## Initialize SEACells model
    model = SEACells.core.SEACells(
        adata_train, 
        build_kernel_on=build_kernel_on, 
        n_SEACells=n_SEACells, 
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon = 1e-5,
        max_franke_wolfe_iters=100,
        use_gpu=True if torch.cuda.is_available() else False
        )

    model.construct_kernel_matrix()
    M = model.kernel_matrix

    # Initialize archetypes
    model.initialize_archetypes()

    # Fit SEACells model
    model.fit(min_iter=10, max_iter=100)

    ## Get number of cells in each SEACell
    n_cells_per_SEACell = adata_train.obs['SEACell'].value_counts(sort=False)
    assert np.all(n_cells_per_SEACell.index == adata_train.obs['SEACell'].unique()) # unique() is how SEACells gather cells into SEACells

    ## add SEACell labels to apply data
    adata_apply.obs['SEACell'] = adata_train.obs['SEACell'].values

    ## summarize by SEACell
    SEACell_ad_train = SEACells.core.summarize_by_SEACell(adata_train, SEACells_label='SEACell', summarize_layer='raw')
    SEACell_ad_apply = SEACells.core.summarize_by_SEACell(adata_apply, SEACells_label='SEACell', summarize_layer='raw')

    ## divide aggregated SEACell counts by number of cells in each SEACell
    SEACell_ad_train.X = SEACell_ad_train.X.multiply(1 / n_cells_per_SEACell.values[:,None]).tocsr()
    SEACell_ad_apply.X = SEACell_ad_apply.X.multiply(1 / n_cells_per_SEACell.values[:,None]).tocsr()

    if redo_umap:
        #sc.pp.pca(adata, n_comps=15)
        sc.pp.neighbors(adata_train)
        sc.tl.umap(adata_train)

        SEACells.plot.plot_2D(adata_train, key='X_umap', colour_metacells=False)
        SEACells.plot.plot_2D(adata_train, key='X_umap', colour_metacells=True, save_as=os.path.join(save_dir, f'seacells_umap.png'))

    else:
        SEACells.plot.plot_2D(adata_train, key=key, colour_metacells=False)
        SEACells.plot.plot_2D(adata_train, key=key, colour_metacells=True, save_as=os.path.join(save_dir, f'seacells_umap.png'))

    ## add proportion of cells per SEACell
    SEACell_ad_train.obs['proportion_of_cells'] = n_cells_per_SEACell# / n_cells_per_SEACell.sum()
    SEACell_ad_apply.obs['proportion_of_cells'] = n_cells_per_SEACell# / n_cells_per_SEACell.sum()

    ## remove raw attribute
    adata_train.raw = None
    adata_apply.raw = None

    return SEACell_ad_train, SEACell_ad_apply


def get_metrics(method, job_id, target_only=False):

    method_job_id = f'{method}_{job_id}'
    paths_root = os.path.join(os.environ['OUTPATH'], method_job_id)

    ## retain leaf directories only
    paths, all_data_source, all_data_target = [], [], []
    paths = glob(os.path.join(paths_root, '**', '**', '**', f'*_metrics_target_valid.csv'))

    ## For scMulticlip, will not find metrics with previous command
    if paths == []:
        paths = glob(os.path.join(paths_root, '**', '**', f'*_metrics_target_valid.csv'))
    #for dirpath, dirnames, filenames in os.walk(paths_root): paths.append(dirpath) if not dirnames else None

    paths = sorted(paths)

    for path in paths:

        path = os.path.dirname(path)

        ## Get source and target dataset names
        path_split = path.split('/')
        method_job_id_idx = np.where([split==method_job_id for split in path_split])[0][0]

        target = path_split[method_job_id_idx + 1]
        source = path_split[method_job_id_idx + 2] if len(path_split) > method_job_id_idx + 2 else None
        if source in ['0', '1', '2']: source = None

        ## Read metrics
        try:
            metrics_target_valid = pd.read_csv(glob(os.path.join(path, f'*_metrics_target_valid.csv'))[0], index_col=0)
            metrics_source_valid = None if target_only else pd.read_csv(glob(os.path.join(path, f'*_metrics_source_valid.csv'))[0], index_col=0)
        except:
            print(f'Error reading {path}')
            continue

        ## Drop foscttm_score_ct & rank_score
        if 'foscttm_score_ct' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['foscttm_score_ct'])
            if ('foscttm_score_ct' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['foscttm_score_ct'])

        if 'rank_score' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['rank_score'])
            if ('rank_score' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['rank_score'])

        ## Transpose, such that metrics as columns rather than indices
        metrics_target_valid = metrics_target_valid.T
        if not target_only:
            metrics_source_valid = metrics_source_valid.T

        ## Add target dataset names, then append to all_data
        metrics_target_valid['target'] = target
        metrics_target_valid['source'] = source
        all_data_target.append(metrics_target_valid)

        if not target_only:
            metrics_source_valid['target'] = target
            metrics_source_valid['source'] = source
            all_data_source.append(metrics_source_valid)

    ## Concatenate all_data and set multi-index
    target_metrics_valid_df = pd.concat(all_data_target).set_index(['source', 'target'])
    target_metrics_valid_df = target_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in target_metrics_valid_df.columns:
        target_metrics_valid_df['1 - foscttm_score'] = 1 - target_metrics_valid_df['foscttm_score']
        target_metrics_valid_df = target_metrics_valid_df.drop(columns=['foscttm_score'])

    if target_only:
        return target_metrics_valid_df

    ## --- SOURCE --- ##
    source_metrics_valid_df = pd.concat(all_data_source).set_index(['source', 'target'])
    source_metrics_valid_df = source_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in source_metrics_valid_df.columns:
        source_metrics_valid_df['1 - foscttm_score'] = 1 - source_metrics_valid_df['foscttm_score']
        source_metrics_valid_df = source_metrics_valid_df.drop(columns=['foscttm_score'])

    ## Create source-only metrics dataframe, and sort according to existing order of target in original df
    source_only_metrics_valid_df = source_metrics_valid_df.reset_index(level='target').drop(columns='target')
    source_only_metrics_valid_df = source_only_metrics_valid_df.loc[source_metrics_valid_df.index.get_level_values(0).unique().values]

    return source_metrics_valid_df, target_metrics_valid_df, source_only_metrics_valid_df

def compare_dataframes_target(dataframes, value_column, dataset_labels, target_source_combinations, ax=None):

    dataset_label_mapper = {
        'AD_Anderson_et_al': 'DLPFC_Anderson',
        'PD_Adams_et_al': 'Midbrain_Adams',
        'human_dlpfc': 'DLPFC_Ma',
        'roussos': 'PFC_Zhu',
        'mdd': 'MDD'
    }

    hue_order = list(dataset_label_mapper.values())

    combined_df = pd.concat(
        [df.assign(dataset=label).rename(index=dataset_label_mapper)
         for df, label in zip(dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(combined_df["target"].unique())

    colors = sns.color_palette("Dark2", 7)
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    #target_to_color = {target: colors[i] if len(unique_targets)>1 else 'black' for i, target in enumerate(unique_targets)}
    if len(unique_targets) > 1:
        target_to_color = {hue_order[i]: colors[i] if len(unique_targets)>1 else 'black' for i in range(len(unique_targets))}
    elif unique_targets[0] == 'MDD':
        target_to_color = {unique_targets[0]: colors[6]}

    if target_source_combinations:
        source_to_marker = {source: markers[i % len(markers)] for i, source in enumerate(unique_sources)}
    else:
        source_to_marker = {source: 'o' for i, source in enumerate(unique_sources)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    ).tick_params(axis='x', rotation=30)
    ax.set_xlabel("method")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for target in unique_targets:
        for dataset in dataset_positions:
            for source in unique_sources:

                if pd.isna(source):
                    subset = combined_df[(combined_df["target"] == target) &
                                        (combined_df["source"].isna()) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker='o',
                        edgecolor=None,
                        s=50,
                        zorder=10,
                        alpha=0.4
                    )

                else:
                    subset = combined_df[(combined_df["source"] == source) &
                                        (combined_df["target"] == target) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker=source_to_marker[source],
                        edgecolor=None,
                        s=50 if source_to_marker[source] == '*' else 50,
                        zorder=10,
                        alpha=0.4
                    )

    return source_to_marker, target_to_color
        

def compare_dataframes_target_only(dataframes, value_column, dataset_labels, ax=None):
    """
    Compare multiple dataframes with boxplots and scatter points with centered jitter,
    distinguishing only between targets (shapes).

    Parameters:
    - dataframes: list of pandas DataFrames (must have same structure and multi-index)
    - value_column: str, the column name to be plotted
    - dataset_labels: list of str, labels for each dataset
    - ax: Matplotlib Axes object (optional)
    """
    combined_df = pd.concat(
        [
            (df.droplevel('source').assign(dataset=label) if 'source' in df.index.names else df.assign(dataset=label))
            for df, label in zip(dataframes, dataset_labels)
        ]
    ).reset_index()

    unique_targets = sorted(combined_df["target"].unique())
    colors = sns.color_palette("tab10", len(unique_targets))

    target_to_color = {target: colors[i % len(colors)] for i, target in enumerate(unique_targets)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    )

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for target in unique_targets:
        for dataset in dataset_positions:
            subset = combined_df[(combined_df["target"] == target) &
                                 (combined_df["dataset"] == dataset)]
            x_position = position_mapping[dataset]
            jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
            jittered_positions = x_position + jitter
            ax.scatter(
                x=jittered_positions,
                y=subset[value_column],
                color=target_to_color[target],
                edgecolor='black',
                s=150,
                zorder=10,
                alpha=0.9
            )

def compare_dataframes_source_only(dataframes, value_column, dataset_labels, ax=None):

    combined_df = pd.concat(
        [df.assign(dataset=label) for df, label in zip(dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique())
    colors = sns.color_palette("Dark2", len(unique_sources))
    source_to_color = {source: colors[i] for i, source in enumerate(unique_sources)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    ).tick_params(axis='x', rotation=30)
    ax.set_xlabel("method")

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for source in unique_sources:
        for dataset in dataset_positions:
            subset = combined_df[(combined_df["source"] == source) &
                                 (combined_df["dataset"] == dataset)]
            x_position = position_mapping[dataset]
            jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
            jittered_positions = x_position + jitter
            ax.scatter(
                x=jittered_positions,
                y=subset[value_column],
                color=source_to_color[source],
                edgecolor='black',
                s=50,
                zorder=10,
                alpha=0.6
            )

    return source_to_color


def combined_plot(target_dataframes, value_columns, dataset_labels, target_source_combinations=False, source_only_dataframes=None, figsize=(6,6)):

    # Create a figure with two subplots
    if source_only_dataframes is None:
        fig, axs = plt.subplots(1, len(value_columns), figsize=figsize, sharex=True, squeeze=False)
    else:
        fig, axs = plt.subplots(2, len(value_columns), figsize=figsize, sharey='col')

    for value_column, ax in zip(value_columns, axs[-1]):
        # Pass axes to the individual plotting functions
        source_to_marker, target_to_color = compare_dataframes_target(target_dataframes, value_column, dataset_labels, target_source_combinations, ax=ax)

    # Create handles for color-coding by target
    color_handles = [
        plt.Line2D([0], [0], marker='D' if target_source_combinations else 'o', color=color, label=f"{target}", linestyle='None', markersize=10)
        for target, color in target_to_color.items()
    ]

    # Create handles for marker-coding by source
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', label=f"{source}", linestyle='None', markersize=10)
        if not pd.isna(source) else
        plt.Line2D([0], [0], marker='o', color='black', label=f"all sources (ensemble)", linestyle='None', markersize=10)
        for source, marker in source_to_marker.items()
    ]


    # Add labels for color and marker sections
    if ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) > 1) ):

        # Add a blank entry and title for marker section
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles +
            [plt.Line2D([0], [0], color='none', label="")] +  # Blank separator
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +  # Title for the second block
            marker_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()] +
            [''] +  # Blank separator
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )

    elif ( (len(set(source_to_marker.values())) == 1) and ( len(set(target_to_color.keys())) > 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()]
        )
    elif ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) == 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +
            marker_handles
        )
        combined_labels = (
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )
    else:
        combined_handles = None
        combined_labels = None

    axs[-1,-1].legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    # Set a title for the entire figure
    for value_column, ax, letter in zip(value_columns, axs[0], ascii_uppercase):

        ax.set_title(f"{letter}) {value_column}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        if source_only_dataframes is not None:
            source_to_target = compare_dataframes_source_only(source_only_dataframes, value_column, dataset_labels, ax=ax)

            axs[0,0].set_ylabel(f"source to source")
            axs[-1,0].set_ylabel(f"source to target")

            combined_handles = (
                [plt.Line2D([0], [0], color='none', label="Source, by color:")] +
                color_handles
            )
            combined_labels = (
                ['Source, by color:'] +
                [f"{target}" for target in target_to_color.keys()]
            )

            axs[0,-1].legend(
                handles=combined_handles,
                labels=combined_labels,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=False
            )

    plt.tight_layout()
    plt.show()

    return fig

def combined_plot_target_only(target_dataframes, value_column, dataset_labels):

    '''
    combined_df = pd.concat(
        [
            (df.droplevel('source').assign(dataset=label) if 'source' in df.index.names else df.assign(dataset=label))
            for df, label in zip(target_dataframes, dataset_labels)
        ]
    ).reset_index()
    '''

    combined_df = pd.concat(
        [df.assign(dataset=label) for df, label in zip(target_dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(combined_df["target"].unique())

    colors = sns.color_palette("tab10", len(unique_sources))
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    source_to_color = {source: colors[i % len(colors)] for i, source in enumerate(unique_sources)}
    target_to_marker = {target: markers[i % len(markers)] for i, target in enumerate(unique_targets)}

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    # Pass axes to the individual plotting functions
    compare_dataframes_target(target_dataframes, value_column, dataset_labels, ax=axs)

    # Add the legend from compare_dataframes_target
    shape_legend = [
        plt.Line2D([0], [0], color=color, label=target, markerfacecolor='gray', markersize=12)
        for target, color in target_to_color.items()
    ]
    full_legend = shape_legend

    axs.legend(
        handles=full_legend,
        title="Targets",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()

## updated functions
# Create multi-index for CLIP metrics DataFrame
def extract_target_source_replicate(df, has_source=True):

    if df.empty or 'run_name' not in df.columns:
        return df
    
    # Extract source and target from Name column
    sources = []
    targets = []
    replicates = []

    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    if has_source:

        for name in df['run_name']:
            try:
                source = name.split('_to_')[0]
                target = name.split('_to_')[1].split('-')[0]
                replicate = name.split('_to_')[1].split('-')[1]
            except:
                source = name.split('-to-')[0]
                target = name.split('-to-')[1].split('-')[0]
                replicate = name.split('-to-')[1].split('-')[1]
            finally:
                sources.append(source)
                targets.append(target)
                replicates.append(replicate)

        result_df['source'] = sources

    elif not has_source:

        for name in df['run_name']:

            target      = name.split('-')[0]
            replicate   = name.split('-')[1]

            targets.append(target)
            replicates.append(replicate)
    
    # Add source and target columns
    result_df['target'] = targets
    result_df['replicate'] = replicates
    
    # Set multi-index
    #result_df = result_df.set_index(['source', 'target', 'replicate'])
    
    return result_df

def metric_boxplot(df, metric, target_to_color, source_to_marker, unique_targets, unique_sources, ax=None):

    # Try to convert the metric column to float, replacing errors with NaN
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Check if there's only one row for any dataset-metric combination
    dataset_metric_counts = df.groupby(['dataset']).size()
    has_single_values = (dataset_metric_counts == 1).all()
    
    if has_single_values:
        # Use barplot for datasets with single values
        sns.barplot(
            x="dataset",
            y=metric,
            data=df,
            ax=ax,
            color="lightgray"
        ).tick_params(axis='x', rotation=45)

    else:
        sns.boxplot(
            x="dataset",
            y=metric,
            data=df,
            ax=ax,
            color="lightgray",
            showfliers=False,
            boxprops=dict(alpha=0.4),
            whiskerprops=dict(alpha=0.4),
            capprops=dict(alpha=0.4),
            medianprops=dict(alpha=0.7)
        ).tick_params(axis='x', rotation=45)

    ax.set_xlabel("method")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Add scatter points with jitter
    dataset_positions = df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}

    for target in unique_targets:
        for dataset in dataset_positions:
            for source in unique_sources:

                if pd.isna(source):
                    subset = df[(df["target"] == target) &
                                (df["source"].isna()) &
                                (df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[metric],
                        color=target_to_color[target],
                        marker='o',
                        edgecolor=None,
                        s=50,
                        zorder=10,
                        alpha=0.4
                    )

                else:
                    subset = df[(df["source"] == source) &
                                (df["target"] == target) &
                                (df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[metric],
                        color=target_to_color[target],
                        marker=source_to_marker[source],
                        edgecolor=None,
                        s=50 if source_to_marker[source] == '*' else 50,
                        zorder=10,
                        alpha=0.4
                    )

def metric_boxplots(df, target_source_combinations=False, include_paired=True):

    hue_order = list(["PFC_Zhu", "DLPFC_Anderson", "DLPFC_Ma", "Midbrain_Adams", "mouse_brain_10x", "pbmc_10x", "MDD"])

    unique_sources = sorted(df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(df["target"].unique())

    colors = sns.color_palette("Dark2", 8)[:4] + [sns.color_palette("Dark2", 8)[5]] + [sns.color_palette("Dark2", 8)[-1]]
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    #target_to_color = {target: colors[i] if len(unique_targets)>1 else 'black' for i, target in enumerate(unique_targets)}
    if len(unique_targets) > 1:
        target_to_color = {hue_order[i]: colors[i] if len(unique_targets)>1 else 'black' for i in range(len(unique_targets))}
    elif unique_targets[0] == 'MDD':
        target_to_color = {unique_targets[0]: sns.color_palette("Dark2", 8)[6]}

    if target_source_combinations:
        source_to_marker = {source: markers[i % len(markers)] for i, source in enumerate(unique_sources)}
    else:
        source_to_marker = {source: 'o' for i, source in enumerate(unique_sources)}

    ## loop over metrics and plot
    unpaired_metrics = ['multimodal_ilisi', 'ari', 'nmi', 'silhouette_celltype', 'batches_ilisi']
    paired_metrics = ['1-foscttm']
    all_metrics = paired_metrics + unpaired_metrics if include_paired else unpaired_metrics

    fig, axs = plt.subplots(1, len(all_metrics), figsize=(14, 4), sharex=True)

    for metric, ax in zip(all_metrics, axs):
        metric_boxplot(df, metric, target_to_color, source_to_marker, unique_targets, unique_sources, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel('')

    # Create handles for color-coding by target
    color_handles = [
        plt.Line2D([0], [0], marker='D' if target_source_combinations else 'o', color=color, label=f"{target}", linestyle='None', markersize=10)
        for target, color in target_to_color.items()
    ]

    # Create handles for marker-coding by source
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', label=f"{source}", linestyle='None', markersize=10)
        if not pd.isna(source) else
        plt.Line2D([0], [0], marker='o', color='black', label=f"all sources (ensemble)", linestyle='None', markersize=10)
        for source, marker in source_to_marker.items()
    ]


    # Add labels for color and marker sections
    if ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) > 1) ):

        # Add a blank entry and title for marker section
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles +
            [plt.Line2D([0], [0], color='none', label="")] +  # Blank separator
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +  # Title for the second block
            marker_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()] +
            [''] +  # Blank separator
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )

    elif ( (len(set(source_to_marker.values())) == 1) and ( len(set(target_to_color.keys())) > 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()]
        )
    elif ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) == 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +
            marker_handles
        )
        combined_labels = (
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )
    else:
        combined_handles = None
        combined_labels = None

    axs[-1].legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    fig.tight_layout()
    fig.show()

def project_mdd_nuclei_into_latent_space(mdd_rna, mdd_atac, rna_sex_key, atac_sex_key, rna_celltype_key, atac_celltype_key, rna_condition_key, atac_condition_key, eclare_student_model, kd_clip_student_model):

    mdd_rna_sampled, mdd_atac_sampled, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition = \
    sample_proportional_celltypes_and_condition(mdd_rna, mdd_atac, batch_size=5000)

    ## extract sex labels (ideally, extract from sample_proportional_celltypes_and_condition)
    mdd_rna_sex = mdd_rna_sampled.obs[rna_sex_key].str.lower()
    mdd_atac_sex = mdd_atac_sampled.obs[atac_sex_key].str.lower()

    ## get latents
    eclare_rna_latents, eclare_atac_latents = get_latents(eclare_student_model, mdd_rna_sampled, mdd_atac_sampled, return_tensor=False)
    kd_clip_rna_latents, kd_clip_atac_latents = get_latents(kd_clip_student_model, mdd_rna_sampled, mdd_atac_sampled, return_tensor=False)

    ## concatenate latents
    eclare_latents = np.concatenate([eclare_rna_latents, eclare_atac_latents], axis=0)
    kd_clip_latents = np.concatenate([kd_clip_rna_latents, kd_clip_atac_latents], axis=0)
    #clip_latents = np.concatenate([clip_rna_latents, clip_atac_latents], axis=0)

    ## concatenate sexes
    mdd_sexes = np.concatenate([mdd_rna_sex, mdd_atac_sex], axis=0)

    ## plot umaps
    color_map_ct = create_celltype_palette(unique_celltypes, unique_celltypes, plot_color_palette=False)
    _, fig, rna_atac_df_umap = plot_umap_embeddings(eclare_rna_latents, eclare_atac_latents, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition, color_map_ct=color_map_ct, umap_embedding=None)
    #plot_umap_embeddings(kd_clip_rna_latents, kd_clip_atac_latents, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition, color_map_ct=color_map_ct, umap_embedding=None)

def load_model_and_metadata(student_job_id, best_model_idx, device, target_dataset='MDD'):

    ## replace 'multiome' by '10x'
    best_model_idx = best_model_idx.replace('multiome', '10x')
 
    paths_root = os.path.join(os.environ['OUTPATH'], student_job_id)
    student_model_path = os.path.join(paths_root, target_dataset, best_model_idx, 'model_uri.txt')

    ## Load the model and metadata
    with open(student_model_path, 'r') as f:
        model_uris = f.read().strip().splitlines()
        model_uri = model_uris[0]

    ## Set tracking URI to ECLARE_ROOT/mlruns and load model & metadata
    mlflow.set_tracking_uri(os.path.join(os.environ['ECLARE_ROOT'], 'mlruns'))

    run_id = model_uri.split('/')[1]
    print(f'run_id: {run_id}')

    model_dir = os.path.join(os.environ['ECLARE_ROOT'], "mlruns", '*', run_id, "artifacts", "trained_model")
    model_dir = glob(model_dir)[0]
    print(f'model_dir: {model_dir}')

    model_uri = f"file://{model_dir}"
    student_model = mlflow.pytorch.load_model(model_uri, map_location=device)
    student_model_metadata = Model.load(model_dir)

    #student_model = mlflow.pytorch.load_model(model_uri)
    #student_model_metadata = Model.load(model_uri)

    return student_model, student_model_metadata

## functions for pyDESeq2 on counts data
def get_pseudo_replicates_counts(sex, celltype, rna_scaled_with_counts, mdd_rna_var, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, pseudo_replicates='Subjects', overlapping_only=False):

    if pseudo_replicates == 'SEACells':
        ## learn SEACell assignments based on processed data to apply onto counts data

        mdd_seacells_counts_dict = {}

        for condition in unique_conditions:

            ## select cell indices
            rna_indices = pd.DataFrame({
                'is_celltype': rna_scaled_with_counts.obs[rna_celltype_key].str.startswith(celltype),
                'is_condition': rna_scaled_with_counts.obs[rna_condition_key].str.startswith(condition),
                'is_sex': rna_scaled_with_counts.obs[rna_sex_key].str.lower().str.contains(sex.lower()),
            }).prod(axis=1).astype(bool).values.nonzero()[0]

            rna_sampled = rna_scaled_with_counts[rna_indices]

            ## learn SEACell assignments for this condition
            seacells_model = SEACells.core.SEACells(
                rna_sampled, 
                build_kernel_on='X_pca', # could also opt for batch-corrected PCA (Harmony), but ok if pseudo-bulk within batch
                n_SEACells=max(rna_sampled.n_obs // 100, 15), 
                n_waypoint_eigs=15,
                convergence_epsilon = 1e-5,
                max_franke_wolfe_iters=100,
                use_gpu=True if torch.cuda.is_available() else False
                )
            
            seacells_model.construct_kernel_matrix()
            seacells_model.initialize_archetypes()
            seacells_model.fit(min_iter=10, max_iter=100)

            ## summarize counts data by SEACell - remove 'nan' obs_names which groups cells not corresponding to celltype or sex
            mdd_rna_counts.obs.loc[rna_sampled.obs_names,'SEACell'] = rna_sampled.obs['SEACell'].add(f'_{condition}_{sex}_{celltype}')
            mdd_seacells_counts = SEACells.core.summarize_by_SEACell(mdd_rna_counts, SEACells_label='SEACell', summarize_layer='X')
            mdd_seacells_counts = mdd_seacells_counts[mdd_seacells_counts.obs_names != 'nan']

            mdd_seacells_counts.obs[rna_condition_key] = condition
            mdd_seacells_counts.obs[rna_sex_key] = sex
            mdd_seacells_counts.obs[rna_celltype_key] = celltype

            mdd_seacells_counts_dict[condition] = mdd_seacells_counts

        ## concatenate all SEACell counts data across conditions
        mdd_seacells_counts_adata = anndata.concat(mdd_seacells_counts_dict.values(), axis=0)
        mdd_seacells_counts_adata = mdd_seacells_counts_adata[:, mdd_seacells_counts_adata.var_names.isin(mdd_rna.var_names)]
        mdd_seacells_counts_adata.var = mdd_rna.var

        counts = mdd_seacells_counts_adata.X.astype(int).toarray()
        metadata = mdd_seacells_counts_adata.obs

    elif pseudo_replicates == 'Subjects':

        subjects_from_sex = rna_scaled_with_counts.obs[rna_scaled_with_counts.obs[rna_sex_key].str.lower() == sex.lower()][rna_subject_key].unique()
        if overlapping_only:
            subjects_from_sex = subjects_from_sex[np.isin(subjects_from_sex, [os.split('_')[-1] for os in overlapping_subjects])]

        mdd_subjects_counts_dict = {}

        for subject in subjects_from_sex:

            rna_indices =  pd.DataFrame({
                'is_subject': rna_scaled_with_counts.obs[rna_subject_key] == subject,
                'is_celltype': rna_scaled_with_counts.obs[rna_celltype_key].str.startswith(celltype)
            })
            rna_indices = rna_indices.prod(axis=1).astype(bool).values.nonzero()[0]

            rna_sampled = rna_scaled_with_counts[rna_indices]
            rna_subject_counts = rna_sampled.raw.X.sum(axis=0).A1.astype(int)
            rna_subject_var = rna_sampled.raw.var.set_index('_index')

            subject_condition = rna_scaled_with_counts.obs[rna_condition_key][rna_scaled_with_counts.obs[rna_subject_key] == subject].iloc[0]
            batch = rna_scaled_with_counts.obs['Batch'][rna_scaled_with_counts.obs[rna_subject_key] == subject].iloc[0]

            rna_subject_obs = pd.DataFrame(
                np.hstack([batch, subject_condition, sex, celltype]).reshape(1, -1),
                columns=['Batch', rna_condition_key, rna_sex_key, rna_celltype_key],
                index=[subject],
            )

            rna_subject_counts_ad = anndata.AnnData(
                X=rna_subject_counts.reshape(1, -1),
                var=rna_subject_var,
                obs=rna_subject_obs,
            )
            mdd_subjects_counts_dict[subject] = rna_subject_counts_ad

        mdd_subjects_counts_adata = anndata.concat(mdd_subjects_counts_dict.values(), axis=0)
        mdd_subjects_counts_adata = mdd_subjects_counts_adata[:, mdd_subjects_counts_adata.var_names.isin(mdd_rna_var.index)]
        mdd_subjects_counts_adata.var = mdd_rna_var

        counts = mdd_subjects_counts_adata.X.astype(int)#.toarray()
        metadata = mdd_subjects_counts_adata.obs

    return mdd_subjects_counts_adata, counts, metadata

def run_pyDESeq2(mdd_subjects_counts_adata, counts, metadata, rna_condition_key, save_dir=None):

    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=['Batch', rna_condition_key],
        ref_level=[rna_condition_key, "Control"],
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    stat_res = DeseqStats(dds, inference=inference)
    stat_res.summary()

    ## get results and volcano plot
    results = stat_res.results_df
    results['signif_padj'] = results['padj'] < 0.05
    results['signif_lfc'] = results['log2FoldChange'].abs() > 1.5
    results['signif'] = results['signif_padj'] & results['signif_lfc']
    results['-log10(padj)'] = -np.log10(results['padj'])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.scatterplot(data=results, x='log2FoldChange', y='-log10(padj)', hue='signif_padj', marker='o', alpha=0.5)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'volcano_plot.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    ## extract significant genes
    significant_genes = mdd_subjects_counts_adata.var_names[results['signif_padj']]
    mdd_subjects_counts_adata.var['signif_padj'] = False
    mdd_subjects_counts_adata.var.loc[significant_genes, 'signif_padj'] = True

    ## violin plot
    df = mdd_subjects_counts_adata[:,mdd_subjects_counts_adata.var_names.isin(significant_genes[:10])].to_df()
    df = df.reset_index()
    df = pd.melt(df, id_vars=['index'], var_name='gene', value_name='expression')
    df = df.merge(mdd_subjects_counts_adata.obs, left_on='index', right_index=True)
    df = df.sort_values(rna_condition_key, ascending=False) # forces controls to be listed first, putting controls on the left-hand violin plots

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=df, x='gene', y='expression', hue=rna_condition_key, split=True, inner=None, cut=0, ax=ax)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'violin_plot.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    return mdd_subjects_counts_adata, results, significant_genes

## functions for dicts
def tree(): return defaultdict(tree)
def initialize_dicts():
    global X_rna_dict, X_atac_dict, overlapping_target_genes_dict, overlapping_tfs_dict, scompreg_loglikelihoods_dict, std_errs_dict, tg_expressions_dict, tfrps_dict, tfrp_predictions_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict

    X_rna_dict = tree()
    X_atac_dict = tree()
    overlapping_target_genes_dict = tree()
    overlapping_tfs_dict = tree()

    genes_by_peaks_corrs_dict = tree()
    genes_by_peaks_masks_dict = tree()
    n_dict = tree()

    scompreg_loglikelihoods_dict = tree()
    std_errs_dict = tree()
    slopes_dict = tree()
    intercepts_dict = tree()
    intercept_stderrs_dict = tree()

    tg_expressions_dict = tree()
    tfrps_dict = tree()
    tfrp_predictions_dict = tree()

    return X_rna_dict, X_atac_dict, overlapping_target_genes_dict, overlapping_tfs_dict, genes_by_peaks_corrs_dict, genes_by_peaks_masks_dict, n_dict, scompreg_loglikelihoods_dict, std_errs_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, tfrps_dict, tfrp_predictions_dict
def assign_to_dicts(sex, celltype, condition, X_rna, X_atac, overlapping_target_genes, overlapping_tfs, scompreg_loglikelihoods, std_errs, tg_expressions, tfrps, tfrp_predictions, slopes, intercepts, intercept_stderrs):
    
    X_rna_dict[sex][celltype][condition if condition != '' else 'all'] = X_rna
    X_atac_dict[sex][celltype][condition if condition != '' else 'all'] = X_atac
    overlapping_target_genes_dict[sex][celltype][condition if condition != '' else 'all'] = overlapping_target_genes
    overlapping_tfs_dict[sex][celltype][condition if condition != '' else 'all'] = overlapping_tfs

    scompreg_loglikelihoods_dict[sex][celltype][condition if condition != '' else 'all'] = scompreg_loglikelihoods
    std_errs_dict[sex][celltype][condition if condition != '' else 'all'] = std_errs
    tg_expressions_dict[sex][celltype][condition if condition != '' else 'all'] = tg_expressions
    tfrps_dict[sex][celltype][condition if condition != '' else 'all'] = tfrps
    tfrp_predictions_dict[sex][celltype][condition if condition != '' else 'all'] = tfrp_predictions
    slopes_dict[sex][celltype][condition if condition != '' else 'all'] = slopes
    intercepts_dict[sex][celltype][condition if condition != '' else 'all'] = intercepts
    intercept_stderrs_dict[sex][celltype][condition if condition != '' else 'all'] = intercept_stderrs


## check overlap of DEGs and DARs
def check_overlap_of_degs_and_dars(significant_genes):
    maitra_female_degs_df   = pd.read_excel(os.path.join(os.environ['DATAPATH'], 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData7', header=2)
    doruk_peaks_df          = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'combined', 'cluster_DAR_0.2.tsv'), sep='\t')
    #maitra_male_degs_df = pd.read_excel(os.path.join(datapath, 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData5', header=2)

    deg_dict = maitra_female_degs_df.groupby('cluster_id.female')['gene'].unique().to_dict()

    deg_overlap_df = pd.DataFrame.from_dict({key: np.isin(deg_dict[key], significant_genes).sum() for key in deg_dict.keys()}, orient='index').rename(columns={0: 'Case'})
    deg_overlap_df = pd.concat([deg_overlap_df, deg_overlap_df.sum(axis=0).to_frame().T.rename(index={0: 'TOTAL'})], axis=0)

    for celltype in deg_overlap_df.index:
        try:
            unique_celltype = [unique_celltype for unique_celltype in unique_celltypes if unique_celltype in celltype][0]
            deg_overlap_df.loc[celltype, 'unique_celltype'] = unique_celltype
        except:
            print(f"Could not find unique celltype for {celltype}")
            continue

    deg_overlap_grouped_df = deg_overlap_df.groupby('unique_celltype').sum()
    display(deg_overlap_grouped_df.sort_values(by='Case', ascending=False).T)

## sc-compReg functions
def compute_scompreg_loglikelihoods(sex, celltype, 
    scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, slopes_dict, std_errs_dict, intercepts_dict, intercept_stderrs_dict,
    Z_type='hawkins-wixley'):
    
    ## pre-computed log likelihoods
    scompreg_loglikelihoods_case_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['Case']
    scompreg_loglikelihoods_control_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['Control']
    scompreg_loglikelihoods_all_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['all']

    scompreg_loglikelihoods_precomputed_df = pd.DataFrame({
        'case': scompreg_loglikelihoods_case_precomputed,
        'control': scompreg_loglikelihoods_control_precomputed,
        'all': scompreg_loglikelihoods_all_precomputed
    })
    genes_names = scompreg_loglikelihoods_precomputed_df.index

    LR_precomputed = -2 * (scompreg_loglikelihoods_precomputed_df.apply(lambda gene: gene['all'] - (gene['case'] + gene['control']), axis=1))

    ## re-computed log likelihoods and confirm that same as pre-computed
    for condition in ['Case', 'Control']:

        tfrps               = tfrps_dict[sex][celltype][condition]
        tg_expressions      = tg_expressions_dict[sex][celltype][condition]

        for gene in genes_names:

            log_gaussian_likelihood_pre = scompreg_loglikelihoods_dict[sex][celltype][condition][gene]
            if np.isnan(log_gaussian_likelihood_pre):
                continue

            tfrp                = tfrps.loc[:,gene].values.astype(float)
            tg_expression       = tg_expressions.loc[:,gene].values.astype(float)

            ## compute slope and intercept of linear regression - tg_expression is sparse (so is tfrp)
            slope, intercept, r_value, p_value, std_err = linregress(tg_expression, tfrp)
            tfrp_prediction = slope * tg_expression + intercept

            ## compute residuals and variance
            n = len(tfrp)
            sq_residuals = (tfrp - tfrp_prediction)**2
            var = sq_residuals.sum() / n
            log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()

            diff = np.abs(log_gaussian_likelihood - log_gaussian_likelihood_pre)
            assert diff < 1e-6

    ## re-compute log likelihoods for pooled conditions
    tfrps_all = pd.concat([ tfrps_dict[sex][celltype]['Case'], tfrps_dict[sex][celltype]['Control'] ]).reset_index(drop=True)
    tg_expressions_all = pd.concat([ tg_expressions_dict[sex][celltype]['Case'], tg_expressions_dict[sex][celltype]['Control'] ]).reset_index(drop=True)
    tfrp_predictions_all = pd.concat([ tfrp_predictions_dict[sex][celltype]['Case'], tfrp_predictions_dict[sex][celltype]['Control'] ]).reset_index(drop=True)

    ## compute correlation between tfrp predictions and tfrps - could potential use to filter genes
    tfrp_corrs = tfrp_predictions_all.corrwith(tfrps_all, method='kendall')
    tfrp_corrs = tfrp_corrs.dropna()

    #print(f'Computing pooled log likelihoods')
    diffs_all = []
    scompreg_loglikelihoods_all = {}

    for gene in genes_names:

        tfrp = tfrps_all.loc[:,gene].values.astype(float)
        tg_expression = tg_expressions_all.loc[:,gene].values.astype(float)

        try:
            slope, intercept, r_value, p_value, std_err = linregress(tg_expression, tfrp)
            tfrp_predictions = slope * tg_expression + intercept
        except:
            #print(f'{gene} has no variance in tg_expression')
            tfrp_predictions = np.ones_like(tg_expression) * np.nan
            log_gaussian_likelihood = np.array([np.nan])
        else:
            n = len(tfrp)
            sq_residuals = (tfrp - tfrp_predictions)**2
            var = sq_residuals.sum() / n
            log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()
        finally:
            scompreg_loglikelihoods_all[gene] = log_gaussian_likelihood

        #diff = np.abs(log_gaussian_likelihood - scompreg_loglikelihoods_dict[sex][celltype]['all'][gene])
        #diffs_all.append(diff)

    ## collect all log likelihoods
    scompreg_loglikelihoods_df = pd.DataFrame({
        'case': scompreg_loglikelihoods_case_precomputed,
        'control': scompreg_loglikelihoods_control_precomputed,
        'all': scompreg_loglikelihoods_all
    }).dropna() # drop rows with nan

    ## extract gene names from dataframe since some genes were dropped due to nan
    genes_names = scompreg_loglikelihoods_df.index

    ## compute log likelihood ratio LR
    LR = -2 * (scompreg_loglikelihoods_df.apply(lambda gene: gene['all'] - (gene['case'] + gene['control']), axis=1))

    if Z_type == 'wilson-hilferty':
        ## apply Wilson-Hilferty cube-root transformation
        dof = 3
        Z = (9*dof/2)**(1/2) * ( (LR/dof)**(1/3) - (1 - 2/(9*dof)) )
    elif Z_type == 'hawkins-wixley':
        ## Hawkins & Wixley 1986
        lambd = 0.2887  # optimal value for dof=3
        Z = (LR**lambd - 1) / lambd

    ## Recenter Z distribution
    Z = Z - np.median(Z)

    if (LR < 0).any():
        print(f'Found negative LR for {celltype} {sex}')

    ## Compute t-statistic between regression slopes
    slopes_case = slopes_dict[sex][celltype]['Case']
    slopes_control = slopes_dict[sex][celltype]['Control']

    std_errs_case = std_errs_dict[sex][celltype]['Case']
    std_errs_control = std_errs_dict[sex][celltype]['Control']

    t_df = pd.DataFrame({
        'case': slopes_case,
        'control': slopes_control,
        'std_err_case': std_errs_case,
        'std_err_control': std_errs_control
    }).dropna()

    t_slopes = (t_df['case'] - t_df['control']) / np.sqrt(t_df['std_err_case']**2 + t_df['std_err_control']**2)
    t_slopes = t_slopes.dropna() # might have residual nan due to std_err being 0

    ## compute correlation between LR and absolute value of slopes t-statistic
    LR.name = 'LR'
    t_slopes.name = 't_slopes'
    LR_t_df = pd.merge(left=LR, right=t_slopes.abs(), left_index=True, right_index=True, how='inner')
    LR_t_corr = LR_t_df.corr(method='spearman')['LR']['t_slopes']
    #print(f'Correlation between LR and absolute value of slopes t-statistic: {LR_t_corr:.2f}')

    ## separate up and down LR and Z based on sign of t-statistic
    LR_up = LR[t_slopes > 0]
    LR_down = LR[t_slopes < 0]

    Z_up = Z[t_slopes > 0]
    Z_down = Z[t_slopes < 0]

    return LR, Z, LR_up, LR_down, Z_up, Z_down

def filter_LR_stats(LR, Z, LR_up, LR_down, save_dir=None):

    # 1) Suppose `lr` is your 1-D array of empirical LR statistics:
    lr = np.array(LR)

    # 2) Compute subset of empirical quantiles:
    max_null_quantile = 0.5

    probs = np.linspace(0.05, max_null_quantile, 10)
    emp_q = np.quantile(lr, probs)

    # 3) Define the objective G(a, b) = sum_i [Gamma.ppf(probs[i], a, scale=b) - emp_q[i]]^2
    def G(params):
        a, scale = params
        # enforce positivity
        if a <= 0 or scale <= 0:
            return np.inf
        theor_q = gamma.ppf(probs, a, scale=scale)
        return np.sum((theor_q - emp_q)**2)

    # 4) Method-of-moments init on the lower half to get a reasonable starting point
    m1 = emp_q.mean()
    v1 = emp_q.var()
    init_a = m1**2 / v1
    init_scale = v1 / m1

    # 5) Minimize G(a, scale) over a>0, scale>0
    res = minimize(G,
                x0=[init_a, init_scale],
                bounds=[(1e-8, None), (1e-8, None)],
                method='L-BFGS-B')


    # Plot empirical histogram and fitted gamma distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lr, bins=100, density=True, alpha=0.6, label='Empirical LR distribution')

    # Generate points for fitted gamma distribution
    x = np.linspace(0, max(lr), 1000)
    fitted_pdf = gamma.pdf(x, a=res.x[0], scale=res.x[1])
    plt.plot(x, fitted_pdf, 'r-', linewidth=2, label='Fitted gamma distribution')

    # Add veritcal line at last null quantile
    plt.axvline(emp_q[-1], color='black', linestyle='--', label=f'Last null quantile of empirical distribution (q={probs[-1]:.2f})')

    # Add veritcal line at threshold
    p_threshold = 0.05
    lr_at_p = gamma.ppf(1-p_threshold, a=res.x[0], scale=res.x[1])
    plt.axvline(lr_at_p, color='green', linestyle='--', label=f'Threshold (p={p_threshold:.2f}) @ LR={lr_at_p:.2f}')

    plt.xlabel('LR statistic')
    plt.ylabel('Density')
    plt.title('Comparison of Empirical and Fitted LR Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'lr_gamma_fit_grn.png'), bbox_inches='tight', dpi=150)
    plt.close()

    ## filter LR values with fitted null distribution
    lr_fitted_cdf = gamma.cdf(lr, a=res.x[0], scale=res.x[1])
    where_filtered = lr_fitted_cdf > (1-p_threshold)

    lr_filtered = LR[where_filtered]
    Z_filtered = Z[where_filtered]
    genes_names_filtered = lr_filtered.index.to_list()

    ## filter LR_up and LR_down, note that splits gene set in half, could use more leniant threshold to include more genes for up and down, or even create separate null distributions for each
    lr_up_filtered = LR_up[LR_up.index.isin(genes_names_filtered)]
    lr_down_filtered = LR_down[LR_down.index.isin(genes_names_filtered)]

    return lr_filtered, Z_filtered, lr_up_filtered, lr_down_filtered, lr_fitted_cdf

def get_deg_gene_sets(LR, lr_filtered, lr_fitted_cdf, significant_genes):
    
    deg_df = pd.DataFrame(significant_genes, columns=['names'], index=significant_genes)

    ## LR values for DEG genes (not all DEG genes are in LR)
    LR_deg = LR.loc[LR.index.isin(deg_df['names'])]
    deg_not_in_lr = deg_df[~deg_df['names'].isin(LR.index)]['names'].drop_duplicates().to_list()
    deg_not_in_lr_series = pd.Series(np.nan, index=deg_not_in_lr)
    LR_deg = pd.concat([LR_deg, deg_not_in_lr_series])

    ## DEG genes with top filtered LR genes
    #where_filtered_idxs = np.sort(np.flip(lr_fitted_cdf.argsort())[:len(lr_filtered)]); top_lr_filtered = LR[where_filtered_idxs]; assert top_lr_filtered.equals(lr_filtered)
    try:
        n_lr_deg = len(lr_filtered) + deg_df['names'].nunique()
        n_lr_minus_deg = len(lr_filtered) - deg_df['names'].nunique(); assert n_lr_minus_deg > 0, 'list of DEG genes is longer than list of filtered LR genes'

        where_filtered_with_excess = np.flip(lr_fitted_cdf.argsort(kind='stable'))[:n_lr_deg]
        where_filtered_with_excess_top = where_filtered_with_excess[:n_lr_minus_deg]
        where_filtered_with_excess_bottom = where_filtered_with_excess[-n_lr_minus_deg:]

        top_lr_filtered = LR[where_filtered_with_excess_top]
        bottom_lr_filtered = LR[where_filtered_with_excess_bottom]

        top_lr_filtered_wDeg = pd.concat([top_lr_filtered, LR_deg])
        bottom_lr_filtered_wDeg = pd.concat([bottom_lr_filtered, LR_deg])

        n_genes = np.unique([len(lr_filtered), len(top_lr_filtered_wDeg), len(bottom_lr_filtered_wDeg)]); assert len(n_genes) == 1, 'number of genes in each list must be the same'
        n_genes = n_genes[0]
    except:
        print('Skipping DEG + LR gene set')
        top_lr_filtered_wDeg, bottom_lr_filtered_wDeg, lr_filtered_woDeg = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ## filtered LR values ex-DEG (replace with other non-DEG genes)
    lr_filtered_is_deg = lr_filtered.index.isin(deg_df['names'])

    if lr_filtered_is_deg.any() and (n_lr_minus_deg > 0):

        lr_filtered_woDeg_short = lr_filtered[~lr_filtered_is_deg]
        deg_woLR = lr_filtered[lr_filtered_is_deg]
        print(f'{lr_filtered_is_deg.sum()} DEG genes removed from LR')

        replace_degs_with_non_degs = where_filtered_with_excess[ len(lr_filtered) : (len(lr_filtered) + lr_filtered_is_deg.sum()) ]
        lr_filtered_woDeg = pd.concat([lr_filtered_woDeg_short, LR[replace_degs_with_non_degs]])

    else:
        print('No DEG genes in LR')
        lr_filtered_woDeg = pd.DataFrame()

    return top_lr_filtered_wDeg, bottom_lr_filtered_wDeg, lr_filtered_woDeg


def merge_grn_lr_filtered(mean_grn_df, lr_filtered, output_dir):

    mean_grn_df_filtered = mean_grn_df.merge(lr_filtered, left_on='TG', right_index=True, how='right')
    mean_grn_df_filtered['lrCorr'] = mean_grn_df_filtered[['Correlation','LR']].product(axis=1, skipna=True).abs()

    ## create graph from mean_grn_df_filtered with lrCorr as edge weight
    mean_grn_filtered_graph = nx.from_pandas_edgelist(
        mean_grn_df_filtered,
        source='TG',
        target='enhancer',
        edge_attr='lrCorr',
        create_using=nx.DiGraph())

    peaks = scglue.genomics.Bed(mean_grn_df_filtered.assign(name=mean_grn_df_filtered['enhancer']))

    gene_ad = anndata.AnnData(var=pd.DataFrame(index=mean_grn_df_filtered['TG'].to_list()))
    scglue.data.get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')
    genes = scglue.genomics.Bed(gene_ad.var)

    #genes = scglue.genomics.Bed(mdd_rna[:, mdd_rna.var['chrom'].notna()].var.assign(name=mdd_rna[:, mdd_rna.var['chrom'].notna()].var_names))
    #genes = genes[genes['name'].isin(mean_grn_df_filtered['TG'])]
    tss = genes.strand_specific_start_site()

    peaks.write_bed(os.path.join(output_dir, 'peaks.bed'))
    scglue.genomics.write_links(
        mean_grn_filtered_graph,
        tss,
        peaks,
        os.path.join(output_dir, 'gene2peak_lrCorr.links'),
        keep_attrs=["lrCorr"]
        )

    '''
    tg_dist_counts_sorted = mean_grn_df_filtered.groupby('TG')[['dist']].mean().merge(mean_grn_df_filtered['TG'].value_counts(), left_index=True, right_on='TG').sort_values('dist').head(20)
    display(tg_dist_counts_sorted)

    gene = tg_dist_counts_sorted.iloc[10].name
    tg_grn = mean_grn_df_filtered[mean_grn_df_filtered['TG']==gene].sort_values('dist')[['enhancer','dist','lrCorr']].groupby('enhancer').mean()
    tg_grn_bounds = np.stack(tg_grn.index.str.split(':|-')).flatten()
    tg_grn_bounds = [int(bound) for bound in tg_grn_bounds if bound.isdigit()] + [genes.loc[gene, 'chromStart'].drop_duplicates().values[0]] + [genes.loc[gene, 'chromEnd'].drop_duplicates().values[0]]
    display(tg_grn)
    print(f'{genes.loc[gene, "chrom"].drop_duplicates().values[0]}:{min(tg_grn_bounds)}-{max(tg_grn_bounds)}')
    print(genes.loc[gene,['chrom','chromStart','chromEnd','name']].drop_duplicates())
    '''

    return mean_grn_df_filtered

## gene set enrichment analyses
def get_brain_gmt():
    
    brain_gmt = gp.parser.read_gmt(os.path.join(os.environ['DATAPATH'], 'BrainGMTv2_HumanOrthologs.gmt'))
    brain_gmt_names = pd.Series(list(brain_gmt.keys()))
    brain_gmt_prefix_unique = np.unique([name.split('_')[0] for name in brain_gmt_names])

    brain_gmt_wGO = gp.parser.read_gmt(os.path.join(os.environ['DATAPATH'], 'BrainGMTv2_wGO_HumanOrthologs.gmt'))
    brain_gmt_wGO_names = pd.Series(list(brain_gmt_wGO.keys()))
    brain_gmt_wGO_prefix_unique = np.unique([name.split('_')[0] for name in brain_gmt_wGO_names])

    non_cortical_blacklist = [
        'DropViz',
        'HippoSeq',
        'Coexpression_Hippocampus',
        'Birt_Hagenauer_2020',
        'Gray_2014',
        'Bagot_2016',
        'Bagot_2017',
        'Pena_2019',
        'GeneWeaver',
        'GenWeaver', # probably same as GeneWeaver
        'Gemma'
    ]

    #gset_by_blacklist_df = pd.DataFrame([brain_gmt_names.str.contains(ncbl) for ncbl in non_cortical_blacklist], index=non_cortical_blacklist, columns=brain_gmt_names)
    gset_by_blacklist_df = pd.DataFrame([brain_gmt_names.str.contains(ncbl) for ncbl in non_cortical_blacklist])
    gset_by_blacklist_df.index = non_cortical_blacklist
    gset_by_blacklist_df.columns = brain_gmt_names

    keep_cortical_gmt = gset_by_blacklist_df.loc[:,~gset_by_blacklist_df.any(axis=0).values].columns.tolist()
    brain_gmt_cortical = {k:v for k,v in brain_gmt.items() if k in keep_cortical_gmt}

    gset_by_blacklist_df_wGO = pd.DataFrame([brain_gmt_wGO_names.str.contains(ncbl) for ncbl in non_cortical_blacklist])
    gset_by_blacklist_df_wGO.index = non_cortical_blacklist
    gset_by_blacklist_df_wGO.columns = brain_gmt_wGO_names

    keep_cortical_gmt_wGO = gset_by_blacklist_df_wGO.loc[:,~gset_by_blacklist_df_wGO.any(axis=0).values].columns.tolist()
    brain_gmt_cortical_wGO = {k:v for k,v in brain_gmt_wGO.items() if k in keep_cortical_gmt_wGO}

    return brain_gmt_cortical, brain_gmt_cortical_wGO

def run_h_magma_prep(output_dir):

    MAGMAPATH = glob(os.path.join(os.environ['ECLARE_ROOT'], 'magma_v1.10*'))[0]
    os.makedirs(os.path.join(os.environ['DATAPATH'], 'MDD_H_MAGMA'), exist_ok=True)

    bfile_path = os.path.join(os.environ['DATAPATH'], 'g1000_eur', 'g1000_eur')
    pval_path = os.path.join(os.environ['DATAPATH'], 'DS_10283_3203', 'PGC_UKB_depression_genome-wide.txt')
    gene_annot_path = os.path.join(os.environ['DATAPATH'], 'Adult_brain.genes.annot')
    out_path = os.path.join(os.environ['DATAPATH'], 'MDD_H_MAGMA', 'mdd_h_magma')

    h_magma_cmd = f"""{MAGMAPATH}/magma \
    --bfile {bfile_path} \
    --pval {pval_path} use=MarkerName,P N=500199 \
    --gene-annot {gene_annot_path} \
    --out {out_path}"""

    subprocess.run(h_magma_cmd, shell=True, check=True)

    return out_path

def run_magma(lr_filtered, Z, significant_genes, output_dir, fuma_job_id='604461'):
    
    mg = mygene.MyGeneInfo()

    ## paths for MAGMA
    MAGMAPATH = glob(os.path.join(os.environ['ECLARE_ROOT'], 'magma_v1.10*'))[0]

    if fuma_job_id is not None:
        magma_genes_raw_path = os.path.join(os.environ['DATAPATH'], 'FUMA_public_jobs', f'FUMA_public_job{fuma_job_id}', 'magma.genes.raw')  # https://fuma.ctglab.nl/browse#GwasList
    else:
        magma_genes_raw_path = os.path.join(os.environ['DATAPATH'], 'MDD_H_MAGMA', 'mdd_h_magma.genes.raw')
    
    # Create unique output path to avoid race conditions
    import threading
    thread_id = threading.current_thread().ident
    magma_out_path = os.path.join(output_dir, f'sc-compReg_significant_genes_thread_{thread_id}')

    ## read "genes.raw" file and see if IDs start with "ENSG" or is integer
    genes_raw_df = pd.read_csv(magma_genes_raw_path, sep='/t', header=None, skiprows=2)
    genes_raw_ids = genes_raw_df[0].apply(lambda x: x.split(' ')[0])

    #genes_out_df = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'FUMA_public_jobs', f'FUMA_public_job{fuma_job_id}', 'magma.genes.out'), sep='\t')
    #genes_magma_control = genes_out_df.loc[genes_out_df['ZSTAT'].argsort()][-len(significant_genes):]['GENE'].to_list()

    #lr_filtered_top = lr_filtered[lr_filtered.argsort().values][-len(significant_genes):]
    #lr_filtered_union = pd.concat([lr_filtered, pd.Series(significant_genes, index=significant_genes)]).drop_duplicates()

    gene_sets = {
        'sc-compReg': lr_filtered.index.to_list(),
        'Z_IGNORE': Z.index.to_list()}
    
    if not significant_genes.empty:
        gene_sets['pyDESeq2'] = significant_genes.to_list()

    ## get entrez IDs or Ensembl IDs for significant genes
    set_file_dict = {}
    sig_ensembl_ids_all_dict = {}

    for gene_set_name, gene_set in gene_sets.items():

        if pd.Series(gene_set).str.startswith('ENSG').all():
            #print(f"IDs are already Ensembl IDs")
            set_file_dict[gene_set_name] = gene_set

        else:
            #print(f"IDs are not Ensembl IDs, converting to Ensembl IDs")

            if genes_raw_ids.str.startswith('ENSG').all():

                #print(f"IDs are Ensembl IDs")

                significant_genes_mg_df = mg.querymany(
                    gene_set,
                    scopes="symbol",
                    fields="ensembl.gene",
                    species="human",
                    size=1,
                    as_dataframe=True,
                    verbose=False
                    )
                
                sig_ensembl_ids = significant_genes_mg_df['ensembl.gene'].dropna()
                sig_ensembl_ids_exploded = (significant_genes_mg_df['ensembl'].dropna().apply(lambda gene: [ensembl['gene'] for ensembl in gene]).explode() if ('ensembl' in significant_genes_mg_df.columns) else pd.Series(dtype=str))
                sig_ensembl_ids_all = pd.concat([sig_ensembl_ids, sig_ensembl_ids_exploded])

                significant_genes_ensembl_ids = list(set(sig_ensembl_ids_all.to_list()))

                set_file_dict[gene_set_name] = significant_genes_ensembl_ids
                sig_ensembl_ids_all_dict[gene_set_name] = sig_ensembl_ids_all

            else:

                #print(f"IDs are not Ensembl (defaulting to Entrez IDs)")

                significant_genes_entrez_ids = []
                for gene in tqdm(gene_set, total=len(gene_set), desc='Getting entrez IDs for significant genes'):
                    entrez_id_res = mg.query(gene, species='human', scopes='entrezgenes', size=1, verbose=False)
                    entrez_id = entrez_id_res['hits'][0]['_id']
                    significant_genes_entrez_ids.append(entrez_id)

                set_file_dict[gene_set_name] = significant_genes_entrez_ids

    ## write gene set file with unique path
    magma_set_file_path = os.path.join(os.path.dirname(output_dir), f'significant_gene_sets_thread_{thread_id}.gmt')

    with open(magma_set_file_path, 'w') as f:
        for gene_set_name, gene_set in set_file_dict.items():
            if not gene_set_name.endswith('IGNORE'):
                f.write(f"{gene_set_name} {' '.join(gene_set)}\n")

    ## write gene covariate file
    Z_IGNORE_ensembl = sig_ensembl_ids_all_dict['Z_IGNORE']
    Z_IGNORE_ensembl.name = 'ensembl'

    Z.name = 'ZSTAT'
    Z.index.name = 'gene'

    Z_ensembl = pd.merge(Z, Z_IGNORE_ensembl, left_index=True, right_index=True, how='right')
    Z_ensembl = Z_ensembl.set_index('ensembl', drop=True)

    magma_genecovar_path = os.path.join(os.path.dirname(output_dir), f'Z.covariates_thread_{thread_id}.txt')
    Z_ensembl.to_csv(magma_genecovar_path, sep='\t', header=True)

    ## write list file for interaction analysis
    magma_list_file_path = os.path.join(os.path.dirname(output_dir), f'significant_genes_thread_{thread_id}.list')
    list_file_dict = {'condition-interaction': ['sc-compReg', 'ZSTAT']}
    pd.DataFrame.from_dict(list_file_dict, orient='index').to_csv(magma_list_file_path, sep='\t', header=False, index=False)

    ## run MAGMA using subprocess instead of os.system for better thread safety
    
    magma_gs_cmd = f"""{MAGMAPATH}/magma \
        --gene-results {magma_genes_raw_path} \
        --set-annot {magma_set_file_path} \
        --model self-contained correct=all\
        --out {magma_out_path}"""

    magma_cvar_cmd = f"""{MAGMAPATH}/magma \
        --gene-results {magma_genes_raw_path} \
        --gene-covar {magma_genecovar_path} \
        --model correct=all\
        --out {magma_out_path}"""

    magma_gs_cvar_interaction_cmd = f"""{MAGMAPATH}/magma \
        --gene-results {magma_genes_raw_path} \
        --set-annot {magma_set_file_path} \
        --gene-covar {magma_genecovar_path} \
        --model interaction={magma_list_file_path}\
        --out {magma_out_path}"""

    # Use subprocess.run instead of os.system for better thread safety
    #subprocess.run(magma_gs_cmd, shell=True, check=True)
    #subprocess.run(magma_cvar_cmd, shell=True, check=True)
    subprocess.run(magma_gs_cvar_interaction_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ## check content of output file and log file
    logfile = magma_out_path + '.log'
    outfile = magma_out_path + '.gsa.out'
    #outfile_self = magma_out_path + '.gsa.self.out'

    '''
    with open(logfile, 'r') as f:
        for line in f:
            print(line)
    with open(outfile, 'r') as f:
        for line in f:
            print(line)
    with open(outfile_self, 'r') as f:
        for line in f:
            print(line)
    '''

    ## check number of lines in outfile to establish skiprows - depends on whether first line is # TOTAL_GENES or # MEAN_SAMPLE_SIZE
    with open(outfile, 'r') as f:
        n_lines = sum(1 for _ in f)
    skiprows = n_lines - (3 + 1)

    ## read in MAGMA results
    magma_results_path = magma_out_path + '.gsa.out'
    magma_results = pd.read_csv(magma_results_path, sep=r'\s+', skiprows=skiprows)
    magma_results_series = magma_results.set_index('VARIABLE').iloc[:,-1]

    return magma_results_series, magma_results

def run_gseapy(rank, brain_gmt_cortical):

    ranked_list = pd.DataFrame({'gene': rank.index.to_list(), 'score': rank.values})
    ranked_list = ranked_list.sort_values(by='score', ascending=False)

    pre_res = gp.prerank(rnk=ranked_list,
                        gene_sets=brain_gmt_cortical,
                        outdir=os.path.join(os.environ['OUTPATH'], 'gseapy_results'),
                        min_size=2,
                        max_size=len(ranked_list),
                        permutation_num=1000)

    pre_res.res2d.sort_values('FWER p-val', ascending=True).head(20)

    ## plot enrichment map
    term2 = pre_res.res2d.Term
    axes = pre_res.plot(terms=term2[0])

    ## dotplot
    from gseapy import dotplot
    # to save your figure, make sure that ``ofname`` is not None
    ax = dotplot(pre_res.res2d,
                column="NOM p-val",
                title='',
                cmap=plt.cm.viridis,
                size=6, # adjust dot size
                top_term=20,
                figsize=(4,7), cutoff=0.25, show_ring=False) 

    return pre_res

def get_pathway_ranks(enr, enr_deg, enr_top_wDeg, enr_bottom_wDeg, rank_type='Adjusted P-value'):

    ## rank by adjusted p-value
    enr_sorted = enr.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
    enr_sorted_deg = enr_deg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
    enr_sorted_top_wDeg = enr_top_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
    enr_sorted_bottom_wDeg = enr_bottom_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')

    ## compute normalized ranks
    enr_sorted['normalized_rank'] = enr_sorted['rank'] / len(enr_sorted)
    enr_sorted_deg['normalized_rank'] = enr_sorted_deg['rank'] / len(enr_sorted_deg)
    enr_sorted_top_wDeg['normalized_rank'] = enr_sorted_top_wDeg['rank'] / len(enr_sorted_top_wDeg)
    enr_sorted_bottom_wDeg['normalized_rank'] = enr_sorted_bottom_wDeg['rank'] / len(enr_sorted_bottom_wDeg)

    pathways = [
        'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN',
        'ASTON_MAJOR_DEPRESSIVE_DISORDER_UP',
        'LU_AGING_BRAIN_DN',
        'LU_AGING_BRAIN_UP',
        'BLALOCK_ALZHEIMERS_DISEASE_DN',
        'BLALOCK_ALZHEIMERS_DISEASE_UP',
        'Gandal_2018_BipolarDisorder_Downregulated_Cortex',
        'Gandal_2018_BipolarDisorder_Upregulated_Cortex',
        ]

    pathway_ranks = pd.DataFrame(index=pathways, \
        columns=['ALL LR', 'DEG + Top LR', 'DEG + Bottom LR', 'DEG'])
    pathway_ranks.attrs['rank_type'] = rank_type


    def get_pathway_rank(enr_res2d, pathway, rank_type='Adjusted P-value', return_nan=True):

        if len(enr_res2d[enr_res2d['Term'] == pathway]) > 0:
            return enr_res2d[enr_res2d['Term'] == pathway][rank_type].values[0]
        else:
            if return_nan:
                return np.nan
            else:
                return max(enr_res2d[rank_type])

    for pathway in pathways:
        pathway_ranks.loc[pathway, 'ALL LR'] = get_pathway_rank(enr_sorted, pathway, rank_type)

        pathway_ranks.loc[pathway, 'DEG + Top LR'] = get_pathway_rank(enr_sorted_top_wDeg, pathway, rank_type)
        pathway_ranks.loc[pathway, 'DEG + Bottom LR'] = get_pathway_rank(enr_sorted_bottom_wDeg, pathway, rank_type)
        pathway_ranks.loc[pathway, 'DEG'] = get_pathway_rank(enr_sorted_deg, pathway, rank_type)

    return pathway_ranks

def plot_pathway_ranks(pathway_ranks, stem=True, save_path=None):

    # ─── 3. Sort pathways by "ALL LR" (optional) ───
    order = pathway_ranks["ALL LR"].sort_values().index.tolist()

    pathway_ranks = pathway_ranks.loc[order]

    pathway_ranks_long = (
        pathway_ranks
        .reset_index()
        .melt(id_vars="index", var_name="method", value_name="rank")
        .rename(columns={"index": "pathway"})
    )

    # ─── 4. Plot the stripplot ───
    fig, ax = plt.subplots(1, 2, figsize=(12, 4 + 0.5 * pathway_ranks.shape[0]), sharex=False, sharey=True)
    sns.stripplot(
        data=pathway_ranks_long,
        y="pathway",
        x="rank",
        hue="method",
        dodge=True,       # side‐by‐side dots per pathway
        jitter=False,     # no vertical jitter
        size=6,
        palette="tab10",
        ax=ax[0]
    )
    sns.stripplot(
        data=pathway_ranks_long,
        y="pathway",
        x="rank",
        hue="method",
        dodge=True,       # side‐by‐side dots per pathway
        jitter=False,     # no vertical jitter
        size=6,
        palette="tab10",
        legend=False,
        ax=ax[1]
    ); ax[1].set_xscale('log')

    if stem:

        # ─── Add horizontal stems from the rightmost edge to each marker ───
        for subax in ax:
            # 1) Record the current x‐axis limits:
            x_min, x_max = subax.get_xlim()

            # 2) Loop over each PathCollection in this subplot.
            #    Each collection corresponds to one hue level (one "method").
            for collec in subax.collections:
                # get_offsets() is an Nx2 array of (x, y) positions for every dot in that hue.
                offsets = collec.get_offsets()
                # Choose a stem color (e.g. light gray); you could also pull collec.get_facecolor()[0]
                stem_color = "lightgray"
                for (x_pt, y_pt) in offsets:
                    # draw a line from x_max → x_pt at y=y_pt
                    subax.plot(
                        [x_max, x_pt],        # x data: from right edge to the point
                        [y_pt, y_pt],         # y data: horizontal at the same y
                        color=stem_color,
                        linewidth=0.6,
                        alpha=0.7,
                        zorder=0  # put stems "underneath" the markers
                    )

            # 3) Re‐enforce the original x‐limits so the stems don't stretch the view
            subax.set_xlim(x_min, x_max)

    # ─── 5. Add horizontal dividing lines ───
    n_paths = len(pathways)
    for i in range(n_paths - 1):
        ypos = i + 0.5
        ax[0].axhline(y=ypos, color="lightgray", linewidth=2, linestyle="--")
        ax[1].axhline(y=ypos, color="lightgray", linewidth=2, linestyle="--")

    # ─── 6. Tidy up labels & legend ───
    ax[0].set_ylabel("Brain GMT Pathway")
    ax[0].set_xlabel(f"{pathway_ranks.attrs['rank_type']}")
    ax[1].set_xlabel(f"{pathway_ranks.attrs['rank_type']} (log scale)")
    ax[0].set_title("Per‐Pathway Ranks Across Methods")

    # Place legend outside so it doesn't overlap
    #ax[0].legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def do_enrichr(lr_filtered_type, pathways, filter_var='Adjusted P-value', outdir=None):

    enr = gp.enrichr(lr_filtered_type.index.to_list(),
                        gene_sets=pathways,
                        outdir=None)

    display(enr.res2d.sort_values('Adjusted P-value', ascending=True).head(20)[['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Combined Score', 'Genes']])

    ofname = None if outdir is None else os.path.join(outdir, f'enrichr_dotplot_{lr_filtered_type.attrs["type"]}.png')

    if len(enr.res2d) > 0:
        # dotplot
        max_pval = enr.res2d['Adjusted P-value'].max()
        fig = gp.dotplot(enr.res2d,
                column='Adjusted P-value',
                figsize=(3,7),
                title=f'{lr_filtered_type.attrs["sex"]} - {lr_filtered_type.attrs["celltype"]} ({lr_filtered_type.attrs["type"]})',
                cmap=plt.cm.winter,
                size=12, # adjust dot size
                cutoff=max_pval,
                top_term=15,
                show_ring=False,
                ofname=ofname)
        # Remove plt.show() - it's not thread-safe
        if ofname is not None:
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()

        enr_sig_pathways_df = enr.res2d[enr.res2d[filter_var] < 0.05]
        #enr_sig_pathways_padj = enr_sig_pathways_df['Adjusted P-value'].to_list()
        #enr_sig_pathways = enr_sig_pathways_df['Term'].to_list()
        #enr_sig_pathways_set = set(enr_sig_pathways) if enr_sig_pathways is not None else set()
        #enr_sig_pathways_genes = enr_sig_pathways_df['Genes'].str.split(';').to_list()

        return enr_sig_pathways_df
    else:
        print(f'No significant pathways found for {lr_filtered_type.attrs["type"]}')
        return None

def differential_grn_analysis(
        condition, sex, celltype,
        mdd_rna, mdd_atac,
        rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, rna_batch_key, atac_celltype_key, atac_condition_key, atac_sex_key, atac_subject_key, atac_batch_key,
        eclare_student_model, mean_grn_df, overlapping_subjects, subjects_by_condition_n_sex_df, cutoff=5025, ot_alignment_type='all', subdir=None
        ):

    mdd_rna_aligned = []
    mdd_atac_aligned = []

    all_rna_indices = []
    all_atac_indices = []

    if ot_alignment_type == 'subjects':

        for subject in subjects_by_condition_n_sex_df[condition, sex.lower()]:

            #print(f'sex: {sex} - celltype: {celltype} - condition: {condition} - DAR peaks: {mdd_atac.n_vars} - DEG genes: {mdd_rna.n_vars}')

            ## select cell indices
            rna_indices = pd.DataFrame({
                'is_celltype': mdd_rna.obs[rna_celltype_key].str.startswith(celltype),
                'is_condition': mdd_rna.obs[rna_condition_key].str.startswith(condition),
                'is_sex': mdd_rna.obs[rna_sex_key].str.lower().str.contains(sex.lower()),
                'is_subject': mdd_rna.obs[rna_subject_key] == subject # do not use startswith to avoid multiple subjects
            }).prod(axis=1).astype(bool).values.nonzero()[0]

            atac_indices = pd.DataFrame({
                'is_celltype': mdd_atac.obs[atac_celltype_key].str.startswith(celltype),
                'is_condition': mdd_atac.obs[atac_condition_key].str.startswith(condition),
                'is_sex': mdd_atac.obs[atac_sex_key].str.lower().str.contains(sex.lower()),
                'is_subject': mdd_atac.obs[atac_subject_key] == subject # do not use startswith to avoid multiple subjects
            }).prod(axis=1).astype(bool).values.nonzero()[0]

            all_rna_indices.append(pd.DataFrame(np.vstack([rna_indices, [subject]*len(rna_indices), [celltype]*len(rna_indices), [condition]*len(rna_indices), [sex]*len(rna_indices)]).T, columns=['index', 'subject', 'celltype', 'condition', 'sex']))
            all_atac_indices.append(pd.DataFrame(np.vstack([atac_indices, [subject]*len(atac_indices), [celltype]*len(atac_indices), [condition]*len(atac_indices), [sex]*len(atac_indices)]).T, columns=['index', 'subject', 'celltype', 'condition', 'sex']))

            assert len(rna_indices) > 0 and len(atac_indices) > 0, f"No indices found for sex: {sex} - celltype: {celltype} - condition: {condition} - subject: {subject}"

            ## sample indices
            if len(rna_indices) > cutoff:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
                _, sampled_indices = next(sss.split(rna_indices, mdd_rna.obs[rna_celltype_key].iloc[rna_indices]))
                rna_indices = rna_indices[sampled_indices]
            if len(atac_indices) > cutoff:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
                _, sampled_indices = next(sss.split(atac_indices, mdd_atac.obs[atac_celltype_key].iloc[atac_indices]))
                atac_indices = atac_indices[sampled_indices]

            ## sample data
            mdd_rna_sampled_group = mdd_rna[rna_indices]
            mdd_atac_sampled_group = mdd_atac[atac_indices]

            ## get latents
            rna_latents, atac_latents = get_latents(eclare_student_model, mdd_rna_sampled_group, mdd_atac_sampled_group, return_tensor=True)
            rna_latents = rna_latents.cpu()
            atac_latents = atac_latents.cpu()

            ## get logits - already normalized during clip loss, but need to normalize before to be consistent with Concerto
            rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
            atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
            student_logits = torch.matmul(atac_latents, rna_latents.T)

            a, b = torch.ones((len(atac_latents),)) / len(atac_latents), torch.ones((len(rna_latents),)) / len(rna_latents)

            cells_gap = np.abs(len(atac_latents) - len(rna_latents))

            ## if imbalance, use partial wasserstein to find cells to reject and resample accordingly
            if cells_gap > 0:
                print(f'cells_gap: {cells_gap} (atac: {len(atac_latents)} - rna: {len(rna_latents)})')
                mdd_atac_sampled_group, mdd_rna_sampled_group, student_logits = \
                    cell_gap_ot(student_logits, atac_latents, rna_latents, mdd_atac_sampled_group, mdd_rna_sampled_group, cells_gap, type='emd')

            ## compute optimal transport plan for alignment on remaining cells
            res = ot_solve(1 - student_logits)
            plan = res.plan
            value = res.value_linear

            ## re-order ATAC latents to match plan (can rerun OT analysis to ensure diagonal matching structure)
            atac_latents = atac_latents[plan.argmax(axis=0)]
            mdd_atac_sampled_group = mdd_atac_sampled_group[plan.argmax(axis=0).numpy()]

            ## append to list
            mdd_rna_aligned.append(mdd_rna_sampled_group)
            mdd_atac_aligned.append(mdd_atac_sampled_group)

        ## concatenate aligned anndatas
        mdd_rna_aligned = anndata.concat(mdd_rna_aligned, axis=0)
        mdd_atac_aligned = anndata.concat(mdd_atac_aligned, axis=0)

        assert np.equal(mdd_rna_aligned.obs[rna_subject_key].values, mdd_atac_aligned.obs[atac_subject_key].values).all()
        assert (mdd_rna_aligned.obs_names.nunique() == mdd_rna_aligned.n_obs) & (mdd_atac_aligned.obs_names.nunique() == mdd_atac_aligned.n_obs)

    elif ot_alignment_type == 'all':

        ## select cell indices
        rna_indices = pd.DataFrame({
            'is_celltype': mdd_rna.obs[rna_celltype_key].str.startswith(celltype),
            'is_condition': mdd_rna.obs[rna_condition_key].str.startswith(condition),
            'is_sex': mdd_rna.obs[rna_sex_key].str.lower().str.contains(sex.lower()),
            'is_subject': mdd_rna.obs[rna_subject_key].isin(overlapping_subjects)
        }).prod(axis=1).astype(bool).values.nonzero()[0]

        atac_indices = pd.DataFrame({
            'is_celltype': mdd_atac.obs[atac_celltype_key].str.startswith(celltype),
            'is_condition': mdd_atac.obs[atac_condition_key].str.startswith(condition),
            'is_sex': mdd_atac.obs[atac_sex_key].str.lower().str.contains(sex.lower()),
            'is_subject': mdd_atac.obs[atac_subject_key].isin(overlapping_subjects)
        }).prod(axis=1).astype(bool).values.nonzero()[0]

        ## sample indices
        if len(rna_indices) > cutoff:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
            _, sampled_indices = next(sss.split(rna_indices, mdd_rna.obs[rna_celltype_key].iloc[rna_indices]))
            rna_indices = rna_indices[sampled_indices]
        if len(atac_indices) > cutoff:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
            _, sampled_indices = next(sss.split(atac_indices, mdd_atac.obs[atac_celltype_key].iloc[atac_indices]))
            atac_indices = atac_indices[sampled_indices]

        ## sample data
        mdd_rna_sampled_group = mdd_rna[rna_indices]
        mdd_atac_sampled_group = mdd_atac[atac_indices]

        ## get latents
        rna_latents, atac_latents = get_latents(eclare_student_model, mdd_rna_sampled_group, mdd_atac_sampled_group, return_tensor=True)
        rna_latents = rna_latents.cpu()
        atac_latents = atac_latents.cpu()

        ## get logits - already normalized during clip loss, but need to normalize before to be consistent with Concerto
        rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
        atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
        student_logits = torch.matmul(atac_latents, rna_latents.T)

        a, b = torch.ones((len(atac_latents),)) / len(atac_latents), torch.ones((len(rna_latents),)) / len(rna_latents)

        cells_gap = np.abs(len(atac_latents) - len(rna_latents))

        ## if imbalance, use partial wasserstein to find cells to reject and resample accordingly
        if cells_gap > 0:
            print(f'cells_gap: {cells_gap} (atac: {len(atac_latents)} - rna: {len(rna_latents)})')
            mdd_atac_sampled_group, mdd_rna_sampled_group, student_logits = \
                cell_gap_ot(student_logits, atac_latents, rna_latents, mdd_atac_sampled_group, mdd_rna_sampled_group, cells_gap, type='emd')

        ## compute optimal transport plan for alignment on remaining cells
        res = ot_solve(1 - student_logits)
        plan = res.plan
        value = res.value_linear

        ## re-order ATAC latents to match plan (can rerun OT analysis to ensure diagonal matching structure)
        print((f'dimensions of plan: {plan.shape}'))
        atac_latents = atac_latents[plan.argmax(axis=0)]
        mdd_atac_sampled_group = mdd_atac_sampled_group[plan.argmax(axis=0).numpy()]

        ## append to list
        mdd_rna_aligned = mdd_rna_sampled_group
        mdd_atac_aligned = mdd_atac_sampled_group

    ## add var to aligned anndatas
    mdd_rna_aligned.var = mdd_rna.var
    mdd_atac_aligned.var = mdd_atac.var

    ## plot UMAP of aligned RNA latents and ATAC latents
    rna_latents, atac_latents = get_latents(eclare_student_model, mdd_rna_aligned, mdd_atac_aligned, return_tensor=True)
    umap_embedder, _, _ = plot_umap_embeddings(rna_latents, atac_latents, [celltype]*len(rna_latents), [celltype]*len(atac_latents), [condition]*len(rna_latents), [condition]*len(atac_latents), color_map_ct={celltype: 'black'}, title=f'{sex} - {celltype} - {condition}', umap_embedding=None, save_path=os.path.join(subdir, f'umap_rna_atac.png'))

    ## get unpaired metrics
    rna_atac_latents = torch.cat([rna_latents, atac_latents], dim=0)
    rna_atac_labels     = np.hstack([mdd_rna_aligned.obs[rna_celltype_key], mdd_atac_aligned.obs[atac_celltype_key]])
    rna_atac_batches    = np.hstack([mdd_rna_aligned.obs[rna_batch_key], mdd_atac_aligned.obs[atac_batch_key]])
    rna_atac_modalities = np.hstack([ np.repeat('rna', len(rna_latents)) , np.repeat('atac', len(atac_latents)) ])

    unpaired_metrics_dict = unpaired_metrics(rna_atac_latents, rna_atac_labels, rna_atac_modalities, rna_atac_batches)

    ## get mean GRN from brainSCOPE
    mean_grn_df, mdd_rna_sampled_group, mdd_atac_sampled_group = filter_mean_grn(mean_grn_df, mdd_rna_aligned, mdd_atac_aligned)

    overlapping_target_genes = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_target_gene']].index.values
    overlapping_tfs = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_tf']].index.values

    ## remove excess cells beyond cutoff with stratified sampling - or else SEACells way too slow
    if len(mdd_rna_sampled_group) > cutoff:
        print(f'Removing {len(mdd_rna_sampled_group) - cutoff} cells')

        sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
        _, sampled_indices = next(sss.split(mdd_rna_sampled_group.obs_names, mdd_rna_sampled_group.obs[rna_subject_key]))

        mdd_rna_sampled_group = mdd_rna_sampled_group[sampled_indices]
        mdd_atac_sampled_group = mdd_atac_sampled_group[sampled_indices]

    ## save original PCA and UMAP
    mdd_rna_sampled_group.obsm['X_pca_orig'] = mdd_rna_sampled_group.obsm['X_pca']
    mdd_rna_sampled_group.obsm['X_umap_orig'] = mdd_rna_sampled_group.obsm['X_UMAP_pca']

    ## PCA and UMAP
    sc.pp.pca(mdd_rna_sampled_group, n_comps=50)
    sc.pp.neighbors(mdd_rna_sampled_group, n_neighbors=15)
    sc.tl.umap(mdd_rna_sampled_group)

    sc.pp.pca(mdd_atac_sampled_group, n_comps=50)

    ## run SEACells to obtain pseudobulked counts
    mdd_rna_sampled_group_seacells, mdd_atac_sampled_group_seacells = \
        run_SEACells(mdd_rna_sampled_group, mdd_atac_sampled_group, build_kernel_on='X_pca', key='X_umap', save_dir=subdir)

    X_rna = torch.from_numpy(mdd_rna_sampled_group_seacells.X.toarray())
    X_atac = torch.from_numpy(mdd_atac_sampled_group_seacells.X.toarray())

    ## detect cells with no gene expression and no chromatin accessibility
    no_rna_cells = X_rna.sum(1) == 0
    no_atac_cells = X_atac.sum(1) == 0
    no_rna_atac_cells = no_rna_cells | no_atac_cells

    ## remove cells with no expression
    X_rna = X_rna[~no_rna_atac_cells]
    X_atac = X_atac[~no_rna_atac_cells]

    ## get tfrp
    scompreg_loglikelihoods, tg_expressions, tfrps, tfrp_predictions, slopes, intercepts, std_errs, intercept_stderrs = \
        get_scompreg_loglikelihood(mean_grn_df, X_rna, X_atac, overlapping_target_genes, overlapping_tfs)

    ## assign to dicts
    return_tuple = (sex, celltype, condition, mdd_rna_sampled_group_seacells, mdd_atac_sampled_group_seacells, overlapping_target_genes, overlapping_tfs, scompreg_loglikelihoods, std_errs, tg_expressions, tfrps, tfrp_predictions, slopes, intercepts, intercept_stderrs)
    return return_tuple

def compute_LR_grns(sex, celltype, mean_grn_df_filtered_dict, X_rna_dict, X_atac_dict, output_dir=None):

    X_rna_control = torch.from_numpy(X_rna_dict[sex][celltype]['Control'].X.toarray())
    X_atac_control = torch.from_numpy(X_atac_dict[sex][celltype]['Control'].X.toarray())

    X_rna_case = torch.from_numpy(X_rna_dict[sex][celltype]['Case'].X.toarray())
    X_atac_case = torch.from_numpy(X_atac_dict[sex][celltype]['Case'].X.toarray())

    X_rna_all = torch.cat([X_rna_control, X_rna_case], dim=0)
    X_atac_all = torch.cat([X_atac_control, X_atac_case], dim=0)

    mean_grn_df_filtered = mean_grn_df_filtered_dict[sex][celltype]

    overlapping_target_genes = mean_grn_df_filtered['TG'].unique()
    overlapping_tfs = mean_grn_df_filtered['TF'].unique()

    mean_grn_df_filtered, tfrps_control, tg_expressions_control = get_scompreg_loglikelihood_full(mean_grn_df_filtered, X_rna_control, X_atac_control, overlapping_target_genes, overlapping_tfs, 'll_control')
    mean_grn_df_filtered, tfrps_case, tg_expressions_case = get_scompreg_loglikelihood_full(mean_grn_df_filtered, X_rna_case, X_atac_case, overlapping_target_genes, overlapping_tfs, 'll_case')

    assert list(tfrps_control.keys()) == list(tfrps_case.keys())
    assert list(tg_expressions_control.keys()) == list(tg_expressions_case.keys())


    for gene in overlapping_target_genes:

        tfrps_control_gene = tfrps_control[gene]
        tfrps_case_gene = tfrps_case[gene]
        tg_expressions_control_gene = tg_expressions_control[gene]
        tg_expressions_case_gene = tg_expressions_case[gene]
        
        tfrps_all_gene = pd.concat([tfrps_control_gene, tfrps_case_gene], axis=1)
        tg_expressions_all_gene = torch.cat([tg_expressions_control_gene, tg_expressions_case_gene], dim=0).numpy()

        scompreg_likelihood_ratio_lambda = lambda tfrp: scompreg_likelihood_ratio(tg_expressions_all_gene, tfrp)
        tfrps_all_gene['scompreg_likelihood_ratio'] = tfrps_all_gene.apply(scompreg_likelihood_ratio_lambda, axis=1)

        mean_grn_df_filtered.loc[tfrps_all_gene.index, 'll_all'] = tfrps_all_gene['scompreg_likelihood_ratio']
        

    LR_grns = -2 * (mean_grn_df_filtered.apply(lambda grn: grn['ll_all'] - (grn['ll_case'] + grn['ll_control']), axis=1))
    LR_grns = LR_grns.dropna()

    Z_grns = pd.Series(np.zeros_like(LR_grns))
    LR_grns_up = pd.Series(np.zeros_like(LR_grns))
    LR_grns_down = pd.Series(np.zeros_like(LR_grns))

    save_dir = os.path.join(output_dir, f'{sex}_{celltype}') if output_dir is not None else None

    LR_grns_filtered, _, _, _, _ = filter_LR_stats(LR_grns, Z_grns, LR_grns_up, LR_grns_down, save_dir=save_dir)

    mean_grn_df_filtered['LR_grns'] = LR_grns
    mean_grn_df_filtered_pruned = mean_grn_df_filtered.loc[LR_grns_filtered.index]
    
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    LR_grns.hist(bins=100, ax=ax)

    #ax.axvline(x=lr_at_p, color='black', linestyle='--')
    ax.axvline(x=LR_grns.quantile(0.95), color='green', linestyle='--')

    ax.set_xlabel('LR')
    ax.set_ylabel('Frequency')
    ax.set_title('LR distribution')
    plt.show()
    '''

    return mean_grn_df_filtered_pruned

def perform_enrichr_comparison(sex, celltype, lr_filtered, lr_filtered_woDeg, top_lr_filtered_wDeg, bottom_lr_filtered_wDeg, pydeseq2_results_dict, mdd_rna_var_names, significant_genes, pathways, output_dir=os.environ['OUTPATH']):

    ## Perform EnrichR on different gene sets
    deg_df = pd.DataFrame(index=significant_genes)

    results_match_length = pydeseq2_results_dict[sex][celltype].sort_values('padj')[:len(lr_filtered)]
    deg_match_length_df = pd.DataFrame(index=mdd_rna_var_names[results_match_length.index.astype(int)])

    lr_filtered.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'lr_filtered'})
    lr_filtered_woDeg.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'lr_filtered_woDeg'})
    top_lr_filtered_wDeg.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'top_lr_filtered_wDeg'})
    bottom_lr_filtered_wDeg.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'bottom_lr_filtered_wDeg'})
    deg_df.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'deg_df'})
    deg_match_length_df.attrs.update({'sex': sex, 'celltype': celltype, 'type': 'deg_match_length_df'})

    enr_sig_df = do_enrichr(lr_filtered, pathways, outdir=output_dir)
    enr_deg_match_length_df = do_enrichr(deg_match_length_df, pathways, outdir=output_dir)
    enr_deg_df = do_enrichr(deg_df, pathways, outdir=output_dir) if deg_df.empty is False else None

    #enr_woDeg, enr_sig_pathways_woDeg_set, enr_sig_pathways_woDeg_padj = do_enrichr(lr_filtered_woDeg, pathways) if lr_filtered_woDeg.empty is False else (None, None, None)
    #enr_top_wDeg, enr_sig_pathways_top_wDeg_set, enr_sig_pathways_top_wDeg_padj = do_enrichr(top_lr_filtered_wDeg, pathways) if top_lr_filtered_wDeg.empty is False else (None, None, None)
    #enr_bottom_wDeg, enr_sig_pathways_bottom_wDeg_set, enr_sig_pathways_bottom_wDeg_padj = do_enrichr(bottom_lr_filtered_wDeg, pathways) if bottom_lr_filtered_wDeg.empty is False else (None, None, None)

    ## save enrs_dict
    enrs_dict = {
        'All LR': set(enr_sig_df['Term'].to_list()),
        'DEG (matched length)': set(enr_deg_match_length_df['Term'].to_list()),
    }
    if enr_deg_df is not None:
        enrs_dict.update({'DEG': set(enr_deg_df['Term'].to_list())})
    else:
        print('No gene set enrichment for DEG genes')

    venn(enrs_dict)
    venn_path = os.path.join(output_dir, f'venn_{sex}_{celltype}.png')
    plt.savefig(venn_path, bbox_inches='tight', dpi=150)
    plt.close()

    all_enrs_dict = {
        'All LR': enr_sig_df,
        'DEG (matched length)': enr_deg_match_length_df,
    }
    if enr_deg_df is not None:
        all_enrs_dict.update({'DEG': enr_deg_df})
    
    return all_enrs_dict

def perform_gene_set_enrichment(sex, celltype, scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, mean_grn_df, significant_genes_dict, mdd_rna_var_names, pydeseq2_results_dict, pathways, slopes_dict, std_errs_dict, intercepts_dict, intercept_stderrs_dict, subdir=None):

    ## get MDD-association gene scores
    LR, Z, LR_up, LR_down, Z_up, Z_down = compute_scompreg_loglikelihoods(
        sex, celltype,
        scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict,
        slopes_dict, std_errs_dict, intercepts_dict, intercept_stderrs_dict
    )

    ## filter LR values with fitted null distribution (sc-compReg)
    lr_filtered, Z_filtered, lr_up_filtered, lr_down_filtered, lr_fitted_cdf = filter_LR_stats(LR, Z, LR_up, LR_down, save_dir=subdir)

    ## Merge lr_filtered into mean_grn_df
    mean_grn_df_filtered = merge_grn_lr_filtered(mean_grn_df, lr_filtered, subdir)

    ## Get gene sets that include DEG genes from pyDESeq2 analysis
    top_lr_filtered_wDeg, bottom_lr_filtered_wDeg, lr_filtered_woDeg = get_deg_gene_sets(LR, lr_filtered, lr_fitted_cdf, significant_genes_dict[sex][celltype])

    ## run MAGMA
    magma_results_series, magma_results = run_magma(lr_filtered, Z, significant_genes_dict[sex][celltype], subdir, fuma_job_id=None) # set fuma_job_id to None to use MDD GWAS from H-MAGMA

    ## run EnrichR
    merged_dict = perform_enrichr_comparison(
        sex, celltype, lr_filtered, lr_filtered_woDeg, top_lr_filtered_wDeg, bottom_lr_filtered_wDeg,
        pydeseq2_results_dict, mdd_rna_var_names, significant_genes_dict[sex][celltype], pathways, output_dir=subdir
    )

    return merged_dict, magma_results_series, mean_grn_df_filtered

def find_hits_overlap(gene_list, enrs, shared_TF_TG_pairs_df):

    hits_df = pd.DataFrame(index=gene_list).assign(n_hits=0, terms_hits=None)

    for gene in gene_list:

        terms_hits = []
        for term_row in enrs.itertuples():

            term_name = term_row.Term
            term_genes = term_row.Genes
            term_genes_list = term_genes.split(';')

            if gene in term_genes_list:
                hits_df.loc[gene, 'n_hits'] += 1
                terms_hits.append(term_name)

        hits_df.loc[gene, 'terms_hits'] = '; '.join(terms_hits)

    hits_df.sort_values(by='n_hits', inplace=True, ascending=False)

    ## Merge hits_df with shared_TF_TG_pairs_df and find TFs with multiple hits
    hits_df = hits_df.merge(shared_TF_TG_pairs_df.groupby('TG').agg({'TF': list}), left_index=True, right_on='TG', how='left')

    # Create dummy encoded columns for each unique TF
    unique_tfs = np.unique(np.hstack(hits_df['TF'].values))
    for tf in unique_tfs:
        hits_df[f'TF_{tf}'] = hits_df['TF'].apply(lambda x: 1 if tf in x else 0)

    tfs_multiple_hits_bool = hits_df.loc[:, hits_df.columns.str.startswith('TF_')].sum(0) > 1
    tfs_multiple_hits_names = tfs_multiple_hits_bool[tfs_multiple_hits_bool].index.values
    tfs_multiple_hits = hits_df.loc[(hits_df.loc[:,tfs_multiple_hits_names] == 1).values.any(1)]
    #hits_df.to_csv(os.path.join(output_dir, 'hits_df.csv'), index=True, header=True)

    tfs_multiple_hits = tfs_multiple_hits.drop(columns=hits_df.columns[hits_df.columns.str.startswith('TF_')])

    return hits_df, tfs_multiple_hits

def set_env_variables(config_path='../config'):

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

        sys.path.insert(0, config_path)

        from export_env_variables import export_env_variables
        export_env_variables(config_path)

def download_mlflow_runs(experiment_name):
    
    # Initialize MLflow client with tracking URI if provided
    tracking_uri = os.path.join(os.environ['ECLARE_ROOT'], 'mlruns')
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Try to find experiment in alternative locations
        alt_tracking_uri = os.path.join(os.environ.get('OUTPATH', '.'), 'mlruns')
        if tracking_uri != alt_tracking_uri:
            client = MlflowClient(tracking_uri=alt_tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found. Set MLFLOW_TRACKING_URI environment variable to specify mlruns directory.")
    
    experiment_id = experiment.experiment_id
    
    # Get all runs for the experiment
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    # Convert runs to DataFrame
    runs_data = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'parent_run_id': run.data.tags.get('mlflow.parentRunId', None),
            'run_name': run.info.run_name,  # Include the name of the run
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time/1000.0),
            'end_time': datetime.fromtimestamp(run.info.end_time/1000.0) if run.info.end_time else None,
            'artifact_uri': run.info.artifact_uri,
            **run.data.params,
            **run.data.metrics
        }
        runs_data.append(run_data)
    
    runs_df = pd.DataFrame(runs_data)
    
    # Save to CSV
    output_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
    runs_df.to_csv(output_path, index=False)
    print(f"Saved {len(runs_df)} runs to {output_path}")
    
    return output_path

def process_celltype(sex, celltype, rna_scaled_with_counts, mdd_rna_var, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key):
    # Run pyDESeq2 on subject-level pseudo-replicates
    mdd_subjects_counts_adata, counts, metadata = get_pseudo_replicates_counts(
        sex, celltype, rna_scaled_with_counts, mdd_rna_var, 
        rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key,
        pseudo_replicates='Subjects', overlapping_only=False
    )
    return run_pyDESeq2(mdd_subjects_counts_adata, counts, metadata, rna_condition_key)

def magma_dicts_to_df(magma_results_dict):
    sex_dfs = {
        sex: pd.concat(cell_dict, axis=1).T
        for sex, cell_dict in magma_results_dict.items()
    }
    df = pd.concat(sex_dfs, names=['sex', 'celltype'])
    return df

def get_next_version_dir(base_dir):
    """Find the next available version number for the output directory"""
    version = 0
    while True:
        version_dir = f"{base_dir}_{version}"
        if not os.path.exists(version_dir):
            return version_dir
        version += 1