from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import torch
from umap import UMAP
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ot
import pickle as pkl
from scipy.special import softmax

import jax.numpy as jnp
from ott.geometry import costs as ot_costs
from ott.problems.linear.barycenter_problem import FreeBarycenterProblem
from ott.solvers.linear.continuous_barycenter import FreeWassersteinBarycenter
from ott.problems.quadratic.gw_barycenter import GWBarycenterProblem
from ott.solvers.quadratic.gw_barycenter import GromovWassersteinBarycenter

from models import load_scTripletgrate_model
from setup_utils import return_setup_func_from_dataset
from setup_utils import \
    snMultiome_388_human_brains_one_subject_setup, snMultiome_388_human_brains_setup, mdd_setup, Roussos_cerebral_cortex_setup, human_dlpfc_setup


def get_model_and_data(model_path, load_mdd=False):

    ## Load the model
    model, model_args_dict = load_scTripletgrate_model(model_path, device='cpu')

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

    rna = torch.from_numpy(rna.X.toarray()).float()
    atac = torch.from_numpy(atac.X.toarray()).float()

    with torch.inference_mode():
        _, rna_latent = model(rna, modality='rna', task='pretrain')
        _, atac_latent = model(atac, modality='atac', task='pretrain')

    if return_tensor:
        rna_latent, atac_latent = rna_latent.detach(), atac_latent.detach()
    else:
        rna_latent, atac_latent = rna_latent.detach().numpy(), atac_latent.detach().numpy()

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


def plot_umap_embeddings(rna_latents, atac_latents, rna_celltypes, atac_celltypes, rna_condition, atac_condition, color_map_ct, umap_embedding=None):

    #rna_celltypes = rna.obs['ClustersMapped'].values
    #atac_celltypes = atac.obs['ClustersMapped'].values

    #rna_condition = rna.obs['Condition'].values
    #atac_condition = atac.obs['condition'].values

    if umap_embedding is None:
        rna_atac_latents = np.concatenate([rna_latents, atac_latents], axis=0)
        umap_embedding = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        umap_embedding.fit(rna_atac_latents)

    rna_umap = umap_embedding.transform(rna_latents)
    atac_umap = umap_embedding.transform(atac_latents)

    rna_df_umap = pd.DataFrame(data={'umap_1': rna_umap[:, 0], 'umap_2': rna_umap[:, 1], 'celltypes': rna_celltypes, 'condition': rna_condition, 'modality': 'RNA'})
    atac_df_umap = pd.DataFrame(data={'umap_1': atac_umap[:, 0], 'umap_2': atac_umap[:, 1], 'celltypes': atac_celltypes, 'condition': atac_condition, 'modality': 'ATAC'})
    rna_atac_df_umap = pd.concat([rna_df_umap, atac_df_umap], axis=0)#.sample(frac=1) # shuffle

    multiple_conditions = (len(np.unique(rna_condition)) > 1) or (len(np.unique(atac_condition)) > 1)
    if multiple_conditions:
        hue_order = np.unique(np.concatenate([rna_condition, atac_condition]))

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='condition', hue_order=hue_order, alpha=0.5, ax=ax[2], legend=True)
        ax[2].set_xticklabels([]); ax[2].set_yticklabels([]); ax[2].set_xlabel(''); ax[2].set_ylabel('')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='celltypes', palette=color_map_ct, alpha=0.8, ax=ax[0], legend=False)
    sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, ax=ax[1], legend=True)

    ax[0].set_xticklabels([]); ax[0].set_yticklabels([]); ax[1].set_xticklabels([]); ax[1].set_yticklabels([])
    ax[0].set_xlabel(''); ax[0].set_ylabel(''); ax[1].set_xlabel(''); ax[1].set_ylabel('')

    fig.tight_layout()

    return umap_embedding

def plot_umap_embeddings_multidatasets(rna_latents_dict, atac_latents_dict, mdd_rna_latents_dict, mdd_atac_latents_dict, umap_embedding=None, split_plots='domains_and_modalities'):

    all_source_datasets = np.concatenate([np.repeat(dataset, latents.shape[0]) for dataset, latents in rna_latents_dict.items()] + \
                                        [np.repeat(dataset, latents.shape[0]) for dataset, latents in atac_latents_dict.items()], axis=0)
    
    all_mdd_datasets = np.concatenate([np.repeat(dataset, latents.shape[0]) for dataset, latents in mdd_rna_latents_dict.items()] + \
                                        [np.repeat(dataset, latents.shape[0]) for dataset, latents in mdd_atac_latents_dict.items()], axis=0)

    source_latents_dict = { 'RNA': np.concatenate([rna_latents for rna_latents in rna_latents_dict.values()], axis=0),
                            'ATAC': np.concatenate([atac_latents for atac_latents in atac_latents_dict.values()], axis=0) }
    
    mdd_latents_dict = { 'RNA': np.concatenate([mdd_rna_latents for mdd_rna_latents in mdd_rna_latents_dict.values()], axis=0),
                         'ATAC': np.concatenate([mdd_atac_latents for mdd_atac_latents in mdd_atac_latents_dict.values()], axis=0) }

    all_source_latents = np.concatenate([latents for latents in source_latents_dict.values()], axis=0)
    all_mdd_latents = np.concatenate([latents for latents in mdd_latents_dict.values()], axis=0)

    all_rna_latents = np.concatenate([rna_latents for rna_latents in rna_latents_dict.values()], axis=0)
    all_atac_latents = np.concatenate([atac_latents for atac_latents in atac_latents_dict.values()], axis=0)

    all_source_modalities = np.concatenate([np.repeat(dataset, latents.shape[0]) for dataset, latents in source_latents_dict.items()], axis=0)
    all_mdd_modalities = np.concatenate([np.repeat(dataset, latents.shape[0]) for dataset, latents in mdd_latents_dict.items()], axis=0)

    ## combine latents across "source" and "MDD" domains
    all_latents = np.concatenate([all_source_latents, all_mdd_latents], axis=0)
    all_datasets = np.concatenate([all_source_datasets, all_mdd_datasets], axis=0)
    all_modalities = np.concatenate([all_source_modalities, all_mdd_modalities], axis=0)
    all_domains = np.concatenate([np.repeat('source', all_source_latents.shape[0]), np.repeat('MDD', all_mdd_latents.shape[0])], axis=0)

    if umap_embedding is None:
        umap_embedding = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        umap_embedding.fit(all_latents)

    all_latents_umap = umap_embedding.transform(all_latents)
    all_df_umap = pd.DataFrame(data={'umap_1': all_latents_umap[:, 0], 'umap_2': all_latents_umap[:, 1], 'dataset': all_datasets, 'modality': all_modalities, 'domain': all_domains})

    '''
    all_source_umap = umap_embedding.transform(all_source_latents)
    all_mdd_umap = umap_embedding.transform(all_mdd_latents)
    all_source_df_umap = pd.DataFrame(data={'umap_1': all_source_umap[:, 0], 'umap_2': all_source_umap[:, 1], 'dataset': all_source_datasets, 'modality': all_source_modalities, 'domain': 'source'})
    all_mdd_df_umap = pd.DataFrame(data={'umap_1': all_mdd_umap[:, 0], 'umap_2': all_mdd_umap[:, 1], 'dataset': all_mdd_datasets, 'modality': all_mdd_modalities, 'domain': 'MDD'})
    all_df_umap = pd.concat([all_source_df_umap, all_mdd_df_umap], axis=0)
    '''

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    if split_plots == 'domains_and_modalities':
        sns.scatterplot(data=all_df_umap.loc[all_df_umap.domain=='source'], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[0,0], legend=True)
        sns.scatterplot(data=all_df_umap.loc[all_df_umap.domain=='MDD'], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[0,1], legend=True)

        sns.scatterplot(data=all_df_umap.loc[all_df_umap.modality=='RNA'], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[1,0], legend=True)
        sns.scatterplot(data=all_df_umap.loc[all_df_umap.modality=='ATAC'], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[1,1], legend=True)

        ax[0,0].set_title('Source nuclei')
        ax[0,1].set_title('MDD nuclei')
        ax[1,0].set_title('RNA nuclei')
        ax[1,1].set_title('ATAC nuclei')

    elif split_plots == 'all':
        sns.scatterplot(data=all_df_umap[(all_df_umap.domain=='source') * (all_df_umap.modality=='RNA')], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[0,0], legend=True)
        sns.scatterplot(data=all_df_umap[(all_df_umap.domain=='source') * (all_df_umap.modality=='ATAC')], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[0,1], legend=True)
        sns.scatterplot(data=all_df_umap[(all_df_umap.domain=='MDD') * (all_df_umap.modality=='RNA')], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[1,0], legend=True)
        sns.scatterplot(data=all_df_umap[(all_df_umap.domain=='MDD') * (all_df_umap.modality=='ATAC')], x='umap_1', y='umap_2', hue='dataset', alpha=0.5, ax=ax[1,1], legend=True)

        ax[0,0].set_title('Source RNA')
        ax[0,1].set_title('Source ATAC')
        ax[1,0].set_title('MDD RNA')
        ax[1,1].set_title('MDD ATAC')

    ax[0,0].set_xticklabels([]); ax[0,0].set_yticklabels([]); ax[0,0].set_xlabel(''); ax[0,0].set_ylabel('')
    ax[0,1].set_xticklabels([]); ax[0,1].set_yticklabels([]); ax[0,1].set_xlabel(''); ax[0,1].set_ylabel('')
    ax[1,0].set_xticklabels([]); ax[1,0].set_yticklabels([]); ax[1,0].set_xlabel(''); ax[1,0].set_ylabel('')
    ax[1,1].set_xticklabels([]); ax[1,1].set_yticklabels([]); ax[1,1].set_xlabel(''); ax[1,1].set_ylabel('')

    fig.tight_layout()

    return umap_embedding

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

    sort_indices_1d = torch.flip(similarities_masked.flatten().argsort(), dims=(0,))
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


def MSDA_loss_and_metrics_plot(slurm_id, ignore_losses=['cLISI'], remove_MDD=True):

    data = pd.read_csv(f'/Users/dmannk/cisformer/outputs/multisource_align_{slurm_id}/training_log.csv', index_col=0)
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