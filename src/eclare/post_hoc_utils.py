import os
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from umap import UMAP
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ot
import copy

from eclare import CLIP
from eclare import load_CLIP_model
from eclare import return_setup_func_from_dataset
from eclare import mdd_setup


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

    ## normalize
    rna = torch.nn.functional.normalize(rna, p=2, dim=1)
    atac = torch.nn.functional.normalize(atac, p=2, dim=1)

    #with torch.inference_mode():
    rna_latent = model(rna, modality='rna', task='align')[0].detach()
    atac_latent = model(atac, modality='atac', task='align')[0].detach()


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

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='condition', hue_order=hue_order, alpha=0.5, ax=ax[3], legend=True)
        ax[3].set_xticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xlabel(''); ax[3].set_ylabel('')
    else:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='celltypes', palette=color_map_ct, alpha=0.8, ax=ax[1], legend=False)
    sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, ax=ax[2], legend=True)

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
