import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from argparse import Namespace
from glob import glob
import mlflow
import torch
from anndata import read_h5ad
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from anndata import AnnData
import phate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scanpy as sc

from eclare.setup_utils import return_setup_func_from_dataset
#from eclare.data_utils import fetch_data_from_loader_light


if __name__ == '__main__':

    source_dataset = 'PFC_V1_Wang'
    target_dataset = 'Cortex_Velmeshev'

    args = Namespace(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        genes_by_peaks_str='6124_by_19914',
        ordinal_job_id='01134633',
    )
    '''
    args = Namespace(
        source_dataset=source_dataset,
        target_dataset=None,
        genes_by_peaks_str='9584_by_66620',
        ordinal_job_id='25195201',
    )
    '''
    '''
    source_dataset = 'PFC_Zhu'
    target_dataset = 'MDD'
    genes_by_peaks_str = '6816_by_55284'
    ordinal_job_id = '22130216'
    args = Namespace(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        genes_by_peaks_str=genes_by_peaks_str,
        ordinal_job_id=ordinal_job_id,
    )
    '''

    if torch.cuda.is_available():
        print('CUDA available')
        num_gpus = torch.cuda.device_count()
        device   = torch.device(f'cuda:{num_gpus-1}')
        torch.cuda.set_device(device.index)  # default device used by e.g. AnnLoader
    else:
        print('CUDA not available, set default to CPU')
        device   = 'cpu'

    ## SOURCE dataset setup function
    developmental_datasets = ['PFC_Zhu', 'PFC_V1_Wang', 'Cortex_Velmeshev']
    assert args.source_dataset in developmental_datasets, "ORDINAL only implemented for developmental datasets"
    source_setup_func = return_setup_func_from_dataset(args.source_dataset)

    ## TARGET dataset setup function
    assert args.target_dataset in developmental_datasets, "ORDINAL only implemented for developmental datasets"
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)

    ## load ordinal model
    ordinal_model_uri_paths_str = f'ordinal_*{args.ordinal_job_id}/model_uri.txt'
    ordinal_model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], ordinal_model_uri_paths_str))
    assert len(ordinal_model_uri_paths) > 0, f'Model URI path not found @ {ordinal_model_uri_paths_str}'

    with open(ordinal_model_uri_paths[0], 'r') as f:
        model_uris = f.read().strip().splitlines()
        model_uri = model_uris[0]

    mlflow.set_tracking_uri('file:///home/mcb/users/dmannk/scMultiCLIP/ECLARE/mlruns')
    ordinal_model = mlflow.pytorch.load_model(model_uri, device=device)

    ## Extract model metrics from MLflow
    # Parse run_id from model_uri (format: "runs:/<run_id>/model")
    run_id = model_uri.split('/')[1]
    
    # Get the run and extract metrics
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.get_run(run_id)
    
    # Extract only valid_mae_atac and valid_mae_rna metrics
    all_metrics = run.data.metrics
    target_metrics = ['valid_mae_atac', 'valid_mae_rna']
    model_metrics = {k: v for k, v in all_metrics.items() if k in target_metrics}
    print(f"Model metrics: {model_metrics}")

    ## define dev group and cell group keys and dev stages
    dev_group_key_dict = {
        'PFC_V1_Wang': 'dev_stage',
        'Cortex_Velmeshev': 'Age_Range',
        'PFC_Zhu': 'dev_stage',
        'MDD': 'Age'
        }
    cell_group_key_dict = {
        'PFC_V1_Wang': 'type',
        'Cortex_Velmeshev': 'Lineage',
        'PFC_Zhu': 'Cell type',
        'MDD': 'ClustersMapped'
        }
    dev_stages_dict = {
        'PFC_V1_Wang': ['FirstTrim', 'SecTrim', 'ThirdTrim', 'Inf', 'Adol'],
        'Cortex_Velmeshev': ['2nd trimester', '3rd trimester', '0-1 years', '1-2 years', '2-4 years', '4-10 years', '10-20 years', 'Adult'],
        'PFC_Zhu': ['EaFet', 'LaFet', 'Inf', 'Child', 'Adol', 'Adult']
        }

    ## load source and target data
    source_rna, source_atac, cell_group, _, _, _, _ = source_setup_func(args, return_type='data', keep_group=[''])
    target_rna, target_atac, cell_group, _, _, _, _ = target_setup_func(args, return_type='data', keep_group=[''])

    '''
    atac_fullpath = os.path.join(os.environ['DATAPATH'], args.source_dataset, 'atac', f'atac_{args.genes_by_peaks_str}.h5ad')
    rna_fullpath = os.path.join(os.environ['DATAPATH'], args.source_dataset, 'rna', f'rna_{args.genes_by_peaks_str}.h5ad')
    
    atac = read_h5ad(atac_fullpath, backed='r')
    rna  = read_h5ad(rna_fullpath, backed='r+')

    train_len = int(0.8*len(rna))
    valid_len = int(0.2*len(rna))
    train_valid_random_state = int(os.environ.get('RANDOM_STATE', 42))
    #train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=train_valid_random_state).split(X=np.empty(len(rna)), y=rna.obs[dev_group_key]))

    from imblearn.under_sampling import RandomUnderSampler
    from collections import Counter

    N_target = 10000

    classes = np.unique(rna.obs[dev_group_key])
    k = len(classes)
    per_class = N_target // k
    counts = Counter(rna.obs[dev_group_key])
    strategy = {c: min(counts[c], per_class) for c in classes}
    rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
    valid_idx_rna, valid_devs_rna = rus.fit_resample(np.arange(len(rna)).reshape(-1, 1), rna.obs[dev_group_key])

    #valid_idx_rna = valid_idx[::10]
    rna = rna[valid_idx_rna.flatten()].to_memory()

    classes = np.unique(atac.obs[dev_group_key])
    k = len(classes)
    per_class = N_target // k
    counts = Counter(atac.obs[dev_group_key])
    strategy = {c: min(counts[c], per_class) for c in classes}
    rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
    valid_idx_atac, valid_devs_atac = rus.fit_resample(np.arange(len(atac)).reshape(-1, 1), atac.obs[dev_group_key])

    #valid_idx_atac = valid_idx[::10]
    atac = atac[valid_idx_atac.flatten()].to_memory()
    '''

    if source_dataset == 'PFC_V1_Wang' and target_dataset is None:

        cell_group_key = cell_group_key_dict[source_dataset]

        ## map cell types
        cell_type_map = {0: "RG-vRG", 1: "RG-tRG", 2: "RG-oRG", 3: "IPC-EN", 4: "EN-newborn", 5: "EN-IT-immature", 6: "EN-L2_3-IT", 7: "EN-L4-IT", 8: "EN-L5-IT", 9: "EN-L6-IT", 10: "EN-non-IT-immature", 11: "EN-L5-ET", 12: "EN-L5_6-NP", 13: "EN-L6-CT", 14: "EN-L6b", 15: "IN-dLGE-immature", 16: "IN-CGE-immature", 17: "IN-CGE-VIP", 18: "IN-CGE-SNCG", 19: "IN-mix-LAMP5", 20: "IN-MGE-immature", 21: "IN-MGE-SST", 22: "IN-MGE-PV", 23: "IPC-glia", 24: "Astrocyte-immature", 25: "Astrocyte-protoplasmic", 26: "Astrocyte-fibrous", 27: "OPC", 28: "Oligodendrocyte-immature", 29: "Oligodendrocyte", 30: "Cajal–Retzius cell", 31: "Microglia", 32: "Vascular", 33: "Unknown"}
        source_rna.obs[cell_group_key] = source_rna.obs[cell_group_key].map(cell_type_map)
        source_atac.obs[cell_group_key] = source_atac.obs[cell_group_key].map(cell_type_map)

        ## map dev stages
        dev_stage_mapper = {0: 'FirstTrim', 1: 'SecTrim', 2:'ThirdTrim', 3:'Inf', 4:'Adol'}
        source_rna.obs[dev_group_key_dict.get(source_dataset)] = source_rna.obs[dev_group_key_dict.get(source_dataset)].map(dev_stage_mapper)
        source_atac.obs[dev_group_key_dict.get(source_dataset)] = source_atac.obs[dev_group_key_dict.get(source_dataset)].map(dev_stage_mapper)

    ## get cells, dev stages, batches
    source_rna_cells, source_rna_labels, source_rna_batches = torch.from_numpy(source_rna.X.toarray()), source_rna.obs[dev_group_key_dict.get(source_dataset)], None
    target_rna_cells, target_rna_labels, target_rna_batches = torch.from_numpy(target_rna.X.toarray()), target_rna.obs[dev_group_key_dict.get(target_dataset)], None
    source_atac_cells, source_atac_labels, source_atac_batches = torch.from_numpy(source_atac.X.toarray()), source_atac.obs[dev_group_key_dict.get(source_dataset)], None
    target_atac_cells, target_atac_labels, target_atac_batches = torch.from_numpy(target_atac.X.toarray()), target_atac.obs[dev_group_key_dict.get(target_dataset)], None

    ## get latents
    ordinal_model = ordinal_model.to('cpu')
    source_rna_logits, source_rna_probas, source_rna_latents = ordinal_model(source_rna_cells.to(device='cpu', dtype=torch.float32), modality=0, normalize=0)
    target_rna_logits, target_rna_probas, target_rna_latents = ordinal_model(target_rna_cells.to(device='cpu', dtype=torch.float32), modality=0, normalize=0)
    source_atac_logits, source_atac_probas, source_atac_latents = ordinal_model(source_atac_cells.to(device='cpu', dtype=torch.float32), modality=1, normalize=0)
    target_atac_logits, target_atac_probas, target_atac_latents = ordinal_model(target_atac_cells.to(device='cpu', dtype=torch.float32), modality=1, normalize=0)

    ## create anndata with latents
    X = np.concatenate([source_rna_latents.detach().cpu().numpy(), target_rna_latents.detach().cpu().numpy()], axis=0)
    obs = pd.concat([source_rna_labels.to_frame(name=dev_group_key_dict.get(source_dataset)), target_rna_labels.to_frame(name=dev_group_key_dict.get(target_dataset))], axis=0)
    obs['source_or_target'] = ['source'] * len(source_rna_labels) + ['target'] * len(target_rna_labels)
    obs['cell_type'] = source_rna.obs[cell_group_key_dict.get(source_dataset)].tolist() + target_rna.obs[cell_group_key_dict.get(target_dataset)].tolist()
    ordinal_rna = AnnData(X=X, obs=obs)

    X = np.concatenate([source_atac_latents.detach().cpu().numpy(), target_atac_latents.detach().cpu().numpy()], axis=0)
    obs = pd.concat([source_atac_labels.to_frame(name=dev_group_key_dict.get(source_dataset)), target_atac_labels.to_frame(name=dev_group_key_dict.get(target_dataset)  )], axis=0)
    obs['source_or_target'] = ['source'] * len(source_atac_labels) + ['target'] * len(target_atac_labels)
    obs['cell_type'] = source_atac.obs[cell_group_key_dict.get(source_dataset)].tolist() + target_atac.obs[cell_group_key_dict.get(target_dataset)].tolist()
    ordinal_atac = AnnData(X=X, obs=obs)

    ## get pseudotimes from CORAL layer - RNA
    source_rna_coral_prebias = ordinal_model.ordinal_layer_rna.coral_weights(source_rna_latents)
    source_rna_pt = torch.sigmoid(source_rna_coral_prebias).flatten().detach().cpu().numpy()

    target_rna_coral_prebias = ordinal_model.ordinal_layer_rna.coral_weights(target_rna_latents)
    target_rna_pt = torch.sigmoid(target_rna_coral_prebias).flatten().detach().cpu().numpy()

    ## get pseudotimes from CORAL layer - ATAC
    source_atac_coral_prebias = ordinal_model.ordinal_layer_atac.coral_weights(source_atac_latents)
    source_atac_pt = torch.sigmoid(source_atac_coral_prebias).flatten().detach().cpu().numpy()

    target_atac_coral_prebias = ordinal_model.ordinal_layer_atac.coral_weights(target_atac_latents)
    target_atac_pt = torch.sigmoid(target_atac_coral_prebias).flatten().detach().cpu().numpy()

    ## add pseudotimes to anndata
    ordinal_rna.obs['pseudotime'] = np.concatenate([source_rna_pt, target_rna_pt], axis=0)
    ordinal_atac.obs['pseudotime'] = np.concatenate([source_atac_pt, target_atac_pt], axis=0)

    ## reorder dev stages
    dev_stage_order = [
        '2nd trimester',
        'EaFet',
        'LaFet',
        '3rd trimester',
        '0-1 years',
        '1-2 years',
        '2-4 years',
        'Inf',
        '4-10 years',
        'Child',
        'Adol',
        '10-20 years',
        'Adult',
        ]
    #ordinal_rna.obs['dev_stage'] = ordinal_rna.obs['dev_stage'].cat.reorder_categories(dev_stage_order, ordered=True)
    #ordinal_rna.obs['dev_stage'] = pd.Categorical(ordinal_rna.obs[dev_group_key_dict.get(source_dataset)], categories=dev_stage_order, ordered=True)
    #ordinal_atac.obs['dev_stage'] = pd.Categorical(ordinal_atac.obs[dev_group_key_dict.get(source_dataset)], categories=dev_stage_order, ordered=True)

    ## subsample data
    subsample_factor = 2 if target_dataset=='MDD' else 100
    ordinal_rna = ordinal_rna[::subsample_factor]
    ordinal_atac = ordinal_atac[::subsample_factor]

    ## get and plot UMAPs
    target_ordinal_rna = ordinal_rna[ordinal_rna.obs['source_or_target']=='target']
    sc.pp.neighbors(target_ordinal_rna)
    sc.tl.umap(target_ordinal_rna, min_dist=0.01)
    sc.pl.umap(target_ordinal_rna, color=['source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    source_ordinal_rna = ordinal_rna[ordinal_rna.obs['source_or_target']=='source']
    sc.pp.neighbors(source_ordinal_rna)
    sc.tl.umap(source_ordinal_rna, min_dist=0.01)
    sc.pl.umap(source_ordinal_rna, color=['source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    #source_EN_lineage_ordinal_rna = source_ordinal_rna[source_ordinal_rna.obs['cell_type'].str.contains('RG|IPC|EN')]
    #sc.pl.umap(source_EN_lineage_ordinal_rna, color=['source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    ## get and plot UMAPs - ATAC
    target_ordinal_atac = ordinal_atac[ordinal_atac.obs['source_or_target']=='target']
    sc.pp.neighbors(target_ordinal_atac)
    sc.tl.umap(target_ordinal_atac, min_dist=0.01)
    sc.pl.umap(target_ordinal_atac, color=['source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    source_ordinal_atac = ordinal_atac[ordinal_atac.obs['source_or_target']=='source']
    sc.pp.neighbors(source_ordinal_atac)
    sc.tl.umap(source_ordinal_atac, min_dist=0.01)
    sc.pl.umap(source_ordinal_atac, color=['source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    def dev_figS1(
        source_ordinal_rna,
        target_ordinal_rna,
        source_ordinal_atac,
        target_ordinal_atac,
        source_dataset, target_dataset,
        model_metrics,
        manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'dev_figS1.pdf')):

        sc.settings._vector_friendly = True
        sc.settings.figdir = manuscript_figpath

        # Create a 2x4 figure: rows = modalities (RNA, ATAC)
        # columns = [source dev_group, source pseudotime, target dev_group, target pseudotime]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        size = 50
        
        # Row 0: RNA
        sc.pl.umap(source_ordinal_rna, color=dev_group_key_dict.get(source_dataset), ax=axes[0, 0], show=False, title=dev_group_key_dict.get(source_dataset), size=size)
        sc.pl.umap(source_ordinal_rna, color='pseudotime', cmap='viridis', ax=axes[0, 1], show=False, title='pseudotime', size=size)
        sc.pl.umap(target_ordinal_rna, color=dev_group_key_dict.get(target_dataset), ax=axes[0, 2], show=False, title=dev_group_key_dict.get(target_dataset), size=size)
        sc.pl.umap(target_ordinal_rna, color='pseudotime', cmap='viridis', ax=axes[0, 3], show=False, title='pseudotime', size=size)
        
        # Row 1: ATAC
        sc.pl.umap(source_ordinal_atac, color=dev_group_key_dict.get(source_dataset), ax=axes[1, 0], show=False, size=size)
        sc.pl.umap(source_ordinal_atac, color='pseudotime', cmap='viridis', ax=axes[1, 1], show=False, size=size)
        sc.pl.umap(target_ordinal_atac, color=dev_group_key_dict.get(target_dataset), ax=axes[1, 2], show=False, size=size)
        sc.pl.umap(target_ordinal_atac, color='pseudotime', cmap='viridis', ax=axes[1, 3], show=False, size=size)
        
        # Add row labels (a, b)
        axes[0, 0].text(-0.15, 1.1, 'a', transform=axes[0, 0].transAxes, 
                       fontsize=20, fontweight='bold', va='top', ha='right')
        axes[0, 2].text(-0.15, 1.1, 'b', transform=axes[0, 2].transAxes, 
                       fontsize=20, fontweight='bold', va='top', ha='right')
        
        # Add modality labels
        axes[0, 0].set_ylabel(f'RNA - MAE: {model_metrics["valid_mae_rna"]:.2f}', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylabel(f'ATAC - MAE: {model_metrics["valid_mae_atac"]:.2f}', fontsize=16, fontweight='bold')
        
        # add suptitle
        fig.suptitle(f'a. {source_dataset} - b. {target_dataset}', fontsize=20, fontweight='bold')
        
        # Increase tick label sizes
        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        manuscript_figpath = manuscript_figpath.replace('.pdf', f'_{source_dataset}_{target_dataset}.pdf')
        fig.savefig(manuscript_figpath, bbox_inches='tight', dpi=300)
        print(f'Saving figure to {manuscript_figpath}')
        plt.close()

    ## combine UMAPs
    import anndata as ad
    ordinal_adata = ad.concat([ordinal_rna, ordinal_atac], axis=0)
    ordinal_adata.obs['modality'] = ['RNA'] * len(ordinal_rna) + ['ATAC'] * len(ordinal_atac)
    
    sc.pp.neighbors(ordinal_adata)
    sc.tl.umap(ordinal_adata)
    sc.pl.umap(ordinal_adata, color=['modality', 'source_or_target', 'dev_stage', 'pseudotime', 'cell_type'], cmap='viridis', wspace=0.3)

    ## store pseudotimes in dataframes
    dev_stage_order = dev_stages

    rna_pt_df = rna_labels.to_frame(name=dev_group_key).assign(pseudotime=rna_pt)
    rna_pt_df[dev_group_key] = pd.Categorical(rna_pt_df[dev_group_key], categories=dev_stage_order, ordered=True)
    rna_pt_df[cell_group_key] = rna.obs[cell_group_key]
    rna_pt_df['Seurat_clusters'] = rna.obs['Seurat_clusters']
    rna_pt_df = rna_pt_df.sort_values(dev_group_key)

    atac_pt_df = atac_labels.to_frame(name=dev_group_key).assign(pseudotime=atac_pt)
    atac_pt_df[dev_group_key] = pd.Categorical(atac_pt_df[dev_group_key], categories=dev_stage_order, ordered=True)
    atac_pt_df[cell_group_key] = atac.obs[cell_group_key]
    atac_pt_df = atac_pt_df.sort_values(dev_group_key)

    pt_df_vertical = pd.concat([rna_pt_df.assign(modality='RNA'), atac_pt_df.assign(modality='ATAC')])
    pt_df_horizontal = pd.merge(rna_pt_df, atac_pt_df, left_index=True, right_index=True, suffixes=('_rna', '_atac'))

    pt_df_horizontal[cell_group_key] = pt_df_horizontal[f'{cell_group_key}_rna']
    pt_df_horizontal[dev_group_key] = pt_df_horizontal[f'{dev_group_key}_rna']
    pt_df_horizontal = pt_df_horizontal.drop([f'{dev_group_key}_rna', f'{dev_group_key}_atac', f'{cell_group_key}_rna', f'{cell_group_key}_atac'], axis=1)

    #pt_df_vertical['pseudotime'] = pt_df_vertical['pseudotime'].apply(lambda x: np.sign(x) * np.log1p(np.abs(x)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    sns.histplot(pt_df_vertical, x='pseudotime', stat='count', hue=dev_group_key, bins=80, alpha=1., ax=axs[0])
    sns.histplot(pt_df_vertical, x='pseudotime', stat='count', hue='modality', bins=80, alpha=0.5, ax=axs[1])
    plt.show()

    g = sns.jointplot(pt_df_horizontal, x='pseudotime_rna', y='pseudotime_atac', hue=dev_group_key, kind='hist', bins=30, legend=False)

    ## plot dev stage and modality counts
    fig, ax = plt.subplots(3,1, figsize=[4,9], sharex=True)
    pt_df_vertical[['Age_Range', 'modality']].value_counts(normalize=True).plot(kind='barh', ax=ax[0])
    pt_df_vertical[pt_df_vertical['modality']=='RNA'][['Age_Range', 'modality']].value_counts(normalize=True).plot(kind='barh', ax=ax[1])
    pt_df_vertical[pt_df_vertical['modality']=='ATAC'][['Age_Range', 'modality']].value_counts(normalize=True).plot(kind='barh', ax=ax[2])
    plt.show()

    ## plot difference in pseudotimes
    diff = pt_df_horizontal['pseudotime_rna'] - pt_df_horizontal['pseudotime_atac']
    pt_df_horizontal['diff'] = diff
    pt_df_horizontal['diff_binary'] = (diff > 0)

    pt_df_horizontal.hist(column='diff', by=dev_group_key, layout=(1,5), figsize=[10,3], density=True, cumulative=False, bins=20)
    plt.suptitle('Difference in pseudotime: RNA - ATAC')
    plt.tight_layout()
    plt.show()

    sns.barplot(pt_df_horizontal, x=dev_group_key, y='diff', hue=dev_group_key)
    sns.barplot(pt_df_horizontal, y='diff')

    # Sort cell_group values by the mean of 'pseudotime'
    cell_group_order = pt_df_vertical.groupby(cell_group_key)['pseudotime'].mean().sort_values().index
    plt.figure(figsize=[5,10])
    sns.barplot(pt_df_vertical, y=cell_group_key, x='pseudotime', orient="y", order=cell_group_order)
    plt.show()

    ## get Seurat clusters
    Seurat_pseudotime_map = rna_pt_df.groupby('Seurat_clusters')['pseudotime'].mean().sort_values()
    Seurat_cell_group_map = rna.obs[['Seurat_clusters', cell_group_key]].groupby('Seurat_clusters').apply(lambda x: np.unique(x).item())
    Seurat_pseudotime_map.name = 'pseudotime'
    Seurat_cell_group_map.name = 'cell_group'
    Seurat_df = pd.merge(Seurat_cell_group_map, Seurat_pseudotime_map, left_index=True, right_index=True, how='left')

    ## create phate embeddings
    cmap_colors = plt.get_cmap('plasma', len(ordinal_rna.obs[dev_group_key].unique()))
    cmap = {group: cmap_colors(i) for i, group in enumerate(dev_stages)}

    phate_op = phate.PHATE(
        k=50,                # denser graph
        n_pca=10,            # moderate denoising
        verbose=True
    )

    phate_op.fit(ordinal_rna.X)
    phate.plot.scatter2d(phate_op, c=rna_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

    phate_op.fit(ordinal_atac.X)
    phate.plot.scatter2d(phate_op, c=atac_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)


cell_type_branches = {
    "RG-vRG": "0",
    "IPC-EN": "1",
    "EN-newborn": "2",
    "EN-IT-immature": "3.IT",
    "EN-non-IT-immature": "3.non-IT",
    "EN-L2/3-IT": "4.IT.BP3",
    "EN-L4-IT-V1": "4.IT.BP3",
    "EN-L4-IT": "4.IT.BP4",
    "EN-L5-IT": "4.IT.BP4",
    "EN-L6-IT": "4.IT.BP2",
    "EN-L5/6-NP": "4.non-IT.BP5",
    "EN-L5-ET": "4.non-IT.BP5",
    "EN-L6-CT": "4.non-IT.BP5",
    "EN-L6b": "4.non-IT.BP5"
}
