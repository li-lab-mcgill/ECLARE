from argparse import Namespace
import os
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

from eclare.setup_utils import return_setup_func_from_dataset
#from eclare.data_utils import fetch_data_from_loader_light


if __name__ == '__main__':

    args = Namespace(
        source_dataset='PFC_V1_Wang',
        target_dataset=None,
        genes_by_peaks_str='9914_by_63404',
        ordinal_job_id='20180433'
    )

    if torch.cuda.is_available():
        print('CUDA available')
        num_gpus = torch.cuda.device_count()
        device   = torch.device(f'cuda:{num_gpus-1}')
        torch.cuda.set_device(device.index)  # default device used by e.g. AnnLoader
    else:
        print('CUDA not available, set default to CPU')
        device   = 'cpu'

    ## SOURCE dataset setup function
    developmental_datasets = ['PFC_Zhu', 'PFC_V1_Wang']
    assert args.source_dataset in developmental_datasets, "ORDINAL only implemented for developmental datasets"
    source_setup_func = return_setup_func_from_dataset(args.source_dataset)

    ## load ordinal model
    ordinal_model_uri_paths_str = f'ordinal_*{args.ordinal_job_id}/model_uri.txt'
    ordinal_model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], ordinal_model_uri_paths_str))
    assert len(ordinal_model_uri_paths) > 0, f'Model URI path not found @ {ordinal_model_uri_paths_str}'

    with open(ordinal_model_uri_paths[0], 'r') as f:
        model_uris = f.read().strip().splitlines()
        model_uri = model_uris[0]

    mlflow.set_tracking_uri('file:///home/mcb/users/dmannk/scMultiCLIP/ECLARE/mlruns')
    ordinal_model = mlflow.pytorch.load_model(model_uri, device=device)

    ## load data
    atac_fullpath = os.path.join(os.environ['DATAPATH'], args.source_dataset, 'atac', f'atac_{args.genes_by_peaks_str}.h5ad')
    rna_fullpath = os.path.join(os.environ['DATAPATH'], args.source_dataset, 'rna', f'rna_{args.genes_by_peaks_str}.h5ad')

    atac = read_h5ad(atac_fullpath, backed='r')
    rna  = read_h5ad(rna_fullpath, backed='r+')

    train_len = int(0.8*len(rna))
    valid_len = int(0.2*len(rna))
    train_valid_random_state = int(os.environ.get('RANDOM_STATE', 42))
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=train_valid_random_state).split(X=np.empty(len(rna)), y=rna.obs['Group']))
    valid_idx = valid_idx[::10]

    atac = atac[valid_idx].to_memory()
    rna = rna[valid_idx].to_memory()

    ## map cell types
    cell_type_map = {0: "RG-vRG", 1: "RG-tRG", 2: "RG-oRG", 3: "IPC-EN", 4: "EN-newborn", 5: "EN-IT-immature", 6: "EN-L2_3-IT", 7: "EN-L4-IT", 8: "EN-L5-IT", 9: "EN-L6-IT", 10: "EN-non-IT-immature", 11: "EN-L5-ET", 12: "EN-L5_6-NP", 13: "EN-L6-CT", 14: "EN-L6b", 15: "IN-dLGE-immature", 16: "IN-CGE-immature", 17: "IN-CGE-VIP", 18: "IN-CGE-SNCG", 19: "IN-mix-LAMP5", 20: "IN-MGE-immature", 21: "IN-MGE-SST", 22: "IN-MGE-PV", 23: "IPC-glia", 24: "Astrocyte-immature", 25: "Astrocyte-protoplasmic", 26: "Astrocyte-fibrous", 27: "OPC", 28: "Oligodendrocyte-immature", 29: "Oligodendrocyte", 30: "Cajal–Retzius cell", 31: "Microglia", 32: "Vascular", 33: "Unknown"}
    atac.obs['type'] = atac.obs['type'].map(cell_type_map)
    rna.obs['type'] = rna.obs['type'].map(cell_type_map)

    ## map dev stages
    dev_stage_mapper = {0: 'FirstTrim', 1: 'SecTrim', 2:'ThirdTrim', 3:'Inf', 4:'Adol'}
    rna.obs['Group'] = rna.obs['Group'].map(dev_stage_mapper)
    atac.obs['Group'] = atac.obs['Group'].map(dev_stage_mapper)

    ## get cells, dev stages, batches
    rna_cells, rna_labels, rna_batches = torch.from_numpy(rna.X.toarray()), rna.obs['Group'], None
    atac_cells, atac_labels, atac_batches = torch.from_numpy(atac.X.toarray()), atac.obs['Group'], None

    ## get latents
    rna_logits, rna_probas, rna_latents = ordinal_model(rna_cells.to(device=device, dtype=torch.float32), modality=0, normalize=0)
    atac_logits, atac_probas, atac_latents = ordinal_model(atac_cells.to(device=device, dtype=torch.float32), modality=1, normalize=0)

    ## create anndata with latents
    ordinal_rna = AnnData(rna_latents.detach().cpu().numpy())
    ordinal_rna.obs['Group'] = rna_labels.values
    ordinal_atac = AnnData(atac_latents.detach().cpu().numpy())
    ordinal_atac.obs['Group'] = atac_labels.values

    ## create phate embeddings
    cmap_colors = plt.get_cmap('plasma', len(ordinal_rna.obs['Group'].unique()))
    cmap = {group: cmap_colors(i) for i, group in enumerate(['FirstTrim', 'SecTrim', 'ThirdTrim', 'Inf', 'Adol'])}

    phate_op = phate.PHATE(
        k=50,                # denser graph
        n_pca=10,            # moderate denoising
        verbose=True
    )

    phate_op.fit(ordinal_rna.X)
    phate.plot.scatter2d(phate_op, c=rna_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

    phate_op.fit(ordinal_atac.X)
    phate.plot.scatter2d(phate_op, c=atac_labels, xticklabels=False, yticklabels=False, xlabel='PHATE 1', ylabel='PHATE 2', cmap=cmap)

    ## get pseudotimes from CORAL layer
    rna_coral_prebias = ordinal_model.ordinal_layer_rna.coral_weights(rna_latents)
    rna_pt = torch.sigmoid(rna_coral_prebias).flatten().detach().cpu().numpy()

    atac_coral_prebias = ordinal_model.ordinal_layer_atac.coral_weights(atac_latents)
    atac_pt = torch.sigmoid(atac_coral_prebias).flatten().detach().cpu().numpy()

    ## store pseudotimes in dataframes
    dev_stage_order = ['FirstTrim', 'SecTrim', 'ThirdTrim', 'Inf', 'Adol']

    rna_pt_df = rna_labels.to_frame(name='Group').assign(pseudotime=rna_pt)
    rna_pt_df['Group'] = pd.Categorical(rna_pt_df['Group'], categories=dev_stage_order, ordered=True)
    rna_pt_df['type'] = rna.obs['type']
    rna_pt_df = rna_pt_df.sort_values('Group')

    atac_pt_df = atac_labels.to_frame(name='Group').assign(pseudotime=atac_pt)
    atac_pt_df['Group'] = pd.Categorical(atac_pt_df['Group'], categories=dev_stage_order, ordered=True)
    atac_pt_df['type'] = atac.obs['type']
    atac_pt_df = atac_pt_df.sort_values('Group')

    pt_df_vertical = pd.concat([rna_pt_df.assign(modality='RNA'), atac_pt_df.assign(modality='ATAC')])
    pt_df_horizontal = pd.merge(rna_pt_df, atac_pt_df, left_index=True, right_index=True, suffixes=('_rna', '_atac'))

    pt_df_horizontal['type'] = pt_df_horizontal['type_rna']
    pt_df_horizontal['Group'] = pt_df_horizontal['Group_rna']
    pt_df_horizontal = pt_df_horizontal.drop(['Group_rna', 'Group_atac', 'type_rna', 'type_atac'], axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    sns.histplot(pt_df_vertical, x='pseudotime', hue='Group', bins=80, alpha=1., ax=axs[0])
    sns.histplot(pt_df_vertical, x='pseudotime', hue='modality', bins=80, alpha=0.5, ax=axs[1])
    plt.show()

    g = sns.jointplot(pt_df_horizontal, x='pseudotime_rna', y='pseudotime_atac', hue='Group', kind='hist', bins=30, legend=False)


    diff = pt_df_horizontal['pseudotime_rna'] - pt_df_horizontal['pseudotime_atac']
    pt_df_horizontal['diff'] = diff
    pt_df_horizontal['diff_binary'] = (diff > 0)

    pt_df_horizontal.hist(column='diff', by='Group', layout=(1,5), figsize=[10,3], density=True, cumulative=False, bins=20)
    plt.suptitle('Difference in pseudotime: RNA - ATAC')
    plt.tight_layout()
    plt.show()

    sns.barplot(pt_df_horizontal, x='Group', y='diff', hue='Group')
    sns.barplot(pt_df_horizontal, y='diff')

    # Sort 'type' values by the mean of 'pseudotime'
    type_order = pt_df_vertical.groupby('type')['pseudotime'].mean().sort_values().index
    plt.figure(figsize=[5,10])
    sns.barplot(pt_df_vertical, y='type', x='pseudotime', orient="y", order=type_order)
    plt.show()
