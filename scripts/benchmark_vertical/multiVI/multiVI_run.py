import os
import sys
from numpy import any as np_any
import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    default_outdir = outpath = '/home/dmannk/scratch/'
    CLARE_root = '/home/dmannk/projects/def-liyue/dmannk/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    default_outdir = outpath = '/Users/dmannk/cisformer/outputs/'
    CLARE_root = '/Users/dmannk/cisformer/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root  ## not path to CLARE, but to the directory right before that also includes data

elif np_any([mcb_server in hostname for mcb_server in ['mcb', 'buckeridge' ,'hlr', 'ri', 'wh', 'yl']]):
    os.environ['machine'] = 'mcb'
    default_outdir = outpath = '/home/mcb/users/dmannk/scMultiCLIP/outputs'
    CLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root

#import pooch
import anndata as ad
import scanpy as sc
import scvi
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import empty as np_empty
from numpy import array as np_array
import pandas as pd
from numpy import arange as np_arange
import celltypist

from setup_utils import return_setup_func_from_dataset, retain_feature_overlap
from eval_utils import align_metrics

from argparse import ArgumentParser

parser = ArgumentParser(description='')
parser.add_argument('--outdir', type=str, default=default_outdir,
                    help='output directory')
parser.add_argument('--n_epochs', type=int, default=2,
                    help='number of epochs')
parser.add_argument('--source_dataset', type=str, default='roussos',
                    help='dataset to use')
parser.add_argument('--target_dataset', type=str, default='mdd',
                    help='target dataset')
parser.add_argument('--save_latents', action='store_true',
                    help='save latents during training')
parser.add_argument('--valid_subsample', type=int, default=5000,
                    help='number of nuclei to subsample for validation')
parser.add_argument('--genes_by_peaks_str', type=str, default=None,  # '6816_by_55284' for roussos
                    help='genes by peaks string')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

if torch.cuda.is_available():
    print('CUDA available')
    device   = 'cuda'
    num_gpus = torch.cuda.device_count()
else:
    print('CUDA not available, set default to CPU')
    device   = 'cpu'

## Check number of cpus (does not work in interactive SALLOC environment)
cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
if cpus_per_task:
    cpus_per_task = int(cpus_per_task)
else:
    cpus_per_task = 1  # Default to 1 if not set

print(f"Allocated CPUs: {cpus_per_task}")

def setup(rna, atac, cell_group, source_or_target):

    #del rna_hvg, atac_hvg
    rna.var['modality'] = 'Gene Expression'
    atac.var['modality'] = 'Peaks'

    ## Obtain cell types and obtain train-valid split indices
    assert (atac.obs[cell_group] == rna.obs[cell_group]).all()
    cell_types = atac.obs[cell_group]

    train_len = len(atac) - args.valid_subsample
    valid_len = args.valid_subsample
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(atac)), y=cell_types))

    ## Create paired object
    paired = ad.concat([rna, atac], join='inner', axis=1)
    paired.obs['modality'] = 'paired'

    # split to three datasets by modality (RNA, ATAC, Multiome), and corrupt data
    # by remove some data to create single-modality data
    n = len(paired)//3
    adata_rna = paired[:n, paired.var.modality == "Gene Expression"].copy()
    adata_paired = paired[n : 2 * n].copy()
    adata_atac = paired[2 * n :, paired.var.modality == "Peaks"].copy()

    paired = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)

    #sc.pp.filter_genes(paired, min_cells=int(paired.shape[0] * 0.01))  # comment out, since results in different features for source and target datasets

    if source_or_target == 'target':
        return paired, cell_types, train_idx, valid_idx

    elif source_or_target == 'source':
        
        scvi.model.MULTIVI.setup_anndata(paired, batch_key="modality")

        model = scvi.model.MULTIVI(
            paired,
            n_genes=(paired.var["modality"] == "Gene Expression").sum(),
            n_regions=(paired.var["modality"] == "Peaks").sum(),
        )

        model.view_anndata_setup()

        return model, cell_types, train_idx, valid_idx
    

def train(model, train_idx, valid_idx, test_idx_target, epochs=1):
    model.train(max_epochs=epochs, datasplitter_kwargs={'external_indexing': [train_idx, valid_idx, test_idx_target]})
    return model


if __name__ == '__main__':

    source_setup_func = return_setup_func_from_dataset(args.source_dataset)
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)

    ## Get all features to skip preprocessing
    source_rna, source_atac, source_cell_group, _, _, source_atac_datapath, source_rna_datapath \
        = source_setup_func(args, return_raw_data=True, hvg_only=False, protein_coding_only=True, pretrain=None, return_type='data')
    
    target_rna, target_atac, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, target_atac_datapath, target_rna_datapath \
        = target_setup_func(args, return_raw_data=True, hvg_only=False, protein_coding_only=True, pretrain=None, return_type='data')
    
    source_rna, target_rna, source_atac, target_atac, target_genes_to_peaks_binary_mask, target_genes_peaks_dict \
        = retain_feature_overlap(args, source_rna, target_rna, source_atac, target_atac, source_cell_group, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, return_type='data')
    
    if (source_cell_group not in source_rna.obs.columns) or (target_cell_group not in target_atac.obs.columns):

        ## currently fails if both source and target datasets do not have cell types
        if (source_cell_group not in source_rna.obs.columns):
            rna_celltypist = source_rna.copy()
        if (target_cell_group not in target_atac.obs.columns):
            rna_celltypist = target_rna.copy()

        sc.pp.normalize_total(rna_celltypist, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna_celltypist)

        celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')
        predictions = celltypist.annotate(rna_celltypist, majority_voting=False, model=celltypist_model_path)

        ## Keep top N cell types
        print('Subsetting cell types')
        num_celltypes = 30; print(f'Keeping top {num_celltypes} cell types')
        value_counts_threshold = predictions.predicted_labels.value_counts().sort_values(ascending=False).iloc[num_celltypes]
        keep_celltypes = (predictions.predicted_labels.value_counts() > value_counts_threshold)
        keep_celltypes = keep_celltypes[keep_celltypes].index.get_level_values(0).values.to_list()
        keep_celltypes_idx = np_array(predictions.probability_matrix.columns.isin(keep_celltypes))

        ## Assign cell types to data based on top probabilities according to retained cell-types
        final_probability_matrix = predictions.probability_matrix.loc[:, keep_celltypes_idx]
        final_predicted_labels = final_probability_matrix.columns[final_probability_matrix.values.argmax(1)]

        print('Assigning predicted cell types to data')
        if (source_cell_group not in source_rna.obs.columns):
            source_rna.obs[source_cell_group] = final_predicted_labels
            source_atac.obs[source_cell_group] = final_predicted_labels
        if (target_cell_group not in target_atac.obs.columns):
            target_rna.obs[target_cell_group] = final_predicted_labels
            target_atac.obs[target_cell_group] = final_predicted_labels

    ## Setup
    paired_target, cell_types_target, train_idx_target, valid_idx_target = setup(target_rna, target_atac, target_cell_group, 'target')
    model, cell_types, train_idx, valid_idx = setup(source_rna, source_atac, source_cell_group, 'source')

    ## Incorporate target data into source model, perhaps should incorporate via setup_anndata instead
    model.adata = model.adata.concatenate(paired_target[valid_idx_target], batch_categories=['source', 'target'])
    test_idx = np_arange( len(model.adata) - len(valid_idx), len(model.adata) )
    model.test_indices = test_idx

    ## Train model
    model = train(model, train_idx, valid_idx, test_idx, epochs=args.n_epochs)  # about 10 seconds per epoch

    ## Get latent data - source valid
    rna_latents_valid   = model.get_latent_representation(indices=model.validation_indices, modality='expression')
    atac_latents_valid  = model.get_latent_representation(indices=model.validation_indices, modality='accessibility')

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_valid = torch.from_numpy(rna_latents_valid)
    atac_latents_valid = torch.from_numpy(atac_latents_valid)

    cell_types_valid = cell_types[valid_idx]

    ilisis_valid, clisis_valid, nmi_valid, ari_valid, diag_concentration_minimizer_valid, foscttm_score_valid, _, acc_valid, acc_top5_valid, clip_loss_valid, clip_loss_censored_valid, \
                        foscttm_score_ct_valid, accuracy_ct_valid, accuracy_top5_ct_valid, clip_loss_ct_valid, clip_loss_ct_split_valid = \
                            align_metrics(None, rna_latents_valid, cell_types_valid, atac_latents_valid, cell_types_valid, paired=True, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_valid,
        'clisis': clisis_valid,
        'nmi': nmi_valid,
        'ari': ari_valid,
        'foscttm_score': foscttm_score_valid,
        'foscttm_score_ct': foscttm_score_ct_valid,
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'multiVI_metrics_source_valid.csv'))

    ## --- TARGET DATASET --- ##

    ## Get latent data - target valid
    rna_latents_target_valid = model.get_latent_representation(indices=model.test_indices, modality='expression')
    atac_latents_target_valid = model.get_latent_representation(indices=model.test_indices, modality='accessibility')

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_target_valid = torch.from_numpy(rna_latents_target_valid)
    atac_latents_target_valid = torch.from_numpy(atac_latents_target_valid)

    cell_types_target_valid = cell_types_target[valid_idx_target]

    ilisis_target_valid, clisis_target_valid, nmi_target_valid, ari_target_valid, diag_concentration_minimizer_target_valid, foscttm_score_target_valid, _, acc_target_valid, acc_top5_target_valid, clip_loss_target_valid, clip_loss_censored_target_valid, \
                        foscttm_score_ct_target_valid, accuracy_ct_target_valid, accuracy_top5_ct_target_valid, clip_loss_ct_target_valid, clip_loss_ct_split_target_valid = \
                            align_metrics(None, rna_latents_target_valid, cell_types_target_valid, atac_latents_target_valid, cell_types_target_valid, paired=True, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_target_valid,
        'clisis': clisis_target_valid,
        'nmi': nmi_target_valid,
        'ari': ari_target_valid,
        'foscttm_score': foscttm_score_target_valid,
        'foscttm_score_ct': foscttm_score_ct_target_valid,
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'multiVI_metrics_target_valid.csv'))

    
    '''
    ## Get latent data - train
    rna_latents_train   = model.get_latent_representation(indices=model.train_indices, modality='expression')
    atac_latents_train  = model.get_latent_representation(indices=model.train_indices, modality='accessibility')

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_train = torch.from_numpy(rna_latents_train)
    atac_latents_train = torch.from_numpy(atac_latents_train)

    cell_types_train = cell_types[train_idx]

    ilisis_train, clisis_train, nmi_train, ari_train, diag_concentration_minimizer_train, foscttm_score_train, acc_train, acc_top5_train, clip_loss_train, clip_loss_censored_train, \
                        foscttm_score_ct_train, accuracy_ct_train, accuracy_top5_ct_train, clip_loss_ct_train, clip_loss_ct_split_train = \
                            align_metrics(None, rna_latents_train, cell_types_train, atac_latents_train, cell_types_train, paired=True, is_latents=True)
    '''

    print('Done')