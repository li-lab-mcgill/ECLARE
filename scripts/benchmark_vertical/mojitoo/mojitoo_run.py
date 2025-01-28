import matplotlib
matplotlib.use('Agg')  # Set backend to 'Agg' for non-GUI rendering

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

#import muon
import mojitoo
import scanpy as sc
import episcanpy as epi
from mudata import MuData
from torch import from_numpy
from numpy import empty as np_empty
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from setup_utils import return_setup_func_from_dataset
from eval_utils import align_metrics


def process_data(rna, atac):
    ## Preprocess ATAC data (MOJITOO documentation)
    epi.pp.cal_var(atac)
    epi.pp.select_var_feature(atac, nb_features=5000)
    epi.tl.tfidf(atac)
    epi.tl.lsi(atac, n_components=50)

    ## Preprocess RNA data (MOJITOO documentation)
    sc.pp.normalize_total(rna, target_sum=1e4)
    #sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, svd_solver='arpack')

    ## Create MuData object
    mdata = MuData({"atac": atac, "rna": rna})
    mdata.obsm["pca"] = rna.obsm["X_pca"]
    mdata.obsm["lsi"] = atac.obsm["X_lsi"]

    return mdata

from argparse import ArgumentParser
parser = ArgumentParser(description='')
parser.add_argument('--outdir', type=str, default=default_outdir,
                    help='output directory')
parser.add_argument('--n_epochs', type=int, default=2,
                    help='number of epochs')
parser.add_argument('--source_dataset', type=str, default='roussos',
                    help='dataset to use')
parser.add_argument('--target_dataset', type=str, default='human_dlpfc',
                    help='target dataset')
parser.add_argument('--save_latents', action='store_true',
                    help='save latents during training')
parser.add_argument('--valid_subsample', type=int, default=5000,
                    help='number of nuclei to subsample for validation')
parser.add_argument('--genes_by_peaks_str', type=str, default='6920_by_57298',
                    help='genes by peaks string')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

## Load data
source_setup_func = return_setup_func_from_dataset(args.source_dataset)
target_setup_func = return_setup_func_from_dataset(args.target_dataset)

## Get HVG features
rna_hvg, atac_hvg, _, _, _, _, _ \
    = source_setup_func(args, hvg_only=True, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=False)

## Get all features to skip preprocessing
rna, atac, cell_group, _, _, atac_datapath, rna_datapath \
    = source_setup_func(args, hvg_only=False, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=True)

## Retain only HVG features
rna = rna[:, rna.var_names.isin(rna_hvg.var_names) ].copy()
atac = atac[:, atac.var_names.isin(atac_hvg.var_names) ].copy()
del rna_hvg, atac_hvg

mdata = process_data(rna, atac)

## Obtain cell types
assert (mdata['atac'].obs[cell_group] == mdata['rna'].obs[cell_group]).all()
cell_types = mdata['atac'].obs[cell_group]

## Train-valid split
train_len = len(mdata) - args.valid_subsample
valid_len = args.valid_subsample
train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(mdata)), y=cell_types))

mdata_train = mdata[train_idx]
mdata_valid = mdata[valid_idx]

cell_types_train = cell_types[train_idx]
cell_types_valid = cell_types[valid_idx]

## Run MOJITOO
rna_latents_train, atac_latents_train, cca, fdr_bool = mojitoo.mojitoo(mdata_train, reduction_list=["pca", "lsi"],  dims_list=(range(50), range(1,50)),reduction_name='mojitoo', overwrite=True)
rna_latents_valid, atac_latents_valid, _, _ = mojitoo.mojitoo(mdata_valid, cca=cca, fdr_bool=fdr_bool, reduction_list=["pca", "lsi"],  dims_list=(range(50), range(1,50)),reduction_name='mojitoo', overwrite=True)
                                        
## Cast latents as torch tensors for compatibility with eval_utils
rna_latents_valid = from_numpy(rna_latents_valid)
atac_latents_valid = from_numpy(atac_latents_valid)

## Align metrics for valid data
ilisis_valid, clisis_valid, nmi_valid, ari_valid, diag_concentration_minimizer_valid, foscttm_score_valid, rank_score_valid, acc_valid, acc_top5_valid, clip_loss_valid, clip_loss_censored_valid, \
                    foscttm_score_ct_valid, accuracy_ct_valid, accuracy_top5_ct_valid, clip_loss_ct_valid, clip_loss_ct_split_valid = \
                        align_metrics(None, rna_latents_valid, cell_types_valid, atac_latents_valid, cell_types_valid, paired=True, is_latents=True)

metrics_df = pd.Series({
    'ilisis': ilisis_valid,
    'clisis': clisis_valid,
    'nmi': nmi_valid,
    'ari': ari_valid,
    'foscttm_score': foscttm_score_valid,
    'foscttm_score_ct': foscttm_score_ct_valid,
    'rank_score': rank_score_valid
})

metrics_df.to_csv(os.path.join(args.outdir, f'mojitoo_metrics_source_valid.csv'))

## --- TARGET DATASET --- ##

## Get HVG features
target_hvg, target_atac_hvg, _, _, _, _, _ \
    = target_setup_func(args, hvg_only=True, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=False)

## Get all features to skip preprocessing
target_rna, target_atac, target_cell_group, _, _, target_atac_datapath, target_rna_datapath \
    = target_setup_func(args, hvg_only=False, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=True)

## Retain only HVG features
target_rna = target_rna[:, target_rna.var_names.isin(target_hvg.var_names) ].copy()
target_atac = target_atac[:, target_atac.var_names.isin(target_atac_hvg.var_names) ].copy()
del target_hvg, target_atac_hvg

target_mdata = process_data(target_rna, target_atac)

## Obtain cell types
assert (target_mdata['atac'].obs[target_cell_group] == target_mdata['rna'].obs[target_cell_group]).all()
target_cell_types = target_mdata['atac'].obs[target_cell_group]

## Train-valid split
train_len = len(target_mdata) - args.valid_subsample
valid_len = args.valid_subsample
_, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(target_mdata)), y=target_cell_types))

target_mdata_valid = target_mdata[valid_idx]
target_cell_types_valid = target_cell_types[valid_idx]

## Inference with MOJITOO
target_rna_latents_valid, target_atac_latents_valid, _, _ = mojitoo.mojitoo(target_mdata_valid, cca=cca, fdr_bool=fdr_bool, reduction_list=["pca", "lsi"],  dims_list=(range(50), range(1,50)),reduction_name='mojitoo', overwrite=True)

## Cast latents as torch tensors for compatibility with eval_utils
target_rna_latents_valid = from_numpy(target_rna_latents_valid)
target_atac_latents_valid = from_numpy(target_atac_latents_valid)

## Align metrics for valid data
ilisis_target_valid, clisis_target_valid, nmi_target_valid, ari_target_valid, diag_concentration_minimizer_target_valid, foscttm_score_target_valid, rank_score_target_valid, acc_target_valid, acc_top5_target_valid, clip_loss_target_valid, clip_loss_censored_target_valid, \
                    foscttm_score_ct_target_valid, accuracy_ct_target_valid, accuracy_top5_ct_target_valid, clip_loss_ct_target_valid, clip_loss_ct_split_target_valid = \
                        align_metrics(None, target_rna_latents_valid, target_cell_types_valid, target_atac_latents_valid, target_cell_types_valid, paired=True, is_latents=True)

metrics_df = pd.Series({
    'ilisis': ilisis_target_valid,
    'clisis': clisis_target_valid,
    'nmi': nmi_target_valid,
    'ari': ari_target_valid,
    'foscttm_score': foscttm_score_target_valid,
    'foscttm_score_ct': foscttm_score_ct_target_valid,
    'rank_score': rank_score_target_valid
})

metrics_df.to_csv(os.path.join(args.outdir, f'mojitoo_metrics_target_valid.csv'))

'''
## Align metrics for train data
rna_latents_train = from_numpy(rna_latents_train)
atac_latents_train = from_numpy(atac_latents_train)

ilisis_train, clisis_train, nmi_train, ari_train, diag_concentration_minimizer_train, foscttm_score_train, rank_score_train, acc_train, acc_top5_train, clip_loss_train, clip_loss_censored_train, \
                    foscttm_score_ct_train, accuracy_ct_train, accuracy_top5_ct_train, clip_loss_ct_train, clip_loss_ct_split_train = \
                        align_metrics(None, rna_latents_train, cell_types_train, atac_latents_train, cell_types_train, paired=True, is_latents=True)

metrics_df = pd.Series({
    'ilisis': ilisis_train,
    'clisis': clisis_train,
    'nmi': nmi_train,
    'ari': ari_train,
    'foscttm_score': foscttm_score_train,
    'foscttm_score_ct': foscttm_score_ct_train,
    'rank_score': rank_score_train
})

metrics_df.to_csv(os.path.join(args.outdir, f'mojitoo_metrics_train.csv'))

## Run clustering and UMAP
sc.pp.neighbors(mdata_valid, use_rep='mojitoo')
sc.tl.leiden(mdata_valid, resolution=0.5)     # replaced louvain by leiden
sc.tl.umap(mdata_valid)
sc.pl.embedding(mdata_valid, color='leiden', basis='umap')
'''