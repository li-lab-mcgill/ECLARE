import os
import sys
import numpy as np
import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    default_outdir = outpath = '/home/dmannk/scratch/'
    CLARE_root = '/home/dmannk/projects/def-liyue/dmannk/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    sys.path.insert(0, '/home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_diagonal/scDART/scDART')
    gtf = "/home/dmannk/projects/def-liyue/dmannk/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    default_outdir = outpath = '/Users/dmannk/cisformer/outputs/'
    CLARE_root = '/Users/dmannk/cisformer/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    sys.path.insert(0, '/Users/dmannk/cisformer/CLARE/benchmark_diagonal/scDART/scDART')
    gtf = "/Users/dmannk/cisformer/workspace/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif np.any([mcb_server in hostname for mcb_server in ['mcb', 'buckeridge' ,'hlr', 'ri', 'wh', 'yl']]):
    os.environ['machine'] = 'mcb'
    default_outdir = outpath = '/home/mcb/users/dmannk/scMultiCLIP/outputs'
    CLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    sys.path.insert(0, '/home/mcb/users/dmannk/scMultiCLIP/CLARE/benchmark_diagonal/scDART/scDART')
    gtf = "/home/mcb/users/dmannk/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

import pandas as pd

import torch
from sklearn.decomposition import PCA

import scDART.utils as utils
import scDART.TI as ti
import scDART
from sklearn.model_selection import StratifiedShuffleSplit

from setup_utils import return_setup_func_from_dataset
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
parser.add_argument('--train_subsample', type=int, default=25000,
                    help='number of nuclei to subsample for training')
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

os.environ['device'] = device

if __name__ == '__main__':

    setup_func = return_setup_func_from_dataset(args.source_dataset)
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)

    ## Get HVG features
    rna_hvg, atac_hvg, _, _, _, _, _ \
        = setup_func(args, hvg_only=True, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=False)
    
    ## Get all features to skip preprocessing
    rna, atac, cell_group, _, _, atac_datapath, rna_datapath \
        = setup_func(args, hvg_only=False, protein_coding_only=True, pretrain=None, return_type='data', return_raw_data=True)

    ## Retain only HVG features
    rna = rna[:, rna.var_names.isin(rna_hvg.var_names) ].copy()
    atac = atac[:, atac.var_names.isin(atac_hvg.var_names) ].copy()
    del rna_hvg, atac_hvg

    rna = rna[:,::2].copy()
    atac = atac[:,::2].copy()

    rna.X = rna.X.astype('float32')
    atac.X = atac.X.astype('float32')

    ## Obtain cell types and obtain train-valid split indices
    assert (atac.obs[cell_group] == rna.obs[cell_group]).all()
    cell_types = atac.obs[cell_group]

    ## Subsample train and valid sets, due to memory constraints of computing diffusion distances
    print(f'Subsample train and valid sets to {args.train_subsample} and {args.valid_subsample} nuclei, respectively')
    train_len = args.train_subsample if args.train_subsample>0 else len(atac) - args.valid_subsample
    valid_len = args.valid_subsample
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np.empty(len(atac)), y=cell_types))

    rna_train = rna[train_idx].copy()
    rna_valid = rna[valid_idx].copy()

    atac_train = atac[train_idx].copy()
    atac_valid = atac[valid_idx].copy()

    cell_types_train = cell_types[train_idx]
    cell_types_valid = cell_types[valid_idx]
    del rna, atac

    # all in one
    batch_size = 128
    seeds = [0]
    latent_dim = 4
    learning_rate = 3e-4
    n_epochs = args.n_epochs
    use_anchor = False
    reg_d = 1
    reg_g = 1
    reg_mmd = 1
    ts = [30, 50, 70]
    use_potential = True

    coarse_reg = np.ones([atac_train.n_vars, rna_train.n_vars])

    scDART_op = scDART.scDART(n_epochs = n_epochs, latent_dim = latent_dim, batch_size = batch_size, \
            ts = ts, use_anchor = use_anchor, use_potential = use_potential, k = 10, \
            reg_d = 1, reg_g = 1, reg_mmd = 1, l_dist_type = 'kl', seed = seeds[0],\
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    scDART_op = scDART_op.fit(rna_count = rna_train.X.toarray(), atac_count = atac_train.X.toarray(), reg = coarse_reg, rna_anchor = None, atac_anchor = None)

    rna_latents_valid, atac_latents_valid = scDART_op.transform(rna_count = rna_valid.X.toarray(), atac_count = atac_valid.X.toarray())

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_valid = torch.from_numpy(rna_latents_valid)
    atac_latents_valid = torch.from_numpy(atac_latents_valid)

    ## Evaluate alignment
    paired = (args.source_dataset != 'mdd')
    ilisis_valid, clisis_valid, nmi_valid, ari_valid, diag_concentration_minimizer_valid, foscttm_score_valid, rank_score_valid, acc_valid, acc_top5_valid, clip_loss_valid, clip_loss_censored_valid, \
                    foscttm_score_ct_valid, accuracy_ct_valid, accuracy_top5_ct_valid, clip_loss_ct_valid, clip_loss_ct_split_valid = \
                        align_metrics(None, rna_latents_valid, cell_types_valid, atac_latents_valid, cell_types_valid, paired=paired, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_valid,
        'clisis': clisis_valid,
        'nmi': nmi_valid,
        'ari': ari_valid,
        'foscttm_score': foscttm_score_valid,
        'foscttm_score_ct': foscttm_score_ct_valid,
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'scdart_metrics_source_valid.csv'))


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

    target_rna = target_rna[:,::2].copy()
    target_atac = target_atac[:,::2].copy()

    target_rna.X = target_rna.X.astype('float32')
    target_atac.X = target_atac.X.astype('float32')

    ## Obtain cell types and obtain train-valid split indices
    assert (target_atac.obs[target_cell_group] == target_rna.obs[target_cell_group]).all()
    target_cell_types = target_atac.obs[target_cell_group]

    train_len = len(target_atac) - args.valid_subsample
    valid_len = args.valid_subsample
    _, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np.empty(len(target_atac)), y=target_cell_types))

    target_rna_valid = target_rna[valid_idx].copy()
    target_atac_valid = target_atac[valid_idx].copy()

    target_cell_types_valid = target_cell_types[valid_idx]
    del target_rna, target_atac

    ## Inference with scDART
    target_rna_latents_valid, target_atac_latents_valid = scDART_op.transform(rna_count = target_rna_valid.X.toarray(), atac_count = target_atac_valid.X.toarray())

    ## Cast latents as torch tensors for compatibility with eval_utils
    target_rna_latents_valid = torch.from_numpy(target_rna_latents_valid)
    target_atac_latents_valid = torch.from_numpy(target_atac_latents_valid)

    ## Align metrics for valid data
    paired = (args.target_dataset != 'mdd')
    ilisis_target_valid, clisis_target_valid, nmi_target_valid, ari_target_valid, diag_concentration_minimizer_target_valid, foscttm_score_target_valid, rank_score_target_valid, acc_target_valid, acc_top5_target_valid, clip_loss_target_valid, clip_loss_censored_target_valid, \
                        foscttm_score_ct_target_valid, accuracy_ct_target_valid, accuracy_top5_ct_target_valid, clip_loss_ct_target_valid, clip_loss_ct_split_target_valid = \
                            align_metrics(None, target_rna_latents_valid, target_cell_types_valid, target_atac_latents_valid, target_cell_types_valid, paired=paired, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_target_valid,
        'clisis': clisis_target_valid,
        'nmi': nmi_target_valid,
        'ari': ari_target_valid,
        'foscttm_score': foscttm_score_target_valid,
        'foscttm_score_ct': foscttm_score_ct_target_valid,
        'rank_score': rank_score_target_valid
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'scdart_metrics_target_valid.csv'))

    print('Done!')