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
    sys.path.insert(0, '/home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_diagonal/scJoint/scJoint')
    datapath = '/home/dmannk/scratch/scJoint_data_tmp/'
    gtf = "/home/dmannk/projects/def-liyue/dmannk/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    default_outdir = outpath = '/Users/dmannk/cisformer/CLARE/benchmark_diagonal/scJoint/output/'
    CLARE_root = '/Users/dmannk/cisformer/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    sys.path.insert(0, '/Users/dmannk/cisformer/CLARE/benchmark_diagonal/scJoint/scJoint')
    datapath = '/Users/dmannk/cisformer/CLARE/benchmark_diagonal/scJoint/data/'
    gtf = "/Users/dmannk/cisformer/workspace/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif np.any([mcb_server in hostname for mcb_server in ['mcb', 'buckeridge' ,'hlr', 'ri', 'wh', 'yl']]):
    os.environ['machine'] = 'mcb_server'
    default_outdir = outpath = '/home/mcb/users/dmannk/scMultiCLIP/outputs/'
    default_outdir = outpath = '/home/mcb/users/dmannk/scMultiCLIP/outputs/'
    CLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    sys.path.insert(0, '/home/mcb/users/dmannk/scMultiCLIP/CLARE/benchmark_diagonal/scJoint')
    sys.path.insert(0, '/home/mcb/users/dmannk/scMultiCLIP/CLARE/benchmark_diagonal/scJoint/scJoint')
    datapath = '/home/mcb/users/dmannk/scMultiCLIP/outputs/scJoint_data_tmp/'
    gtf = "/home/mcb/users/dmannk/scMultiCLIP/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

from setup_utils import return_setup_func_from_dataset, gene_activity_score_adata
from eval_utils import align_metrics

import h5py
from scipy.sparse import csr_matrix, csc_matrix
from pathlib import Path
import pandas as pd
import process_db
from torch import from_numpy
from torch.cuda import is_available as cuda_available
from sklearn.model_selection import StratifiedShuffleSplit

def write_10X_h5(adata, file, cell_group, path=datapath):
    """ https://github.com/scverse/anndata/issues/595
    
    Writes adata to a 10X-formatted h5 file.
    
    Note that this function is not fully tested and may not work for all cases.
    It will not write the following keys to the h5 file compared to 10X:
    '_all_tag_keys', 'pattern', 'read', 'sequence'

    Args:
        adata (AnnData object): AnnData object to be written.
        file (str): File name to be written to. If no extension is given, '.h5' is appended.

    Raises:
        FileExistsError: If file already exists.

    Returns:
        None
    """
    
    if '.h5' not in file: file = f'{file}.h5'
    filepath = os.path.join(path, file)
    #if not Path(filepath).exists():
    def int_max(x):
        return int(max(np.floor(len(str(int(max(x)))) / 4), 1) * 4)
    def str_max(x):
        return max([len(i) for i in x])

    w = h5py.File(filepath, 'w')
    grp = w.create_group("matrix")
    grp.create_dataset("barcodes", data=np.array(adata.obs_names, dtype=f'|S{str_max(adata.obs_names)}'))
    grp.create_dataset("data", data=np.array(adata.X.data, dtype=f'<i{int_max(adata.X.data)}'))
    ftrs = grp.create_group("features")
    # this group will lack the following keys:
    # '_all_tag_keys', 'feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence'
    #ftrs.create_dataset("feature_type", data=np.array(adata.var.feature_types, dtype=f'|S{str_max(adata.var.feature_types)}'))
    #ftrs.create_dataset("genome", data=np.array(adata.var.genome, dtype=f'|S{str_max(adata.var.genome)}'))
    #ftrs.create_dataset("id", data=np.array(adata.var.gene_ids, dtype=f'|S{str_max(adata.var.gene_ids)}'))
    ftrs.create_dataset("name", data=np.array(adata.var.index, dtype=f'|S{str_max(adata.var.index)}'))
    grp.create_dataset("indices", data=np.array(adata.X.indices, dtype=f'<i{int_max(adata.X.indices)}'))
    grp.create_dataset("indptr", data=np.array(adata.X.indptr, dtype=f'<i{int_max(adata.X.indptr)}'))
    grp.create_dataset("shape", data=np.array(list(adata.X.shape)[::-1], dtype=f'<i{int_max(adata.X.shape)}'))

    ## save celltypes to csv
    celltypes = adata.obs[cell_group]
    filename = file.split('.')[0]
    celltypes.to_csv(f'{path}/{filename}_celltypes.csv', header=[cell_group], index=True)
        
    #else:
    #    print(f"{file} already exists.")


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
parser.add_argument('--replicate_idx', type=int, default=0,
                    help='replicate index')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

os.environ['outdir'] = args.outdir

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

    ## Convert to csc_matrix for proper indptr
    rna.X = csc_matrix(rna.X)
    atac.X = csc_matrix(atac.X)
    atac, rna = gene_activity_score_adata(atac, rna)

    write_10X_h5(rna, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}', cell_group)
    write_10X_h5(atac, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}', cell_group)

    rna_h5_files = [os.path.join(datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.h5')]
    rna_label_files = [os.path.join(datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv')]

    atac_h5_files = [os.path.join(datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.h5')]
    atac_label_files = [os.path.join(datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv')]

    process_db.data_parsing(rna_h5_files, atac_h5_files)

    rna_label = pd.read_csv(rna_label_files[0], header=0, index_col=0)
    print(rna_label.value_counts(sort = False))
    process_db.label_parsing(rna_label_files, atac_label_files)

    ## run main scJoint file
    import main  # import from scJoint
    from config import Config

    config = Config()

    config.datapath = datapath
    config.device = 'cuda' if cuda_available() else 'cpu'
    config.use_cuda = cuda_available()

    config.number_of_class = rna_label.nunique().item()
    config.input_size = rna.n_vars
    config.rna_paths = [os.path.join(config.datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.npz')]
    config.rna_labels = [os.path.join(config.datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.txt')]
    config.atac_paths = [os.path.join(config.datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.npz')]
    config.atac_labels = [os.path.join(config.datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.txt')]
    config.epochs_stage1 = args.n_epochs
    config.epochs_stage3 = args.n_epochs
    config.valid_subsample = args.valid_subsample

    config.batch_size = 10000

    model, index_dict = main.main(config)

    ## get indices, originally defined in Stage 1
    rna_train_idx = index_dict['rna_train']
    rna_valid_idx = index_dict['rna_valid']
    atac_train_idx = index_dict['atac_train']
    atac_valid_idx = index_dict['atac_valid']

    ## fetch embeddings
    rna_latents_valid = pd.read_csv(os.path.join(args.outdir, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')
    atac_latents_valid = pd.read_csv(os.path.join(args.outdir, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')

    rna_latents_valid = from_numpy(rna_latents_valid.values)
    atac_latents_valid = from_numpy(atac_latents_valid.values)

    ## cell types
    atac_cell_types = pd.read_csv(os.path.join(datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv'), header=0, index_col=0)  # to be updated
    atac_cell_types_train = atac_cell_types.iloc[atac_train_idx][cell_group].values
    atac_cell_types_valid = atac_cell_types.iloc[atac_valid_idx][cell_group].values

    rna_cell_types = pd.read_csv(os.path.join(datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv'), header=0, index_col=0)  # to be updated
    rna_cell_types_train = rna_cell_types.iloc[rna_train_idx][cell_group].values
    rna_cell_types_valid = rna_cell_types.iloc[rna_valid_idx][cell_group].values

    ## Evaluate alignment
    paired = (args.source_dataset != 'mdd')
    ilisis_valid, clisis_valid, nmi_valid, ari_valid, diag_concentration_minimizer_valid, foscttm_score_valid, rank_score_valid, acc_valid, acc_top5_valid, clip_loss_valid, clip_loss_censored_valid, \
                    foscttm_score_ct_valid, accuracy_ct_valid, accuracy_top5_ct_valid, clip_loss_ct_valid, clip_loss_ct_split_valid = \
                        align_metrics(None, rna_latents_valid, rna_cell_types_valid, atac_latents_valid, atac_cell_types_valid, paired=paired, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_valid,
        'clisis': clisis_valid,
        'nmi': nmi_valid,
        'ari': ari_valid,
        'foscttm_score': foscttm_score_valid,
        'foscttm_score_ct': foscttm_score_ct_valid,
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'scJoint_metrics_source_valid.csv'))


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

    ## Convert to csc_matrix for proper indptr
    target_rna.X = csc_matrix(target_rna.X)
    target_atac.X = csc_matrix(target_atac.X)
    target_atac, target_rna = gene_activity_score_adata(target_atac, target_rna)

    write_10X_h5(target_rna, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}', target_cell_group)
    write_10X_h5(target_atac, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}', target_cell_group)

    target_rna_h5_files = [os.path.join(datapath, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}.h5')]
    target_rna_label_files = [os.path.join(datapath, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.csv')]

    target_atac_h5_files = [os.path.join(datapath, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}.h5')]
    target_atac_label_files = [os.path.join(datapath, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.csv')]

    process_db.data_parsing(target_rna_h5_files, target_atac_h5_files)

    target_rna_label = pd.read_csv(target_rna_label_files[0], header=0, index_col=0)
    print(target_rna_label.value_counts(sort = False))
    process_db.label_parsing(target_rna_label_files, target_atac_label_files)


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

    from scJoint.util.dataloader_stage3 import collate_fn_binarize, collate_fn_binarize_with_labels
    from anndata.experimental import AnnLoader
    
    target_rna_valid.obs = target_rna_valid.obs.rename(columns={target_cell_group: 'labels'}, inplace=False)
    target_atac_valid.obs = target_atac_valid.obs.rename(columns={target_cell_group: 'labels'}, inplace=False)

    #target_rna_valid_loader = AnnLoader(target_rna_valid, use_cuda=cuda_available(), batch_size=len(target_rna_valid), shuffle=True, collate_fn=collate_fn_binarize)
    target_rna_valid.obs['labels'] = target_rna_valid.obs['labels'].factorize()[0]
    target_rna_valid_loader = AnnLoader(target_rna_valid, use_cuda=cuda_available(), batch_size=len(target_rna_valid), shuffle=True, collate_fn=collate_fn_binarize_with_labels)
    target_atac_valid_loader = AnnLoader(target_atac_valid, use_cuda=cuda_available(), batch_size=len(target_atac_valid), shuffle=True, collate_fn=collate_fn_binarize)

    model.test_rna_loaders = [target_rna_valid_loader]
    model.test_atac_loaders = [target_atac_valid_loader]

    model.config.rna_paths = [os.path.join(config.datapath, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}.npz')]
    model.config.rna_labels = [os.path.join(config.datapath, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.txt')]
    model.config.atac_paths = [os.path.join(config.datapath, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}.npz')]
    model.config.atac_labels = [os.path.join(config.datapath, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.txt')]

    model.write_embeddings()

    ## fetch embeddings
    target_rna_latents_valid = pd.read_csv(os.path.join(args.outdir, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')
    target_atac_latents_valid = pd.read_csv(os.path.join(args.outdir, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')

    target_rna_latents_valid = from_numpy(target_rna_latents_valid.values)
    target_atac_latents_valid = from_numpy(target_atac_latents_valid.values)

    ## cell types
    target_atac_cell_types = pd.read_csv(os.path.join(datapath, f'atac_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.csv'), header=0, index_col=0)  # to be updated
    target_atac_cell_types_valid = target_atac_cell_types.iloc[valid_idx][target_cell_group].values

    target_rna_cell_types = pd.read_csv(os.path.join(datapath, f'rna_target_{args.target_dataset}_source_{args.source_dataset}_{args.replicate_idx}_celltypes.csv'), header=0, index_col=0)  # to be updated
    target_rna_cell_types_valid = target_rna_cell_types.iloc[valid_idx][target_cell_group].values

    ## Evaluate alignment
    paired = (args.target_dataset != 'mdd')
    ilisis_valid, clisis_valid, nmi_valid, ari_valid, diag_concentration_minimizer_valid, foscttm_score_valid, rank_score_valid, acc_valid, acc_top5_valid, clip_loss_valid, clip_loss_censored_valid, \
                    foscttm_score_ct_valid, accuracy_ct_valid, accuracy_top5_ct_valid, clip_loss_ct_valid, clip_loss_ct_split_valid = \
                        align_metrics(None, target_rna_latents_valid, target_rna_cell_types_valid, target_atac_latents_valid, target_atac_cell_types_valid, paired=paired, is_latents=True)
    
    metrics_df = pd.Series({
        'ilisis': ilisis_valid,
        'clisis': clisis_valid,
        'nmi': nmi_valid,
        'ari': ari_valid,
        'foscttm_score': foscttm_score_valid,
        'foscttm_score_ct': foscttm_score_ct_valid,
    })

    metrics_df.to_csv(os.path.join(args.outdir, f'scJoint_metrics_target_valid.csv'))

    print('Done.')
    
