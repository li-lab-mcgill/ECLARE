import os
import sys
import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    default_outdir = outpath = '/home/dmannk/scratch/'
    sys.path.insert(0, '/home/dmannk/CLARE/')

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    default_outdir = outpath = '/Users/dmannk/cisformer/outputs/'
    sys.path.insert(0, '/Users/dmannk/cisformer/CLARE/')

from pyWNN.pyWNN import pyWNN
import scanpy as sc
import episcanpy as epi
from mudata import MuData
from torch import from_numpy
from numpy import empty as np_empty
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from setup_utils import return_setup_func_from_dataset
from scib_metrics import nmi_ari_cluster_labels_leiden

def process_data(rna, atac):
    ## Preprocess ATAC data (MOJITOO documentation)
    epi.pp.cal_var(atac)
    epi.pp.select_var_feature(atac, nb_features=5000)
    epi.tl.tfidf(atac)
    epi.tl.lsi(atac, n_components=50)

    ## Preprocess RNA data (MOJITOO documentation)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
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

## Load data
setup_func = return_setup_func_from_dataset(args.source_dataset)

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

mdata = process_data(rna, atac)

## Obtain cell types
assert (mdata['atac'].obs['Cell type'] == mdata['rna'].obs['Cell type']).all()
cell_types = mdata['atac'].obs['Cell type']

## Train-valid split - does not make much sense since WNN is instance-based learning, so no model is trained
train_len = len(mdata) - args.valid_subsample
valid_len = args.valid_subsample
train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(mdata)), y=cell_types))

cell_types_train = cell_types[train_idx]
cell_types_valid = cell_types[valid_idx]

mdata_train = mdata[train_idx]
mdata_valid = mdata[valid_idx]

mdata_train.obs['celltype'] = cell_types_train
mdata_valid.obs['celltype'] = cell_types_valid


## Run pyWNN (Seurat v4)
WNNobj = pyWNN(mdata_valid, reps=['pca', 'lsi'], npcs=[30,30], n_neighbors=20, seed=14)

mdata_train = WNNobj.compute_wnn(mdata_valid)

nmi_itr, ari_itr = nmi_ari_cluster_labels_leiden(mdata_valid.obsp['WNN_distance'], cell_types_train, optimize_resolution=True).values()

sc.tl.umap(mdata_train, neighbors_key='WNN')
(fig,ax) = plt.subplots(1,1, figsize=(3,3), dpi=200)
sc.pl.umap(mdata_train, color='celltype', ax=ax)