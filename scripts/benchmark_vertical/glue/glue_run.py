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
    gtf = "/home/dmannk/projects/def-liyue/dmannk/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    default_outdir = outpath = '/Users/dmannk/cisformer/outputs/'
    CLARE_root = '/Users/dmannk/cisformer/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    gtf = "/Users/dmannk/cisformer/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif np_any([mcb_server in hostname for mcb_server in ['mcb', 'buckeridge' ,'hlr', 'ri', 'wh', 'yl']]):
    os.environ['machine'] = 'mcb'
    default_outdir = outpath = '/home/mcb/users/dmannk/scMultiCLIP/outputs'
    CLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/CLARE'
    sys.path.insert(0, CLARE_root)
    os.environ['CLARE_root'] = CLARE_root
    gtf = "/home/mcb/users/dmannk/scMultiCLIP/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

import scglue
import scanpy as sc
import episcanpy as epi
from itertools import chain
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import empty as np_empty
from numpy import ceil as np_ceil
from torch import from_numpy
import pandas as pd
import celltypist
from numpy import array as np_array
from scipy.sparse import csr_matrix

from setup_utils import return_setup_func_from_dataset, retain_feature_overlap
from eval_utils import align_metrics

def process_data(rna, atac):

    ## Ensure that all count values are integers
    atac_non_integer_mask = (atac.X.data % 1 != 0)
    rna_non_integer_mask = (rna.X.data % 1 != 0)

    if np_any(atac_non_integer_mask):
        print(f'Converting ATAC data to integers: {sum(atac_non_integer_mask)} values')
        new_data = atac.X.data.copy()
        new_data[atac_non_integer_mask] = np_ceil(new_data[atac_non_integer_mask])
        atac.X = csr_matrix((new_data, atac.X.indices, atac.X.indptr), shape=atac.X.shape)

    if np_any(rna_non_integer_mask):
        print(f'Converting RNA data to integers: {sum(rna_non_integer_mask)} values')
        new_data = rna.X.data.copy()
        new_data[rna_non_integer_mask] = np_ceil(new_data[rna_non_integer_mask])
        rna.X = csr_matrix((new_data, rna.X.indices, rna.X.indptr), shape=rna.X.shape)

    ## Preprocess ATAC data (MOJITOO documentation)
    epi.pp.cal_var(atac)
    epi.pp.select_var_feature(atac, nb_features=5000)
    epi.tl.tfidf(atac)
    epi.tl.lsi(atac, n_components=50)

    ## Preprocess RNA data (MOJITOO documentation)
    rna.layers["counts"] = rna.X.ceil().copy()

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, svd_solver='arpack')

    print('In tutorial, flavor="seurat_v3" is used for highly_variable_genes()')
    sc.pp.highly_variable_genes(rna, n_top_genes=2000)

    return rna, atac

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

if __name__ == '__main__':

    ## Load data
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


    rna, atac = process_data(source_rna, source_atac)

    ## Obtain cell types and obtain train-valid split indices
    assert (atac.obs[source_cell_group] == rna.obs[source_cell_group]).all()
    cell_types = atac.obs[source_cell_group]

    train_len = len(atac) - args.valid_subsample
    valid_len = args.valid_subsample
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(atac)), y=cell_types))

    rna_train = rna[train_idx].copy()
    rna_valid = rna[valid_idx].copy()

    atac_train = atac[train_idx].copy()
    atac_valid = atac[valid_idx].copy()

    cell_types_train = cell_types[train_idx]
    cell_types_valid = cell_types[valid_idx]

    ## Obtain genomic coordinates 
    scglue.data.get_gene_annotation(rna_train, gtf=gtf, gtf_by='gene_name')
    non_float_coords = rna_train.var[["chromStart", "chromEnd"]].isna().any(axis=1)
    rna_train = rna_train[:,~non_float_coords]
    print(f"Number of genes with non-float coordinates: {sum(non_float_coords)} -> remove from RNA data")

    split = atac_train.var_names.str.split(r"[:-]")
    atac_train.var["chrom"] = split.map(lambda x: x[0])
    atac_train.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac_train.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    ## Construct guidance graph
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna_train, atac_train)
    scglue.graph.check_graph(guidance, [rna_train, atac_train])

    ## Configure data
    scglue.models.configure_dataset(
        rna_train, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca"
    )

    scglue.models.configure_dataset(
        atac_train, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )

    ## Extract subgraph for highly variable features
    guidance_hvf = guidance.subgraph(chain(
        rna_train.var.query("highly_variable").index,
        atac_train.var.query("highly_variable").index
    )).copy()

    ## Train GLUE model
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna_train, "atac": atac_train},
        guidance_hvf,
        fit_kws={"directory": os.path.join(args.outdir, "glue"), "max_epochs": args.n_epochs}
    )

    ## Project nuclei into latent embedding
    rna_latents_valid = glue.encode_data("rna", rna_valid)
    atac_latents_valid = glue.encode_data("atac", atac_valid)

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_valid = from_numpy(rna_latents_valid)
    atac_latents_valid = from_numpy(atac_latents_valid)

    ## Evaluate alignment
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

    metrics_df.to_csv(os.path.join(args.outdir, f'glue_metrics_source_valid.csv'))


    ## --- TARGET DATASET --- ##

    target_mdata = process_data(target_rna, target_atac)

    ## Obtain cell types
    assert (target_atac.obs[target_cell_group] == target_atac.obs[target_cell_group]).all()
    target_cell_types = target_atac.obs[target_cell_group]

    ## Train-valid split
    train_len = len(target_atac) - args.valid_subsample
    valid_len = args.valid_subsample
    _, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(target_atac)), y=target_cell_types))

    target_rna_valid = target_rna[valid_idx]
    target_atac_valid = target_atac[valid_idx]
    target_cell_types_valid = target_cell_types[valid_idx]

    ## Inference with GLUE, based on PCA & LSI. Perhaps better to use train data singular vectors to project valid data
    target_rna_latents_valid = glue.encode_data("rna", target_rna_valid)
    target_atac_latents_valid = glue.encode_data("atac", target_atac_valid)

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

    metrics_df.to_csv(os.path.join(args.outdir, f'glue_metrics_target_valid.csv'))

    '''
    ## Training set
    rna_latents_train = glue.encode_data("rna", rna_train)
    atac_latents_train = glue.encode_data("atac", atac_train)

    ## Cast latents as torch tensors for compatibility with eval_utils
    rna_latents_train = from_numpy(rna_latents_train)
    atac_latents_train = from_numpy(atac_latents_train)

    ## Evaluate alignment
    ilisis_train, clisis_train, nmi_train, ari_train, diag_concentration_minimizer_train, foscttm_score_train, acc_train, acc_top5_train, clip_loss_train, clip_loss_censored_train, \
                    foscttm_score_ct_train, accuracy_ct_train, accuracy_top5_ct_train, clip_loss_ct_train, clip_loss_ct_split_train = \
                        align_metrics(None, rna_latents_train, cell_types_train, atac_latents_train, cell_types_train, paired=True, is_latents=True)

    metrics_df = pd.Series({
        'ilisis': ilisis_train,
        'clisis': clisis_train,
        'nmi': nmi_train,
        'ari': ari_train,
        'foscttm_score': foscttm_score_train,
        'foscttm_score_ct': foscttm_score_ct_train,
    })
    
    metrics_df.to_csv(os.path.join(args.outdir, f'glue_metrics_train.csv'))

    losses = glue.get_losses({'rna':rna_train, 'atac':atac_train}, guidance_hvf)
    '''
