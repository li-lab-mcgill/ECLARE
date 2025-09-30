import sys
import os
ECLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/ECLARE/'
sys.path.insert(0, ECLARE_root)
sys.path.insert(0, os.path.join(ECLARE_root, 'src'))

from eclare import set_env_variables
set_env_variables(os.path.join(ECLARE_root, 'config'))

gtf = "/home/mcb/users/dmannk/scMultiCLIP/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

from eclare.setup_utils import return_setup_func_from_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import empty as np_empty
from itertools import chain
import scglue
import episcanpy as epi
from glob import glob
import pickle

from argparse import ArgumentParser
parser = ArgumentParser(description='')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--source_dataset', type=str, default='Cortex_Velmeshev',
                    help='dataset to use')
parser.add_argument('--target_dataset', type=str, default=None,
                    help='target dataset')
parser.add_argument('--valid_subsample', type=int, default=5000,
                    help='number of nuclei to subsample for validation')
parser.add_argument('--genes_by_peaks_str', type=str, default='9584_by_66620',
                    help='genes by peaks string')
parser.add_argument('--eclare_job_id', type=str, default=None,
                    help='eclare job id')
parser.add_argument('--replicate_idx', type=int, default=0,
                    help='replicate index')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

if __name__ == '__main__':

    ## get data
    source_setup_func = return_setup_func_from_dataset(args.source_dataset)

    rna, atac, cell_group, _, _, atac_datapath, rna_datapath \
        = source_setup_func(args, return_raw_data=True, hvg_only=False, protein_coding_only=True, return_type='data')

    ## LSI for ATAC
    epi.tl.tfidf(atac)
    epi.tl.lsi(atac, n_components=50)

    ## subsample data
    if args.eclare_job_id is not None:

        ## get eclare valid cell ids, if eclare_job_id is provided
        eclare_student_model_ids_path_str = f'eclare_*{args.eclare_job_id}/{args.source_dataset}/{args.replicate_idx}/valid_cell_ids.pkl'
        eclare_student_model_ids_paths = glob(os.path.join(os.environ['OUTPATH'], eclare_student_model_ids_path_str))
        assert len(eclare_student_model_ids_paths) > 0, f'Model IDs path not found @ {eclare_student_model_ids_path_str}'
        with open(eclare_student_model_ids_paths[0], 'rb') as f:
            valid_cell_ids = pickle.load(f)

        ## identify validation set cells
        valid_bool_rna = rna.obs_names.isin(valid_cell_ids['valid_cell_ids_rna'])
        valid_bool_atac = atac.obs_names.isin(valid_cell_ids['valid_cell_ids_atac'])

        ## split into train and valid sets
        rna_valid = rna[valid_bool_rna].copy()
        atac_valid = atac[valid_bool_atac].copy()

        rna_train = rna[~valid_bool_rna].copy()
        atac_train = atac[~valid_bool_atac].copy()

        del rna, atac

    else:

        ## subsample rna data
        rna_cell_types = rna.obs[cell_group]
        train_len = len(rna) - args.valid_subsample
        valid_len = args.valid_subsample
        train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(rna)), y=rna_cell_types))

        rna_train = rna[train_idx].copy()
        rna_valid = rna[valid_idx].copy()
        del rna

        ## subsample atac data
        atac_cell_types = atac.obs[cell_group]
        train_len = len(atac) - args.valid_subsample
        valid_len = args.valid_subsample
        train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=42).split(X=np_empty(len(atac)), y=atac_cell_types))

        atac_train = atac[train_idx].copy()
        atac_valid = atac[valid_idx].copy()
        del atac

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
        use_layer="counts", use_rep="X_lsi"
    )

    ## Extract subgraph for highly variable features
    guidance_hvf = guidance.subgraph(chain(
        rna_train.var.query("highly_variable").index,
        atac_train.var.query("highly_variable").index
    )).copy()

    ## Train GLUE model
    print("Training GLUE model... \n WARNING: Skipping balance step")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna_train, "atac": atac_train},
        guidance_hvf,
        skip_balance=True,
        fit_kws={"directory": os.path.join(os.environ['OUTPATH'], "glue"), "max_epochs": args.n_epochs}
    )

    ## Project nuclei into latent embedding
    rna_latents_valid = glue.encode_data("rna", rna_valid)
    atac_latents_valid = glue.encode_data("atac", atac_valid)

    ## create adata from latents
    import pandas as pd
    from anndata import AnnData
    import numpy as np
    import scanpy as sc

    X = np.concatenate([rna_latents_valid, atac_latents_valid], axis=0)
    obs = pd.concat([rna_valid.obs, atac_valid.obs], axis=0)
    obs['Age_Range'] = pd.Categorical(obs['Age_Range'], categories=rna_valid.obs['Age_Range'].cat.categories, ordered=True)
    adata_valid = AnnData(X=X, obs=obs)
    adata_valid.obs['modality'] = ['RNA'] * len(rna_valid) + ['ATAC'] * len(atac_valid)

    ## create unique identifier for this run based on date and time
    import datetime
    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(os.environ['OUTPATH'], 'glue', unique_id), exist_ok=True)

    ## umap
    sc.pp.neighbors(adata_valid)
    sc.tl.umap(adata_valid)
    fig = sc.pl.umap(adata_valid, color=['modality', 'Age_Range', 'Lineage'], return_fig=True)

    import matplotlib.pyplot as plt
    fig.savefig(os.path.join(os.environ['OUTPATH'], 'glue', unique_id, 'glue_latents.png'), dpi=96, bbox_inches='tight')
    plt.close()

    ## save adata
    for col in adata_valid.obs.columns:
        if adata_valid.obs[col].dtype == 'object':
            adata_valid.obs[col] = adata_valid.obs[col].astype(str)

    adata_valid.attrs = {'eclare_job_id': args.eclare_job_id}
    adata_valid.write_h5ad(os.path.join(os.environ['OUTPATH'], 'glue', unique_id, 'glue_latents.h5ad'))



