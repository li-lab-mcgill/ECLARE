ECLARE_root = '/home/mcb/users/dmannk/scMultiCLIP/ECLARE/'

import sys
import os
sys.path.insert(0, os.path.join(ECLARE_root, 'scripts', 'benchmark_diagonal', 'scJoint', 'scJoint'))
from config import Config
import main  # import from scJoint

sys.path.insert(0, os.path.join(ECLARE_root, 'src'))
sys.path.insert(0, ECLARE_root)

import numpy as np
import socket
hostname = socket.gethostname()
if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    datapath = '/home/dmannk/scratch/scJoint_data_tmp/'
    gtf = "/home/dmannk/projects/def-liyue/dmannk/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    datapath = '/Users/dmannk/cisformer/CLARE/benchmark_diagonal/scJoint/data/'
    gtf = "/Users/dmannk/cisformer/workspace/genome_annot/gencode.v45lift37.basic.annotation.gtf"

elif np.any([mcb_server in hostname for mcb_server in ['mcb', 'buckeridge' ,'hlr', 'ri', 'wh', 'yl']]):
    os.environ['machine'] = 'mcb_server'
    datapath = '/home/mcb/users/dmannk/scMultiCLIP/outputs/scJoint_data_tmp/'
    gtf = "/home/mcb/users/dmannk/scMultiCLIP/data/genome_annot/gencode.v45lift37.basic.annotation.gtf"


from eclare import set_env_variables
set_env_variables(os.path.join(ECLARE_root, 'config'))

## create unique identifier for this run based on date and time
import datetime
unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ['outdir'] = os.path.join(os.environ['OUTPATH'], 'scJoint_data_tmp', unique_id, '')
os.makedirs(os.path.join(os.environ['outdir']), exist_ok=True)

# Remove ECLARE root after setting up environment variables, ensures that scJoint 'config' is prioritized over ECLARE 'config'
while ECLARE_root in sys.path:
    sys.path.remove(ECLARE_root)

from eclare.setup_utils import return_setup_func_from_dataset
from eclare.data_utils import gene_activity_score_glue
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import h5py
from scipy.sparse import csc_matrix
import pandas as pd
import process_db
from torch import from_numpy
from torch.cuda import is_available as cuda_available
import scanpy as sc
from glob import glob
import pickle

scjoint_datapath = os.path.join(os.environ['OUTPATH'], 'scJoint_data_tmp')

def write_10X_h5(adata, file, cell_group, path=scjoint_datapath):
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
parser.add_argument('--n_epochs', type=int, default=2,
                    help='number of epochs')
parser.add_argument('--source_dataset', type=str, default='Cortex_Velmeshev',
                    help='dataset to use')
parser.add_argument('--target_dataset', type=str, default=None,
                    help='target dataset')
parser.add_argument('--valid_subsample', type=int, default=5000,
                    help='number of nuclei to subsample for validation')
parser.add_argument('--genes_by_peaks_str', type=str, default='9584_by_66620',
                    help='genes by peaks string')
parser.add_argument('--replicate_idx', type=int, default=0,
                    help='replicate index')
parser.add_argument('--eclare_job_id', type=str, default=None,
                    help='eclare job id')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

if __name__ == '__main__':

    ## get data
    setup_func = return_setup_func_from_dataset(args.source_dataset)
    rna, atac, cell_group, _, _, atac_datapath, rna_datapath = setup_func(args, return_type='data')

    ## Convert to csc_matrix for proper indptr
    rna.X = csc_matrix(rna.layers['counts'])
    atac.X = csc_matrix(atac.layers['counts'])

    ## Get gene activity scores
    atac.var.rename(columns={'peak_og': 'interval'}, inplace=True)
    atac, rna = gene_activity_score_glue(atac, rna)

    ## Script requires equal number of cells - Subsample RNA using stratified sampling based on Age_Range parameter
    rna_dev_groups = rna.obs['Age_Range'].to_list()
    n_cells = len(atac)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_cells, test_size=len(rna) - n_cells, random_state=42)
    subsample_idx, _ = next(sss.split(np.empty_like(rna_dev_groups), rna_dev_groups))
    rna = rna[subsample_idx].copy()

    write_10X_h5(rna, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}', cell_group)
    write_10X_h5(atac, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}', cell_group)

    rna_h5_files = [os.path.join(scjoint_datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.h5')]
    rna_label_files = [os.path.join(scjoint_datapath, f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv')]

    atac_h5_files = [os.path.join(scjoint_datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}.h5')]
    atac_label_files = [os.path.join(scjoint_datapath, f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_celltypes.csv')]

    process_db.data_parsing(rna_h5_files, atac_h5_files)

    rna_label = pd.read_csv(rna_label_files[0], header=0, index_col=0)
    print(rna_label.value_counts(sort = False))
    process_db.label_parsing(rna_label_files, atac_label_files)

    config = Config()

    config.datapath = scjoint_datapath
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

    if args.eclare_job_id is not None:

        ## get eclare valid cell ids, if eclare_job_id is provided
        eclare_student_model_ids_path_str = f'eclare_*{args.eclare_job_id}/{args.source_dataset}/{args.replicate_idx}/valid_cell_ids.pkl'
        eclare_student_model_ids_paths = glob(os.path.join(os.environ['OUTPATH'], eclare_student_model_ids_path_str))
        assert len(eclare_student_model_ids_paths) > 0, f'Model IDs path not found @ {eclare_student_model_ids_path_str}'
        with open(eclare_student_model_ids_paths[0], 'rb') as f:
            valid_cell_ids = pickle.load(f)

        valid_ids_rna = valid_cell_ids['valid_cell_ids_rna']
        valid_ids_atac = valid_cell_ids['valid_cell_ids_atac']

        ## find indices of valid cell ids in rna and atac
        rna_bool_valid = rna.obs_names.isin(valid_ids_rna)
        atac_bool_valid = atac.obs_names.isin(valid_ids_atac)

        ## return indices of valid and train cell ids in rna and atac
        rna_valid_idx = np.where(rna_bool_valid)[0]
        atac_valid_idx = np.where(atac_bool_valid)[0]

        rna_train_idx = np.where(~rna_bool_valid)[0]
        atac_train_idx = np.where(~atac_bool_valid)[0]

        ## set valid cell ids to config
        config.rna_valid_idx = [rna_valid_idx]
        config.atac_valid_idx = [atac_valid_idx]

        config.rna_train_idx = [rna_train_idx]
        config.atac_train_idx = [atac_train_idx]

    ## train model
    model, index_dict = main.main(config)

    ## get indices, originally defined in Stage 1
    rna_train_idx = index_dict['rna_train']
    rna_valid_idx = index_dict['rna_valid']
    atac_train_idx = index_dict['atac_train']
    atac_valid_idx = index_dict['atac_valid']

    ## fetch embeddings
    rna_latents_valid = pd.read_csv(os.path.join(os.environ['outdir'], f'rna_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')
    atac_latents_valid = pd.read_csv(os.path.join(os.environ['outdir'], f'atac_source_{args.source_dataset}_target_{args.target_dataset}_{args.replicate_idx}_embeddings.txt'), header=None, sep=' ')

    rna_latents_valid = from_numpy(rna_latents_valid.values)
    atac_latents_valid = from_numpy(atac_latents_valid.values)

    X = np.concatenate([rna_latents_valid, atac_latents_valid], axis=0)
    obs = pd.concat([rna.obs.iloc[rna_valid_idx], atac.obs.iloc[atac_valid_idx]], axis=0)
    adata_valid = sc.AnnData(X, obs=obs)
    adata_valid.obs['modality'] = ['RNA'] * len(rna.obs.iloc[rna_valid_idx]) + ['ATAC'] * len(atac.obs.iloc[atac_valid_idx])
    adata_valid.obs['Age_Range'] = pd.Categorical(adata_valid.obs['Age_Range'], categories=rna.obs['Age_Range'].cat.categories, ordered=True)

    ## umap
    sc.pp.neighbors(adata_valid)
    sc.tl.umap(adata_valid)
    fig = sc.pl.umap(adata_valid, color=['modality', 'Lineage', 'Age_Range'], return_fig=True)

    import matplotlib.pyplot as plt
    fig.savefig(os.path.join(os.environ['outdir'], 'scJoint_latents.png'), dpi=96, bbox_inches='tight')
    plt.close()

    ## save adata
    for col in adata_valid.obs.columns:
        if adata_valid.obs[col].dtype == 'object':
            adata_valid.obs[col] = adata_valid.obs[col].astype(str)

    adata_valid.attrs = {'eclare_job_id': args.eclare_job_id}
    adata_valid.write_h5ad(os.path.join(os.environ['outdir'], 'scJoint_latents.h5ad'))