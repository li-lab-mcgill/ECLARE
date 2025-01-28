from anndata import read_h5ad
from scanpy.external.pp import magic
import scipy.sparse
import pandas as pd
import numpy as np
import os
from torch.cuda import is_available as cuda_is_available
from torch.cuda import device_count as cuda_device_count

import warnings
np.warnings = warnings
#from scalex import SCALEX

import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    datapath = '/home/dmannk/projects/def-liyue/dmannk/data/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc'

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    datapath = '/Users/dmannk/cisformer/workspace/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc'


if cuda_is_available():
    print('CUDA available')
    device   = 'cuda'
    num_gpus = cuda_device_count()
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


def unwind_full_mask(mask, adata):
        # Iterate over the dataset names and fill in the mask
        cell_by_interval_mask = pd.DataFrame(
            np.zeros((adata.n_obs, mask.shape[0]), dtype=bool),
            index=mask.index,
            columns=mask.index
        )

        for dataset in original_peaks_mask.columns:
            # Identify cells that belong to this dataset
            cells_in_dataset = adata.obs['dataset'] == dataset
            
            # Assign the boolean mask from the original_peaks_mask
            cell_by_interval_mask.loc[cells_in_dataset, :] = original_peaks_mask[dataset].values
        
        return cell_by_interval_mask.values

if __name__ == "__main__":

    filepath = os.path.join(datapath, 'rna_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad')
    rna = read_h5ad(filepath)

    # original_peaks_mask = pd.read_csv(os.path.join(datapath, 'original_peaks_mask_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.csv'), index_col=0)
    # original_peaks_mask = unwind_full_mask(original_peaks_mask, atac.n_obs)

    ## impute data using MAGIC
    print('Imputing data...', flush=True)
    magic(rna, solver='approximate', n_pca=20, n_jobs=cpus_per_task, verbose=True)

    #sparsity = rna.raw.X.nnz / np.multiply(*rna.shape)
    impute_thresh = np.flip(np.sort(rna.X.flatten()))[rna.raw.X.nnz]
    rna.X[rna.X < impute_thresh] = 0

    rna.obsm['imputed'] = scipy.sparse.csr_matrix(rna.X)
    del rna.X
    rna.X = rna.raw.X
    
    ## load original peaks mask, and 'unwind' into a full nuclei x peaks matrix
    #print('Replacing missing peaks by imputed values...', flush=True)
    #original_peaks_mask = unwind_full_mask(atac_imp.var, atac_imp.n_obs)
    #original_peaks_mask = scipy.sparse.csr_matrix(~original_peaks_mask.values)
    #imputed_peaks_masked = atac_imp['impute'].multiply(original_peaks_mask)
    #atac_imp.X = atac_imp.X + imputed_peaks_masked

    ## save imputed data
    print('Saving imputed data...', flush=True)
    rna.write(os.path.join(datapath, 'rna_imputed_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc_imputed.h5ad'))

    print('Done!')