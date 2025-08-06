from argparse import ArgumentParser
from scipy.sparse import save_npz
from pickle import dump as pkl_dump
from anndata import read_h5ad
import numpy as np
import os

from eclare import return_setup_func_from_dataset
from eclare.setup_utils import retain_feature_overlap


if __name__ == "__main__":

    parser = ArgumentParser(description='CAtlas_celltyping')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='source dataset')
    parser.add_argument('--target_dataset', type=str, default='mdd',
                        help='target dataset')
    parser.add_argument('--atac_datapath', type=str, default='/Users/dmannk/cisformer/workspace',
                        help='path to ATAC data')
    parser.add_argument('--rna_datapath', type=str, default='/Users/dmannk/cisformer/workspace',
                        help='path to RNA data'),
    parser.add_argument('--genes_by_peaks_str', type=str, default=None,
                        help='indicator of peaks to genes mapping to skip processing')
    parser.add_argument('--feature', type=str, default=None)
    args = parser.parse_args()

    ## TARGET dataset setup function
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)

    ## extract data
    print('Extracting target data')
    target_rna, target_atac, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, target_atac_datapath, target_rna_datapath \
        = target_setup_func(args, hvg_only=False, return_type='data')
    
    ## delete atac.raw if it exists
    if (args.target_dataset=='MDD') and ('raw' in target_atac.uns.keys()):
        del target_atac.raw

    ## Overwrite paths where data is saved
    #print('!! --- Overwriting save paths --- !!')
    #source_rna_datapath = source_atac_datapath = target_rna_datapath = target_atac_datapath = '/home/dmannk/scratch'

    if args.source_dataset is not None:

        ## SOURCE dataset setup function
        print('Extracting source data')
        source_setup_func = return_setup_func_from_dataset(args.source_dataset)

        source_rna, source_atac, source_cell_group, _, _, source_atac_datapath, source_rna_datapath \
            = source_setup_func(args, hvg_only=False, return_type='data')

        ## harmonize data
        print('Harmonizing data')
        source_rna, target_rna, source_atac, target_atac, target_genes_to_peaks_binary_mask, target_genes_peaks_dict \
            = retain_feature_overlap(args, source_rna, target_rna, source_atac, target_atac, source_cell_group, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, return_type='data')
        
        ## save data
        print('Saving source data')
        source_rna.write_h5ad(os.path.join(source_rna_datapath, f'rna_{source_rna.n_vars}_by_{source_atac.n_vars}_aligned_target_{args.target_dataset}.h5ad'))
        source_atac.write_h5ad(os.path.join(source_atac_datapath, f'atac_{source_rna.n_vars}_by_{source_atac.n_vars}_aligned_target_{args.target_dataset}.h5ad'))

        print('Saving target data')
        target_rna.write_h5ad(os.path.join(target_rna_datapath, f'rna_{target_rna.n_vars}_by_{target_atac.n_vars}_aligned_source_{args.source_dataset}.h5ad'))
        target_atac.write_h5ad(os.path.join(target_atac_datapath, f'atac_{target_rna.n_vars}_by_{target_atac.n_vars}_aligned_source_{args.source_dataset}.h5ad'))
        
        print('Saving genes-peaks mask')
        genes_to_peaks_binary_mask_path = os.path.join(source_atac_datapath, f'genes_to_peaks_binary_mask_{target_rna.n_vars}_by_{target_atac.n_vars}_aligned_target_{args.target_dataset}.npz')
        save_npz(genes_to_peaks_binary_mask_path, target_genes_to_peaks_binary_mask)
        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        with open(pkl_path, 'wb') as f: pkl_dump(target_genes_peaks_dict, f)


    elif args.source_dataset is None:

        ## save data
        print('Saving target data (no source data)')
        target_rna.write_h5ad(os.path.join(target_rna_datapath, f'rna_{target_rna.n_vars}_by_{target_atac.n_vars}.h5ad'))
        target_atac.write_h5ad(os.path.join(target_atac_datapath, f'atac_{target_rna.n_vars}_by_{target_atac.n_vars}.h5ad'))
        
        print('Saving genes-peaks mask')
        genes_to_peaks_binary_mask_path = os.path.join(target_atac_datapath, f'genes_to_peaks_binary_mask_{target_rna.n_vars}_by_{target_atac.n_vars}.npz')
        save_npz(genes_to_peaks_binary_mask_path, target_genes_to_peaks_binary_mask)
        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        with open(pkl_path, 'wb') as f: pkl_dump(target_genes_peaks_dict, f)

    
    ## convert atac.X and rna.X back to csr
    #source_rna.X = source_rna.X.astype(np.float64).tocsr()
    #source_atac.X = source_atac.X.astype(np.float64).tocsr()

    print('Done')

    
