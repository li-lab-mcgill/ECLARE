import argparse
import os
import gc
from pickle import load as pkl_load
from pickle import dump as pkl_dump
from scipy.sparse import load_npz, save_npz

from eclare.setup_utils import return_setup_func_from_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split dev stages')
    parser.add_argument('--source_dataset', type=str, default='PFC_V1_Wang')
    parser.add_argument('--target_dataset', type=str, default=None)
    parser.add_argument('--genes_by_peaks_str', type=str, default='9914_by_63404')
    args = parser.parse_args()

    ## SOURCE dataset setup function
    developmental_datasets = ['PFC_Zhu', 'PFC_V1_Wang']
    assert args.source_dataset in developmental_datasets, "ORDINAL only implemented for developmental datasets"
    source_setup_func = return_setup_func_from_dataset(args.source_dataset)

    dev_stages_dict = {
        'PFC_Zhu': ['EaFet', 'LaFet', 'Inf', 'Child', 'Adol', 'Adult'],
        'PFC_V1_Wang': ['FirstTrim', 'SecTrim', 'ThirdTrim', 'Inf', 'Adol'],
        'Cortex_Velmeshev': ['2nd trimester', '3rd trimester', '0-1 years', '1-2 years', '2-4 years', '4-10 years', '10-20 years', 'Adult']
    }
    dev_stages = dev_stages_dict[args.source_dataset]

    ## get data - preload all dev stages, then split into dev stages in loop
    rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = \
        source_setup_func(args, return_type='data', keep_group=[''])

    print('Total number of cells:', len(rna))

    for dev_stage in dev_stages:

        ## target of new dataset is source dataset
        RNA_file = f"{dev_stage}_rna_{args.genes_by_peaks_str}_aligned_target_{args.source_dataset}.h5ad"
        ATAC_file = f"{dev_stage}_atac_{args.genes_by_peaks_str}_aligned_target_{args.source_dataset}.h5ad"

        rna_dat = rna[rna.obs['dev_stage'].str.contains(dev_stage)]
        rna_dat.obs['original_target'] = str(args.target_dataset)
        rna_dat.write_h5ad(os.path.join(rna_datapath, RNA_file))

        atac_dat = atac[atac.obs['dev_stage'].str.contains(dev_stage)]
        atac_dat.obs['original_target'] = str(args.target_dataset)
        atac_dat.write_h5ad(os.path.join(atac_datapath, ATAC_file))

        print(f'Number of cells in {dev_stage}:', len(rna_dat))
        del rna_dat, atac_dat
        gc.collect()

    ## load binary mask and genes_peaks_dict and save copies of them
    if args.target_dataset is not None:
        binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
        genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"
    else:
        binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}.npz"
        genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}.pkl"

    genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, binary_mask_file)
    genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
    pkl_path = os.path.join(atac_datapath, genes_peaks_dict_file)
    with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

    ## save copies of binary mask and genes_peaks_dict
    genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f'genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.source_dataset}.npz')
    save_npz(genes_to_peaks_binary_mask_path, genes_to_peaks_binary_mask)
    pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
    with open(pkl_path, 'wb') as f: pkl_dump(genes_peaks_dict, f)

    print('Done')
