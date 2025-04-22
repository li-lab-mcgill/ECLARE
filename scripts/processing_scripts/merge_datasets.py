from argparse import ArgumentParser
import os
from numpy import isin as np_isin

from setup_utils import \
    CAtlas_Tabula_Sapiens_setup, pbmc_10x_setup, Roussos_cerebral_cortex_setup, merge_datasets_union, snMultiome_388_human_brains_setup, AD_Anderson_et_al_setup, PD_Adams_et_al_setup, human_dlpfc_setup

import socket
hostname = socket.gethostname()

if hostname.startswith('narval'):
    os.environ['machine'] = 'narval'
    datapath = '/home/dmannk/projects/def-liyue/dmannk/data/merged_data'

elif (hostname == 'MBP-de-Dylan.lan') or (hostname == 'MacBook-Pro-de-Dylan.local'):
    os.environ['machine'] = 'local'
    datapath = '/Users/dmannk/cisformer/workspace/merged_data'

if __name__ == "__main__":

    parser = ArgumentParser(description='CAtlas_celltyping')
    parser.add_argument('--datasets', type=list, default=['roussos', 'AD_Anderson_et_al', 'PD_Adams_et_al', 'human_dlpfc'],
                        help='list of datasets to merge')
    parser.add_argument('--genes_by_peaks_str', type=str, default=None,
                        help='indicator of peaks to genes mapping to skip processing')
    args = parser.parse_args()

    setup_funcs = []
    dataset_names = []

    rna_datasets = []
    atac_datasets = []
    cell_groups = []

    ## set setup func for CLIP dataset
    if np_isin('CAtlas_Tabula_Sapiens', args.datasets):
        setup_func = CAtlas_Tabula_Sapiens_setup

    if np_isin('pbmc_10x', args.datasets):
        setup_funcs.append(pbmc_10x_setup)

    if np_isin('roussos', args.datasets):
        setup_funcs.append(Roussos_cerebral_cortex_setup)

    if np_isin('388_human_brains', args.datasets):
        setup_funcs.append(snMultiome_388_human_brains_setup)

    if np_isin('AD_Anderson_et_al', args.datasets):
        setup_funcs.append(AD_Anderson_et_al_setup)

    if np_isin('PD_Adams_et_al', args.datasets):
        setup_funcs.append(PD_Adams_et_al_setup)

    if np_isin('human_dlpfc', args.datasets):
        setup_funcs.append(human_dlpfc_setup)

    ## extract data
    print('Extracting data')
    for setup_func in setup_funcs:

        rna, atac, cell_group, _, _, _, _ = setup_func(args, pretrain=None, return_type='data')

        cell_groups.append(cell_group)
        rna_datasets.append(rna)
        atac_datasets.append(atac)

    '''
    with open("cell_groups.pkl", "wb") as f:
        pkl.dump(cell_groups, f)

    for i, atac in enumerate(atac_datasets):
        atac.write(f"atac_{i}.h5ad")

    for i, rna in enumerate(rna_datasets):
        rna.write(f"rna_{i}.h5ad")
    '''

    '''
    with open("cell_groups.pkl", "rb") as f:
        cell_groups = pkl.load(f)

    atac_datasets = []
    for i in range(4):
        atac = read_h5ad(f"atac_{i}.h5ad")
        atac_datasets.append(atac)

    rna_datasets = []
    for i in range(4):
        rna = read_h5ad(f"rna_{i}.h5ad")
        rna_datasets.append(rna)
    '''

    ## merge data
    merged_rna, merged_atac, original_genes_mask, original_peaks_mask = merge_datasets_union(rna_datasets, atac_datasets, cell_groups, args.datasets)
    merge_dataset_name = '_'.join(args.datasets)

    ## sort peaks & genes in mask matrix and add to ATAC anndata
    original_peaks_mask = original_peaks_mask.iloc[original_peaks_mask.index.argsort()]
    original_peaks_mask = original_peaks_mask.rename(columns={col: f'original_peaks_mask_{col}' for col in original_peaks_mask.columns})
    merged_atac.var = atac.var.merge(original_peaks_mask, left_index=True, right_index=True)

    original_genes_mask = original_genes_mask.iloc[original_genes_mask.index.argsort()]
    original_genes_mask = original_genes_mask.rename(columns={col: f'original_genes_mask_{col}' for col in original_genes_mask.columns})
    merged_rna.var = merged_rna.var.merge(original_genes_mask, left_index=True, right_index=True)

    ## create new directory for merged data
    merged_datapath = os.path.join(datapath, merge_dataset_name)
    if not os.path.exists(merged_datapath):
        os.makedirs(merged_datapath)

    ## save data
    print('Saving merged data and masks')
    merged_rna.write_h5ad(os.path.join(merged_datapath, f'rna_merged_{merge_dataset_name}.h5ad'))
    merged_atac.write_h5ad(os.path.join(merged_datapath, f'atac_merged_{merge_dataset_name}.h5ad'))

    ## save original_genes_mask and original_peaks_mask
    original_genes_mask.to_csv(os.path.join(merged_datapath, f'original_genes_mask_{merge_dataset_name}.csv'), columns=args.datasets)
    original_peaks_mask.to_csv(os.path.join(merged_datapath, f'original_peaks_mask_{merge_dataset_name}.csv'), columns=args.datasets)

    print('Done')

    
