from argparse import ArgumentParser
from setup_utils import \
    CAtlas_Tabula_Sapiens_setup, mdd_setup, pbmc_multiome_setup, splatter_sim_setup, toy_simulation_setup, Roussos_cerebral_cortex_setup, retain_feature_overlap, snMultiome_388_human_brains_setup, snMultiome_388_human_brains_one_subject_setup, AD_Anderson_et_al_setup, PD_Adams_et_al_setup, human_dlpfc_setup, sea_ad_setup
from scipy.sparse import save_npz
from pickle import dump as pkl_dump
from anndata import read_h5ad
import numpy as np
import os

import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'

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


    ## SOURCE dataset setup function
    if args.source_dataset == 'CAtlas_Tabula_Sapiens':
        setup_func = CAtlas_Tabula_Sapiens_setup

    elif args.source_dataset == 'pbmc_multiome':
        source_setup_func = pbmc_multiome_setup
        #source_rna_datapath = source_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/10x_pbmc'
        #source_RNA_file = "pbmcMultiome_rna.h5ad"
        #source_ATAC_file = "pbmcMultiome_atac.h5ad"

    elif args.source_dataset == 'roussos':
        source_setup_func = Roussos_cerebral_cortex_setup
        #source_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/Roussos_lab'
        #rna_datapath = os.path.join(source_datapath, 'rna')
        #atac_datapath = os.path.join(source_datapath, 'atac')

    elif args.source_dataset == '388_human_brains':
        source_setup_func = snMultiome_388_human_brains_setup
        #source_datapath = source_rna_datapath = source_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/388_human_brains'

    elif args.source_dataset == '388_human_brains_one_subject':
        source_setup_func = snMultiome_388_human_brains_one_subject_setup
        #subject = 'RT00391N'
        #datapath = rna_datapath = atac_datapath = os.path.join('/home/dmannk/projects/def-liyue/dmannk/data/388_human_brains', subject)

    elif args.source_dataset == 'AD_Anderson_et_al':
        source_setup_func = AD_Anderson_et_al_setup
        #datapath = rna_datapath = atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/AD_Anderson_et_al'

    elif args.source_dataset == 'PD_Adams_et_al':
        source_setup_func = PD_Adams_et_al_setup

    elif args.source_dataset == 'human_dlpfc':
        source_setup_func = human_dlpfc_setup

    elif args.source_dataset == 'sea_ad':
        source_setup_func = sea_ad_setup


    ## TARGET dataset setup function
    if args.target_dataset == 'mdd':
        target_setup_func = mdd_setup

    elif args.target_dataset == 'pbmc_multiome':
        target_setup_func = pbmc_multiome_setup
        #target_rna_datapath = target_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/10x_pbmc'
        #target_RNA_file = "pbmcMultiome_rna.h5ad"
        #target_ATAC_file = "pbmcMultiome_atac.h5ad"

    elif args.target_dataset == 'roussos':
        target_setup_func = Roussos_cerebral_cortex_setup
        #target_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/Roussos_lab'
        #target_rna_datapath = os.path.join(target_datapath, 'rna')
        #target_atac_datapath = os.path.join(target_datapath, 'atac')

    elif args.target_dataset == '388_human_brains':
        target_setup_func = snMultiome_388_human_brains_setup
        #target_datapath = target_rna_datapath = target_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/388_human_brains'

    elif args.target_dataset == '388_human_brains_one_subject':
        target_setup_func = snMultiome_388_human_brains_one_subject_setup
        #subject = 'RT00391N'
        #datapath = rna_datapath = atac_datapath = os.path.join('/home/dmannk/projects/def-liyue/dmannk/data/388_human_brains', subject)

    elif args.target_dataset == 'AD_Anderson_et_al':
        target_setup_func = AD_Anderson_et_al_setup

    elif args.target_dataset == 'PD_Adams_et_al':
        target_setup_func = PD_Adams_et_al_setup

    elif args.target_dataset == 'human_dlpfc':
        target_setup_func = human_dlpfc_setup

    elif args.target_dataset == 'sea_ad':
        target_setup_func = sea_ad_setup
    


    ## extract data
    print('Extracting data')

    if args.source_dataset == 'merged_roussos_pbmc_multiome':
        source_rna_datapath = source_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/merged_data'
        source_rna = read_h5ad(os.path.join(source_rna_datapath, 'rna_merged_roussos_pbmc_multiome.h5ad'))
        source_atac = read_h5ad(os.path.join(source_atac_datapath, 'atac_merged_roussos_pbmc_multiome.h5ad'))
        source_cell_group = 'Cell type'

    elif (args.source_dataset == 'merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc') or (args.source_dataset == 'imputed_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc'):
        if os.environ['machine'] == 'narval':
            source_rna_datapath = soure_atac_datapath = '/home/dmannk/projects/def-liyue/dmannk/data/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc'
        elif os.environ['machine'] == 'local':
            source_rna_datapath = source_atac_datapath = '/Users/dmannk/cisformer/workspace/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc'

        source_cell_group = 'Cell type'

        if args.dataset == 'merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc':
            source_rna = read_h5ad(os.path.join(source_rna_datapath, 'rna_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'))
            source_atac = read_h5ad(os.path.join(source_atac_datapath, 'atac_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'))

        elif args.dataset == 'imputed_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc':
            source_rna = read_h5ad(os.path.join(source_rna_datapath, 'rna_imputed_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'))
            source_atac = read_h5ad(os.path.join(source_atac_datapath, 'atac_imputed_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'))

            del source_rna.X, source_atac.X
            source_rna.X = source_rna.obsm['imputed']
            source_atac.X = source_atac.obsm['imputed']

    elif args.source_dataset is not None:
        source_rna, source_atac, source_cell_group, _, _, source_atac_datapath, source_rna_datapath \
            = source_setup_func(args, hvg_only=True, protein_coding_only=True, pretrain=None, return_type='data')

    target_rna, target_atac, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, target_atac_datapath, target_rna_datapath \
        = target_setup_func(args, pretrain=None, return_type='data')
    
    ## delete atac.raw if it exists
    if (args.target_dataset=='mdd') and ('raw' in target_atac.uns.keys()):
        del target_atac.raw

    ## Overwrite paths where data is saved
    #print('!! --- Overwriting save paths --- !!')
    #source_rna_datapath = source_atac_datapath = target_rna_datapath = target_atac_datapath = '/home/dmannk/scratch'

    if args.source_dataset is not None:

        ## harmonize data
        print('Harmonizing data')
        source_rna, target_rna, source_atac, target_atac, target_genes_to_peaks_binary_mask, target_genes_peaks_dict \
            = retain_feature_overlap(args, source_rna, target_rna, source_atac, target_atac, source_cell_group, target_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, return_type='data')
        
        print('Saving source data')
        source_rna.write_h5ad(os.path.join(source_rna_datapath, f'rna_{source_rna.n_vars}_by_{source_atac.n_vars}_aligned_target_{args.target_dataset}.h5ad'))
        source_atac.write_h5ad(os.path.join(source_atac_datapath, f'atac_{source_rna.n_vars}_by_{source_atac.n_vars}_aligned_target_{args.target_dataset}.h5ad'))

        ## save data
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

    
