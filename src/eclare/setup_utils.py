import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import os
from argparse import Namespace

import socket
hostname = socket.gethostname()
problematic_hostname = "hlr-10gpu"  # replace this with actual hostname
if hostname != problematic_hostname:
    from pybedtools import BedTool
else:
    print(f"Warning: Skipping pybedtools import on {hostname} due to GLIBC incompatibility.")
    BedTool = None  # or define a dummy fallback if necessary


#import bbknn
from muon import atac as ac
from muon import read_10x_h5 as muon_read_10x_h5
from muon import read_h5mu as muon_read_h5mu
from muon import MuData

from scipy.sparse import csr_matrix, save_npz, load_npz
from pickle import dump as pkl_dump
from pickle import load as pkl_load
from functools import reduce
from collections import defaultdict
import celltypist
from anndata import AnnData
from glob import glob
from datetime import datetime
import re
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
from copy import deepcopy
from eclare.data_utils import create_loaders

def return_setup_func_from_dataset(dataset_name):

    if dataset_name == 'MDD':
        setup_func = mdd_setup

    elif (dataset_name == 'PFC_Zhu') or (dataset_name == 'roussos'):
        setup_func = pfc_zhu_setup

    elif (dataset_name == 'DLPFC_Anderson') or (dataset_name == 'AD_Anderson_et_al'):
        setup_func = dlpfc_anderson_setup

    elif (dataset_name == 'Midbrain_Adams') or (dataset_name == 'PD_Adams_et_al'):
        setup_func = midbrain_adams_setup

    elif (dataset_name == 'DLPFC_Ma') or (dataset_name == 'human_dlpfc'):
        setup_func = dlpfc_ma_setup

    elif (dataset_name == 'spatialLIBD'):
        setup_func = spatialLIBD_setup

    elif (dataset_name == 'pbmc_10x'):
        setup_func = pbmc_10x_setup

    elif (dataset_name == 'mouse_brain_10x'):
        setup_func = mouse_brain_10x_setup

    elif (dataset_name == 'PFC_V1_Wang'):
        setup_func = pfc_v1_wang_setup

    elif (dataset_name == 'Cortex_Velmeshev'):
        setup_func = cortex_velmeshev_setup

    return setup_func

def get_protein_coding_genes(rna):

    ensembl_path = os.path.join(os.environ['DATAPATH'], 'ensembl_gene_positions.csv')

    protein_coding_genes = pd.read_csv(ensembl_path)
    protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
    protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
    rna.var['protein_coding'] = rna.var.index.isin(protein_coding_genes_list)
    rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()
    return rna

def get_genes_by_peaks(rna, atac, genes_to_peaks_mask_path, window_size = 1e6, feature_selection_method = None):

        ## Merge gene coordinates to RNA
        if np.isin( ['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)'] , rna.var.columns ).all():
            rna_var_tmp = rna.var.copy()
        else:

            gene_positions_path = os.path.join(os.environ['DATAPATH'], 'ensembl_gene_positions.csv')

            results = pd.read_csv(gene_positions_path)
            results = results.rename(columns={'Transcription start site (TSS)':'TSS start (bp)'})
            results['TSS end (bp)'] = results['TSS start (bp)'] + 1

            filt_chroms = [str(i) for i in range(1,23+1)] + ['X','Y']
            results = results[results['Chromosome/scaffold name'].isin(filt_chroms)]
            rna_var_tmp = pd.merge(rna.var, results, left_index=True, right_on='Gene name', how='left').set_index('Gene name')

            rna_var_tmp = pd.merge(rna.var, results, left_index=True, right_on='Gene name', how='left').set_index('Gene name')
            rna_var_tmp = rna_var_tmp.copy()
            rna_var_tmp = rna_var_tmp[rna_var_tmp['protein_coding']]
            rna_var_tmp = rna_var_tmp[~rna_var_tmp[['Chromosome/scaffold name', 'TSS start (bp)', 'TSS end (bp)']].isna().all(1)]
            rna_var_tmp = rna_var_tmp.groupby('Gene name', sort=False).agg({
                        'Chromosome/scaffold name': 'max',
                        'TSS start (bp)': 'mean',
                        'TSS end (bp)': 'mean',
                        'Gene start (bp)': 'mean',
                        'Gene end (bp)': 'mean'})
        
            rna_var_tmp['TSS start (bp)'] = rna_var_tmp['TSS start (bp)'].astype(int).astype(str)
            rna_var_tmp['TSS end (bp)'] = rna_var_tmp['TSS end (bp)'].astype(int).astype(str)
            rna_var_tmp['Gene start (bp)'] = rna_var_tmp['Gene start (bp)'].astype(int).astype(str)
            rna_var_tmp['Gene end (bp)'] = rna_var_tmp['Gene end (bp)'].astype(int).astype(str)
            rna_var_tmp['Chromosome/scaffold name'] = 'chr' + rna_var_tmp['Chromosome/scaffold name']

            #rna.var = pd.merge(rna.var, rna_var_tmp, left_index=True, right_index=True, how='left')

        #rna = rna[:,~rna.var.isna().values.any(1)]

        ## Create bedtools objects
        atac_bed_df = pd.DataFrame(list(atac.var.index.str.split(':|-', expand=True).values), columns=['chrom','start','end'])
        atac_bed_df = atac_bed_df.copy()
        atac_bed_df['name'] = atac.var.index
        atac_bed = BedTool.from_dataframe(atac_bed_df)

        #rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'TSS start (bp)', 'TSS end (bp)', 'index']])
        rna_bed = BedTool.from_dataframe(rna_var_tmp.reset_index()[['Chromosome/scaffold name', 'TSS start (bp)', 'TSS end (bp)', 'Gene start (bp)', 'Gene end (bp)', 'Gene name']])

        ## Find in-cis overlapping intervals
        cis_overlaps = rna_bed.window(atac_bed, w=window_size).to_dataframe()
        i, r = pd.factorize(cis_overlaps['strand'])       # genes
        j, c = pd.factorize(cis_overlaps['blockCount'])   # peaks
        n, m = len(r), len(c)

        ## check if some genes have no peaks in cis
        if n != len(rna_var_tmp):
            print(f'{len(rna_var_tmp) - n} genes have no peaks in cis')
            genes_missing_peaks = set(rna_var_tmp.index).difference(set(r[i]))
            print(genes_missing_peaks)

        ## Feature selection
        if feature_selection_method == 'gradient boosting regression':

            print('Using gradient boosting regression for feature selection')
            significant_associations_mask = np.ones_like(i, dtype=bool)

            params = {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 5,
                "learning_rate": 0.01,
                "loss": "squared_error",
            }

            def process_gene(g, gene):
                peaks_in_cis_with_gene_match = (i == g)
                peaks_in_cis_with_gene_idxs = j[peaks_in_cis_with_gene_match]
                g_idx = np.where(rna.var_names == gene)[0]

                x = atac[:, peaks_in_cis_with_gene_idxs].X.toarray()
                y = rna[:, g_idx].X.toarray().squeeze()
                #y = rna[:, g].X.toarray().squeeze()

                ## check if gene has no associated peak
                if x.shape[1] == 0:
                    return g, peaks_in_cis_with_gene_match, False
                
                else:
                    reg = GradientBoostingRegressor(**params)
                    reg.fit(x, y)
                    feature_importances = reg.feature_importances_

                    feature_importance_thresh = np.max([np.percentile(feature_importances, q=50),
                                                        feature_importances[feature_importances > 0].min()])  # set threshold to median value, or smallest non-zero value if median = 0
                    keep_peaks = (feature_importances >= feature_importance_thresh)

                    return g, peaks_in_cis_with_gene_match, keep_peaks

            ## Run gradient boosting regression in parallel
            results = Parallel(n_jobs=-1)(delayed(process_gene)(g, gene) for g, gene in enumerate(r))

            for g, peaks_in_cis_with_gene_match, keep_peaks in results:
                significant_associations_mask[peaks_in_cis_with_gene_match] = keep_peaks

            ## Save variables needed to create mask, for troubleshooting - wrong path definition
            pkl_dump({'r':r, 'c':c, 'i':i, 'j':j, 'significant_associations_mask':significant_associations_mask, 'results':results}, open( \
                os.path.join(genes_to_peaks_binary_mask_path, 'tmp_gbr_params.pkl') \
                    , 'wb'))

            ## retain peaks and genes indices arising from significant feature importances
            i = i[significant_associations_mask]
            j = j[significant_associations_mask]

            ## retain peaks and genes arising from significant feature importances
            r = r[np.unique(i)]
            c = c[np.unique(j)]
            n, m = len(r), len(c)

            genes_to_peaks_binary_mask_path = os.path.join(genes_to_peaks_binary_mask_path, f'genes_to_peaks_binary_mask_{n}_by_{m}_with_GBR.npz')

        ## Compute distances
        tss = cis_overlaps['start']
        gene_start = cis_overlaps['name']
        gene_end = cis_overlaps['score']
        gene_centre = (tss + gene_end)/2 - 5000/2
        gene_size = gene_end - tss

        peak_start = cis_overlaps['thickEnd']
        peak_end = cis_overlaps['itemRgb']
        peak_centre = (peak_start + peak_end)/2

        distance = np.abs(gene_centre - peak_centre)
        distance_with_gb = distance.copy()
        distance_with_gb = distance_with_gb
        distance_with_gb[distance_with_gb < 0] = 0

        archr_weights = np.exp(-np.abs(distance_with_gb/5000)) + np.exp(-1)
        archr_weights[distance_with_gb > window_size] = 0
        archr_weights = archr_weights.values

        ''' visualise weights by distance
        x = (gene_centre - peak_centre)
        y = archr_weights
        fig, ax = plt.subplots(1,2,figsize=[8,4])
        trim = np.abs(x) < window_size
        ax[0].scatter(x[trim], y[trim])
        trim = np.abs(x) < 2e4
        ax[1].scatter(x[trim], y[trim])
        fig.show()
        '''

        ## Create mask
        genes_to_peaks_distance_mask = np.zeros((n, m), dtype=np.float32)
        np.add.at(genes_to_peaks_distance_mask, (i, j), archr_weights)
        genes_peaks_dict = {'genes':r, 'peaks':c}
        #genes_to_peaks_binary_mask_df = pd.DataFrame(genes_to_peaks_binary_mask, index=r, columns=c)

        ## Save mask
        if genes_to_peaks_mask_path:
            genes_to_peaks_distance_mask = csr_matrix(genes_to_peaks_distance_mask)
            save_npz(genes_to_peaks_mask_path, genes_to_peaks_distance_mask)

            pkl_path = os.path.splitext(genes_to_peaks_mask_path)[0] + '.pkl'
            with open(pkl_path, 'wb') as f: pkl_dump(genes_peaks_dict, f)

        return genes_to_peaks_distance_mask, genes_peaks_dict

def get_gas(rna, atac):
        
        ## Merge gene coordinates to RNA
        if np.isin( ['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)'] , rna.var.columns ).all():
            rna_var_tmp = rna.var
        else:
            results = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'ensembl_gene_positions.csv'))
            filt_chroms = [str(i) for i in range(1,23+1)] + ['X','Y']
            results = results[results['Chromosome/scaffold name'].isin(filt_chroms)]
            rna_var_tmp = pd.merge(rna.var, results, left_index=True, right_on='Gene name', how='left').set_index('Gene name')

        ## if replicate gene names (splicing variants?), take average position
        rna_var_tmp = rna_var_tmp.groupby('Gene name', sort=False).agg({
                    'Chromosome/scaffold name': 'max',
                    'Gene start (bp)': 'mean',
                    'Gene end (bp)': 'mean'})
        
        rna.var = pd.merge(rna.var, rna_var_tmp, left_index=True, right_index=True, how='left')
        rna = rna[:,~rna.var.isna().values.any(1)]

        rna.var['Gene start (bp)'] = rna.var['Gene start (bp)'].astype(int).astype(str)
        rna.var['Gene end (bp)'] = rna.var['Gene end (bp)'].astype(int).astype(str)
        rna.var['Chromosome/scaffold name'] = 'chr' + rna.var['Chromosome/scaffold name']

        ## Create bedtools objects
        rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)', 'index']])
        atac_bed_df = pd.DataFrame(list(atac.var.index.str.split('-', expand=True).values), columns=['chrom','start','end'])
        atac_bed_df['name'] = atac.var.index
        atac_bed = BedTool.from_dataframe(atac_bed_df)

        ## Find peaks at gene and 2000bp upstream for gene activity scores
        gene_size=1e2
        upstream=2e3
        gas_overlaps = rna_bed.window(atac_bed, l=upstream, r=gene_size, c=False).to_dataframe()
        #gas_overlaps['thickEnd'] = gas_overlaps[['chrom','start','end']].astype(str).apply('-'.join, axis=1)

        i, r = pd.factorize(gas_overlaps['name'])
        j, c = pd.factorize(gas_overlaps['thickEnd'])
        n, m = len(r), len(c)

        gas_binary_mask = np.zeros((n, m), dtype=np.int64)
        np.add.at(gas_binary_mask, (i, j), 1)
        gas_binary_mask_partial = pd.DataFrame(gas_binary_mask, index=r, columns=c)

        gas_binary_mask = pd.DataFrame(np.zeros([rna.n_vars, atac.n_vars]), index=rna.var.index, columns=atac.var.index)
        gas_binary_mask = gas_binary_mask.merge(gas_binary_mask_partial, left_index=True, right_index=True, how='left', suffixes=('_x',None))
        gas_binary_mask = gas_binary_mask.loc[:,~gas_binary_mask.columns.str.endswith('_x')]
        gas_binary_mask = gas_binary_mask.fillna(0)
        
        atac_gas = np.matmul( atac.X.toarray() , gas_binary_mask.T)
        atac_gas.index = atac.obs.index
        return atac_gas, rna


def get_multimodal_clusters(rna, atac_gas, neighbors_type='standard'):

    atac_gas = anndata.AnnData(X=atac_gas)
    sc.pp.normalize_total(atac_gas, target_sum=1e4)


    if neighbors_type == 'standard':
        rna_atac_gas = anndata.concat([rna, atac_gas], axis=1)

        sc.pp.pca(rna_atac_gas, n_comps=100, zero_center=False)
        sc.pp.neighbors(rna_atac_gas)
        sc.tl.leiden(rna_atac_gas, resolution=1)

        return rna_atac_gas.obs['leiden']

    elif neighbors_type == 'bbknn':
        rna_atac_gas = anndata.concat([rna, atac_gas], axis=0)
        sc.pp.pca(rna_atac_gas, n_comps=100, zero_center=False)
        rna_atac_gas.obs['modality'] = ['RNA']*rna.n_obs + ['ATAC']*atac_gas.n_obs
        sc.external.pp.bbknn(rna_atac_gas, batch_key='modality', metric='cosine')

        rna_leiden = rna_atac_gas.obs['leiden'].iloc[:rna.n_obs]
        atac_gas_leiden = rna_atac_gas.obs['leiden'].iloc[rna.n_obs:]

        return rna_leiden, atac_gas_leiden


def retain_feature_overlap(
        args: Namespace,
        source_rna: anndata.AnnData,
        target_rna: anndata.AnnData,
        source_atac: anndata.AnnData,
        target_atac: anndata.AnnData,
        source_cell_group: str,
        target_cell_group: str,
        genes_to_peaks_binary_mask: pd.DataFrame,
        genes_peaks_dict: dict,
        peak_overlap_method: str = 'bedtools',
        return_type: str = 'loaders'):
    
    ## align source genes
    overlapping_rna_genes = set(source_rna.var.index).intersection(target_rna.var.index)
    target_genes_filt = target_rna.var.index.isin(overlapping_rna_genes)
    target_rna = target_rna[:, target_genes_filt]
    
    #source_rna = source_rna[:, source_rna.var.index.isin(overlapping_rna_genes)]
    order_genes_idxs = np.array([np.where(gene == source_rna.var.index)[0][0] for gene in target_rna.var.index if np.isin(gene, list(overlapping_rna_genes)).item()]) # long
    source_rna = source_rna[:, order_genes_idxs]

    ## align source peaks

    ## check how peak names are formatted with delimiters
    source_peak_name_str_pattern = len(source_atac.var.index[0].split('-'))
    source_peak_name_str_pattern = 'all dashes' if source_peak_name_str_pattern==3 else 'standard'

    target_peak_name_str_pattern = len(target_atac.var.index[0].split('-'))
    target_peak_name_str_pattern = 'all dashes' if target_peak_name_str_pattern==3 else 'standard'

    ## find overlapping peaks with bedtools
    source_atac_bed = BedTool.from_dataframe(pd.DataFrame(np.stack(source_atac.var.index.str.split(':|-')), columns=['chrom','start','end']))
    target_atac_bed = BedTool.from_dataframe(pd.DataFrame(np.stack(target_atac.var.index.str.split(':|-')), columns=['chrom','start','end']))
    overlapping_atac_peaks = source_atac_bed.intersect(target_atac_bed, wa=True, wb=True).to_dataframe()

    if source_peak_name_str_pattern == 'all dashes':
        source_peak_names = overlapping_atac_peaks.iloc[:,:3].astype(str).apply('-'.join, axis=1)
        target_peak_names = overlapping_atac_peaks.iloc[:,3:].astype(str).apply('-'.join, axis=1)

    elif source_peak_name_str_pattern == 'standard':
        source_peak_names = overlapping_atac_peaks.iloc[:,:3].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
        target_peak_names = overlapping_atac_peaks.iloc[:,3:].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)

    peaks_names_mapper = dict(zip(source_peak_names, target_peak_names)) # not necessarly one-to-one map, can have many-to-one

    source_peaks_filt = source_atac.var.index.isin(source_peak_names) # three lines below "performs alignment"
    source_atac = source_atac[:, source_peaks_filt].to_memory()
    source_atac.var.index = source_atac.var.index.map(peaks_names_mapper)

    duplicates = source_atac.var_names.duplicated(keep='first')
    source_atac = source_atac[:, ~duplicates].copy()

    #source_atac = source_atac[:,~source_atac.var.index.isna()]
    overlapping_atac_peaks = list(set(source_atac.var.index).intersection(set(target_atac.var.index))) # should be equivalent to set(source_atac.var.index)

    ## subset target genes and peaks
    target_peaks_filt = target_atac.var.index.isin(overlapping_atac_peaks)
    target_atac = target_atac[:, target_peaks_filt].to_memory()
    target_atac = target_atac[:, target_atac.var.index.argsort()].to_memory()

    source_peaks_filt = source_atac.var.index.isin(overlapping_atac_peaks)
    source_atac = source_atac[:, source_atac.var.index.argsort()]

    ## check if features align
    print(f'Genes match across datasets: {(source_rna.var.index == target_rna.var.index).all()}')
    print(f'Peaks match across datasets: {(source_atac.var.index == target_atac.var.index).all()}')

    ## subset genes_to_peaks_binary_mask
    if genes_to_peaks_binary_mask is not None:
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[ list(target_rna.var.index) , list(target_atac.var.index) ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[target_genes_filt, :]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[:, target_peaks_filt]

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][target_genes_filt]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][target_peaks_filt]

        print(f"Genes match with mask: {(genes_peaks_dict['genes'] == target_rna.var.index).all()}")
        print(f"Peaks match with mask: {(genes_peaks_dict['peaks'] == target_atac.var.index).all()}")

        print(f'Number of peaks and genes remaining: {target_atac.n_vars} peaks & {target_rna.n_vars} genes')
    else:
        genes_peaks_dict = None

    '''
    if 'X_pca_harmony' in source_rna.obsm:
        harmony_integrate(source_rna, key='subject')
        harmony_integrate(source_atac, key='subject')

        source_rna.X = csr_matrix(source_rna.obsm['X_pca_harmony'].dot(source_rna.varm['PCs'].T))
        source_atac.X = csr_matrix(source_atac.obsm['X_pca_harmony'].dot(source_atac.varm['PCs'].T))
    '''

    if return_type == 'loaders':

        source_rna_train_loader, source_rna_valid_loader, source_rna_valid_idx, _, _, _, _, _, _ = create_loaders(source_rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=source_cell_group)
        source_atac_train_loader, source_atac_valid_loader, source_atac_valid_idx, source_atac_train_num_batches, source_atac_valid_num_batches, source_atac_train_n_batches_str_length, source_atac_valid_n_batches_str_length, source_atac_train_n_epochs_str_length, source_atac_valid_n_epochs_str_length = create_loaders(source_atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=source_cell_group)

        _, target_rna_valid_loader, _, _, _, _, _, _, _ = create_loaders(target_rna, 'mdd', args.batch_size, args.total_epochs, cell_group_key=target_cell_group, valid_only=True)
        _, target_atac_valid_loader, _, target_atac_train_num_batches, target_atac_valid_num_batches, _, target_atac_valid_n_batches_str_length, _, target_atac_valid_n_epochs_str_length = create_loaders(target_atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=target_cell_group, valid_only=True)

        return \
            source_rna_train_loader, source_atac_train_loader, source_atac_train_num_batches, source_atac_train_n_batches_str_length, source_atac_train_n_epochs_str_length, source_rna_valid_loader, source_atac_valid_loader, source_atac_valid_num_batches, source_atac_valid_n_batches_str_length, source_atac_valid_n_epochs_str_length, \
            None, None, target_atac_train_num_batches, None, None, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, \
            genes_to_peaks_binary_mask, genes_peaks_dict
    

    elif return_type == 'data':
        return source_rna, target_rna, source_atac, target_atac, genes_to_peaks_binary_mask, genes_peaks_dict


def merge_datasets_intersection(
        rna_datasets: list,
        atac_datasets: list,
        cell_groups: list,
        cre_path: str = '/home/dmannk/projects/def-liyue/dmannk/data/GRCh38-cCREs.bed'
        ):

    cell_group_mapper = [(cell_group,'Cell type') for cell_group in cell_groups]

    ## RNA genes
    overlapping_rna_genes = reduce(lambda x, y: set(x.var.index) & set(y.var.index), rna_datasets)

    for r, rna_dataset in enumerate(rna_datasets):
        rna_dataset = rna_dataset[:, rna_dataset.var.index.isin(overlapping_rna_genes)]
        rna_dataset = rna_dataset[:, rna_dataset.var.index.argsort()]

        rna_dataset.obs.rename(columns=dict([cell_group_mapper[r]]), inplace=True)
        rna_datasets[r] = rna_dataset

    all_gene_names = np.stack([rna_dataset.var_names.values for rna_dataset in rna_datasets])
    check_gene_names_alignment = (np.apply_along_axis(lambda x: len(np.unique(x)), 0, all_gene_names) == 1).all()  # check if gene names match across datasets by counting the number of unique gene names per row, and checking whether this count is 1 for all rows
    print('Gene names match across datasets:', check_gene_names_alignment)


    ## ATAC peaks
    cre_bedtool = BedTool(cre_path)
    atac_datasets_bedtools = [\
        BedTool.from_dataframe(pd.DataFrame(np.stack(atac.var.index.str.split('-')), columns=['chrom','start','end'])).sort()\
                for atac in atac_datasets]
    
    overlapping_cres = [cre_bedtool.intersect(bed, u=True) for bed in atac_datasets_bedtools]
    overlapping_atac_peaks = reduce(lambda x, y: x.intersect(y, u=True), overlapping_cres)

    # a=1; atac_bed, atac_dataset = (atac_datasets_bedtools[a], atac_datasets[a])
    for a, (atac_bed, atac_dataset) in enumerate(zip(atac_datasets_bedtools, atac_datasets)):
        overlap = overlapping_atac_peaks.intersect(atac_bed, wa=True, wb=True)
        overlap = overlap.to_dataframe(header=None)
        #overlap = overlap[overlap['name']==len(atac_datasets_bedtools)]  # 'name' or 3 corresponds to the number of Bedtools with which an intersection was detected

        source_peak_names = overlap.iloc[:,-3:].astype(str).apply('-'.join, axis=1) # smaller number of unique peaks in source than target, conflict for mapper
        target_peak_names = overlap.iloc[:,:3].astype(str).apply('-'.join, axis=1) # correct number of unique peaks
        #peaks_names_mapper = dict(zip(source_peak_names, target_peak_names))

        aggregated_peaks_names_mapper = defaultdict(list)
        #for key, value in peaks_names_mapper.items(): aggregated_peaks_names_mapper[key].append(value)
        for key, value in zip(source_peak_names, target_peak_names): aggregated_peaks_names_mapper[key].append(value)
        aggregated_peaks_names_mapper = dict(aggregated_peaks_names_mapper)
        #tmp = [len(vals) for vals in aggregated_peaks_names_mapper.values()]

        # Apply the mapping to the DataFrame index
        expanded_rows = []
        retained_peaks_idxs = []

        for p, (peak, row) in enumerate(atac_dataset.var.iterrows()):
            if peak in aggregated_peaks_names_mapper:
                for target_peak in aggregated_peaks_names_mapper[peak]:
                    new_row = row.copy()
                    new_row.name = target_peak

                    expanded_rows.append(new_row)
                    retained_peaks_idxs.append(p)

        expanded_atac_dataset = pd.DataFrame(expanded_rows)
        retained_peaks_idxs = np.asarray(retained_peaks_idxs)

        atac_dataset = atac_dataset[:, retained_peaks_idxs].copy()
        atac_dataset.var = expanded_atac_dataset

        #atac_dataset.var.index = atac_dataset.var.index.map(peaks_names_mapper)
        #atac_dataset = atac_dataset[:,~atac_dataset.var_names.isna()]

        duplicates = atac_dataset.var_names.duplicated(keep='first')
        atac_dataset = atac_dataset[:, ~duplicates].copy()
        atac_dataset = atac_dataset[:, atac_dataset.var_names.argsort()].copy()

        atac_dataset.obs.rename(columns=dict([cell_group_mapper[a]]), inplace=True)
        atac_datasets[a] = atac_dataset

    all_peak_names = np.stack([atac_dataset.var_names.values for atac_dataset in atac_datasets])
    check_peak_names_alignment = (np.apply_along_axis(lambda x: len(np.unique(x)), 0, all_peak_names) == 1).all()  # check if peak names match across datasets by counting the number of unique peak names per row, and checking whether this count is 1 for all rows
    print('Peak names match across datasets:', check_peak_names_alignment)

    ## concatenate
    rna = anndata.concat(rna_datasets, axis=0)
    atac = anndata.concat(atac_datasets, axis=0)

    return rna, atac

def merge_datasets_union(
    rna_datasets: list,
    atac_datasets: list,
    cell_groups: list,
    dataset_names: list,
    cre_path: str = '/home/dmannk/projects/def-liyue/dmannk/data/GRCh38-cCREs.bed'
    ):

    cell_group_mapper = [(cell_group,'Cell type') for cell_group in cell_groups]

    ## RNA genes
    all_gene_names = np.hstack([rna_dataset.var_names.values for rna_dataset in rna_datasets])
    all_gene_names = np.unique(all_gene_names)

    ## intialize mask matrix that indicates whether an overlapping peak is present in the original dataset or not
    original_genes_mask = pd.DataFrame(np.ones((len(all_gene_names), len(dataset_names)), dtype=bool), index=all_gene_names, columns=dataset_names)

    for r, (rna_dataset, dataset_name) in enumerate(zip(rna_datasets, dataset_names)):
        genes_not_in_rna_dataset = np.setdiff1d(all_gene_names, rna_dataset.var_names)

        rna_dataset = anndata.concat([rna_dataset, \
            anndata.AnnData(X=np.zeros((rna_dataset.n_obs, len(genes_not_in_rna_dataset))), obs=rna_dataset.obs, var=pd.DataFrame(index=genes_not_in_rna_dataset))],\
                join='outer', merge='same', axis=1)

        ## set name of dataset in obs column
        rna_dataset.obs['dataset'] = dataset_name

        ## set to False the overlapping peaks not in current dataset
        original_genes_mask.loc[genes_not_in_rna_dataset, dataset_name] = False  

        rna_dataset = rna_dataset[:, rna_dataset.var.index.argsort()]
        rna_dataset.obs.rename(columns=dict([cell_group_mapper[r]]), inplace=True)
        rna_datasets[r] = rna_dataset.copy()

    all_gene_names = np.stack([rna_dataset.var_names.values for rna_dataset in rna_datasets])
    check_gene_names_alignment = (np.apply_along_axis(lambda x: len(np.unique(x)), 0, all_gene_names) == 1).all()  # check if gene names match across datasets by counting the number of unique gene names per row, and checking whether this count is 1 for all rows
    print('Gene names match across datasets:', check_gene_names_alignment)

    ## ATAC peaks

    ## create Bedtool from cCREs (highjack cre_bedtool via MDD peaks)
    #cre_bedtool = BedTool(cre_path)

    mdd_atac_path = os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_broad.h5ad')

    mdd_atac = anndata.read_h5ad(mdd_atac_path, backed='r')
    mdd_peak_names = mdd_atac.var_names.str.split('-', expand=True)
    mdd_peak_names_df = pd.DataFrame(mdd_peak_names.to_list(), columns=['chrom','start','end'])
    cre_bedtool = BedTool.from_dataframe(mdd_peak_names_df)

    ## create Bedtools from ATAC datasets. NB, not all peak names are formatted as chrom-start-end
    atac_datasets_bedtools = [\
        BedTool.from_dataframe(pd.DataFrame(atac.var.index.str.replace(':','-').str.split('-').to_list(), columns=['chrom','start','end'])).sort()\
                for atac in atac_datasets]

    ## check cCREs that overlap with ATAC datasets
    overlapping_cres = [cre_bedtool.intersect(bed, u=True) for bed in atac_datasets_bedtools]
    overlapping_atac_peaks = reduce(lambda x, y: x.cat(y, postmerge=True), overlapping_cres)  # check postmerge
    overlapping_atac_peaks_names = [f"{interval.chrom}:{interval.start}-{interval.end}" for interval in overlapping_atac_peaks]

    ## intialize mask matrix that indicates whether an overlapping peak is present in the original dataset or not
    original_peaks_mask = pd.DataFrame(np.ones((len(overlapping_atac_peaks_names), len(dataset_names)), dtype=bool), index=overlapping_atac_peaks_names, columns=dataset_names)

    ## loop through datasets
    for a, (atac_bed, atac_dataset, dataset_name) in enumerate(zip(atac_datasets_bedtools, atac_datasets, dataset_names)):

        if len(atac_dataset.var_names.str.split('-')[0]) == 3:
            atac_dataset.var_names = atac_dataset.var_names.str.replace('-', ':', n=1)

        ## check which peaks from current dataset overlaps with retained cCREs
        overlap = overlapping_atac_peaks.intersect(atac_bed, wa=True, wb=True)
        overlap = overlap.to_dataframe(header=None)

        ## create mapper
        source_peak_names = overlap.iloc[:,-3:].astype(str).apply(lambda x: f'{x.iloc[0]}:{x.iloc[1]}-{x.iloc[2]}', axis=1) # overlap.iloc[:,-3:].astype(str).apply('-'.join, axis=1)
        target_peak_names = overlap.iloc[:,:3].astype(str).apply(lambda x: f'{x.iloc[0]}:{x.iloc[1]}-{x.iloc[2]}', axis=1) # overlap.iloc[:,:3].astype(str).apply('-'.join, axis=1)
        target_peak_names_bed = BedTool.from_dataframe(overlap.iloc[:,:3])

        aggregated_peaks_names_mapper = defaultdict(list)
        for key, value in zip(source_peak_names, target_peak_names): aggregated_peaks_names_mapper[key].append(value)
        aggregated_peaks_names_mapper = dict(aggregated_peaks_names_mapper)

        ## apply mapper, while taking care of duplicates
        expanded_rows = []
        retained_peaks_idxs = []

        for p, (peak, row) in enumerate(atac_dataset.var.iterrows()):
            if peak in aggregated_peaks_names_mapper:
                for target_peak in aggregated_peaks_names_mapper[peak]: ## account for duplicates
                    new_row = row.copy()
                    new_row.name = target_peak

                    expanded_rows.append(new_row)
                    retained_peaks_idxs.append(p)

        ## update dataset
        expanded_atac_dataset = pd.DataFrame(expanded_rows); 
        retained_peaks_idxs = np.asarray(retained_peaks_idxs)

        atac_dataset = atac_dataset[:, retained_peaks_idxs].copy()
        atac_dataset.var = expanded_atac_dataset

        ## remove duplicates, although would be better to combine values from duplicates
        duplicates = atac_dataset.var_names.duplicated(keep='first')
        atac_dataset = atac_dataset[:, ~duplicates].copy()

        ## add cCREs not in current dataset
        peaks_not_in_atac_dataset = overlapping_atac_peaks.intersect(target_peak_names_bed, v=True) #peaks_not_in_atac_dataset = list( set(overlapping_atac_peaks) - set(expanded_atac_dataset.drop_duplicates().index) )
        peaks_not_in_atac_dataset = pd.unique([f"{interval.chrom}:{interval.start}-{interval.end}" for interval in peaks_not_in_atac_dataset])

        if hasattr(atac_dataset, 'varm'): # varm interferes with anndata concat
            del atac_dataset.varm

        atac_dataset = anndata.concat([atac_dataset,\
            anndata.AnnData(X=np.zeros((atac_dataset.n_obs, len(peaks_not_in_atac_dataset))), obs=atac_dataset.obs, var=pd.DataFrame(index=peaks_not_in_atac_dataset))],\
                join='outer', merge='same', axis=1)

        ## set to False the overlapping peaks not in current dataset
        original_peaks_mask.loc[peaks_not_in_atac_dataset, dataset_name] = False

        ## set name of dataset in obs column
        atac_dataset.obs['dataset'] = dataset_name
        
        ## SHOULD AVOID: reformat to chrom-start-end for harmonization with MDD data
        #atac_dataset.var.index = atac_dataset.var.index.str.replace(':','-')

        ## sort cCREs by name/interval
        atac_dataset = atac_dataset[:, atac_dataset.var_names.argsort()].copy()
        atac_dataset.obs.rename(columns=dict([cell_group_mapper[a]]), inplace=True)
        atac_datasets[a] = atac_dataset

    ## check alignment of cCREs
    all_peak_names = np.stack([atac_dataset.var_names.values for atac_dataset in atac_datasets])
    check_peak_names_alignment = (np.apply_along_axis(lambda x: len(np.unique(x)), 0, all_peak_names) == 1).all()  # check if peak names match across datasets by counting the number of unique peak names per row, and checking whether this count is 1 for all rows
    print('Peak names match across datasets:', check_peak_names_alignment)
    
    ## concatenate
    rna = anndata.concat(rna_datasets, axis=0)
    atac = anndata.concat(atac_datasets, axis=0)

    return rna, atac, original_genes_mask, original_peaks_mask

def merged_dataset_setup(args, cell_group='Cell type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', dataset=None):

    dataset = dataset.replace('merged_','').replace('imputed_','')

    rna_datapath = atac_datapath = os.path.join(os.environ['DATAPATH'], f'merged_data/{dataset}')


    if args.genes_by_peaks_str is not None:

        RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_mdd.h5ad"
        ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_mdd.h5ad"

        atac = anndata.read_h5ad( os.path.join(atac_datapath, ATAC_file))
        rna  = anndata.read_h5ad( os.path.join(rna_datapath, RNA_file) )
        
        genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_mdd.npz")
        genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

    elif args.genes_by_peaks_str is None:
        raise NotImplementedError()
        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath


def mdd_setup(
        args,
        cell_groups=dict({'atac':'ClustersMapped','rna':'Broad'}),
        batch_groups=dict({'atac':'BrainID','rna':'OriginalSub'}),
        dev_group_key="Age",
        hvg_only=True,
        protein_coding_only=True,
        do_gas=False,
        do_pseudobulk=False,
        return_type='loaders',
        overlapping_subjects_only=False,
        return_raw_data=False,
        dataset='MDD',
        keep_group=None,
        return_backed=False):

    rna_datapath = atac_datapath = os.path.join(os.environ['DATAPATH'], 'mdd_data')

    if args.genes_by_peaks_str is not None:

        ## manually re-introducing the 'merge' string, but should be more self-consistent across filenames and directory names
        if overlapping_subjects_only:
            RNA_file = f"rna_overlap_{args.genes_by_peaks_str}_aligned_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_overlap_{args.genes_by_peaks_str}_aligned_{args.source_dataset}.h5ad"
        else:
            if args.source_dataset == dataset:
                RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
                ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"

            elif args.target_dataset == dataset:
                RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
                ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"

        rna_fullpath = os.path.join(rna_datapath, RNA_file)
        atac_fullpath = os.path.join(atac_datapath, ATAC_file)

        if return_backed:
            rna = anndata.read_h5ad(rna_fullpath, backed='r')
            atac = anndata.read_h5ad(atac_fullpath, backed='r')
            dev_stages = pd.Categorical(\
                np.unique(rna.obs[dev_group_key].tolist() + atac.obs[dev_group_key].tolist()).astype(str),
                ordered=True)
            return rna, atac, cell_groups, dev_group_key, dev_stages

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## already being loaded by other dataset, datapath of which genes-to-peaks mask is stored
        genes_to_peaks_binary_mask = genes_peaks_dict = None

        cell_group = cell_groups['atac']

    elif args.genes_by_peaks_str is None:
    
        RNA_file = "mdd_rna.h5ad"
        ATAC_file = "mdd_atac_broad.h5ad"

        rna_fullpath = os.path.join(rna_datapath, RNA_file)
        atac_fullpath = os.path.join(atac_datapath, ATAC_file)

        rna  = anndata.read_h5ad( rna_fullpath, backed='r+' )
        atac = anndata.read_h5ad( atac_fullpath, backed='r+' )

        ## rename peaks using more conventional delimiters
        source_peak_name_str_pattern = 'all dashes' if len(atac.var.index[0].split('-'))==3 else 'standard'
        if source_peak_name_str_pattern == 'all dashes':
            atac.var_names = atac.var_names.str.split('-').map(lambda x: f'{x[0]}:{x[1]}-{x[2]}') # format to standard

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        ## !! in other setup scripts, HVG applied here

        ## Remove genes with zero counts
        zero_count_genes = np.asarray(rna.X.sum(0) == 0).squeeze()
        rna = rna[:, ~zero_count_genes]

        if do_pseudobulk:

            rna_ct_indices, rna_ct_factors = rna.obs[cell_groups['rna']].factorize()
            rna_ct_map = pd.get_dummies(rna_ct_indices).astype(float)
            rna_ct_weight = rna_ct_map / rna_ct_map.sum(0)
            rna_pseudobulk = np.matmul( rna_ct_weight.T , rna.X.toarray() )

            atac_ct_indices, atac_ct_factors = atac.obs[cell_groups['atac']].factorize()
            atac_ct_map = pd.get_dummies(atac_ct_indices).astype(float)
            atac_ct_weight = atac_ct_map / atac_ct_map.sum(0)
            atac_pseudobulk = np.matmul( atac_ct_weight.T , atac_gas )

            from ot import solve_sample
            #a = rna_ct_map.mean()
            #b = atac_ct_map.mean()
            res = solve_sample(rna_pseudobulk.values, atac_pseudobulk.values, a=None, b=None, metric='cosine', reg=None)
            Gs = res.plan
            ct_map = np.where(Gs == Gs.max(1, keepdims=True))

            ct_map_dict = dict({*zip(
                np.take(rna_ct_factors.values, ct_map[0]),
                np.take(atac_ct_factors.values, ct_map[1])
            )})

            rna.obs = rna.obs.rename(columns={cell_groups['rna']:cell_groups['atac']})
            rna.obs[cell_groups['atac']] = rna.obs[cell_groups['atac']].map(ct_map_dict)
            cell_group = cell_groups['atac']

        elif not do_pseudobulk:
            ct_map_dict = dict({
                1: 'ExN',
                0: 'InN',
                4: 'Oli',
                2: 'Ast',
                3: 'OPC',
                6: 'End',
                5: 'Mix',
                7: 'Mic'
            })

            rna.obs[cell_groups['rna']] = rna.obs[cell_groups['rna']].map(ct_map_dict)
            rna.obs = rna.obs.rename(columns={cell_groups['rna']:cell_groups['atac']})
            cell_group = cell_groups['atac']

            ## remove 'Mix' RNA cells since not part of ATAC data
            rna = rna[~(rna.obs[cell_group] == 'Mix')]

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath
        
        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=None)
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
        

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])

        #genes_sort_idxs = np.isin(genes_peaks_dict['genes'], list(overlapping_genes))
        #peaks_sort_idxs = np.isin(genes_peaks_dict['peaks'], list(overlapping_peaks))
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , peaks_sort_idxs ]
        #genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        #genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].to_memory()  # bottleneck on limited RAM (e.g. local)
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())
        atac = atac.to_memory()

        ## retain peaks by number of cells with non-zero count
        atac_sum = np.asarray((atac.X > 0).sum(0)).flatten()
        keep_peaks = (atac_sum >= np.percentile(atac_sum, q=20))
        atac = atac[:,keep_peaks]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][keep_peaks]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , keep_peaks ]

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=100000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10)
            genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][atac.var['highly_variable'].astype(bool)]
            genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , atac.var['highly_variable'].astype(bool) ]
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            #sc.pp.filter_genes(rna, min_counts=3)
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=18000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False, max_value=10)
            genes_peaks_dict['genes'] = genes_peaks_dict['genes'][rna.var['highly_variable'].astype(bool)]
            genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ rna.var['highly_variable'].astype(bool) , : ]
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        if do_gas:

            ## Merge gene coordinates to RNA
            results = pd.read_csv(os.path.join('/home/dmannk/projects/def-liyue/dmannk/data', 'ensembl_gene_positions.csv'))
            filt_chroms = [str(i) for i in range(1,23+1)] + ['X','Y']
            results = results[results['Chromosome/scaffold name'].isin(filt_chroms)]
            rna_var_tmp = pd.merge(rna.var, results, left_index=True, right_on='Gene name', how='left').set_index('Gene name')
            rna_var_tmp = rna_var_tmp.groupby('Gene name', sort=False).agg({
                        'Chromosome/scaffold name': 'max',
                        'Gene start (bp)': 'mean',
                        'Gene end (bp)': 'mean'})
            
            rna.var = pd.merge(rna.var, rna_var_tmp, left_index=True, right_index=True, how='left')
            rna = rna[:,~rna.var.isna().values.any(1)]

            rna.var['Gene start (bp)'] = rna.var['Gene start (bp)'].astype(int).astype(str)
            rna.var['Gene end (bp)'] = rna.var['Gene end (bp)'].astype(int).astype(str)
            rna.var['Chromosome/scaffold name'] = 'chr' + rna.var['Chromosome/scaffold name']

            ## Create bedtools objects
            rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)', 'index']])
            atac_bed_df = pd.DataFrame(list(atac.var.index.str.split('-', expand=True).values), columns=['chrom','start','end'])
            atac_bed_df['name'] = atac.var.index
            atac_bed = BedTool.from_dataframe(atac_bed_df)

            ## Find peaks at gene and 2000bp upstream for gene activity scores
            gene_size=1e2
            upstream=2e3
            gas_overlaps = rna_bed.window(atac_bed, l=upstream, r=gene_size, c=False).to_dataframe()
            #gas_overlaps['thickEnd'] = gas_overlaps[['chrom','start','end']].astype(str).apply('-'.join, axis=1)

            i, r = pd.factorize(gas_overlaps['name'])
            j, c = pd.factorize(gas_overlaps['thickEnd'])
            n, m = len(r), len(c)

            gas_binary_mask = np.zeros((n, m), dtype=np.int64)
            np.add.at(gas_binary_mask, (i, j), 1)
            gas_binary_mask_partial = pd.DataFrame(gas_binary_mask, index=r, columns=c)

            gas_binary_mask = pd.DataFrame(np.zeros([rna.n_vars, atac.n_vars]), index=rna.var.index, columns=atac.var.index)
            gas_binary_mask = gas_binary_mask.merge(gas_binary_mask_partial, left_index=True, right_index=True, how='left', suffixes=('_x',None))
            gas_binary_mask = gas_binary_mask.loc[:,~gas_binary_mask.columns.str.endswith('_x')]
            gas_binary_mask = gas_binary_mask.fillna(0)
            
            atac_gas = np.matmul( atac.X.toarray() , gas_binary_mask.T)
            atac_gas.index = atac.obs.index
            return atac_gas, rna

    ## delete atac.raw if it exists
    if 'raw' in atac.uns.keys():
        del atac.raw

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, 'MDD', args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_groups['rna'], valid_only=False)
        atac.obs = atac.obs.rename(columns={'batch':'batch_og'}) # original batch column, different from subjects
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, 'mdd', args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_groups['atac'], valid_only=False)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    elif return_type == 'data':
        return rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath
    
def pfc_zhu_setup(args, cell_group='Cell type', batch_group='Donor ID', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='PFC_Zhu', \
    keep_group=[''], dev_group_key='dev_stage', dev_stages = ['EaFet', 'LaFet', 'Inf', 'Child', 'Adol', 'Adult']):

    datapath = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu')
    rna_datapath = os.path.join(datapath, 'rna')
    atac_datapath = os.path.join(datapath, 'atac')

    EN_cell_type_branches = {
        "RG": "0",
        "IPC": "1",
        "EN-fetal-early": "2",
        "EN-fetal-late": "3",
        "EN-postnatal": "4",
    }

    if args.genes_by_peaks_str is not None:

        if (args.source_dataset == dataset) and (args.target_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif (args.target_dataset == dataset) and (args.source_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        elif (args.source_dataset == dataset) and (args.target_dataset is None):
            RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        ## load data for a single developmental stage, should already be processed
        if (len(keep_group) == 1) and (keep_group != ['']):
            dev_stage = keep_group[0]
            RNA_file = f"{dev_stage}_{RNA_file}"
            ATAC_file = f"{dev_stage}_{ATAC_file}"

        atac_fullpath = os.path.join(atac_datapath, ATAC_file)
        rna_fullpath = os.path.join(rna_datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## retain from specific developmental stages
        keep_atac_subj = atac.obs['Donor ID'].str.contains('|'.join(keep_group), regex=True) # if keep_group=[''], then keeps all subjects
        keep_rna_subj = rna.obs['Donor ID'].str.contains('|'.join(keep_group), regex=True)

        fetal_present = pd.Series(keep_group).str.contains('Fet').any()
        if (not fetal_present) and (keep_group != ['']):
            keep_atac_ct = ~atac.obs[cell_group].str.contains('fetal')
            keep_rna_ct = ~rna.obs[cell_group].str.contains('fetal')

            keep_atac = keep_atac_subj & keep_atac_ct
            keep_rna = keep_rna_subj & keep_rna_ct
        else:
            keep_atac = keep_atac_subj
            keep_rna = keep_rna_subj

        assert (keep_atac == keep_rna).all()

        ## TMP - keep only ExNeu cells
        #print('!!!! TMP - keeping only cells from EN lineage !!!!')
        #keep_atac = keep_atac & atac.obs[cell_group].isin(EN_cell_type_branches.keys())
        #keep_rna = keep_rna & rna.obs[cell_group].isin(EN_cell_type_branches.keys())

        atac = atac[keep_atac].to_memory()
        rna = rna[keep_rna].to_memory()


    elif args.genes_by_peaks_str is None:

        RNA_file = "PFC_Zhu_rna.h5ad"
        ATAC_file = "PFC_Zhu_atac.h5ad"

        atac_fullpath = os.path.join(atac_datapath, ATAC_file)
        rna_fullpath = os.path.join(rna_datapath, RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## import cell-type labels and assign to data
        atac_barcodes = pd.read_csv( os.path.join(atac_datapath, 'atac_barcodes.tsv.gz'), compression='gzip', sep='\t', index_col=0)
        rna_barcodes  = pd.read_csv( os.path.join(rna_datapath, 'rna_barcodes.tsv.gz'), compression='gzip', sep='\t', index_col=0)

        atac_barcodes_data = np.stack(atac.obs.index.str.split('_', expand=True))[:,1]
        rna_barcodes_data = np.stack(rna.obs.index.str.split('_', expand=True))[:,1]
        assert (rna_barcodes_data == rna_barcodes.index).all() and (atac_barcodes_data == rna_barcodes.index).all()# and (atac_barcodes_data == atac_barcodes.index).all()

        rna.obs = rna_barcodes.reset_index().set_index(rna.obs.index)
        atac.obs = atac_barcodes.reset_index().set_index(atac.obs.index)

        ## rename ATAC peak names to follow chrom:start-end format (rather than chrom-start-end)
        atac.var.index = pd.DataFrame(atac.var.index.str.split('-', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=100000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        # min-max scaling
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(datapath, 'atac', f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

            ## ensure chrom:start-end format
            genes_peaks_dict['peaks'] = pd.DataFrame(genes_peaks_dict['peaks'].str.split('[-:]', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

        ## create dev_stage labels
        rna.obs['dev_stage'] = rna.obs['Donor ID'].apply(lambda x: x[:-1])
        atac.obs['dev_stage'] = atac.obs['Donor ID'].apply(lambda x: x[:-1])

    ## Count number of cells per dev stage
    dev_stage_counts_df = pd.merge(
        rna.obs[dev_group_key].value_counts().to_frame(), atac.obs[dev_group_key].value_counts().to_frame(),
        left_index=True, right_index=True, how='outer')
    dev_stage_counts_df.columns = ['rna_n_cells', 'atac_n_cells']
    dev_stage_counts_df.index = pd.Categorical(dev_stage_counts_df.index, categories=dev_stages, ordered=True)
    dev_stage_counts_df = dev_stage_counts_df.sort_index()

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    ## ensure that developmental stages are in the correct order
    atac.obs[cell_group] = pd.Categorical(atac.obs[dev_group_key], categories=dev_stages, ordered=True)
    rna.obs[cell_group] = pd.Categorical(rna.obs[dev_group_key], categories=dev_stages, ordered=True)

    ## Set split type and key
    split_key = 'cell_type'
    split_type = 'stratified'

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'loaders_with_dev_stages':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return dev_stage_counts_df, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath


def cortex_velmeshev_setup(args, cell_group='Lineage', batch_group='subject', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, return_backed=False, dataset='Cortex_Velmeshev',\
    keep_group=[''], dev_group_key='Age_Range', dev_stages=['2nd trimester', '3rd trimester', '0-1 years', '1-2 years', '2-4 years', '4-10 years', '10-20 years', 'Adult']):
    
    datapath = os.path.join(os.environ['DATAPATH'], 'Cortex_Velmeshev')
    atac_datapath = os.path.join(datapath, 'atac')
    rna_datapath = os.path.join(datapath, 'rna')

    adult_celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')
    dev_celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Developing_Human_Brain.pkl')

    pfc_zhu_marker_genes = {
        #"NPC": ["PAX6"],
        "NPC - Radial glia (RG)": ["PAX6", "VIM"],  # "HES5" # NPC subtype
        "NPC - Intermediate progenitors (IPCs)": ["PAX6", "EOMES"],  # NPC subtype

        #"Excitatory neurons (EN)": ["SATB2", "SLC17A7", "NEUROD2"],
        "EN-fetal-early": ["SATB2", "SLC17A7", "NEUROD2"],  # enriched in early fetal
        "EN-fetal-late": ["SATB2", "SLC17A7", "NEUROD2"],  # enriched in late fetal
        "EN-postnatal": ["SATB2", "SLC17A7", "NEUROD2"],   # enriched in postnatal

        "Inhibitory neurons (IN)": ["GAD1", "GAD2"],
        "IN-MGE": ["GAD1", "GAD2", "LHX6"],       # MGE-derived
        "IN-CGE": ["GAD1", "GAD2", "VIP", "ADARB2"],  # CGE-derived
        "IN-fetal": ["GAD1", "GAD2"],             # enriched in fetal samples

        "OPC": ["OLIG1", "SOX10"],  # Oligodendrocyte progenitor cells
        "Astrocytes": ["AQP4", "GFAP"],
        "Oligodendrocytes": ["MOBP", "OPALIN"],
        "Microglia": ["CX3CR1"], # "PTPRC"
        "Endothelial": ["CLDN5"],
        "Pericytes": ["PDGFRB"],
        "VSMCs": ["COL1A2"],  # Vascular smooth muscle cells
    }

    if args.genes_by_peaks_str is not None:

        if (args.source_dataset == dataset) and (args.target_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"
            
            genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif (args.target_dataset == dataset) and (args.source_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        elif (args.source_dataset == dataset) and (args.target_dataset is None):
            RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        ## load data for a single developmental stage, should already be processed
        if (len(keep_group) == 1) and (keep_group != ['']):
            dev_stage = keep_group[0]
            RNA_file = f"{dev_stage}_{RNA_file}"
            ATAC_file = f"{dev_stage}_{ATAC_file}"

        atac_fullpath = os.path.join(atac_datapath, ATAC_file)
        rna_fullpath = os.path.join(rna_datapath, RNA_file)

        if return_backed:
            rna = anndata.read_h5ad(rna_fullpath, backed='r')
            atac = anndata.read_h5ad(atac_fullpath, backed='r')
            return rna, atac, cell_group, dev_group_key, dev_stages

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        #rna.obs['Seurat_cell_type'] = rna.obs[['cell_type', 'Seurat_clusters']].apply(lambda x: f'{x[0]} - {x[1]}', axis=1)

        ## create dev_stage labels by copying from dev_group_key
        rna.obs['dev_stage'] = rna.obs[dev_group_key].copy()
        atac.obs['dev_stage'] = atac.obs[dev_group_key].copy()

        ## retain from specific developmental stages
        keep_atac_subj = atac.obs['dev_stage'].str.contains('|'.join(keep_group), regex=True) # if keep_group=[''], then keeps all subjects
        keep_rna_subj = rna.obs['dev_stage'].str.contains('|'.join(keep_group), regex=True)

        ## remove fetal cells if not part of keep_group
        fetal_present = pd.Series(keep_group).str.contains('trimester').any()
        if (not fetal_present) and (keep_group != ['']):
            fetal_cell_types = ["pLaCeHoLdEr"]

            keep_atac_ct = ~atac.obs[cell_group].isin(fetal_cell_types)
            keep_rna_ct = ~rna.obs[cell_group].isin(fetal_cell_types)

            keep_atac = keep_atac_subj & keep_atac_ct
            keep_rna = keep_rna_subj & keep_rna_ct
        else:
            keep_atac = keep_atac_subj
            keep_rna = keep_rna_subj


        ## TMP - keep only ExNeu cells
        #print('!!!! TMP - keeping only ExNeu and IN cells !!!!')
        #keep_atac = keep_atac & atac.obs['Lineage'].str.contains('ExNeu|IN')
        #keep_rna = keep_rna & rna.obs['Lineage'].str.contains('ExNeu|IN')

        atac = atac[keep_atac].to_memory()
        rna = rna[keep_rna].to_memory()

    elif args.genes_by_peaks_str is None:

        #rna = anndata.read_h5ad(os.path.join(rna_datapath, "velmeshev_snRNA_seq.h5ad"))
        #rna_meta = pd.read_csv(os.path.join(rna_datapath, "meta.tsv"), sep='\t')
        #rna.obs = rna_meta.merge(rna.obs, left_on='Cell_ID', right_index=True).set_index('Cell_ID')

        atac = anndata.read_h5ad(os.path.join(atac_datapath, "atac_unprocessed.h5ad"))
        rna = anndata.read_h5ad(os.path.join(rna_datapath, "rna.h5ad"))

        ## add generlal meta data
        rna_meta = pd.read_csv(os.path.join(rna_datapath, "meta.tsv"), sep='\t')
        rna.obs = rna_meta.merge(rna.obs, left_on='Cell_ID', right_index=True).set_index('Cell_ID')
        rna.var['protein_coding'] = (rna.var['feature_type'] == 'protein_coding')
        rna.var = rna.var.reset_index().set_index('feature_name')
        rna.obs['Individual'] = rna.obs['Individual'].astype(str)

        ## concatenate cell type meta data
        ct_meta_dfs = []
        ct_meta_files = glob(os.path.join(datapath, '*', "meta", "*_meta.tsv"))
        for ct_meta_file in ct_meta_files:
            ct_meta = pd.read_csv(ct_meta_file, sep='\t')
            if any(col.lower() == "cell_type" for col in ct_meta.columns):
                cell_id_col = ct_meta.columns[0]
                ct_meta.set_index(cell_id_col, inplace=True)
                ct_meta_dfs.append(ct_meta)
            else:
                print(f'{os.path.splitext(os.path.basename(ct_meta_file))[0]} does not contain cell_type column')

        ## merge cell type meta data
        ct_meta_df = pd.concat(ct_meta_dfs, axis=0)
        ct_meta_df.rename(columns={'Cell_Type':'sub_cell_type', 'Pseudotime':'velmeshev_pseudotime'}, inplace=True)
        ct_meta_df = ct_meta_df[['sub_cell_type','velmeshev_pseudotime']]

        ## merge cell type meta data
        rna.obs = rna.obs.merge(ct_meta_df, left_index=True, right_index=True, how='left')
        atac.obs = atac.obs.merge(ct_meta_df, left_index=True, right_index=True, how='left')

        ## fill NaNs in 'sub_cell_type' with 'Lineage'
        rna.obs['sub_cell_type'] = rna.obs['sub_cell_type'].fillna(rna.obs['Lineage'])

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = rna[:, rna.var['feature_type'] == 'protein_coding'].copy()

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## create counts layer
        atac.layers['counts'] = atac.X.copy()

        rna_counts = rna.raw[:,rna.raw.var_names.isin(rna.var['index'])].X.copy()
        rna.layers['counts'] = rna_counts
        del rna.raw # raw replaced by counts layer

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        ac.tl.lsi(atac)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=100000, subset=False, batch_key=None) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000, subset=False, batch_key=None) # sc.pl.highly_variable_genes(rna) # batch_key=None to avoid error
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        ''' not needed now that we have cell type meta data
        # min-max scaling
        adult_predictions = celltypist.annotate(rna, majority_voting=True, model=adult_celltypist_model_path, over_clustering='Seurat_clusters')
        dev_predictions = celltypist.annotate(rna, majority_voting=True, model=dev_celltypist_model_path, over_clustering='Seurat_clusters')
        #celltypist.dotplot(adult_predictions, use_as_reference = 'Lineage', use_as_prediction = 'majority_voting')

        is_fetal = rna.obs[dev_group_key].str.contains('trimester').astype(bool)
        rna.obs['Seurat_clusters_celltypist'] = None
        rna.obs['celltypist'] = None
        rna.obs.loc[~is_fetal, 'Seurat_clusters_celltypist'] = adult_predictions.predicted_labels['majority_voting'][~is_fetal]
        rna.obs.loc[is_fetal, 'Seurat_clusters_celltypist'] = dev_predictions.predicted_labels['majority_voting'][is_fetal]
        rna.obs.loc[~is_fetal, 'celltypist'] = adult_predictions.predicted_labels['predicted_labels'][~is_fetal]
        rna.obs.loc[is_fetal, 'celltypist'] = dev_predictions.predicted_labels['predicted_labels'][is_fetal]
        '''

        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(datapath, 'atac', f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

            ## ensure chrom:start-end format
            genes_peaks_dict['peaks'] = pd.DataFrame(genes_peaks_dict['peaks'].str.split('[-:]', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

    ## Count number of cells per dev stage
    dev_stage_counts_df = pd.merge(
        rna.obs[dev_group_key].value_counts().to_frame(), atac.obs[dev_group_key].value_counts().to_frame(),
        left_index=True, right_index=True, how='outer')
    dev_stage_counts_df.columns = ['rna_n_cells', 'atac_n_cells']
    dev_stage_counts_df.index = pd.Categorical(dev_stage_counts_df.index, categories=dev_stages, ordered=True)
    dev_stage_counts_df = dev_stage_counts_df.sort_index()
        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    ## ensure that developmental stages are in the correct order
    atac.obs[dev_group_key] = pd.Categorical(atac.obs[dev_group_key], categories=dev_stages, ordered=True)
    rna.obs[dev_group_key] = pd.Categorical(rna.obs[dev_group_key], categories=dev_stages, ordered=True)

    ## define variable used to split data into train and valid sets
    split_key = 'cell_type'
    split_type = 'stratified'

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'loaders_with_dev_stages':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return dev_stage_counts_df, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath



def pfc_v1_wang_setup(args, cell_group='type', batch_group='subject', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='PFC_V1_Wang',\
    keep_group=[''], dev_group_key='Group', dev_stages=['FirstTrim', 'SecTrim', 'ThirdTrim', 'Inf', 'Adol']):

    datapath = os.path.join(os.environ['DATAPATH'], 'PFC_V1_Wang')
    atac_datapath = os.path.join(datapath, 'atac')
    rna_datapath = os.path.join(datapath, 'rna')

    EN_cell_type_branches = { # missing in data: "EN-L4-IT-V1"
        "RG-vRG": "0",
        "IPC-EN": "1",
        "EN-newborn": "2",
        "EN-IT-immature": "3.IT",
        "EN-non-IT-immature": "3.non-IT",
        "EN-L2_3-IT": "4.IT.BP3",
        "EN-L4-IT-V1": "4.IT.BP3",
        "EN-L4-IT": "4.IT.BP4",
        "EN-L5-IT": "4.IT.BP4",
        "EN-L6-IT": "4.IT.BP2",
        "EN-L5_6-NP": "4.non-IT.BP5",
        "EN-L5-ET": "4.non-IT.BP5",
        "EN-L6-CT": "4.non-IT.BP5",
        "EN-L6b": "4.non-IT.BP5"
    }


    if args.genes_by_peaks_str is not None:

        if (args.source_dataset == dataset) and (args.target_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif (args.target_dataset == dataset) and (args.source_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        elif (args.source_dataset == dataset) and (args.target_dataset is None):
            RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        ## load data for a single developmental stage, should already be processed
        if (len(keep_group) == 1) and (keep_group != ['']):
            dev_stage = keep_group[0]
            RNA_file = f"{dev_stage}_{RNA_file}"
            ATAC_file = f"{dev_stage}_{ATAC_file}"

        atac_fullpath = os.path.join(atac_datapath, ATAC_file)
        rna_fullpath = os.path.join(rna_datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## retain from specific developmental stages
        keep_atac_subj = atac.obs['dev_stage'].str.contains('|'.join(keep_group), regex=True) # if keep_group=[''], then keeps all subjects
        keep_rna_subj = rna.obs['dev_stage'].str.contains('|'.join(keep_group), regex=True)

        ## remove fetal cells if not part of keep_group
        fetal_present = pd.Series(keep_group).str.contains('Trim').any() # 'Trim' for 'Trimester' during gestation
        if (not fetal_present) and (keep_group != ['']):
            fetal_cell_types = ["RG-vRG", "RG-tRG", "RG-oRG", "IPC-EN", "IPC-glia", "EN-newborn", "EN-IT-immature", "EN-non-IT-immature", "Cajal–Retzius cell"]

            keep_atac_ct = ~atac.obs[cell_group].isin(fetal_cell_types)
            keep_rna_ct = ~rna.obs[cell_group].isin(fetal_cell_types)

            keep_atac = keep_atac_subj & keep_atac_ct
            keep_rna = keep_rna_subj & keep_rna_ct
        else:
            keep_atac = keep_atac_subj
            keep_rna = keep_rna_subj

        assert (keep_atac == keep_rna).all()

        ## TMP - keep only ExNeu cells
        #print('!!!! TMP - keeping only cells from EN lineage !!!!')
        #keep_atac = keep_atac & atac.obs[cell_group].isin(EN_cell_type_branches.keys())
        #keep_rna = keep_rna & rna.obs[cell_group].isin(EN_cell_type_branches.keys())

        atac = atac[keep_atac].to_memory()
        rna = rna[keep_rna].to_memory()

    elif args.genes_by_peaks_str is None:

        rna_file = "snMultiome_atlas_RNA.h5ad"
        atac_file = "snMultiome_atlas_ATAC.h5ad"

        rna_fullpath = os.path.join(rna_datapath, rna_file)
        atac_fullpath = os.path.join(atac_datapath, atac_file)

        rna = anndata.read_h5ad(rna_fullpath)
        atac = anndata.read_h5ad(atac_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=100000)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        # min-max scaling
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(datapath, 'atac', f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

        ## ensure chrom:start-end format
        genes_peaks_dict['peaks'] = pd.DataFrame(genes_peaks_dict['peaks'].str.split('[-:]', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values
        atac.var.index = genes_peaks_dict['peaks']

        ## map cell types to cell_group if not already done, according to Fig 1B from Wang et al.
        if pd.api.types.is_integer_dtype(rna.obs[cell_group]) or pd.api.types.is_integer_dtype(atac.obs[cell_group]):
            cell_type_map = {0: "RG-vRG", 1: "RG-tRG", 2: "RG-oRG", 3: "IPC-EN", 4: "EN-newborn", 5: "EN-IT-immature", 6: "EN-L2_3-IT", 7: "EN-L4-IT", 8: "EN-L5-IT", 9: "EN-L6-IT", 10: "EN-non-IT-immature", 11: "EN-L5-ET", 12: "EN-L5_6-NP", 13: "EN-L6-CT", 14: "EN-L6b", 15: "IN-dLGE-immature", 16: "IN-CGE-immature", 17: "IN-CGE-VIP", 18: "IN-CGE-SNCG", 19: "IN-mix-LAMP5", 20: "IN-MGE-immature", 21: "IN-MGE-SST", 22: "IN-MGE-PV", 23: "IPC-glia", 24: "Astrocyte-immature", 25: "Astrocyte-protoplasmic", 26: "Astrocyte-fibrous", 27: "OPC", 28: "Oligodendrocyte-immature", 29: "Oligodendrocyte", 30: "Cajal–Retzius cell", 31: "Microglia", 32: "Vascular", 33: "Unknown"}
            atac.obs[cell_group] = atac.obs[cell_group].map(cell_type_map)
            rna.obs[cell_group] = rna.obs[cell_group].map(cell_type_map)

        ## create dev_stage labels by mapping dev_group_key to dev_stage
        dev_stage_mapper = {0: 'FirstTrim', 1: 'SecTrim', 2:'ThirdTrim', 3:'Inf', 4:'Adol'}
        rna.obs['dev_stage'] = rna.obs[dev_group_key].map(dev_stage_mapper)
        atac.obs['dev_stage'] = atac.obs[dev_group_key].map(dev_stage_mapper)

    ## Count number of cells per dev stage
    dev_stage_counts_df = pd.merge(
        rna.obs['dev_stage'].value_counts().to_frame(), atac.obs['dev_stage'].value_counts().to_frame(),
        left_index=True, right_index=True, how='outer')
    dev_stage_counts_df.columns = ['rna_n_cells', 'atac_n_cells']
    dev_stage_counts_df.index = pd.Categorical(dev_stage_counts_df.index, categories=dev_stages, ordered=True)
    dev_stage_counts_df = dev_stage_counts_df.sort_index()

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    ## ensure that developmental stages are in the correct order
    atac.obs['dev_stage'] = pd.Categorical(atac.obs['dev_stage'], categories=dev_stages, ordered=True)
    rna.obs['dev_stage'] = pd.Categorical(rna.obs['dev_stage'], categories=dev_stages, ordered=True)

    ## Set split type and key
    split_key = 'cell_type'
    split_type = 'stratified'

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'loaders_with_dev_stages':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, split_type=split_type, split_key=split_key, cell_group_key=cell_group, batch_key=batch_group)
        return dev_stage_counts_df, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath
    


def dlpfc_anderson_setup(args, cell_group='predicted.id', batch_group='id', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='DLPFC_Anderson'):
        
    atac_datapath = rna_datapath = datapath = os.path.join(os.environ['DATAPATH'], 'DLPFC_Anderson', 'snMultiome')

    if args.genes_by_peaks_str is not None:

        if (args.source_dataset == dataset) and (args.target_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif (args.target_dataset == dataset) and (args.source_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        elif (args.source_dataset == dataset) and (args.target_dataset is None):
            RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)


    elif args.genes_by_peaks_str is None:

        RNA_file = "rna_ctrl.h5ad"
        ATAC_file = "atac_ctrl.h5ad"

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        # min-max scaling
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=None)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

    ## check if batches are non-nan
    assert (atac.obs[batch_group].notna().all() or rna.obs[batch_group].notna().all())
        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath

def midbrain_adams_setup(args, cell_group='cell_type', batch_group='subject', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='Midbrain_Adams'):
    
    atac_datapath = rna_datapath = datapath = os.path.join(os.environ['DATAPATH'], 'Midbrain_Adams')
    celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
        
    
    elif args.genes_by_peaks_str is None:

        RNA_file = "rna_ctrl.h5ad"
        ATAC_file = "atac_ctrl.h5ad"

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        predictions = celltypist.annotate(rna, majority_voting=False, model=celltypist_model_path)
        sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
        sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling

        ## Use celltypist to annotate cells by cell types
        #sc.pp.scale(atac, zero_center=False, max_value=10)
        #sc.pp.scale(rna, zero_center=False, max_value=10)

        ## Keep top N cell types
        num_celltypes = 30; print(f'Keeping top {num_celltypes} cell types')
        value_counts_threshold = predictions.predicted_labels.value_counts().sort_values(ascending=False).iloc[num_celltypes]
        keep_celltypes = (predictions.predicted_labels.value_counts() > value_counts_threshold)
        keep_celltypes = keep_celltypes[keep_celltypes].index.get_level_values(0).values.to_list()
        keep_celltypes_idx = np.array(predictions.probability_matrix.columns.isin(keep_celltypes))

        ## Assign cell types to data based on top probabilities according to retained cell-types
        final_probability_matrix = predictions.probability_matrix.loc[:, keep_celltypes_idx]
        final_predicted_labels = final_probability_matrix.columns[final_probability_matrix.values.argmax(1)]

        rna.obs['cell_type'] = final_predicted_labels
        atac.obs['cell_type'] = final_predicted_labels

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=None)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath

def dlpfc_ma_setup(args, cell_group='subclass', batch_group='samplename', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='DLPFC_Ma'):
        
    atac_datapath = rna_datapath = datapath = os.path.join(os.environ['DATAPATH'], 'DLPFC_Ma')

    if args.genes_by_peaks_str is not None:

        if (args.source_dataset == dataset) and (args.target_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif (args.target_dataset == dataset) and (args.source_dataset is not None):
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        elif (args.source_dataset == dataset) and (args.target_dataset is None):
            RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
    
    elif args.genes_by_peaks_str is None:

        RNA_file = "GSE207334_Multiome_rna_counts.mtx.gz"
        ATAC_file = "GSE207334_Multiome_atac_counts.mtx.gz"

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)
    
        ## Load data
        atac = sc.read_mtx(atac_fullpath).T
        rna  = sc.read_mtx(rna_fullpath).T

        ## Load metadata
        atac_peaks = pd.read_csv(os.path.join(datapath, 'GSE207334_Multiome_atac_peaks.txt.gz'), delimiter=None, header=None)
        rna_genes = pd.read_csv(os.path.join(datapath, 'GSE207334_Multiome_rna_genes.txt.gz'), delimiter=None, header=None)
        cell_metadata = pd.read_csv(os.path.join(datapath, 'GSE207334_Multiome_cell_meta.txt.gz'), delimiter='\t')

        ## Assign var and obs variables to ATAC and RNA data
        atac_peaks = atac_peaks[0].str.split('-', expand=True).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
        atac.var.index = atac_peaks
        atac.obs = cell_metadata

        rna.var.index = rna_genes[0].values
        rna.obs = cell_metadata

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        # min-max scaling
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=None)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath


def get_genes_by_peaks_str(datasets = ["PFC_Zhu", "DLPFC_Anderson", "DLPFC_Ma", "Midbrain_Adams", "mouse_brain_10x", "pbmc_10x"]):
    """
    Get the genes_by_peaks_str for the given source and target datasets

    import re, os
    from glob import glob
    from datetime import datetime
    import pandas as pd
    """

    datapath = os.environ['DATAPATH']

    ## initialize dataframes
    genes_by_peaks_str_df = pd.DataFrame(index = datasets, columns = datasets)
    genes_by_peaks_str_df.index.name = 'source'
    genes_by_peaks_str_df.columns.name = 'target'

    genes_by_peaks_str_timestamp_df = pd.DataFrame(index = datasets, columns = datasets)
    genes_by_peaks_str_timestamp_df.index.name = 'source'
    genes_by_peaks_str_timestamp_df.columns.name = 'target'

    ## loop through source datasets
    for source in datasets:
        datapath_full = os.path.join(datapath, source)
        genes_by_peaks_str_paths = glob(os.path.join(datapath_full, 'genes_to_peaks_binary_mask_*_by_*_aligned_target_*.pkl'))

        if len(genes_by_peaks_str_paths) == 0:  # check one subdirectory deeper
            genes_by_peaks_str_paths = glob(os.path.join(datapath_full, '*', 'genes_to_peaks_binary_mask_*_by_*_aligned_target_*.pkl'))

        ## loop through target datasets to find genes_by_peaks_str
        for genes_by_peaks_str_path in genes_by_peaks_str_paths:
            target = re.search(r'aligned_target_(\w+)', genes_by_peaks_str_path).group(1)
            genes_by_peaks_str = re.search(r'\d+_by_\d+', genes_by_peaks_str_path).group(0)
            genes_by_peaks_str_df.loc[source, target] = genes_by_peaks_str

            timestamp = os.path.getmtime(genes_by_peaks_str_path)
            genes_by_peaks_str_timestamp_df.loc[source, target] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

    ## save dataframes
    genes_by_peaks_str_df.to_csv(os.path.join(datapath, 'genes_by_peaks_str.csv'))
    genes_by_peaks_str_timestamp_df.to_csv(os.path.join(datapath, 'genes_by_peaks_str_timestamp.csv'))


def teachers_setup(model_paths, args, device, return_type='loaders', dataset_idx_dict=None):
    import mlflow.pytorch
    from mlflow.models import Model
    
    datasets = []
    models = {}

    if return_type == 'loaders':
        teacher_rna_train_loaders = {}
        teacher_atac_train_loaders = {}
        teacher_rna_valid_loaders = {}
        teacher_atac_valid_loaders = {}
    elif return_type == 'data':
        teacher_rna_adatas = {}
        teacher_atac_adatas = {}
    
    for m, model_path in enumerate(model_paths):

        print(model_path)

        ## Load the model
        with open(model_path, 'r') as f:
            model_uris = f.read().strip().splitlines()
            model_uri = model_uris[0]
        
        ## Get metadata and determine the dataset
        model_metadata = Model.load(model_uri)
        #dataset = model_metadata.metadata['source_dataset']
        dataset = model_path.split('/')[-3] # TEMPORARY - get dataset from model path
        genes_by_peaks_str = model_metadata.metadata['genes_by_peaks_str']
        teacher_setup_func = return_setup_func_from_dataset(args.target_dataset)

        ## Check if dataset contains 'multiome', in which case convert to '10x'
        if 'multiome' in dataset:
            print(f"Warning: Dataset {dataset} contains 'multiome', converting to '10x'")
            dataset = dataset.replace('multiome', '10x')

        ## Check if dataset is in args.ignore_sources
        if dataset in args.ignore_sources:
            print(f"Skipping dataset {dataset} because it is in args.ignore_sources")
            continue

        ## Load the model
        model = mlflow.pytorch.load_model(model_uri, device=device)

        print(dataset)
        datasets.append(dataset)

        models[dataset] = model

        if args.source_dataset_embedder:
            dataset_idx_dict[dataset] = m

        #if args.ordinal_job_id is None:
        ## TMP - create deepcopy of args
        args_tmp = deepcopy(args)
        args_tmp.genes_by_peaks_str = genes_by_peaks_str

        ## if disagreement between model and args, probably involves individual dev stages
        are_dev_teachers = (dataset != model_metadata.metadata['source_dataset'])
        if not are_dev_teachers:
            args_tmp.source_dataset = dataset
        elif are_dev_teachers:
            print("Teachers are likely dev stages")
            args_tmp.source_dataset = dataset.split(model_metadata.metadata['source_dataset']+'_')[-1] # remove dev stage from dataset name to isolate dataset name
        
        overlapping_subjects_only = False #True if args.dataset == 'roussos' else False

        if (not are_dev_teachers) or (m==0):

            if return_type == 'loaders':

                teacher_rna_train_loader, teacher_atac_train_loader, _, _, _, teacher_rna_valid_loader, teacher_atac_valid_loader, _, _, _, _, _, _, _, _ =\
                    teacher_setup_func(args_tmp, return_type=return_type)

                teacher_rna_train_loaders[dataset] = teacher_rna_train_loader
                teacher_atac_train_loaders[dataset] = teacher_atac_train_loader
                teacher_rna_valid_loaders[dataset] = teacher_rna_valid_loader
                teacher_atac_valid_loaders[dataset] = teacher_atac_valid_loader

            elif return_type == 'data':

                teacher_rna_adata, teacher_atac_adata, _, _, _ = \
                    teacher_setup_func(args_tmp, return_type=return_type, return_backed=True)
            
                teacher_rna_adatas[dataset] = teacher_rna_adata
                teacher_atac_adatas[dataset] = teacher_atac_adata

    if are_dev_teachers:
        ## reuse first teacher for all dev stages, since they all the same teacher data is dev teachers
        for dataset in datasets:
            if dataset != model_metadata.metadata['source_dataset']:
                teacher_rna_train_loaders[dataset] = teacher_rna_train_loaders[datasets[0]]
                teacher_atac_train_loaders[dataset] = teacher_atac_train_loaders[datasets[0]]
                teacher_rna_valid_loaders[dataset] = teacher_rna_valid_loaders[datasets[0]]
                teacher_atac_valid_loaders[dataset] = teacher_atac_valid_loaders[datasets[0]]

    if return_type == 'loaders':
        return datasets, models, teacher_rna_train_loaders, teacher_atac_train_loaders, teacher_rna_valid_loaders, teacher_atac_valid_loaders
    elif return_type == 'data':
        return datasets, models, teacher_rna_adatas, teacher_atac_adatas
    else:
        return datasets, models


def spatialLIBD_setup(batch_size, total_epochs, cell_group='Cluster', hvg_only=True, protein_coding_only=True, return_type='loaders', return_raw_data=False, dataset='spatialLIBD'):

    datapath = os.path.join(os.environ['DATAPATH'], 'spatialLIBD')

    ## Load the data
    sp = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'spatialLIBD', 'spatialLIBD.h5ad'))
    coords = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'spatialLIBD', 'spatialCoords.csv'))

    ## Change ENSG to gene name
    sp.var_names = sp.var['gene_name']

    ## Subset to protein-coding genes
    if protein_coding_only:
        sp = get_protein_coding_genes(sp)

    if hvg_only:
        sp = sp[:, sp.var['is_top_hvg'].astype(bool)]

    if return_raw_data and return_type == 'data':
        return sp.to_memory(), cell_group, datapath
    
    if not return_raw_data:
        sc.pp.normalize_total(sp, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(sp)
        sc.pp.scale(sp, zero_center=False,  max_value=10) # min-max scaling
    
    if return_type == 'loaders':
        sp_train_loader, sp_valid_loader, sp_valid_idx, sp_train_num_batches, sp_valid_num_batches, sp_train_n_batches_str_length, sp_valid_n_batches_str_length, sp_train_n_epochs_str_length, sp_valid_n_epochs_str_length = create_loaders(sp, dataset, batch_size, total_epochs, cell_group_key=cell_group)
        return sp_train_loader, sp_train_num_batches, sp_train_n_batches_str_length, sp_train_n_epochs_str_length, sp_valid_loader, sp_valid_num_batches, sp_valid_n_batches_str_length, sp_valid_n_epochs_str_length, sp.n_vars, sp_valid_idx
        
    
    elif return_type == 'data':
        return sp.to_memory(), cell_group, datapath

    return sp, cell_group, datapath

def mouse_brain_10x_setup(args, cell_group='GEX Graph-based', batch_group=None, hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='mouse_brain_10x', chain_file_name=None):

    atac_datapath = rna_datapath = datapath = os.path.join(os.environ['DATAPATH'], 'mouse_brain_10x')
    
    ## Lift data from mm10 to hg38 and save as MuData object. Skip the rest of setup
    if chain_file_name is not None: #'mm10ToHg38.over.chain.gz'
        
        ## Load the original data
        mudata = muon_read_10x_h5(os.path.join(os.environ['DATAPATH'], 'mouse_brain_10x', 'M_Brain_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5'))

        rna = mudata['rna']
        atac = mudata['atac']

        atac_intervals = atac.var['interval'].reset_index().drop(columns=['index'])
        atac_intervals_df = pd.DataFrame(atac_intervals['interval'].str.split(':|-', expand=True).values, columns=['chrom', 'start', 'end'])
        atac_intervals_df['original_peak_idx'] = np.arange(atac_intervals_df.shape[0])
        atac_intervals_df['original_peak'] = atac_intervals['interval']
        atac_intervals_bed = BedTool.from_dataframe(atac_intervals_df)

        os.environ["PATH"] = os.path.expanduser("~/bin") + ":" + os.environ["PATH"] # add path to liftOver
        chain_file_path = os.path.join(os.environ['DATAPATH'], chain_file_name)
        lifted_intervals_bed = atac_intervals_bed.liftover(chain_file_path, unmapped=os.path.join(datapath, 'mouse_brain_10x_peak_beds_unmapped_mm10ToHg38.bed'), liftover_args='-minMatch=0.8')
        lifted_intervals_df = lifted_intervals_bed.to_dataframe()
        lifted_intervals_df['intervals'] = lifted_intervals_df['chrom'] + ':' + lifted_intervals_df['start'].astype(str) + '-' + lifted_intervals_df['end'].astype(str)

        ## Subset ATAC to lifted peaks
        atac = atac[:, lifted_intervals_df['name']].copy() # 'name' column is the original peak index
        atac.var = atac.var.reset_index(drop=False, inplace=False)

        atac.var['intervals'] = lifted_intervals_df['intervals']
        atac.var['original_peak_idx'] = lifted_intervals_df['name']
        atac.var['original_peak'] = lifted_intervals_df['score']

        assert (atac.var['index'] == atac.var['original_peak']).all(), "Original peak index is not preserved"
        atac.var = atac.var.drop(columns=['index', 'gene_ids', 'interval']) # all seem like the same as original_peak_idx
        atac.var = atac.var.set_index('intervals', inplace=False)

        ## Save lifted ATAC
        mudata_lifted = MuData({'atac': atac, 'rna': rna})
        mudata_lifted.write_h5mu(os.path.join(datapath, 'mouse_brain_10x_atac_lifted.h5mu'), compression='gzip')  # A value is trying to be set on a copy of a slice from a DataFrame.
        return

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

    elif args.genes_by_peaks_str is None:

        ## Load the lifted data
        mudata = muon_read_h5mu(os.path.join(datapath, 'mouse_brain_10x_atac_lifted.h5mu'))
        rna = mudata['rna']
        atac = mudata['atac']

        ## Load cluster labels
        cluster_labels_file = os.path.join(datapath, 'gex_graph_cluster_annotations.csv')
        cluster_labels = pd.read_csv(cluster_labels_file)
        cluster_labels = cluster_labels.set_index('Barcode')

        rna.obs = rna.obs.join(cluster_labels, how='left')
        atac.obs = atac.obs.join(cluster_labels, how='left')

        ## Map mouse genes to human genes
        mouse_to_human_dict = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'mouse_CRGm38_with_human_orthologs.tsv'), sep='\t')
        mouse_to_human_dict = mouse_to_human_dict.drop_duplicates(subset=['Gene name'])
        mouse_to_human_map = mouse_to_human_dict.set_index('Gene name')['Human gene name']
        rna.var_names = rna.var_names.map(mouse_to_human_map)
        rna = rna[:, rna.var_names.notna()].copy()

        ## Subset to unique human genes
        unique_human_genes_mask = ~rna.var.index.duplicated(keep='first')
        rna = rna[:, unique_human_genes_mask].copy()

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, atac_datapath, rna_datapath

        ## Normalize data
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(atac)
        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        sc.pp.log1p(rna)

        ## Subset to variable features
        if hvg_only:
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        # min-max scaling
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=None)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        ## find peaks and genes that overlap
        overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
        overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

        ## subset RNA genes
        rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

        ## subset ATAC peaks - requires ordering of peaks
        atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
        #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

        ## sort peaks and genes
        genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
        peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

        genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
        genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

        genes_sort_idxs = np.argsort(rna.var.index.tolist())
        rna = rna[:,genes_sort_idxs]

        peaks_sort_idxs = np.argsort(atac.var.index.tolist())
        atac = atac[:,peaks_sort_idxs]

        ## check alignment
        print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
        print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath

def pbmc_10x_setup(args, cell_group='seurat_annotations', batch_group=None, hvg_only=False, protein_coding_only=True, do_gas=False, do_peak_gene_alignment=True, return_type='loaders', return_raw_data=False, dataset='pbmc_10x'):

    atac_datapath = rna_datapath = datapath = os.path.join(os.environ['DATAPATH'], 'pbmc_10x')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(datapath, ATAC_file)
        rna_fullpath = os.path.join(datapath, RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

    elif args.genes_by_peaks_str is None:

        RNA_file = "pbmcMultiome_rna.h5ad"
        ATAC_file = "pbmcMultiome_atac.h5ad"
        
        atac = anndata.read_h5ad( os.path.join(datapath, ATAC_file))
        rna  = anndata.read_h5ad( os.path.join(datapath, RNA_file) )

        ## Rename peaks to standard format
        atac.var.index = pd.DataFrame(atac.var.index.str.split('-', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna, atac, cell_group, datapath

        ## Preprocess data, usually in preparation for HVG
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        #sc.pp.log1p(atac)
        atac = sc.pp.scale(atac, zero_center=False, max_value=10, copy=True) # min-max scaling

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        #sc.pp.log1p(rna)
        rna = sc.pp.scale(rna, zero_center=False, max_value=10, copy=True) # min-max scaling, max_value not working if copy=False

        ## Subset to variable features. For this dataset, very restrictive HVG selection
        if hvg_only:
            atac = atac[:, atac.var['vst.variable'].astype(bool)].to_memory()
            rna = rna[:, rna.var['vst.variable'].astype(bool)].to_memory()

        if do_gas:

            ## Import ATAC data with gene activity scores (GAS)
            args.ATAC_gas_file = "pbmcMultiome_atac_gas.h5ad"
            atac_gas = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_gas_file) )
            
            ## Keep genes from gene activity scores that overlap with RNA genes
            order_gas_idxs = np.array([np.where(gene == atac_gas.var.index)[0][0] for gene in rna.var.index if gene in atac_gas.var.index])
            atac_gas = atac_gas[:, order_gas_idxs].to_memory()
            #atac = anndata.concat([atac, atac_gas], axis=1)

            ## Add column to RNA var indicating whether gene is in ATAC_gas var
            rna.var['in_ATAC_gas'] = rna.var.index.isin(atac_gas.var.index)
            print('Genes in RNA subset and ATAC_gas match:', (rna.var.index[rna.var['in_ATAC_gas']] == atac_gas.var.index).all())

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

        if not os.path.exists(genes_to_peaks_binary_mask_path):
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])
            
        else:
            print(f'peaks to genes mask found, loading {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
            #genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask.toarray(), index=genes_peaks_dict['genes'], columns=genes_peaks_dict['peaks'])

        if do_peak_gene_alignment:

            ## find peaks and genes that overlap
            overlapping_genes = set(rna.var.index).intersection(genes_peaks_dict['genes'])
            overlapping_peaks = set(atac.var.index).intersection(genes_peaks_dict['peaks'])
            #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ list(overlapping_genes) , list(overlapping_peaks) ]

            ## subset RNA genes
            rna = rna[:, rna.var.index.isin(overlapping_genes)].copy()
            #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[rna.var.index,:]

            ## subset ATAC peaks - requires ordering of peaks
            atac = atac[:, atac.var.index.isin(overlapping_peaks)].copy()
            #genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.loc[:,atac.var.index]

            ## sort peaks and genes
            genes_sort_idxs = np.argsort(genes_peaks_dict['genes'].tolist())
            peaks_sort_idxs = np.argsort(genes_peaks_dict['peaks'].tolist())

            genes_peaks_dict['genes'] = genes_peaks_dict['genes'][genes_sort_idxs]
            genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][peaks_sort_idxs]
            genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ genes_sort_idxs , : ]
            genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , peaks_sort_idxs ]

            genes_sort_idxs = np.argsort(rna.var.index.tolist())
            rna = rna[:,genes_sort_idxs]

            peaks_sort_idxs = np.argsort(atac.var.index.tolist())
            atac = atac[:,peaks_sort_idxs]

            ## check alignment
            print('Genes match:', (rna.var.index == genes_peaks_dict['genes']).all())
            print('Peaks match:', (atac.var.index == genes_peaks_dict['peaks']).all())

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key=batch_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath