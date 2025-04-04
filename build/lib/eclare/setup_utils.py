import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import os
from argparse import Namespace
from pybedtools import BedTool
from sklearn.model_selection import StratifiedShuffleSplit
from pybiomart import Dataset as biomart_Dataset
#import bbknn
from muon import atac as ac
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

from eclare.models import load_CLIP_model
from eclare.data_utils import create_loaders, keep_CREs_and_adult_only, merge_major_cell_group

def return_setup_func_from_dataset(dataset_name):

    if dataset_name == 'CAtlas_Tabula_Sapiens':
        setup_func = CAtlas_Tabula_Sapiens_setup

    elif dataset_name == 'mdd':
        setup_func = mdd_setup
        
    elif dataset_name == 'pbmc_multiome':
        setup_func = pbmc_multiome_setup

    elif dataset_name == 'roussos':
        setup_func = Roussos_cerebral_cortex_setup

    elif dataset_name == 'merged_roussos_pbmc_multiome':
        setup_func = merged_dataset_setup

    elif dataset_name == 'splatter_sim':
        setup_func = splatter_sim_setup

    elif dataset_name == 'toy_simulation':
        setup_func = toy_simulation_setup

    elif dataset_name == '388_human_brains':
        setup_func = snMultiome_388_human_brains_setup

    elif dataset_name == '388_human_brains_one_subject':
        setup_func = snMultiome_388_human_brains_one_subject_setup

    elif dataset_name == 'AD_Anderson_et_al':
        setup_func = AD_Anderson_et_al_setup

    elif dataset_name == 'PD_Adams_et_al':
        setup_func = PD_Adams_et_al_setup

    elif dataset_name == 'human_dlpfc':
        setup_func = human_dlpfc_setup

    elif dataset_name == 'sea_ad':
        setup_func = sea_ad_setup

    return setup_func

def get_protein_coding_genes(rna):

    ensembl_path = os.path.join(os.environ['DATAPATH'], 'ensembl_gene_positions.csv')

    protein_coding_genes = pd.read_csv(ensembl_path)
    protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
    protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
    rna.var['protein_coding'] = rna.var.index.isin(protein_coding_genes_list)
    rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()
    return rna

def get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size = 1e6, feature_selection_method = 'gradient boosting regression'):

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
                        'TSS end (bp)': 'mean'})
        
            rna_var_tmp['TSS start (bp)'] = rna_var_tmp['TSS start (bp)'].astype(int).astype(str)
            rna_var_tmp['TSS end (bp)'] = rna_var_tmp['TSS end (bp)'].astype(int).astype(str)
            rna_var_tmp['Chromosome/scaffold name'] = 'chr' + rna_var_tmp['Chromosome/scaffold name']

            #rna.var = pd.merge(rna.var, rna_var_tmp, left_index=True, right_index=True, how='left')

        #rna = rna[:,~rna.var.isna().values.any(1)]

        ## Create bedtools objects
        atac_bed_df = pd.DataFrame(list(atac.var.index.str.split(':|-', expand=True).values), columns=['chrom','start','end'])
        atac_bed_df = atac_bed_df.copy()
        atac_bed_df['name'] = atac.var.index
        atac_bed = BedTool.from_dataframe(atac_bed_df)

        #rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'TSS start (bp)', 'TSS end (bp)', 'index']])
        rna_bed = BedTool.from_dataframe(rna_var_tmp.reset_index()[['Chromosome/scaffold name', 'TSS start (bp)', 'TSS end (bp)', 'Gene name']])

        ## Find in-cis overlapping intervals
        cis_overlaps = rna_bed.window(atac_bed, w=window_size).to_dataframe()
        i, r = pd.factorize(cis_overlaps['name'])       # genes
        j, c = pd.factorize(cis_overlaps['thickEnd'])   # peaks
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

            ## Save variables needed to create mask, for troubleshooting
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

        genes_to_peaks_binary_mask = np.zeros((n, m), dtype=np.int64)
        np.add.at(genes_to_peaks_binary_mask, (i, j), 1)
        #genes_to_peaks_binary_mask_df = pd.DataFrame(genes_to_peaks_binary_mask, index=r, columns=c)

        ## Save mask
        genes_to_peaks_binary_mask = csr_matrix(genes_to_peaks_binary_mask)
        save_npz(genes_to_peaks_binary_mask_path, genes_to_peaks_binary_mask)

        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        genes_peaks_dict = {'genes':r, 'peaks':c}
        with open(pkl_path, 'wb') as f: pkl_dump(genes_peaks_dict, f)

        return genes_to_peaks_binary_mask, genes_peaks_dict

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


    if peak_overlap_method == 'set':
        overlapping_atac_peaks = set(source_atac.var.index).intersection(target_atac.var.index)

        source_atac = source_atac[:, source_atac.var.index.isin(overlapping_atac_peaks)]
        order_peaks_idxs = np.array([np.where(peak == source_atac.var.index)[0][0] for peak in target_atac.var.index if np.isin(peak, overlapping_atac_peaks).item()])
        source_atac = source_atac[:, order_peaks_idxs]

    elif peak_overlap_method == 'bedtools':
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

def merged_dataset_setup(args, pretrain=False, cell_group='Cell type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', dataset=None):

    dataset = dataset.replace('merged_','').replace('imputed_','')

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], f'merged_data/{dataset}')


    if args.genes_by_peaks_str is not None:

        args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_mdd.h5ad"
        args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_mdd.h5ad"

        atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
        rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )
        
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_mdd.npz")
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
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath

def toy_simulation_setup(args, pretrain=False, cell_group='dummy_celltype'):

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'toy_simulation')

    args.RNA_file = "toy_simulation_rna.h5ad"
    args.ATAC_file = "toy_simulation_atac.h5ad"

    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file) )
    rna = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    if pretrain:
        #rna_train_loader, rna_valid_loader, _, rna_train_num_batches, rna_valid_num_batches, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, _, atac_train_num_batches, atac_valid_num_batches, _, _, _, _ = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return atac_train_loader, atac_train_num_batches, atac_valid_loader, atac_valid_num_batches, n_peaks, n_genes
    
    elif not pretrain:
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx

def splatter_sim_setup(args, pretrain=False, cell_group='Group'):

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], '10x_pbmc')

    args.RNA_file = "pbmcMultiome_rna_sim_gas.h5ad"
    args.ATAC_file = "pbmcMultiome_atac_sim.h5ad"

    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file) )
    rna = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    if pretrain:
        #rna_train_loader, rna_valid_loader, _, rna_train_num_batches, rna_valid_num_batches, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, _, atac_train_num_batches, atac_valid_num_batches, _, _, _, _ = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return atac_train_loader, atac_train_num_batches, atac_valid_loader, atac_valid_num_batches, n_peaks, n_genes
    
    elif not pretrain:
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx

def pbmc_multiome_setup(args, pretrain=False, cell_group='seurat_annotations', hvg_only=False, protein_coding_only=True, do_gas=False, return_type='loaders'):

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], '10x_pbmc')

    args.RNA_file = "pbmcMultiome_rna.h5ad"
    args.ATAC_file = "pbmcMultiome_atac.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )


    if args.genes_by_peaks_str is not None:

        args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_mdd.h5ad"
        args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_mdd.h5ad"

        atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
        rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )
        
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_mdd.npz")
        genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

    elif args.genes_by_peaks_str is None:

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        ## Preprocess data, usually in preparation for HVG
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
        #sc.pp.log1p(atac)
        atac = sc.pp.scale(atac, zero_center=False, max_value=10, copy=True) # min-max scaling

        sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
        #sc.pp.log1p(rna)
        rna = sc.pp.scale(rna, zero_center=False, max_value=10, copy=True) # min-max scaling, max_value not working if copy=False

        ## Subset to variable features
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
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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

    '''
    if pretrain:
        rna_train_loader, rna_valid_loader, _, rna_train_num_batches, rna_valid_num_batches, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, _, atac_train_num_batches, atac_valid_num_batches, _, _, _, _ = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return align_setup_completed, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx

    elif not pretrain:
    '''

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict


def mdd_setup(args, pretrain=None, cell_groups=dict({'atac':'ClustersMapped','rna':'Broad'}), hvg_only=True, protein_coding_only=True, do_gas=False, do_pseudobulk=False, return_type='loaders', overlapping_subjects_only=False, return_raw_data=False, dataset='mdd'):

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'mdd_data')

    if args.genes_by_peaks_str is not None:

        ## manually re-introducing the 'merge' string, but should be more self-consistent across filenames and directory names
        if overlapping_subjects_only:
            args.RNA_file = f"rna_overlap_{args.genes_by_peaks_str}_aligned_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_overlap_{args.genes_by_peaks_str}_aligned_{args.source_dataset}.h5ad"
        else:
            if args.source_dataset == dataset:
                args.RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
                args.ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"

            elif args.target_dataset == dataset:
                args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
                args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"

        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## already being loaded by other dataset, datapath of which genes-to-peaks mask is stored
        genes_to_peaks_binary_mask = genes_peaks_dict = None

        cell_group = cell_groups['atac']

    elif args.genes_by_peaks_str is None:
    
        args.RNA_file = "mdd_rna.h5ad"
        args.ATAC_file = "mdd_atac_broad.h5ad"

        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)

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
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath
        
        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
            sc.pp.highly_variable_genes(atac, n_top_genes=56354) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10)
            genes_peaks_dict['peaks'] = genes_peaks_dict['peaks'][atac.var['highly_variable'].astype(bool)]
            genes_to_peaks_binary_mask = genes_to_peaks_binary_mask[ : , atac.var['highly_variable'].astype(bool) ]
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            #sc.pp.filter_genes(rna, min_counts=3)
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=10112) # sc.pl.highly_variable_genes(rna)
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

    ## delete atac.raw if it exists
    if 'raw' in atac.uns.keys():
        del atac.raw

    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, 'mdd', args.batch_size, args.total_epochs, cell_group_key=cell_group, valid_only=False)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, 'mdd', args.batch_size, args.total_epochs, cell_group_key=cell_group, valid_only=False)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    elif return_type == 'data':
        return rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath
    
def CAtlas_Tabula_Sapiens_setup(args, pretrain=False, cell_group='major_cell_type'):

    if pretrain:

        if args.subsample > 0:
            atac = anndata.read_h5ad( os.path.join(args.atac_datapath , args.ATAC_file), backed='r+')
            subsample_atac_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(atac)), y=atac.obs['cell type']))
            atac = atac[subsample_atac_idxs].to_memory()  # long, almost 1 minute
        else:
            atac = anndata.read_h5ad( os.path.join(args.atac_datapath , args.ATAC_file))
        
        atac, CRE_to_genes_mapper = keep_CREs_and_adult_only(atac, args, check_adult_only=False)       

        n_peaks = atac.n_vars
        n_genes = 2085  # hard-coded, prevents needing to import RNA data for now

        atac_train_loader, atac_valid_loader, _, atac_train_num_batches, atac_valid_num_batches, _, _, _, _ = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key='cell type')

        return atac_train_loader, atac_train_num_batches, atac_valid_loader, atac_valid_num_batches, n_peaks, n_genes, CRE_to_genes_mapper
        
    elif not pretrain:
        if args.subsample > 0:
            rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file), backed='r')
            atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file), backed='r+')

        else:
            rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file))
            atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))

        if cell_group == 'major_cell_type':
                
            ## Merge 'major cell group' annotations to ATAC data
            atac, rna, onto_overlap = merge_major_cell_group(atac, rna, args)

            if args.subsample > 0:
                subsample_rna_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(rna)), y=rna.obs['major cell group']))
                rna = rna[subsample_rna_idxs].to_memory()
                subsample_atac_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(atac)), y=atac.obs['major cell group']))
                atac = atac[subsample_atac_idxs].to_memory()

            ## Retain cells from overlapping celltypes only, if appropriate
            args.overlap_only = (not args.all_celltypes)
            if args.overlap_only:
                rna  = rna[np.isin( rna.obs['major cell group'] , onto_overlap )].copy()
                atac = atac[np.isin( atac.obs['major cell group'] , onto_overlap )].copy()

            atac, CRE_to_genes_mapper = keep_CREs_and_adult_only(atac, args)

        elif cell_group == 'bbknn':

            if args.subsample > 0:
                subsample_atac_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(atac)), y=atac.obs['cell type']))
                atac = atac[subsample_atac_idxs].to_memory()
                subsample_rna_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(rna)), y=rna.obs['cell_ontology_class']))
                rna = rna[subsample_rna_idxs].to_memory()

            ## Retain peaks included in CRE_to_genes_matrix
            atac, CRE_to_genes_mapper = keep_CREs_and_adult_only(atac, args)
                
            ## Run BBKNN and find leiden clusters
            ac.tl.lsi(atac)
            atac.obsm['X_pca'] = atac.obsm['X_lsi']
            atac.obs['modality'] = 'atac'
            rna.obs['modality'] = 'rna'
            adata_merged = sc.concat([rna, atac], join='inner')

            bbknn.bbknn(adata_merged, use_rep='X_pca', batch_key='modality', computation='pynndescent', metric='cosine', neighbors_within_batch=100)
            sc.tl.leiden(adata_merged)

            ## high-jack cell-types into leiden clusters
            rna.obs['major cell group'] = adata_merged.obs['leiden'].loc[adata_merged.obs['modality'] == 'rna']
            atac.obs['major cell group'] = adata_merged.obs['leiden'].loc[adata_merged.obs['modality'] == 'atac']

            ## Retain cells belonging to cell groups with at least two members
            rna_celltype_counts  = rna.obs['major cell group'].value_counts()
            rna  = rna[np.isin( rna.obs['major cell group'] , rna_celltype_counts[rna_celltype_counts > 1].index )].copy()

            atac_celltype_counts = atac.obs['major cell group'].value_counts()
            atac = atac[np.isin( atac.obs['major cell group'] , atac_celltype_counts[atac_celltype_counts > 1].index )].copy()

        elif cell_group == 'modality':

            if args.subsample > 0:
                subsample_atac_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(atac)), y=atac.obs['major cell group']))
                atac = atac[subsample_atac_idxs].to_memory()
                subsample_rna_idxs, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.subsample).split(X=np.zeros(len(rna)), y=rna.obs['major cell group']))
                rna = rna[subsample_rna_idxs].to_memory()

            ## Merge 'major cell group' annotations to ATAC data
            atac, rna = merge_major_cell_group(atac, rna, args)
            atac, CRE_to_genes_mapper = keep_CREs_and_adult_only(atac, args)

            n_genes, n_peaks = rna.n_vars, atac.n_vars
            

        ## Model setup
        n_genes, n_peaks = rna.n_vars, atac.n_vars

        ## Create data loader
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args)

        align_setup_completed = True
        return align_setup_completed, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx

def Bm_1469_setup(args, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/1469')

    args.RNA_file = "rna.h5ad"
    args.ATAC_file = "atac.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'region2gene.csv')
    genes_to_peaks_binary_mask = pd.read_csv(genes_to_peaks_binary_mask_path, index_col=0)
    genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.T

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def Bm_P0_setup(args, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        raise NotImplemented()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/1469')
    elif args.machine == 'narval':
        args.rna_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/P0/RNA')
        args.atac_datapath = os.path.jon(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/P0/ATAC')

    args.RNA_file = "rna.h5ad"
    args.ATAC_file = "atac.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'region2gene.mtx')
    genes_to_peaks_binary_mask = sc.read_mtx(genes_to_peaks_binary_mask_path).X.toarray()
    genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask, index=atac.var['region_id'], columns=rna.var['gene_id'])

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def Bm_uterus_setup(args, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        raise NotImplementedError()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/1469')
    elif args.machine == 'narval':
        args.rna_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/uterus/RNA')
        args.atac_datapath = os.path.jon(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/uterus/ATAC')

    args.RNA_file = "Uterus_Wang_2020.h5ad"
    args.ATAC_file = "adata_anno.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'region2gene.mtx')
    genes_to_peaks_binary_mask = sc.read_mtx(genes_to_peaks_binary_mask_path).X.toarray()
    genes_to_peaks_binary_mask = pd.DataFrame(genes_to_peaks_binary_mask, index=atac.var['region_id'], columns=rna.var['gene_id'])

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def bm_bmmc_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        assert NotImplementedError()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/bmmc')
    elif args.machine == 'narval':
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/bmmc')

    args.RNA_file = "multiome_bmmc_site1_or_donor1_RNA.h5ad"
    args.ATAC_file = "multiome_bmmc_site1_or_donor1_ATAC_pmat.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'bmmc_peak_GxR.csv')
    genes_to_peaks_binary_mask = pd.read_csv(genes_to_peaks_binary_mask_path, index_col=0)

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def bm_mouse_skin_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        raise NotImplementedError()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/mouse_skin')
    elif args.machine == 'narval':
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/mouse_skin')

    args.RNA_file = "shareseq_mouse_skin_rna.h5ad"      # also have mouse_skin_shareseq_rna_10k.h5ad
    args.ATAC_file = "shareseq_mouse_skin_atac.h5ad"    # also have mouse_skin_shareseq_atac_10k.h5ad
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'mouse_skin_peak_GxR.csv')
    genes_to_peaks_binary_mask = pd.read_csv(genes_to_peaks_binary_mask_path, index_col=0)

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def bm_multiome_pbmc_10k_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        raise NotImplementedError()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/multiome_pbmc_10k')
    elif args.machine == 'narval':
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/multiome_pbmc_10k')

    args.RNA_file = "pbmc_10x_rna_public.h5ad"
    args.ATAC_file = "pbmc_10x_atac_public.h5ad"
    
    atac = anndata.read_h5ad( os.path.join(args.atac_datapath, args.ATAC_file))
    rna  = anndata.read_h5ad( os.path.join(args.rna_datapath, args.RNA_file) )

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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

    ## Load dummy-encoded overlapping intervals, use later as mask
    genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, 'pbmc_peak_GxR.csv')
    genes_to_peaks_binary_mask = pd.read_csv(genes_to_peaks_binary_mask_path, index_col=0)

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask

def bm_hpap_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=False, do_gas=False):

    if args.machine == 'local':
        raise NotImplementedError()
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'Benchmarking_single-cell_RNA_and_ATAC_data_integration_dataset/hpap')
    elif args.machine == 'narval':
        args.rna_datapath = os.path.join('benchmark_sc_multiomic_integration_dataset/hpap/unpaired_RNA')      # also have paired_RNA
        args.atac_datapath = os.path.join('benchmark_sc_multiomic_integration_dataset/hpap/unpaired_ATAC')    # also have paired_ATAC

    args.RNA_file = "RNA_counts.mtx"
    args.ATAC_file = "ATAC_counts.mtx"
    
    atac = sc.read_10x_mtx(args.atac_datapath)
    rna  = sc.read_10x_mtx(args.rna_datapath)

    ## Subset to protein-coding genes
    if protein_coding_only:

        raise NotImplementedError()
    
        dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
        protein_coding_genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
        protein_coding_genes = protein_coding_genes[protein_coding_genes['Gene type'] == 'protein_coding']
        protein_coding_genes_list = protein_coding_genes['Gene name'].tolist()
        rna = rna[:, rna.var.index.isin(protein_coding_genes_list)].to_memory()

    ## Subset to variable features
    if hvg_only:
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
        # sc.pl.highly_variable_genes(atac)

        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        #sc.pl.highly_variable_genes(rna)

        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

    if do_gas:

        raise NotImplementedError()

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
    genes_to_peaks_binary_mask_path = os.path.join('/home/dmannk/projects/def-liyue/dmannk/data/benchmark_sc_multiomic_integration_dataset/hpap', 'genes_to_peaks_binary_mask.csv')

    if not os.path.exists(genes_to_peaks_binary_mask_path):

        print('peaks to genes mask not found, saving to csv')

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
        rna = rna[:,~rna.var.isna().any(1)]

        rna.var['Gene start (bp)'] = rna.var['Gene start (bp)'].astype(int).astype(str)
        rna.var['Gene end (bp)'] = rna.var['Gene end (bp)'].astype(int).astype(str)
        rna.var['Chromosome/scaffold name'] = 'chr' + rna.var['Chromosome/scaffold name']

        ## Create bedtools objects
        rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)', 'index']])
        atac_bed_df = pd.DataFrame(list(atac.var.index.str.split('-', expand=True).values), columns=['chrom','start','end'])
        atac_bed_df['name'] = atac.var.index
        atac_bed = BedTool.from_dataframe(atac_bed_df)

        ## Find in-cis overlapping intervals
        window_size = 1e6
        cis_overlaps = rna_bed.window(atac_bed, w=window_size).to_dataframe()
        genes_to_peaks_binary_mask = pd.get_dummies(cis_overlaps.set_index('name')['thickEnd'])
        genes_to_peaks_binary_mask = genes_to_peaks_binary_mask.groupby('name', sort=False).sum()

        ## Save mask
        genes_to_peaks_binary_mask.to_csv(genes_to_peaks_binary_mask_path)
        
    else:
        print('peaks to genes mask found, loading csv')
        genes_to_peaks_binary_mask = pd.read_csv(genes_to_peaks_binary_mask_path, index_col=0)

    rna_var = rna.var.loc[genes_to_peaks_binary_mask.index]
    atac_var = atac.var.loc[genes_to_peaks_binary_mask.columns]

    ## subset RNA genes
    rna = rna[:, rna.var.index.isin(rna_var.index)]
    print('Genes match:', (rna.var.index == rna_var.index).all())
    rna.var = rna_var

    ## subset ATAC peaks - requires ordering of peaks
    order_peaks_idxs = np.array([np.where(peak == atac.var.index)[0][0] for peak in atac_var.index if peak in atac.var.index])
    atac = atac[:,order_peaks_idxs]
    print('Peaks match:', (atac.var.index == atac_var.index).all())
    atac.var = atac_var

    n_peaks, n_genes = atac.n_vars, rna.n_vars

    rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
    return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask


def Roussos_cerebral_cortex_setup(args, pretrain=False, cell_group='Cell type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='roussos', \
    keep_group=['Fet', 'Inf', 'Child', 'Adol', 'Adult']):

    datapath = os.path.join(os.environ['DATAPATH'], 'Roussos_lab')
    args.rna_datapath = os.path.join(datapath, 'rna')
    args.atac_datapath = os.path.join(datapath, 'atac')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(args.atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## retain from specific developmental stages
        keep_atac_subj = atac.obs['Donor ID'].str.startswith(tuple(keep_group)) 
        keep_rna_subj = rna.obs['Donor ID'].str.startswith(tuple(keep_group))

        if 'Fet' not in keep_group:
            keep_atac_ct = ~atac.obs['Cell type'].str.contains('fetal')
            keep_rna_ct = ~rna.obs['Cell type'].str.contains('fetal')

            keep_atac = keep_atac_subj & keep_atac_ct
            keep_rna = keep_rna_subj & keep_rna_ct
        else:
            keep_atac = keep_atac_subj
            keep_rna = keep_rna_subj


        assert (keep_atac == keep_rna).all()

        atac = atac[keep_atac].to_memory()
        rna = rna[keep_rna].to_memory()

    elif args.genes_by_peaks_str is None:

        args.RNA_file = "roussos_rna.h5ad"
        args.ATAC_file = "roussos_atac.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## import cell-type labels and assign to data
        atac_barcodes = pd.read_csv( os.path.join(args.atac_datapath, 'atac_barcodes.tsv.gz'), compression='gzip', sep='\t', index_col=0)
        rna_barcodes  = pd.read_csv( os.path.join(args.rna_datapath, 'rna_barcodes.tsv.gz'), compression='gzip', sep='\t', index_col=0)

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
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

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
            genes_peaks_dict['peaks'] = pd.DataFrame(genes_peaks_dict['peaks'].str.split('-', expand=True).to_list()).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).values

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


        '''
        if pretrain:
            rna_train_loader, rna_valid_loader, _, rna_train_num_batches, rna_valid_num_batches, _, _, _, _ = create_loaders(rna, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
            atac_train_loader, atac_valid_loader, _, atac_train_num_batches, atac_valid_num_batches, _, _, _, _ = create_loaders(atac, args.dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
            return align_setup_completed, rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx

        elif not pretrain:
        '''
        
    n_peaks, n_genes = atac.n_vars, rna.n_vars
    print(f'Number of peaks and genes remaining: {n_peaks} peaks & {n_genes} genes')

    if return_type == 'loaders':
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath
    

def snMultiome_388_human_brains_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', dataset='388_human_brains'):
    
    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], '388_human_brains')
    celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')

    if args.genes_by_peaks_str is not None:

        args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_mdd.h5ad"
        args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_mdd.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
        
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_mdd.npz")
        genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
        pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
        with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

    else:

        args.RNA_file = "rna.h5ad"
        args.ATAC_file = "atac.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
        
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Remove subjects from blacklist, according to Science paper Fig. S23
        subjects_blacklist = ['RT00377N', 'RT00373N', 'RT00379N', 'RT00385N']
        atac = atac[~rna.obs['subject'].isin(subjects_blacklist)].to_memory() # use rna.obs to remove subjects from atac, since atac does not have subject information yet
        rna = rna[~rna.obs['subject'].isin(subjects_blacklist)].to_memory()
        print('Subjects removed:', subjects_blacklist)

        ## Remove genes with zero counts
        zero_count_genes = np.asarray(rna.X.sum(0) == 0).squeeze()
        rna = rna[:, ~zero_count_genes]

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)

            predictions = celltypist.annotate(rna, majority_voting=False, model=celltypist_model_path)

            sc.pp.highly_variable_genes(rna, n_top_genes=rna.n_vars) # sc.pl.highly_variable_genes(rna)
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        ## Use celltypist to annotate cells by cell types
        sc.pp.scale(atac, zero_center=False, max_value=10)
        sc.pp.scale(rna, zero_center=False,  max_value=10)

        ## Keep top N cell types
        print('Subsetting cell types')
        num_celltypes = 30; print(f'Keeping top {num_celltypes} cell types')
        value_counts_threshold = predictions.predicted_labels.value_counts().sort_values(ascending=False).iloc[num_celltypes]
        keep_celltypes = (predictions.predicted_labels.value_counts() > value_counts_threshold)
        keep_celltypes = keep_celltypes[keep_celltypes].index.get_level_values(0).values.to_list()
        keep_celltypes_idx = np.array(predictions.probability_matrix.columns.isin(keep_celltypes))

        ## Assign cell types to data based on top probabilities according to retained cell-types
        final_probability_matrix = predictions.probability_matrix.loc[:, keep_celltypes_idx]
        final_predicted_labels = final_probability_matrix.columns[final_probability_matrix.values.argmax(1)]

        print('Assigning predicted cell types to data')
        rna.obs['cell_type'] = final_predicted_labels
        atac.obs['cell_type'] = final_predicted_labels

        ## Get genes by peaks, using gradient boosting regression to select features
        feature_selection_method = None

        if feature_selection_method == 'gradient boosting regression':
            genes_to_peaks_binary_mask_path = args.atac_datapath   # just first part of full path, to be completed inside of get_genes_by_peaks
                
            print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=feature_selection_method)

        else:
            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

            if not os.path.exists(genes_to_peaks_binary_mask_path):
                print(f'peaks to genes mask not found, saving to {os.path.splitext(genes_to_peaks_binary_mask_path)[-1]}')
                genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000, feature_selection_method=feature_selection_method)
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
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key='subject')
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group, batch_key='subject')
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath


def snMultiome_388_human_brains_one_subject_setup(args, subject='RT00391N', pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders'):
        
        args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], '388_human_brains', subject)
        celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')
    
        if args.genes_by_peaks_str is not None:
    
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_mdd.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_mdd.h5ad"

            atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
            rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
            atac = anndata.read_h5ad(atac_fullpath)
            rna  = anndata.read_h5ad(rna_fullpath)
            
            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_mdd.npz")
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.splitext(genes_to_peaks_binary_mask_path)[0] + '.pkl'
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)
    
        else: # not recommended, better to run process_datasets first
    
            args.RNA_file = f"{subject}_rna.h5ad"
            args.ATAC_file = f"{subject}_atac.h5ad"

            atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
            rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
            
            atac = anndata.read_h5ad(atac_fullpath)
            rna  = anndata.read_h5ad(rna_fullpath)

            ## Remove genes with zero counts
            zero_count_genes = np.asarray(rna.X.sum(0) == 0).squeeze()
            rna = rna[:, ~zero_count_genes]
    
            ## Subset to protein-coding genes
            if protein_coding_only:
                rna = get_protein_coding_genes(rna)
    
            ## Subset to variable features
            if hvg_only:
                ac.pp.tfidf(atac, scale_factor=1e4)
                sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
                sc.pp.log1p(atac)
                sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
                atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

                sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
                sc.pp.log1p(rna)
                sc.pp.highly_variable_genes(rna, n_top_genes=rna.n_vars) # sc.pl.highly_variable_genes(rna)
                rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

            ## Use celltypist to annotate cells by cell types
            predictions = celltypist.annotate(rna, majority_voting=False, model=celltypist_model_path)
            sc.pp.scale(atac, zero_center=False, max_value=10)
            sc.pp.scale(rna, zero_center=False, max_value=10)

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

            ## Get genes by peaks, using gradient boosting regression to select features
            genes_to_peaks_binary_mask_path = args.atac_datapath   # just first part of full path, to be completed inside of get_genes_by_peaks
            genes_to_peaks_binary_mask, genes_peaks_dict = get_genes_by_peaks(rna, atac, genes_to_peaks_binary_mask_path, window_size=250000)

            ''' CANNOT PREDICT AHEAD OF TIME NUMBER VARS, PRIOR TO GRADIENT BOOSTING REGRESSION

            ## Save dummy-encoded overlapping intervals, use later as mask
            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
            '''

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
            rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, '388_human_brains_one_subject', args.batch_size, args.total_epochs, cell_group_key=cell_group)
            atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, '388_human_brains_one_subject', args.batch_size, args.total_epochs, cell_group_key=cell_group)
            return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
        
        elif return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath
    

def AD_Anderson_et_al_setup(args, pretrain=False, cell_group='predicted.id', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='AD_Anderson_et_al'):
        
    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'AD_Anderson_et_al/snMultiome')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(args.atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
        

    elif args.genes_by_peaks_str is None:

        args.RNA_file = "rna_ctrl.h5ad"
        args.ATAC_file = "atac_ctrl.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath


def PD_Adams_et_al_setup(args, pretrain=False, cell_group='cell_type', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='PD_Adams_et_al'):
    
    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'PD_Adams_et_al')
    celltypist_model_path = os.path.join(os.environ['DATAPATH'], 'Adult_Human_PrefrontalCortex.pkl')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(args.atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
        
    
    elif args.genes_by_peaks_str is None:

        args.RNA_file = "rna_ctrl.h5ad"
        args.ATAC_file = "atac_ctrl.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)

            predictions = celltypist.annotate(rna, majority_voting=False, model=celltypist_model_path)

            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()


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
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath


def human_dlpfc_setup(args, pretrain=False, cell_group='subclass', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='human_dlpfc'):
        
    datapath = args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'human_dlpfc')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(args.atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
    
    elif args.genes_by_peaks_str is None:

        args.RNA_file = "GSE207334_Multiome_rna_counts.mtx.gz"
        args.ATAC_file = "GSE207334_Multiome_atac_counts.mtx.gz"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
        ## Load data
        atac = sc.read_mtx(atac_fullpath).T
        rna  = sc.read_mtx(rna_fullpath).T

        ## Load metadata
        atac_peaks = pd.read_csv(os.path.join(args.atac_datapath, 'GSE207334_Multiome_atac_peaks.txt.gz'), delimiter=None, header=None)
        rna_genes = pd.read_csv(os.path.join(args.rna_datapath, 'GSE207334_Multiome_rna_genes.txt.gz'), delimiter=None, header=None)
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
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath


def sea_ad_setup(args, pretrain=False, cell_group='Subclass', hvg_only=True, protein_coding_only=True, do_gas=False, return_type='loaders', return_raw_data=False, dataset='sea_ad'):
        
    args.rna_datapath = args.atac_datapath = os.path.join(os.environ['DATAPATH'], 'SEA-AD')

    if args.genes_by_peaks_str is not None:

        if args.source_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.h5ad"
            binary_mask_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.npz"
            genes_peaks_dict_file = f"genes_to_peaks_binary_mask_{args.genes_by_peaks_str}_aligned_target_{args.target_dataset}.pkl"

            genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, binary_mask_file)
            genes_to_peaks_binary_mask = load_npz(genes_to_peaks_binary_mask_path)
            pkl_path = os.path.join(args.atac_datapath, genes_peaks_dict_file)
            with open(pkl_path, 'rb') as f: genes_peaks_dict = pkl_load(f)

        elif args.target_dataset == dataset:
            args.RNA_file = f"rna_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            args.ATAC_file = f"atac_{args.genes_by_peaks_str}_aligned_source_{args.source_dataset}.h5ad"
            binary_mask_file = genes_peaks_dict_file = genes_to_peaks_binary_mask = genes_peaks_dict = None

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)

        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)
    
    elif args.genes_by_peaks_str is None:

        args.RNA_file = "rna_paired_not_and_low_AD.h5ad"
        args.ATAC_file = "atac_paired_not_and_low_AD.h5ad"

        atac_fullpath = os.path.join(args.atac_datapath, args.ATAC_file)
        rna_fullpath = os.path.join(args.rna_datapath, args.RNA_file)
    
        atac = anndata.read_h5ad(atac_fullpath)
        rna  = anndata.read_h5ad(rna_fullpath)

        ## Subset to protein-coding genes
        if protein_coding_only:
            rna = get_protein_coding_genes(rna)

        if return_raw_data and return_type == 'data':
            return rna.to_memory(), atac.to_memory(), cell_group, None, None, args.atac_datapath, args.rna_datapath

        ## Subset to variable features
        if hvg_only:
            ac.pp.tfidf(atac, scale_factor=1e4)
            sc.pp.normalize_total(atac, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(atac)
            sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
            sc.pp.scale(atac, zero_center=False, max_value=10) # min-max scaling
            atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

            sc.pp.normalize_total(rna, target_sum=1e4, exclude_highly_expressed=False)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
            sc.pp.scale(rna, zero_center=False,  max_value=10) # min-max scaling
            rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()

        ## Save dummy-encoded overlapping intervals, use later as mask
        genes_to_peaks_binary_mask_path = os.path.join(args.atac_datapath, f'genes_to_peaks_binary_mask_{rna.n_vars}_by_{atac.n_vars}.npz')

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
        rna_train_loader, rna_valid_loader, rna_valid_idx, _, _, _, _, _, _ = create_loaders(rna, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        atac_train_loader, atac_valid_loader, atac_valid_idx, atac_train_num_batches, atac_valid_num_batches, atac_train_n_batches_str_length, atac_valid_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_n_epochs_str_length = create_loaders(atac, dataset, args.batch_size, args.total_epochs, cell_group_key=cell_group)
        return rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask
    
    elif return_type == 'data':
        return rna.to_memory(), atac.to_memory(), cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, args.atac_datapath, args.rna_datapath
    

def gene_activity_score_adata(atac, rna):

    ensembl_path = os.path.join(os.environ['DATAPATH'], 'ensembl_gene_positions.csv')

    ## Merge gene coordinates to RNA
    results = pd.read_csv(ensembl_path)
    filt_chroms = [str(i) for i in range(1,23+1)] + ['X','Y']
    results = results[results['Chromosome/scaffold name'].isin(filt_chroms)]
    rna_var_tmp = pd.merge(rna.var, results, left_index=True, right_on='Gene name', how='left').set_index('Gene name')
    rna_var_tmp = rna_var_tmp.groupby('Gene name', sort=False).agg({
                'Chromosome/scaffold name': 'max',
                'Gene start (bp)': 'mean',
                'Gene end (bp)': 'mean'})
    
    rna.var = pd.merge(rna.var, rna_var_tmp, left_index=True, right_index=True, how='left')
    #rna = rna[:,~rna.var.isna().values.any(1)]  # removes genes with nan variable, e.g. dispersions (but perhaps don't care)

    rna.var['Gene start (bp)'] = rna.var['Gene start (bp)'].astype(int).astype(str)
    rna.var['Gene end (bp)'] = rna.var['Gene end (bp)'].astype(int).astype(str)
    rna.var['Chromosome/scaffold name'] = 'chr' + rna.var['Chromosome/scaffold name']

    ## Create bedtools objects
    rna_bed = BedTool.from_dataframe(rna.var.reset_index()[['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)', 'index']])
    atac_bed_df = pd.DataFrame(list(atac.var.index.str.split(':|-', expand=True).values), columns=['chrom','start','end'])
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

    assert (gas_binary_mask.index == rna.var_names).all(), "Gene activity score matrix is not aligned with RNA var names"
    gas_binary_mask_sparse_transposed = csr_matrix(gas_binary_mask.values.T)

    atac_gas_X = atac.X @ gas_binary_mask_sparse_transposed
    atac_gas = AnnData(X=atac_gas_X, obs=atac.obs, var=rna.var)

    return atac_gas, rna


def get_genes_by_peaks_str(datasets = ["Roussos_lab", "AD_Anderson_et_al", "human_dlpfc", "PD_Adams_et_al"]):

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


    ## rename Roussos_lab to roussos
    genes_by_peaks_str_df['Roussos_lab'] = genes_by_peaks_str_df['roussos']
    genes_by_peaks_str_df = genes_by_peaks_str_df.drop(columns='roussos')
    genes_by_peaks_str_df = genes_by_peaks_str_df.rename(index={'Roussos_lab':'roussos'} , columns={'Roussos_lab':'roussos'})

    genes_by_peaks_str_timestamp_df['Roussos_lab'] = genes_by_peaks_str_timestamp_df['roussos']
    genes_by_peaks_str_timestamp_df = genes_by_peaks_str_timestamp_df.drop(columns='roussos')
    genes_by_peaks_str_timestamp_df = genes_by_peaks_str_timestamp_df.rename(index={'Roussos_lab':'roussos'} , columns={'Roussos_lab':'roussos'})

    ## save dataframes
    genes_by_peaks_str_df.to_csv(os.path.join(datapath, 'genes_by_peaks_str.csv'))
    genes_by_peaks_str_timestamp_df.to_csv(os.path.join(datapath, 'genes_by_peaks_str_timestamp.csv'))


def teachers_setup(model_paths, device, args, dataset_idx_dict=None):
    datasets = []
    models = {}
    target_rna_train_loaders = {}
    target_atac_train_loaders = {}
    target_rna_valid_loaders = {}
    target_atac_valid_loaders = {}
    
    for m, model_path in enumerate(model_paths):

        print(model_path)

        ## Load the model
        model, model_args_dict = load_CLIP_model(model_path, device=device)

        ## Determine the dataset
        dataset = model_args_dict['args'].source_dataset
        source_setup_func = return_setup_func_from_dataset(dataset)
        target_setup_func = return_setup_func_from_dataset(model_args_dict['args'].target_dataset)
        genes_by_peaks_str = model_args_dict['args'].genes_by_peaks_str

        print(dataset)
        datasets.append(dataset)

        ## Load the data loaders
        #rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
        #    source_setup_func(model_args_dict['args'], pretrain=None, return_type='loaders', dataset=dataset)
        
        overlapping_subjects_only = False #True if args.dataset == 'roussos' else False
        target_rna_train_loader, target_atac_train_loader, _, _, _, target_rna_valid_loader, target_atac_valid_loader, _, _, _, _, _, _, _, _ =\
            target_setup_func(model_args_dict['args'], pretrain=None, return_type='loaders')
        
        if args.train_encoders:
            model.train()

        models[dataset] = model

        if args.source_dataset_embedder:
            dataset_idx_dict[dataset] = m

        target_rna_train_loaders[dataset] = target_rna_train_loader
        target_atac_train_loaders[dataset] = target_atac_train_loader
        target_rna_valid_loaders[dataset] = target_rna_valid_loader
        target_atac_valid_loaders[dataset] = target_atac_valid_loader

    return datasets, models, target_rna_train_loaders, target_atac_train_loaders, target_rna_valid_loaders, target_atac_valid_loaders
