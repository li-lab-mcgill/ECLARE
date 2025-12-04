from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

import pandas as pd
import numpy as np
import anndata
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm
import ast
import ray

from scipy import stats
from statsmodels.stats.multitest import multipletests

from pybedtools import BedTool

# Optional (covariate regression)
try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
from eclare import set_env_variables
set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')


def main(subset_type=None, gene_activity_score_type=None, analyze_per_celltype=False):

    ## load data (RNA and ATAC gene activity score)
    mdd_rna_scaled_sub = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_rna_scaled_sub_15582.h5ad'))
    mdd_atac_sub = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_gas_broad_sub_14814.h5ad'))
    
    ## load data (ATAC broad peaks)
    mdd_atac_broad = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_broad.h5ad'), backed='r')
    mdd_atac_broad_sub = mdd_atac_broad[mdd_atac_broad.obs_names.isin(mdd_atac_sub.obs_names)].to_memory()

    mdd_atac_broad_sub.X = mdd_atac_broad_sub.raw.X
    mdd_atac_broad_sub.var_names = mdd_atac_broad_sub.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values

    ## load GWAS hits
    zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
    gwas_hits = pd.read_excel(zhu_supp_tables, sheet_name='Table S12', header=2)
    gwas_catalog_bedtool, gwas_catalog_metadata = get_gwas_catalogue_hits()
    #get_brain_gmt()

    ## load peak-gene links and km gene sets
    peak_gene_links, female_ExN, peak_names_mapper_reverse = get_peak_gene_links()
    km_gene_sets, km_gene_sets_mapper = load_km_gene_sets()

    ## restrict peak-enhancer-gene triplets for which all correlation signs agree (all positive or all negative)
    #peak_gene_links = peak_gene_links.loc[peak_gene_links['abs_sign_score_grn'].ge(0.5)]
    #peak_gene_links = peak_gene_links.loc[peak_gene_links['Correlation'].ge(0)]

    ## subset data
    mdd_rna_scaled_sub = subset_data(mdd_rna_scaled_sub, gwas_hits, subset_type)

    if gene_activity_score_type == 'promoter':
        mdd_atac_sub = subset_data(mdd_atac_sub, gwas_hits, subset_type, hvg_genes=mdd_rna_scaled_sub.var_names.tolist())

    elif gene_activity_score_type == 'in_cis':

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=os.cpu_count())
        
        genes_list = peak_gene_links['gene'].unique()
        
        # Put large data structures into Ray's object store for efficient sharing
        var_names_ref = ray.put(mdd_atac_broad_sub.var_names.to_numpy())
        raw_X_ref = ray.put(mdd_atac_broad_sub.raw.X)
        
        # Create remote tasks for each gene
        print(f"Submitting {len(genes_list)} gene tasks to Ray...")
        futures = []

        # Get peaks linked to this gene
        for gene in tqdm(genes_list, desc='Submitting tasks'):
            peak_gene_links_subset = peak_gene_links.loc[peak_gene_links['gene'].eq(gene), ['peak', 'Correlation']].copy()
            
            # Submit remote task
            future = compute_gene_activity_score.remote(gene, peak_gene_links_subset, var_names_ref, raw_X_ref)
            futures.append(future)
        
        # Collect results with progress tracking as they complete
        print(f"\nProcessing {len(futures)} genes in parallel...")
        results = []
        remaining_futures = futures.copy()
        
        with tqdm(total=len(futures), desc='Computing gene activity scores') as pbar:
            while remaining_futures:
                # Wait for at least one task to complete
                ready_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1)
                
                # Get the completed result
                for future in ready_futures:
                    results.append(ray.get(future))
                    pbar.update(1)
        
        # Sort results by gene name to maintain consistent ordering
        results.sort(key=lambda x: x[0])
        
        # Extract genes and activity scores separately
        genes_list_ordered = [gene for gene, _ in results]
        all_gene_activity_scores = [scores for _, scores in results]
        
        print(f"Completed processing {len(genes_list_ordered)} genes")
        
        # Shutdown Ray
        ray.shutdown()

        from scipy.sparse import csr_matrix
        gene_activity_scores_matrix = np.concatenate(all_gene_activity_scores, axis=1)
        gene_activity_scores_csr_matrix = csr_matrix(gene_activity_scores_matrix)
        var = pd.Series(genes_list_ordered, name='gene').to_frame().set_index('gene')
        mdd_atac_sub = anndata.AnnData(X=gene_activity_scores_csr_matrix, var=var, obs=mdd_atac_sub.obs)
        mdd_atac_sub.raw = mdd_atac_sub.copy()

        mdd_atac_sub = subset_data(mdd_atac_sub, gwas_hits, subset_type)

    elif gene_activity_score_type == None:
        ## stay at peak level
        peak_counts = np.array(mdd_atac_broad_sub.X.sum(axis=0)).flatten()
        obs = mdd_atac_sub.obs.copy()
        mdd_atac_sub = mdd_atac_broad_sub[:, peak_counts > 100].copy()
        mdd_atac_sub.obs = obs
        # mdd_atac_sub.write_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_broad_sub_14814.h5ad'))


    for sex in ['male', 'female']:

        if analyze_per_celltype:
            ## per celltype with power-tilted approach
            # Defaults: max_min_n_cells=20, max_imbalance_ratio=2.0, N_max_cells_per_donor=200
            rna_per_celltype, rna_var = run_pyDESeq2_on_celltypes(mdd_rna_scaled_sub, sex, max_min_n_cells=0, max_imbalance_ratio=np.inf, N_max_cells_per_donor=np.inf)
            atac_per_celltype, atac_var = run_pyDESeq2_on_celltypes(mdd_atac_sub, sex, max_min_n_cells=0, max_imbalance_ratio=np.inf, N_max_cells_per_donor=np.inf)

            ## ensure that atac_var in proper format (and that rna_var an index)
            atac_var = atac_var.reset_index().loc[:,'index'].str.split(':|-', expand=True).apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values
            atac_var = pd.Index(atac_var)
            rna_var = rna_var.index

            ## concatenate results across celltypes
            rna_results = concat_results(rna_per_celltype, rna_var)
            atac_results = concat_results(atac_per_celltype, atac_var)
        else:
            ## across all celltypes (combined analysis)
            rna_all, rna_var = run_pyDESeq2_on_celltypes(mdd_rna_scaled_sub, sex, test_type='all')
            atac_all, atac_var = run_pyDESeq2_on_celltypes(mdd_atac_sub, sex, test_type='all')
            
            ## format rna_all results
            rna_var = rna_var.index
            rna_all.index = rna_var
            rna_all.index.name = 'gene'
            rna_all.reset_index(inplace=True)
            rna_all['km'] = rna_all['gene'].map(gwas_hits.set_index('Target gene name')['km'].to_dict())
            rna_all['traits'] = rna_all['gene'].map(gwas_hits.groupby('Target gene name')['Trait'].unique().str.join('_').to_dict())
            rna_all['celltype'] = 'all'
            rna_all.set_index('celltype', inplace=True)
            rna_results = rna_all
            
            ## format atac_all results
            atac_var = atac_var.reset_index().loc[:,'index'].str.split(':|-', expand=True).apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values
            atac_var = pd.Index(atac_var)
            atac_all.index = atac_var
            atac_all.index.name = 'gene'
            atac_all.reset_index(inplace=True)
            atac_all['km'] = atac_all['gene'].map(gwas_hits.set_index('Target gene name')['km'].to_dict())
            atac_all['traits'] = atac_all['gene'].map(gwas_hits.groupby('Target gene name')['Trait'].unique().str.join('_').to_dict())
            atac_all['celltype'] = 'all'
            atac_all.set_index('celltype', inplace=True)
            atac_results = atac_all

        ## extract significant results
        rna_sig_results, rna_mdd_sig_results, rna_sig_genes_per_celltype, rna_sig_km_per_celltype = extract_results(rna_results, p_metric='padj')
        #atac_sig_results, atac_mdd_sig_results, atac_sig_genes_per_celltype, atac_sig_km_per_celltype = extract_results(atac_results, p_metric='padj')

        def do_enrichr(sig_results, celltype, gene_sets):
            from gseapy import enrichr
            
            universe = list(mdd_rna_scaled_sub.raw.var_names)

            if celltype is not None:
                hits = sig_results[sig_results.index.isin(celltype)]['gene'].unique()
            else:
                hits = sig_results['gene'].unique()

            enr = enrichr(
                gene_list=list(hits),
                gene_sets=gene_sets,
                background=list(universe),
                cutoff=1.0,
            )
            enr_res = enr.results
            return enr_res

        def do_gsea(results, celltype, gene_sets):
            from gseapy import prerank

            if celltype is not None:
                ranks = results[results.index.isin(celltype)].set_index('gene').get('log2FoldChange').sort_values(ascending=False).dropna()
            else:
                ranks = results.set_index('gene').get('log2FoldChange').sort_values(ascending=False).dropna()

            res = prerank(
                ranks,
                gene_sets=gene_sets,
                max_size=5000
            )
            gsea_res = res.res2d

            return gsea_res

        '''
        ## run enrichr and gsea, celltypes combined - test all km_gene_sets
        rna_enrichr_res_all = do_enrichr(rna_sig_results, rna_sig_results.index.unique().tolist(), km_gene_sets)

        ## run enrichr and gsea for each celltype - test km_gene_sets
        rna_enrichr_res = {celltype: do_enrichr(rna_sig_results, [celltype], km_gene_sets) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running EnrichR on RNA')}
        atac_enrichr_res = {celltype: do_enrichr(atac_sig_results, [celltype], km_gene_sets) for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running EnrichR on ATAC')}
        rna_gsea_res = {celltype: do_gsea(rna_results, [celltype], km_gene_sets) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running GSEA on RNA')}
        atac_gsea_res = {celltype: do_gsea(atac_results, [celltype], km_gene_sets) for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running GSEA on ATAC')}

        ## run enrichr - test ChEA_2022
        rna_enrichr_res_chea = do_enrichr(rna_all.loc[rna_all['padj'] < 0.05], None, 'ChEA_2022')
        #atac_enrichr_res_chea = do_enrichr(atac_all, None, 'ChEA_2022')

        ## run enrichr, celltypes combined - test ChEA_2022
        rna_enrichr_res_chea_all = do_enrichr(rna_sig_results, rna_sig_results.index.unique().tolist(), 'ChEA_2022')
        #rna_gsea_res_chea_all = do_gsea(rna_results, rna_sig_results.index.unique().tolist(), 'ChEA_2022')

        ## run enrichr for each celltype - test ChEA_2022
        rna_enrichr_res_chea_celltypes = {celltype: do_enrichr(rna_sig_results, [celltype], 'ChEA_2022') for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running EnrichR on RNA - ChEA_2022')}
        atac_enrichr_res_chea_celltypes = {celltype: do_enrichr(atac_sig_results, [celltype], 'ChEA_2022') for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running EnrichR on ATAC - ChEA_2022')}
        '''

        ## run enrichr for each celltype - test brainSCOPE TF-TG links
        tf_tg_links = peak_gene_links.groupby('TF')['gene'].unique().to_dict()
        tf_tg_links_with_kms = tf_tg_links.copy()
        tf_tg_links_with_kms.update(km_gene_sets)

        rna_enrichr_res_brainscope = {celltype: do_enrichr(rna_sig_results, [celltype], tf_tg_links_with_kms) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running EnrichR on RNA - brainSCOPE TF-TG links')}  

        N_tops = 10
        rna_enrichr_res_brainscope_tops = {celltype: rna_enrichr_res_brainscope[celltype].sort_values('Combined Score', ascending=False).head(N_tops) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc=f'Extracting top {N_tops} terms from brainSCOPE TF-TG links')}
        rna_enrichr_res_brainscope_tops = {celltype: rna_enrichr_res_brainscope_tops[celltype].assign(log10_1_FDR=rna_enrichr_res_brainscope_tops[celltype].get('Adjusted P-value').apply(lambda x: -np.log10(x))) for celltype in rna_sig_genes_per_celltype.keys()}

        from gseapy import dotplot as gp_dotplot
        from matplotlib_venn import venn2, venn3
        
        female_ExN_TF = female_ExN.loc[female_ExN['TF'].eq('EGR1')]
        female_ExN_TF_genes = set(female_ExN_TF.get('TG').unique())

        merged_eqtl_edges_df = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'eqtl_edges', 'merged_eqtl_edges.csv'))
        merged_eqtl_edges_egr1 = merged_eqtl_edges_df[merged_eqtl_edges_df['GRN.TF'].eq('EGR1')]
        merged_eqtl_edges_egr1_genes = set(merged_eqtl_edges_egr1['GRN.TG'].unique())

        egr1_scompreg_hits_dict = {}

        for celltype in ['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']:

            if celltype not in rna_enrichr_res_brainscope_tops.keys():
                print(f'{celltype} not in rna_enrichr_res_brainscope_tops.keys()')
                continue

            plt.figure()
            gp_dotplot(
                rna_enrichr_res_brainscope_tops[celltype],
                column='log10_1_FDR',
                title=f'EnrichR for {celltype} celltype',
                top_term=10,
                size=20,
                cutoff=1e3, # purposefully too large even for log10(1/FDR)
                cmap='copper',
            )
            plt.show()

            ## Venn diagram between genes overlapping EGR1 and km3 terms
            egr1_genes = set(rna_enrichr_res_brainscope[celltype].loc[rna_enrichr_res_brainscope[celltype]['Term'].eq('EGR1'), 'Genes'].str.split(';').iloc[0])
            km3_genes = set(rna_enrichr_res_brainscope[celltype].loc[rna_enrichr_res_brainscope[celltype]['Term'].eq('km3'), 'Genes'].str.split(';').iloc[0])

            plt.figure()
            venn2([egr1_genes, female_ExN_TF_genes], set_labels=['DE TGs of EGR1', 'sc-compReg'])
            egr1_scompreg_hits = egr1_genes & female_ExN_TF_genes

            plt.figure()
            venn3([egr1_genes, km3_genes, female_ExN_TF_genes], set_labels=['EGR1', 'km3', 'sc-compReg'])
            egr1_km3_scompreg_hits = egr1_genes & km3_genes & female_ExN_TF_genes
            plt.title(f'Threeway hits: {egr1_km3_scompreg_hits}')

            ## GRNs for EGR1 and sc-compReg hits
            egr1_scompreg_hits = list(egr1_genes & female_ExN_TF_genes)
            egr1_scompreg_hits_grn = female_ExN_TF.loc[female_ExN_TF['TG'].isin(egr1_scompreg_hits)]
            assert egr1_scompreg_hits_grn['TF'].eq('EGR1').all()

            ## filter by peaks' p-values (merge egr1_scompreg_hits_grn with pvalues from atac_results)
            egr1_scompreg_hits_grn = egr1_scompreg_hits_grn.merge(atac_results, left_on='enhancer', right_on='gene', how='left')
            egr1_scompreg_hits_grn.dropna(subset='enhancer', inplace=True)
            #egr1_scompreg_hits_grn = egr1_scompreg_hits_grn.loc[egr1_scompreg_hits_grn['pvalue'] < 0.05]

            ## intersect GWAS hits with EGR1 and sc-compReg hits
            #egr1_scompreg_hits_bedtool_intersect_df = intersect_gwas_hits_with_egr1_scompreg_hits(egr1_scompreg_hits_grn, gwas_catalog_metadata)

            
            # Store the intersection results in the dictionary
            egr1_scompreg_hits_dict[celltype] = {
                'egr1_genes': egr1_genes,
                'km3_genes': km3_genes,
                'egr1_scompreg_hits': egr1_scompreg_hits,
                'egr1_scompreg_hits_grn': egr1_scompreg_hits_grn
            }

        ## Concatenate all 'egr1_scompreg_hits_grn' from egr1_scompreg_hits_dict
        egr1_scompreg_hits_grn_all = pd.concat([
            egr1_scompreg_hits_dict[celltype]['egr1_scompreg_hits_grn'].assign(celltype=celltype) for celltype in egr1_scompreg_hits_dict.keys()
            ])

        ## Group results be celltype
        egr1_scompreg_hits_grn_by_celltypes = egr1_scompreg_hits_grn_all.groupby(['enhancer', 'TG'])['celltype'].agg([
            ('n_celltypes', 'nunique'),
            ('celltypes', 'unique')
        ]).sort_values(by='n_celltypes', ascending=False)

        ## Group results by celltype
        hits_by_celltypes = pd.merge(
            pd.get_dummies(egr1_scompreg_hits_grn_all.celltype).assign(RG=False)[['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']],
            egr1_scompreg_hits_grn_all[['TG','enhancer']].apply(lambda x: ' - '.join(x), axis=1).rename('TG - enhancer'),
            left_index=True, right_index=True, how='right'
            ).groupby('TG - enhancer').any()
        assert hits_by_celltypes.index.value_counts().eq(1).all()

        ## sort by number of celltypes for each hit (presort alphabetically)
        hits_by_celltypes.sort_index(inplace=True)
        sort_idxs = pd.DataFrame(np.stack(np.where(hits_by_celltypes)).T).groupby(0).sum().loc[:,1].argsort().values
        hits_by_celltypes = hits_by_celltypes.iloc[sort_idxs]

        ## plot heatmap of hits by celltype
        plt.figure(figsize=(3, 5))
        ax = sns.heatmap(hits_by_celltypes, cmap='viridis', cbar=False, annot=hits_by_celltypes.map(lambda x: 'x' if x else ''), fmt='', 
                         annot_kws={'size': 8, 'weight': 'bold', 'color': 'black'})
        plt.xticks(rotation=45)
        plt.xlabel('Imputed cell-type labels')
        plt.show()

        ## save intermediate results
        output_dir = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis')
        os.makedirs(output_dir, exist_ok=True)
        egr1_scompreg_hits_grn_all.to_csv(os.path.join(output_dir,  'egr1_scompreg_hits_grn_all.csv'), index=False)
        hits_by_celltypes.to_csv(os.path.join(output_dir, 'hits_by_celltypes.csv'), index=False)

        ## preparations for volcano plots
        best_egr1_term = 'EGR1'
        best_nr4a2_term = 'NR4A2'
        best_sox2_term = 'SOX2'
        best_spi1_term = 'SPI1'
        terms_dict = {'EGR1': best_egr1_term, 'NR4A2': best_nr4a2_term, 'SOX2': best_sox2_term, 'SPI1': best_spi1_term}

        """ ## search for EGR1 term in enrichr results
        egr1_terms = rna_enrichr_res_chea_all.loc[rna_enrichr_res_chea_all['Term'].str.contains('EGR1'), 'Term'].unique()
        best_egr1_term = egr1_terms[0]

        ## searchr for NR4A2 term in enrichr results
        nr4a2_terms = rna_enrichr_res_chea_all.loc[rna_enrichr_res_chea_all['Term'].str.contains('NR4A2'), 'Term'].unique()
        best_nr4a2_term = nr4a2_terms[0]
        terms_dict = {'EGR1': best_egr1_term, 'NR4A2': best_nr4a2_term, 'SOX2': best_sox2_term, 'SPI1': best_spi1_term} """


        ## volcano plots - plot term linked genes for RNA or ATAC results
        def plot_term_linked_volcano_plots(TF_name, results_data, term_linked_genes_dict, celltypes, 
                                           output_dir, sex, data_type='RNA', km_gene_sets_mapper=None):
            """
            Create volcano plots for term-linked genes across cell types.
            
            Parameters:
            -----------
            TF_name : str
                Name of the transcription factor
            results_data : DataFrame
                Results data indexed by celltype (RNA or ATAC results)
            term_linked_genes_dict : dict or DataFrame
                Either a dictionary mapping celltypes to term-linked genes/peaks,
                or a DataFrame with celltype index and 'gene'/'peak' columns (enrichment results)
            celltypes : list
                List of celltypes to plot
            output_dir : str
                Base output directory path
            sex : str
                Sex identifier for output filename
            data_type : str, optional
                Type of data being plotted ('RNA' or 'ATAC'), default 'RNA'
            km_gene_sets_mapper : dict, optional
                Mapping of genes to km clusters, if None will skip km annotation
            """
            # Convert enrichment results DataFrame to dict if needed
            if isinstance(term_linked_genes_dict, pd.DataFrame):
                # Extract genes/peaks per celltype from DataFrame
                term_genes_dict = {}
                # For ATAC data, use 'peak' column; for RNA data, use 'gene' column
                feature_col = 'peak' if data_type == 'ATAC' else 'gene'
                
                for celltype in celltypes:
                    if celltype in term_linked_genes_dict.index:
                        celltype_data = term_linked_genes_dict.loc[celltype]
                        # Handle both single row and multiple rows per celltype
                        if isinstance(celltype_data, pd.Series):
                            feature_value = celltype_data.get(feature_col)
                            term_genes_dict[celltype] = [feature_value] if pd.notna(feature_value) else []
                        else:
                            if feature_col in celltype_data.columns:
                                term_genes_dict[celltype] = celltype_data[feature_col].dropna().unique()
                            else:
                                term_genes_dict[celltype] = []
                    else:
                        term_genes_dict[celltype] = []
            else:
                term_genes_dict = term_linked_genes_dict

            fig, ax = plt.subplots(2, 4, figsize=(20, 11), sharex=True, sharey=True)
            plt.suptitle(f"{terms_dict[TF_name]} - {data_type}")

            x_absmax = max(abs(results_data['log2FoldChange'].max()), abs(results_data['log2FoldChange'].min())) + 0.5
            xlims = [x_absmax * -1, x_absmax]

            for c, celltype in enumerate(celltypes):
                results = results_data.loc[celltype].copy()

                if (km_gene_sets_mapper is not None) and (data_type == 'RNA'):
                    results['km'] = results['gene'].map(km_gene_sets_mapper)
                    p_metric = 'padj'

                elif (km_gene_sets_mapper is not None) and (data_type == 'ATAC'):
                    results['km'] = results['gene_linked_to_peak'].map(km_gene_sets_mapper)
                    p_metric = 'pvalue'

                term_genes = term_genes_dict.get(celltype, [])
                volcano_plot(results, term_genes, (ax[0, c], ax[1, c]), p_metric=p_metric)

                ax[0, c].set_title(celltype)
                ax[1, c].set_title(celltype)
                ax[0, c].xlim(xlims)
                ax[1, c].xlim(xlims)

            plt.tight_layout()
            os.makedirs(os.path.join(output_dir, 'pychromVAR', TF_name, sex), exist_ok=True)
            save_path = os.path.join(output_dir, 'pychromVAR', TF_name, sex, f'volcano_plot_{data_type.lower()}_{sex}.png')
            plt.savefig(save_path)
            plt.close()

            print(f'Volcano plot saved to {save_path}')

        ## find term linked genes and peaks
        def find_term_linked_genes_and_peaks(TF_name, rna_results, rna_enrichr_res_celltypes, peak_gene_links):

            term_linked_genes = {}
            term_linked_peaks = {}

            if list(rna_enrichr_res_celltypes.keys()).pop() == 'all':
                term_genes = rna_enrichr_res_celltypes['all'].loc[rna_enrichr_res_celltypes['all']['Term'].eq(terms_dict[TF_name]), 'Genes'].str.split(';').iloc[0]
                peak_gene_link_in_term = (peak_gene_links['TF'].eq(TF_name) & peak_gene_links['gene'].isin(term_genes))
                term_linked_genes['all'] = peak_gene_links.loc[peak_gene_link_in_term, 'gene'].unique()
                term_linked_peaks['all'] = peak_gene_links.loc[peak_gene_link_in_term, 'peak'].unique()

            else:
                for celltype in ['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']:

                    ## extract linked genes and peaks for genes overlapping with term (EnrichR)
                    try:
                        term_genes = rna_enrichr_res_celltypes[celltype].loc[rna_enrichr_res_celltypes[celltype]['Term'].eq(terms_dict[TF_name]), 'Genes'].str.split(';').iloc[0]
                        peak_gene_link_in_term = (peak_gene_links['TF'].eq(TF_name) & peak_gene_links['gene'].isin(term_genes))
                    except:
                        print(f"No term-linked genes found for {celltype} and {terms_dict[TF_name]}")
                        term_genes = []
                        peak_gene_link_in_term = pd.Series([False] * len(peak_gene_links))

                    term_linked_genes[celltype] = peak_gene_links.loc[peak_gene_link_in_term, 'gene'].unique()
                    term_linked_peaks[celltype] = peak_gene_links.loc[peak_gene_link_in_term, 'peak'].unique()

            return term_linked_genes, term_linked_peaks

        ## find term linked genes and peaks for EGR1
        def atac_enrichment_analysis(TF_name, rna_results, rna_enrichr_res_brainscope, peak_gene_links, atac_results, gene_activity_score_type):

            term_linked_genes, term_linked_peaks = find_term_linked_genes_and_peaks(TF_name, rna_results, rna_enrichr_res_brainscope, peak_gene_links)

            ## get female ExN sc-compReg results for TF
            female_ExN_TF = female_ExN.loc[female_ExN['TF'].eq(TF_name)]

            if gene_activity_score_type is not None:
                
                res = run_pychromVAR_case_control(
                    term_linked_genes,
                    term_linked_peaks,
                    peak_gene_links,
                    mdd_atac_broad_sub,
                    sex,
                    genome_fasta_path=os.path.join(os.environ['DATAPATH'], 'hg38.fa'))

                per_pb = res["per_pseudobulk_z"]
                per_pb['set'] = pd.Categorical(per_pb['set'], categories=per_pb['set'].unique(), ordered=True)

            else:

                ## extract results in term
                all_terms_genes = np.unique(np.hstack(list(term_linked_genes.values())))
                all_terms_peaks = np.unique(np.hstack(list(term_linked_peaks.values())))
                results_in_term = atac_results.loc[atac_results['gene'].isin(all_terms_peaks)]

                ## filter results by logFC
                results_in_term_filtered = results_in_term.loc[results_in_term['log2FoldChange'].abs().ge(0.5)]
                results_in_term_filtered = results_in_term_filtered.loc[results_in_term_filtered['gene'].isin(female_ExN_TF['enhancer'])] # remember, 'gene' in this context refers to peaks

                ## FDR correction on filtered results
                padj_term = multipletests(results_in_term_filtered['pvalue'], method='fdr_bh')[1]

                ## assign FDR corrected p-values and significant results
                results_in_term_filtered = results_in_term_filtered.assign(
                    padj_term=padj_term,
                    signif_term=padj_term<0.05,
                    mlog10_pvalue=-np.log10(results_in_term_filtered['pvalue'])
                )
                print(f"{np.sum(results_in_term_filtered['signif_term'])}/{len(results_in_term_filtered)}")

                ## extract significant results and peaks
                sig_results_in_term = results_in_term_filtered.loc[results_in_term_filtered['signif_term']]
                sig_peaks = sig_results_in_term.rename(columns={'gene':'peak'}).get('peak').to_frame()

                ## extract genes linked to significant peaks
                sig_genes_linked_to_peaks = sig_peaks.reset_index().merge(
                    peak_gene_links.loc[
                        peak_gene_links['TF'].eq(TF_name) & \
                        peak_gene_links['gene'].isin(all_terms_genes) & \
                        peak_gene_links['peak'].isin(sig_peaks['peak'])
                        ],
                    left_on='peak', right_on='peak', how='left').set_index('celltype')

                ## ensure that the genes linked to significant peaks are also linked to the term-linked genes for each celltype
                filtered_sig_genes_linked_to_peaks = sig_genes_linked_to_peaks[sig_genes_linked_to_peaks.apply(
                    lambda row: row['peak'] in term_linked_peaks.get(row.name, []) and 
                                row['gene'] in term_linked_genes.get(row.name, []), axis=1
                )]

            ## check overlap between filtered sig genes linked to peaks and sc-compReg results for female ExN EGR1
            #set(sig_genes_linked_to_peaks['gene']) & set(female_ExN_TF['TG'])
            rna_overlap_with_scCompReg = set(all_terms_genes) & set(female_ExN_TF['TG'])
            atac_overlap_with_scCompReg = set(filtered_sig_genes_linked_to_peaks['gene']) & set(female_ExN_TF['TG']) if len(filtered_sig_genes_linked_to_peaks) > 0 else set()

            outputs_dict = {
                'sig_genes_linked_to_peaks': filtered_sig_genes_linked_to_peaks,
                'rna_overlap_with_scCompReg': rna_overlap_with_scCompReg,
                'atac_overlap_with_scCompReg': atac_overlap_with_scCompReg,
                'term_linked_genes': term_linked_genes,
                'term_linked_peaks': term_linked_peaks
            }

            return outputs_dict

        ## find enriched peaks per candidate TF
        egr1_outputs_dict = atac_enrichment_analysis('EGR1', rna_results, rna_enrichr_res_brainscope, peak_gene_links, atac_results, gene_activity_score_type)
        nr4a2_outputs_dict = atac_enrichment_analysis('NR4A2', rna_results, rna_enrichr_res_brainscope, peak_gene_links, atac_results, gene_activity_score_type)
        sox2_outputs_dict = atac_enrichment_analysis('SOX2', rna_results, rna_enrichr_res_brainscope, peak_gene_links, atac_results, gene_activity_score_type)
        spi1_outputs_dict = atac_enrichment_analysis('SPI1', rna_results, rna_enrichr_res_brainscope, peak_gene_links, atac_results, gene_activity_score_type)

        ## Plot volcano plots for RNA results with enrichment-derived genes
        for TF_name, outputs_dict in [
            ('EGR1', egr1_outputs_dict),
            ('NR4A2', nr4a2_outputs_dict),
            ('SOX2', sox2_outputs_dict),
            ('SPI1', spi1_outputs_dict)
        ]:
            plot_term_linked_volcano_plots(
                TF_name=TF_name,
                results_data=rna_results,
                term_linked_genes_dict=outputs_dict['term_linked_genes'],
                celltypes=['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN'] if analyze_per_celltype else ['all'],
                output_dir=os.path.join(os.environ['DATAPATH'], 'mdd_data', 'developmental_analysis'),
                sex=sex,
                data_type='RNA',
                km_gene_sets_mapper=km_gene_sets_mapper
            )

        ## Plot volcano plots for ATAC results with enrichment-derived genes
        for TF_name, outputs_dict in [
            ('EGR1', egr1_outputs_dict),
            ('NR4A2', nr4a2_outputs_dict),
            ('SOX2', sox2_outputs_dict),
            ('SPI1', spi1_outputs_dict)
        ]:

            ## add genes linked to peaks to ATAC results
            all_term_linked_peaks = np.unique(np.hstack(list(outputs_dict['term_linked_peaks'].values())))
            peaks_linked_to_term = peak_gene_links.loc[peak_gene_links['TF'].eq(TF_name) & peak_gene_links['peak'].isin(all_term_linked_peaks)]
            atac_results.loc[:, 'gene_linked_to_peak'] = atac_results['gene'].map(dict(zip(peaks_linked_to_term['peak'], peaks_linked_to_term['gene'])))

            plot_term_linked_volcano_plots(
                TF_name=TF_name,
                results_data=atac_results,
                term_linked_genes_dict=outputs_dict['term_linked_peaks'],
                celltypes=['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN'] if analyze_per_celltype else ['all'],
                output_dir=os.path.join(os.environ['DATAPATH'], 'mdd_data', 'developmental_analysis'),
                sex=sex,
                data_type='ATAC',
                km_gene_sets_mapper=km_gene_sets_mapper
            )


        def ttest_case_control(per_pb, condition_col="condition", value_col="z", set_col="set",
                            case_label="Case", control_label="Control"):
            results = []
            for set_name, subset in per_pb.groupby(set_col):
                case = subset.loc[subset[condition_col] == case_label, value_col].dropna()
                ctrl = subset.loc[subset[condition_col] == control_label, value_col].dropna()
                if len(case) < 2 or len(ctrl) < 2:
                    continue
                t, p = stats.ttest_ind(case, ctrl, equal_var=False)
                pooled_sd = np.sqrt((case.var(ddof=1) + ctrl.var(ddof=1)) / 2)
                cohen_d = (case.mean() - ctrl.mean()) / pooled_sd
                results.append({
                    set_col: set_name,
                    "n_case": len(case),
                    "n_control": len(ctrl),
                    "mean_case": case.mean(),
                    "mean_control": ctrl.mean(),
                    "diff": case.mean() - ctrl.mean(),
                    "t_stat": t,
                    "p_val": p,
                    "cohen_d": cohen_d
                })
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                results_df["q_val"] = multipletests(results_df["p_val"], method="fdr_bh")[1]
            return results_df

        ## t-test for motif
        motif_name = 'MA0160.1.NR4A2'
        for set_name in per_pb['set'].unique():
            set_pb_motif = per_pb[per_pb['set'].eq(set_name) & per_pb['motif'].eq(motif_name)]
            results_df = ttest_case_control(set_pb_motif)
            print(results_df['p_val'])

        plot_pychromvar_case_control_multi_sets(
            per_pb,
            set_names=None,
            motif_name="MA0162.4.EGR1",
            condition_col="condition",
            celltype_col="ClustersMapped",
            ncols=4,
            save_path=os.path.join(os.environ['OUTPATH'], 'pychromVAR', 'EGR1', sex)
        )
        
        # Save enrichr results
        for celltype, res in rna_enrichr_res.items():
            res['sex'] = sex
            res['celltype'] = celltype
            res['modality'] = 'RNA'
            res['method'] = 'enrichr'
            res.to_csv(os.path.join(output_dir, f'enrichr_RNA_{celltype}_{sex}.csv'), index=False)
        
        for celltype, res in atac_enrichr_res.items():
            res['sex'] = sex
            res['celltype'] = celltype
            res['modality'] = 'ATAC'
            res['method'] = 'enrichr'
            res.to_csv(os.path.join(output_dir, f'enrichr_ATAC_{celltype}_{sex}.csv'), index=False)
        
        # Save gsea results
        for celltype, res in rna_gsea_res.items():
            res['sex'] = sex
            res['celltype'] = celltype
            res['modality'] = 'RNA'
            res['method'] = 'gsea'
            res.to_csv(os.path.join(output_dir, f'gsea_RNA_{celltype}_{sex}.csv'), index=False)
        
        for celltype, res in atac_gsea_res.items():
            res['sex'] = sex
            res['celltype'] = celltype
            res['modality'] = 'ATAC'
            res['method'] = 'gsea'
            res.to_csv(os.path.join(output_dir, f'gsea_ATAC_{celltype}_{sex}.csv'), index=False)

    # After both sexes are processed, load all results and create comprehensive plots
    print("\n" + "="*80)
    print("Creating comprehensive enrichment plots...")
    print("="*80)
    
    create_comprehensive_plot(output_dir, 'RNA')
    create_comprehensive_plot(output_dir, 'ATAC')
    create_rna_atac_comparison(output_dir)
    create_enrichr_rna_atac_stacked_heatmap(output_dir)
    
    print("\n" + "="*80)
    print("All plots and analyses complete!")
    print("="*80)


def load_km_gene_sets():
    km_gene_sets = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv'), index_col=0, header=0)
    km_gene_sets = km_gene_sets['clusters.1'].to_dict()
    km_gene_sets = {k: ast.literal_eval(v) for k, v in km_gene_sets.items()} # or else, just a single string that looks like a list of genes

    km_gene_sets_df = pd.DataFrame.from_dict(km_gene_sets, orient='index').unstack().reset_index().set_index(0).drop(columns=['level_0']).rename(columns={'level_1': 'km'})
    km_gene_sets_mapper = km_gene_sets_df.to_dict().get('km')
    return km_gene_sets, km_gene_sets_mapper

def get_peak_gene_links():

    ## get peak names mapper and reverse mapper
    peak_names_mapper = pd.read_pickle(os.path.join(os.environ['OUTPATH'], 'peak_names_mapper.pkl')) # derived from female data
    peak_names_mapper_reverse = {v: k for k, v in peak_names_mapper.items()}

    ## get peak-gene links
    mean_grn_df_filtered = pd.read_csv(os.path.join(os.environ['OUTPATH'], 'mean_grn_df_filtered.csv'))
    peak_gene_links = mean_grn_df_filtered.copy()
    peak_gene_links['peak'] = peak_gene_links['enhancer'].map(peak_names_mapper_reverse)
    peak_gene_links.rename(columns={'TG': 'gene'}, inplace=True)

    ## get sc-compReg results for female ExN
    female_ExN = pd.read_csv(os.path.join(os.environ['OUTPATH'], 'enrichment_analyses_16103846_41', 'mean_grn_df_filtered_female_ExN.csv'))
    female_ExN['enhancer'] = female_ExN['enhancer'].map(peak_names_mapper_reverse)

    return peak_gene_links, female_ExN, peak_names_mapper_reverse

def get_gwas_catalogue_hits():
    gwas_catalogue_hits = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'gwas-catalog-download-associations-v1.0-full.tsv'), sep='\t')

    depressive_traits = ['depressive disorder', 'postpartum depression', 'treatment resistant depression', 'major depressive disorder', 'mixed anxiety and depressive disorder', 'age of onset of depressive disorder', 'psychotic symptom measurement', 'major depressive episode']
    #mdd_gwas_catalogue_hits = gwas_catalogue_hits.loc[gwas_catalogue_hits['DISEASE/TRAIT'].fillna('').str.lower().isin(depressive_traits)]
    mdd_gwas_catalogue_hits = gwas_catalogue_hits

    # Split semicolon-separated values and explode to separate rows
    mdd_gwas_exploded = mdd_gwas_catalogue_hits.copy()

    # Split CHR_ID if it also has semicolons
    if 'CHR_ID' in mdd_gwas_exploded.columns:
        mdd_gwas_exploded['CHR_ID'] = mdd_gwas_exploded['CHR_ID'].astype(str).str.split(';')
        mdd_gwas_exploded = mdd_gwas_exploded.explode('CHR_ID')

    # Split and explode CHR_POS
    if 'CHR_POS' in mdd_gwas_exploded.columns:
        mdd_gwas_exploded['CHR_POS'] = mdd_gwas_exploded['CHR_POS'].astype(str).str.split(';')
        mdd_gwas_exploded = mdd_gwas_exploded.explode('CHR_POS')
        
    # Clean up CHR_POS: convert to numeric (coercing errors to NaN)
    mdd_gwas_exploded['CHR_POS'] = pd.to_numeric(mdd_gwas_exploded['CHR_POS'], errors='coerce')

    # Remove NaN values and empty strings
    mdd_gwas_exploded = mdd_gwas_exploded.dropna(subset=['CHR_ID', 'CHR_POS'])
    mdd_gwas_exploded = mdd_gwas_exploded[mdd_gwas_exploded['CHR_ID'] != '']

    # Reset index to track which row is which
    mdd_gwas_exploded = mdd_gwas_exploded.reset_index(drop=True)

    # Create BED format dataframe WITH metadata preservation
    gwas_catalog_bed_df = pd.DataFrame({
        'chr': 'chr' + mdd_gwas_exploded['CHR_ID'].astype(str),
        'start': mdd_gwas_exploded['CHR_POS'].astype(int),
        'end': mdd_gwas_exploded['CHR_POS'].astype(int) + 1,  # 1bp interval for SNP
        'snp_id': mdd_gwas_exploded.index,  # Track the index to map back later
        'trait': mdd_gwas_exploded['DISEASE/TRAIT'],  # Add trait info
        'rsid': mdd_gwas_exploded.get('SNPS', '')  # Add SNP RS ID if available
    })

    # Remove any remaining invalid entries
    gwas_catalog_bed_df = gwas_catalog_bed_df.dropna(subset=['chr', 'start', 'end'])

    # Store the full metadata for later mapping
    gwas_catalog_metadata = mdd_gwas_exploded.copy()

    # Create BedTool (will use first 3 columns as BED format)
    gwas_catalog_bedtool = BedTool.from_dataframe(gwas_catalog_bed_df)

    return gwas_catalog_bedtool, gwas_catalog_metadata

def subset_data(adata, gwas_hits, subset_type, hvg_genes=None):

    if subset_type == 'gwas':

        ## subset genes to those in GWAS hits
        genes_in_gwas_hits_bool = adata.var_names.isin(gwas_hits['Target gene name'].unique())
        adata = adata[:, genes_in_gwas_hits_bool]

        ## filter raw data to only include genes in full data
        raw_genes_in_gwas_hits_bool = adata.raw.var_names.isin(gwas_hits['Target gene name'].unique())
        filtered_raw_X = adata.raw.X[:, raw_genes_in_gwas_hits_bool]
        filtered_raw_var = adata.raw.var.loc[raw_genes_in_gwas_hits_bool].copy()
        raw_adata = anndata.AnnData(X=filtered_raw_X, var=filtered_raw_var)
        adata.raw = raw_adata

    elif subset_type == 'gwas_and_hvgs':

        ## get list of genes in GWAS hits and highly variable genes
        hvg_genes = adata.var_names.tolist() if hvg_genes is None else hvg_genes
        gwas_and_hvgs_genes = gwas_hits['Target gene name'].unique().tolist() + hvg_genes

        ## subset data to only include genes in GWAS hits and highly variable genes
        gwas_and_hvgs_genes_bool = adata.var_names.isin(gwas_and_hvgs_genes)
        adata = adata[:, gwas_and_hvgs_genes_bool]

        ## filter raw data to only include genes in GWAS hits and highly variable genes
        raw_genes_in_gwas_and_hvgs_bool = adata.raw.var_names.isin(gwas_and_hvgs_genes)
        filtered_raw_X = adata.raw.X[:, raw_genes_in_gwas_and_hvgs_bool]
        filtered_raw_var = adata.raw.var.loc[raw_genes_in_gwas_and_hvgs_bool].copy()
        raw_adata = anndata.AnnData(X=filtered_raw_X, var=filtered_raw_var)
        adata.raw = raw_adata

    elif subset_type is None:
        pass

    return adata

def concat_results(per_celltype, genes):
    results = pd.concat([
        df.assign(celltype=celltype, gene=genes).assign(
            km=genes.map(gwas_hits.set_index('Target gene name')['km'].to_dict()),
            traits=genes.map(gwas_hits.groupby('Target gene name')['Trait'].unique().str.join('_').to_dict())
            ).set_index('celltype') 
        for celltype, df in per_celltype.items() if df is not None
    ])

    return results

def extract_results(results, p_metric='pvalue'):
    significant_results = results[results[p_metric] < 0.05]
    mdd_significant_results = significant_results.loc[significant_results.traits.str.contains('MDD').fillna(False)]
    sig_genes_per_celltype = significant_results.groupby('celltype')['gene'].apply(np.sort).to_dict()
    sig_km_per_celltype = significant_results.dropna(subset=['km']).groupby('celltype')['km'].unique().to_dict()
    return significant_results, mdd_significant_results, sig_genes_per_celltype, sig_km_per_celltype
    
def run_pyDESeq2_on_celltypes(adata, sex, test_type='split', max_min_n_cells=20, 
                              max_imbalance_ratio=2.0, N_max_cells_per_donor=300, random_state=42):
    """
    Run pyDESeq2 on cell types with donor-level replicates, prioritizing power.
    
    This implementation uses a "power-tilted" approach:
    - Uses as many donors as possible per cell type (after quality filtering)
    - Only trims donors when imbalance between conditions is extreme (ratio > max_imbalance_ratio)
    - Allows richer cell types to contribute more signal via higher N_max
    
    Parameters
    ----------
    adata : AnnData
        Input data
    sex : str
        Sex to analyze
    test_type : str
        Type of test ('split' or 'all')
    max_min_n_cells : int
        Minimum cells per donor to include (default: 20 for quality)
    max_imbalance_ratio : float
        Maximum allowed ratio between conditions before trimming (default: 2.0)
    N_max_cells_per_donor : int or None
        Maximum cells to sample per donor (default: 200). Set to None for no cap.
    random_state : int
        Random seed for reproducibility
    """
    
    np.random.seed(random_state)
    
    all_mdd_subjects_counts_adata = []
    all_counts = []
    all_metadata = []

    ## loop through celltypes and get pseudo-replicates counts
    for celltype in ['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']:

        print(f"\n{'='*80}")
        print(f"Processing {celltype} - {sex}")
        print(f"{'='*80}")

        mdd_subjects_counts_adata, counts, metadata = get_pseudo_replicates_counts(
            sex, celltype, adata, adata.raw.var.copy(), 
            'most_common_cluster', 'Condition', 'Sex', 'OriginalSub',
            pseudo_replicates='Subjects', overlapping_only=False
        )

        ## STEP 1: Drop donors with < N_min cells
        min_n_cells = max_min_n_cells
        keep_pseudobulk = metadata['n_cells'].astype(int).ge(min_n_cells)
        
        print(f"\nStep 1 - Filtering donors with < {min_n_cells} cells:")
        print(f"  Before: {len(metadata)} donors")
        
        mdd_subjects_counts_adata = mdd_subjects_counts_adata[keep_pseudobulk]
        counts = counts[keep_pseudobulk.values]
        metadata = metadata[keep_pseudobulk.values]
        
        print(f"  After: {len(metadata)} donors")
        
        ## STEP 2: Use as many donors as possible, with mild balancing
        n_ctrl = (metadata['Condition'] == 'Control').sum()
        n_case = (metadata['Condition'] == 'Case').sum()
        
        print(f"\nStep 2 - Determining donor counts (power-tilted approach):")
        print(f"  Available donors - Control: {n_ctrl}, Case: {n_case}")
        
        # Check for minimum donors
        if n_ctrl < 2 or n_case < 2:
            print(f"  WARNING: Insufficient donors for {celltype}. Skipping.")
            continue
        
        # Calculate imbalance ratio
        imbalance_ratio = max(n_ctrl, n_case) / min(n_ctrl, n_case)
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        # Decide whether to trim based on imbalance
        if imbalance_ratio <= max_imbalance_ratio:
            # Mild imbalance is OK - use all donors
            K_ctrl = n_ctrl
            K_case = n_case
            print(f"  Imbalance is acceptable (≤ {max_imbalance_ratio:.1f}) - using all donors")
        else:
            # Extreme imbalance - trim majority to at most 2x minority
            K_min = min(n_ctrl, n_case)
            K_max = int(max_imbalance_ratio * K_min)
            
            if n_ctrl > n_case:
                K_ctrl = K_max
                K_case = n_case
                print(f"  Trimming Control donors from {n_ctrl} to {K_ctrl} (max {max_imbalance_ratio:.1f}× minority)")
            else:
                K_ctrl = n_ctrl
                K_case = K_max
                print(f"  Trimming Case donors from {n_case} to {K_case} (max {max_imbalance_ratio:.1f}× minority)")
        
        # Select donors from each condition
        ctrl_indices = np.where(metadata['Condition'] == 'Control')[0]
        case_indices = np.where(metadata['Condition'] == 'Case')[0]
        
        # Select top K donors by cell count from each condition
        ctrl_n_cells = metadata.iloc[ctrl_indices]['n_cells'].astype(int).values
        case_n_cells = metadata.iloc[case_indices]['n_cells'].astype(int).values
        
        selected_ctrl_idx = ctrl_indices[np.argsort(ctrl_n_cells)[-K_ctrl:]]
        selected_case_idx = case_indices[np.argsort(case_n_cells)[-K_case:]]
        
        selected_idx = np.concatenate([selected_ctrl_idx, selected_case_idx])
        
        mdd_subjects_counts_adata = mdd_subjects_counts_adata[selected_idx]
        counts = counts[selected_idx]
        metadata = metadata.iloc[selected_idx]
        
        print(f"  Final: {len(metadata)} donors total ({K_ctrl} Control + {K_case} Case)")
        
        ## STEP 3: Optionally downsample very deep donors
        if N_max_cells_per_donor is not None:
            print(f"\nStep 3 - Downsampling donors with > {N_max_cells_per_donor} cells:")
            print(f"  Cell counts per donor before downsampling:")
            print(f"    Median: {metadata['n_cells'].astype(int).median()}")
            print(f"    Range: [{metadata['n_cells'].astype(int).min()}, {metadata['n_cells'].astype(int).max()}]")
            
            n_downsampled = (metadata['n_cells'].astype(int) > N_max_cells_per_donor).sum()
            print(f"  Donors to downsample: {n_downsampled}/{len(metadata)}")
        else:
            print(f"\nStep 3 - No downsampling (N_max=None):")
            print(f"  Cell counts per donor:")
            print(f"    Median: {metadata['n_cells'].astype(int).median()}")
            print(f"    Range: [{metadata['n_cells'].astype(int).min()}, {metadata['n_cells'].astype(int).max()}]")
        
        # This requires going back to the original data to resample cells (if needed)
        downsampled_counts_list = []
        downsampled_metadata_list = []
        
        for idx, (subject, row) in enumerate(metadata.iterrows()):
            n_cells_donor = int(row['n_cells'])
            
            # Get cells for this donor
            donor_mask = (adata.obs['OriginalSub'] == subject) & \
                         (adata.obs['most_common_cluster'].str.startswith(celltype)) & \
                         (adata.obs['Sex'].str.lower() == sex.lower())
            donor_cell_indices = np.where(donor_mask)[0]
            
            # Downsample only if N_max is set and donor exceeds it
            if N_max_cells_per_donor is not None and n_cells_donor > N_max_cells_per_donor:
                sampled_cell_indices = np.random.choice(
                    donor_cell_indices, 
                    size=N_max_cells_per_donor, 
                    replace=False
                )
                actual_n_cells = N_max_cells_per_donor
            else:
                sampled_cell_indices = donor_cell_indices
                actual_n_cells = n_cells_donor
            
            # Sum counts across sampled cells
            donor_counts = adata.raw.X[sampled_cell_indices].sum(axis=0).A1.astype(int)
            downsampled_counts_list.append(donor_counts)
            
            # Update metadata
            row_updated = row.copy()
            row_updated['n_cells'] = actual_n_cells
            downsampled_metadata_list.append(row_updated)
        
        # Reconstruct arrays
        counts = np.vstack(downsampled_counts_list)
        metadata = pd.DataFrame(downsampled_metadata_list)
        
        # Reconstruct AnnData
        mdd_subjects_counts_adata = anndata.AnnData(
            X=counts,
            var=adata.raw.var.copy(),
            obs=metadata
        )
        
        if N_max_cells_per_donor is not None:
            print(f"  Cell counts per donor after downsampling:")
            print(f"    Median: {metadata['n_cells'].astype(int).median()}")
            print(f"    Range: [{metadata['n_cells'].astype(int).min()}, {metadata['n_cells'].astype(int).max()}]")

        all_mdd_subjects_counts_adata.append(mdd_subjects_counts_adata)
        all_counts.append(counts)
        all_metadata.append(metadata)

    ## Print summary statistics across all cell types
    print(f"\n{'='*80}")
    print(f"SUMMARY - Power-tilted donor counts across cell types ({sex})")
    print(f"{'='*80}")
    print(f"{'Cell Type':<20} {'Ctrl donors':<15} {'Case donors':<15} {'Median cells':<15} {'Range':<20}")
    print(f"{'-'*80}")
    
    for celltype_data in all_metadata:
        ct_name = celltype_data['most_common_cluster'].iloc[0]
        n_ctrl = (celltype_data['Condition'] == 'Control').sum()
        n_case = (celltype_data['Condition'] == 'Case').sum()
        median_cells = celltype_data['n_cells'].astype(int).median()
        min_cells = celltype_data['n_cells'].astype(int).min()
        max_cells = celltype_data['n_cells'].astype(int).max()
        print(f"{ct_name:<20} {n_ctrl:<15} {n_case:<15} {median_cells:<15.0f} [{min_cells}-{max_cells}]")
    print(f"{'='*80}")
    print(f"Note: Imbalance ratio capped at {max_imbalance_ratio}×, N_min={max_min_n_cells}, N_max={N_max_cells_per_donor}")
    print(f"{'='*80}\n")

    ## concatenate
    if len(all_mdd_subjects_counts_adata) == 0:
        raise ValueError(f"No cell types had sufficient donors for {sex}")
    
    mdd_subjects_counts_adata = anndata.concat(all_mdd_subjects_counts_adata, axis=0)
    counts = np.concatenate(all_counts, axis=0)
    metadata = pd.concat(all_metadata, axis=0)

    ## remove features with less than 100 counts
    peak_counts = np.array(counts.sum(axis=0)).flatten()
    mdd_subjects_counts_adata = mdd_subjects_counts_adata[:, peak_counts > 100]
    counts = counts[:, peak_counts > 100]

    ## check if all genes have at least one zero-count (used in pyDESeq2 to switch to iterative fitting)
    if (counts==0).any(0).all():
        print("All genes have at least one zero-count, setting size factors fit type to 'poscounts'")
        sf_type = 'poscounts'
    else:
        print("Not all genes have at least one zero-count, keeping size factors fit type as 'poscounts'")
        sf_type = 'poscounts'

    ## run pyDESeq2
    #per_celltype = run_pyDESeq2_per_celltype(counts, metadata, 'most_common_cluster')
    if test_type == 'split':
        per_celltype = run_pyDESeq2_contrasts(counts, metadata, 'Condition', sf_type=sf_type)
    elif test_type == 'all':
        per_celltype = run_pyDESeq2_all_celltypes(counts, metadata, sf_type=sf_type)

    return per_celltype, mdd_subjects_counts_adata.var

## functions for pyDESeq2 on counts data
def get_pseudo_replicates_counts(sex, celltype, rna_scaled_with_counts, mdd_rna_var, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, pseudo_replicates='Subjects', overlapping_only=False):

    if pseudo_replicates == 'SEACells':
        ## learn SEACell assignments based on processed data to apply onto counts data

        mdd_seacells_counts_dict = {}

        unique_conditions = rna_scaled_with_counts.obs[rna_condition_key].unique().tolist()
        for condition in unique_conditions:

            ## select cell indices
            rna_indices = pd.DataFrame({
                'is_celltype': rna_scaled_with_counts.obs[rna_celltype_key].str.startswith(celltype),
                'is_condition': rna_scaled_with_counts.obs[rna_condition_key].str.startswith(condition),
                'is_sex': rna_scaled_with_counts.obs[rna_sex_key].str.lower().str.contains(sex.lower()),
            }).prod(axis=1).astype(bool).values.nonzero()[0]

            rna_sampled = rna_scaled_with_counts[rna_indices]

            ## learn SEACell assignments for this condition
            seacells_model = SEACells.core.SEACells(
                rna_sampled, 
                build_kernel_on='X_pca', # could also opt for batch-corrected PCA (Harmony), but ok if pseudo-bulk within batch
                n_SEACells=max(rna_sampled.n_obs // 50, 15), 
                n_waypoint_eigs=15,
                convergence_epsilon = 1e-5,
                max_franke_wolfe_iters=100,
                use_gpu=True if torch.cuda.is_available() else False
                )
            
            seacells_model.construct_kernel_matrix()
            seacells_model.initialize_archetypes()
            seacells_model.fit(min_iter=10, max_iter=100)

            ## create mdd_rna_counts anndata using raw counts
            mdd_rna_counts = anndata.AnnData(
                X=rna_sampled.raw.X.toarray(),
                var=rna_sampled.raw.var.copy(),
                obs=rna_sampled.obs.copy(),
            )

            ## summarize counts data by SEACell - remove 'nan' obs_names which groups cells not corresponding to celltype or sex
            mdd_rna_counts.obs.loc[rna_sampled.obs_names,'SEACell'] = rna_sampled.obs['SEACell'].add(f'_{condition}_{sex}_{celltype}')
            mdd_seacells_counts = SEACells.core.summarize_by_SEACell(mdd_rna_counts, SEACells_label='SEACell', summarize_layer='X')
            mdd_seacells_counts = mdd_seacells_counts[mdd_seacells_counts.obs_names != 'nan']

            mdd_seacells_counts.obs[rna_condition_key] = condition
            mdd_seacells_counts.obs[rna_sex_key] = sex
            mdd_seacells_counts.obs[rna_celltype_key] = celltype

            batch = rna_sampled.obs['Batch'].value_counts(normalize=True).astype(str).reset_index().values.flatten().tolist()
            batch = '_'.join(batch)
            mdd_seacells_counts.obs['Batch'] = batch

            mdd_seacells_counts_dict[condition] = mdd_seacells_counts

        ## concatenate all SEACell counts data across conditions
        mdd_seacells_counts_adata = anndata.concat(mdd_seacells_counts_dict.values(), axis=0)
        mdd_seacells_counts_adata = mdd_seacells_counts_adata[:, mdd_seacells_counts_adata.var_names.isin(mdd_rna_var.index)]
        mdd_seacells_counts_adata.var = mdd_rna_var

        counts = mdd_seacells_counts_adata.X.astype(int).toarray()
        metadata = mdd_seacells_counts_adata.obs
        mdd_counts_adata = mdd_seacells_counts_adata

    elif pseudo_replicates == 'Subjects':

        subjects_from_sex = rna_scaled_with_counts.obs[rna_scaled_with_counts.obs[rna_sex_key].str.lower() == sex.lower()][rna_subject_key].unique()
        if overlapping_only:
            subjects_from_sex = subjects_from_sex[np.isin(subjects_from_sex, [os.split('_')[-1] for os in overlapping_subjects])]

        mdd_subjects_counts_dict = {}

        for subject in subjects_from_sex:

            rna_indices =  pd.DataFrame({
                'is_subject': rna_scaled_with_counts.obs[rna_subject_key] == subject,
                'is_celltype': rna_scaled_with_counts.obs[rna_celltype_key].str.startswith(celltype)
            })
            rna_indices = rna_indices.prod(axis=1).astype(bool).values.nonzero()[0]

            rna_sampled = rna_scaled_with_counts[rna_indices]
            rna_subject_counts = rna_sampled.raw.X.sum(axis=0).A1.astype(int)
            rna_subject_var = rna_sampled.raw.var

            subject_condition = rna_scaled_with_counts.obs[rna_condition_key][rna_scaled_with_counts.obs[rna_subject_key] == subject].iloc[0]
            batch = rna_scaled_with_counts.obs['Batch'][rna_scaled_with_counts.obs[rna_subject_key] == subject].iloc[0]
            n_cells = len(rna_indices)

            rna_subject_obs = pd.DataFrame(
                np.hstack([batch, subject_condition, sex, celltype, n_cells]).reshape(1, -1),
                columns=['Batch', rna_condition_key, rna_sex_key, rna_celltype_key, 'n_cells'],
                index=[subject],
            )

            rna_subject_counts_ad = anndata.AnnData(
                X=rna_subject_counts.reshape(1, -1),
                var=rna_subject_var,
                obs=rna_subject_obs,
            )
            mdd_subjects_counts_dict[subject] = rna_subject_counts_ad

        mdd_subjects_counts_adata = anndata.concat(mdd_subjects_counts_dict.values(), axis=0)
        mdd_subjects_counts_adata = mdd_subjects_counts_adata[:, mdd_subjects_counts_adata.var_names.isin(mdd_rna_var.index)]
        mdd_subjects_counts_adata.var = mdd_rna_var

        counts = mdd_subjects_counts_adata.X.astype(int)#.toarray()
        metadata = mdd_subjects_counts_adata.obs
        mdd_counts_adata = mdd_subjects_counts_adata

    return mdd_counts_adata, counts, metadata

def run_pyDESeq2_per_celltype(counts, metadata, rna_celltype_key, save_dir=None):

    inference = DefaultInference(n_cpus=8)

    per_celltype = {}
    for celltype in tqdm(metadata[rna_celltype_key].unique(), desc='Running pyDESeq2 per celltype'):

        is_celltype = metadata[rna_celltype_key].eq(celltype)
        metadata_celltype = metadata[is_celltype]
        counts_celltype = counts[is_celltype]

        dds = DeseqDataSet(
            counts=counts_celltype,
            metadata=metadata_celltype,
            design="~ Batch + Condition",
            refit_cooks=True,
            inference=inference
        )
        dds.deseq2()

        stat = DeseqStats(dds, contrast=("Condition", "Case", "Control"))
        stat.run_wald_test()
        stat.summary()
        results_celltype = stat.results_df

        per_celltype[celltype] = results_celltype

    return per_celltype

def run_pyDESeq2_contrasts(counts, metadata, rna_condition_key, sf_type='ratio', save_dir=None):

    inference = DefaultInference(n_cpus=8)

    cell_col = "most_common_cluster"
    cond_col = rna_condition_key     # ["Control","Case"]

    md = metadata.copy()
    md[cell_col] = pd.Categorical(md[cell_col])
    md[cond_col] = pd.Categorical(md[cond_col], categories=["Control","Case"], ordered=True)

    # 1) Make a combined factor: "<cell>_<cond>"
    md["group"] = md[cell_col].astype(str) + "_" + md[cond_col].astype(str)
    # Optional: set a stable category order (first level is the base)
    group_levels = sorted(md["group"].unique())
    md["group"] = pd.Categorical(md["group"], categories=group_levels, ordered=True)

    # 2) Single fit with Batch + group
    dds = DeseqDataSet(
        counts=counts,
        metadata=md,
        design="~ Batch + group",
        refit_cooks=True,
        inference=inference,
        size_factors_fit_type=sf_type
    )
    dds.deseq2()

    # 3) For each cell type, test Case vs Control *within that cell*
    # If your pyDESeq2 supports named contrasts (factor, numerator, denominator):
    def run_named_contrast(level_case, level_ctrl):
        stat = DeseqStats(dds, contrast=("group", level_case, level_ctrl))
        stat.run_wald_test()
        stat.summary()
        return stat.results_df

    per_celltype = {}
    for ct in md[cell_col].cat.categories:
        level_case = f"{ct}_Case"
        level_ctrl = f"{ct}_Control"
        if (level_case in group_levels) and (level_ctrl in group_levels):
            per_celltype[ct] = run_named_contrast(level_case, level_ctrl)
        else:
            # one of the levels missing => that cell type lacks one condition
            per_celltype[ct] = None

    return per_celltype

def run_pyDESeq2_all_celltypes(counts, metadata, sf_type='ratio'):

    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~ Batch + Condition",
        refit_cooks=True,
        inference=inference,
        size_factors_fit_type=sf_type
    )
    dds.deseq2()

    stat = DeseqStats(dds, contrast=("Condition", "Case", "Control"))
    stat.run_wald_test()
    stat.summary()
    results = stat.results_df

    return results



# ---------- your existing helpers (lightly adapted) ----------

def build_mask_from_intervals(var_names, intervals_np):
    idx = {p: i for i, p in enumerate(var_names)}
    hits = [idx[s] for s in map(str, intervals_np) if s in idx]
    mask = np.zeros(len(var_names), dtype=bool)
    if hits:
        mask[np.asarray(hits, dtype=int)] = True
    return mask, len(intervals_np), len(hits)

def deviations_to_df(dev_adata, row_name="sample", col_name="annotation"):
    df = pd.DataFrame(
        dev_adata.X,
        index=getattr(dev_adata, "obs_names", None),
        columns=getattr(dev_adata, "var_names", None),
    )
    df.index.name = row_name
    df.columns.name = col_name
    return df

def stouffer(z, weights=None):
    z = np.asarray(pd.to_numeric(pd.Series(z), errors="coerce").dropna())
    if z.size == 0:
        return np.nan
    if weights is None:
        weights = np.ones_like(z)
    else:
        weights = np.asarray(weights)[: z.size]
    num = np.sum(weights * z)
    den = np.sqrt(np.sum(weights**2))
    return num / den if den > 0 else np.nan


# ---------- main function with critical changes ----------

def run_pychromVAR_case_control(
    term_linked_genes,          # dict: {celltype -> ...} (you use keys(celltype) to slice columns later)
    term_linked_peaks,          # dict: {celltype or set_name -> np.array of "chr:start-end"}
    peak_gene_links,            # not directly used below, but keep if needed
    mdd_atac_broad_sub,         # AnnData (cells x peaks), with .obs['sex'], .obs['condition'], .obs['ClustersMapped'], .obs['BrainID']
    sex,
    genome_fasta_path,          # hg38 fasta path
    n_jobs=8,
    condition_col="condition",
    case_label="Case",
    control_label="Control",
    min_cells_per_pseudobulk=50,
):
    import pychromvar as pc
    from pyjaspar import jaspardb

    # ----------------------------
    # 1) Filter by sex, build pseudobulks ONCE
    # ----------------------------
    sex_mask = mdd_atac_broad_sub.obs['sex'].str.lower() == sex.lower()
    adata = mdd_atac_broad_sub[sex_mask].copy()

    # pseudobulk by celltype x donor
    meta = adata.obs[['ClustersMapped','BrainID', condition_col, 'sex']].copy()
    meta['celltype_donor'] = meta[['ClustersMapped','BrainID']].agg(lambda x: f"{x[0]}_{x[1]}", axis=1)

    groups = pd.get_dummies(meta['celltype_donor'], sparse=False).astype(int)
    pb_counts = adata.X.T.dot(groups.values)  # peaks x pseudobulks
    pb_counts = pb_counts.T                   # pseudobulks x peaks

    pb = anndata.AnnData(X=pb_counts, obs=groups.columns.to_frame(), var=adata.var)
    pb.obs['n_cells'] = groups.sum(axis=0)
    # retain metadata for those pseudobulks
    meta_pb = meta.drop_duplicates(subset='celltype_donor').set_index('celltype_donor').loc[pb.obs.index]
    pb.obs = pd.concat([pb.obs, meta_pb], axis=1)

    # filter small pseudobulks and empty peaks
    pb = pb[pb.obs['n_cells'] >= min_cells_per_pseudobulk].copy()
    pb = pb[:, np.asarray(pb.X.sum(axis=0)).ravel() > 0].copy()

    if pb.n_obs < 2:
        raise ValueError("Not enough pseudobulks after filtering.")

    # ----------------------------
    # 2) Add sequences, GC bias, backgrounds ONCE on ALL pseudobulks
    # ----------------------------
    pc.add_peak_seq(pb, genome_file=genome_fasta_path, delimiter=':|-')
    pc.add_gc_bias(pb)
    pc.get_bg_peaks(pb)  # matched backgrounds stored in pb.uns["bg_peaks"]

    # ----------------------------
    # 3) Fetch motifs and match ONCE
    # ----------------------------
    jdb = jaspardb(release='JASPAR2020')
    motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'])
    pc.match_motif(pb, motifs=motifs)  # creates pb.varm["motif_match"], pb.uns["motif_name"]

    orig_motif_match = pb.varm["motif_match"].copy()
    orig_motif_names = np.asarray(pb.uns["motif_name"]).copy()

    # ----------------------------
    # 4) For each peak set (e.g., per celltype), compute deviations ONCE on ALL pseudobulks
    #    (Critical change: we do NOT split by condition before compute_deviations)
    # ----------------------------
    all_results = []  # rows of per-pseudobulk Z; we’ll test later
    overlap_registry = {}

    for set_name, ivals in term_linked_peaks.items():
        # mask peaks for this set
        mask, n_in, n_hit = build_mask_from_intervals(pb.var_names, ivals)
        if n_hit == 0:
            print(f"[warn] {set_name}: 0/{n_in} peaks matched — skipping")
            continue

        # restrict motif matching to peaks in this set
        M_subset = orig_motif_match * mask[:, None]
        keep = np.asarray(M_subset.sum(axis=0)).ravel() > 0
        if keep.sum() == 0:
            print(f"[warn] {set_name}: no motifs overlap — skipping")
            continue
        M_subset = M_subset[:, keep]
        names_subset = orig_motif_names[keep].astype(object)

        # swap in and compute deviations ONCE across all pseudobulks
        pb.varm["motif_match"] = M_subset
        pb.uns["motif_name"] = names_subset

        dev = pc.compute_deviations(pb, n_jobs=n_jobs)  # SAME background/statistics for all samples

        df_dev = deviations_to_df(dev, row_name="pseudobulk", col_name="motif")
        # annotate multi-indexed columns to carry set_name
        df_dev.columns = pd.MultiIndex.from_product([[set_name], df_dev.columns], names=["set", "motif"])

        # record per-motif overlap size (for later reporting)
        overlap = np.asarray(M_subset.sum(axis=0)).ravel()
        overlap_registry[set_name] = pd.Series(overlap, index=df_dev.columns.get_level_values("motif"))

        # collect long-form with metadata for testing
        long = df_dev.stack(level=["set", "motif"]).to_frame("z").reset_index()
        # attach metadata: condition, celltype, donor, etc.
        meta_cols = [condition_col, "ClustersMapped", "BrainID", "sex", "n_cells"]
        long = long.merge(pb.obs[meta_cols], left_on="pseudobulk", right_index=True, how="left")
        all_results.append(long)

    # restore originals (tidy)
    pb.varm["motif_match"] = orig_motif_match
    pb.uns["motif_name"] = orig_motif_names

    if not all_results:
        raise ValueError("No deviations computed for any set.")

    dev_long = pd.concat(all_results, ignore_index=True)

    # ----------------------------
    # 5) Case vs control testing per (set, motif) — replicate-level inference
    # ----------------------------
    # Keep only desired labels
    dev_long = dev_long[dev_long[condition_col].isin([case_label, control_label])].copy()

    def welch_test(group):
        # group: rows for one (set, motif)
        x = pd.to_numeric(group.loc[group[condition_col] == case_label, "z"], errors="coerce").dropna().values
        y = pd.to_numeric(group.loc[group[condition_col] == control_label, "z"], errors="coerce").dropna().values
        out = {
            "n_case": x.size,
            "n_ctrl": y.size,
            "mean_case": float(np.mean(x)) if x.size else np.nan,
            "mean_ctrl": float(np.mean(y)) if y.size else np.nan,
            "diff_mean": float(np.mean(x) - np.mean(y)) if (x.size and y.size) else np.nan,
            "t_stat": np.nan,
            "p_welch": np.nan,
            "stouffer_case": stouffer(x) if x.size else np.nan,
            "stouffer_ctrl": stouffer(y) if y.size else np.nan,
        }
        if x.size >= 2 and y.size >= 2:
            t, p = stats.ttest_ind(x, y, equal_var=False)
            out.update(t_stat=float(t), p_welch=float(p))
        return pd.Series(out)

    tests = (
        dev_long
        .groupby(["set", "motif"], sort=False)
        .apply(welch_test)
        .reset_index()
    )

    # multiple testing over all (set, motif) combinations
    tests["q_welch"] = np.nan
    mask = tests["p_welch"].notna()
    if mask.sum():
        tests.loc[mask, "q_welch"] = multipletests(tests.loc[mask, "p_welch"], method="fdr_bh")[1]

    # ----------------------------
    # 6) (Optional) OLS with covariates on Z ~ condition + covars
    # ----------------------------
    # You can add e.g. + C(ClustersMapped) + depth proxies, etc., if you stored them.
    if HAS_SM:
        def fit_ols(group):
            g = group[[condition_col, "z", "ClustersMapped"]].dropna().copy()
            if g[condition_col].nunique() < 2 or g.shape[0] < 4:
                return pd.Series({"p_ols": np.nan})
            # binary coding for condition
            g["_cond"] = (g[condition_col] == case_label).astype(int)
            try:
                # include celltype fixed effect if you want: + C(ClustersMapped)
                model = smf.ols("z ~ _cond", data=g).fit()
                p = model.pvalues.get("_cond", np.nan)
            except Exception:
                p = np.nan
            return pd.Series({"p_ols": p})

        ols_res = (
            dev_long
            .groupby(["set", "motif"], sort=False)
            .apply(fit_ols)
            .reset_index()
        )
        tests = tests.merge(ols_res, on=["set","motif"], how="left")
        mask2 = tests["p_ols"].notna()
        if mask2.sum():
            tests.loc[mask2, "q_ols"] = multipletests(tests.loc[mask2, "p_ols"], method="fdr_bh")[1]

    # ----------------------------
    # 7) Tidy outputs
    # ----------------------------
    # Per-replicate Z scores table (for plotting/QA)
    per_pb = dev_long.copy()

    # Per (set, motif) test table
    results = tests.sort_values(["q_welch", "p_welch"], na_position="last")

    return {
        "per_pseudobulk_z": per_pb,   # rows: pseudobulk x (set,motif) with metadata
        "tests": results,             # stats per (set,motif): Welch and (optional) OLS p/q
    }

def volcano_plot(results, term_genes, axes, p_metric='padj'):

    results['term_genes'] = results['gene'].isin(term_genes)
    results['-log10(padj)'] = -np.log10(results[p_metric])
    results['signif_padj'] = results[p_metric] < 0.05
    
    # Identify significant peaks without km assignment
    has_km = results['km'].isin(['km1', 'km2', 'km3', 'km4'])
    is_significant = results['signif_padj']
    
    # Assign categories
    results.loc[~is_significant, 'km'] = 'Not significant'
    results.loc[is_significant & ~has_km, 'km'] = 'Significant (no km)'
    
    results['km'] = pd.Categorical(results['km'], categories=['km1', 'km2', 'km3', 'km4', 'Significant (no km)', 'Not significant'], ordered=True)

    # Define default colors and add grey for both 'Not significant' and 'Significant (no km)'
    default_colors = sns.color_palette()[:4]  # Get the first four default colors
    custom_colors = default_colors + [(0.6, 0.6, 0.6)] + [(0.6, 0.6, 0.6)]  # Add grey for both categories
    
    # Create a marker style mapping
    marker_styles = {'km1': 'o', 'km2': 'o', 'km3': 'o', 'km4': 'o', 'Significant (no km)': '^', 'Not significant': 'o'}
    
    # Define plotting order: plot less important categories first, km clusters last (so they appear on top)
    plot_order = ['Not significant', 'Significant (no km)', 'km1', 'km2', 'km3', 'km4']
    
    # Plot each category separately to control marker styles and order
    for category in plot_order:
        if category not in results['km'].cat.categories:
            continue
        subset = results[results['km'] == category].reset_index()
        if not subset.empty:
            color_idx = list(results['km'].cat.categories).index(category)
            sns.scatterplot(
                data=subset, 
                x='log2FoldChange', 
                y='-log10(padj)', 
                marker=marker_styles[category],
                color=custom_colors[color_idx],
                alpha=0.5, 
                ax=axes[1], 
                label=category
            )
    
    # Top row: plot False first, then True (so True appears on top)
    plot_data = results.reset_index()
    
    # Plot False term_genes first (bottom layer)
    false_subset = plot_data[plot_data['term_genes'] == False]
    if not false_subset.empty:
        sns.scatterplot(
            data=false_subset, 
            x='log2FoldChange', 
            y='-log10(padj)', 
            marker='o', 
            alpha=0.5, 
            ax=axes[0], 
            label='False',
            color=sns.color_palette()[0]
        )
    
    # Plot True term_genes last (top layer)
    true_subset = plot_data[plot_data['term_genes'] == True]
    if not true_subset.empty:
        sns.scatterplot(
            data=true_subset, 
            x='log2FoldChange', 
            y='-log10(padj)', 
            marker='o', 
            alpha=0.5, 
            ax=axes[0], 
            label='True',
            color=sns.color_palette()[1]
        )

    # Explicitly set legend locations to avoid slow "best" calculation
    if axes[0].get_legend() is not None:
        axes[0].legend(loc='upper right', frameon=True, fontsize='small')
    if axes[1].get_legend() is not None:
        axes[1].legend(loc='upper right', frameon=True, fontsize='small')

    # Add vertical and horizontal lines for reference
    axes[0].axhline(y=-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)  # Horizontal line at significance threshold
    axes[0].axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = 1.5
    axes[0].axvline(x=-0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = -1.5

    axes[1].axhline(y=-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)  # Horizontal line at significance threshold
    axes[1].axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = 1.5
    axes[1].axvline(x=-0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = -1.5


def _plot_combined(
    motif_df,
    set_names,
    motif_name,
    condition_col,
    celltype_col,
    figsize,
    title,
    save_path
):
    """
    Helper function to create a combined plot with all sets in one boxplot,
    with scatter points color-coded by set_name.
    """
    # Determine figure size
    if figsize is None:
        figsize = (8, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a color palette for set_names
    n_sets = len(set_names)
    palette = sns.color_palette("tab10", n_colors=n_sets)
    set_colors = dict(zip(set_names, palette))
    
    # Plot overall boxplot (without hue for set_name to keep it simple)
    sns.boxplot(
        data=motif_df,
        x=condition_col,
        y="z",
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        ax=ax
    )
    
    # Plot scatter points colored by set_name
    for set_name in set_names:
        subset = motif_df[motif_df["set"] == set_name]
        if not subset.empty:
            sns.stripplot(
                data=subset,
                x=condition_col,
                y="z",
                color=set_colors[set_name],
                label=set_name,
                dodge=False,
                size=5,
                linewidth=0.5,
                edgecolor="k",
                alpha=0.7,
                ax=ax,
                jitter=True
            )
    
    # Add horizontal line at 0
    ax.axhline(0, ls="--", c="gray", lw=1)
    
    # Labels and title
    ax.set_ylabel("pychromVAR Z-score", fontsize=12)
    ax.set_xlabel(condition_col.capitalize(), fontsize=12)
    
    if title is None:
        title = f"{motif_name} deviations across all sets"
    ax.set_title(title, fontsize=14)
    
    # Add legend for set_names
    ax.legend(
        title="Set",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fontsize='small'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(
            os.path.join(save_path, f'combined_{motif_name}.png'),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()


def plot_pychromvar_case_control_multi_sets(
    per_pb_df,
    set_names,
    motif_name,
    condition_col="condition",
    celltype_col="ClustersMapped",
    figsize=None,
    ncols=None,
    title=None,
    save_path=None,
    combined=False
):
    """
    Plot per-pseudobulk pychromVAR Z-scores for a given motif across multiple sets
    in a single figure with subplots or as a combined plot.

    Parameters
    ----------
    per_pb_df : DataFrame
        Output of run_pychromVAR_case_control()["per_pseudobulk_z"].
        Must contain columns ["set","motif","z", condition_col, celltype_col].
    set_names : list of str or None
        List of set names to plot. If None, uses all unique sets in the data.
    motif_name : str
        Motif to plot (e.g. "EGR1" or "MA0470.2").
    condition_col : str, default "condition"
        Column indicating case vs control.
    celltype_col : str, default "ClustersMapped"
        Optional column for faceting.
    figsize : tuple or None
        Figure size. If None, automatically calculated based on number of subplots.
    ncols : int or None
        Number of columns for subplot grid. If None, automatically determined.
        (Only used when combined=False)
    title : str or None
        Overall figure title (suptitle).
    save_path : str or None
        If given, saves the figure to this path.
    combined : bool, default False
        If True, creates a single plot with all sets combined, with scatter points
        color-coded by set_name. If False, creates separate subplots for each set.
    """
    
    # If set_names is None, use all unique sets
    if set_names is None:
        set_names = per_pb_df["set"].unique()
    
    # Filter data for the specified motif
    motif_df = per_pb_df[per_pb_df["motif"] == motif_name].copy()
    
    if motif_df.empty:
        print(f"[warn] No rows found for motif={motif_name}")
        return
    
    # Filter for the specified sets
    motif_df = motif_df[motif_df["set"].isin(set_names)].copy()
    
    if combined:
        # Create a single combined plot
        _plot_combined(
            motif_df, 
            set_names, 
            motif_name, 
            condition_col, 
            celltype_col, 
            figsize, 
            title, 
            save_path
        )
        return
    
    # Determine subplot layout
    n_sets = len(set_names)
    if ncols is None:
        ncols = min(3, n_sets)  # Default to 3 columns max
    nrows = int(np.ceil(n_sets / ncols))
    
    # Determine figure size
    if figsize is None:
        figsize = (3 * ncols, 6 * nrows)
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Get consistent y-axis limits across all subplots
    all_z = motif_df["z"].dropna()
    if len(all_z) > 0:
        y_min, y_max = all_z.min(), all_z.max()
        y_margin = (y_max - y_min) * 0.1
        y_lim = (y_min - y_margin, y_max + y_margin)
    else:
        y_lim = None
    
    # Plot each set
    for idx, set_name in enumerate(set_names):
        ax = axes[idx]
        plt.sca(ax)
        
        subset = motif_df[motif_df["set"] == set_name].copy()
        
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for\n{set_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(set_name)
            continue
        
        # Plot boxplot and stripplot
        sns.boxplot(
            data=subset,
            x=condition_col,
            y="z",
            hue=celltype_col if celltype_col in subset.columns else None,
            showfliers=False,
            boxprops=dict(alpha=0.6),
            dodge=True,
            ax=ax
        )
        sns.stripplot(
            data=subset,
            x=condition_col,
            y="z",
            hue=celltype_col if celltype_col in subset.columns else None,
            dodge=True,
            size=4,
            linewidth=0.5,
            edgecolor="k",
            alpha=0.7,
            ax=ax
        )
        
        # Add horizontal line at 0
        ax.axhline(0, ls="--", c="gray", lw=1)
        
        # Labels and title
        ax.set_ylabel("pychromVAR Z-score")
        ax.set_xlabel(condition_col.capitalize())
        ax.set_title(set_name)
        
        # Set consistent y-axis limits
        if y_lim:
            ax.set_ylim(y_lim)
        
        # Handle legend - only show on first subplot
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicate legend entries from boxplot+stripplot
            n_categories = len(set(labels))
            if idx == len(axes) - 1:  # Only show legend on last subplot
                ax.legend(handles[:n_categories], list(set(labels))[:n_categories],
                         title=celltype_col, bbox_to_anchor=(1.05, 1), loc="upper left",
                         fontsize='small')
            else:
                ax.legend([], [], frameon=False)
        else:
            ax.legend([], [], frameon=False)
    
    # Hide unused subplots
    for idx in range(n_sets, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    if title is None:
        title = f"{motif_name} deviations across sets"
    fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'multi_sets_{motif_name}.png'), 
                   dpi=300, bbox_inches="tight")
    plt.show()


def create_comprehensive_plot(output_dir, modality_filter):
    """
    Create a comprehensive plot showing -log10(p) results for:
    - enrichr vs gsea
    - male vs female
    - individual cell-types
    """
    # Load all saved results (only enrichr and gsea result files, not summary stats)
    all_results = []
    for csv_file in glob.glob(os.path.join(output_dir, '*.csv')):
        # Skip summary statistics files
        if 'summary_stats' in csv_file or 'comparison' in csv_file:
            continue
        
        df = pd.read_csv(csv_file)
        
        # Extract metadata from filename if not present in dataframe
        # Filename format: {method}_{modality}_{celltype}_{sex}.csv
        basename = os.path.basename(csv_file).replace('.csv', '')
        
        # Parse filename to extract metadata
        # Handle both underscore and hyphen separators in celltype names
        if basename.startswith('enrichr_') or basename.startswith('gsea_'):
            method = basename.split('_')[0]
            remainder = basename[len(method)+1:]  # Remove method and underscore
            
            # Next should be modality (RNA or ATAC)
            if remainder.startswith('RNA_'):
                modality = 'RNA'
                remainder = remainder[4:]
            elif remainder.startswith('ATAC_'):
                modality = 'ATAC'
                remainder = remainder[5:]
            else:
                print(f"Warning: Could not parse modality from {basename}")
                continue
            
            # Last part is sex (male or female)
            if remainder.endswith('_male'):
                sex = 'male'
                celltype = remainder[:-5]
            elif remainder.endswith('_female'):
                sex = 'female'
                celltype = remainder[:-7]
            else:
                print(f"Warning: Could not parse sex from {basename}")
                continue
            
            # Add metadata columns if they don't exist
            if 'method' not in df.columns:
                df['method'] = method
            if 'modality' not in df.columns:
                df['modality'] = modality
            if 'sex' not in df.columns:
                df['sex'] = sex
            if 'celltype' not in df.columns:
                df['celltype'] = celltype
        
        all_results.append(df)
    
    if not all_results:
        print("No results found to plot")
        return
    
    # Concatenate all results
    combined_results = pd.concat(all_results, ignore_index=True)

    ## Filter by modality
    combined_results = combined_results[combined_results['modality'] == modality_filter]
    
    # Verify required columns are present
    required_cols = ['modality', 'method', 'sex', 'celltype']
    missing_cols = [col for col in required_cols if col not in combined_results.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {combined_results.columns.tolist()}")
        return
    
    # Extract p-values based on method
    # For enrichr, typically use 'Adjusted P-value' or 'P-value'
    # For gsea, typically use 'FDR q-val' or 'NOM p-val'
    
    # Determine which p-value column to use
    if 'Adjusted P-value' in combined_results.columns:
        pval_col = 'Adjusted P-value'
    elif 'P-value' in combined_results.columns:
        pval_col = 'P-value'
    elif 'FDR q-val' in combined_results.columns:
        pval_col = 'FDR q-val'
    elif 'NOM p-val' in combined_results.columns:
        pval_col = 'NOM p-val'
    else:
        # Try to find any column with 'p' or 'P' in it
        pval_cols = [col for col in combined_results.columns if 'p-val' in col.lower() or 'p_val' in col.lower() or 'pval' in col.lower()]
        if pval_cols:
            pval_col = pval_cols[0]
        else:
            print("No p-value column found in results")
            print("Available columns:", combined_results.columns.tolist())
            return
    
    # Calculate -log10(p)
    combined_results['-log10_p'] = -np.log10(combined_results[pval_col].clip(lower=1e-300))
    
    # Get term/pathway name (different naming in enrichr vs gsea)
    if 'Term' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term']
    elif 'Term name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term name']
    elif 'Name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Name']
    else:
        combined_results['pathway'] = combined_results.index.astype(str)

    # Filter out non-significant results and prepare summary
    summary_results = combined_results.groupby(['modality', 'method', 'sex', 'celltype', 'pathway']).agg({
        '-log10_p': 'max'  # Take maximum -log10(p) if there are duplicates
    }).reset_index()
    
    # Create a pivot for heatmap: rows=pathway, columns=method_sex_celltype
    summary_results['method_sex_celltype'] = (summary_results['method'] + '_' + 
                                              summary_results['sex'] + '_' + 
                                              summary_results['celltype'] + '_' +
                                              summary_results['modality'])
    
    # For a cleaner plot, let's take the top pathways by average significance
    pathway_avg_sig = summary_results.groupby('pathway')['-log10_p'].mean().sort_values(ascending=False)
    top_pathways = pathway_avg_sig.head(20).index.tolist()
    
    plot_data = summary_results[summary_results['pathway'].isin(top_pathways)]

    ## code as ordered categorical
    celltype_order = ['RG', 'IPC', 'EN-fetal-early', 'EN-fetal-late', 'EN']
    available_celltypes = [ct for ct in celltype_order if ct in plot_data['celltype'].unique()]
    plot_data['celltype'] = pd.Categorical(plot_data['celltype'], categories=available_celltypes, ordered=True)
    plot_data['pathway'] = pd.Categorical(plot_data['pathway'], categories=plot_data['pathway'].sort_values().unique(), ordered=True)
    
    # Create figure with subplots - 2x3 grid for single modality
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    fig.suptitle(f'{modality_filter} Enrichment Analysis: -log10(p) Comparison', fontsize=20, fontweight='bold', y=0.995)
    
    # Define color schemes
    method_colors = ['#2E86AB', '#A23B72']  # Blue for enrichr, Purple for gsea
    sex_colors = ['#4ECDC4', '#FF6B6B']  # Teal for female, Coral for male
    celltype_palette = sns.color_palette("Set2", n_colors=len(available_celltypes))
    
    # Plot 1: EnrichR vs GSEA (averaged across sex and celltype)
    ax1 = fig.add_subplot(gs[0, 0])
    method_comparison = plot_data.groupby(['method', 'pathway'])['-log10_p'].mean().unstack(level=0)
    if not method_comparison.empty and method_comparison.shape[1] >= 2:
        method_comparison_top = method_comparison.loc[pathway_avg_sig[pathway_avg_sig.index.isin(method_comparison.index)].head(10).index]
        method_comparison_top.plot(kind='barh', ax=ax1, color=method_colors, width=0.75)
        ax1.set_xlabel('-log10(p-value)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Pathway', fontsize=13, fontweight='bold')
        ax1.set_title('EnrichR vs GSEA', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(title='Method', fontsize=11, title_fontsize=12, frameon=True, shadow=True)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.tick_params(axis='both', labelsize=11)
    
    # Plot 2: Male vs Female (averaged across method and celltype)
    ax2 = fig.add_subplot(gs[0, 1])
    sex_comparison = plot_data.groupby(['sex', 'pathway'])['-log10_p'].mean().unstack(level=0)
    if not sex_comparison.empty and sex_comparison.shape[1] >= 2:
        sex_comparison_top = sex_comparison.loc[pathway_avg_sig[pathway_avg_sig.index.isin(sex_comparison.index)].head(10).index]
        # Ensure consistent ordering: female, male
        if 'female' in sex_comparison_top.columns and 'male' in sex_comparison_top.columns:
            sex_comparison_top = sex_comparison_top[['female', 'male']]
        sex_comparison_top.plot(kind='barh', ax=ax2, color=sex_colors, width=0.75)
        ax2.set_xlabel('-log10(p-value)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Pathway', fontsize=13, fontweight='bold')
        ax2.set_title('Male vs Female', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(title='Sex', fontsize=11, title_fontsize=12, frameon=True, shadow=True)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='both', labelsize=11)
    
    # Plot 3: Cell-types (averaged across method and sex)
    ax3 = fig.add_subplot(gs[0, 2])
    celltype_comparison = plot_data.groupby(['celltype', 'pathway'])['-log10_p'].mean().unstack(level=0)
    if not celltype_comparison.empty:
        # Reorder columns by celltype order
        celltype_comparison = celltype_comparison[[ct for ct in available_celltypes if ct in celltype_comparison.columns]]
        celltype_comparison_top = celltype_comparison.loc[pathway_avg_sig[pathway_avg_sig.index.isin(celltype_comparison.index)].head(10).index]
        celltype_comparison_top.plot(kind='barh', ax=ax3, color=celltype_palette, width=0.75)
        ax3.set_xlabel('-log10(p-value)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Pathway', fontsize=13, fontweight='bold')
        ax3.set_title('Cell-type Comparison', fontsize=15, fontweight='bold', pad=15)
        ax3.legend(title='Cell Type', fontsize=10, title_fontsize=11, frameon=True, shadow=True, 
                  bbox_to_anchor=(1.02, 1), loc='upper left')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.tick_params(axis='both', labelsize=11)
    
    # Plot 4: Heatmap showing all combinations (method × sex × celltype)
    ax4 = fig.add_subplot(gs[1, :2])  # Span first two columns
    heatmap_data = plot_data.pivot_table(
        values='-log10_p', 
        index='pathway', 
        columns=['method', 'sex', 'celltype'],
        aggfunc='mean'
    )
    if not heatmap_data.empty:
        heatmap_top = heatmap_data.loc[pathway_avg_sig[pathway_avg_sig.index.isin(heatmap_data.index)].head(15).index]
        heatmap_top.columns = [f"{m[:3].upper()}_{s[0].upper()}_{c}" for m, s, c in heatmap_top.columns]
        
        sns.heatmap(heatmap_top, cmap='YlOrRd', annot=False, 
                   cbar_kws={'label': '-log10(p-value)', 'shrink': 0.8}, ax=ax4,
                   xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgray')
        ax4.set_xlabel('Method_Sex_CellType', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Pathway', fontsize=13, fontweight='bold')
        ax4.set_title('Comprehensive Heatmap: All Combinations', fontsize=15, fontweight='bold', pad=15)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax4.get_yticklabels(), fontsize=10)
    
    # Plot 5: Top significant pathways by average -log10(p)
    ax5 = fig.add_subplot(gs[1, 2])
    top_pathways_plot = pathway_avg_sig.head(15)
    top_pathways_plot.plot(kind='barh', ax=ax5, color='#06A77D', width=0.75)
    ax5.set_xlabel('Average -log10(p-value)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Pathway', fontsize=13, fontweight='bold')
    ax5.set_title('Top 15 Pathways\n(averaged across all)', fontsize=15, fontweight='bold', pad=15)
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    ax5.tick_params(axis='both', labelsize=10)
    
    # Save the plot with modality-specific filename
    output_file = os.path.join(output_dir, f'comprehensive_enrichment_plot_{modality_filter}.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{modality_filter} comprehensive plot saved to: {output_file}")
    plt.close(fig)
    
    # Also create a summary statistics table for this modality
    summary_stats = plot_data.groupby(['method', 'sex', 'celltype']).agg({
        'pathway': 'count',
        '-log10_p': ['mean', 'max', 'min']
    }).round(3)
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats_file = os.path.join(output_dir, f'enrichment_summary_stats_{modality_filter}.csv')
    summary_stats.to_csv(summary_stats_file)
    print(f"{modality_filter} summary statistics saved to: {summary_stats_file}")

def create_enrichr_rna_atac_stacked_heatmap(output_dir):
    """
    Create stacked comprehensive heatmaps for RNA and ATAC (EnrichR only) with km clusters sorted by labels.
    Simplified version based on create_comprehensive_plot function, focusing only on the heatmap.
    - Rows: km clusters sorted by label (km1 < km2 < km3 < km4), not by significance
    - Columns: sex_celltype combinations  
    - Two heatmaps stacked vertically (RNA on top, ATAC on bottom)
    - x-tick labels show only sex_celltype (no "method" prefix since EnrichR-only)
    """
    import glob
    
    # Load all saved results (reusing logic from create_comprehensive_plot)
    all_results = []
    for csv_file in glob.glob(os.path.join(output_dir, '*.csv')):
        # Skip summary statistics files
        if 'summary_stats' in csv_file or 'comparison' in csv_file:
            continue
        
        df = pd.read_csv(csv_file)
        basename = os.path.basename(csv_file).replace('.csv', '')
        
        # Parse filename to extract metadata
        if basename.startswith('enrichr_') or basename.startswith('gsea_'):
            method = basename.split('_')[0]
            remainder = basename[len(method)+1:]
            
            # Next should be modality (RNA or ATAC)
            if remainder.startswith('RNA_'):
                modality = 'RNA'
                remainder = remainder[4:]
            elif remainder.startswith('ATAC_'):
                modality = 'ATAC'
                remainder = remainder[5:]
            else:
                continue
            
            # Last part is sex (male or female)
            if remainder.endswith('_male'):
                sex = 'male'
                celltype = remainder[:-5]
            elif remainder.endswith('_female'):
                sex = 'female'
                celltype = remainder[:-7]
            else:
                continue
            
            # Add metadata columns if they don't exist
            if 'method' not in df.columns:
                df['method'] = method
            if 'modality' not in df.columns:
                df['modality'] = modality
            if 'sex' not in df.columns:
                df['sex'] = sex
            if 'celltype' not in df.columns:
                df['celltype'] = celltype
        
        all_results.append(df)
    
    if not all_results:
        print("No results found to plot")
        return
    
    # Concatenate all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Filter to EnrichR only
    combined_results = combined_results[combined_results['method'] == 'enrichr']
    
    if combined_results.empty:
        print("No EnrichR results found to create stacked heatmap")
        return
    
    # Determine which p-value column to use
    if 'Adjusted P-value' in combined_results.columns:
        pval_col = 'Adjusted P-value'
    elif 'P-value' in combined_results.columns:
        pval_col = 'P-value'
    elif 'FDR q-val' in combined_results.columns:
        pval_col = 'FDR q-val'
    elif 'NOM p-val' in combined_results.columns:
        pval_col = 'NOM p-val'
    else:
        pval_cols = [col for col in combined_results.columns if 'p-val' in col.lower() or 'p_val' in col.lower() or 'pval' in col.lower()]
        if pval_cols:
            pval_col = pval_cols[0]
        else:
            print("No p-value column found in results")
            return
    
    # Calculate -log10(p)
    combined_results['-log10_p'] = -np.log10(combined_results[pval_col].clip(lower=1e-300))
    
    # Get term/pathway name
    if 'Term' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term']
    elif 'Term name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term name']
    elif 'Name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Name']
    else:
        combined_results['pathway'] = combined_results.index.astype(str)
    
    # Define celltype order
    celltype_order = ['RG', 'IPC', 'EN-fetal-early', 'EN-fetal-late', 'EN']
    available_celltypes = [ct for ct in celltype_order if ct in combined_results['celltype'].unique()]
    
    # Code as ordered categorical
    combined_results['celltype'] = pd.Categorical(combined_results['celltype'], 
                                                   categories=available_celltypes, 
                                                   ordered=True)
    
    # Extract km cluster number for sorting
    def extract_km_number(km_str):
        import re
        match = re.search(r'km(\d+)', str(km_str).lower())
        if match:
            return int(match.group(1))
        return 999  # Put non-matching at end
    
    combined_results['km_number'] = combined_results['pathway'].apply(extract_km_number)
    
    # Sort by km number and get unique pathways in sorted order
    pathway_sorted = combined_results.sort_values('km_number')['pathway'].unique()
    
    # Separate RNA and ATAC data
    rna_data = combined_results[combined_results['modality'] == 'RNA'].copy()
    atac_data = combined_results[combined_results['modality'] == 'ATAC'].copy()
    
    # Create pivot tables (same style as comprehensive heatmap in create_comprehensive_plot)
    rna_heatmap = rna_data.pivot_table(
        values='-log10_p',
        index='pathway',
        columns=['sex', 'celltype'],
        aggfunc='mean'
    )
    
    atac_heatmap = atac_data.pivot_table(
        values='-log10_p',
        index='pathway',
        columns=['sex', 'celltype'],
        aggfunc='mean'
    )
    
    if rna_heatmap.empty and atac_heatmap.empty:
        print("No data to plot")
        return
    
    # Reorder rows by km label (not by significance)
    if not rna_heatmap.empty:
        rna_heatmap = rna_heatmap.reindex([p for p in pathway_sorted if p in rna_heatmap.index])
        # Format column names: F_RG, F_EN, M_RG, M_EN (no method prefix)
        rna_heatmap.columns = [f"{s[0].upper()}_{c}" for s, c in rna_heatmap.columns]
    
    if not atac_heatmap.empty:
        atac_heatmap = atac_heatmap.reindex([p for p in pathway_sorted if p in atac_heatmap.index])
        # Format column names: F_RG, F_EN, M_RG, M_EN (no method prefix)
        atac_heatmap.columns = [f"{s[0].upper()}_{c}" for s, c in atac_heatmap.columns]
    
    # Create figure with two subplots stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Comprehensive Heatmap: EnrichR RNA and ATAC (All Combinations)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Significance threshold: -log10(0.05) ≈ 1.3
    sig_threshold = -np.log10(0.05)
    
    # Plot RNA heatmap (with its own color scale)
    if not rna_heatmap.empty:
        # Create annotation matrix with asterisks for significant values
        rna_annot = rna_heatmap.applymap(lambda x: '*' if x > sig_threshold else '')
        
        sns.heatmap(rna_heatmap, cmap='YlOrRd', annot=rna_annot, fmt='',
                   cbar_kws={'label': '-log10(p-value)', 'shrink': 0.8}, ax=axes[0],
                   xticklabels=True, yticklabels=True, 
                   linewidths=0.5, linecolor='lightgray',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold', 'color': 'black'})
        axes[0].set_xlabel('Sex_CellType', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Pathway', fontsize=13, fontweight='bold')
        axes[0].set_title('RNA', fontsize=15, fontweight='bold', pad=15)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'No RNA data', ha='center', va='center', fontsize=14)
        axes[0].set_title('RNA', fontsize=15, fontweight='bold', pad=15)
    
    # Plot ATAC heatmap (with its own color scale)
    if not atac_heatmap.empty:
        # Create annotation matrix with asterisks for significant values
        atac_annot = atac_heatmap.applymap(lambda x: '*' if x > sig_threshold else '')
        
        sns.heatmap(atac_heatmap, cmap='YlOrRd', annot=atac_annot, fmt='',
                   cbar_kws={'label': '-log10(p-value)', 'shrink': 0.8}, ax=axes[1],
                   xticklabels=True, yticklabels=True, 
                   linewidths=0.5, linecolor='lightgray',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold', 'color': 'black'})
        axes[1].set_xlabel('Sex_CellType', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Pathway', fontsize=13, fontweight='bold')
        axes[1].set_title('ATAC', fontsize=15, fontweight='bold', pad=15)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No ATAC data', ha='center', va='center', fontsize=14)
        axes[1].set_title('ATAC', fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'enrichr_rna_atac_stacked_heatmap.svg')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nEnrichR RNA-ATAC stacked comprehensive heatmap saved to: {output_file}")
    plt.close(fig)

def create_rna_atac_comparison(output_dir):
    """
    Create a comparison plot between RNA and ATAC modalities
    """
    import glob
    
    # Load all saved results
    all_results = []
    for csv_file in glob.glob(os.path.join(output_dir, '*.csv')):
        # Skip summary statistics files
        if 'summary_stats' in csv_file or 'comparison' in csv_file:
            continue
        
        df = pd.read_csv(csv_file)
        basename = os.path.basename(csv_file).replace('.csv', '')
        
        # Parse filename to extract metadata
        if basename.startswith('enrichr_') or basename.startswith('gsea_'):
            method = basename.split('_')[0]
            remainder = basename[len(method)+1:]
            
            if remainder.startswith('RNA_'):
                modality = 'RNA'
                remainder = remainder[4:]
            elif remainder.startswith('ATAC_'):
                modality = 'ATAC'
                remainder = remainder[5:]
            else:
                continue
            
            if remainder.endswith('_male'):
                sex = 'male'
                celltype = remainder[:-5]
            elif remainder.endswith('_female'):
                sex = 'female'
                celltype = remainder[:-7]
            else:
                continue
            
            if 'method' not in df.columns:
                df['method'] = method
            if 'modality' not in df.columns:
                df['modality'] = modality
            if 'sex' not in df.columns:
                df['sex'] = sex
            if 'celltype' not in df.columns:
                df['celltype'] = celltype
        
        all_results.append(df)
    
    if len(all_results) < 2:
        print("Not enough results to create RNA vs ATAC comparison")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Determine p-value column
    if 'Adjusted P-value' in combined_results.columns:
        pval_col = 'Adjusted P-value'
    elif 'P-value' in combined_results.columns:
        pval_col = 'P-value'
    elif 'FDR q-val' in combined_results.columns:
        pval_col = 'FDR q-val'
    elif 'NOM p-val' in combined_results.columns:
        pval_col = 'NOM p-val'
    else:
        pval_cols = [col for col in combined_results.columns if 'p-val' in col.lower() or 'p_val' in col.lower() or 'pval' in col.lower()]
        if pval_cols:
            pval_col = pval_cols[0]
        else:
            return
    
    combined_results['-log10_p'] = -np.log10(combined_results[pval_col].clip(lower=1e-300))
    
    # Get pathway name
    if 'Term' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term']
    elif 'Term name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Term name']
    elif 'Name' in combined_results.columns:
        combined_results['pathway'] = combined_results['Name']
    else:
        combined_results['pathway'] = combined_results.index.astype(str)
    
    # Create comparison table
    modality_comparison = combined_results.groupby(['modality', 'pathway'])['-log10_p'].mean().unstack(level=0)
    
    if 'RNA' in modality_comparison.columns and 'ATAC' in modality_comparison.columns:
        # Save comparison table
        modality_comparison.to_csv(os.path.join(output_dir, 'rna_vs_atac_comparison.csv'))
        print(f"\nRNA vs ATAC comparison table saved to: {os.path.join(output_dir, 'rna_vs_atac_comparison.csv')}")
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        scatter_data = modality_comparison.dropna()
        ax.scatter(scatter_data['RNA'], scatter_data['ATAC'], alpha=0.6, s=100, c='#2E86AB', edgecolors='black')
        
        # Add diagonal line
        max_val = max(scatter_data['RNA'].max(), scatter_data['ATAC'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='y=x')
        
        # Add labels for top pathways
        top_n = 10
        for idx in scatter_data.nlargest(top_n, 'RNA').index[:5]:
            ax.annotate(idx[:30], (scatter_data.loc[idx, 'RNA'], scatter_data.loc[idx, 'ATAC']),
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('RNA -log10(p-value)', fontsize=14, fontweight='bold')
        ax.set_ylabel('ATAC -log10(p-value)', fontsize=14, fontweight='bold')
        ax.set_title('RNA vs ATAC Enrichment Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        comparison_plot_file = os.path.join(output_dir, 'rna_vs_atac_scatter.png')
        fig.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
        print(f"RNA vs ATAC scatter plot saved to: {comparison_plot_file}")
        plt.close(fig)



@ray.remote
def compute_gene_activity_score(gene, peak_gene_links_subset, var_names, raw_X):
    peaks_linked_to_gene = peak_gene_links_subset.groupby('peak').mean()
    peaks_mask = np.isin(var_names, peaks_linked_to_gene.index)
    gene_activity_scores = raw_X[:, peaks_mask].dot(peaks_linked_to_gene.values)
    return gene, gene_activity_scores


def balance_cell_types(adata, cell_type_col='most_common_cluster', cell_types=None, random_state=None):
    """
    Balance cell type composition by sampling equal numbers from each cell type.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object with cell type annotations
    cell_type_col : str
        Column name in adata.obs containing cell type labels
    cell_types : list or None
        List of cell types to include. If None, uses all unique cell types.
    random_state : int or None
        Random seed for reproducibility
        
    Returns
    -------
    AnnData
        Balanced AnnData object with equal cells per type
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Filter to specified cell types if provided
    if cell_types is not None:
        adata = adata[adata.obs[cell_type_col].isin(cell_types)].copy()
    else:
        cell_types = adata.obs[cell_type_col].unique().tolist()
    
    # Get cell type counts
    cell_type_counts = adata.obs[cell_type_col].value_counts()
    print(f"\nCell type counts before balancing ({cell_type_col}):")
    print(cell_type_counts)
    
    # Find minimum number of cells across cell types
    min_cells = cell_type_counts.min()
    print(f"\nSampling {min_cells} cells per cell type for balanced analysis...")
    
    # Sample equal number of cells from each cell type
    balanced_indices = []
    for cell_type in cell_types:
        cell_type_mask = adata.obs[cell_type_col] == cell_type
        cell_type_indices = np.where(cell_type_mask)[0]
        
        if len(cell_type_indices) >= min_cells:
            sampled_indices = np.random.choice(cell_type_indices, size=min_cells, replace=False)
            balanced_indices.extend(sampled_indices)
        else:
            print(f"Warning: {cell_type} has fewer cells ({len(cell_type_indices)}) than min_cells ({min_cells})")
    
    # Subset to balanced samples
    adata_balanced = adata[balanced_indices].copy()
    
    # Verify balanced composition
    balanced_counts = adata_balanced.obs[cell_type_col].value_counts()
    print(f"\nCell type counts after balancing:")
    print(balanced_counts)
    print(f"Total cells: {adata_balanced.n_obs}\n")
    
    return adata_balanced

def intersect_gwas_hits_with_egr1_scompreg_hits(egr1_scompreg_hits_dict, gwas_catalog_metadata):
    # Prepare BED format dataframes with proper handling of coordinates
    egr1_bed_df = egr1_scompreg_hits_grn['enhancer'].str.split('[:-]', expand=True)
    egr1_bed_df = egr1_bed_df.dropna()  # Remove any rows with NaN
    egr1_bed_df.columns = ['chr', 'start', 'end']
    egr1_bed_df['start'] = egr1_bed_df['start'].astype(int)
    egr1_bed_df['end'] = egr1_bed_df['end'].astype(int)
    
    gwas_bed_df = gwas_hits['Peak coordinates (hg38)'].str.split('[:-]', expand=True)
    gwas_bed_df = gwas_bed_df.dropna()  # Remove any rows with NaN
    gwas_bed_df.columns = ['chr', 'start', 'end']
    gwas_bed_df['start'] = gwas_bed_df['start'].astype(int)
    gwas_bed_df['end'] = gwas_bed_df['end'].astype(int)
    
    egr1_scompreg_hits_bedtool = BedTool.from_dataframe(egr1_bed_df)
    
    # Use wa=True and wb=True to get both intervals and SNP metadata
    egr1_scompreg_hits_bedtool_intersect = egr1_scompreg_hits_bedtool.intersect(
        gwas_catalog_bedtool, 
        wa=True,  # Write original egr1 intervals
        wb=True   # Write overlapping GWAS intervals with metadata
    )
    
    # Convert to dataframe and parse columns
    egr1_scompreg_hits_bedtool_intersect_df = egr1_scompreg_hits_bedtool_intersect.to_dataframe()
    
    # The columns are: chr, start, end (from egr1) + chr, start, end, snp_id, trait, rsid (from GWAS)
    egr1_scompreg_hits_bedtool_intersect_df.columns = [
        'egr1_chr', 'egr1_start', 'egr1_end', 
        'gwas_chr', 'gwas_start', 'gwas_end', 
        'snp_id', 'trait', 'rsid'
    ]
    
    print(f"\n{celltype}: Found {len(egr1_scompreg_hits_bedtool_intersect_df)} SNPs intersecting with EGR1 enhancers")
    print(f"Unique traits: {egr1_scompreg_hits_bedtool_intersect_df['trait'].nunique()}")
    print(f"\nTop traits:\n{egr1_scompreg_hits_bedtool_intersect_df['trait'].value_counts().head(10)}")

    return egr1_scompreg_hits_bedtool_intersect_df

if __name__ == '__main__':
    main(subset_type=None)