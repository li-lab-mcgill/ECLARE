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

sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
from eclare import set_env_variables
set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')


def main(subset_type=None):

    ## load data
    mdd_rna_scaled_sub = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_rna_scaled_sub.h5ad'))
    mdd_atac_sub = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_gas_broad_sub.h5ad'))
    
    mdd_atac_broad = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'mdd_atac_broad.h5ad'), backed='r')
    mdd_atac_broad_sub = mdd_atac_broad[mdd_atac_broad.obs_names.isin(mdd_atac_sub.obs_names)].to_memory()
    mdd_atac_broad_sub.X = mdd_atac_broad_sub.raw.X
    mdd_atac_broad_sub.var_names = mdd_atac_broad_sub.var_names.str.split(':|-', expand=True).to_frame().apply(axis=1, func=lambda x: f'{x[0]}:{x[1]}-{x[2]}').values

    ## subset data
    mdd_rna_scaled_sub = subset_data(mdd_rna_scaled_sub, gwas_hits, subset_type)
    mdd_atac_sub = subset_data(mdd_atac_sub, gwas_hits, subset_type, hvg_genes=mdd_rna_scaled_sub.var_names.tolist())

    ## load GWAS hits
    zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
    gwas_hits = pd.read_excel(zhu_supp_tables, sheet_name='Table S12', header=2)

    ## load peak-gene links and km gene sets
    peak_gene_links = get_peak_gene_links()
    km_gene_sets, km_gene_sets_mapper = load_km_gene_sets()


    for sex in ['male', 'female']:

        ## across all celltypes
        rna_all = run_pyDESeq2_on_celltypes(mdd_rna_scaled_sub, sex, test_type='all')
        rna_all.index = mdd_rna_scaled_sub.raw.var_names
        rna_all.index.name = 'gene'
        rna_all.reset_index(inplace=True)

        ## per celltype
        rna_per_celltype = run_pyDESeq2_on_celltypes(mdd_rna_scaled_sub, sex)
        atac_per_celltype = run_pyDESeq2_on_celltypes(mdd_atac_sub, sex)

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

        rna_results = concat_results(rna_per_celltype, mdd_rna_scaled_sub.raw.var_names)
        atac_results = concat_results(atac_per_celltype, mdd_atac_sub.raw.var_names)

        rna_sig_results, rna_mdd_sig_results, rna_sig_genes_per_celltype, rna_sig_km_per_celltype = extract_results(rna_results, p_metric='padj')
        atac_sig_results, atac_mdd_sig_results, atac_sig_genes_per_celltype, atac_sig_km_per_celltype = extract_results(atac_results, p_metric='padj')

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
                max_size=100
            )
            gsea_res = res.res2d

            return gsea_res


        ## run enrichr and gsea, celltypes combined - test all gene sets
        rna_enrichr_res_all = do_enrichr(rna_sig_results, rna_sig_results.index.unique().tolist(), km_gene_sets)

        ## run enrichr and gsea for each celltype - test km_gene_sets
        rna_enrichr_res = {celltype: do_enrichr(rna_sig_results, [celltype], km_gene_sets) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running EnrichR on RNA')}
        atac_enrichr_res = {celltype: do_enrichr(atac_sig_results, [celltype], km_gene_sets) for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running EnrichR on ATAC')}
        rna_gsea_res = {celltype: do_gsea(rna_results, celltype) for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running GSEA on RNA')}
        atac_gsea_res = {celltype: do_gsea(atac_results, celltype) for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running GSEA on ATAC')}

        ## run enrichr - test ChEA_2022
        rna_enrichr_res_chea = do_enrichr(rna_all.loc[rna_all['log2FoldChange'] < 0.05], None, 'ChEA_2022')
        #atac_enrichr_res_chea = do_enrichr(atac_all, None, 'ChEA_2022')

        ## run enrichr, celltypes combined - test ChEA_2022
        rna_enrichr_res_chea_all = do_enrichr(rna_sig_results, rna_sig_results.index.unique().tolist(), 'ChEA_2022')
        rna_gsea_res_chea_all = do_gsea(rna_results, rna_sig_results.index.unique().tolist(), 'ChEA_2022')

        ## run enrichr for each celltype - test ChEA_2022
        rna_enrichr_res_chea_celltypes = {celltype: do_enrichr(rna_sig_results, [celltype], 'ChEA_2022') for celltype in tqdm(rna_sig_genes_per_celltype.keys(), desc='Running EnrichR on RNA - ChEA_2022')}
        atac_enrichr_res_chea_celltypes = {celltype: do_enrichr(atac_sig_results, [celltype], 'ChEA_2022') for celltype in tqdm(atac_sig_genes_per_celltype.keys(), desc='Running EnrichR on ATAC - ChEA_2022')}

        def volcano_plot(results, term_genes, axes):
            results['term_genes'] = results['gene'].isin(term_genes)
            results['-log10(padj)'] = -np.log10(results['padj'])
            results['signif_padj'] = results['padj'] < 0.05
            results.loc[~results['signif_padj'], 'km'] = 'Not significant'
            results['km'] = pd.Categorical(results['km'], categories=['km1', 'km2', 'km3', 'km4', 'Not significant'], ordered=True)

            # Define default colors and add grey for 'Not significant'
            default_colors = sns.color_palette()[:4]  # Get the first four default colors
            custom_colors = default_colors + [(0.6, 0.6, 0.6)]  # Add grey
            sns.scatterplot(data=results.reset_index(), x='log2FoldChange', y='-log10(padj)', hue='km', marker='o', alpha=0.5, palette=custom_colors, ax=axes[1])
            sns.scatterplot(data=results.reset_index(), x='log2FoldChange', y='-log10(padj)', hue='term_genes', marker='o', alpha=0.5, ax=axes[0])

            # Add vertical and horizontal lines for reference
            axes[0].axhline(y=-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)  # Horizontal line at significance threshold
            axes[0].axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = 1.5
            axes[0].axvline(x=-0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = -1.5

            axes[1].axhline(y=-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)  # Horizontal line at significance threshold
            axes[1].axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = 1.5
            axes[1].axvline(x=-0.5, color='grey', linestyle='--', linewidth=0.8)  # Vertical line at log2FoldChange = -1.5

            x_absmax = np.max(abs(results['log2FoldChange'])) + 0.5
            axes[0].set_xlim(x_absmax * -1, x_absmax)
            axes[1].set_xlim(x_absmax * -1, x_absmax)


        ## search for EGR1 term in enrichr results
        egr1_terms = rna_enrichr_res_chea.loc[rna_enrichr_res_chea['Term'].str.contains('EGR1'), 'Term'].unique()
        best_egr1_term = egr1_terms[0]

        ## searchr for NR4A2 term in enrichr results
        nr4a2_terms = rna_enrichr_res_chea.loc[rna_enrichr_res_chea['Term'].str.contains('NR4A2'), 'Term'].unique()
        best_nr4a2_term = nr4a2_terms[0]

        terms_dict = {'EGR1': best_egr1_term, 'NR4A2': best_nr4a2_term}

        ## volcano plots - best EGR1 term

        fig, ax = plt.subplots(2, 4, figsize=(20, 11), sharex=True, sharey=True)
        plt.suptitle(best_egr1_term)

        term_linked_genes = {}
        term_linked_peaks = {}
        for c, celltype in enumerate(['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']):
            results = rna_results.loc[celltype].copy()
            results['km'] = results['gene'].map(km_gene_sets_mapper)

            term_genes = rna_enrichr_res_chea_celltypes[celltype].loc[rna_enrichr_res_chea_celltypes[celltype]['Term'].eq(terms_dict['EGR1']), 'Genes'].str.split(';').iloc[0]
            peak_gene_link_in_term = (peak_gene_links['TF'].eq('EGR1') & peak_gene_links['gene'].isin(term_genes))
            term_linked_genes[celltype] = peak_gene_links.loc[peak_gene_link_in_term, 'gene'].unique()
            term_linked_peaks[celltype] = peak_gene_links.loc[peak_gene_link_in_term, 'peak'].unique()

            volcano_plot(results, term_genes, (ax[0, c], ax[1, c]))
            ax[0, c].set_title(celltype)
            ax[1, c].set_title(celltype)

        plt.tight_layout()

        ## volcano plots - best NR4A2 term
        fig, ax = plt.subplots(2, 4, figsize=(20, 11), sharex=True, sharey=True)
        plt.suptitle(best_nr4a2_term)

        for c, celltype in enumerate(['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']):
            results = rna_results.loc[celltype].copy()
            results['km'] = results['gene'].map(km_gene_sets_mapper)
            try:
                term_genes = rna_enrichr_res_chea_celltypes[celltype].loc[rna_enrichr_res_chea_celltypes[celltype]['Term'].eq(best_nr4a2_term), 'Genes'].str.split(';').iloc[0]
            except:
                term_genes = []
                print(f'No NR4A2 term found for {celltype}')

            volcano_plot(results, term_genes, (ax[0, c], ax[1, c]))
            ax[0, c].set_title(celltype)
            ax[1, c].set_title(celltype)
        plt.tight_layout()
        
        ## Save results on a celltype & sex basis
        output_dir = os.path.join(os.environ['DATAPATH'], 'mdd_data', 'developmental_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
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
    peak_names_mapper = pd.read_pickle(os.path.join(os.environ['OUTPATH'], 'peak_names_mapper.pkl')) # derived from female data
    peak_names_mapper_reverse = {v: k for k, v in peak_names_mapper.items()}
    mean_grn_df_filtered = pd.read_csv(os.path.join(os.environ['OUTPATH'], 'mean_grn_df_filtered.csv'))

    peak_gene_links = mean_grn_df_filtered.copy()
    peak_gene_links['peak'] = peak_gene_links['enhancer'].map(peak_names_mapper_reverse)
    peak_gene_links.rename(columns={'TG': 'gene'}, inplace=True)

    return peak_gene_links

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


def run_pyDESeq2_on_celltypes(adata, sex, test_type='split'):
    
    all_mdd_subjects_counts_adata = []
    all_counts = []
    all_metadata = []

    ## loop through celltypes and get pseudo-replicates counts
    for celltype in ['RG', 'EN-fetal-early', 'EN-fetal-late', 'EN']:

        #results = process_celltype(sex, celltype, mdd_rna_scaled_sub, mdd_rna_scaled_sub.raw.var.set_index('_index').copy(), 'most_common_cluster', 'Condition', 'Sex', 'OriginalSub')

        mdd_subjects_counts_adata, counts, metadata = get_pseudo_replicates_counts(
            sex, celltype, adata, adata.raw.var.copy(), 
            'most_common_cluster', 'Condition', 'Sex', 'OriginalSub',
            pseudo_replicates='Subjects', overlapping_only=False
        )

        all_mdd_subjects_counts_adata.append(mdd_subjects_counts_adata)
        all_counts.append(counts)
        all_metadata.append(metadata)

    ## concatenate
    mdd_subjects_counts_adata = anndata.concat(all_mdd_subjects_counts_adata, axis=0)
    counts = np.concatenate(all_counts, axis=0)
    metadata = pd.concat(all_metadata, axis=0)

    ## run pyDESeq2
    #per_celltype = run_pyDESeq2_per_celltype(counts, metadata, 'most_common_cluster')
    if test_type == 'split':
        per_celltype = run_pyDESeq2_contrasts(counts, metadata, 'Condition')
    elif test_type == 'all':
        per_celltype = run_pyDESeq2_all_celltypes(counts, metadata)

    return per_celltype

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

            rna_subject_obs = pd.DataFrame(
                np.hstack([batch, subject_condition, sex, celltype]).reshape(1, -1),
                columns=['Batch', rna_condition_key, rna_sex_key, rna_celltype_key],
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

def run_pyDESeq2_contrasts(counts, metadata, rna_condition_key, save_dir=None):

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

def run_pyDESeq2_all_celltypes(counts, metadata):

    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~ Batch + Condition",
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    stat = DeseqStats(dds, contrast=("Condition", "Case", "Control"))
    stat.run_wald_test()
    stat.summary()
    results = stat.results_df

    return results


def run_pychromVAR(term_linked_genes, term_linked_peaks, peak_gene_links):
    from pyjaspar import jaspardb
    import pychromvar as pc
    from gseapy import enrichr
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    import pickle
    def tree(): return defaultdict(tree)

    from scipy.sparse import csr_matrix
    def build_mask_from_intervals(var_names, intervals_np):
        """
        Returns: (mask_bool, n_provided, n_matched)
        """
        idx = {p: i for i, p in enumerate(var_names)}
        hits = [idx[s] for s in map(str, intervals_np) if s in idx]
        mask = np.zeros(len(var_names), dtype=bool)
        if hits:
            mask[np.asarray(hits, dtype=int)] = True
        return mask, len(intervals_np), len(hits)

    def make_peakset_annotation(var_names, sets_dict):
        """
        sets_dict: {"setname": np.array([...interval strings...]), ...}
        Returns: (CSR n_peaks × n_sets, kept_names[list])
        """
        cols = []
        names = []
        for name, ivals in sets_dict.items():
            mask, n_in, n_hit = build_mask_from_intervals(var_names, ivals)
            if n_hit == 0:
                print(f"[warn] {name}: 0/{n_in} intervals matched var_names — skipping")
                continue
            cols.append(mask.astype(np.uint8)[:, None])
            names.append(name)
        if not cols:
            raise ValueError("No intervals matched var_names for any set.")
        M = np.hstack(cols)  # small dense then CSR
        return csr_matrix(M), names

    def deviations_to_df(dev_adata, row_name="sample", col_name="annotation"):
        """
        Convert the AnnData returned by compute_deviations into a tidy DataFrame (samples × annotations).
        """
        df = pd.DataFrame(
            dev_adata.X,
            index=getattr(dev_adata, "obs_names", None),
            columns=getattr(dev_adata, "var_names", None),
        )
        df.index.name = row_name
        df.columns.name = col_name
        return df

    ## pseudobulk ATAC data
    celltype_donor_condition_df = mdd_atac_broad_sub.obs[['ClustersMapped','BrainID','condition']]
    celltype_donor_condition_df['celltype_donor'] = celltype_donor_condition_df[['ClustersMapped','BrainID']].apply(lambda x: f"{x[0]}_{x[1]}", axis=1)
    pseudobulk_groups = pd.get_dummies(celltype_donor_condition_df['celltype_donor'], sparse=False).astype(int)
    pseudobulk_weights = pseudobulk_groups.sum(axis=0) / pseudobulk_groups.sum().sum()
    pseudobulk_matrix = mdd_atac_broad_sub.X.T.dot(pseudobulk_groups.values)
    pseudobulk_matrix = pseudobulk_matrix.T
    mdd_atac_broad_sub_pseudobulked = anndata.AnnData(X=pseudobulk_matrix, obs=pseudobulk_groups.columns.to_frame(), var=mdd_atac_broad_sub.var)
    mdd_atac_broad_sub_pseudobulked.obs['n_cells'] = pseudobulk_groups.sum(axis=0)
    mdd_atac_broad_sub_pseudobulked = mdd_atac_broad_sub_pseudobulked[mdd_atac_broad_sub_pseudobulked.obs['n_cells'] > 50]

    celltype_donor_condition_df = celltype_donor_condition_df.drop_duplicates(subset='celltype_donor')
    mdd_atac_broad_sub_pseudobulked.obs = mdd_atac_broad_sub_pseudobulked.obs.merge(celltype_donor_condition_df, left_index=True, right_on='celltype_donor', how='left').drop(columns=0)

    ## subset peaks to reduce memory usage (if enabled)
    subset_peaks = False

    if subset_peaks:
        peaks = peak_gene_links['peak'].unique()
        keep_peaks = \
            mdd_atac_broad_sub_pseudobulked.var_names[mdd_atac_broad_sub_pseudobulked.var_names.isin(peaks)].tolist() + \
            mdd_atac_broad_sub_pseudobulked.var_names.to_series().sample(120000).tolist()
        keep_peaks = np.unique(keep_peaks)

        mdd_atac_broad_sub_pseudobulked = mdd_atac_broad_sub_pseudobulked[:,mdd_atac_broad_sub_pseudobulked.var_names.isin(keep_peaks)]
        adata = mdd_atac_broad_sub_pseudobulked.copy()
    else:
        adata = mdd_atac_broad_sub_pseudobulked.copy()

    #pc.get_genome("hg38", output_dir="./")
    pc.add_peak_seq(adata, genome_file=os.path.join(os.environ['DATAPATH'], 'hg38.fa'), delimiter=':|-')
    pc.add_gc_bias(adata)
    pc.get_bg_peaks(adata) # ~5 minutes

    ## get motifs
    jdb_obj = jaspardb(release='JASPAR2020')
    motifs = jdb_obj.fetch_motifs(
        collection = 'CORE',
        tax_group = ['vertebrates'])

    pc.match_motif(adata, motifs=motifs)

    orig_M = adata.varm["motif_match"]
    orig_names = np.asarray(adata.uns["motif_name"]).copy()

    motif_backup = orig_M.copy()
    name_backup  = orig_names.copy()

    dfs_per_set = []
    n_matching_peaks_per_set = {}
    for set_name, ivals in term_linked_peaks.items():

        print(f"Processing {set_name} (n={len(ivals)})")

        # 1) build mask for this set
        mask, n_in, n_hit = build_mask_from_intervals(adata.var_names, ivals)
        if n_hit == 0:
            print(f"[warn] {set_name}: 0/{n_in} intervals matched var_names — skipping")
            continue

        # 2) intersect motif annotations with this set
        #M_subset = csr_matrix(orig_M.multiply(mask[:, None]))      # zero peaks outside the set
        M_subset = orig_M * mask[:, None]

        # drop motifs with no peaks in this set
        keep = np.asarray(M_subset.sum(axis=0)).ravel() > 0
        if keep.sum() == 0:
            print(f"[warn] {set_name}: no motifs overlap the set — skipping")
            continue
        M_subset = M_subset[:, keep]
        names_subset = orig_names[keep].astype(object)

        # 3) swap in, compute, restore
        adata.varm["motif_match"] = M_subset
        adata.uns["motif_name"]   = names_subset

        dev_this = pc.compute_deviations(adata, n_jobs=10)

        # tidy DF; tag columns with the set name to keep them distinct
        df_this = deviations_to_df(dev_this, col_name="motif")
        df_this.columns = [f"{m}__in__{set_name}" for m in df_this.columns]
        dfs_per_set.append(df_this)

        overlap = pd.Series(M_subset.sum(axis=0), index=df_this.columns, name='Overlap').astype(str) + f"/{len(M_subset)}"
        n_matching_peaks_per_set[set_name] = overlap

    # restore original motif table
    adata.varm["motif_match"] = motif_backup
    adata.uns["motif_name"]   = name_backup

    # Join across sets (samples × (motif×set))
    df_dev_motif_in_sets = pd.concat(dfs_per_set, axis=1).sort_index(axis=1)
    #w = pseudobulk_weights.values[:,None]
    #assert df_dev_motif_in_sets.index.equals(pseudobulk_weights.index)

    celltype_devs_dict = {}
    all_motif_names = []
    for celltype in term_linked_genes.keys():

        celltype_dev = df_dev_motif_in_sets.loc[:, df_dev_motif_in_sets.columns.str.contains(f"__in__{celltype}")]
        celltype_dev.columns = celltype_dev.columns.str.split('__in__').str[0]
        all_motif_names.extend(celltype_dev.columns.tolist())

        Z = celltype_dev.values
        #stouffer_Z = np.sum(Z * w, axis=0) / np.sqrt(np.sum(w**2))
        stouffer_Z = np.sum(Z, axis=0) / np.sqrt(len(Z))
        p = norm.sf(abs(stouffer_Z)) * 2  # Two-tailed p-value
        reject, q, _, _ = multipletests(p, method='fdr_bh')

        celltype_dev_adata = anndata.AnnData(X=celltype_dev.values, var=celltype_dev.columns.to_frame(), obs=celltype_dev.index.to_frame())
        celltype_dev_adata.var['q_value'] = q
        celltype_dev_adata.var['p_value'] = p
        celltype_dev_adata.var['stouffer_Z'] = stouffer_Z
        celltype_dev_adata.var['reject'] = reject

        celltype_devs_dict[celltype] = celltype_dev_adata

        print(f"Hits in {celltype}: {celltype_dev_adata.var['reject'].sum()}")

    return adata

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

if __name__ == '__main__':
    main(subset_type=None)