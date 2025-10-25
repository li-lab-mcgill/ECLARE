from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from collections import defaultdict

import pandas as pd
import numpy as np
import anndata
import os
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

    ## load GWAS hits
    zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
    gwas_hits = pd.read_excel(zhu_supp_tables, sheet_name='Table S12', header=2)

    ## subset data
    #subset_type = 'gwas_and_hvgs'
    subset_type = 'gwas'
    mdd_rna_scaled_sub = subset_data(mdd_rna_scaled_sub, gwas_hits, subset_type)
    mdd_atac_sub = subset_data(mdd_atac_sub, gwas_hits, subset_type, hvg_genes=mdd_rna_scaled_sub.var_names.tolist())

    for sex in ['male', 'female']:

        rna_per_celltype = run_pyDESeq2_on_celltypes(mdd_rna_scaled_sub, sex)
        atac_per_celltype = run_pyDESeq2_on_celltypes(mdd_atac_sub, sex)



        def get_significant_results(per_celltype, genes, p_metric='pvalue'):
            results = pd.concat([
                df.assign(celltype=celltype, gene=genes).assign(
                    km=genes.map(gwas_hits.set_index('Target gene name')['km'].to_dict()),
                    traits=genes.map(gwas_hits.groupby('Target gene name')['Trait'].unique().str.join('_').to_dict())
                    ).set_index('celltype') 
                for celltype, df in per_celltype.items() if df is not None
            ])

            significant_results = results[results[p_metric] < 0.05]

            return significant_results

        def extract_results(significant_results):
            mdd_significant_results = significant_results.loc[significant_results.traits.str.contains('MDD').fillna(False)]
            sig_genes_per_celltype = significant_results.groupby('celltype')['gene'].apply(np.sort).to_dict()
            sig_km_per_celltype = significant_results.dropna(subset=['km']).groupby('celltype')['km'].unique().to_dict()
            return mdd_significant_results, sig_genes_per_celltype, sig_km_per_celltype

        rna_significant_results = get_significant_results(rna_per_celltype, mdd_rna_scaled_sub.raw.var_names, p_metric='pvalue')
        atac_significant_results = get_significant_results(atac_per_celltype, mdd_atac_sub.raw.var_names, p_metric='pvalue')

        rna_mdd_significant_results, rna_sig_genes_per_celltype, rna_sig_km_per_celltype = extract_results(rna_significant_results)
        atac_mdd_significant_results, atac_sig_genes_per_celltype, atac_sig_km_per_celltype = extract_results(atac_significant_results)


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

    '''
    ## get results and volcano plot
    results = stat_res.results_df
    results['signif_padj'] = results['padj'] < 0.05
    results['signif_lfc'] = results['log2FoldChange'].abs() > 1.5
    results['signif'] = results['signif_padj'] & results['signif_lfc']
    results['-log10(padj)'] = -np.log10(results['padj'])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.scatterplot(data=results, x='log2FoldChange', y='-log10(padj)', hue='signif_padj', marker='o', alpha=0.5)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'volcano_plot.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    ## extract significant genes
    significant_genes = mdd_subjects_counts_adata.var_names[results['signif_padj']]
    mdd_subjects_counts_adata.var['signif_padj'] = False
    mdd_subjects_counts_adata.var.loc[significant_genes, 'signif_padj'] = True

    ## violin plot
    df = mdd_subjects_counts_adata[:,mdd_subjects_counts_adata.var_names.isin(significant_genes[:10])].to_df()
    df = df.reset_index()
    df = pd.melt(df, id_vars=['index'], var_name='gene', value_name='expression')
    df = df.merge(mdd_subjects_counts_adata.obs, left_on='index', right_index=True)
    df = df.sort_values(rna_condition_key, ascending=False) # forces controls to be listed first, putting controls on the left-hand violin plots

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=df, x='gene', y='expression', hue=rna_condition_key, split=True, inner=None, cut=0, ax=ax)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'violin_plot.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    return mdd_subjects_counts_adata, results, significant_genes
    '''
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