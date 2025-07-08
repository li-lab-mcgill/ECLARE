#%% set env variables
import os
import sys

# Check if environment variables are already set
eclare_root = os.environ.get('ECLARE_ROOT')
outpath = os.environ.get('OUTPATH')
datapath = os.environ.get('DATAPATH')

# Print status of environment variables
if all([eclare_root, outpath, datapath]):
    print(f"Environment variables already set:")
    print(f"ECLARE_ROOT: {eclare_root}")
    print(f"OUTPATH: {outpath}")
    print(f"DATAPATH: {datapath}")
else:
    print(f"Missing environment variables")

    config_path = '../config'
    sys.path.insert(0, config_path)

    from export_env_variables import export_env_variables
    export_env_variables(config_path)

#%% import data

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from eclare.post_hoc_utils import tree

## Create dict for methods and job_ids
methods_id_dict = {'eclare_mdd': ['16103846']}
base_output_dir = os.path.join(os.environ['OUTPATH'], f"enrichment_analyses_{methods_id_dict['eclare_mdd'][0]}")
output_dir = base_output_dir + '_41'

# Load all saved dictionaries
dicts_to_load = [
    'X_rna_dict',
    'X_atac_dict',
    'pydeseq2_results_dict',
    'significant_genes_dict',
    'overlapping_target_genes_dict',
    'overlapping_tfs_dict',
    'scompreg_loglikelihoods_dict',
    'std_errs_dict',
    'tg_expressions_dict',
    'tfrps_dict',
    'tfrp_predictions_dict',
    'slopes_dict',
    'intercepts_dict',
    'intercept_stderrs_dict',
    'enrs_dict',
    'magma_results_dict',
    'mean_grn_df_filtered_dict',
    'gene_set_scores_dict',
    'ttest_comp_df_dict'
]

loaded_dicts = {}
for dict_name in dicts_to_load:
    dict_path = os.path.join(output_dir, f"{dict_name}.pkl")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            loaded_dicts[dict_name] = pickle.load(f)
        print(f"Loaded {dict_name}")
    else:
        print(f"Warning: {dict_path} not found")

# Unpack loaded dictionaries into individual variables
X_rna_dict = loaded_dicts.get('X_rna_dict', tree())
X_atac_dict = loaded_dicts.get('X_atac_dict', tree())
pydeseq2_results_dict = loaded_dicts.get('pydeseq2_results_dict', tree())
significant_genes_dict = loaded_dicts.get('significant_genes_dict', tree())
overlapping_target_genes_dict = loaded_dicts.get('overlapping_target_genes_dict', tree())
overlapping_tfs_dict = loaded_dicts.get('overlapping_tfs_dict', tree())
scompreg_loglikelihoods_dict = loaded_dicts.get('scompreg_loglikelihoods_dict', tree())
std_errs_dict = loaded_dicts.get('std_errs_dict', tree())
tg_expressions_dict = loaded_dicts.get('tg_expressions_dict', tree())
tfrps_dict = loaded_dicts.get('tfrps_dict', tree())
tfrp_predictions_dict = loaded_dicts.get('tfrp_predictions_dict', tree())
slopes_dict = loaded_dicts.get('slopes_dict', tree())
intercepts_dict = loaded_dicts.get('intercepts_dict', tree())
intercept_stderrs_dict = loaded_dicts.get('intercept_stderrs_dict', tree())
enrs_dict = loaded_dicts.get('enrs_dict', tree())
magma_results_dict = loaded_dicts.get('magma_results_dict', tree())
mean_grn_df_filtered_dict = loaded_dicts.get('mean_grn_df_filtered_dict', tree())
gene_set_scores_dict = loaded_dicts.get('gene_set_scores_dict', tree())
ttest_comp_df_dict = loaded_dicts.get('ttest_comp_df_dict', tree())

## Load CSV files and other file types
shared_TF_TG_pairs_df = pd.read_csv(os.path.join(output_dir, 'shared_TF_TG_pairs.csv'))

enrs_mdd_dn_hits_df = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_hits_df.csv'))
enrs_mdd_dn_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_tfs_multiple_hits.csv'))

all_sccompreg_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_tfs_multiple_hits.csv'))
all_sccompreg_hits_df = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_hits_df.csv'))

pydeseq2_match_length_genes_hits_df = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_hits_df.csv'))
pydeseq2_match_length_genes_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_tfs_multiple_hits.csv'))

#%% Extract basic information
unique_sexes = list(enrs_dict.keys())
unique_celltypes = list(enrs_dict[unique_sexes[0]].keys())

#%% EnrichR results for MDD-DN pathway


def get_enrs_mdd_dn_hits_df(enrs_dict, filtered_type='All LR'):

    enrs_mdd_dn_hits_df = pd.DataFrame(columns=['ngenes', 'padj', 'mlog10_padj'], 
                                    index=pd.MultiIndex.from_product([unique_sexes, unique_celltypes], names=['sex', 'celltype']))

    for sex in unique_sexes:
        for celltype in unique_celltypes:

            enrs = enrs_dict[sex][celltype][filtered_type]
            enrs_mdd_dn = enrs[enrs.Term == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']

            if len(enrs_mdd_dn) == 0:
                continue

            ngenes = int(enrs_mdd_dn['Overlap'].str.split('/').item()[0])
            padj = enrs_mdd_dn['Adjusted P-value'].item()
            mlog10_padj = -np.log10(padj)

            enrs_mdd_dn_hits_df.loc[(sex, celltype), 'ngenes'] = ngenes
            enrs_mdd_dn_hits_df.loc[(sex, celltype), 'padj'] = padj
            enrs_mdd_dn_hits_df.loc[(sex, celltype), 'mlog10_padj'] = mlog10_padj


    enrs_mdd_dn_hits_df.reset_index(inplace=True)
    enrs_mdd_dn_hits_df['mlog10_padj'] = enrs_mdd_dn_hits_df['mlog10_padj'].fillna(0)
    enrs_mdd_dn_hits_df['ngenes'] = enrs_mdd_dn_hits_df['ngenes'].fillna(0)
    enrs_mdd_dn_hits_df['size_ngenes'] = (enrs_mdd_dn_hits_df['ngenes']/enrs_mdd_dn_hits_df['ngenes'].max())**2 * 500

    return enrs_mdd_dn_hits_df

def plot_enrichment_significance(enrs_mdd_dn_hits_df, figsize=(6, 2)):


    # Ensure x-axis is in alphabetical order
    enrs_mdd_dn_hits_df = enrs_mdd_dn_hits_df.sort_values('celltype')

    # Single plot with three marker types
    fig, ax = plt.subplots(figsize=figsize)

    # Define thresholds
    sig_threshold = -np.log10(0.05)
    bonferroni_threshold = 0.05 / len(enrs_mdd_dn_hits_df)
    bonferroni_threshold_mlog10 = -np.log10(bonferroni_threshold)

    # Split data by significance levels
    non_sig_data = enrs_mdd_dn_hits_df[enrs_mdd_dn_hits_df['mlog10_padj'] == 0]
    sig_not_bonferroni_data = enrs_mdd_dn_hits_df[(enrs_mdd_dn_hits_df['mlog10_padj'] > sig_threshold) & (enrs_mdd_dn_hits_df['mlog10_padj'] <= bonferroni_threshold_mlog10)]
    bonferroni_sig_data = enrs_mdd_dn_hits_df[enrs_mdd_dn_hits_df['mlog10_padj'] > bonferroni_threshold_mlog10]

    # Fix range of colormap so applies to all data
    vmin = min(enrs_mdd_dn_hits_df['mlog10_padj'].min(), sig_not_bonferroni_data['mlog10_padj'].min(), bonferroni_sig_data['mlog10_padj'].min())
    vmax = max(enrs_mdd_dn_hits_df['mlog10_padj'].max(), sig_not_bonferroni_data['mlog10_padj'].max(), bonferroni_sig_data['mlog10_padj'].max())
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot empty scatter plot for colorbar
    ax.scatter(enrs_mdd_dn_hits_df['celltype'], enrs_mdd_dn_hits_df['sex'], alpha=0, s=0)

    # Plot non-significant as grey crosses
    if len(non_sig_data) > 0:
        ax.scatter(non_sig_data['celltype'], non_sig_data['sex'], 
                   color='grey', alpha=0.7, 
                   marker='+', linewidths=1, label='Non-significant')

    # Plot significant but not Bonferroni as circles
    if len(sig_not_bonferroni_data) > 0:
        ax.scatter(sig_not_bonferroni_data['celltype'], sig_not_bonferroni_data['sex'], 
                   s=sig_not_bonferroni_data['size_ngenes'], c=sig_not_bonferroni_data['mlog10_padj'], 
                   cmap=cmap, norm=norm, alpha=0.6, marker='o', hatch=3*'.', label='Significant (p<0.05)', edgecolors='black')

    # Plot Bonferroni significant as stars
    if len(bonferroni_sig_data) > 0:
        ax.scatter(bonferroni_sig_data['celltype'], bonferroni_sig_data['sex'], 
                   s=bonferroni_sig_data['size_ngenes'], c=bonferroni_sig_data['mlog10_padj'], 
                   cmap=cmap, norm=norm, alpha=0.8, marker='o', edgecolors='black',
                   label=f'Bonferroni significant (p<{bonferroni_threshold:.3f})')

    ax.set_ylim(-0.5, 1.5)

    # Add colorbar for mlog10_padj
    scatter = ax.scatter(bonferroni_sig_data['celltype'], bonferroni_sig_data['sex'], c=bonferroni_sig_data['mlog10_padj'], cmap='viridis', alpha=0.6, s=0)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('$-log_{10}(p)$', pad=10, fontsize=9)

    # Add size legend for ngenes
    # Create dummy scatter plot for size legend
    sizes_describe = enrs_mdd_dn_hits_df[['size_ngenes','ngenes']].loc[enrs_mdd_dn_hits_df['ngenes']>0].describe()
    sizes = [sizes_describe.loc['min','size_ngenes'], sizes_describe.loc['50%','size_ngenes'], sizes_describe.loc['max','size_ngenes']]
    size_labels = np.array([sizes_describe.loc['min','ngenes'], sizes_describe.loc['50%','ngenes'], sizes_describe.loc['max','ngenes']]).astype(int)

    # Create a separate axis for the size legend
    ax2 = fig.add_axes([0.88, 0.15, 0.2, 0.8])  # Position for size legend
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1, 2 * len(sizes))  # More vertical space

    for i, (size, label) in enumerate(zip(sizes, size_labels)):
        y = i * 2  # Spread out vertically
        ax2.scatter(0.5, y, s=size, color='grey', alpha=0.7, edgecolors='black')
        ax2.text(0.7, y, f'{label} genes', va='center', fontsize=8)

    ax2.set_title('Num. genes', fontsize=9, pad=1, x=0.7, y=0.9)
    ax2.axis('off')

    # Ensure x-axis is in alphabetical order
    celltype_order = sorted(enrs_mdd_dn_hits_df['celltype'].unique())
    ax.set_xticks(celltype_order)
    ax.set_xticklabels(celltype_order, rotation=45, ha='right')

    fig.tight_layout()
    fig.suptitle('EnrichR results for MDD-DN pathway', fontsize=10, y=0.96)
    
    return fig, ax

# Call the function
enrs_mdd_dn_hits_lr_df = get_enrs_mdd_dn_hits_df(enrs_dict, filtered_type='All LR')
enrs_mdd_dn_hits_lr_fig, enrs_mdd_dn_hits_lr_ax = plot_enrichment_significance(enrs_mdd_dn_hits_lr_df)

enrs_mdd_dn_hits_deg_df = get_enrs_mdd_dn_hits_df(enrs_dict, filtered_type='DEG (matched length)')
#enrs_mdd_dn_hits_deg_fig, enrs_mdd_dn_hits_deg_ax = plot_enrichment_significance(enrs_mdd_dn_hits_deg_df)


#%% Differential expression of MDD-DN pathway genes

def get_ttest_df(ttest_comp_df_dict, pathway_name='ASTON_MAJOR_DEPRESSIVE_DISORDER_DN'):

    ttest_comp_results_df = pd.DataFrame(columns=['tstat', 'pvalue', 'df'], 
                                    index=pd.MultiIndex.from_product([unique_sexes, unique_celltypes], names=['sex', 'celltype']))

    for sex in unique_sexes:
        for celltype in unique_celltypes:

            ttests = ttest_comp_df_dict[sex][celltype]
            ttest_mdd_dn = ttests[ttests.index == pathway_name]

            if len(ttest_mdd_dn) == 0:
                continue

            tstat = ttest_mdd_dn['tstat'].item()
            pvalue = ttest_mdd_dn['pvalue'].item()
            df = ttest_mdd_dn['df'].item()

            if pvalue > 0:
                mlog10_pvalue = -np.log10(pvalue)
            else:
                mlog10_pvalue = -np.log10(5) + 324 # "If you see a p-value of exactly 0.0, it means the computed survival function underflowed below ∼5×10⁻³²⁴."

            ttest_comp_results_df.loc[(sex, celltype), 'tstat'] = tstat
            ttest_comp_results_df.loc[(sex, celltype), 'pvalue'] = pvalue
            ttest_comp_results_df.loc[(sex, celltype), 'mlog10_pvalue'] = mlog10_pvalue
            ttest_comp_results_df.loc[(sex, celltype), 'df'] = df

    ttest_comp_results_df.reset_index(inplace=True)
    ttest_comp_results_df['tstat'] = ttest_comp_results_df['tstat'].fillna(0)
    ttest_comp_results_df['mlog10_pvalue'] = ttest_comp_results_df['mlog10_pvalue'].fillna(0)
    ttest_comp_results_df['size_mlog10_pvalue'] = np.log1p(ttest_comp_results_df['mlog10_pvalue']) * 100

    return ttest_comp_results_df

def plot_ttest(ttest_mdd_dn_df):

    # Ensure x-axis is in alphabetical order
    ttest_mdd_dn_df = ttest_mdd_dn_df.sort_values('celltype')

    # Single plot with three marker types
    fig, ax = plt.subplots(figsize=(6, 2))

    # Empty scatter plot for colorbar
    ax.scatter(ttest_mdd_dn_df['celltype'], ttest_mdd_dn_df['sex'], alpha=0, s=0)

    # Plot non-significant as grey crosses
    non_sig_data = ttest_mdd_dn_df[ttest_mdd_dn_df['mlog10_pvalue'] == 0]
    if len(non_sig_data) > 0:
        ax.scatter(non_sig_data['celltype'], non_sig_data['sex'], 
                   color='grey', alpha=0.7, marker='+', linewidths=1)

    # Create a diverging colormap with 0 centered

    # Find min and max tstat for normalization
    vmax = max(abs(ttest_mdd_dn_df['tstat']))
    vmin = -vmax
    norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax)

    # Main scatter plot
    scatter = ax.scatter(ttest_mdd_dn_df['celltype'], ttest_mdd_dn_df['sex'], 
               c=ttest_mdd_dn_df['tstat'], cmap='PuOr', norm=norm, alpha=0.6, s=ttest_mdd_dn_df['size_mlog10_pvalue'], edgecolors='black')
    
    if pathway_name == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN':
        assert (ttest_mdd_dn_df['mlog10_pvalue'][ttest_mdd_dn_df['mlog10_pvalue']>0] > -np.log10(0.05)).all()
        assert (ttest_mdd_dn_df['mlog10_pvalue'][ttest_mdd_dn_df['mlog10_pvalue']>0] > -np.log10(0.05 / len(ttest_mdd_dn_df))).all()
    
    ax.set_ylim(-0.5, 1.5)

    # Add colorbar for tstat
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('t-stat', pad=10, fontsize=9)

    # Add size legend for mlog10_pvalue
    sizes_describe = ttest_mdd_dn_df[['size_mlog10_pvalue','mlog10_pvalue']].loc[ttest_mdd_dn_df['mlog10_pvalue']>0].describe()
    sizes = [sizes_describe.loc['min','size_mlog10_pvalue'], sizes_describe.loc['50%','size_mlog10_pvalue'], sizes_describe.loc['max','size_mlog10_pvalue']]
    size_labels = np.array([sizes_describe.loc['min','mlog10_pvalue'], sizes_describe.loc['50%','mlog10_pvalue'], sizes_describe.loc['max','mlog10_pvalue']]).astype(int)

    # Create a separate axis for the size legend
    ax2 = fig.add_axes([0.88, 0.1, 0.2, 0.8])  # Position for size legend
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.2, 2 * len(sizes))  # More vertical space

    for i, (size, label) in enumerate(zip(sizes, size_labels)):
        y = i * 2  # Spread out vertically
        ax2.scatter(0.5, y, s=size, color='grey', alpha=0.7, edgecolors='black')
        ax2.text(0.7, y, f'{label}', va='center', fontsize=8)

    ax2.set_title('$-log_{10}(p)$', fontsize=9, pad=0, loc='left', x=0.3, y=0.96)
    ax2.axis('off')

    # Ensure x-axis is in alphabetical order
    celltype_order = sorted(ttest_mdd_dn_df['celltype'].unique())
    ax.set_xticks(celltype_order)
    ax.set_xticklabels(celltype_order, rotation=45, ha='right')

    fig.tight_layout()
    fig.suptitle('Differential expression of MDD-DN pathway genes', fontsize=10, y=0.96)
    
    return fig, ax

ttest_mdd_dn_df = get_ttest_df(ttest_comp_df_dict, pathway_name='ASTON_MAJOR_DEPRESSIVE_DISORDER_DN')
ttest_mdd_dn_fig, ttest_mdd_dn_ax = plot_ttest(ttest_mdd_dn_df)


#%% Module scores for pathways of interest

pathways_of_interest = [
    "ASTON_MAJOR_DEPRESSIVE_DISORDER_DN",
    "Oligodendrocyte_Mature_Darmanis_PNAS_2015",
    "FAN_EMBRYONIC_CTX_OLIG",
    "LEIN_OLIGODENDROCYTE_MARKERS",
    "ZHONG_PFC_C4_PTGDS_POS_OPC",
    "DESCARTES_FETAL_CEREBRUM_OLIGODENDROCYTES",
    #"COLIN_PILOCYTIC_ASTROCYTOMA_VS_GLIOBLASTOMA_UP",
    "DESCARTES_MAIN_FETAL_OLIGODENDROCYTES",
    "DESCARTES_FETAL_CEREBELLUM_OLIGODENDROCYTES",
    "LU_AGING_BRAIN_UP",
    #"DURANTE_ADULT_OLFACTORY_NEUROEPITHELIUM_OLFACTORY_ENSHEATHING_GLIA",
    "DESCARTES_MAIN_FETAL_SCHWANN_CELLS",
    "Oligodendrocyte_All_Zeisel_Science_2015",
    "GOBERT_OLIGODENDROCYTE_DIFFERENTIATION_DN",
    "BLALOCK_ALZHEIMERS_DISEASE_UP"
]

for pathway_name in pathways_of_interest:
    ttest_df = get_ttest_df(ttest_comp_df_dict, pathway_name=pathway_name)
    ttest_df_fig, ttest_df_ax = plot_ttest(ttest_df)








# %%
