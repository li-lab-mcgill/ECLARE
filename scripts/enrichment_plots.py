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
from matplotlib.colors import SymLogNorm, ListedColormap
import json
import seaborn as sns

from eclare.post_hoc_utils import tree, get_brain_gmt, do_enrichr

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
    'mean_grn_df_filtered_pruned_dict',
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
mean_grn_df_filtered_pruned_dict = loaded_dicts.get('mean_grn_df_filtered_pruned_dict', tree())
gene_set_scores_dict = loaded_dicts.get('gene_set_scores_dict', tree())
ttest_comp_df_dict = loaded_dicts.get('ttest_comp_df_dict', tree())

## Load CSV files and other file types
shared_TF_TG_pairs_df = pd.read_csv(os.path.join(output_dir, 'shared_TF_TG_pairs.csv'))
with open(os.path.join(output_dir, 'enriched_TF_TG_pairs_dict.json'), 'r') as f:
    enriched_TF_TG_pairs_dict = json.load(f)

enrs_mdd_dn_hits_df = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_hits_df.csv'))
enrs_mdd_dn_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'enrs_mdd_dn_tfs_multiple_hits.csv'))

all_sccompreg_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_tfs_multiple_hits.csv'))
all_sccompreg_hits_df = pd.read_csv(os.path.join(output_dir, 'all_sccompreg_hits_df.csv'))

pydeseq2_match_length_genes_hits_df = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_hits_df.csv'))
pydeseq2_match_length_genes_tfs_multiple_hits = pd.read_csv(os.path.join(output_dir, 'pydeseq2_match_length_genes_tfs_multiple_hits.csv'))

#%% Load GREAT results from R script output

great_csv_outputs_path = os.path.join(output_dir, 'great_csv_outputs')
great_csv_outputs_files = os.listdir(great_csv_outputs_path)

great_results_df_dict = tree()
for file in great_csv_outputs_files:
    if file.endswith('.csv'):
        great_results_df = pd.read_csv(os.path.join(great_csv_outputs_path, file), index_col=0)

        sex = file.split('_')[0]
        celltype = file.split('_')[1].split('.')[0]
        great_results_df_dict[sex][celltype] = great_results_df


#%% Extract basic information
unique_sexes = list(enrs_dict.keys())
unique_celltypes = list(enrs_dict[unique_sexes[0]].keys())

brain_gmt_cortical, brain_gmt_cortical_wGO = get_brain_gmt()

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]


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

def plot_enrichment_significance(enrs_mdd_dn_hits_df, title='', figsize=(6, 2)):

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
    sig_data = enrs_mdd_dn_hits_df[enrs_mdd_dn_hits_df['mlog10_padj'] > 0]
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
                   cmap=cmap, norm=norm, alpha=0.6, marker='o', hatch=3*'//', edgecolors='white', label='Significant (p<0.05)')

    # Plot Bonferroni significant as stars
    if len(bonferroni_sig_data) > 0:
        ax.scatter(bonferroni_sig_data['celltype'], bonferroni_sig_data['sex'], 
                   s=bonferroni_sig_data['size_ngenes'], c=bonferroni_sig_data['mlog10_padj'], 
                   cmap=cmap, norm=norm, alpha=0.8, marker='o', edgecolors='black',
                   label=f'Bonferroni significant (p<{bonferroni_threshold:.3f})')

    ax.set_ylim(-0.5, 1.5)

    # Add colorbar for mlog10_padj
    scatter = ax.scatter(sig_data['celltype'], sig_data['sex'], c=sig_data['mlog10_padj'], cmap='viridis', alpha=0.6, s=0)
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
    fig.suptitle(title, fontsize=10, y=0.96)
    
    return fig, ax

# Call the function
enrs_mdd_dn_hits_lr_df = get_enrs_mdd_dn_hits_df(enrs_dict, filtered_type='All LR')
enrs_mdd_dn_hits_lr_fig, enrs_mdd_dn_hits_lr_ax = plot_enrichment_significance(enrs_mdd_dn_hits_lr_df)

enrs_mdd_dn_hits_deg_df = get_enrs_mdd_dn_hits_df(enrs_dict, filtered_type='DEG (matched length)')
#enrs_mdd_dn_hits_deg_fig, enrs_mdd_dn_hits_deg_ax = plot_enrichment_significance(enrs_mdd_dn_hits_deg_df)

#enrs_mdd_dn_hits_lr_fig.savefig(os.path.join(output_dir, 'enrs_mdd_dn_hits_lr_fig.png'),  dpi=300, bbox_inches='tight')

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
    ttest_comp_results_df['size_mlog10_pvalue'] = np.log1p(ttest_comp_results_df['mlog10_pvalue']) * 60

    return ttest_comp_results_df

def plot_ttest(ttest_mdd_dn_df, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2))

    # Define thresholds
    sig_threshold = -np.log10(0.05)
    bonferroni_threshold = 0.05 / len(ttest_mdd_dn_df)
    bonferroni_threshold_mlog10 = -np.log10(bonferroni_threshold)

    # Further filter to only include celltypes with significant FDR p-values by setting mlog10_pvalue to 0
    ttest_mdd_dn_df.loc[ttest_mdd_dn_df['mlog10_pvalue'] < sig_threshold, 'mlog10_pvalue'] = 0
    ttest_mdd_dn_df.loc[ttest_mdd_dn_df['mlog10_pvalue'] < sig_threshold, 'size_mlog10_pvalue'] = 0

    # Ensure x-axis is in alphabetical order
    ttest_mdd_dn_df = ttest_mdd_dn_df.sort_values('celltype')

    # Split data by significance levels
    non_sig_data = ttest_mdd_dn_df[ttest_mdd_dn_df['mlog10_pvalue'] == 0]
    sig_not_bonferroni_data = ttest_mdd_dn_df[(ttest_mdd_dn_df['mlog10_pvalue'] > sig_threshold) & (ttest_mdd_dn_df['mlog10_pvalue'] <= bonferroni_threshold_mlog10)]
    bonferroni_sig_data = ttest_mdd_dn_df[ttest_mdd_dn_df['mlog10_pvalue'] > bonferroni_threshold_mlog10]
    sig_data = ttest_mdd_dn_df[ttest_mdd_dn_df['mlog10_pvalue'] > 0]

    # Empty scatter plot for colorbar
    ax.scatter(ttest_mdd_dn_df['celltype'], ttest_mdd_dn_df['sex'], alpha=0, s=0)

    # Plot non-significant as grey crosses
    if len(non_sig_data) > 0:
        ax.scatter(non_sig_data['celltype'], non_sig_data['sex'], 
                   color='grey', alpha=0.7, marker='+', linewidths=1, label='Non-significant')

    # Create a diverging colormap with 0 centered
    vmax = max(abs(ttest_mdd_dn_df['tstat']))
    vmin = -vmax
    norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax)
    cmap = ListedColormap(['orange', 'purple']) # recreate PuOr colormap

    # Plot significant but not Bonferroni as circles
    if len(sig_not_bonferroni_data) > 0:
        ax.scatter(sig_not_bonferroni_data['celltype'], sig_not_bonferroni_data['sex'], 
                   s=sig_not_bonferroni_data['size_mlog10_pvalue'], c=sig_not_bonferroni_data['tstat'], 
                   cmap=cmap, norm=norm, alpha=0.6, marker='s', hatch=3*'//', edgecolors='white', label='Significant (p<0.05)')

    # Plot Bonferroni significant as stars
    if len(bonferroni_sig_data) > 0:
        ax.scatter(bonferroni_sig_data['celltype'], bonferroni_sig_data['sex'], 
                   s=bonferroni_sig_data['size_mlog10_pvalue'], c=bonferroni_sig_data['tstat'], 
                   cmap=cmap, norm=norm, alpha=0.8, marker='s', edgecolors='black',
                   label=f'Bonferroni significant (p<{bonferroni_threshold:.3f})')

    ax.set_ylim(-0.5, 1.5)

    # Add colorbar for tstat
    scatter = ax.scatter(sig_data['celltype'], sig_data['sex'], c=sig_data['tstat'], cmap=cmap, norm=norm, alpha=0.6, s=0)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([])  # Remove legend tick labels

    # Add "downregulated" and "upregulated" labels to colorbar
    cbar_min, cbar_max = cbar.ax.get_ylim()
    mid_bottom = cbar_min + (cbar_max - cbar_min) / 2.1
    mid_top = cbar_max - (cbar_max - cbar_min) / 2.1
    cbar.ax.text(2, mid_bottom, 'down \n reg.', va='bottom', ha='left', fontsize=8, rotation=0, transform=cbar.ax.transData)
    cbar.ax.text(2, mid_top, 'up \n reg.', va='top', ha='left', fontsize=8, rotation=0, transform=cbar.ax.transData)

    # Add size legend for mlog10_pvalue
    sizes_describe = ttest_mdd_dn_df[['size_mlog10_pvalue','mlog10_pvalue']].loc[ttest_mdd_dn_df['mlog10_pvalue']>0].describe()
    sizes = [sizes_describe.loc['min','size_mlog10_pvalue'], sizes_describe.loc['50%','size_mlog10_pvalue'], sizes_describe.loc['max','size_mlog10_pvalue']]
    size_labels = np.array([sizes_describe.loc['min','mlog10_pvalue'], sizes_describe.loc['50%','mlog10_pvalue'], sizes_describe.loc['max','mlog10_pvalue']])
    size_labels = [f'{int(x)}' if x>=10 else f'{x:.2g}' for x in size_labels]

    ax2 = ax.inset_axes([1.0, 0.1, 0.4, 0.8])  # Position for size legend
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.2, 2 * len(sizes))  # More vertical space

    for i, (size, label) in enumerate(zip(sizes, size_labels)):
        y = i * 2  # Spread out vertically
        ax2.scatter(0.5, y, s=size, marker='s', color='grey', alpha=0.7, edgecolors='black')
        ax2.text(0.625, y, f'{label}', va='center', fontsize=8)

    ax2.set_title('$-log_{10}(p)$', fontsize=9, pad=0, loc='left', x=0.4, y=0.96)
    ax2.axis('off')

    # Ensure x-axis is in alphabetical order
    celltype_order = sorted(ttest_mdd_dn_df['celltype'].unique())
    ax.set_xticks(celltype_order)
    ax.set_xticklabels(celltype_order, rotation=45, ha='right')

    return ax

ttest_mdd_dn_df = get_ttest_df(ttest_comp_df_dict, pathway_name='ASTON_MAJOR_DEPRESSIVE_DISORDER_DN')
ttest_mdd_dn_ax = plot_ttest(ttest_mdd_dn_df)



#%% Module scores for pathways of interest

pathways_of_interest = [
    "ASTON_MAJOR_DEPRESSIVE_DISORDER_DN",
    #"FAN_EMBRYONIC_CTX_OLIG",
    "Oligodendrocyte_Mature_Darmanis_PNAS_2015",
    #"LEIN_OLIGODENDROCYTE_MARKERS",
    #"ZHONG_PFC_C4_PTGDS_POS_OPC",
    #"DESCARTES_FETAL_CEREBRUM_OLIGODENDROCYTES",
    #"COLIN_PILOCYTIC_ASTROCYTOMA_VS_GLIOBLASTOMA_UP",
    #"DESCARTES_MAIN_FETAL_OLIGODENDROCYTES",
    #"DESCARTES_FETAL_CEREBELLUM_OLIGODENDROCYTES",
    #"LU_AGING_BRAIN_UP",
    #"DURANTE_ADULT_OLFACTORY_NEUROEPITHELIUM_OLFACTORY_ENSHEATHING_GLIA",
    #"DESCARTES_MAIN_FETAL_SCHWANN_CELLS",
    #"Oligodendrocyte_All_Zeisel_Science_2015",
    #"GOBERT_OLIGODENDROCYTE_DIFFERENTIATION_DN",
    #"BLALOCK_ALZHEIMERS_DISEASE_UP"
]

# Call the function
n = len(pathways_of_interest)
#fig, axes = plt.subplots(n, 1, figsize=(8, 2.25*n), sharex=True)
fig, axes = plt.subplots(n, 1, figsize=(6.5, 1.75*n), sharex=True)
if n == 1:
    axes = [axes]

for i, pathway_name in enumerate(pathways_of_interest):
    ttest_df = get_ttest_df(ttest_comp_df_dict, pathway_name=pathway_name)
    plot_ttest(ttest_df, ax=axes[i])
    axes[i].set_title(pathway_name)

#fig.suptitle('Pathway-level differential expression of genes identified from MDD pathway enrichment', fontsize=10, y=0.96)
fig.tight_layout()
plt.show()

#fig.savefig(os.path.join(output_dir, 'ttest_mdd_dn_fig.png'),  dpi=300, bbox_inches='tight')

#%% GRN plot of ABHD17B

TFs_of_EGR1 = mean_grn_df[mean_grn_df['TG'] == 'EGR1']['TF'].to_list()

hit1 = 'NR4A2'
assert hit1 in TFs_of_EGR1

#NR4A2_targets_ExN_male = mean_grn_df_filtered_dict['male']['ExN'].loc[mean_grn_df_filtered_dict['male']['ExN']['TF'] == 'NR4A2']
grn_female_exn = mean_grn_df_filtered_pruned_dict['female']['ExN']
NR4A2_targets_ExN_female = grn_female_exn[grn_female_exn['TF']==hit1]
EGR1_targets_ExN_female = grn_female_exn[grn_female_exn['TF']=='EGR1']
SOX2_targets_ExN_female = grn_female_exn[grn_female_exn['TF']=='SOX2']

hit2 = 'ABHD17B'
assert hit2 in NR4A2_targets_ExN_female['TG'].to_list()

NR4A2_hit2 = NR4A2_targets_ExN_female[NR4A2_targets_ExN_female['TG'] == 'ABHD17B']['enhancer'].item()
EGR1_hit2 = EGR1_targets_ExN_female[EGR1_targets_ExN_female['TG'] == 'ABHD17B']['enhancer'].item()
assert NR4A2_hit2==EGR1_hit2

SOX2_hit2 = SOX2_targets_ExN_female[SOX2_targets_ExN_female['TG'] == 'ABHD17B']['enhancer']
assert not SOX2_hit2.isin([NR4A2_hit2, EGR1_hit2]).any()

enriched_TFs = np.array(list(enriched_TF_TG_pairs_dict.keys()))
female_exn_TFs_of_ABHD17B = grn_female_exn[grn_female_exn['TG']==hit2]['TF'].values
female_exn_enriched_TFs_of_ABHD17B = enriched_TFs[np.isin(enriched_TFs, female_exn_TFs_of_ABHD17B)]


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import networkx as nx

G = nx.DiGraph()

edge_color_map = {
    'a priori (brainSCOPE)': 'gray',
    'female ExN': 'blue',
    'all': 'green'
}

G.add_edge('NR4A2', 'EGR1', interaction='a priori (brainSCOPE)', color=edge_color_map['a priori (brainSCOPE)'])
G.add_edge('NR4A2', NR4A2_hit2, interaction='female ExN', color=edge_color_map['female ExN'])
G.add_edge('EGR1', EGR1_hit2, interaction='female ExN', color=edge_color_map['female ExN'])
G.add_edge(NR4A2_hit2, 'ABHD17B', interaction='female ExN', color=edge_color_map['female ExN'])
G.add_edge(EGR1_hit2, 'ABHD17B', interaction='female ExN', color=edge_color_map['female ExN'])
G.add_edge('SOX2', 'ABHD17B', interaction='female ExN', color=edge_color_map['female ExN'])
#G.add_edge('NR4A2', 'ABHD17B', interaction='female ExN', color=edge_color_map['female ExN'])
#G.add_edge('EGR1', 'ABHD17B', interaction='female ExN', color=edge_color_map['female ExN'])

all_targets = []
for tf in female_exn_enriched_TFs_of_ABHD17B:
    tf_targets = enriched_TF_TG_pairs_dict[tf]

    tf_targets_df = mean_grn_df[(mean_grn_df['TF']==tf) & mean_grn_df['TG'].isin(tf_targets)]
    if_enhancer_more_than_one_tg = tf_targets_df.groupby('enhancer')['TG'].nunique() > 1
    if if_enhancer_more_than_one_tg.any():
        None # currently, no "all TF"-enhancer has more than one TG target

    for tf_target in tf_targets:
        G.add_edge(tf, tf_target, interaction='all', color=edge_color_map['all'])
        all_targets.append(tf_target)

# Assign layer information to each node
for node in G.nodes():
    if node == 'NR4A2':
        G.nodes[node]['layer'] = 0
    elif node in ['EGR1', 'SOX2']:
        G.nodes[node]['layer'] = 1
    elif (node==EGR1_hit2) and (node==NR4A2_hit2):
        G.nodes[node]['layer'] = 2
    elif node == 'ABHD17B':
        G.nodes[node]['layer'] = 3
    elif node in all_targets:
        G.nodes[node]['layer'] = 4

pos = nx.multipartite_layout(G, subset_key='layer', scale=2)

for egr1_target in enriched_TF_TG_pairs_dict['EGR1']:
    pos[egr1_target] += np.array([0, 0.4])

colors = nx.get_edge_attributes(G, 'color').values()

fig, ax = plt.subplots(figsize=(4, 4))

# Draw nodes in groups with different styles
# 1. Draw regular nodes (excluding the special ones)
regular_nodes = [n for n in G.nodes() if n not in [NR4A2_hit2, EGR1_hit2]]
nx.draw_networkx_nodes(G, pos,
    nodelist=regular_nodes,
    node_size=1200,
    node_color='lightgrey',
    edgecolors='k',
    ax=ax
)

# 2. Draw the special nodes (NR4A2_hit2 and EGR1_hit2) with different style
#special_nodes = [n for n in G.nodes() if n in [NR4A2_hit2, EGR1_hit2]]
nx.draw_networkx_nodes(G, pos,
    nodelist=[NR4A2_hit2],
    node_size=100,  # smaller size
    node_color='lightgrey',
    edgecolors='k',
    node_shape='s',  # square shape
    ax=ax
)

# Draw edges
nx.draw_networkx_edges(G, pos,
    arrowstyle='-|>',
    arrowsize=15,
    width=2,
    edge_color=list(nx.get_edge_attributes(G, 'color').values()),
    min_source_margin=0.5,
    min_target_margin=0.5,
    ax=ax
)

# Draw labels for regular nodes only (excluding the special ones and ABHD17B)
labels = {n: n for n in G.nodes() if n not in ['ABHD17B', NR4A2_hit2, EGR1_hit2]}
nx.draw_networkx_labels(G, pos, labels,
    font_size=8,
    font_color='black',
    ax=ax
)

# Draw ABHD17B label separately (as in original code)
x, y = pos['ABHD17B']
plt.text(
    x, y,
    'ABHD17B',
    fontsize=6,
    fontweight='bold',
    ha='center',
    va='center'
)

axins = inset_axes(ax,
                   width="40%",    # width = 20% of parent_bbox width
                   height="20%",   # height= 20%
                   loc='lower left',
                   borderpad=1)

H = nx.DiGraph()
H.add_edge('TF','TG')

pos2 = {'TF': (0.4, 0.5),
        'TG': (0.6, 0.5)}

nx.draw_networkx_nodes(H, pos2, node_size=800, node_color='lightgrey', edgecolors='k', ax=axins)
nx.draw_networkx_edges(H, pos2, arrowstyle='-|>', arrowsize=20, edge_color='k',min_source_margin=0.2, min_target_margin=0.2, ax=axins)
nx.draw_networkx_labels(H, pos2, font_size=10, ax=axins)

# Add legend
legend_handles = [
    mpatches.Patch(color=edge_color_map['a priori (brainSCOPE)'], label='a priori (brainSCOPE)'),
    mpatches.Patch(color=edge_color_map['female ExN'], label='female ExN'),
    mpatches.Patch(color=edge_color_map['all'], label='all'),
]
legend_labels = edge_color_map.keys()
ax.legend(handles=legend_handles, title='relevant group', loc='upper left')

axins.set_xlim(0.3, 0.7)
axins.set_ylim(0.3, 0.7)

axins.axis('off')
ax.axis('off')

plt.tight_layout()
plt.show()


## Create a new, empty figure just for the legend
fig_legend = plt.figure(figsize=(3, 1))  # adjust size as needed
fig_legend.legend(legend_handles, legend_labels,
                  loc='center',        # put legend in the center
                  title='edge type',
                  frameon=False)

## Remove axes, and save tight around the legend
fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig_legend.savefig(os.path.join(output_dir, 'grn_legend_only.png'), 
                   bbox_inches='tight', 
                   pad_inches=0.1,
                   dpi=300)

#nx.write_graphml(G, os.path.join(output_dir, "ABHD17B_GRN.graphml"))
nx.write_gexf(G, os.path.join(output_dir, "ABHD17B_GRN.gexf"))



#%% MAGMA results

def plot_magma(magma_results_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2))

    # Ensure x-axis is in alphabetical order
    magma_results_df = magma_results_df.sort_values('celltype')

    # Empty scatter plot for colorbar
    ax.scatter(magma_results_df['celltype'], magma_results_df['sex'], alpha=0, s=0)

    # Plot non-significant as grey crosses
    non_sig_data = magma_results_df[magma_results_df['mlog10_pvalue'] == 0]
    if len(non_sig_data) > 0:
        ax.scatter(non_sig_data['celltype'], non_sig_data['sex'], 
                   color='grey', alpha=0.7, marker='+', linewidths=1)

    # Main scatter plot
    scatter = ax.scatter(magma_results_df['celltype'], magma_results_df['sex'], 
               c=magma_results_df['BETA'], s=magma_results_df['size_mlog10_pvalue'], alpha=0.6, edgecolors='black')
    
    ax.set_ylim(-0.5, 1.5)

    # Add colorbar for tstat
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('BETA', pad=10, fontsize=9)

    # Add size legend for mlog10_pvalue
    sizes_describe = magma_results_df[['size_mlog10_pvalue','mlog10_pvalue']].loc[magma_results_df['mlog10_pvalue']>0].describe()
    sizes = [sizes_describe.loc['min','size_mlog10_pvalue'], sizes_describe.loc['50%','size_mlog10_pvalue'], sizes_describe.loc['max','size_mlog10_pvalue']]
    size_labels = np.array([sizes_describe.loc['min','mlog10_pvalue'], sizes_describe.loc['50%','mlog10_pvalue'], sizes_describe.loc['max','mlog10_pvalue']]).round(2)

    # Create a separate axis for the size legend
    ax2 = ax.inset_axes([1.1, 0.1, 0.4, 0.8])  # Position for size legend
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.2, 2 * len(sizes))  # More vertical space

    for i, (size, label) in enumerate(zip(sizes, size_labels)):
        y = i * 2  # Spread out vertically
        ax2.scatter(0.5, y, s=size, color='grey', alpha=0.7, edgecolors='black')
        ax2.text(0.7, y, f'{label}', va='center', fontsize=8)

    ax2.set_title('$-log_{10}(p)$', fontsize=9, pad=0, loc='left', x=0.4, y=0.96)
    ax2.axis('off')

    # Ensure x-axis is in alphabetical order
    celltype_order = sorted(magma_results_df['celltype'].unique())
    ax.set_xticks(celltype_order)
    ax.set_xticklabels(celltype_order, rotation=45, ha='right')

    fig.tight_layout()
    fig.suptitle('MAGMA results for MDD GWAS')

    return ax

#magma_results_df = pd.read_csv('data/magma_results/magma_results_mdd_dn_pathways.txt', sep='\t')
magma_results_df = magma_results_df.reset_index()
magma_results_df['mlog10_pvalue'] = -np.log10(magma_results_df['P'])
magma_results_df['size_mlog10_pvalue'] = np.log1p(magma_results_df['mlog10_pvalue']) * 1000

plot_magma(magma_results_df)

#%% GREAT results

great_results_mdd_dn_df = pd.DataFrame(columns=['ngenes', 'padj', 'mlog10_padj'], 
                                index=pd.MultiIndex.from_product([unique_sexes, unique_celltypes], names=['sex', 'celltype']))

for sex in unique_sexes:
    for celltype in unique_celltypes:

        great_results_df = great_results_df_dict[sex][celltype]
        great_mdd_dn = great_results_df[great_results_df.index == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']

        if len(great_mdd_dn) == 0:
            continue

        ngenes = great_mdd_dn['observed_gene_hits'].item()
        padj = great_mdd_dn['p_adjust_hyper'].item()
        mlog10_padj = -np.log10(padj)

        great_results_mdd_dn_df.loc[(sex, celltype), 'ngenes'] = ngenes
        great_results_mdd_dn_df.loc[(sex, celltype), 'padj'] = padj
        great_results_mdd_dn_df.loc[(sex, celltype), 'mlog10_padj'] = mlog10_padj

great_results_mdd_dn_df.reset_index(inplace=True)
great_results_mdd_dn_df['mlog10_padj'] = great_results_mdd_dn_df['mlog10_padj'].fillna(0)
great_results_mdd_dn_df['ngenes'] = great_results_mdd_dn_df['ngenes'].fillna(0)
great_results_mdd_dn_df['size_ngenes'] = (great_results_mdd_dn_df['ngenes']/great_results_mdd_dn_df['ngenes'].max())**2 * 500


plot_enrichment_significance(great_results_mdd_dn_df, title='GREAT results for MDD-DN pathway')

#%% Compare number of enriched pathways per type of gene set

## load filtered results
with open(os.path.join(output_dir, 'broad_gene_series_dict.pkl'), 'rb') as f:
    
    gene_series_dict = pickle.load(f)

    enrs_mdd_dn_genes_series = gene_series_dict['enrs_mdd_dn_genes_series']
    all_sccompreg_genes_series = gene_series_dict['all_sccompreg_genes_series']
    pydeseq2_match_length_genes_series = gene_series_dict['pydeseq2_match_length_genes_series']

## enrichment of pooled gene sets, without filtering for p-value
enrs_mdd_dn_genes_enrichr_all = do_enrichr(enrs_mdd_dn_genes_series, brain_gmt_cortical, filter_var=None, outdir=None)
all_sccompreg_genes_enrichr_all = do_enrichr(all_sccompreg_genes_series, brain_gmt_cortical, filter_var=None, outdir=None)
pydeseq2_match_length_genes_enrichr_all = do_enrichr(pydeseq2_match_length_genes_series, brain_gmt_cortical, filter_var=None, outdir=None)

results_to_index_mapper = {
    #'mdd_dn_genes': enrs_mdd_dn_genes_enrichr_all,
    'sc_compreg_all': all_sccompreg_genes_enrichr_all,
    'pydeseq2_all': pydeseq2_match_length_genes_enrichr_all
}

## extract p-values to place into dataframe
enrs_compare_mlog10_pval_df = pd.DataFrame(0, columns=brain_gmt_cortical.keys(), index=results_to_index_mapper.keys())

for result_name, result_df in results_to_index_mapper.items():

    for pathway in brain_gmt_cortical.keys():
        if pathway in result_df['Term'].to_list():

            result_pathway = result_df.set_index('Term').loc[pathway]
            pval = result_pathway['Adjusted P-value']
            mlog10_pval = -np.log10(pval)
            enrs_compare_mlog10_pval_df.loc[result_name, pathway] = mlog10_pval

enrs_compare_mlog10_pval_df.dropna(inplace=True)

## extract topk pathways for each experiment type
topk = 15
topk_pathways_dict = {}
for result_name, result_df in enrs_compare_mlog10_pval_df.iterrows():

    topk_pathways = result_df.sort_values(ascending=False).head(topk).index.to_list()
    topk_pathways_dict[result_name] = topk_pathways

keep_pathways = np.hstack(list(topk_pathways_dict.values()))

enrs_compare_mlog10_pval_ladder = enrs_compare_mlog10_pval_df[keep_pathways].T
enrs_compare_mlog10_pval_ladder.drop_duplicates(keep='last', inplace=True)

## plot heatmap of topk pathways
fig, ax = plt.subplots(figsize=(3, 14))
heatmap = sns.heatmap(enrs_compare_mlog10_pval_ladder, cmap='viridis', vmin=0, vmax=5, ax=ax, cbar_kws={'label': '$-log_{10}$(p-adj.)'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

# Add red stars where value > -log10(0.05)
threshold = -np.log10(0.05)
data = enrs_compare_mlog10_pval_ladder.values
for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        if data[y, x] > threshold:
            ax.plot(x + 0.5, y + 0.5, marker='*', color='red', markersize=10, markeredgecolor='black')

#plt.savefig(os.path.join(output_dir, 'enrs_compare_mlog10_pval_ladder.png'), dpi=300, bbox_inches='tight')