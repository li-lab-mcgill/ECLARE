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

#%% import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math
import torch
from collections import defaultdict
from string import ascii_uppercase
import pybedtools
from types import SimpleNamespace
from tqdm import tqdm
from glob import glob
from datetime import datetime
import pickle

from mlflow.tracking import MlflowClient
import mlflow.pytorch
from mlflow.models import Model

from eclare.models import load_CLIP_model, CLIP
from eclare.setup_utils import mdd_setup
from eclare.post_hoc_utils import get_latents, sample_proportional_celltypes_and_condition, plot_umap_embeddings, create_celltype_palette
from eclare.data_utils import get_unified_grns, filter_mean_grn, get_scompreg_loglikelihood

from ot import solve as ot_solve
import SEACells
import scanpy as sc
import scglue
import networkx as nx
from pyjaspar import jaspardb
import pyranges as pr
import gseapy as gp
from Bio.motifs.jaspar import calculate_pseudocounts
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP
from torchmetrics.functional import kendall_rank_corrcoef
from scipy.stats import norm, linregress

def cell_gap_ot(student_logits, atac_latents, rna_latents, mdd_atac_sampled_group, mdd_rna_sampled_group, cells_gap, type='emd'):

    if type == 'emd':
        res = ot_solve(1 - student_logits)
        plan = res.plan
        value = res.value_linear

    elif type == 'partial':
        a, b = torch.ones((len(student_logits.shape[0]),)) / len(student_logits.shape[0]), torch.ones((len(student_logits.shape[1]),)) / len(student_logits.shape[1])
        mass = 1 - (a[0] * cells_gap).item()
        plan = ot.partial.partial_wasserstein(a, b, 1-student_logits, m=mass)

    if student_logits.shape[0] > student_logits.shape[1]:
        keep_atac_cells = plan.max(1).values.argsort()[cells_gap:].sort().values.detach().cpu().numpy()
        atac_latents = atac_latents[keep_atac_cells]
        mdd_atac_sampled_group = mdd_atac_sampled_group[keep_atac_cells]
    else:
        keep_rna_cells = plan.max(0).values.argsort()[cells_gap:].sort().values.detach().cpu().numpy()
        rna_latents = rna_latents[keep_rna_cells]
        mdd_rna_sampled_group = mdd_rna_sampled_group[keep_rna_cells]

    student_logits = torch.matmul(atac_latents, rna_latents.T)

    return mdd_atac_sampled_group, mdd_rna_sampled_group, student_logits

def run_SEACells(adata_train, adata_apply, build_kernel_on, redo_umap=False, key='X_umap', n_SEACells=None):

    # Copy the counts to ".raw" attribute of the anndata since it is necessary for downstream analysis
    # This step should be performed after filtering 
    raw_ad = sc.AnnData(adata_train.X)
    raw_ad.obs_names, raw_ad.var_names = adata_train.obs_names, adata_train.var_names
    adata_train.raw = raw_ad

    raw_ad = sc.AnnData(adata_apply.X)
    raw_ad.obs_names, raw_ad.var_names = adata_apply.obs_names, adata_apply.var_names
    adata_apply.raw = raw_ad

    # Normalize cells, log transform and compute highly variable genes
    #sc.pp.normalize_per_cell(adata_train)
    #sc.pp.log1p(adata_train)
    #sc.pp.highly_variable_genes(adata_train, n_top_genes=1500)

    ## User defined parameters

    ## Core parameters 
    n_SEACells = n_SEACells if n_SEACells is not None else max(adata_train.n_obs // 100, 15) # 75 cells per SEACell, for a minimum of 15 SEACells
    print(f'Number of SEACells: {n_SEACells}')
    n_waypoint_eigs = 15 # Number of eigenvalues to consider when initializing metacells

    ## Initialize SEACells model
    model = SEACells.core.SEACells(adata_train, 
                    build_kernel_on=build_kernel_on, 
                    n_SEACells=n_SEACells, 
                    n_waypoint_eigs=n_waypoint_eigs,
                    convergence_epsilon = 1e-5,
                    max_franke_wolfe_iters=100,
                    use_gpu=True if torch.cuda.is_available() else False)

    model.construct_kernel_matrix()
    M = model.kernel_matrix

    # Initialize archetypes
    model.initialize_archetypes()

    # Fit SEACells model
    model.fit(min_iter=10, max_iter=100)

    ## Get number of cells in each SEACell
    n_cells_per_SEACell = adata_train.obs['SEACell'].value_counts(sort=False)
    assert np.all(n_cells_per_SEACell.index == adata_train.obs['SEACell'].unique()) # unique() is how SEACells gather cells into SEACells

    ## add SEACell labels to apply data
    adata_apply.obs['SEACell'] = adata_train.obs['SEACell'].values

    ## summarize by SEACell
    SEACell_ad_train = SEACells.core.summarize_by_SEACell(adata_train, SEACells_label='SEACell', summarize_layer='raw')
    SEACell_ad_apply = SEACells.core.summarize_by_SEACell(adata_apply, SEACells_label='SEACell', summarize_layer='raw')

    ## divide aggregated SEACell counts by number of cells in each SEACell
    SEACell_ad_train.X = SEACell_ad_train.X.multiply(1 / n_cells_per_SEACell.values[:,None]).tocsr()
    SEACell_ad_apply.X = SEACell_ad_apply.X.multiply(1 / n_cells_per_SEACell.values[:,None]).tocsr()

    if redo_umap:
        #sc.pp.pca(adata, n_comps=15)
        sc.pp.neighbors(adata_train)
        sc.tl.umap(adata_train)

        SEACells.plot.plot_2D(adata_train, key='X_umap', colour_metacells=False)
        SEACells.plot.plot_2D(adata_train, key='X_umap', colour_metacells=True)

    else:
        SEACells.plot.plot_2D(adata_train, key=key, colour_metacells=False)
        SEACells.plot.plot_2D(adata_train, key=key, colour_metacells=True)

    return SEACell_ad_train, SEACell_ad_apply


def get_metrics(method, job_id, target_only=False):

    method_job_id = f'{method}_{job_id}'
    paths_root = os.path.join(os.environ['OUTPATH'], method_job_id)

    ## retain leaf directories only
    paths, all_data_source, all_data_target = [], [], []
    paths = glob(os.path.join(paths_root, '**', '**', '**', f'*_metrics_target_valid.csv'))

    ## For scMulticlip, will not find metrics with previous command
    if paths == []:
        paths = glob(os.path.join(paths_root, '**', '**', f'*_metrics_target_valid.csv'))
    #for dirpath, dirnames, filenames in os.walk(paths_root): paths.append(dirpath) if not dirnames else None

    paths = sorted(paths)

    for path in paths:

        path = os.path.dirname(path)

        ## Get source and target dataset names
        path_split = path.split('/')
        method_job_id_idx = np.where([split==method_job_id for split in path_split])[0][0]

        target = path_split[method_job_id_idx + 1]
        source = path_split[method_job_id_idx + 2] if len(path_split) > method_job_id_idx + 2 else None
        if source in ['0', '1', '2']: source = None

        ## Read metrics
        try:
            metrics_target_valid = pd.read_csv(glob(os.path.join(path, f'*_metrics_target_valid.csv'))[0], index_col=0)
            metrics_source_valid = None if target_only else pd.read_csv(glob(os.path.join(path, f'*_metrics_source_valid.csv'))[0], index_col=0)
        except:
            print(f'Error reading {path}')
            continue

        ## Drop foscttm_score_ct & rank_score
        if 'foscttm_score_ct' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['foscttm_score_ct'])
            if ('foscttm_score_ct' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['foscttm_score_ct'])

        if 'rank_score' in metrics_target_valid.index:
            metrics_target_valid = metrics_target_valid.drop(index=['rank_score'])
            if ('rank_score' in metrics_source_valid.index) and (not target_only):
                metrics_source_valid = metrics_source_valid.drop(index=['rank_score'])

        ## Transpose, such that metrics as columns rather than indices
        metrics_target_valid = metrics_target_valid.T
        if not target_only:
            metrics_source_valid = metrics_source_valid.T

        ## Add target dataset names, then append to all_data
        metrics_target_valid['target'] = target
        metrics_target_valid['source'] = source
        all_data_target.append(metrics_target_valid)

        if not target_only:
            metrics_source_valid['target'] = target
            metrics_source_valid['source'] = source
            all_data_source.append(metrics_source_valid)

    ## Concatenate all_data and set multi-index
    target_metrics_valid_df = pd.concat(all_data_target).set_index(['source', 'target'])
    target_metrics_valid_df = target_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in target_metrics_valid_df.columns:
        target_metrics_valid_df = target_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in target_metrics_valid_df.columns:
        target_metrics_valid_df['1 - foscttm_score'] = 1 - target_metrics_valid_df['foscttm_score']
        target_metrics_valid_df = target_metrics_valid_df.drop(columns=['foscttm_score'])

    if target_only:
        return target_metrics_valid_df

    ## --- SOURCE --- ##
    source_metrics_valid_df = pd.concat(all_data_source).set_index(['source', 'target'])
    source_metrics_valid_df = source_metrics_valid_df.astype(float)

    ## TEMPORARY
    if 'foscttm' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'foscttm': 'foscttm_score'})

    if 'ilisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'ilisi': 'ilisis'})

    if 'clisi' in source_metrics_valid_df.columns:
        source_metrics_valid_df = source_metrics_valid_df.rename(columns={'clisi': 'clisis'})

    ## Replace foscttm_score by 1 - foscttm_score
    if 'foscttm_score' in source_metrics_valid_df.columns:
        source_metrics_valid_df['1 - foscttm_score'] = 1 - source_metrics_valid_df['foscttm_score']
        source_metrics_valid_df = source_metrics_valid_df.drop(columns=['foscttm_score'])

    ## Create source-only metrics dataframe, and sort according to existing order of target in original df
    source_only_metrics_valid_df = source_metrics_valid_df.reset_index(level='target').drop(columns='target')
    source_only_metrics_valid_df = source_only_metrics_valid_df.loc[source_metrics_valid_df.index.get_level_values(0).unique().values]

    return source_metrics_valid_df, target_metrics_valid_df, source_only_metrics_valid_df

def compare_dataframes_target(dataframes, value_column, dataset_labels, target_source_combinations, ax=None):

    dataset_label_mapper = {
        'AD_Anderson_et_al': 'DLPFC_Anderson',
        'PD_Adams_et_al': 'Midbrain_Adams',
        'human_dlpfc': 'DLPFC_Ma',
        'roussos': 'PFC_Zhu',
        'mdd': 'MDD'
    }

    hue_order = list(dataset_label_mapper.values())

    combined_df = pd.concat(
        [df.assign(dataset=label).rename(index=dataset_label_mapper)
         for df, label in zip(dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(combined_df["target"].unique())

    colors = sns.color_palette("Dark2", 7)
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    #target_to_color = {target: colors[i] if len(unique_targets)>1 else 'black' for i, target in enumerate(unique_targets)}
    if len(unique_targets) > 1:
        target_to_color = {hue_order[i]: colors[i] if len(unique_targets)>1 else 'black' for i in range(len(unique_targets))}
    elif unique_targets[0] == 'MDD':
        target_to_color = {unique_targets[0]: colors[6]}

    if target_source_combinations:
        source_to_marker = {source: markers[i % len(markers)] for i, source in enumerate(unique_sources)}
    else:
        source_to_marker = {source: 'o' for i, source in enumerate(unique_sources)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    ).tick_params(axis='x', rotation=30)
    ax.set_xlabel("method")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for target in unique_targets:
        for dataset in dataset_positions:
            for source in unique_sources:

                if pd.isna(source):
                    subset = combined_df[(combined_df["target"] == target) &
                                        (combined_df["source"].isna()) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker='o',
                        edgecolor=None,
                        s=50,
                        zorder=10,
                        alpha=0.4
                    )

                else:
                    subset = combined_df[(combined_df["source"] == source) &
                                        (combined_df["target"] == target) &
                                        (combined_df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[value_column],
                        color=target_to_color[target],
                        marker=source_to_marker[source],
                        edgecolor=None,
                        s=50 if source_to_marker[source] == '*' else 50,
                        zorder=10,
                        alpha=0.4
                    )

    return source_to_marker, target_to_color
        

def compare_dataframes_target_only(dataframes, value_column, dataset_labels, ax=None):
    """
    Compare multiple dataframes with boxplots and scatter points with centered jitter,
    distinguishing only between targets (shapes).

    Parameters:
    - dataframes: list of pandas DataFrames (must have same structure and multi-index)
    - value_column: str, the column name to be plotted
    - dataset_labels: list of str, labels for each dataset
    - ax: Matplotlib Axes object (optional)
    """
    combined_df = pd.concat(
        [
            (df.droplevel('source').assign(dataset=label) if 'source' in df.index.names else df.assign(dataset=label))
            for df, label in zip(dataframes, dataset_labels)
        ]
    ).reset_index()

    unique_targets = sorted(combined_df["target"].unique())
    colors = sns.color_palette("tab10", len(unique_targets))

    target_to_color = {target: colors[i % len(colors)] for i, target in enumerate(unique_targets)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    )

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for target in unique_targets:
        for dataset in dataset_positions:
            subset = combined_df[(combined_df["target"] == target) &
                                 (combined_df["dataset"] == dataset)]
            x_position = position_mapping[dataset]
            jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
            jittered_positions = x_position + jitter
            ax.scatter(
                x=jittered_positions,
                y=subset[value_column],
                color=target_to_color[target],
                edgecolor='black',
                s=150,
                zorder=10,
                alpha=0.9
            )

def compare_dataframes_source_only(dataframes, value_column, dataset_labels, ax=None):

    combined_df = pd.concat(
        [df.assign(dataset=label) for df, label in zip(dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique())
    colors = sns.color_palette("Dark2", len(unique_sources))
    source_to_color = {source: colors[i] for i, source in enumerate(unique_sources)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        x="dataset",
        y=value_column,
        data=combined_df,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7)
    ).tick_params(axis='x', rotation=30)
    ax.set_xlabel("method")

    # Add scatter points with jitter
    dataset_positions = combined_df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}
    for source in unique_sources:
        for dataset in dataset_positions:
            subset = combined_df[(combined_df["source"] == source) &
                                 (combined_df["dataset"] == dataset)]
            x_position = position_mapping[dataset]
            jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
            jittered_positions = x_position + jitter
            ax.scatter(
                x=jittered_positions,
                y=subset[value_column],
                color=source_to_color[source],
                edgecolor='black',
                s=50,
                zorder=10,
                alpha=0.6
            )

    return source_to_color


def combined_plot(target_dataframes, value_columns, dataset_labels, target_source_combinations=False, source_only_dataframes=None, figsize=(6,6)):

    # Create a figure with two subplots
    if source_only_dataframes is None:
        fig, axs = plt.subplots(1, len(value_columns), figsize=figsize, sharex=True, squeeze=False)
    else:
        fig, axs = plt.subplots(2, len(value_columns), figsize=figsize, sharey='col')

    for value_column, ax in zip(value_columns, axs[-1]):
        # Pass axes to the individual plotting functions
        source_to_marker, target_to_color = compare_dataframes_target(target_dataframes, value_column, dataset_labels, target_source_combinations, ax=ax)

    # Create handles for color-coding by target
    color_handles = [
        plt.Line2D([0], [0], marker='D' if target_source_combinations else 'o', color=color, label=f"{target}", linestyle='None', markersize=10)
        for target, color in target_to_color.items()
    ]

    # Create handles for marker-coding by source
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', label=f"{source}", linestyle='None', markersize=10)
        if not pd.isna(source) else
        plt.Line2D([0], [0], marker='o', color='black', label=f"all sources (ensemble)", linestyle='None', markersize=10)
        for source, marker in source_to_marker.items()
    ]


    # Add labels for color and marker sections
    if ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) > 1) ):

        # Add a blank entry and title for marker section
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles +
            [plt.Line2D([0], [0], color='none', label="")] +  # Blank separator
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +  # Title for the second block
            marker_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()] +
            [''] +  # Blank separator
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )

    elif ( (len(set(source_to_marker.values())) == 1) and ( len(set(target_to_color.keys())) > 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()]
        )
    elif ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) == 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +
            marker_handles
        )
        combined_labels = (
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )
    else:
        combined_handles = None
        combined_labels = None

    axs[-1,-1].legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    # Set a title for the entire figure
    for value_column, ax, letter in zip(value_columns, axs[0], ascii_uppercase):

        ax.set_title(f"{letter}) {value_column}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        if source_only_dataframes is not None:
            source_to_target = compare_dataframes_source_only(source_only_dataframes, value_column, dataset_labels, ax=ax)

            axs[0,0].set_ylabel(f"source to source")
            axs[-1,0].set_ylabel(f"source to target")

            combined_handles = (
                [plt.Line2D([0], [0], color='none', label="Source, by color:")] +
                color_handles
            )
            combined_labels = (
                ['Source, by color:'] +
                [f"{target}" for target in target_to_color.keys()]
            )

            axs[0,-1].legend(
                handles=combined_handles,
                labels=combined_labels,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=False
            )

    plt.tight_layout()
    plt.show()

    return fig

def combined_plot_target_only(target_dataframes, value_column, dataset_labels):

    '''
    combined_df = pd.concat(
        [
            (df.droplevel('source').assign(dataset=label) if 'source' in df.index.names else df.assign(dataset=label))
            for df, label in zip(target_dataframes, dataset_labels)
        ]
    ).reset_index()
    '''

    combined_df = pd.concat(
        [df.assign(dataset=label) for df, label in zip(target_dataframes, dataset_labels)]
    ).reset_index()

    unique_sources = sorted(combined_df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(combined_df["target"].unique())

    colors = sns.color_palette("tab10", len(unique_sources))
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    source_to_color = {source: colors[i % len(colors)] for i, source in enumerate(unique_sources)}
    target_to_marker = {target: markers[i % len(markers)] for i, target in enumerate(unique_targets)}

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    # Pass axes to the individual plotting functions
    compare_dataframes_target(target_dataframes, value_column, dataset_labels, ax=axs)

    # Add the legend from compare_dataframes_target
    shape_legend = [
        plt.Line2D([0], [0], color=color, label=target, markerfacecolor='gray', markersize=12)
        for target, color in target_to_color.items()
    ]
    full_legend = shape_legend

    axs.legend(
        handles=full_legend,
        title="Targets",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()

## updated functions
# Create multi-index for CLIP metrics DataFrame
def extract_target_source_replicate(df, has_source=True):

    if df.empty or 'run_name' not in df.columns:
        return df
    
    # Extract source and target from Name column
    sources = []
    targets = []
    replicates = []

    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    if has_source:

        for name in df['run_name']:
            try:
                source = name.split('_to_')[0]
                target = name.split('_to_')[1].split('-')[0]
                replicate = name.split('_to_')[1].split('-')[1]
            except:
                source = name.split('-to-')[0]
                target = name.split('-to-')[1].split('-')[0]
                replicate = name.split('-to-')[1].split('-')[1]
            finally:
                sources.append(source)
                targets.append(target)
                replicates.append(replicate)

        result_df['source'] = sources

    elif not has_source:

        for name in df['run_name']:

            target      = name.split('-')[0]
            replicate   = name.split('-')[1]

            targets.append(target)
            replicates.append(replicate)
    
    # Add source and target columns
    result_df['target'] = targets
    result_df['replicate'] = replicates
    
    # Set multi-index
    #result_df = result_df.set_index(['source', 'target', 'replicate'])
    
    return result_df

def metric_boxplot(df, metric, target_to_color, source_to_marker, unique_targets, unique_sources, ax=None):

    # Try to convert the metric column to float, replacing errors with NaN
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Check if there's only one row for any dataset-metric combination
    dataset_metric_counts = df.groupby(['dataset']).size()
    has_single_values = (dataset_metric_counts == 1).all()
    
    if has_single_values:
        # Use barplot for datasets with single values
        sns.barplot(
            x="dataset",
            y=metric,
            data=df,
            ax=ax,
            color="lightgray"
        ).tick_params(axis='x', rotation=45)

    else:
        sns.boxplot(
            x="dataset",
            y=metric,
            data=df,
            ax=ax,
            color="lightgray",
            showfliers=False,
            boxprops=dict(alpha=0.4),
            whiskerprops=dict(alpha=0.4),
            capprops=dict(alpha=0.4),
            medianprops=dict(alpha=0.7)
        ).tick_params(axis='x', rotation=45)

    ax.set_xlabel("method")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Add scatter points with jitter
    dataset_positions = df["dataset"].unique()
    position_mapping = {dataset: i for i, dataset in enumerate(dataset_positions)}

    for target in unique_targets:
        for dataset in dataset_positions:
            for source in unique_sources:

                if pd.isna(source):
                    subset = df[(df["target"] == target) &
                                (df["source"].isna()) &
                                (df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[metric],
                        color=target_to_color[target],
                        marker='o',
                        edgecolor=None,
                        s=50,
                        zorder=10,
                        alpha=0.4
                    )

                else:
                    subset = df[(df["source"] == source) &
                                (df["target"] == target) &
                                (df["dataset"] == dataset)]
                    x_position = position_mapping[dataset]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
                    jittered_positions = x_position + jitter
                    ax.scatter(
                        x=jittered_positions,
                        y=subset[metric],
                        color=target_to_color[target],
                        marker=source_to_marker[source],
                        edgecolor=None,
                        s=50 if source_to_marker[source] == '*' else 50,
                        zorder=10,
                        alpha=0.4
                    )

def metric_boxplots(df, target_source_combinations=False, include_paired=True):

    hue_order = list(["PFC_Zhu", "DLPFC_Anderson", "DLPFC_Ma", "Midbrain_Adams", "mouse_brain_10x", "pbmc_10x", "MDD"])

    unique_sources = sorted(df["source"].unique(), key=lambda item: (math.isnan(item), item) if isinstance(item, float) else (False, item))
    unique_targets = sorted(df["target"].unique())

    colors = sns.color_palette("Dark2", 8)[:4] + [sns.color_palette("Dark2", 8)[5]] + [sns.color_palette("Dark2", 8)[-1]]
    markers = ['*', 's', '^', 'P', 'D', 'v', '<', '>']

    #target_to_color = {target: colors[i] if len(unique_targets)>1 else 'black' for i, target in enumerate(unique_targets)}
    if len(unique_targets) > 1:
        target_to_color = {hue_order[i]: colors[i] if len(unique_targets)>1 else 'black' for i in range(len(unique_targets))}
    elif unique_targets[0] == 'MDD':
        target_to_color = {unique_targets[0]: sns.color_palette("Dark2", 8)[6]}

    if target_source_combinations:
        source_to_marker = {source: markers[i % len(markers)] for i, source in enumerate(unique_sources)}
    else:
        source_to_marker = {source: 'o' for i, source in enumerate(unique_sources)}

    ## loop over metrics and plot
    unpaired_metrics = ['multimodal_ilisi', 'ari', 'nmi', 'silhouette_celltype', 'batches_ilisi']
    paired_metrics = ['1-foscttm']
    all_metrics = paired_metrics + unpaired_metrics if include_paired else unpaired_metrics

    fig, axs = plt.subplots(1, len(all_metrics), figsize=(14, 4), sharex=True)

    for metric, ax in zip(all_metrics, axs):
        metric_boxplot(df, metric, target_to_color, source_to_marker, unique_targets, unique_sources, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel('')

    # Create handles for color-coding by target
    color_handles = [
        plt.Line2D([0], [0], marker='D' if target_source_combinations else 'o', color=color, label=f"{target}", linestyle='None', markersize=10)
        for target, color in target_to_color.items()
    ]

    # Create handles for marker-coding by source
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', label=f"{source}", linestyle='None', markersize=10)
        if not pd.isna(source) else
        plt.Line2D([0], [0], marker='o', color='black', label=f"all sources (ensemble)", linestyle='None', markersize=10)
        for source, marker in source_to_marker.items()
    ]


    # Add labels for color and marker sections
    if ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) > 1) ):

        # Add a blank entry and title for marker section
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles +
            [plt.Line2D([0], [0], color='none', label="")] +  # Blank separator
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +  # Title for the second block
            marker_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()] +
            [''] +  # Blank separator
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )

    elif ( (len(set(source_to_marker.values())) == 1) and ( len(set(target_to_color.keys())) > 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Target, by color:")] +
            color_handles
        )
        combined_labels = (
            ['Target, by color:'] +
            [f"{target}" for target in target_to_color.keys()]
        )
    elif ( (len(set(source_to_marker.values())) > 1) and ( len(set(target_to_color.keys())) == 1) ):
        combined_handles = (
            [plt.Line2D([0], [0], color='none', label="Source, by marker:")] +
            marker_handles
        )
        combined_labels = (
            ["Source, by marker:"] +
            [   f"{source}" if not pd.isna(source) else "all sources (ensemble)"
                for source in source_to_marker.keys()]
        )
    else:
        combined_handles = None
        combined_labels = None

    axs[-1].legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    fig.tight_layout()
    fig.show()


cuda_available = torch.cuda.is_available()

#%%
## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '21140748',
    'kd_clip': '22222341',
    'eclare': ['22222419'],
    'clip_mdd': '15093733',
    'kd_clip_mdd': '16105523',
    'eclare_mdd': ['16103846'], #16105437
    'mojitoo': '20212916',
    'multiVI': '18175921',
    'glue': '18234131_19205223',
    'scDART': '22083939', #'19150754',
    'scjoint': '19115016'
}

## define search strings
search_strings = {
    'clip': 'CLIP' + '_' + methods_id_dict['clip'],
    'kd_clip': 'KD_CLIP' + '_' + methods_id_dict['kd_clip'],
    'eclare': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare']],
    'clip_mdd': 'CLIP' + '_' + methods_id_dict['clip_mdd'],
    'kd_clip_mdd': 'KD_CLIP' + '_' + methods_id_dict['kd_clip_mdd'],
    'eclare_mdd': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare_mdd']]
}

## for ECLARE, map search_strings to 'dataset' column
dataset_column = [
    'eclare',
    ]
search_strings_to_dataset = {
    'ECLARE' + '_' + job_id: dataset_column[j] for j, job_id in enumerate(methods_id_dict['eclare'])
}


def download_mlflow_runs(experiment_name):
    
    # Initialize MLflow client with tracking URI if provided
    tracking_uri = os.path.join(os.environ['ECLARE_ROOT'], 'mlruns')
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Try to find experiment in alternative locations
        alt_tracking_uri = os.path.join(os.environ.get('OUTPATH', '.'), 'mlruns')
        if tracking_uri != alt_tracking_uri:
            client = MlflowClient(tracking_uri=alt_tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found. Set MLFLOW_TRACKING_URI environment variable to specify mlruns directory.")
    
    experiment_id = experiment.experiment_id
    
    # Get all runs for the experiment
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    # Convert runs to DataFrame
    runs_data = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'parent_run_id': run.data.tags.get('mlflow.parentRunId', None),
            'run_name': run.info.run_name,  # Include the name of the run
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time/1000.0),
            'end_time': datetime.fromtimestamp(run.info.end_time/1000.0) if run.info.end_time else None,
            'artifact_uri': run.info.artifact_uri,
            **run.data.params,
            **run.data.metrics
        }
        runs_data.append(run_data)
    
    runs_df = pd.DataFrame(runs_data)
    
    # Save to CSV
    output_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
    runs_df.to_csv(output_path, index=False)
    print(f"Saved {len(runs_df)} runs to {output_path}")
    
    return output_path

#%% paired data
experiment_name = f"clip_{methods_id_dict['clip']}"

if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
else:
    print(f"Downloading runs.csv for {experiment_name} from MLflow")
    all_metrics_csv_path = download_mlflow_runs(experiment_name)

all_metrics_df = pd.read_csv(all_metrics_csv_path)

CLIP_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['clip']))[0]
KD_CLIP_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['kd_clip']))[0]
ECLARE_header_idx = np.where(all_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['eclare'])))[0]

CLIP_run_id = all_metrics_df.iloc[CLIP_header_idx]['run_id']
KD_CLIP_run_id = all_metrics_df.iloc[KD_CLIP_header_idx]['run_id']
ECLARE_run_id = all_metrics_df.iloc[ECLARE_header_idx]['run_id']

CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(CLIP_run_id)]
KD_CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(KD_CLIP_run_id)]
ECLARE_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(ECLARE_run_id)]

CLIP_metrics_df = extract_target_source_replicate(CLIP_metrics_df)
KD_CLIP_metrics_df = extract_target_source_replicate(KD_CLIP_metrics_df)
ECLARE_metrics_df = extract_target_source_replicate(ECLARE_metrics_df, has_source=False)

## add dataset column
CLIP_metrics_df.loc[:, 'dataset']      = 'clip'
KD_CLIP_metrics_df.loc[:, 'dataset']   = 'kd_clip'
ECLARE_metrics_df.loc[:, 'dataset']    = 'eclare'

if len(methods_id_dict['eclare']) > 1:
    eclare_dfs = {}
    for search_key, dataset_name in search_strings_to_dataset.items():
        runs_from_eclare_experiment = ECLARE_metrics_df['parent_run_id'].isin(all_metrics_df.loc[all_metrics_df['run_name']==search_key]['run_id'].values)
        dataset_df = ECLARE_metrics_df.loc[runs_from_eclare_experiment]
        dataset_df.loc[:, 'dataset'] = dataset_name
        dataset_df.loc[:, 'source'] = np.nan
        eclare_dfs[dataset_name] = dataset_df

    combined_metrics_df = pd.concat([
        *eclare_dfs.values(),
        KD_CLIP_metrics_df,
        CLIP_metrics_df
        ])

else:
    combined_metrics_df = pd.concat([
        ECLARE_metrics_df,
        KD_CLIP_metrics_df,
        CLIP_metrics_df
        ]) # determines order in which metrics are plotted
    
## if source and/or target contain 'multiome', convert to '10x'
combined_metrics_df.loc[:, 'source'] = combined_metrics_df['source'].str.replace('multiome', '10x')
combined_metrics_df.loc[:, 'target'] = combined_metrics_df['target'].str.replace('multiome', '10x')

## only keep runs with 'FINISHED' status
combined_metrics_df = combined_metrics_df[combined_metrics_df['status'] == 'FINISHED']

## plot boxplots
#metric_boxplots(combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['eclare', 'kd_clip', 'clip'])])
metric_boxplots(
    combined_metrics_df.loc[
        ~combined_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])
        & ~combined_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])
        ]
    )

if len(methods_id_dict['eclare']) > 1:
    metric_boxplots(pd.concat([*eclare_dfs.values()]))

#%% unpaired MDD data
experiment_name = f"clip_mdd_{methods_id_dict['clip_mdd']}"

if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
else:
    print(f"Downloading runs.csv for {experiment_name} from MLflow")
    all_metrics_csv_path = download_mlflow_runs(experiment_name)

mdd_metrics_df = pd.read_csv(all_metrics_csv_path)

CLIP_mdd_header_idxs = np.where(mdd_metrics_df['run_name'].str.startswith(search_strings['clip_mdd']))[0]
KD_CLIP_mdd_header_idxs = np.where(mdd_metrics_df['run_name'].str.startswith(search_strings['kd_clip_mdd']))[0]
ECLARE_mdd_header_idxs = np.where(mdd_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['eclare_mdd'])))[0]

CLIP_mdd_run_id = mdd_metrics_df.iloc[CLIP_mdd_header_idxs]['run_id']
KD_CLIP_mdd_run_id = mdd_metrics_df.iloc[KD_CLIP_mdd_header_idxs]['run_id']
ECLARE_mdd_run_id = mdd_metrics_df.iloc[ECLARE_mdd_header_idxs]['run_id']

CLIP_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(CLIP_mdd_run_id)]
KD_CLIP_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(KD_CLIP_mdd_run_id)]
ECLARE_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(ECLARE_mdd_run_id)]

CLIP_mdd_metrics_df = extract_target_source_replicate(CLIP_mdd_metrics_df)
KD_CLIP_mdd_metrics_df = extract_target_source_replicate(KD_CLIP_mdd_metrics_df)
ECLARE_mdd_metrics_df = extract_target_source_replicate(ECLARE_mdd_metrics_df, has_source=False)

## add dataset column
CLIP_mdd_metrics_df.loc[:, 'dataset'] = 'clip_mdd'
KD_CLIP_mdd_metrics_df.loc[:, 'dataset'] = 'kd_clip_mdd'
ECLARE_mdd_metrics_df.loc[:, 'dataset'] = 'eclare_mdd'

if len(methods_id_dict['eclare_mdd']) > 1:
    eclare_dfs = {}
    for search_key, dataset_name in search_strings_to_dataset.items():
        runs_from_eclare_experiment = ECLARE_metrics_df['parent_run_id'].isin(all_metrics_df.loc[all_metrics_df['run_name']==search_key]['run_id'].values)
        dataset_df = ECLARE_metrics_df.loc[runs_from_eclare_experiment]
        dataset_df.loc[:, 'dataset'] = dataset_name
        eclare_dfs[dataset_name] = dataset_df

    combined_metrics_df = pd.concat([
        *eclare_dfs.values(),
        KD_CLIP_metrics_df,
        CLIP_metrics_df
        ])

else:
    combined_mdd_metrics_df = pd.concat([
        ECLARE_mdd_metrics_df,
        KD_CLIP_mdd_metrics_df,
        CLIP_mdd_metrics_df
        ]) # determines order in which metrics are plotted

## if source and/or target contain 'multiome', convert to '10x'
combined_mdd_metrics_df.loc[:, 'source'] = combined_mdd_metrics_df['source'].str.replace('multiome', '10x')
combined_mdd_metrics_df.loc[:, 'target'] = combined_mdd_metrics_df['target'].str.replace('multiome', '10x')

## only keep runs with 'FINISHED' status
combined_mdd_metrics_df = combined_mdd_metrics_df[combined_mdd_metrics_df['status'] == 'FINISHED']

## plot boxplots for main models
metric_boxplots(combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['eclare_mdd', 'kd_clip_mdd', 'clip_mdd'])], 
                target_source_combinations=True, include_paired=False)

## plot boxplots for ablation studies
if len(methods_id_dict['eclare_mdd']) > 1:
    metric_boxplots(combined_mdd_metrics_df.loc[
        combined_mdd_metrics_df['dataset'].isin(['eclare_mdd', 'w/o_p_mdd', 'w/o_m_mdd', 'w/o_pm_mdd'])
        ], target_source_combinations=True, include_paired=False)


'''
## Get metrics
source_df_clip, target_df_clip, source_only_df_clip = get_metrics('clip', methods_id_dict['clip'])   # may need to rename 'triplet_align_<job_id>' by 'clip_<job_id>'
target_df_kd_clip = get_metrics('kd_clip', methods_id_dict['kd_clip'], target_only=True)
target_df_multiclip = get_metrics('scMulticlip', methods_id_dict['scMulticlip'], target_only=True) # may need to rename 'multisource_align_<job_id>' by 'multiclip_<job_id>'

mdd_df_clip = get_metrics('clip_mdd', methods_id_dict['clip_mdd'], target_only=True)
mdd_df_kd_clip = get_metrics('kd_clip_mdd', methods_id_dict['kd_clip_mdd'], target_only=True)
mdd_df_multiclip = get_metrics('scMulticlip_mdd', methods_id_dict['scMulticlip_mdd'], target_only=True)

source_df_mojitoo, target_df_mojitoo, source_only_df_mojitoo = get_metrics('mojitoo', methods_id_dict['mojitoo'])
source_df_multiVI, target_df_multiVI, source_only_df_multiVI = get_metrics('multiVI', methods_id_dict['multiVI'])
source_df_glue, target_df_glue, source_only_df_glue = get_metrics('glue', methods_id_dict['glue'])

source_df_scdart, target_df_scdart, source_only_df_scdart = get_metrics('scDART', methods_id_dict['scDART'])
source_df_scjoint, target_df_scjoint, source_only_df_scjoint = get_metrics('scjoint', methods_id_dict['scjoint'])

## check length of dataframes
if len(target_df_mojitoo) != 12*3:
    print(f'mojitoo: {len(target_df_mojitoo)}')
if len(target_df_multiVI) != 12*3:
    print(f'multiVI: {len(target_df_multiVI)}')
if len(target_df_glue) != 12*3:
    print(f'glue: {len(target_df_glue)}')
if len(target_df_scdart) != 12*3:
    print(f'scDART: {len(target_df_scdart)}')
if len(target_df_scjoint) != 12*3:
    print(f'scjoint: {len(target_df_scjoint)}')

if len(target_df_clip) != 12*3:
    print(f'clip: {len(target_df_clip)}')
if len(target_df_kd_clip) != 12*3:
    print(f'kd_clip: {len(target_df_kd_clip)}')
if len(target_df_multiclip) != 4*3:
    print(f'multiclip: {len(target_df_multiclip)}')

if len(mdd_df_clip) != 4*3:
    print(f'clip_mdd: {len(mdd_df_clip)}')
if len(mdd_df_kd_clip) != 4*3:
    print(f'kd_clip_mdd: {len(mdd_df_kd_clip)}')
if len(mdd_df_multiclip) != 1*3:
    print(f'multiclip_mdd: {len(mdd_df_multiclip)}')
'''

#%% compare KD-CLIP with pbmc and mouse brain VS ECLARE with pbmc + mouse brain

## paired data

## CLIP
ablation_clip_metrics_df = combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['clip'])]
ablation_clip_metrics_df = ablation_clip_metrics_df.loc[ablation_clip_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_clip_metrics_df = ablation_clip_metrics_df.loc[~ablation_clip_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_clip_metrics_df.loc[ablation_clip_metrics_df['source']=='pbmc_10x', 'dataset'] = 'pbmc (clip)'
ablation_clip_metrics_df.loc[ablation_clip_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'mb (clip)'

## KD-CLIP
ablation_kd_clip_metrics_df = combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['kd_clip'])]
ablation_kd_clip_metrics_df = ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_kd_clip_metrics_df = ablation_kd_clip_metrics_df.loc[~ablation_kd_clip_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source']=='pbmc_10x', 'dataset'] = 'pbmc (kd)'
ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'mb (kd)'

## ECLARE
ablation_eclare_metrics_df = combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['eclare'])]
ablation_eclare_metrics_df.loc[:, 'dataset'] = 'both'

## Combined ablation
ablation_metrics_df = pd.concat([
    ablation_eclare_metrics_df,
    ablation_kd_clip_metrics_df,
    ablation_clip_metrics_df
    ])

## Reorder rows using "dataset" field
ablation_metrics_df = ablation_metrics_df.sort_values('dataset')

metric_boxplots(ablation_metrics_df, target_source_combinations=False, include_paired=True)

## MDD

## CLIP
ablation_clip_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['clip_mdd'])]
ablation_clip_mdd_metrics_df = ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_clip_mdd_metrics_df = ablation_clip_mdd_metrics_df.loc[~ablation_clip_mdd_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source']=='pbmc_10x', 'dataset'] = 'pbmc (clip)'
ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'mb (clip)'

## KD-CLIP
ablation_kd_clip_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['kd_clip_mdd'])]
ablation_kd_clip_mdd_metrics_df = ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_kd_clip_mdd_metrics_df = ablation_kd_clip_mdd_metrics_df.loc[~ablation_kd_clip_mdd_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source']=='pbmc_10x', 'dataset'] = 'pbmc (kd)'
ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'mb (kd)'

## ECLARE
ablation_eclare_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['eclare_mdd'])]
ablation_eclare_mdd_metrics_df.loc[:, 'dataset'] = 'both'

## Combined ablation
ablation_mdd_metrics_df = pd.concat([
    ablation_eclare_mdd_metrics_df,
    ablation_kd_clip_mdd_metrics_df,
    ablation_clip_mdd_metrics_df
    ])

## Reorder rows using "dataset" field
ablation_mdd_metrics_df = ablation_mdd_metrics_df.sort_values('dataset')

metric_boxplots(ablation_mdd_metrics_df, target_source_combinations=True, include_paired=False)


#%%
metrics = ['1 - foscttm_score', 'ilisis', 'ari', 'nmi', 'asw_ct']
metrics_mdd = ['ilisis', 'ari', 'nmi', 'asw_ct']

methods = ['clip', 'mojitoo', 'multiVI', 'glue', 'scDART', 'scJoint']
methods_clips = ['eclare', 'kd-clip', 'clip']
methods_clips_mdd = ['eclare', 'kd-clip', 'clip']

source_only_dataframes = [source_only_df_clip, source_only_df_mojitoo, source_only_df_multiVI, source_only_df_glue, source_only_df_scdart, source_only_df_scjoint]
target_dataframes = [target_df_clip, target_df_mojitoo, target_df_multiVI, target_df_glue, target_df_scdart, target_df_scjoint]
target_dataframes_clips = [target_df_multiclip, target_df_kd_clip, target_df_clip]
target_dataframes_clips_mdd = [mdd_df_multiclip, mdd_df_kd_clip, mdd_df_clip]

## replace 'clisis' by 'asw-ct' dataframes
for method in methods:
    if 'clisis' in source_only_dataframes[methods.index(method)].columns:
        source_only_dataframes[methods.index(method)] = source_only_dataframes[methods.index(method)].rename(columns={'clisis': 'asw_ct'})

    if 'clisis' in target_dataframes[methods.index(method)].columns:
        target_dataframes[methods.index(method)] = target_dataframes[methods.index(method)].rename(columns={'clisis': 'asw_ct'})

    if 'clisis' in source_df_clip.columns:
        source_df_clip = source_df_clip.rename(columns={'clisis': 'asw_ct'})

for method in methods_clips:
    if 'clisis' in target_dataframes_clips[methods_clips.index(method)].columns:
        target_dataframes_clips[methods_clips.index(method)] = target_dataframes_clips[methods_clips.index(method)].rename(columns={'clisis': 'asw_ct'})

for method in methods_clips_mdd:
    if 'clisis' in target_dataframes_clips_mdd[methods_clips_mdd.index(method)].columns:
        target_dataframes_clips_mdd[methods_clips_mdd.index(method)] = target_dataframes_clips_mdd[methods_clips_mdd.index(method)].rename(columns={'clisis': 'asw_ct'})


#fig2_benchmark_source_only = combined_plot(source_only_dataframes, metrics, methods, target_source_combinations=False, figsize=(14,6))
fig2_benchmark      = combined_plot(target_dataframes, metrics, methods, target_source_combinations=False, figsize=(14,4))
fig2_benchmark_with_source_to_source = combined_plot(target_dataframes, metrics, methods, target_source_combinations=False, source_only_dataframes=source_only_dataframes, figsize=(14,11))
fig2_benchmark_with_source_shape_coding = combined_plot(target_dataframes_clips, metrics, methods_clips, target_source_combinations=True, figsize=(14,4))

fig3_clips_paired   = combined_plot(target_dataframes_clips, metrics, methods_clips, target_source_combinations=False, figsize=(14,4))
fig4_clips_mdd      = combined_plot(target_dataframes_clips_mdd, metrics_mdd, methods_clips_mdd, target_source_combinations=True, figsize=(11,4))

## Save figures
figpath = "/Users/dmannk/Library/CloudStorage/OneDrive-McGillUniversity/Doctorat/QLS/Li_Lab/ISMB_2025/figures"

fig2_benchmark.savefig(os.path.join(figpath, 'fig2_benchmark.png'), bbox_inches='tight', dpi=300)
fig2_benchmark_with_source_to_source.savefig(os.path.join(figpath, 'fig2_benchmark_with_source_to_source.png'), bbox_inches='tight', dpi=300)
fig2_benchmark_with_source_shape_coding.savefig(os.path.join(figpath, 'fig2_benchmark_with_source_shape_coding.png'), bbox_inches='tight', dpi=300)

fig3_clips_paired.savefig(os.path.join(figpath, 'fig3_clips_paired.png'), bbox_inches='tight', dpi=300)
fig4_clips_mdd.savefig(os.path.join(figpath, 'fig4_clips_mdd.png'), bbox_inches='tight', dpi=300)


#%% MDD GRN analysis

device = 'cuda' if cuda_available else 'cpu'

## Find path to best ECLARE, KD-CLIP and CLIP models
best_eclare_mdd     = str(ECLARE_mdd_metrics_df['multimodal_ilisi'].argmax())
best_kd_clip_mdd    = KD_CLIP_mdd_metrics_df['source'].iloc[KD_CLIP_mdd_metrics_df['multimodal_ilisi'].argmax()]

def load_model_and_metadata(student_job_id, best_model_idx, target_dataset='MDD'):

    ## replace 'multiome' by '10x'
    best_model_idx = best_model_idx.replace('multiome', '10x')
 
    paths_root = os.path.join(os.environ['OUTPATH'], student_job_id)
    student_model_path = os.path.join(paths_root, target_dataset, best_model_idx, 'model_uri.txt')

    ## Load the model and metadata
    with open(student_model_path, 'r') as f:
        model_uris = f.read().strip().splitlines()
        model_uri = model_uris[0]

    ## Set tracking URI to ECLARE_ROOT/mlruns and load model & metadata
    mlflow.set_tracking_uri(os.path.join(os.environ['ECLARE_ROOT'], 'mlruns'))

    run_id = model_uri.split('/')[1]
    print(f'run_id: {run_id}')

    model_dir = os.path.join(os.environ['ECLARE_ROOT'], "mlruns", '*', run_id, "artifacts", "trained_model")
    model_dir = glob(model_dir)[0]
    print(f'model_dir: {model_dir}')

    model_uri = f"file://{model_dir}"
    student_model = mlflow.pytorch.load_model(model_uri, map_location=device)
    student_model_metadata = Model.load(model_dir)

    #student_model = mlflow.pytorch.load_model(model_uri)
    #student_model_metadata = Model.load(model_uri)

    return student_model, student_model_metadata

#eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_mdd_{methods_id_dict["eclare_mdd"]}', best_eclare_mdd)
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_mdd_{methods_id_dict["eclare_mdd"][0]}', best_eclare_mdd)
kd_clip_student_model, kd_clip_student_model_metadata   = load_model_and_metadata(f'kd_clip_mdd_{methods_id_dict["kd_clip_mdd"]}', os.path.join(best_kd_clip_mdd, '0'))

eclare_student_model = eclare_student_model.train().to('cpu')
kd_clip_student_model = kd_clip_student_model.train().to('cpu')

#%% load data

args = SimpleNamespace(
    source_dataset='MDD',
    target_dataset=None,
    genes_by_peaks_str='17563_by_100000'
)

'''
mdd_rna, mdd_atac, mdd_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, _, _ = \
    mdd_setup(args, return_raw_data=True, return_type='data',\
    overlapping_subjects_only=False)
'''
import anndata
rna_datapath = atac_datapath = os.path.join(os.environ['DATAPATH'], 'mdd_data')

RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"

rna_fullpath = os.path.join(rna_datapath, RNA_file)
atac_fullpath = os.path.join(atac_datapath, ATAC_file)

atac = anndata.read_h5ad(atac_fullpath, backed='r')
rna  = anndata.read_h5ad(rna_fullpath, backed='r')

## get data after decimation
decimate_factor = 1

mdd_atac = atac[::decimate_factor].to_memory()
mdd_rna = rna[::decimate_factor].to_memory()

mdd_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(list(mdd_atac.var_names.str.split(':|-', expand=True)), columns=['chrom', 'start', 'end']))

## also load counts data for pyDESeq2 and full processed data for annotations
rna_full = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna.h5ad'), backed='r')
rna_scaled_with_counts = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna_scaled.h5ad'), backed='r')
rna_scaled_with_counts = rna_scaled_with_counts[::decimate_factor].to_memory()

rna_counts_X = rna_scaled_with_counts.raw.X.astype(int).toarray()
rna_counts_obs = rna_scaled_with_counts.obs
rna_counts_var = rna_full[::decimate_factor].var

mdd_rna_counts = anndata.AnnData(
    X=rna_counts_X,
    var=rna_counts_var,
    obs=rna_counts_obs,
)

## define keys
rna_celltype_key='ClustersMapped'
atac_celltype_key='ClustersMapped'

rna_condition_key='Condition'
atac_condition_key='condition'

rna_subject_key='OriginalSub'
atac_subject_key='BrainID'

rna_sex_key = 'Sex'
atac_sex_key = 'sex'

unique_celltypes = np.unique(np.concatenate([mdd_rna.obs[rna_celltype_key], mdd_atac.obs[atac_celltype_key]]))
unique_conditions = np.unique(np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]))
unique_sexes = np.unique(np.concatenate([mdd_rna.obs[rna_sex_key].str.lower(), mdd_atac.obs[atac_sex_key].str.lower()]))


## prepend subject label with 'case_' or 'control_'
mdd_rna.obs[rna_subject_key] = mdd_rna.obs[rna_subject_key].astype(str)
mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Case'] = mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Case'].apply(lambda x: f'case_{x}')
mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Control'] = mdd_rna.obs[rna_subject_key][mdd_rna.obs[rna_condition_key] == 'Control'].apply(lambda x: f'control_{x}')

mdd_atac.obs[atac_subject_key] = mdd_atac.obs[atac_subject_key].astype(str)
mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Case'] = mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Case'].apply(lambda x: f'case_{x}')
mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Control'] = mdd_atac.obs[atac_subject_key][mdd_atac.obs[atac_condition_key] == 'Control'].apply(lambda x: f'control_{x}')

subjects_by_condition_n_sex_df = pd.DataFrame({
    'subject': np.concatenate([mdd_rna.obs[rna_subject_key], mdd_atac.obs[atac_subject_key]]),
    'condition': np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]),
    'sex': np.concatenate([mdd_rna.obs[rna_sex_key].str.lower(), mdd_atac.obs[atac_sex_key].str.lower()])
})
overlapping_subjects = np.intersect1d(mdd_rna.obs[rna_subject_key], mdd_atac.obs[atac_subject_key])
subjects_by_condition_n_sex_df = subjects_by_condition_n_sex_df[subjects_by_condition_n_sex_df['subject'].isin(overlapping_subjects)]
subjects_by_condition_n_sex_df = subjects_by_condition_n_sex_df.groupby(['condition', 'sex'])['subject'].unique()


#%% differential expression analysis

tmp = mdd_rna[(mdd_rna.obs[rna_sex_key]=='Female') & (mdd_rna.obs[rna_celltype_key]=='Mic')]
sc.tl.rank_genes_groups(tmp, groupby=rna_condition_key, reference='Control', method='wilcoxon')

## shows that unique Batch and Chemistry per Sample
confound_vars = ["Batch", "Sample", "Chemistry", "percent.mt", "nCount_RNA"]
display(mdd_rna.obs.groupby('Sample')[confound_vars].nunique())

## pyDESeq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

ct_map_dict = dict({1: 'ExN', 0: 'InN', 4: 'Oli', 2: 'Ast', 3: 'OPC', 6: 'End', 5: 'Mix', 7: 'Mic'})
rna_scaled_with_counts.obs[rna_celltype_key] = rna_scaled_with_counts.obs['Broad'].map(ct_map_dict)

sex = 'Female'
celltype = 'Mic'

pseudo_replicates = 'Subjects'

if pseudo_replicates == 'SEACells':
    ## learn SEACell assignments based on processed data to apply onto counts data

    mdd_seacells_counts_dict = {}

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
            n_SEACells=max(rna_sampled.n_obs // 100, 15), 
            n_waypoint_eigs=15,
            convergence_epsilon = 1e-5,
            max_franke_wolfe_iters=100,
            use_gpu=True if torch.cuda.is_available() else False
            )
        
        seacells_model.construct_kernel_matrix()
        seacells_model.initialize_archetypes()
        seacells_model.fit(min_iter=10, max_iter=100)

        ## summarize counts data by SEACell - remove 'nan' obs_names which groups cells not corresponding to celltype or sex
        mdd_rna_counts.obs.loc[rna_sampled.obs_names,'SEACell'] = rna_sampled.obs['SEACell'].add(f'_{condition}_{sex}_{celltype}')
        mdd_seacells_counts = SEACells.core.summarize_by_SEACell(mdd_rna_counts, SEACells_label='SEACell', summarize_layer='X')
        mdd_seacells_counts = mdd_seacells_counts[mdd_seacells_counts.obs_names != 'nan']

        mdd_seacells_counts.obs[rna_condition_key] = condition
        mdd_seacells_counts.obs[rna_sex_key] = sex
        mdd_seacells_counts.obs[rna_celltype_key] = celltype

        mdd_seacells_counts_dict[condition] = mdd_seacells_counts

    ## concatenate all SEACell counts data across conditions
    mdd_seacells_counts_adata = anndata.concat(mdd_seacells_counts_dict.values(), axis=0)
    mdd_seacells_counts_adata = mdd_seacells_counts_adata[:, mdd_seacells_counts_adata.var_names.isin(mdd_rna.var_names)]
    mdd_seacells_counts_adata.var = mdd_rna.var

    counts = mdd_seacells_counts_adata.X.astype(int).toarray()
    metadata = mdd_seacells_counts_adata.obs

elif pseudo_replicates == 'Subjects':

    mdd_subjects_counts_dict = {}

    for subject in overlapping_subjects:

        rna_indices =  pd.DataFrame({
            'is_subject': mdd_rna.obs[rna_subject_key].str.startswith(subject),
            'is_celltype': mdd_rna.obs[rna_celltype_key].str.startswith(celltype)
        })
        rna_indices = rna_indices.prod(axis=1).astype(bool).values.nonzero()[0]

        rna_sampled = rna_scaled_with_counts[rna_indices]
        rna_subject_counts = rna_sampled.raw.X.sum(axis=0).A1.astype(int)
        rna_subject_var = rna_sampled.raw.var.set_index('_index')

        subject_condition = mdd_rna.obs[rna_condition_key][mdd_rna.obs[rna_subject_key] == subject].iloc[0]
        subject_sex = mdd_rna.obs[rna_sex_key][mdd_rna.obs[rna_subject_key] == subject].iloc[0]
        rna_subject_obs = pd.DataFrame(
            np.hstack([subject_condition, subject_sex, celltype]).reshape(1, -1),
            columns=[rna_condition_key, rna_sex_key, rna_celltype_key],
            index=[subject],
        )

        rna_subject_counts_ad = anndata.AnnData(
            X=rna_subject_counts.reshape(1, -1),
            var=rna_subject_var,
            obs=rna_subject_obs,
        )
        mdd_subjects_counts_dict[subject] = rna_subject_counts_ad

    mdd_subjects_counts_adata = anndata.concat(mdd_subjects_counts_dict.values(), axis=0)
    mdd_subjects_counts_adata = mdd_subjects_counts_adata[:, mdd_subjects_counts_adata.var_names.isin(mdd_rna.var_names)]
    mdd_subjects_counts_adata.var = mdd_rna.var

    ## retain sex of interest
    mdd_subjects_counts_adata = mdd_subjects_counts_adata[mdd_subjects_counts_adata.obs[rna_sex_key] == sex]

    counts = mdd_subjects_counts_adata.X.astype(int).toarray()
    metadata = mdd_subjects_counts_adata.obs


## run pyDESeq2
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts,
    metadata=metadata,
    design_factors=rna_condition_key,
    refit_cooks=True,
    inference=inference,
)
dds.deseq2()
stat_res = DeseqStats(dds, inference=inference)
stat_res.summary()

## get results and volcano plot
results = stat_res.results_df
results['signif_padj'] = results['padj'] < 0.05
results['signif_lfc'] = results['log2FoldChange'].abs() > 1.5
results['signif'] = results['signif_padj'] & results['signif_lfc']
results['pH'] = -np.log10(results['padj'])
sns.scatterplot(data=results, x='log2FoldChange', y='pH', hue='signif_padj', marker='o', alpha=0.5)

## extract significant genes
significant_genes = mdd_subjects_counts_adata.var_names[results['signif_padj']]
mdd_subjects_counts_adata.var.loc[significant_genes, 'signif_padj'] = True

## violin plot
df = mdd_subjects_counts_adata[:,mdd_subjects_counts_adata.var_names.isin(significant_genes[:10])].to_df()
df = df.reset_index()
df = pd.melt(df, id_vars=['index'], var_name='gene', value_name='expression')
df = df.merge(mdd_subjects_counts_adata.obs, left_on='index', right_index=True)

sns.violinplot(data=df, x=rna_condition_key, y='X', hue=rna_celltype_key)


#%% Get mean GRN from brainSCOPE & scglue preprocessing

grn_path = os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'GRNs')
mean_grn_df = get_unified_grns(grn_path)

## get gene annotation and position
scglue.data.get_gene_annotation(
    mdd_rna, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'),
    gtf_by="gene_name"
)
mdd_rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
#mdd_rna_chrom = mdd_rna[:, mdd_rna.var['chrom'].notna()] # will result in number of genes mismatching with student model

## get peak position
split = mean_grn_df['enhancer'].str.split(r"[:-]")
mean_grn_df["chrom"] = split.map(lambda x: x[0])
mean_grn_df["chromStart"] = split.map(lambda x: x[1]).astype(int)
mean_grn_df["chromEnd"] = split.map(lambda x: x[2]).astype(int)

## extract gene and peak positions
gene_ad = anndata.AnnData(var=pd.DataFrame(index=mean_grn_df['TG'].unique()))
scglue.data.get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')
gene_ad = gene_ad[:,gene_ad.var['name'].notna()]

genes = scglue.genomics.Bed(gene_ad.var)
tss = genes.strand_specific_start_site()
promoters = tss.expand(2000, 0)

peaks = scglue.genomics.Bed(mean_grn_df.assign(name=mean_grn_df['enhancer'])).drop_duplicates()

## distance graph
window_size = 1e6
dist_graph = scglue.genomics.window_graph(
    promoters, peaks, window_size,
    attr_fn=lambda l, r, d: {
        "dist": abs(d),
        "weight": scglue.genomics.dist_power_decay(abs(d)),
        "type": "dist"
    }
)
dist_graph = nx.DiGraph(dist_graph)
dist_graph.number_of_edges()

def graph_to_df(G):
    """
    Convert a NetworkX DiGraph G into a pandas DataFrame
    with columns: source, target, weight, type (and any other attrs).
    """
    rows = []
    for u, v, attrs in G.edges(data=True):
        row = {
            "source": u,
            "target": v,
            **attrs
        }
        rows.append(row)
    return pd.DataFrame(rows)

# e.g. for the distance graph
df_dist = graph_to_df(dist_graph)
print(df_dist.head())

## merge distance graph with mean_grn_df
mean_grn_df = mean_grn_df.merge(df_dist, left_on=['TG', 'enhancer'], right_on=['source', 'target'], how='left')

## get JASPAR2018 TFs db
jaspar_db_2020 = jaspardb(release='JASPAR2020')
jaspar_db_2024 = jaspardb(release='JASPAR2024')

tfs = list(mean_grn_df['TF'].unique())
#motifs = {tf:jaspar_db.fetch_motifs(tf_name=tf, tax_group='vertebrates') for tf in tfs} # ~25 seconds
#motifs = jaspar_db_2018.fetch_motifs(tf_name=tfs, tax_group='vertebrates')

mean_grn_df['motif_score'] = 0
tf_enhancer_grns = list(mean_grn_df.groupby('TF')['enhancer'].unique().items())

for tf, enhancers in tqdm(tf_enhancer_grns, total=len(tf_enhancer_grns)):

    try:
        motifs = jaspar_db_2020.fetch_motifs(tf_name=tf, tax_group='vertebrates')
        motif = motifs[0]
    except:
        try:
            motifs = jaspar_db_2024.fetch_motifs_by_name(tf)
            motif = motifs[0]
        except:
            print(f"Could not find motif for TF {tf} in either JASPAR2020 or JASPAR2024")
            continue

    motif.pseudocounts = calculate_pseudocounts(motif)  # also have motif.pseudo_counts, not sure what the difference is...

    enhancers_df = pd.DataFrame(enhancers)[0].str.split(':|-', expand=True).rename(columns={0: 'Chromosome', 1: 'Start', 2: 'End'})
    gr = pr.from_dict(enhancers_df.to_dict()) # can extend with pr.extend(k)
    seqs = pr.get_sequence(gr, os.path.join(os.environ['DATAPATH'], 'hg38.fa')) # ~53 seconds

    motif_scores = {enhancer: motif.pssm.calculate(seq.upper()).max() for enhancer, seq in zip(enhancers, seqs)} # ~44 seconds
    mean_grn_df.loc[mean_grn_df['TF'] == tf, 'motif_score'] = mean_grn_df.loc[mean_grn_df['TF'] == tf, 'enhancer'].map(motif_scores)

## remove TFs with no motif score
print(f"Removing {mean_grn_df[mean_grn_df['motif_score'] == 0]['TF'].nunique()} TFs out of {mean_grn_df['TF'].nunique()} with no motif score")
mean_grn_df = mean_grn_df[mean_grn_df['motif_score'] > 0]
mean_grn_df.reset_index(drop=True, inplace=True)  # need to reset indices to enable alignment with normed values

## normalize motif score by target gene TG
#motif_score_norm = mean_grn_df.groupby('TG')['motif_score'].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # guaranteed to have one zero-score per TG
temperature = mean_grn_df['motif_score'].var()
softmax_temp_func = lambda x, temperature: np.exp(x / temperature)# / np.exp(x / temperature).sum()
motif_score_norm = mean_grn_df.groupby('TG')['motif_score'].apply(lambda x: softmax_temp_func(x, temperature))
motif_score_norm = motif_score_norm.fillna(1)  # NaN motif scores because only one motif score for some TGs

motif_score_norm = motif_score_norm.reset_index(level=0, drop=False).rename(columns={'motif_score': 'motif_score_norm'})
assert (motif_score_norm.index.sort_values() == np.arange(len(motif_score_norm))).all()

## merge motif scores with mean_grn_df
mean_grn_df = mean_grn_df.merge(motif_score_norm, left_index=True, right_index=True, how='left', suffixes=('', '_motifs'))


#%% project MDD nuclei into latent space

mdd_rna_sampled, mdd_atac_sampled, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition = \
   sample_proportional_celltypes_and_condition(mdd_rna, mdd_atac, batch_size=5000)

## extract sex labels (ideally, extract from sample_proportional_celltypes_and_condition)
mdd_rna_sex = mdd_rna_sampled.obs[rna_sex_key].str.lower()
mdd_atac_sex = mdd_atac_sampled.obs[atac_sex_key].str.lower()

# subset to 'Mic' celltype right away
'''
mdd_rna_sampled = mdd_rna[mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_sampled = mdd_atac[mdd_atac.obs[atac_celltype_key] == 'Mic']
mdd_rna_condition = mdd_rna.obs[rna_condition_key][mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_condition = mdd_atac.obs[atac_condition_key][mdd_atac.obs[atac_celltype_key] == 'Mic']
mdd_rna_celltypes = mdd_rna.obs[rna_celltype_key][mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_celltypes = mdd_atac.obs[atac_celltype_key][mdd_atac.obs[atac_celltype_key] == 'Mic']
'''

## get latents
eclare_rna_latents, eclare_atac_latents = get_latents(eclare_student_model, mdd_rna_sampled, mdd_atac_sampled, return_tensor=False)
kd_clip_rna_latents, kd_clip_atac_latents = get_latents(kd_clip_student_model, mdd_rna_sampled, mdd_atac_sampled, return_tensor=False)

## concatenate latents
eclare_latents = np.concatenate([eclare_rna_latents, eclare_atac_latents], axis=0)
kd_clip_latents = np.concatenate([kd_clip_rna_latents, kd_clip_atac_latents], axis=0)
#clip_latents = np.concatenate([clip_rna_latents, clip_atac_latents], axis=0)

## concatenate sexes
mdd_sexes = np.concatenate([mdd_rna_sex, mdd_atac_sex], axis=0)

## plot umaps
color_map_ct = create_celltype_palette(unique_celltypes, unique_celltypes, plot_color_palette=False)
_, fig, rna_atac_df_umap = plot_umap_embeddings(eclare_rna_latents, eclare_atac_latents, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition, color_map_ct=color_map_ct, umap_embedding=None)
#plot_umap_embeddings(kd_clip_rna_latents, kd_clip_atac_latents, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition, color_map_ct=color_map_ct, umap_embedding=None)


#%% get peak-gene correlations

def tree(): return defaultdict(tree)

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]

X_rna_dict = tree()
X_atac_dict = tree()
overlapping_target_genes_dict = tree()
overlapping_tfs_dict = tree()

genes_by_peaks_corrs_dict = tree()
genes_by_peaks_masks_dict = tree()
n_dict = tree()

scompreg_loglikelihoods_dict = tree()
std_errs_dict = tree()
slopes_dict = tree()
intercepts_dict = tree()
intercept_stderrs_dict = tree()

tg_expressions_dict = tree()
tfrps_dict = tree()
tfrp_predictions_dict = tree()

## set cutoff for number of cells to keep for SEACells representation. see Bilous et al. 2024, Liu & Li 2024 (mcRigor) or Li et al. 2025 (MetaQ) for benchmarking experiments
cutoff = 5025 # better a multiple of 75 due to formation of SEACells

maitra_female_degs_df   = pd.read_excel(os.path.join(os.environ['DATAPATH'], 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData7', header=2)
doruk_peaks_df          = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'combined', 'cluster_DAR_0.2.tsv'), sep='\t')
#maitra_male_degs_df = pd.read_excel(os.path.join(datapath, 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData5', header=2)

deg_dict = maitra_female_degs_df.groupby('cluster_id.female')['gene'].unique().to_dict()

deg_overlap_df = pd.DataFrame.from_dict({key: np.isin(deg_dict[key], significant_genes).sum() for key in deg_dict.keys()}, orient='index').rename(columns={0: 'Case'})
deg_overlap_df = pd.concat([deg_overlap_df, deg_overlap_df.sum(axis=0).to_frame().T.rename(index={0: 'TOTAL'})], axis=0)

for celltype in deg_overlap_df.index:
    try:
        unique_celltype = [unique_celltype for unique_celltype in unique_celltypes if unique_celltype in celltype][0]
        deg_overlap_df.loc[celltype, 'unique_celltype'] = unique_celltype
    except:
        print(f"Could not find unique celltype for {celltype}")
        continue

deg_overlap_grouped_df = deg_overlap_df.groupby('unique_celltype').sum()
display(deg_overlap_grouped_df.sort_values(by='Case', ascending=False).T)

## define sex and celltype
sex='Female'
celltype='Oli'
do_corrs_or_cosines = ''

## get HVG features
'''
sc.pp.highly_variable_genes(mdd_rna)
sc.pp.highly_variable_genes(mdd_atac, n_top_genes=10000)

## use all peaks and genes
genes_indices_hvg = mdd_rna.var['highly_variable'].astype(bool)
peaks_indices_hvg = mdd_atac.var['highly_variable'].astype(bool)
'''


if not os.path.exists(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl')):

    for celltype in unique_celltypes:

        celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id.female'].str.startswith(celltype)]
        #celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id'].str.startswith(celltype)]

        celltype_peaks_df = doruk_peaks_df[doruk_peaks_df['cluster'].str.startswith(celltype)]

        #celltype_peaks = celltype_peaks_df['peakName'].str.split('-')
        #celltype_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame({'chrom': celltype_peaks.str[0], 'start': celltype_peaks.str[1], 'end': celltype_peaks.str[2]}))

        celltype_peaks = celltype_peaks_df[['Chromosome', 'Start', 'End']].values
        celltype_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame({'chrom': celltype_peaks[:,0], 'start': celltype_peaks[:,1], 'end': celltype_peaks[:,2]}))
        
        ## select peaks indices using bedtools intersect
        #peaks_indices_dar = mdd_peaks_bed.intersect(celltype_peaks_bed, c=True).to_dataframe()['name'].astype(bool)
        genes_indices_deg = mdd_rna.var_names.isin(celltype_degs_df['gene'])

        #peaks_indices = peaks_indices_hvg.values | peaks_indices_dar.values
        #genes_indices = genes_indices_hvg.values | genes_indices_deg
        #peaks_indices = peaks_indices_dar
        #genes_indices = genes_indices_deg
        #genes_indices = np.ones(len(mdd_rna.var_names), dtype=bool)

        for condition in unique_conditions:

            mdd_rna_aligned = []
            mdd_atac_aligned = []

            all_rna_indices = []
            all_atac_indices = []

            for subject in subjects_by_condition_n_sex_df[condition, sex.lower()]:

                #print(f'sex: {sex} - celltype: {celltype} - condition: {condition} - DAR peaks: {mdd_atac.n_vars} - DEG genes: {mdd_rna.n_vars}')

                ## select cell indices
                rna_indices = pd.DataFrame({
                    'is_celltype': mdd_rna.obs[rna_celltype_key].str.startswith(celltype),
                    'is_condition': mdd_rna.obs[rna_condition_key].str.startswith(condition),
                    'is_sex': mdd_rna.obs[rna_sex_key].str.lower().str.contains(sex.lower()),
                    'is_subject': mdd_rna.obs[rna_subject_key] == subject # do not use startswith to avoid multiple subjects
                }).prod(axis=1).astype(bool).values.nonzero()[0]

                atac_indices = pd.DataFrame({
                    'is_celltype': mdd_atac.obs[atac_celltype_key].str.startswith(celltype),
                    'is_condition': mdd_atac.obs[atac_condition_key].str.startswith(condition),
                    'is_sex': mdd_atac.obs[atac_sex_key].str.lower().str.contains(sex.lower()),
                    'is_subject': mdd_atac.obs[atac_subject_key] == subject # do not use startswith to avoid multiple subjects
                }).prod(axis=1).astype(bool).values.nonzero()[0]

                all_rna_indices.append(pd.DataFrame(np.vstack([rna_indices, [subject]*len(rna_indices), [celltype]*len(rna_indices), [condition]*len(rna_indices), [sex]*len(rna_indices)]).T, columns=['index', 'subject', 'celltype', 'condition', 'sex']))
                all_atac_indices.append(pd.DataFrame(np.vstack([atac_indices, [subject]*len(atac_indices), [celltype]*len(atac_indices), [condition]*len(atac_indices), [sex]*len(atac_indices)]).T, columns=['index', 'subject', 'celltype', 'condition', 'sex']))

                assert len(rna_indices) > 0 and len(atac_indices) > 0, f"No indices found for sex: {sex} - celltype: {celltype} - condition: {condition} - subject: {subject}"

                ## sample indices
                if len(rna_indices) > cutoff:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
                    _, sampled_indices = next(sss.split(rna_indices, mdd_rna.obs[rna_celltype_key].iloc[rna_indices]))
                    rna_indices = rna_indices[sampled_indices]
                if len(atac_indices) > cutoff:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
                    _, sampled_indices = next(sss.split(atac_indices, mdd_atac.obs[atac_celltype_key].iloc[atac_indices]))
                    atac_indices = atac_indices[sampled_indices]

                ## sample data
                mdd_rna_sampled_group = mdd_rna[rna_indices]
                mdd_atac_sampled_group = mdd_atac[atac_indices]

                ## get latents
                rna_latents, atac_latents = get_latents(eclare_student_model, mdd_rna_sampled_group, mdd_atac_sampled_group, return_tensor=True)
                rna_latents = rna_latents.cpu()
                atac_latents = atac_latents.cpu()

                ## get logits - already normalized during clip loss, but need to normalize before to be consistent with Concerto
                rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
                atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
                student_logits = torch.matmul(atac_latents, rna_latents.T)

                a, b = torch.ones((len(atac_latents),)) / len(atac_latents), torch.ones((len(rna_latents),)) / len(rna_latents)

                cells_gap = np.abs(len(atac_latents) - len(rna_latents))

                ## if imbalance, use partial wasserstein to find cells to reject and resample accordingly
                if cells_gap > 0:
                    print(f'cells_gap: {cells_gap}')
                    mdd_atac_sampled_group, mdd_rna_sampled_group, student_logits = \
                        cell_gap_ot(student_logits, atac_latents, rna_latents, mdd_atac_sampled_group, mdd_rna_sampled_group, cells_gap, type='emd')

                ## compute optimal transport plan for alignment on remaining cells
                res = ot_solve(1 - student_logits)
                plan = res.plan
                value = res.value_linear

                ## re-order ATAC latents to match plan (can rerun OT analysis to ensure diagonal matching structure)
                atac_latents = atac_latents[plan.argmax(axis=0)]
                mdd_atac_sampled_group = mdd_atac_sampled_group[plan.argmax(axis=0).numpy()]

                ## append to list
                mdd_rna_aligned.append(mdd_rna_sampled_group)
                mdd_atac_aligned.append(mdd_atac_sampled_group)

            ## concatenate aligned anndatas
            mdd_rna_aligned = anndata.concat(mdd_rna_aligned, axis=0)
            mdd_atac_aligned = anndata.concat(mdd_atac_aligned, axis=0)

            assert np.equal(mdd_rna_aligned.obs[rna_subject_key].values, mdd_atac_aligned.obs[atac_subject_key].values).all()
            assert (mdd_rna_aligned.obs_names.nunique() == mdd_rna_aligned.n_obs) & (mdd_atac_aligned.obs_names.nunique() == mdd_atac_aligned.n_obs)

            mdd_rna_aligned.var = mdd_rna.var
            mdd_atac_aligned.var = mdd_atac.var

            ## plot UMAP of aligned RNA latents and ATAC latents
            rna_latents, atac_latents = get_latents(eclare_student_model, mdd_rna_aligned, mdd_atac_aligned, return_tensor=True)

            umap_embeddings = UMAP(n_neighbors=50, min_dist=0.5, n_components=2, metric='cosine', random_state=42)
            umap_embeddings.fit(np.concatenate([rna_latents, atac_latents], axis=0))
            rna_umap = umap_embeddings.transform(rna_latents)
            atac_umap = umap_embeddings.transform(atac_latents)

            rna_df_umap = pd.DataFrame(data={'umap_1': rna_umap[:, 0], 'umap_2': rna_umap[:, 1],'modality': 'RNA'})
            atac_df_umap = pd.DataFrame(data={'umap_1': atac_umap[:, 0], 'umap_2': atac_umap[:, 1], 'modality': 'ATAC'})
            rna_atac_df_umap = pd.concat([rna_df_umap, atac_df_umap], axis=0)#.sample(frac=1) # shuffle            sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, ax=ax[2], legend=True, marker=marker)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sns.scatterplot(data=rna_atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, legend=True, marker='.', ax=ax[0])
            sns.scatterplot(data=rna_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, legend=False, marker='.', ax=ax[1])
            sns.scatterplot(data=atac_df_umap, x='umap_1', y='umap_2', hue='modality', hue_order=['ATAC','RNA'], alpha=0.5, legend=False, marker='.', ax=ax[2])
            ax[0].set_xticklabels([]); ax[0].set_yticklabels([]); ax[0].set_xlabel(''); ax[0].set_ylabel('')
            ax[1].set_xticklabels([]); ax[1].set_yticklabels([]); ax[1].set_xlabel(''); ax[1].set_ylabel('')
            ax[2].set_xticklabels([]); ax[2].set_yticklabels([]); ax[2].set_xlabel(''); ax[2].set_ylabel('')
            plt.show()

            ## get mean GRN from brainSCOPE
            mean_grn_df, mdd_rna_sampled_group, mdd_atac_sampled_group = filter_mean_grn(mean_grn_df, mdd_rna_aligned, mdd_atac_aligned)

            overlapping_target_genes = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_target_gene']].index.values
            overlapping_tfs = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_tf']].index.values

            ## remove excess cells beyond cutoff with stratified sampling - or else SEACells way too slow
            if len(mdd_rna_sampled_group) > cutoff:
                print(f'Removing {len(mdd_rna_sampled_group) - cutoff} cells')

                sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
                _, sampled_indices = next(sss.split(mdd_rna_sampled_group.obs_names, mdd_rna_sampled_group.obs[rna_subject_key]))

                mdd_rna_sampled_group = mdd_rna_sampled_group[sampled_indices]
                mdd_atac_sampled_group = mdd_atac_sampled_group[sampled_indices]

            ## run SEACells to obtain pseudobulked counts
            mdd_rna_sampled_group_seacells, mdd_atac_sampled_group_seacells = \
                run_SEACells(mdd_rna_sampled_group, mdd_atac_sampled_group, build_kernel_on='X_pca', key='X_UMAP_pca')

            X_rna = torch.from_numpy(mdd_rna_sampled_group_seacells.X.toarray())
            X_atac = torch.from_numpy(mdd_atac_sampled_group_seacells.X.toarray())

            #X_rna = torch.from_numpy(mdd_rna_sampled_group.X.toarray())
            #X_atac = torch.from_numpy(mdd_atac_sampled_group.X.toarray())

            ## select DEG genes and DAR peaks
            #X_rna = X_rna[:,genes_indices]
            #X_atac = X_atac[:,peaks_indices]

            ## detect cells with no gene expression and no chromatin accessibility
            no_rna_cells = X_rna.sum(1) == 0
            no_atac_cells = X_atac.sum(1) == 0
            no_rna_atac_cells = no_rna_cells | no_atac_cells

            ## remove cells with no expression
            X_rna = X_rna[~no_rna_atac_cells]
            X_atac = X_atac[~no_rna_atac_cells]

            #bulk_atac_abc = X_atac.mean(0, keepdim=True).detach().cpu().numpy()
            #genes_to_peaks_binary_mask *= bulk_atac_abc

            #genes_by_peaks_masks_dict[sex][celltype][condition] = genes_to_peaks_binary_mask

            ## get tfrp
            scompreg_loglikelihoods, tg_expressions, tfrps, tfrp_predictions, slopes, intercepts, std_errs, intercept_stderrs = \
                get_scompreg_loglikelihood(mean_grn_df, X_rna, X_atac, overlapping_target_genes, overlapping_tfs)


            ## save to dicts
            X_rna_dict[sex][celltype][condition if condition != '' else 'all'] = X_rna
            X_atac_dict[sex][celltype][condition if condition != '' else 'all'] = X_atac
            overlapping_target_genes_dict[sex][celltype][condition if condition != '' else 'all'] = overlapping_target_genes
            overlapping_tfs_dict[sex][celltype][condition if condition != '' else 'all'] = overlapping_tfs

            scompreg_loglikelihoods_dict[sex][celltype][condition if condition != '' else 'all'] = scompreg_loglikelihoods
            std_errs_dict[sex][celltype][condition if condition != '' else 'all'] = std_errs
            tg_expressions_dict[sex][celltype][condition if condition != '' else 'all'] = tg_expressions
            tfrps_dict[sex][celltype][condition if condition != '' else 'all'] = tfrps
            tfrp_predictions_dict[sex][celltype][condition if condition != '' else 'all'] = tfrp_predictions
            slopes_dict[sex][celltype][condition if condition != '' else 'all'] = slopes
            intercepts_dict[sex][celltype][condition if condition != '' else 'all'] = intercepts
            intercept_stderrs_dict[sex][celltype][condition if condition != '' else 'all'] = intercept_stderrs

            # tg_expressions_degs = tg_expressions.loc[:,tg_expressions.columns.isin(celltype_degs_df['gene'])]
            # tfrps_degs = tfrps.loc[:,tfrps.columns.isin(celltype_degs_df['gene'])]
            # x_degs = tg_expressions_degs.values.flatten()
            # y_degs = tfrps_degs.values.flatten()

            # tf_expressions = mdd_rna_sampled_group_seacells[:,mdd_rna_sampled_group.var['is_tf']].X.toarray()
            # x_tfs = tf_expressions.flatten()
            # y_tfs = tfrps.loc[:,mdd_rna_sampled_group.var['is_tf']].values.flatten()

            if do_corrs_or_cosines is not None:

                ## transpose
                X_rna = X_rna.T
                X_atac = X_atac.T

                if do_corrs_or_cosines == 'correlations':
                    ## concatenate
                    X_rna_atac = torch.cat([X_rna, X_atac], dim=0)

                    corr = torch.corrcoef(X_rna_atac) #torch.corrcoef(X_rna_atac.cuda(9) if cuda_available else X_rna_atac)
                    corr = corr[:X_rna.shape[0], X_rna.shape[0]:]

                    if corr.isnan().any():
                        print(f'NaN values in correlation matrix')
                        corr[corr.isnan()] = 0

                    genes_by_peaks_corrs_dict[sex][celltype][condition] = corr.detach().cpu()
                    n_dict[sex][celltype][condition] = X_rna_atac.shape[1]

                elif do_corrs_or_cosines == 'spearman':

                    X_rna_argsort = torch.argsort(X_rna, dim=1) 
                    X_atac_argsort = torch.argsort(X_atac, dim=1)

                    X_rna_atac = torch.cat([X_rna_argsort, X_atac_argsort], dim=0)
                    corr = torch.corrcoef(X_rna_atac.cuda() if cuda_available else X_rna_atac)
                    corr = corr[:X_rna.shape[0], X_rna.shape[0]:]

                    genes_by_peaks_corrs_dict[sex][celltype][condition] = corr.detach().cpu()
                    n_dict[sex][celltype][condition] = X_rna_atac.shape[1]
                        
                elif do_corrs_or_cosines == 'kendall':

                    corr = torch.zeros(X_rna.shape[0], X_atac.shape[0])
                    p_values = torch.zeros(X_rna.shape[0], X_atac.shape[0])
                    
                    for g, gene in tqdm(enumerate(mdd_rna_sampled_group.var_names), total=len(mdd_rna_sampled_group.var_names), desc="Computing Kendall correlations"):
                        x_rna = X_rna[g,:]
                        X_rna_g = x_rna.repeat(len(X_atac), 1)
                        corrs_g, p_values_g = kendall_rank_corrcoef(X_rna_g.T.cuda(), X_atac.T.cuda(), variant='b', t_test=True, alternative='less')
                        corr[g,:] = corrs_g.detach().cpu()
                        p_values[g,:] = p_values_g.detach().cpu()

                    p_values[p_values==0] = 1e-5
                    p_values[p_values==1] = 1-1e-5

                    z_scores = norm.ppf(p_values)

                    genes_by_peaks_corrs_dict[sex][celltype][condition] = corr.detach().cpu()
                    n_dict[sex][celltype][condition] = X_rna.shape[1]

                elif do_corrs_or_cosines == 'cosines':

                    ## normalize gene expression and chromatin accessibility
                    X_rna = torch.nn.functional.normalize(X_rna, p=2, dim=1)
                    X_atac = torch.nn.functional.normalize(X_atac, p=2, dim=1)

                    ## correlate gene expression with chromatin accessibility
                    cosine = torch.matmul(X_atac, X_rna.T)
                    genes_by_peaks_corrs_dict[sex][celltype][condition] = cosine

else:
    import pickle
    with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
        all_dicts = pickle.load(f)
    genes_by_peaks_corrs_dict, tfrps_dict, tfrp_predictions_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, mean_grn_df = all_dicts



'''
import pickle
all_dicts = (genes_by_peaks_corrs_dict, tfrps_dict, tfrp_predictions_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, mean_grn_df)
with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'wb') as f:
    pickle.dump(all_dicts, f)
'''

#%% get MDD-association gene scores

## pre-computed log likelihoods
scompreg_loglikelihoods_case_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['Case']
scompreg_loglikelihoods_control_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['Control']
scompreg_loglikelihoods_all_precomputed = scompreg_loglikelihoods_dict[sex][celltype]['all']

scompreg_loglikelihoods_precomputed_df = pd.DataFrame({
    'case': scompreg_loglikelihoods_case_precomputed,
    'control': scompreg_loglikelihoods_control_precomputed,
    'all': scompreg_loglikelihoods_all_precomputed
})
genes_names = scompreg_loglikelihoods_precomputed_df.index

LR_precomputed = -2 * (scompreg_loglikelihoods_precomputed_df.apply(lambda gene: gene['all'] - (gene['case'] + gene['control']), axis=1))

## re-computed log likelihoods and confirm that same as pre-computed
for condition in ['Case', 'Control']:

    tfrps               = tfrps_dict[sex][celltype][condition]
    tg_expressions      = tg_expressions_dict[sex][celltype][condition]

    for gene in tqdm(genes_names):

        log_gaussian_likelihood_pre = scompreg_loglikelihoods_dict[sex][celltype][condition][gene]
        if np.isnan(log_gaussian_likelihood_pre):
            continue

        tfrp                = tfrps.loc[:,gene].values.astype(float)
        tg_expression       = tg_expressions.loc[:,gene].values.astype(float)

        ## compute slope and intercept of linear regression - tg_expression is sparse (so is tfrp)
        slope, intercept, r_value, p_value, std_err = linregress(tg_expression, tfrp)
        tfrp_prediction = slope * tg_expression + intercept

        ## compute residuals and variance
        n = len(tfrp)
        sq_residuals = (tfrp - tfrp_prediction)**2
        var = sq_residuals.sum() / n
        log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()

        diff = np.abs(log_gaussian_likelihood - log_gaussian_likelihood_pre)
        assert diff < 1e-6

## re-compute log likelihoods for pooled conditions
tfrps_all = pd.concat([ tfrps_dict[sex][celltype]['Case'], tfrps_dict[sex][celltype]['Control'] ]).reset_index(drop=True)
tg_expressions_all = pd.concat([ tg_expressions_dict[sex][celltype]['Case'], tg_expressions_dict[sex][celltype]['Control'] ]).reset_index(drop=True)
tfrp_predictions_all = pd.concat([ tfrp_predictions_dict[sex][celltype]['Case'], tfrp_predictions_dict[sex][celltype]['Control'] ]).reset_index(drop=True)

## compute correlation between tfrp predictions and tfrps - could potential use to filter genes
tfrp_corrs = tfrp_predictions_all.corrwith(tfrps_all, method='kendall')
tfrp_corrs = tfrp_corrs.dropna()

diffs_all = []
scompreg_loglikelihoods_all = {}

for gene in tqdm(genes_names):

    tfrp = tfrps_all.loc[:,gene].values.astype(float)
    tg_expression = tg_expressions_all.loc[:,gene].values.astype(float)

    try:
        slope, intercept, r_value, p_value, std_err = linregress(tg_expression, tfrp)
        tfrp_predictions = slope * tg_expression + intercept
    except:
        print(f'{gene} has no variance in tg_expression')
        tfrp_predictions = np.ones_like(tg_expression) * np.nan
        log_gaussian_likelihood = np.array([np.nan])
    else:
        n = len(tfrp)
        sq_residuals = (tfrp - tfrp_predictions)**2
        var = sq_residuals.sum() / n
        log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()
    finally:
        scompreg_loglikelihoods_all[gene] = log_gaussian_likelihood

    #diff = np.abs(log_gaussian_likelihood - scompreg_loglikelihoods_dict[sex][celltype]['all'][gene])
    #diffs_all.append(diff)

## collect all log likelihoods
scompreg_loglikelihoods_df = pd.DataFrame({
    'case': scompreg_loglikelihoods_case_precomputed,
    'control': scompreg_loglikelihoods_control_precomputed,
    'all': scompreg_loglikelihoods_all
}).dropna() # drop rows with nan

## extract gene names from dataframe since some genes were dropped due to nan
genes_names = scompreg_loglikelihoods_df.index

## compute log likelihood ratio LR
LR = -2 * (scompreg_loglikelihoods_df.apply(lambda gene: gene['all'] - (gene['case'] + gene['control']), axis=1))

## apply Wilson-Hilferty cube-root transformation
dof = 3
Z = (9*dof/2)**(1/2) * ( (LR/dof)**(1/3) - (1 - 2/(9*dof)) )

## Hawkins & Wixley 1986
lambd = 0.2887  # optimal value for dof=3
Z = (LR**lambd - 1) / lambd

## Recenter Z distribution
Z = Z - np.median(Z)

if (LR < 0).any():
    print(f'LR is negative for {celltype} {sex}')

## Compute t-statistic between regression slopes
slopes_case = slopes_dict[sex][celltype]['Case']
slopes_control = slopes_dict[sex][celltype]['Control']

std_errs_case = std_errs_dict[sex][celltype]['Case']
std_errs_control = std_errs_dict[sex][celltype]['Control']

t_df = pd.DataFrame({
    'case': slopes_case,
    'control': slopes_control,
    'std_err_case': std_errs_case,
    'std_err_control': std_errs_control
}).dropna()

t_slopes = (t_df['case'] - t_df['control']) / np.sqrt(t_df['std_err_case']**2 + t_df['std_err_control']**2)
t_slopes = t_slopes.dropna() # might have residual nan due to std_err being 0

## compute correlation between LR and absolute value of slopes t-statistic
LR.name = 'LR'
t_slopes.name = 't_slopes'
LR_t_df = pd.merge(left=LR, right=t_slopes.abs(), left_index=True, right_index=True, how='inner')
LR_t_corr = LR_t_df.corr(method='spearman')['LR']['t_slopes']
print(f'Correlation between LR and absolute value of slopes t-statistic: {LR_t_corr:.2f}')

## separate up and down LR and Z based on sign of t-statistic
LR_up = LR[t_slopes > 0]
LR_down = LR[t_slopes < 0]

Z_up = Z[t_slopes > 0]
Z_down = Z[t_slopes < 0]



#%% filter LR values with fitted null distribution (sc-compReg)
from scipy.stats import gamma
from scipy.optimize import minimize

# 1) Suppose `lr` is your 1-D array of empirical LR statistics:
lr = np.array(LR)

# 2) Compute subset of empirical quantiles:
max_null_quantile = 0.5

probs = np.linspace(0.05, max_null_quantile, 10)
emp_q = np.quantile(lr, probs)

# 3) Define the objective G(a, b) = sum_i [Gamma.ppf(probs[i], a, scale=b) - emp_q[i]]^2
def G(params):
    a, scale = params
    # enforce positivity
    if a <= 0 or scale <= 0:
        return np.inf
    theor_q = gamma.ppf(probs, a, scale=scale)
    return np.sum((theor_q - emp_q)**2)

# 4) Method-of-moments init on the lower half to get a reasonable starting point
m1 = emp_q.mean()
v1 = emp_q.var()
init_a = m1**2 / v1
init_scale = v1 / m1

# 5) Minimize G(a, scale) over a>0, scale>0
res = minimize(G,
               x0=[init_a, init_scale],
               bounds=[(1e-8, None), (1e-8, None)],
               method='L-BFGS-B')


# Plot empirical histogram and fitted gamma distribution
plt.figure(figsize=(10, 6))
plt.hist(lr, bins=100, density=True, alpha=0.6, label='Empirical LR distribution')

# Generate points for fitted gamma distribution
x = np.linspace(0, max(lr), 1000)
fitted_pdf = gamma.pdf(x, a=res.x[0], scale=res.x[1])
plt.plot(x, fitted_pdf, 'r-', linewidth=2, label='Fitted gamma distribution')

# Add veritcal line at last null quantile
plt.axvline(emp_q[-1], color='black', linestyle='--', label=f'Last null quantile of empirical distribution (q={probs[-1]:.2f})')

# Add veritcal line at threshold
p_threshold = 0.05
lr_at_p = gamma.ppf(1-p_threshold, a=res.x[0], scale=res.x[1])
plt.axvline(lr_at_p, color='green', linestyle='--', label=f'Threshold (p={p_threshold:.2f}) @ LR={lr_at_p:.2f}')

plt.xlabel('LR statistic')
plt.ylabel('Density')
plt.title('Comparison of Empirical and Fitted LR Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

## filter LR values with fitted null distribution
lr_fitted_cdf = gamma.cdf(lr, a=res.x[0], scale=res.x[1])
where_filtered = lr_fitted_cdf > (1-p_threshold)

lr_filtered = LR[where_filtered]
genes_names_filtered = lr_filtered.index.to_list()

## filter LR_up and LR_down, note that splits gene set in half, could use more leniant threshold to include more genes for up and down, or even create separate null distributions for each
lr_up_filtered = LR_up[LR_up.index.isin(genes_names_filtered)]
lr_down_filtered = LR_down[LR_down.index.isin(genes_names_filtered)]
t_slopes_filtered = t_slopes[genes_names_filtered]

#%% Get gene sets that include DEG genes...


## ...from Maitra et al. 2023
celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id.female'].str.startswith(celltype)]

## filtered LR values ex-DEG (replace with other non-DEG genes)
lr_filtered_is_deg = lr_filtered.index.isin(celltype_degs_df.index)

if lr_filtered_is_deg.any():
    lr_filtered_woDeg = lr_filtered[~lr_filtered_is_deg]
else:
    print('No DEG genes in LR')

## LR values for DEG genes (not all DEG genes are in LR)
LR_deg = LR.loc[LR.index.isin(celltype_degs_df['gene'])]
deg_not_in_lr = celltype_degs_df[~celltype_degs_df['gene'].isin(LR.index)]['gene'].drop_duplicates().to_list()
deg_not_in_lr_series = pd.Series(np.nan, index=deg_not_in_lr)
LR_deg = pd.concat([LR_deg, deg_not_in_lr_series])

## DEG genes with top filtered LR genes
#where_filtered_idxs = np.sort(np.flip(lr_fitted_cdf.argsort())[:len(lr_filtered)]); top_lr_filtered = LR[where_filtered_idxs]; assert top_lr_filtered.equals(lr_filtered)
n_lr_deg = len(lr_filtered) + celltype_degs_df['gene'].nunique()
n_lr_minus_deg = len(lr_filtered) - celltype_degs_df['gene'].nunique(); assert n_lr_minus_deg > 0, 'list of DEG genes is longer than list of filtered LR genes'

where_filtered_with_excess = np.flip(lr_fitted_cdf.argsort())[:n_lr_deg]
where_filtered_with_excess_top = where_filtered_with_excess[:n_lr_minus_deg]
where_filtered_with_excess_bottom = where_filtered_with_excess[-n_lr_minus_deg:]

top_lr_filtered = LR[where_filtered_with_excess_top]
top_lr_filtered_wDeg = pd.concat([top_lr_filtered, LR_deg])

bottom_lr_filtered = LR[where_filtered_with_excess_bottom]
bottom_lr_filtered_wDeg = pd.concat([bottom_lr_filtered, LR_deg])

n_genes = np.unique([len(lr_filtered), len(top_lr_filtered_wDeg), len(bottom_lr_filtered_wDeg)]); assert len(n_genes) == 1, 'number of genes in each list must be the same'
n_genes = n_genes[0]

## ...from scanpy DEG analysis
'''
mdd_rna_female_celltype = mdd_rna_female[mdd_rna_female.obs[rna_celltype_key].str.startswith(celltype)]
sc.tl.rank_genes_groups(mdd_rna_female_celltype, rna_condition_key, reference='Control', method=deg_method, key_added=deg_method, pts=True)
sc_deg_df = sc.get.rank_genes_groups_df(mdd_rna_female_celltype, group='Case', key=deg_method)
sc_deg_df = sc_deg_df[sc_deg_df['pvals_adj'] < 0.05]
'''

sc_deg_df = pd.DataFrame(significant_genes, columns=['names'], index=significant_genes)

## LR values for DEG genes (not all DEG genes are in LR)
sc_LR_deg = LR.loc[LR.index.isin(sc_deg_df['names'])]
sc_deg_not_in_lr = sc_deg_df[~sc_deg_df['names'].isin(LR.index)]['names'].drop_duplicates().to_list()
sc_deg_not_in_lr_series = pd.Series(np.nan, index=sc_deg_not_in_lr)
sc_LR_deg = pd.concat([sc_LR_deg, sc_deg_not_in_lr_series])

## DEG genes with top filtered LR genes
#where_filtered_idxs = np.sort(np.flip(lr_fitted_cdf.argsort())[:len(lr_filtered)]); top_lr_filtered = LR[where_filtered_idxs]; assert top_lr_filtered.equals(lr_filtered)
sc_n_lr_deg = len(lr_filtered) + sc_deg_df['names'].nunique()
sc_n_lr_minus_deg = len(lr_filtered) - sc_deg_df['names'].nunique(); assert sc_n_lr_minus_deg > 0, 'list of DEG genes is longer than list of filtered LR genes'

where_filtered_with_excess = np.flip(lr_fitted_cdf.argsort())[:sc_n_lr_deg]
where_filtered_with_excess_top = where_filtered_with_excess[:sc_n_lr_minus_deg]
where_filtered_with_excess_bottom = where_filtered_with_excess[-sc_n_lr_minus_deg:]

sc_top_lr_filtered = LR[where_filtered_with_excess_top]
sc_bottom_lr_filtered = LR[where_filtered_with_excess_bottom]

sc_top_lr_filtered_wDeg = pd.concat([sc_top_lr_filtered, sc_LR_deg])
sc_bottom_lr_filtered_wDeg = pd.concat([sc_bottom_lr_filtered, sc_LR_deg])

sc_n_genes = np.unique([len(lr_filtered), len(top_lr_filtered_wDeg), len(bottom_lr_filtered_wDeg)]); assert len(sc_n_genes) == 1, 'number of genes in each list must be the same'
sc_n_genes = sc_n_genes[0]

## filtered LR values ex-DEG (replace with other non-DEG genes)
sc_lr_filtered_is_deg = lr_filtered.index.isin(sc_deg_df['names'])

if sc_lr_filtered_is_deg.any():
    sc_lr_filtered_woDeg_short = lr_filtered[~sc_lr_filtered_is_deg]
    sc_deg_woLR = lr_filtered[sc_lr_filtered_is_deg]
    print(f'{sc_lr_filtered_is_deg.sum()} DEG genes removed from LR')

    replace_degs_with_non_degs = where_filtered_with_excess[ len(lr_filtered) : (len(lr_filtered) + sc_lr_filtered_is_deg.sum()) ]
    sc_lr_filtered_woDeg = pd.concat([sc_lr_filtered_woDeg_short, LR[replace_degs_with_non_degs]])

else:
    print('No scanpy DEG genes in LR')


#%% Merge lr_filtered into mean_grn_df

mean_grn_df_filtered = mean_grn_df.merge(lr_filtered, left_on='TG', right_index=True, how='right')
#mean_grn_df_filtered['Correlation'].fillna(mean_grn_df_filtered['Correlation'].mean(), inplace=True)
mean_grn_df_filtered['lrCorr'] = mean_grn_df_filtered[['Correlation','LR']].product(axis=1, skipna=True).abs()

## create graph from mean_grn_df_filtered with lrCorr as edge weight
mean_grn_filtered_graph = nx.from_pandas_edgelist(
    mean_grn_df_filtered,
    source='TG',
    target='enhancer',
    edge_attr='lrCorr',
    create_using=nx.DiGraph())

peaks = scglue.genomics.Bed(mean_grn_df_filtered.assign(name=mean_grn_df_filtered['enhancer']))

gene_ad = anndata.AnnData(var=pd.DataFrame(index=mean_grn_df_filtered['TG'].to_list()))
scglue.data.get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')
genes = scglue.genomics.Bed(gene_ad.var)

#genes = scglue.genomics.Bed(mdd_rna[:, mdd_rna.var['chrom'].notna()].var.assign(name=mdd_rna[:, mdd_rna.var['chrom'].notna()].var_names))
#genes = genes[genes['name'].isin(mean_grn_df_filtered['TG'])]
tss = genes.strand_specific_start_site()

peaks.write_bed(os.path.join(os.environ['OUTPATH'], 'peaks.bed'))
scglue.genomics.write_links(
    mean_grn_filtered_graph,
    tss,
    peaks,
    os.path.join(os.environ['OUTPATH'], 'gene2peak_lrCorr.links'),
    keep_attrs=["lrCorr"]
    )

tg_dist_counts_sorted = mean_grn_df_filtered.groupby('TG')[['dist']].mean().merge(mean_grn_df_filtered['TG'].value_counts(), left_index=True, right_on='TG').sort_values('dist').head(20)
display(tg_dist_counts_sorted)

gene = 'INPP5J'
tg_grn = mean_grn_df_filtered[mean_grn_df_filtered['TG']==gene].sort_values('dist')[['enhancer','dist','lrCorr']].groupby('enhancer').mean()
tg_grn_bounds = np.stack(tg_grn.index.str.split(':|-')).flatten()
tg_grn_bounds = [int(bound) for bound in tg_grn_bounds if bound.isdigit()] + [genes.loc[gene, 'chromStart'].drop_duplicates().values[0]] + [genes.loc[gene, 'chromEnd'].drop_duplicates().values[0]]
display(tg_grn)
print(f'{genes.loc[gene, "chrom"].drop_duplicates().values[0]}:{min(tg_grn_bounds)}-{max(tg_grn_bounds)}')
print(genes.loc[gene,['chrom','chromStart','chromEnd','name']].drop_duplicates())

#!pyGenomeTracks --tracks tracks.ini --region chr2:157304654-157336585 -o tracks.png

#%% Get BrainGMT and filter for cortical genes

brain_gmt = gp.parser.read_gmt(os.path.join(os.environ['DATAPATH'], 'BrainGMTv2_HumanOrthologs.gmt'))
brain_gmt_names = pd.Series(list(brain_gmt.keys()))
brain_gmt_prefix_unique = np.unique([name.split('_')[0] for name in brain_gmt_names])

brain_gmt_wGO = gp.parser.read_gmt(os.path.join(os.environ['DATAPATH'], 'BrainGMTv2_wGO_HumanOrthologs.gmt'))
brain_gmt_wGO_names = pd.Series(list(brain_gmt_wGO.keys()))
brain_gmt_wGO_prefix_unique = np.unique([name.split('_')[0] for name in brain_gmt_wGO_names])

non_cortical_blacklist = [
    'DropViz',
    'HippoSeq',
    'Coexpression_Hippocampus',
    'Birt_Hagenauer_2020',
    'Gray_2014',
    'Bagot_2016',
    'Bagot_2017',
    'Pena_2019',
    'GeneWeaver',
    'GenWeaver', # probably same as GeneWeaver
    'Gemma'
]

#gset_by_blacklist_df = pd.DataFrame([brain_gmt_names.str.contains(ncbl) for ncbl in non_cortical_blacklist], index=non_cortical_blacklist, columns=brain_gmt_names)
gset_by_blacklist_df = pd.DataFrame([brain_gmt_names.str.contains(ncbl) for ncbl in non_cortical_blacklist])
gset_by_blacklist_df.index = non_cortical_blacklist
gset_by_blacklist_df.columns = brain_gmt_names

keep_cortical_gmt = gset_by_blacklist_df.loc[:,~gset_by_blacklist_df.any(axis=0).values].columns.tolist()
brain_gmt_cortical = {k:v for k,v in brain_gmt.items() if k in keep_cortical_gmt}

gset_by_blacklist_df_wGO = pd.DataFrame([brain_gmt_wGO_names.str.contains(ncbl) for ncbl in non_cortical_blacklist])
gset_by_blacklist_df_wGO.index = non_cortical_blacklist
gset_by_blacklist_df_wGO.columns = brain_gmt_wGO_names

keep_cortical_gmt_wGO = gset_by_blacklist_df_wGO.loc[:,~gset_by_blacklist_df_wGO.any(axis=0).values].columns.tolist()
brain_gmt_cortical_wGO = {k:v for k,v in brain_gmt_wGO.items() if k in keep_cortical_gmt_wGO}

#%% MAGMA

## get entrez IDs or Ensembl IDs for significant genes
import mygene
mg = mygene.MyGeneInfo()

## paths for MAGMA
MAGMAPATH = os.path.join(os.environ['ECLARE_ROOT'], 'magma_v1.10_mac')
magma_genes_raw_path = os.path.join(os.environ['DATAPATH'], 'FUMA_public_jobs', 'FUMA_public_job500', 'magma.genes.raw')  # https://fuma.ctglab.nl/browse#GwasList
magma_out_path = os.path.join(os.environ['OUTPATH'], 'sc-compReg_significant_genes')

## read "genes.raw" file and see if IDs start with "ENSG" or is integer
genes_raw_df = pd.read_csv(magma_genes_raw_path, sep='/t', header=None, skiprows=2)
genes_raw_ids = genes_raw_df[0].apply(lambda x: x.split(' ')[0])

gene_sets = {'sc-compReg': lr_filtered.index.to_list(), 'pyDESeq2': significant_genes.to_list()}
set_file_dict = {}

for gene_set_name, gene_set in gene_sets.items():

    if genes_raw_ids.str.startswith('ENSG').all():

        print(f"IDs are Ensembl IDs")

        significant_genes_mg_df = mg.querymany(
            gene_set,
            scopes="symbol",
            fields="ensembl.gene",
            species="human",
            size=1,
            as_dataframe=True,
            )

        significant_genes_ensembl_ids = list(set(
            significant_genes_mg_df['ensembl.gene'].dropna().to_list() + \
            significant_genes_mg_df['ensembl'].dropna().apply(lambda gene: [ensembl['gene'] for ensembl in gene]).explode().to_list()
        ))

        set_file_dict[gene_set_name] = significant_genes_ensembl_ids
        
    else:

        print(f"IDs are not Ensembl (defaulting to Entrez IDs)")

        significant_genes_entrez_ids = []
        for gene in tqdm(gene_set, total=len(gene_set), desc='Getting entrez IDs for significant genes'):
            entrez_id_res = mg.query(gene, species='human', scopes='entrezgenes', size=1)
            entrez_id = entrez_id_res['hits'][0]['_id']
            significant_genes_entrez_ids.append(entrez_id)

        set_file_dict[gene_set_name] = significant_genes_entrez_ids

## write gene set file
magma_set_file_path = os.path.join(os.environ['OUTPATH'], 'significant_gene_sets.gmt')

with open(magma_set_file_path, 'w') as f:
    for gene_set_name, gene_set in set_file_dict.items():
        f.write(f"{gene_set_name} {' '.join(gene_set)}\n")

## run MAGMA (a terminal command)
os.system(f"{MAGMAPATH}/magma --gene-results {magma_genes_raw_path} --set-annot {magma_set_file_path} --out {magma_out_path}")

## check content of output file and log file
outfile = magma_out_path + '.gsa.out'
logfile = magma_out_path + '.log'

with open(logfile, 'r') as f:
    for line in f:
        print(line)
with open(outfile, 'r') as f:
    for line in f:
        print(line)

#%% GSEApy

rank = Z.copy()

ranked_list = pd.DataFrame({'gene': rank.index.to_list(), 'score': rank.values})
ranked_list = ranked_list.sort_values(by='score', ascending=False)

pre_res = gp.prerank(rnk=ranked_list,
                     gene_sets=brain_gmt_cortical,
                     outdir=os.path.join(os.environ['OUTPATH'], 'gseapy_results'),
                     min_size=2,
                     max_size=len(ranked_list),
                     permutation_num=1000)

'''
gene sets of interest:
- GO_Biological_Process_2021/2023/2025
- Human_Phenotype_Ontology
- DisGeNET

- Reactome_2022
- KEGG_2021_Human

- ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X
- ENCODE_TF_ChIP-seq_2014/2015
- ChEA_2013/2015/2016/2022
- TRRUST_Transcription_Factors_2019
'''

pre_res.res2d.sort_values('FWER p-val', ascending=True).head(20)

## plot enrichment map
term2 = pre_res.res2d.Term
axes = pre_res.plot(terms=term2[5])

## dotplot
from gseapy import dotplot
# to save your figure, make sure that ``ofname`` is not None
ax = dotplot(pre_res.res2d,
             column="NOM p-val",
             title='',
             cmap=plt.cm.viridis,
             size=6, # adjust dot size
             top_term=20,
             figsize=(4,7), cutoff=0.25, show_ring=False)

#%% EnrichR
from IPython.display import display

def do_enrichr(lr_filtered_type, pathways, outdir=None, gene_sets=None):

    enr = gp.enrichr(lr_filtered_type.index.to_list(),
                        gene_sets=pathways,
                        outdir=None)

    display(enr.res2d.sort_values('Adjusted P-value', ascending=True).head(20)[['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Combined Score', 'Genes']])

    # dotplot
    gp.dotplot(enr.res2d,
            column='Adjusted P-value',
            figsize=(3,7),
            title='',
            cmap=plt.cm.viridis,
            size=12, # adjust dot size
            cutoff=0.25,
            top_term=15,
            show_ring=False)
    plt.show()

    enr_sig_pathways = enr.res2d[enr.res2d['Adjusted P-value'] < 0.05]['Term'].to_list()

    return enr, enr_sig_pathways

pathways = brain_gmt_cortical
enr, enr_sig_pathways = do_enrichr(lr_filtered, pathways)

enr_woDeg, enr_sig_pathways_woDeg = do_enrichr(sc_lr_filtered_woDeg, pathways)
enr_woDeg_short, enr_sig_pathways_woDeg_short = do_enrichr(sc_lr_filtered_woDeg_short, pathways)
enr_deg_woLR, enr_sig_pathways_deg_woLR = do_enrichr(sc_deg_woLR, pathways)

enr_deg, enr_sig_pathways_deg = do_enrichr(LR_deg, pathways)
enr_top_wDeg, enr_sig_pathways_top_wDeg = do_enrichr(top_lr_filtered_wDeg, pathways)
enr_bottom_wDeg, enr_sig_pathways_bottom_wDeg = do_enrichr(bottom_lr_filtered_wDeg, pathways)

sc_enr_deg, sc_enr_sig_pathways_deg = do_enrichr(sc_LR_deg, pathways)
sc_enr_top_wDeg, sc_enr_sig_pathways_top_wDeg = do_enrichr(sc_top_lr_filtered_wDeg, pathways)
sc_enr_bottom_wDeg, sc_enr_sig_pathways_bottom_wDeg = do_enrichr(sc_bottom_lr_filtered_wDeg, pathways)

# Create sets of significant pathways for each case
set1 = set(enr_sig_pathways)

set2 = set(enr_sig_pathways_deg)
set3 = set(enr_sig_pathways_top_wDeg)
set4 = set(enr_sig_pathways_bottom_wDeg)

set5 = set(sc_enr_sig_pathways_deg)
set6 = set(sc_enr_sig_pathways_top_wDeg)
set7 = set(sc_enr_sig_pathways_bottom_wDeg)

set8 = set(enr_sig_pathways_woDeg)
set9 = set(enr_sig_pathways_woDeg_short)
set10 = set(enr_sig_pathways_deg_woLR)

from venn import venn
enrs_dict = {
    'All LR': set1,
    'DEG + Top LR': set3,
    'DEG + Bottom LR': set4,
    'DEG': set2,
}
venn(enrs_dict)

sc_enrs_dict = {
    'All LR': set1,
    'DEG + Top LR': set6,
    'DEG + Bottom LR': set7,
    'DEG': set5,
}
venn(sc_enrs_dict)

sc_enrs_dict_woDeg = {
    'All LR': set1,
    'All LR - ex sc DEG': set8,
    'All LR - ex sc DEG (short)': set9,
    'DEG - ex LR': set10,
}
venn(sc_enrs_dict_woDeg)


#%% check ranks of pre-defined pathways

## rank by adjusted p-value
enr_sorted = enr.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
enr_sorted_deg = enr_deg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
enr_sorted_top_wDeg = enr_top_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
enr_sorted_bottom_wDeg = enr_bottom_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
sc_enr_sorted_deg = sc_enr_deg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
sc_enr_sorted_top_wDeg = sc_enr_top_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')
sc_enr_sorted_bottom_wDeg = sc_enr_bottom_wDeg.res2d.sort_values('Adjusted P-value', ignore_index=True).reset_index(names='rank')

## compute normalized ranks
enr_sorted['normalized_rank'] = enr_sorted['rank'] / len(enr_sorted)
enr_sorted_deg['normalized_rank'] = enr_sorted_deg['rank'] / len(enr_sorted_deg)
enr_sorted_top_wDeg['normalized_rank'] = enr_sorted_top_wDeg['rank'] / len(enr_sorted_top_wDeg)
enr_sorted_bottom_wDeg['normalized_rank'] = enr_sorted_bottom_wDeg['rank'] / len(enr_sorted_bottom_wDeg)
sc_enr_sorted_deg['normalized_rank'] = sc_enr_sorted_deg['rank'] / len(sc_enr_sorted_deg)
sc_enr_sorted_top_wDeg['normalized_rank'] = sc_enr_sorted_top_wDeg['rank'] / len(sc_enr_sorted_top_wDeg)
sc_enr_sorted_bottom_wDeg['normalized_rank'] = sc_enr_sorted_bottom_wDeg['rank'] / len(sc_enr_sorted_bottom_wDeg)

pathways = [
    'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN',
    'ASTON_MAJOR_DEPRESSIVE_DISORDER_UP',
    'LU_AGING_BRAIN_DN',
    'LU_AGING_BRAIN_UP',
    'BLALOCK_ALZHEIMERS_DISEASE_DN',
    'BLALOCK_ALZHEIMERS_DISEASE_UP',
    'Gandal_2018_BipolarDisorder_Downregulated_Cortex',
    'Gandal_2018_BipolarDisorder_Upregulated_Cortex',
    ]

rank_type = 'Adjusted P-value'
pathway_ranks = pd.DataFrame(index=pathways, \
    columns=['ALL LR', 'DEG + Top LR', 'DEG + Bottom LR', 'DEG', 'sc DEG + Top LR', 'sc DEG + Bottom LR', 'sc DEG'])
pathway_ranks.attrs['rank_type'] = rank_type


def get_pathway_rank(enr_res2d, pathway, rank_type='Adjusted P-value', return_nan=True):

    if len(enr_res2d[enr_res2d['Term'] == pathway]) > 0:
        return enr_res2d[enr_res2d['Term'] == pathway][rank_type].values[0]
    else:
        if return_nan:
            return np.nan
        else:
            return max(enr_res2d[rank_type])

for pathway in pathways:
    pathway_ranks.loc[pathway, 'ALL LR'] = get_pathway_rank(enr_sorted, pathway, rank_type)

    pathway_ranks.loc[pathway, 'DEG + Top LR'] = get_pathway_rank(enr_sorted_top_wDeg, pathway, rank_type)
    pathway_ranks.loc[pathway, 'DEG + Bottom LR'] = get_pathway_rank(enr_sorted_bottom_wDeg, pathway, rank_type)
    pathway_ranks.loc[pathway, 'DEG'] = get_pathway_rank(enr_sorted_deg, pathway, rank_type)

    pathway_ranks.loc[pathway, 'sc DEG + Top LR'] = get_pathway_rank(sc_enr_sorted_top_wDeg, pathway, rank_type)
    pathway_ranks.loc[pathway, 'sc DEG + Bottom LR'] = get_pathway_rank(sc_enr_sorted_bottom_wDeg, pathway, rank_type)
    pathway_ranks.loc[pathway, 'sc DEG'] = get_pathway_rank(sc_enr_sorted_deg, pathway, rank_type)

#%% plot ranks across methods and pathways

def plot_pathway_ranks(pathway_ranks, stem=True):

    # ─── 3. Sort pathways by “ALL LR” (optional) ───
    order = pathway_ranks["ALL LR"].sort_values().index.tolist()

    pathway_ranks = pathway_ranks.loc[order]

    pathway_ranks_long = (
        pathway_ranks
        .reset_index()
        .melt(id_vars="index", var_name="method", value_name="rank")
        .rename(columns={"index": "pathway"})
    )

    # ─── 4. Plot the stripplot ───
    fig, ax = plt.subplots(1, 2, figsize=(12, 4 + 0.5 * pathway_ranks.shape[0]), sharex=False, sharey=True)
    sns.stripplot(
        data=pathway_ranks_long,
        y="pathway",
        x="rank",
        hue="method",
        dodge=True,       # side‐by‐side dots per pathway
        jitter=False,     # no vertical jitter
        size=6,
        palette="tab10",
        ax=ax[0]
    )
    sns.stripplot(
        data=pathway_ranks_long,
        y="pathway",
        x="rank",
        hue="method",
        dodge=True,       # side‐by‐side dots per pathway
        jitter=False,     # no vertical jitter
        size=6,
        palette="tab10",
        legend=False,
        ax=ax[1]
    ); ax[1].set_xscale('log')

    if stem:

        # ─── Add horizontal stems from the rightmost edge to each marker ───
        for subax in ax:
            # 1) Record the current x‐axis limits:
            x_min, x_max = subax.get_xlim()

            # 2) Loop over each PathCollection in this subplot.
            #    Each collection corresponds to one hue level (one "method").
            for collec in subax.collections:
                # get_offsets() is an Nx2 array of (x, y) positions for every dot in that hue.
                offsets = collec.get_offsets()
                # Choose a stem color (e.g. light gray); you could also pull collec.get_facecolor()[0]
                stem_color = "lightgray"
                for (x_pt, y_pt) in offsets:
                    # draw a line from x_max → x_pt at y=y_pt
                    subax.plot(
                        [x_max, x_pt],        # x data: from right edge to the point
                        [y_pt, y_pt],         # y data: horizontal at the same y
                        color=stem_color,
                        linewidth=0.6,
                        alpha=0.7,
                        zorder=0  # put stems “underneath” the markers
                    )

            # 3) Re‐enforce the original x‐limits so the stems don’t stretch the view
            subax.set_xlim(x_min, x_max)

    # ─── 5. Add horizontal dividing lines ───
    n_paths = len(pathways)
    for i in range(n_paths - 1):
        ypos = i + 0.5
        ax[0].axhline(y=ypos, color="lightgray", linewidth=2, linestyle="--")
        ax[1].axhline(y=ypos, color="lightgray", linewidth=2, linestyle="--")

    # ─── 6. Tidy up labels & legend ───
    ax[0].set_ylabel("Brain GMT Pathway")
    ax[0].set_xlabel(f"{pathway_ranks.attrs['rank_type']}")
    ax[1].set_xlabel(f"{pathway_ranks.attrs['rank_type']} (log scale)")
    ax[0].set_title("Per‐Pathway Ranks Across Methods")

    # Place legend outside so it doesn’t overlap
    #ax[0].legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()

plot_pathway_ranks(pathway_ranks, stem=True)

#%% isolated analysis for ASTON MDD pathways

mdd_rna_female_celltype = mdd_rna_female[mdd_rna_female.obs[rna_celltype_key] == celltype]
sc.tl.rank_genes_groups(mdd_rna_female_celltype, groupby=rna_condition_key, reference='Control', method='wilcoxon')
deg_celltype_df = sc.get.rank_genes_groups_df(mdd_rna_female_celltype, group='Case', key=deg_method)

ASTON_MAJOR_DEPRESSIVE_DISORDER_DN_genes_df = enr.res2d[enr.res2d['Term'] == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN']
ASTON_MAJOR_DEPRESSIVE_DISORDER_DN_genes = ASTON_MAJOR_DEPRESSIVE_DISORDER_DN_genes_df['Genes'].values[0].split(';') # gene set based on temporal cortex
deg_celltype_df_mdd_dn = deg_celltype_df.set_index('names').loc[ASTON_MAJOR_DEPRESSIVE_DISORDER_DN_genes]

ASTON_MAJOR_DEPRESSIVE_DISORDER_UP_genes_df = enr.res2d[enr.res2d['Term'] == 'ASTON_MAJOR_DEPRESSIVE_DISORDER_UP']
ASTON_MAJOR_DEPRESSIVE_DISORDER_UP_genes = ASTON_MAJOR_DEPRESSIVE_DISORDER_UP_genes_df['Genes'].values[0].split(';') # gene set based on temporal cortex
deg_celltype_df_mdd_up = deg_celltype_df.set_index('names').loc[ASTON_MAJOR_DEPRESSIVE_DISORDER_UP_genes]

ASTON_MAJOR_DEPRESSIVE_DISORDER_df = pd.concat([deg_celltype_df_mdd_dn, deg_celltype_df_mdd_up])


#%% visualize overlap between pathways
import networkx as nx

nodes, edges = gp.enrichment_map(enr.res2d, top_term=1000)

# build graph
G = nx.from_pandas_edgelist(edges,
                            source='src_idx',
                            target='targ_idx',
                            edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes'])

# Subset graph to contain specific pathway
pathway_name = 'ASTON_MAJOR_DEPRESSIVE_DISORDER_DN'
#pathway_name = 'GOBP_POSITIVE_REGULATION_OF_MOLECULAR_FUNCTION'
pathway_idx = nodes[nodes['Term'] == pathway_name].index[0]

# Get edges where target gene is in the pathway
edges_to_keep = [
    (u, v)
    for u, v in G.edges()
    if pathway_idx in set([v, u])
]

# Create subgraph with only edges to pathway genes
G = G.edge_subgraph(edges_to_keep).copy()


# Add missing node if there is any
for node in nodes.index:
    if node not in G.nodes():
        G.add_node(node)

top_pathway_edges = edges[(edges['targ_idx'] == pathway_idx) | (edges['src_idx'] == pathway_idx)].sort_values('jaccard_coef', ascending=False).head(10)
top_pathway_edges['n_genes_overlap'] = top_pathway_edges['overlap_genes'].str.split(',').apply(len)
display(top_pathway_edges)

'''
fig, ax = plt.subplots(figsize=(24, 18))

# init node cooridnates
pos=nx.layout.circular_layout(G)
#node_size = nx.get_node_attributes()
# draw node
nx.draw_networkx_nodes(G,
                       pos=pos,
                       cmap=plt.cm.RdYlBu)
                       #node_color=list(nodes.NES),
                       #node_size=list(nodes.Hits_ratio *1000))
# draw node label
nx.draw_networkx_labels(G,
                        pos=pos,
                        labels=nodes.Term.to_dict())
# draw edge
edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
nx.draw_networkx_edges(G,
                       pos=pos,
                       width=list(map(lambda x: x*10, edge_weight)),
                       edge_color='#CDDBD4')

#plt.xlim(-1.6, 1.6)
#plt.ylim(-0.8,0.6)
plt.axis('off')
plt.show()
'''

#%% obtain LR for TF-peak-TG combinations
from eclare.data_utils import get_scompreg_loglikelihood_full

X_rna_control = X_rna_dict[sex][celltype]['Control']
X_atac_control = X_atac_dict[sex][celltype]['Control']

X_rna_case = X_rna_dict[sex][celltype]['Case']
X_atac_case = X_atac_dict[sex][celltype]['Case']

X_rna_all = torch.cat([X_rna_control, X_rna_case], dim=0)
X_atac_all = torch.cat([X_atac_control, X_atac_case], dim=0)

'''
overlapping_target_genes_control = overlapping_target_genes_dict[sex][celltype]['Control']
overlapping_target_genes_case = overlapping_target_genes_dict[sex][celltype]['Case']
overlapping_tfs_case = overlapping_tfs_dict[sex][celltype]['Case']
overlapping_tfs_control = overlapping_tfs_dict[sex][celltype]['Control']

if np.all(overlapping_tfs_case == overlapping_tfs_control) & np.all(overlapping_target_genes_case == overlapping_target_genes_control):
    overlapping_target_genes = overlapping_target_genes_case
    overlapping_tfs = overlapping_tfs_case
else:
    raise ValueError('overlapping_tfs_case and overlapping_tfs_control are not the same')
'''

overlapping_target_genes = mean_grn_df_filtered['TG'].unique()
overlapping_tfs = mean_grn_df_filtered['TF'].unique()

mean_grn_df_filtered, tfrps_control, tg_expressions_control = get_scompreg_loglikelihood_full(mean_grn_df_filtered, X_rna_control, X_atac_control, overlapping_target_genes, overlapping_tfs, 'll_control')
mean_grn_df_filtered, tfrps_case, tg_expressions_case = get_scompreg_loglikelihood_full(mean_grn_df_filtered, X_rna_case, X_atac_case, overlapping_target_genes, overlapping_tfs, 'll_case')
#mean_grn_df, tfrps_all, tg_expressions_all = get_scompreg_loglikelihood_full(mean_grn_df, X_rna_all, X_atac_all, overlapping_target_genes, overlapping_tfs, 'll_all')

assert list(tfrps_control.keys()) == list(tfrps_case.keys())
assert list(tg_expressions_control.keys()) == list(tg_expressions_case.keys())

def scompreg_likelihood_ratio(tg_expression, tfrp):

    try:
        linregress_res = linregress(tg_expression, tfrp) # if unpack directly, returns only 5 of the 6 outputs..
    except:
        log_gaussian_likelihood = np.nan
    else:
        slope, intercept, r_value, p_value, std_err, intercept_stderr = (linregress_res.slope, linregress_res.intercept, linregress_res.rvalue, linregress_res.pvalue, linregress_res.stderr, linregress_res.intercept_stderr)
        tfrp_predictions = slope * tg_expression + intercept

        ## compute residuals and variance
        n = len(tfrp)
        sq_residuals = (tfrp - tfrp_predictions)**2
        var = sq_residuals.sum() / n
        log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()

    return log_gaussian_likelihood


for gene in tqdm(overlapping_target_genes, total=len(overlapping_target_genes), desc='Null LR'):

    tfrps_control_gene = tfrps_control[gene]
    tfrps_case_gene = tfrps_case[gene]
    tg_expressions_control_gene = tg_expressions_control[gene]
    tg_expressions_case_gene = tg_expressions_case[gene]
    
    tfrps_all_gene = pd.concat([tfrps_control_gene, tfrps_case_gene], axis=1)
    tg_expressions_all_gene = torch.cat([tg_expressions_control_gene, tg_expressions_case_gene], dim=0).numpy()

    scompreg_likelihood_ratio_lambda = lambda tfrp: scompreg_likelihood_ratio(tg_expressions_all_gene, tfrp)
    tfrps_all_gene['scompreg_likelihood_ratio'] = tfrps_all_gene.apply(scompreg_likelihood_ratio_lambda, axis=1)

    mean_grn_df_filtered.loc[tfrps_all_gene.index, 'll_all'] = tfrps_all_gene['scompreg_likelihood_ratio']
    

LR_grns = -2 * (mean_grn_df_filtered.apply(lambda grn: grn['ll_all'] - (grn['ll_case'] + grn['ll_control']), axis=1))
LR_grns_filtered = LR_grns[LR_grns > LR_grns.quantile(0.95)]

fig, ax = plt.subplots(figsize=(10, 5))
LR_grns.hist(bins=100, ax=ax)

ax.axvline(x=lr_at_p, color='black', linestyle='--')
ax.axvline(x=LR_grns.quantile(0.95), color='green', linestyle='--')

ax.set_xlabel('LR')
ax.set_ylabel('Frequency')
ax.set_title('LR distribution')
plt.show()


mean_grn_df_filtered['LR_grns'] = LR_grns
mean_grn_df_filtered_pruned = mean_grn_df_filtered[mean_grn_df_filtered['LR_grns'] > LR_grns.quantile(0.95)]



#%% TF-enhancer-TG-pathway visualization with networkx and dash-cytoscape

pathway_names = ['ASTON_MAJOR_DEPRESSIVE_DISORDER_DN', 'LU_AGING_BRAIN_UP']
pathways = enr.res2d[enr.res2d['Term'].isin(pathway_names)]
pathways_edge = top_pathway_edges.loc[top_pathway_edges[['src_name','targ_name']].apply(lambda x: np.isin(x, pathway_names).all(), axis=1)]

keep_tg = pathways.set_index('Term').Genes.str.split(';')
keep_tg_stacked = np.hstack(keep_tg.values)
keep_grn = mean_grn_df_filtered_pruned[mean_grn_df_filtered_pruned['TG'].isin(keep_tg_stacked)]
keep_peaks = keep_grn['enhancer'].unique()
keep_tf = keep_grn['TF'].unique()
keep_all = np.concatenate([pathway_names, keep_tg_stacked, keep_peaks, keep_tf]) # essentially, nodes of the graph

mapping = {v: f'{v.split(":")[0]}-{vi}' for vi, v in enumerate(keep_peaks)}
keep_peaks = [mapping[peak] for peak in keep_peaks]

'''
for tg in keep_tg:
    mean_grn_filtered_graph.add_edge(tg, pathway_name, interaction='in_pathway', weight=pathway['Combined Score'].values[0])

for tf, enhancer, motif_score_norm in mean_grn_df_filtered_pruned[['TF','enhancer','motif_score_norm']].drop_duplicates().itertuples(index=False):
    mean_grn_filtered_graph.add_edge(tf, enhancer, interaction='binds', weight=motif_score_norm)
'''
G = nx.DiGraph()
#G.add_edge(pathways_edge['src_name'].values[0], pathways_edge['targ_name'].values[0], interaction='pathway_overlap', weight=pathways_edge['jaccard_coef'].values[0])

for pathway_name, keep_tg_pathway in keep_tg.items():

    grn_pathway = mean_grn_df_filtered_pruned[mean_grn_df_filtered_pruned['TG'].isin(keep_tg_pathway)]

    for tf, enhancer, tg in grn_pathway[['TF','enhancer','TG']].drop_duplicates().itertuples(index=False):

        G.add_edge(tf, enhancer, interaction='binds', weight=1)
        G.add_edge(enhancer, tg, interaction='activates', weight=1)
        G.add_edge(tg, pathway_name, interaction='in_pathway', weight=1)

        G.add_node(tf, type='tf', layer=0)
        G.add_node(enhancer, type='enhancer', layer=1)
        G.add_node(tg, type='tg', layer=2)
        G.add_node(pathway_name, type='pathway', layer=3)

nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
nlist = nodes_df.groupby('type').apply(lambda x: x.index.to_list())
nlist = nlist.reindex(['pathway', 'tg', 'enhancer', 'tf'])
nlist_values = list(nlist.values)

pos = nx.shell_layout(G, nlist_values, scale=5000)

# Apply type-specific rotations to node positions
def rotate_nodes(pos, nlist, angle_dict={'pathway': 0, 'tg': np.pi/8, 'enhancer': -np.pi/8, 'tf': -np.pi/2}):
    for node_type, nodes in nlist.items():
        # Get the center point of nodes for this type
        type_pos = np.array([pos[node] for node in nodes])
        center = type_pos.mean(axis=0)
        
        # Calculate rotation angle based on node type
        angle = angle_dict[node_type]
        
        # Apply rotation to each node position
        for node in nodes:
            # Get current position relative to center
            rel_pos = np.array(pos[node]) - center
            
            # Create rotation matrix
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply rotation
            rotated_pos = np.dot(rot_matrix, rel_pos)
            
            # Update position
            pos[node] = tuple(rotated_pos + center)

    return pos
pos = rotate_nodes(pos, nlist)

pos = nx.arf_layout(G, pos, max_iter=10000, a=1.1, dt=0.001)
pos_xy = {k: {'x': xy[0], 'y': xy[1]} for k, xy in pos.items()}

plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=False, node_size=20, alpha=0.5, node_color="blue")

cyto_data = nx.cytoscape_data(G)
elements = cyto_data['elements']

def datashader_bundling(edges, pos_xy_df):
    import datashader as ds
    import datashader.transfer_functions as tf_ds
    from datashader.bundling import hammer_bundle

    #nodes = pd.DataFrame.from_dict(dict(G.nodes), orient='index').reset_index()
    #nodes = nodes.drop(columns=['type', 'layer'])

    edges = pd.DataFrame.from_dict(dict(G.edges), orient='index').reset_index()
    edges.columns = ['source', 'target', 'interaction', 'weight']
    edges = edges.drop(columns=['interaction','weight'])

    pos_xy_df = pd.DataFrame.from_dict(pos_xy, orient='index')
    pos_xy_df['type'] = pd.Categorical(nodes_df['type'])

    cvsopts = dict(plot_height=400, plot_width=400)

    def nodesplot(nodes, name=None, canvas=None, cat=None):
        canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
        aggregator=None if cat is None else ds.count_cat(cat)
        agg=canvas.points(nodes,'x','y',aggregator)
        return tf_ds.spread(tf_ds.shade(agg, cmap=["#FF3333"]), px=3, name=name)

    def edgesplot(edges, name=None, canvas=None):
        canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
        return tf_ds.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)

    def graphplot(nodes, edges, name="", canvas=None, cat=None):
        if canvas is None:
            xr = nodes.x.min(), nodes.x.max()
            yr = nodes.y.min(), nodes.y.max()
            canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)

        np = nodesplot(nodes, name + " nodes", canvas, cat)
        ep = edgesplot(edges, name + " edges", canvas)
        return tf_ds.stack(ep, np, how="over", name=name)

    grid =  [graphplot(pos_xy_df,
                    hammer_bundle(*(pos_xy_df,edges), iterations=5, decay=decay, initial_bandwidth=bw),
                                    "d={:0.2f}, bw={:0.2f}".format(decay, bw), cat='type')
        for decay in [0.1, 0.25, 0.5, 0.9] for bw    in [0.1, 0.2, 0.5, 1]]

    tf_ds.Images(*grid).cols(4)

def holoviews_bundling(edges, pos_xy_df):
    
    import holoviews as hv
    from holoviews import opts
    from holoviews.operation.datashader import bundle_graph

    hv.extension("bokeh")

    kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
    opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

    grn_graph = hv.Graph.from_networkx(G, pos)
    grn_graph.opts(node_size=hv.dim('type').categorize({'tf': 8, 'enhancer': 10, 'tg': 16, 'pathway': 24}), 
                  node_color='type', 
                  cmap='Set1',
                  show_legend=True)

    bundled = bundle_graph(grn_graph, decay=0.5, initial_bandwidth=0.5) # (d=0.9, bw=0.2) for shell+arf, (d=0.25, bw=0.1) for shell only

    edges_only = bundled.opts(node_alpha=0, edge_color="gray", edge_line_width=1)

    nodes_only = hv.Nodes(bundled.nodes).opts(
        color='type', cmap='Set1',
        size=hv.dim('type').categorize({'tf': 8, 'enhancer': 10, 'tg': 16, 'pathway': 24}),
        legend_labels={'tf':'Transcription Factor', 'enhancer':'Enhancer', 'tg':'Target Gene', 'pathway':'Pathway'},
        width=800, height=800)
    
    network_plot = edges_only * nodes_only
    network_plot

    '''
    shaded = (datashade(bundled, normalization='linear', width=800, height=800) * bundled.nodes).opts(opts.Nodes(color='type', size=10, width=1000, cmap='Set1',legend_position='right'))
    all_nodes = hv.Nodes(bundled.nodes)
    highlight = all_nodes.select(index=['SYNJ2']).opts(fill_color='white', line_color='black', size=15)

    shaded * highlight

    highlight_gene = 'DICER1'
    incident_edges_hv = bundled.select(index=[highlight_gene], selection_mode="edges").opts(edge_color="orange", node_color='type', node_cmap='Set1', node_alpha=0.9)
    incident_node_hv = hv.Nodes(bundled.nodes).select(index=[highlight_gene]).opts(color='type', cmap='Set1', size=15)
    shaded_full = (datashade(bundled, normalization='linear', width=800, height=800) * bundled.nodes).opts(opts.Nodes(size=1, alpha=1., width=1000,legend_position='right'))
    (shaded_full * incident_edges_hv)
    '''




# collect only edges where source or target is in `keep`
edges_to_keep = [
    (u, v)
    for u, v in mean_grn_filtered_graph.edges()
    if (u in keep_all) and (v in keep_all)
]

subgraph = mean_grn_filtered_graph.edge_subgraph(edges_to_keep)
subgraph_renamed = nx.relabel_nodes(subgraph, mapping, copy=True)

degree_dict = dict(subgraph_renamed.to_undirected().degree())

import dash
from dash import html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output

cyto.load_extra_layouts()

# ─── (1) Convert NX → Cytoscape elements (no positions) ───
def nx_to_cytoscape_elements(G, keep_tf, keep_peaks):
    elements = []
    for node, _ in G.nodes(data=True):
        if node in keep_tf:
            ntype = "tf"
            layer = 0
        elif node in keep_peaks:
            ntype = "enhancer"
            layer = 1
        elif node in keep_tg:
            ntype = "tg"
            layer = 2
        elif node == pathway_name:
            ntype = "pathway"
            layer = 3
        elements.append({
            "data": {
                "id": str(node),
                "label": str(node),
                "type": ntype,
                "layer": layer,
            }
        })
    for u, v, edge_data in G.edges(data=True):
        interaction = edge_data.get("interaction", "")
        elements.append({
            "data": {
                "source": str(u),
                "target": str(v),
                "interaction": interaction
            }
        })
    return elements

elements = nx_to_cytoscape_elements(
    G=subgraph_renamed,
    keep_tf=set(keep_tf),
    keep_peaks=set(keep_peaks)
)

nodes = []
edges = []
for elem in elements:
    d = elem.get("data", {})
    # If “data” has both “source” and “target”, it’s an edge; otherwise it’s a node.
    if "source" in d and "target" in d:
        edges.append(elem)
    else:
        nodes.append(elem)

cytoscape_ready = {
    "elements": {
        "nodes": nodes,
        "edges": edges
    }
}

# Save elements to a .cyjs file for Cytoscape Desktop import
import json
cyjs_path = os.path.join(os.environ['OUTPATH'], 'mdd_subgraph.cyjs')
with open(cyjs_path, 'w') as f:
    json.dump(cytoscape_ready, f, indent=2)
print(f"Saved Cytoscape elements to {cyjs_path}")

import math

# Set R_max = how big an outer circle you want (in pixels)
R_max = 500.0

# Precompute a mapping from node -> (x,y) position
positions = {}

node_layers = [element['data'].get('layer', np.nan) for element in elements]
node_layers_unique = np.unique(node_layers)[:-1]
layers_radii = {layer: R_max * ((max(node_layers_unique) - layer) / max(node_layers_unique)) for layer in node_layers_unique}

for layer, radius in layers_radii.items():
    nodes_in_layer = [element['data'].get('id', None) for element in elements if element['data'].get('layer', None) == layer]
    N = len(nodes_in_layer)
    for i, node in enumerate(nodes_in_layer):
        θ = 2 * math.pi * (i / float(N))
        x = radius * math.cos(θ)
        y = radius * math.sin(θ)
        positions[node] = {"x": x, "y": y}

positions_df = pd.DataFrame(positions).T.dropna()
sns.scatterplot(data=positions_df, x='x', y='y')

for element in elements:
    if 'id' in element['data']:
        element['data']['position'] = positions[element['data']['id']]

import json
subgraph_renamed_cy = nx.cytoscape_data(subgraph_renamed)
with open(os.path.join(os.environ['OUTPATH'], 'mdd_subgraph.cy'), 'w') as f:
    json.dump(elements, f)


# ─── (2) Stylesheet ───
stylesheet = [
    {
        "selector": 'node[type = "tf"]',
        "style": {
            "shape": "diamond",
            "background-color": "#1f77b4",
            "label": "data(label)",
            "width": 50,
            "height": 50,
            "font-size": 8,
            "color": "#ffffff"
        }
    },
    {
        "selector": 'node[type = "enhancer"]',
        "style": {
            "shape": "triangle",
            "background-color": "#2ca02c",
            "label": "data(label)",
            "width": 40,
            "height": 40,
            "font-size": 8,
            "color": "#ffffff"
        }
    },
    {
        "selector": 'node[type = "tg"]',
        "style": {
            "shape": "ellipse",
            "background-color": "#d62728",
            "label": "data(label)",
            "width": 45,
            "height": 45,
            "font-size": 8,
            "color": "#ffffff"
        }
    },
    {
        "selector": 'node[type = "pathway"]',
        "style": {
            "shape": "rectangle",
            "background-color": "#9467bd",
            "label": "data(label)",
            "width": 45,
            "height": 45,
            "font-size": 8,
            "color": "#ffffff"
        }
    },
    {
        "selector": 'edge[interaction = "binds"]',
        "style": {
            "line-color": "#9467bd",
            "target-arrow-color": "#9467bd",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "width": 2
        }
    },
    {
        "selector": 'edge[interaction = "activates"]',
        "style": {
            "line-color": "#d62728",
            "target-arrow-color": "#d62728",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "width": 2
        }
    },
    {
        "selector": "node:hover",
        "style": {
            "border-color": "#000000",
            "border-width": 3
        }
    }
]


# ─── (3) Choose a dagre layout for left→right flow ───
layout = {
    "name": "preset",
    "positions": pos_xy
}

# ─── (4) Build the Dash app ───

app = dash.Dash(__name__)
app.layout = html.Div([
    cyto.Cytoscape(
        id="grn-network",
        elements=elements,
        layout=layout,
        stylesheet=stylesheet,
        style={"width": "100%", "height": "700px"},
    ),
    html.Div(
        id="node-data-display",
        style={
            "marginTop": "20px",
            "padding": "10px",
            "border": "1px solid #ccc"
        },
        children="Click on a node to see its type"
    )
])

# ─── (5) Callback to show node info on click ───
@app.callback(
    Output("node-data-display", "children"),
    Input("grn-network", "tapNodeData")
)
def display_node_info(node_data):
    if not node_data:
        return "Click on a node to see its type"
    return f"Node ID: {node_data['id']}  •  Type: {node_data['type']}"

if __name__ == "__main__":
    app.run(debug=True)


#%% define function for computing corrected Pearson correlation coefficient based on BigSur paper

def compute_corrected_pearson_correlation(X, Y, c=0.5):

    ## will treat both genes and peaks as features of the same type
    Z = torch.cat([X, Y], dim=1)
    n = Z.shape[0]
    kx = X.shape[1]
    ky = Y.shape[1]

    mus_gene_cells = (Z.sum(0, keepdim=True) * Z.sum(1, keepdim=True)) / Z.sum()
    fano_sampling_n_fluctuation = 1 + c**2 * mus_gene_cells

    corrected_pearson_residual = (Z - mus_gene_cells) / np.sqrt(mus_gene_cells * fano_sampling_n_fluctuation)
    corrected_fano = (corrected_pearson_residual**2).mean(0)

    corrected_pcc_scale = 1 / ( (n-1) * np.sqrt(corrected_fano.unsqueeze(0) * corrected_fano.unsqueeze(1)) )
    corrected_pcc_effect = (corrected_pearson_residual.unsqueeze(1) * corrected_pearson_residual.unsqueeze(2)).sum(0)
    corrected_pcc = corrected_pcc_scale * corrected_pcc_effect
    corrected_pcc_clamped = torch.clamp(corrected_pcc, min=-1, max=1)

    corrected_pcc_X_by_Y = corrected_pcc[:kx, kx:]

    return corrected_pcc_X_by_Y

def c_sweep(X, Y, c_range=[0.1, 0.5, 2, 10]):

    #Z = torch.cat([X, Y], dim=1)
    Z = Y.clone()

    mus_gene_cells = (Z.sum(0, keepdim=True) * Z.sum(1, keepdim=True)) / Z.sum()
    mus_genes = mus_gene_cells.mean(0) #Z.mean(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in c_range:

        fano_sampling_n_fluctuation = 1 + c**2 * mus_gene_cells
        corrected_pearson_residual = (Z - mus_gene_cells) / np.sqrt(mus_gene_cells * fano_sampling_n_fluctuation)
        corrected_fano = (corrected_pearson_residual**2).mean(0)

        ax.scatter(mus_genes, corrected_fano, label=f'c={c}', marker='.')

    ax.legend()
    plt.show()

#%%

#gene_candidates_dict = tree()
#gene_candidates_dict['Female']['Mic'] = ['ROBO2', 'SLIT3', 'ADAMSTL1', 'THSD4', 'SPP1', 'SOCS3', 'GAS6', 'MERTK']
#gene_candidates_dict['Female']['InN'] = gene_candidates_dict['Female']['Mic']
#hits_idxs_dict = tree()
#from scipy.stats import norm

score_type = 'sc-compReg'

for sex in unique_sexes:
    for celltype in unique_celltypes:

        #celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id'].str.startswith(celltype)]
        celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id.female'].str.startswith(celltype)]

        #genes_names = mdd_rna.var_names[mdd_rna.var_names.isin(celltype_degs_df['gene'])]
        genes_names = genes_peaks_dict['genes']

        corr_case = genes_by_peaks_corrs_dict[sex][celltype]['Case']
        corr_control = genes_by_peaks_corrs_dict[sex][celltype]['Control']

        corr_case = corr_case.numpy()
        corr_control = corr_control.numpy()

        genes_by_peaks_masks_case = genes_by_peaks_masks_dict[sex][celltype]['Case']
        genes_by_peaks_masks_control = genes_by_peaks_masks_dict[sex][celltype]['Control']

        corr_case[genes_by_peaks_masks_case==0] = 0
        corr_control[genes_by_peaks_masks_control==0] = 0
        
        if score_type == 'corrdiff':

            corrdiff = corr_case - corr_control
            corrdiff_genes = (corrdiff**2).sum(axis=1)
            chi2 = corrdiff_genes / np.sqrt(len(corrdiff_genes))

        elif score_type == 'logfc':

            logcorr_case = np.log1p(corr_case)
            logcorr_control = np.log1p(corr_control)

            logfc = logcorr_case - logcorr_control
            logfc_genes = (logfc**2).sum(axis=1)
            chi2 = logfc_genes / np.sqrt(len(logfc_genes))

        elif score_type == 'fisher':

            fisher_case = np.arctanh(corr_case)
            fisher_control = np.arctanh(corr_control)

            fisher_sums_case = np.sum(fisher_case, axis=1)
            fisher_sums_control = np.sum(fisher_control, axis=1)

            K_case = genes_by_peaks_masks_case.sum(1)
            K_control = genes_by_peaks_masks_control.sum(1)
            
            n_case = n_dict[sex][celltype]['Case']
            n_control = n_dict[sex][celltype]['Control']

            fisher_sums_var_case = K_case / (n_case - 3)
            fisher_sums_var_control = K_control / (n_control - 3)

            Z = (fisher_sums_case - fisher_sums_control) / np.sqrt(fisher_sums_var_case + fisher_sums_var_control)
            chi2 = Z**2

            p_value = 2 * (1 - norm.cdf(abs(Z)))
            p_value_threshold = 0.05 / K_case
            hits_idxs = np.where(p_value < p_value_threshold)[0]

            hits_idxs_dict[sex][celltype] = hits_idxs

        elif score_type == 'kendall':

            corr_case = np.arctanh(corr_case)
            corr_control = np.arctanh(corr_control)

            ## get weights from distance mask
            weights_case = genes_by_peaks_masks_case.copy()
            weights_control = genes_by_peaks_masks_control.copy()
            assert (weights_case==weights_control).all()

            #weights_case /= weights_case.sum(0)[None,:]
            #weights_control /= weights_control.sum(0)[None,:]

            weights_case[np.isnan(weights_case)] = 0
            weights_control[np.isnan(weights_control)] = 0

            ## corrs linearly correlated with z-scores
            kendall_sums_case = corr_case.copy()
            kendall_sums_control = corr_control.copy()

            K_case = (weights_case**2).sum(1)
            K_control = (weights_control**2).sum(1)

            diffs = kendall_sums_case - kendall_sums_control
            diffs_weights = diffs * weights_case
            Z = np.array([max(diffs_weights[i], key=lambda x: x**2) for i in range(len(diffs_weights))])
            #Z = (np.power( diffs * weights_case, 2)).sum(1) / np.sqrt(K_case + K_control)
            Z[np.isnan(Z)] = 0

            chi2 = Z**2

        elif score_type == 'sc-compReg':



        elif score_type == 'ismb_cosine':

            cosine_case = genes_by_peaks_corrs_dict[sex][celltype]['Case']
            cosine_control = genes_by_peaks_corrs_dict[sex][celltype]['Control']

            cosine_valid_comparison = (cosine_case!=0) * (cosine_control!=0)
            cosine_case = cosine_case[cosine_valid_comparison]
            cosine_control = cosine_control[cosine_valid_comparison]

            log1p_fold_change = torch.log1p(cosine_case) - torch.log1p(cosine_control)

            where_valid = np.where(cosine_valid_comparison)
            genes_where_valid_idxs = where_valid[1]
            logfc_sum_per_gene = np.bincount(genes_where_valid_idxs, weights=log1p_fold_change.abs())
            logfc_mean_per_gene = logfc_sum_per_gene / np.bincount(genes_where_valid_idxs)

            logfc_sum_per_gene_ranked_idxs = np.argsort(logfc_sum_per_gene)[::-1]
            logfc_sum_per_gene_ranked_gene_names = mdd_rna.var_names[genes_indices][logfc_sum_per_gene_ranked_idxs]
            logfc_sum_per_gene_matches = np.concatenate([logfc_sum_per_gene[genes_names == gene] for gene in gene_candidates_dict['Female']['Mic'] if logfc_sum_per_gene[genes_names == gene].size > 0])
            #logfc_sum_per_gene_matches = np.concatenate([logfc_sum_per_gene[mdd_rna.var_names == gene] for gene in gene_candidates_dict['Female']['Mic'] if logfc_sum_per_gene[mdd_rna.var_names == gene].size > 0])

            from scipy import stats
            percentiles = [stats.percentileofscore(logfc_sum_per_gene, gene) for gene in logfc_sum_per_gene_matches]

            # Filter genes with percentiles > 95
            top_percentiles = [
                (percentile, logfc_sum, gene)
                for percentile, logfc_sum, gene in zip(percentiles, logfc_sum_per_gene_matches, gene_candidates_dict['Female']['Mic'])
                if percentile > 95
            ]

            lower_percentiles = [
                (percentile, logfc_sum, gene)
                for percentile, logfc_sum, gene in zip(percentiles, logfc_sum_per_gene_matches, gene_candidates_dict['Female']['Mic'])
                if percentile < 95
            ]

            # increase font size
            plt.rcParams.update({'font.size': 12})

            # Create a histogram
            fig5 = plt.figure(figsize=(10, 5))
            #plt.hist(logfc_sum_per_gene, bins=30, alpha=0.7, color='blue', edgecolor='black')
            sns.histplot(logfc_sum_per_gene, stat='proportion')

            # Annotate the histogram with top-percentile genes
            for percentile, logfc_sum, gene in top_percentiles:
                plt.axvline(logfc_sum, color='black', linestyle='--', label=f'{gene}: {percentile:.0f}th percentile')
                plt.annotate(f'{gene}\n{percentile:.0f}%',  # Gene name with percentile below
                            xy=(logfc_sum, plt.gca().get_ylim()[1] * 0.7),
                            xytext=(logfc_sum + 5, plt.gca().get_ylim()[1] * 0.8),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=12, color='black')

            # Annotate the histogram with lower-percentile genes
            for percentile, logfc_sum, gene in lower_percentiles:
                offsetx = 25 if percentile == np.min([lp[0] for lp in lower_percentiles]) else 0
                offsety = 0.025 if percentile == np.min([lp[0] for lp in lower_percentiles]) else 0
                plt.axvline(logfc_sum, color='black', linestyle='--', label=f'{gene}: {percentile:.0f}th percentile')
                plt.annotate(f'{gene}\n{percentile:.0f}%',  # Gene name with percentile below
                            #xy=(logfc_sum, (plt.gca().get_ylim()[1] * percentile / 100) * 0.8 - offsety) ,
                            #xytext=(logfc_sum + 2 + offsetx, (plt.gca().get_ylim()[1] * percentile / 100) - offsety) ,
                            xy=(logfc_sum, plt.gca().get_ylim()[1] * 0.7),
                            xytext=(logfc_sum + 5, plt.gca().get_ylim()[1] * 0.8),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=12, color='black')

            # Add labels and title
            plt.xlabel("$\phi_g$", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.show()

            fig5.savefig(os.path.join(figpath, f'fig5_cosine_histogram.png'), bbox_inches='tight', dpi=300)

            '''
            cosine_case_hit = (cosine_case==1) * (cosine_control!=1) * cosine_valid_comparison
            cosine_control_hit = (cosine_control==1) * (cosine_case!=1) * cosine_valid_comparison

            cosine_case_hit_by_gene = cosine_case_hit.sum(dim=0)
            cosine_control_hit_by_gene = cosine_control_hit.sum(dim=0)

            cosine_case_hit_ranked_idxs = cosine_case_hit_by_gene.argsort().flip(0)
            cosine_case_hit_ranked_gene_names = mdd_rna.var_names[cosine_case_hit_ranked_idxs]

            cosine_case_hit_idxs = torch.where(cosine_case_hit)
            hits_idxs_dict[sex][celltype] = cosine_case_hit_idxs
            

            logfc_mean_per_gene_ranked_idxs = np.argsort(logfc_mean_per_gene)[::-1]
            logfc_mean_per_gene_ranked_gene_names = mdd_rna.var_names[logfc_mean_per_gene_ranked_idxs]

            log1p_fold_change = torch.log1p(cosine_case) - torch.log1p(cosine_control)
            #log1p_fold_change = log1p_fold_change[cosine_valid_comparison]

            nonzero_log1p_fold_change = log1p_fold_change[log1p_fold_change != 0]

            num_comparisons = len(unique_celltypes)
            corrected_percentile = 1 - (0.05 / num_comparisons)

            log1p_fold_change_95th_percentile = torch.quantile(nonzero_log1p_fold_change, corrected_percentile)
            '''
        break
    break