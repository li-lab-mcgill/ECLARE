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
from ot import solve as ot_solve
from collections import defaultdict
from string import ascii_uppercase
import pybedtools
from types import SimpleNamespace

from mlflow.tracking import MlflowClient
import mlflow.pytorch
from mlflow.models import Model

from eclare.models import load_CLIP_model, CLIP
from eclare.setup_utils import mdd_setup
from eclare.post_hoc_utils import get_latents, sample_proportional_celltypes_and_condition, plot_umap_embeddings, create_celltype_palette

from datetime import datetime
from glob import glob

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
    ).tick_params(axis='x', rotation=30)
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

    fig, axs = plt.subplots(1, len(all_metrics), figsize=(12, 4), sharex=True)

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
    'kd_clip': '22090727',
    'eclare': '22090820',
    'clip_mdd': '16204608',
    'kd_clip_mdd': '17082436',
    'eclare_mdd': '17082437', #16105437
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
    'eclare': 'ECLARE' + '_' + methods_id_dict['eclare'],
    'clip_mdd': 'CLIP' + '_' + methods_id_dict['clip_mdd'],
    'kd_clip_mdd': 'KD_CLIP' + '_' + methods_id_dict['kd_clip_mdd'],
    'eclare_mdd': 'ECLARE' + '_' + methods_id_dict['eclare_mdd']
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
ECLARE_header_idx = np.where(all_metrics_df['run_name'].str.startswith(search_strings['eclare']))[0]

CLIP_run_id = all_metrics_df.iloc[CLIP_header_idx]['run_id']
KD_CLIP_run_id = all_metrics_df.iloc[KD_CLIP_header_idx]['run_id']
ECLARE_run_id = all_metrics_df.iloc[ECLARE_header_idx]['run_id']

CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(CLIP_run_id)]
KD_CLIP_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(KD_CLIP_run_id)]
ECLARE_metrics_df = all_metrics_df.loc[all_metrics_df['parent_run_id'].isin(ECLARE_run_id)]

CLIP_metrics_df = extract_target_source_replicate(CLIP_metrics_df)
KD_CLIP_metrics_df = extract_target_source_replicate(KD_CLIP_metrics_df)
ECLARE_metrics_df = extract_target_source_replicate(ECLARE_metrics_df, has_source=False)

CLIP_metrics_df.loc[:, 'dataset']      = 'clip'
ECLARE_metrics_df.loc[:, 'dataset']    = 'eclare'
KD_CLIP_metrics_df.loc[:, 'dataset']   = 'kd_clip'

combined_metrics_df = pd.concat([ECLARE_metrics_df, KD_CLIP_metrics_df, CLIP_metrics_df]) # determines order in which metrics are plotted

metric_boxplots(combined_metrics_df)

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
ECLARE_mdd_header_idxs = np.where(mdd_metrics_df['run_name'].str.startswith(search_strings['eclare_mdd']))[0]

CLIP_mdd_run_id = mdd_metrics_df.iloc[CLIP_mdd_header_idxs]['run_id']
KD_CLIP_mdd_run_id = mdd_metrics_df.iloc[KD_CLIP_mdd_header_idxs]['run_id']
ECLARE_mdd_run_id = mdd_metrics_df.iloc[ECLARE_mdd_header_idxs]['run_id']

CLIP_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(CLIP_mdd_run_id)]
KD_CLIP_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(KD_CLIP_mdd_run_id)]
ECLARE_mdd_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(ECLARE_mdd_run_id)]

CLIP_mdd_metrics_df = extract_target_source_replicate(CLIP_mdd_metrics_df)
KD_CLIP_mdd_metrics_df = extract_target_source_replicate(KD_CLIP_mdd_metrics_df)
ECLARE_mdd_metrics_df = extract_target_source_replicate(ECLARE_mdd_metrics_df, has_source=False)

CLIP_mdd_metrics_df.loc[:, 'dataset'] = 'clip_mdd'
KD_CLIP_mdd_metrics_df.loc[:, 'dataset'] = 'kd_clip_mdd'
ECLARE_mdd_metrics_df.loc[:, 'dataset'] = 'eclare_mdd'

combined_mdd_metrics_df = pd.concat([ECLARE_mdd_metrics_df, KD_CLIP_mdd_metrics_df, CLIP_mdd_metrics_df])

metric_boxplots(combined_mdd_metrics_df, target_source_combinations=True, include_paired=False)


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

## Find path to best ECLARE model
best_eclare_mdd = str(ECLARE_mdd_metrics_df['multimodal_ilisi'].argmax())
student_job_id = f'eclare_mdd_{methods_id_dict["eclare_mdd"]}'
paths_root = os.path.join(os.environ['OUTPATH'], student_job_id)
student_model_path = os.path.join(paths_root, 'MDD', best_eclare_mdd, 'model_uri.txt')

## Load the model and metadata
with open(student_model_path, 'r') as f:
    model_uris = f.read().strip().splitlines()
    model_uri = model_uris[0]

## Set tracking URI to ECLARE_ROOT/mlruns and load model & metadata
mlflow.set_tracking_uri(os.path.join(os.environ['ECLARE_ROOT'], 'mlruns'))
student_model = mlflow.pytorch.load_model(model_uri)
student_model_metadata = Model.load(model_uri)


#%% load data

args = SimpleNamespace(
    source_dataset='MDD',
    target_dataset=None,
    genes_by_peaks_str='17563_by_100000'
)

mdd_rna, mdd_atac, mdd_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, _, _ = \
    mdd_setup(args, return_raw_data=True, return_type='data',\
    overlapping_subjects_only=False)

mdd_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(list(mdd_atac.var_names.str.split(':|-', expand=True)), columns=['chrom', 'start', 'end']))

## define keys
rna_celltype_key='ClustersMapped'
atac_celltype_key='ClustersMapped'

rna_condition_key='Condition'
atac_condition_key='condition'

rna_subject_key='OriginalSub'
atac_subject_key='BrainID'

unique_celltypes = np.unique(np.concatenate([mdd_rna.obs[rna_celltype_key], mdd_atac.obs[atac_celltype_key]]))
unique_conditions = np.unique(np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]))
unique_sexes = ['Female', 'Male']

#%% project MDD nuclei into latent space

mdd_rna_sampled, mdd_atac_sampled, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition = \
   sample_proportional_celltypes_and_condition(mdd_rna, mdd_atac, batch_size=10000)

# subset to 'Mic' celltype
'''
mdd_rna_sampled = mdd_rna[mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_sampled = mdd_atac[mdd_atac.obs[atac_celltype_key] == 'Mic']
mdd_rna_condition = mdd_rna.obs[rna_condition_key][mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_condition = mdd_atac.obs[atac_condition_key][mdd_atac.obs[atac_celltype_key] == 'Mic']
mdd_rna_celltypes = mdd_rna.obs[rna_celltype_key][mdd_rna.obs[rna_celltype_key] == 'Mic']
mdd_atac_celltypes = mdd_atac.obs[atac_celltype_key][mdd_atac.obs[atac_celltype_key] == 'Mic']
'''


student_model = student_model.to('cpu')
rna_latents, atac_latents = get_latents(student_model, mdd_rna_sampled, mdd_atac_sampled, return_tensor=True)

rna_latents = rna_latents.cpu().numpy()
atac_latents = atac_latents.cpu().numpy()

color_map_ct = create_celltype_palette(unique_celltypes, unique_celltypes, plot_color_palette=False)
plot_umap_embeddings(rna_latents, atac_latents, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition, color_map_ct=color_map_ct, umap_embedding=None)

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


#%% SEACells
import SEACells
import scanpy as sc

def run_SEACells(adata_train, adata_apply, build_kernel_on, redo_umap=False, key='X_umap'):

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
    n_SEACells = max(adata_train.n_obs // 75, 30)
    print(f'Number of SEACells: {n_SEACells}')
    n_waypoint_eigs = 15 # Number of eigenvalues to consider when initializing metacells

    ## Initialize SEACells model
    model = SEACells.core.SEACells(adata_train, 
                    build_kernel_on=build_kernel_on, 
                    n_SEACells=n_SEACells, 
                    n_waypoint_eigs=n_waypoint_eigs,
                    convergence_epsilon = 1e-5,
                    max_franke_wolfe_iters=100,
                    use_gpu=False)

    model.construct_kernel_matrix()
    M = model.kernel_matrix

    # Initialize archetypes
    model.initialize_archetypes()

    # Fit SEACells model
    model.fit(min_iter=10, max_iter=50)

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




#%% get peak-gene correlations

def tree(): return defaultdict(tree)
genes_by_peaks_corrs_dict = tree()
n_dict = tree()

cutoff = 1300  # 1300 for Mic

## sex='Female'; celltype='Mic'
## sex='Female'; celltype='InN'

maitra_female_degs_df   = pd.read_excel(os.path.join(os.environ['DATAPATH'], 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData6', header=2)
doruk_peaks_df          = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'combined', 'cluster_DAR_0.05.tsv'), sep='\t')
#maitra_male_degs_df = pd.read_excel(os.path.join(datapath, 'Maitra_et_al_supp_tables.xlsx'), sheet_name='SupplementaryData5', header=2)

## get HVG features
sc.pp.highly_variable_genes(mdd_rna)
sc.pp.highly_variable_genes(mdd_atac, n_top_genes=10000)

## use all peaks and genes
peaks_indices_hvg = mdd_atac.var['highly_variable'].astype(bool)
genes_indices_hvg = mdd_rna.var['highly_variable'].astype(bool)

do_corrs_or_cosines = 'correlations'

for sex in unique_sexes:
    for celltype in unique_celltypes:

        celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id'].str.startswith(celltype)]

        celltype_peaks_df = doruk_peaks_df[doruk_peaks_df['cluster'].str.startswith(celltype)]
        celltype_peaks = celltype_peaks_df['peakName'].str.split('-')
        celltype_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame({'chrom': celltype_peaks.str[0], 'start': celltype_peaks.str[1], 'end': celltype_peaks.str[2]}))
        
        ## select peaks indices using bedtools intersect
        peaks_indices_dar = mdd_peaks_bed.intersect(celltype_peaks_bed, c=True).to_dataframe()['name'].astype(bool)
        genes_indices_deg = mdd_rna.var_names.isin(celltype_degs_df['gene'])

        #peaks_indices = peaks_indices_hvg.values | peaks_indices_dar.values
        genes_indices = genes_indices_hvg.values | genes_indices_deg
        peaks_indices = peaks_indices_dar
        #genes_indices = genes_indices_deg

        for condition in unique_conditions:

            print(f'sex: {sex} - celltype: {celltype} - condition: {condition} - DAR peaks: {peaks_indices.sum()} - DEG genes: {genes_indices.sum()}')

            ## select cell indices
            rna_indices = np.where((mdd_rna.obs[rna_celltype_key] == celltype) & (mdd_rna.obs[rna_condition_key] == condition) & (mdd_rna.obs['Sex'] == sex))[0]
            atac_indices = np.where((mdd_atac.obs[atac_celltype_key] == celltype) & (mdd_atac.obs[atac_condition_key] == condition) & (mdd_atac.obs['sex'] == sex.lower()))[0]

            if len(rna_indices) > cutoff:
                rna_indices = np.random.choice(rna_indices, cutoff, replace=False)
            if len(atac_indices) > cutoff:
                atac_indices = np.random.choice(atac_indices, cutoff, replace=False)

            mdd_rna_sampled_group = mdd_rna[rna_indices]
            mdd_atac_sampled_group = mdd_atac[atac_indices]

            if len(rna_indices) == 0 or len(atac_indices) == 0:
                print(f'No indices')

            ## get latents
            rna_latents, atac_latents = get_latents(student_model, mdd_rna_sampled_group, mdd_atac_sampled_group, return_tensor=True)
            rna_latents = rna_latents.cpu()
            atac_latents = atac_latents.cpu()

            ## trim latents to smallest size
            min_size = min(rna_latents.shape[0], atac_latents.shape[0])
            rna_latents = rna_latents[:min_size]
            atac_latents = atac_latents[:min_size]
            mdd_rna_sampled_group = mdd_rna_sampled_group[:min_size]
            mdd_atac_sampled_group = mdd_atac_sampled_group[:min_size]

            ## get logits - already normalized during clip loss, but need to normalize before to be consistent with Concerto
            rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
            atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
            student_logits = torch.matmul(atac_latents, rna_latents.T)

            ot_res = ot_solve(1 - student_logits)
            plan = ot_res.plan
            value = ot_res.value_linear

            ## re-order RNA latents to match plan (can rerun OT analysis to ensure diagonal matching structure)
            rna_latents = rna_latents[plan.argmax(axis=1)]
            mdd_rna_sampled_group = mdd_rna_sampled_group[plan.argmax(axis=1).numpy()]

            ## select genes and peaks before SEACells
            mdd_rna_sampled_group = mdd_rna_sampled_group[:,genes_indices]
            mdd_atac_sampled_group = mdd_atac_sampled_group[:,peaks_indices]

            mdd_rna_sampled_group_seacells, mdd_atac_sampled_group_seacells = run_SEACells(mdd_rna_sampled_group, mdd_atac_sampled_group, build_kernel_on='X_pca', key='X_UMAP_Harmony_Batch_Sample_Chemistry')

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

            ## transpose
            X_rna = X_rna.T
            X_atac = X_atac.T

            if do_corrs_or_cosines == 'correlations':
                ## concatenate
                X_rna_atac = torch.cat([X_rna, X_atac], dim=0)

                corr = torch.corrcoef(X_rna_atac.cuda() if cuda_available else X_rna_atac)
                corr = corr[0:genes_indices.sum(), genes_indices.sum():]

                if corr.isnan().any():
                    print(f'NaN values in correlation matrix')
                    corr[corr.isnan()] = 0

                genes_by_peaks_corrs_dict[sex][celltype][condition] = corr.detach().cpu()
                n_dict[sex][celltype][condition] = X_rna_atac.shape[1]

            elif do_corrs_or_cosines == 'cosines':

                ## normalize gene expression and chromatin accessibility
                X_rna = torch.nn.functional.normalize(X_rna, p=2, dim=1)
                X_atac = torch.nn.functional.normalize(X_atac, p=2, dim=1)

                ## correlate gene expression with chromatin accessibility
                cosine = torch.matmul(X_atac, X_rna.T)
                genes_by_peaks_corrs_dict[sex][celltype][condition] = cosine

        break
    break

#%% get MDD-association gene scores

gene_candidates_dict = tree()
gene_candidates_dict['Female']['Mic'] = ['ROBO2', 'SLIT3', 'ADAMSTL1', 'THSD4', 'SPP1', 'SOCS3', 'GAS6', 'MERTK']
gene_candidates_dict['Female']['InN'] = gene_candidates_dict['Female']['Mic']

hits_idxs_dict = tree()
from scipy.stats import norm

for sex in unique_sexes:
    for celltype in unique_celltypes:

        celltype_degs_df = maitra_female_degs_df[maitra_female_degs_df['cluster_id'].str.startswith(celltype)]
        #genes_names = mdd_rna.var_names[mdd_rna.var_names.isin(celltype_degs_df['gene'])]
        genes_names = mdd_rna.var_names[genes_indices]

        corr_case = genes_by_peaks_corrs_dict[sex][celltype]['Case']
        corr_control = genes_by_peaks_corrs_dict[sex][celltype]['Control']

        corr_case = corr_case.numpy()
        corr_control = corr_control.numpy()

        fisher_case = np.arctanh(corr_case)
        fisher_control = np.arctanh(corr_control)

        fisher_sums_case = np.sum(fisher_case, axis=1)
        fisher_sums_control = np.sum(fisher_control, axis=1)

        K_case = len(fisher_sums_case)
        K_control = len(fisher_sums_control)
        n_case = n_dict[sex][celltype]['Case']
        n_control = n_dict[sex][celltype]['Control']

        fisher_sums_var_case = K_case / (n_case - 3)
        fisher_sums_var_control = K_control / (n_control - 3)

        Z = (fisher_sums_case - fisher_sums_control) / np.sqrt(fisher_sums_var_case + fisher_sums_var_control)

        p_value = 2 * (1 - norm.cdf(abs(Z)))
        p_value_threshold = 0.05 / K_case
        hits_idxs = np.where(p_value < p_value_threshold)[0]

        hits_idxs_dict[sex][celltype] = hits_idxs

        '''
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
#%% GSEApy

import gseapy as gp

ranked_list = pd.DataFrame({'gene': genes_names.to_list(), 'score': Z})
ranked_list = ranked_list.sort_values(by='score', ascending=False)

pre_res = gp.prerank(rnk=ranked_list,
                     gene_sets='Reactome_2022', # apparently, GO_Biological_Process_2021 and Reactome_2022 better for neurological disorders
                     outdir=os.path.join(os.environ['OUTPATH'], 'gseapy_results'),
                     min_size=1,
                     max_size=500,
                     permutation_num=1000)

pre_res.res2d.head()

#%%


