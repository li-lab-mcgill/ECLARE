import os
import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    slurm_job_path = '/home/dmannk/scratch'
    default_outdir = '/home/dmannk/scratch'
    os.environ['CLARE_root'] = '/home/dmannk/projects/def-liyue/dmannk/CLARE'

elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    slurm_job_path = '/Users/dmannk/cisformer/outputs'
    default_outdir = '/Users/dmannk/cisformer/outputs'
    os.environ['CLARE_root'] = '/Users/dmannk/cisformer/CLARE'

elif 'mcb-gpu1' in hostname:
    os.environ['machine'] = 'mcb'
    slurm_job_path = '/home/mcb/users/dmannk/scMultiCLIP/outputs'
    default_outdir = '/home/mcb/users/dmannk/scMultiCLIP/outputs'
    os.environ['CLARE_root'] = '/home/mcb/users/dmannk/scMultiCLIP/CLARE'

import sys
sys.path.insert(0, os.environ['CLARE_root'])
os.environ['outdir'] = default_outdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math
import torch
from ot import solve as ot_solve
from collections import defaultdict
from string import ascii_uppercase

from eclare.models import load_CLIP_model, CLIP
from eclare.setup_utils import mdd_setup
from eclare.post_hoc_utils import get_latents, sample_proportional_celltypes_and_condition

from datetime import datetime
from glob import glob

def get_metrics(method, job_id, target_only=False):

    method_job_id = f'{method}_{job_id}'
    paths_root = os.path.join(default_outdir, method_job_id)

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

def load_scMulticlip_mdd_model(student_model_path):
    student_model_args_dict = torch.load(student_model_path, map_location='cpu')

    slurm_job_id = student_model_args_dict['args'].slurm_job_ids
    model_path = glob(os.path.join(os.environ['outdir'], f'clip_mdd_{slurm_job_id}/mdd/AD_Anderson_et_al/{best_multiclip_mdd}/model.pt'))[0]
    _, clip_model_args_dict = load_CLIP_model(model_path, device='cpu')

    clip_model_args_dict['args'].source_dataset = 'mdd'
    clip_model_args_dict['args'].target_dataset = None

    clip_model_args_dict['args'].genes_by_peaks_str = '17563_by_100000'
    clip_model_args_dict['n_genes'] = 17563
    clip_model_args_dict['n_peaks'] = 100000
    clip_model_args_dict['tuned_hyperparameters']['params_num_layers'] = 2
    clip_model_args_dict['pretrain'] = clip_model_args_dict['rna_valid_idx']  = clip_model_args_dict['atac_valid_idx'] = None

    clip_model_args_dict['model_state_dict'] = student_model_args_dict['model_state_dict']

    student_model = CLIP(**clip_model_args_dict, trial=None)

    return student_model

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '15122421',
    'kd_clip': '16101332_17203243',
    'scMulticlip': '16101728',
    'clip_mdd': '15155920',
    'kd_clip_mdd': '20091454',
    'scMulticlip_mdd': '19154717', #16105437
    'mojitoo': '20212916',
    'multiVI': '18175921',
    'glue': '18234131_19205223',
    'scDART': '22083939', #'19150754',
    'scjoint': '19115016'
}

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


## MDD GRN analysis

best_multiclip_mdd = str(mdd_df_multiclip['ilisis'].droplevel(0).argmax())
method_job_id = f'scMulticlip_mdd_{methods_id_dict["scMulticlip_mdd"]}'
paths_root = os.path.join(default_outdir, method_job_id)
student_model_path = os.path.join(paths_root, 'mdd', best_multiclip_mdd, 'student_model.pt')
#model, model_args_dict = load_CLIP_model(model_path, device='cpu')

student_model = load_scMulticlip_mdd_model(student_model_path)
student_model.eval()

mdd_rna, mdd_atac, mdd_cell_group, target_genes_to_peaks_binary_mask, target_genes_peaks_dict, _, _ = mdd_setup(student_model.args, pretrain=None, return_type='data',\
    overlapping_subjects_only=False)

#mdd_rna_sampled, mdd_atac_sampled, mdd_rna_celltypes, mdd_atac_celltypes, mdd_rna_condition, mdd_atac_condition = \
#   sample_proportional_celltypes_and_condition(mdd_rna, mdd_atac, batch_size=50000)

rna_celltype_key='ClustersMapped'
atac_celltype_key='ClustersMapped'

rna_condition_key='Condition'
atac_condition_key='condition'

rna_subject_key='OriginalSub'
atac_subject_key='BrainID'

unique_celltypes = np.unique(np.concatenate([mdd_rna.obs[rna_celltype_key], mdd_atac.obs[atac_celltype_key]]))
unique_conditions = np.unique(np.concatenate([mdd_rna.obs[rna_condition_key], mdd_atac.obs[atac_condition_key]]))
unique_sexes = ['Female', 'Male']
unique_overlapping_subjects = np.intersect1d(mdd_rna.obs[rna_subject_key], mdd_atac.obs[atac_subject_key])

mdd_rna = mdd_rna[mdd_rna.obs[rna_subject_key].isin(unique_overlapping_subjects)]
mdd_atac = mdd_atac[mdd_atac.obs[atac_subject_key].isin(unique_overlapping_subjects)]

unique_atac_subjects = mdd_atac.obs[atac_subject_key].unique()
mdd_rna_sorted = mdd_rna[mdd_rna.obs[rna_subject_key].isin(unique_atac_subjects)]
mdd_rna_sorted = mdd_rna_sorted.obs.set_index(rna_subject_key).loc[unique_atac_subjects].reset_index()
assert (mdd_rna_sorted[rna_subject_key].unique() == unique_atac_subjects).all()

mdd_rna = mdd_rna[mdd_rna.obs[rna_subject_key].argsort()]
mdd_atac = mdd_atac[mdd_atac.obs[atac_subject_key].argsort()]

def tree(): return defaultdict(tree)
genes_by_peaks_cosines_dict = tree()

cutoff = 1300  # 1300 for Mic

## sex='Female'; celltype='Mic'
## sex='Female'; celltype='InN'

for sex in unique_sexes:
    for celltype in unique_celltypes:
        for condition in unique_conditions:
            for subject in unique_overlapping_subjects:

                print(f'sex: {sex} - celltype: {celltype} - condition: {condition} - subject: {subject}')

                rna_indices = np.where((mdd_rna.obs[rna_celltype_key] == celltype) & (mdd_rna.obs[rna_condition_key] == condition) & (mdd_rna.obs['Sex'] == sex) & (mdd_rna.obs[rna_subject_key] == subject))[0]
                atac_indices = np.where((mdd_atac.obs[atac_celltype_key] == celltype) & (mdd_atac.obs[atac_condition_key] == condition) & (mdd_atac.obs['sex'] == sex.lower()) & (mdd_atac.obs[atac_subject_key] == subject))[0]

                if len(rna_indices) == 0 or len(atac_indices) == 0:
                    print(f'No indices')
                    genes_by_peaks_cosines_dict[sex][celltype][condition][subject] = None
                    continue

                if len(rna_indices) > cutoff:
                    rna_indices = np.random.choice(rna_indices, cutoff, replace=False)
                if len(atac_indices) > cutoff:
                    atac_indices = np.random.choice(atac_indices, cutoff, replace=False)

                mdd_rna_sampled_group = mdd_rna[rna_indices]
                mdd_atac_sampled_group = mdd_atac[atac_indices]

                if len(rna_indices) == 0 or len(atac_indices) == 0:
                    print(f'No indices')

                rna_latents, atac_latents = get_latents(student_model, mdd_rna_sampled_group, mdd_atac_sampled_group, return_tensor=True)

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

                X_rna = torch.from_numpy(mdd_rna_sampled_group.X.toarray().T)
                X_atac = torch.from_numpy(mdd_atac_sampled_group.X.toarray().T)

                ## normalize gene expression and chromatin accessibility
                X_rna = torch.nn.functional.normalize(X_rna, p=2, dim=1)
                X_atac = torch.nn.functional.normalize(X_atac, p=2, dim=1)

                ## correlate gene expression with chromatin accessibility
                cosine = torch.matmul(X_atac, X_rna.T)
                genes_by_peaks_cosines_dict[sex][celltype][condition][subject] = cosine

        break
    break

gene_candidates_dict = tree()
gene_candidates_dict['Female']['Mic'] = ['ROBO2', 'SLIT3', 'ADAMSTL1', 'THSD4', 'SPP1', 'SOCS3', 'GAS6', 'MERTK']
gene_candidates_dict['Female']['InN'] = gene_candidates_dict['Female']['Mic']

hits_idxs_dict = tree()

for sex in unique_sexes:
    for celltype in unique_celltypes:

        cosine_case = genes_by_peaks_cosines_dict[sex][celltype]['Case']
        cosine_control = genes_by_peaks_cosines_dict[sex][celltype]['Control']

        cosine_valid_comparison = (cosine_case!=0) * (cosine_control!=0)
        cosine_case = cosine_case[cosine_valid_comparison]
        cosine_control = cosine_control[cosine_valid_comparison]

        log1p_fold_change = torch.log1p(cosine_case) - torch.log1p(cosine_control)

        where_valid = np.where(cosine_valid_comparison)
        genes_where_valid_idxs = where_valid[1]
        logfc_sum_per_gene = np.bincount(genes_where_valid_idxs, weights=log1p_fold_change.abs())
        logfc_mean_per_gene = logfc_sum_per_gene / np.bincount(genes_where_valid_idxs)
        assert len(logfc_sum_per_gene) == mdd_rna.n_vars

        logfc_sum_per_gene_ranked_idxs = np.argsort(logfc_sum_per_gene)[::-1]
        logfc_sum_per_gene_ranked_gene_names = mdd_rna.var_names[logfc_sum_per_gene_ranked_idxs]
        logfc_sum_per_gene_matches = np.concatenate([logfc_sum_per_gene[mdd_rna.var_names == gene] for gene in gene_candidates_dict['Female']['Mic'] if logfc_sum_per_gene[mdd_rna.var_names == gene].size > 0])

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
                        xy=(logfc_sum, (plt.gca().get_ylim()[1] * percentile / 100) * 0.8 - offsety) ,
                        xytext=(logfc_sum + 2 + offsetx, (plt.gca().get_ylim()[1] * percentile / 100) - offsety) ,
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