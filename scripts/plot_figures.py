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

from IPython.display import display
import pickle
import anndata
from venn import venn

from mlflow.tracking import MlflowClient
import mlflow.pytorch
from mlflow.models import Model

from eclare.post_hoc_utils import \
    get_latents, sample_proportional_celltypes_and_condition, plot_umap_embeddings, create_celltype_palette, download_mlflow_runs, extract_target_source_replicate, metric_boxplots
from eclare.data_utils import filter_mean_grn, get_scompreg_loglikelihood

from ot import solve as ot_solve
import SEACells
import scanpy as sc
import scglue
import networkx as nx

import gseapy as gp
import mygene
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import norm, linregress, gamma
from scipy.optimize import minimize

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from eclare.post_hoc_utils import do_enrichr, run_magma, run_gseapy, get_pathway_ranks, plot_pathway_ranks, get_latents, sample_proportional_celltypes_and_condition, plot_umap_embeddings, create_celltype_palette

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

ablation_clip_metrics_df.loc[ablation_clip_metrics_df['source']=='pbmc_10x', 'dataset'] = 'clip (pbmc)'
ablation_clip_metrics_df.loc[ablation_clip_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'clip (mb)'

## KD-CLIP
ablation_kd_clip_metrics_df = combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['kd_clip'])]
ablation_kd_clip_metrics_df = ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_kd_clip_metrics_df = ablation_kd_clip_metrics_df.loc[~ablation_kd_clip_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source']=='pbmc_10x', 'dataset'] = 'kd-clip (pbmc)'
ablation_kd_clip_metrics_df.loc[ablation_kd_clip_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'kd-clip (mb)'

## ECLARE
ablation_eclare_metrics_df = combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['eclare'])]
ablation_eclare_metrics_df.loc[:, 'dataset'] = 'eclare'

## Combined ablation
ablation_metrics_df = pd.concat([
    ablation_eclare_metrics_df,
    ablation_kd_clip_metrics_df,
    ablation_clip_metrics_df
    ])

## Reorder rows using "dataset" field
#ablation_metrics_df = ablation_metrics_df.sort_values('dataset')

## rename 'silhouette_celltype' by 'asw_ct'
ablation_metrics_df = ablation_metrics_df.rename(columns={'silhouette_celltype': 'asw_ct'})

datasets_order = ['eclare', 'kd-clip (mb)', 'clip (mb)', 'kd-clip (pbmc)', 'clip (pbmc)']
order_map = {name: i for i, name in enumerate(datasets_order)}
ablation_metrics_df = ablation_metrics_df.sort_values('dataset', key=lambda x: x.map(order_map))

boxplots_fig = metric_boxplots(ablation_metrics_df, target_source_combinations=False, include_paired=True)

## MDD

## CLIP
ablation_clip_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['clip_mdd'])]
ablation_clip_mdd_metrics_df = ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_clip_mdd_metrics_df = ablation_clip_mdd_metrics_df.loc[~ablation_clip_mdd_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source']=='pbmc_10x', 'dataset'] = 'clip (pbmc)'
ablation_clip_mdd_metrics_df.loc[ablation_clip_mdd_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'clip (mb)'

## KD-CLIP
ablation_kd_clip_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['kd_clip_mdd'])]
ablation_kd_clip_mdd_metrics_df = ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source'].isin(['pbmc_10x', 'mouse_brain_10x'])]
ablation_kd_clip_mdd_metrics_df = ablation_kd_clip_mdd_metrics_df.loc[~ablation_kd_clip_mdd_metrics_df['target'].isin(['pbmc_10x', 'mouse_brain_10x'])]

ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source']=='pbmc_10x', 'dataset'] = 'kd-clip (pbmc)'
ablation_kd_clip_mdd_metrics_df.loc[ablation_kd_clip_mdd_metrics_df['source']=='mouse_brain_10x', 'dataset'] = 'kd-clip (mb)'

## ECLARE
ablation_eclare_mdd_metrics_df = combined_mdd_metrics_df.loc[combined_mdd_metrics_df['dataset'].isin(['eclare_mdd'])]
ablation_eclare_mdd_metrics_df.loc[:, 'dataset'] = 'eclare'

## Combined ablation
ablation_mdd_metrics_df = pd.concat([
    ablation_eclare_mdd_metrics_df,
    ablation_kd_clip_mdd_metrics_df,
    ablation_clip_mdd_metrics_df
    ])

## Reorder rows using "dataset" field
datasets_order = ['eclare', 'kd-clip (mb)', 'clip (mb)', 'kd-clip (pbmc)', 'clip (pbmc)']
order_map = {name: i for i, name in enumerate(datasets_order)}
ablation_mdd_metrics_df = ablation_mdd_metrics_df.sort_values('dataset', key=lambda x: x.map(order_map))

## rename 'silhouette_celltype' by 'asw_ct'
ablation_mdd_metrics_df = ablation_mdd_metrics_df.rename(columns={'silhouette_celltype': 'asw_ct'})

boxplots_mdd_fig = metric_boxplots(ablation_mdd_metrics_df, target_source_combinations=True, include_paired=False)

## save boxplot figures as high-quality figures (300 dpi)
figpath = "/Users/dmannk/OneDrive - McGill University/Doctorat/QLS/Li_Lab/ISMB_2025/figures"
boxplots_fig.savefig(os.path.join(figpath, 'cross_tissue_species_metrics.png'), bbox_inches='tight', dpi=300)
boxplots_mdd_fig.savefig(os.path.join(figpath, 'cross_tissue_species_mdd_metrics.png'), bbox_inches='tight', dpi=300)


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

#eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_mdd_{methods_id_dict["eclare_mdd"]}', best_eclare_mdd)
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_mdd_{methods_id_dict["eclare_mdd"][0]}', best_eclare_mdd)
kd_clip_student_model, kd_clip_student_model_metadata   = load_model_and_metadata(f'kd_clip_mdd_{methods_id_dict["kd_clip_mdd"]}', os.path.join(best_kd_clip_mdd, '0'))

eclare_student_model = eclare_student_model.train().to('cpu')
kd_clip_student_model = kd_clip_student_model.train().to('cpu')

#%% load data

## define decimation factor
decimate_factor = 1

## define args for mdd_setup
args = SimpleNamespace(
    source_dataset='MDD',
    target_dataset=None,
    genes_by_peaks_str='17563_by_100000'
)

rna_datapath = atac_datapath = os.path.join(os.environ['DATAPATH'], 'mdd_data')

RNA_file = f"rna_{args.genes_by_peaks_str}.h5ad"
ATAC_file = f"atac_{args.genes_by_peaks_str}.h5ad"

rna_fullpath = os.path.join(rna_datapath, RNA_file)
atac_fullpath = os.path.join(atac_datapath, ATAC_file)

atac = anndata.read_h5ad(atac_fullpath, backed='r')
rna  = anndata.read_h5ad(rna_fullpath, backed='r')

## define keys
rna_celltype_key='ClustersMapped'
atac_celltype_key='ClustersMapped'

rna_condition_key='Condition'
atac_condition_key='condition'

rna_subject_key='OriginalSub'
atac_subject_key='BrainID'

rna_sex_key = 'Sex'
atac_sex_key = 'sex'

mdd_atac = atac[::decimate_factor].to_memory()
mdd_rna = rna[::decimate_factor].to_memory()

mdd_peaks_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(list(mdd_atac.var_names.str.split(':|-', expand=True)), columns=['chrom', 'start', 'end']))

## also load counts data for pyDESeq2 and full processed data for annotations
rna_full = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna.h5ad'), backed='r')
rna_scaled_with_counts = anndata.read_h5ad(os.path.join(rna_datapath, 'mdd_rna_scaled.h5ad'), backed='r')
rna_scaled_with_counts = rna_scaled_with_counts[::decimate_factor].to_memory()

ct_map_dict = dict({1: 'ExN', 0: 'InN', 4: 'Oli', 2: 'Ast', 3: 'OPC', 6: 'End', 5: 'Mix', 7: 'Mic'})
rna_scaled_with_counts.obs[rna_celltype_key] = rna_scaled_with_counts.obs['Broad'].map(ct_map_dict)

rna_counts_X = rna_scaled_with_counts.raw.X.astype(int).toarray()
rna_counts_obs = rna_scaled_with_counts.obs
rna_counts_var = rna_full.var
rna_counts_var['in_mdd'] = rna_counts_var.index.isin(mdd_rna.var_names)

mdd_rna_counts = anndata.AnnData(
    X=rna_counts_X,
    var=rna_counts_var,
    obs=rna_counts_obs,
)

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

## initialize dicts
pydeseq2_results_dict = tree()
significant_genes_dict = tree()

## TEMPORARY - restrict unique celltypes
unique_celltypes = ['Ast', 'End', 'InN', 'Mic', 'OPC', 'Oli']
     
## loop through sex and celltype
for sex in unique_sexes:
    # Use joblib for parallel processing
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(process_celltype)(
            sex, celltype, rna_scaled_with_counts, mdd_rna.var,
            rna_celltype_key, rna_condition_key, rna_sex_key
        )
        for celltype in unique_celltypes
    )
    
    # Process results
    for celltype, (mdd_subjects_counts_adata, pydeseq2_results, significant_genes) in zip(unique_celltypes, results):
        pydeseq2_results_dict[sex][celltype] = pydeseq2_results
        significant_genes_dict[sex][celltype] = significant_genes


#%% get peak-gene correlations

X_rna_dict, X_atac_dict, overlapping_target_genes_dict, overlapping_tfs_dict, genes_by_peaks_corrs_dict, genes_by_peaks_masks_dict, n_dict, scompreg_loglikelihoods_dict, std_errs_dict, slopes_dict, intercepts_dict, intercept_stderrs_dict, tg_expressions_dict, tfrps_dict, tfrp_predictions_dict \
    = initialize_dicts()

with open(os.path.join(os.environ['OUTPATH'], 'all_dicts_female.pkl'), 'rb') as f:
    all_dicts = pickle.load(f)
mean_grn_df = all_dicts[-1]

## set cutoff for number of cells to keep for SEACells representation. see Bilous et al. 2024, Liu & Li 2024 (mcRigor) or Li et al. 2025 (MetaQ) for benchmarking experiments

for sex in unique_sexes:
    for condition in unique_conditions:

        results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
                delayed(differential_grn_analysis)(
                    condition, sex, celltype, mdd_rna, mdd_atac, rna_celltype_key, rna_condition_key, rna_sex_key, rna_subject_key, atac_celltype_key, atac_condition_key, atac_sex_key, atac_subject_key, eclare_student_model, mean_grn_df, cutoff=5025, ot_alignment_type='all'
                )
            for celltype in unique_celltypes
        )

        for celltype, result in zip(unique_celltypes, results):
            assign_to_dicts(*result)

#!pyGenomeTracks --tracks tracks.ini --region chr2:157304654-157336585 -o tracks.png

#%% Save  all dicts

dicts_to_save = {
    'X_rna_dict': X_rna_dict,
    'X_atac_dict': X_atac_dict,
    'overlapping_target_genes_dict': overlapping_target_genes_dict,
    'overlapping_tfs_dict': overlapping_tfs_dict,
    'scompreg_loglikelihoods_dict': scompreg_loglikelihoods_dict,
    'std_errs_dict': std_errs_dict,
    'tg_expressions_dict': tg_expressions_dict,
    'tfrps_dict': tfrps_dict,
    'tfrp_predictions_dict': tfrp_predictions_dict,
    'slopes_dict': slopes_dict,
    'intercepts_dict': intercepts_dict,
    'intercept_stderrs_dict': intercept_stderrs_dict,
}

if 'pydeseq2_results_dict' in locals() and pydeseq2_results_dict is not None:
    dicts_to_save['pydeseq2_results_dict'] = pydeseq2_results_dict
if 'significant_genes_dict' in locals() and significant_genes_dict is not None:
    dicts_to_save['significant_genes_dict'] = significant_genes_dict

for dict_name, dict_obj in dicts_to_save.items():
    with open(os.path.join(os.environ['OUTPATH'], f"{dict_name}.pkl"), "wb") as f:
        print(f'Saving {dict_name}...')
        pickle.dump(dict_obj, f)




#%% Gene set enrichment analyses

## Get BrainGMT and filter for cortical genes
brain_gmt_cortical, brain_gmt_cortical_wGO = get_brain_gmt()


#%% EnrichR
enrs_dict = tree()
magma_results_dict = tree()

for sex in unique_sexes:
    
    results = Parallel(n_jobs=min(cpu_count(), len(unique_celltypes)), backend='threading')(
        delayed(perform_gene_set_enrichment)(
            sex, celltype, scompreg_loglikelihoods_dict, tfrps_dict, tg_expressions_dict, tfrp_predictions_dict, mean_grn_df, significant_genes_dict, mdd_rna.var_names
        )
        for celltype in unique_celltypes
    )

    for celltype, result in zip(unique_celltypes, results):
        enrs_dict[sex][celltype] = result[0]
        magma_results_dict[sex][celltype] = result[1]

#%% Save all dicts

dicts_to_save = {
    'enrs_dict': enrs_dict,
    'magma_results_dict': magma_results_dict,
}


for dict_name, dict_obj in dicts_to_save.items():
    with open(os.path.join(os.environ['OUTPATH'], f"{dict_name}.pkl"), "wb") as f:
        print(f'Saving {dict_name}...')
        pickle.dump(dict_obj, f)


#%% check ranks of pre-defined pathways

pathway_ranks = get_pathway_ranks(enr, enr_deg, enr_top_wDeg, enr_bottom_wDeg)
plot_pathway_ranks(pathway_ranks, stem=True)

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
    # If "data" has both "source" and "target", it's an edge; otherwise it's a node.
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