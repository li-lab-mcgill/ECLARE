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

#%%
import os
import pickle
import numpy as np
import pandas as pd
import pybedtools
import anndata
import torch
import seaborn as sns
from scanpy.tl import score_genes
from statsmodels.stats.weightstats import DescrStatsW
import json

# Set matplotlib to use a thread-safe backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add logging for better debugging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.environ.get('OUTPATH', '.'), 'enrichment_analyses.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from types import SimpleNamespace
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import threading
from glob import glob

from eclare.post_hoc_utils import \
    extract_target_source_replicate, initialize_dicts, assign_to_dicts, perform_gene_set_enrichment, differential_grn_analysis, process_celltype, load_model_and_metadata, get_brain_gmt, magma_dicts_to_df, get_next_version_dir, compute_LR_grns, do_enrichr, find_hits_overlap, \
    set_env_variables, download_mlflow_runs,\
    tree

set_env_variables()

cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '04201018',
    'kd_clip': '04214651',
    'eclare': ['04214648'],
}

## define search strings
search_strings = {
    'clip': 'CLIP' + '_' + methods_id_dict['clip'],
    'kd_clip': 'KD_CLIP' + '_' + methods_id_dict['kd_clip'],
    'eclare': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare']]
}

## Create output directory with version counter
base_output_dir = os.path.join(os.environ['OUTPATH'], f"cross_species_analysis_{methods_id_dict['eclare'][0]}")
output_dir = get_next_version_dir(base_output_dir)
os.makedirs(output_dir, exist_ok=True)

#%% unpaired MDD data
experiment_name = f"clip_{methods_id_dict['clip']}"

if os.path.exists(os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")):
    print(f"Found runs.csv for {experiment_name} in {os.environ['OUTPATH']}")
    all_metrics_csv_path = os.path.join(os.environ['OUTPATH'], experiment_name, "runs.csv")
else:
    print(f"Downloading runs.csv for {experiment_name} from MLflow")
    all_metrics_csv_path = download_mlflow_runs(experiment_name)

mdd_metrics_df = pd.read_csv(all_metrics_csv_path)
ECLARE_header_idxs = np.where(mdd_metrics_df['run_name'].apply(lambda x: any(x.startswith(s) for s in search_strings['eclare'])))[0]
ECLARE_run_id = mdd_metrics_df.iloc[ECLARE_header_idxs]['run_id']
ECLARE_metrics_df = mdd_metrics_df.loc[mdd_metrics_df['parent_run_id'].isin(ECLARE_run_id)]
ECLARE_metrics_df = extract_target_source_replicate(ECLARE_metrics_df, has_source=False)


#%% Load ECLARE student model and source datasets

target_dataset = 'DLPFC_Anderson'
source_datasets = ['DLPFC_Ma', 'mouse_brain_10x']

## Find path to best ECLARE model
best_eclare     = str(ECLARE_metrics_df['multimodal_ilisi'].argmax())
eclare_student_model, eclare_student_model_metadata     = load_model_and_metadata(f'eclare_{methods_id_dict["eclare"][0]}', best_eclare, device, target_dataset=target_dataset)
eclare_student_model = eclare_student_model.eval().to(device=device)

## Load KD_CLIP student model
best_kd_clip = '0'
kd_clip_student_models = {}

for source_dataset in source_datasets:
    kd_clip_student_model, kd_clip_student_model_metadata     = load_model_and_metadata(f'kd_clip_{methods_id_dict["kd_clip"]}', best_kd_clip, device, target_dataset=os.path.join(target_dataset, source_dataset))
    kd_clip_student_models[source_dataset] = kd_clip_student_model


# %% Setup student and teachers

from eclare.setup_utils import teachers_setup, return_setup_func_from_dataset
from eclare.data_utils import fetch_data_from_loader_light
from eclare.losses_and_distances_utils import cosine_distance

model_uri_paths_str = f"clip_*{methods_id_dict['clip']}/{target_dataset}/**/{best_eclare}/model_uri.txt"
model_uri_paths = glob(os.path.join(outpath, model_uri_paths_str))

genes_by_peaks_path = os.path.join(os.environ['DATAPATH'], 'genes_by_peaks_str.csv')
genes_by_peaks_df = pd.read_csv(genes_by_peaks_path, index_col=0)

## Setup student
student_setup_func = return_setup_func_from_dataset(target_dataset)
genes_by_peaks_str = genes_by_peaks_df.loc[target_dataset, 'MDD']

args = SimpleNamespace(
    source_dataset=target_dataset,
    target_dataset='MDD',
    genes_by_peaks_str=genes_by_peaks_str,
    ignore_sources=['Midbrain_Adams', 'pbmc_10x', 'PFC_Zhu'],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)

student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_total_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_total_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
    student_setup_func(args, return_type='loaders')

## project data through student model
student_rna_cells, student_rna_labels, student_rna_batches = fetch_data_from_loader_light(student_rna_valid_loader)
student_atac_cells, student_atac_labels, student_atac_batches = fetch_data_from_loader_light(student_atac_valid_loader)
student_rna_latents, _ = eclare_student_model(student_rna_cells.to(device=device), modality=0)
student_atac_latents, _ = eclare_student_model(student_atac_cells.to(device=device), modality=1)

## Setup teachers
args = SimpleNamespace(
    source_dataset=None,
    target_dataset=target_dataset,
    genes_by_peaks_str=None,
    ignore_sources=['Midbrain_Adams', 'pbmc_10x', 'PFC_Zhu'],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)
datasets, models, teacher_rna_train_loaders, teacher_atac_train_loaders, teacher_rna_valid_loaders, teacher_atac_valid_loaders = \
    teachers_setup(model_uri_paths, args, device)


def mean_cosine_similarities(rna_latents, atac_latents, rna_celltypes, atac_celltypes):

    rna_similarity = torch.matmul(rna_latents, rna_latents.T).cpu().detach().numpy()
    atac_similarity = torch.matmul(atac_latents, atac_latents.T).cpu().detach().numpy()

    rna_celltypes = np.array(rna_celltypes)
    atac_celltypes = np.array(atac_celltypes)

    df_rna = pd.DataFrame({
        'rna_celltype_x': np.repeat(rna_celltypes, len(rna_celltypes)),
        'rna_celltype_y': np.tile(rna_celltypes, len(rna_celltypes)),
        'rna_similarity': rna_similarity.flatten(),
    })

    df_atac = pd.DataFrame({
        'atac_celltype_x': np.repeat(atac_celltypes, len(atac_celltypes)),
        'atac_celltype_y': np.tile(atac_celltypes, len(atac_celltypes)),
        'atac_similarity': atac_similarity.flatten()
    })

    mean_cosine_similarity_by_label_rna = df_rna.groupby(['rna_celltype_x', 'rna_celltype_y'])['rna_similarity'].mean().unstack(fill_value=float('nan'))
    mean_cosine_similarity_by_label_atac = df_atac.groupby(['atac_celltype_x', 'atac_celltype_y'])['atac_similarity'].mean().unstack(fill_value=float('nan'))
    
    return mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac

fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))

## get mean cosine similarity between teachers and student
mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(student_rna_latents, student_atac_latents, student_rna_labels, student_atac_labels)
sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax1[0, 0]); ax1[0, 0].set_xlabel(''); ax1[0, 0].set_ylabel('')
sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax1[1, 0]); ax1[1, 0].set_xlabel(''); ax1[1, 0].set_ylabel('')
ax1[0, 0].set_title(f'Student: {target_dataset}'); ax1[0, 0].set_ylabel('RNA'); ax1[1, 0].set_ylabel('ATAC')
sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax2[0, 0]); ax2[0, 0].set_xlabel(''); ax2[0, 0].set_ylabel('')
sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax2[1, 0]); ax2[1, 0].set_xlabel(''); ax2[1, 0].set_ylabel('')
ax2[0, 0].set_title(f'Student: {target_dataset}'); ax2[0, 0].set_ylabel('RNA'); ax2[1, 0].set_ylabel('ATAC')

## get data & latents
teachers_rna_latents = {}
teachers_atac_latents = {}

for s, source_dataset in enumerate(source_datasets):

    rna_valid_loader = teacher_rna_valid_loaders[source_dataset]
    atac_valid_loader = teacher_atac_valid_loaders[source_dataset]
    model = models[source_dataset]
    
    teacher_rna_cells, teacher_rna_labels, teacher_rna_batches = fetch_data_from_loader_light(rna_valid_loader)
    teacher_atac_cells, teacher_atac_labels, teacher_atac_batches = fetch_data_from_loader_light(atac_valid_loader)

    ## project data through teacher model
    teacher_rna_latents, _ = model(teacher_rna_cells.to(device=device), modality=0)
    teacher_atac_latents, _ = model(teacher_atac_cells.to(device=device), modality=1)

    teachers_rna_latents[source_dataset] = teacher_rna_latents
    teachers_atac_latents[source_dataset] = teacher_atac_latents

    mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(teacher_rna_latents, teacher_atac_latents, teacher_rna_labels, teacher_atac_labels)
    sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax1[0, s + 1]); ax1[0, s + 1].set_xlabel(''); ax1[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax1[1, s + 1]); ax1[1, s + 1].set_xlabel(''); ax1[1, s + 1].set_ylabel('')
    ax1[0, s + 1].set_title(f'Teacher: {source_dataset}')

    ## project data through KD_CLIP student model
    kd_clip_rna_latents, _ = kd_clip_student_models[source_dataset](student_rna_cells.to(device=device), modality=0)
    kd_clip_atac_latents, _ = kd_clip_student_models[source_dataset](student_atac_cells.to(device=device), modality=1)
    mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(kd_clip_rna_latents, kd_clip_atac_latents, student_rna_labels, student_atac_labels)
    sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax2[0, s + 1]); ax2[0, s + 1].set_xlabel(''); ax2[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax2[1, s + 1]); ax2[1, s + 1].set_xlabel(''); ax2[1, s + 1].set_ylabel('')
    ax2[0, s + 1].set_title(f'Student (KD_CLIP): {source_dataset}')

fig1.tight_layout()
fig2.tight_layout()

#%%
from eclare.post_hoc_utils import plot_umap_embeddings, create_celltype_palette

color_map_ct = create_celltype_palette(teacher_rna_labels, teacher_atac_labels, plot_color_palette=False)

rna_condition = ['nan'] * len(teacher_rna_labels)
atac_condition = ['nan'] * len(teacher_atac_labels)

umap_embedding, umap_figure, _ = plot_umap_embeddings(teacher_rna_latents.cpu().detach().numpy(), teacher_atac_latents.cpu().detach().numpy(), teacher_rna_labels, teacher_atac_labels, rna_condition, atac_condition, color_map_ct, umap_embedding=None)

#%% project data through teacher models and student model

## project data through teacher models
for source_dataset in source_datasets:
    next(teacher_rna_valid_loaders[source_dataset])
    teacher_atac_valid_loaders[source_dataset]





# %%
