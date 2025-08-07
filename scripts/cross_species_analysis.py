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

    eclare_root = os.environ.get('ECLARE_ROOT')
    outpath = os.environ.get('OUTPATH')
    datapath = os.environ.get('DATAPATH')

#%%
import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns

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
from glob import glob

from eclare.post_hoc_utils import \
    extract_target_source_replicate, metric_boxplots, get_next_version_dir, load_model_and_metadata, \
    set_env_variables, download_mlflow_runs

from eclare.setup_utils import teachers_setup, return_setup_func_from_dataset
from eclare.data_utils import fetch_data_from_loader_light


set_env_variables()

cuda_available = torch.cuda.is_available()
n_cudas = torch.cuda.device_count()
device = torch.device(f'cuda:{n_cudas - 1}') if cuda_available else 'cpu'

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '05151710',
    'kd_clip': '06082442',
    'eclare': ['06083053'],
}

## define search strings
search_strings = {
    'clip': 'CLIP' + '_' + methods_id_dict['clip'],
    'kd_clip': 'KD_CLIP' + '_' + methods_id_dict['kd_clip'],
    'eclare': ['ECLARE' + '_' + job_id for job_id in methods_id_dict['eclare']]
}

## for ECLARE, map search_strings to 'dataset' column
dataset_column = [
    'eclare',
    ]
search_strings_to_dataset = {
    'ECLARE' + '_' + job_id: dataset_column[j] for j, job_id in enumerate(methods_id_dict['eclare'])
}

## Create output directory with version counter
base_output_dir = os.path.join(os.environ['OUTPATH'], f"cross_species_analysis_{methods_id_dict['eclare'][0]}")
output_dir = get_next_version_dir(base_output_dir)
os.makedirs(output_dir, exist_ok=True)

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
#combined_metrics_df = combined_metrics_df[combined_metrics_df['status'] == 'FINISHED']

## plot boxplots
#metric_boxplots(combined_metrics_df.loc[combined_metrics_df['dataset'].isin(['eclare', 'kd_clip', 'clip'])])
metric_boxplots(
    combined_metrics_df, target_source_combinations=True, include_paired=True
    )

if len(methods_id_dict['eclare']) > 1:
    metric_boxplots(pd.concat([*eclare_dfs.values()]))


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
model_uri_paths_str = f"clip_*{methods_id_dict['clip']}/{target_dataset}/**/{best_eclare}/model_uri.txt"
model_uri_paths = glob(os.path.join(outpath, model_uri_paths_str))

genes_by_peaks_path = os.path.join(os.environ['DATAPATH'], 'genes_by_peaks_str.csv')
genes_by_peaks_df = pd.read_csv(genes_by_peaks_path, index_col=0)

## Setup student
student_setup_func = return_setup_func_from_dataset(target_dataset)
#genes_by_peaks_str = genes_by_peaks_df.loc[target_dataset, 'MDD']

args = SimpleNamespace(
    source_dataset=target_dataset,
    target_dataset=None,
    genes_by_peaks_str='17987_by_127358',
    ignore_sources=['Midbrain_Adams', 'pbmc_10x', 'PFC_Zhu'],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)

student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_total_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_total_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
    student_setup_func(args, return_type='loaders')

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

## approximate number of cells to subsample
subsample = 2000

## project data through student model
student_rna_cells, student_rna_labels, student_rna_batches = fetch_data_from_loader_light(student_rna_valid_loader, subsample=subsample, shuffle=False)
student_atac_cells, student_atac_labels, student_atac_batches = fetch_data_from_loader_light(student_atac_valid_loader, subsample=subsample, shuffle=False)
student_rna_latents, _ = eclare_student_model(student_rna_cells.to(device=device), modality=0)
student_atac_latents, _ = eclare_student_model(student_atac_cells.to(device=device), modality=1)

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

## init dict to save specific similarity combinations
celltypes_list = ['OPCs', 'Astrocytes']
df_sim = pd.DataFrame(index=['student', f'teacher ({source_datasets[0]})', f'teacher ({source_datasets[1]})'], columns=['RNA', 'ATAC'])

fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))

## get mean cosine similarity between teachers and student
mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(student_rna_latents, student_atac_latents, student_rna_labels, student_atac_labels)
df_sim.loc['student', 'RNA'] = mean_cosine_similarity_by_label_rna.loc[celltypes_list[0], celltypes_list[1]]
df_sim.loc['student', 'ATAC'] = mean_cosine_similarity_by_label_atac.loc[celltypes_list[0], celltypes_list[1]]

sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax1[0, 0]); ax1[0, 0].set_xlabel(''); ax1[0, 0].set_ylabel('')
sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax1[1, 0]); ax1[1, 0].set_xlabel(''); ax1[1, 0].set_ylabel('')
ax1[0, 0].set_title(f'Student: {target_dataset}'); ax1[0, 0].set_ylabel('RNA'); ax1[1, 0].set_ylabel('ATAC')
sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax2[0, 0]); ax2[0, 0].set_xlabel(''); ax2[0, 0].set_ylabel('')
sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax2[1, 0]); ax2[1, 0].set_xlabel(''); ax2[1, 0].set_ylabel('')
ax2[0, 0].set_title(f'Student: {target_dataset}'); ax2[0, 0].set_ylabel('RNA'); ax2[1, 0].set_ylabel('ATAC')

## get data & latents
all_teachers_rna_latents = {}
all_teachers_atac_latents = {}
all_kd_clip_rna_latents = {}
all_kd_clip_atac_latents = {}

for s, source_dataset in enumerate(source_datasets):

    ## get teacher loaders and model
    rna_valid_loader = teacher_rna_valid_loaders[source_dataset]
    atac_valid_loader = teacher_atac_valid_loaders[source_dataset]
    model = models[source_dataset]
    
    ## get teacher data
    teacher_rna_cells, teacher_rna_labels, teacher_rna_batches = fetch_data_from_loader_light(rna_valid_loader, subsample=subsample, shuffle=False)
    teacher_atac_cells, teacher_atac_labels, teacher_atac_batches = fetch_data_from_loader_light(atac_valid_loader, subsample=subsample, shuffle=False)

    ## project data through teacher model to get latents
    teacher_rna_latents, _ = model(teacher_rna_cells.to(device=device), modality=0)
    teacher_atac_latents, _ = model(teacher_atac_cells.to(device=device), modality=1)

    all_teachers_rna_latents[source_dataset] = teacher_rna_latents
    all_teachers_atac_latents[source_dataset] = teacher_atac_latents

    ## get mean cosine similarity between cell types
    mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(teacher_rna_latents, teacher_atac_latents, teacher_rna_labels, teacher_atac_labels)
    df_sim.loc[f'teacher ({source_dataset})', 'RNA'] = mean_cosine_similarity_by_label_rna.loc[celltypes_list[0], celltypes_list[1]]
    df_sim.loc[f'teacher ({source_dataset})', 'ATAC'] = mean_cosine_similarity_by_label_atac.loc[celltypes_list[0], celltypes_list[1]]

    sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax1[0, s + 1]); ax1[0, s + 1].set_xlabel(''); ax1[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax1[1, s + 1]); ax1[1, s + 1].set_xlabel(''); ax1[1, s + 1].set_ylabel('')
    ax1[0, s + 1].set_title(f'Teacher: {source_dataset}')

    ## project data through KD_CLIP student model to get latents
    kd_clip_rna_latents, _ = kd_clip_student_models[source_dataset](student_rna_cells.to(device=device), modality=0)
    kd_clip_atac_latents, _ = kd_clip_student_models[source_dataset](student_atac_cells.to(device=device), modality=1)

    all_kd_clip_rna_latents[source_dataset] = kd_clip_rna_latents
    all_kd_clip_atac_latents[source_dataset] = kd_clip_atac_latents

    mean_cosine_similarity_by_label_rna, mean_cosine_similarity_by_label_atac = mean_cosine_similarities(kd_clip_rna_latents, kd_clip_atac_latents, student_rna_labels, student_atac_labels)
    sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax2[0, s + 1]); ax2[0, s + 1].set_xlabel(''); ax2[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax2[1, s + 1]); ax2[1, s + 1].set_xlabel(''); ax2[1, s + 1].set_ylabel('')
    ax2[0, s + 1].set_title(f'Student (KD_CLIP): {source_dataset}')

fig1.suptitle(f'Mean cosine similarity between cell types - {target_dataset}')
fig1.tight_layout()

fig2.suptitle(f'Mean cosine similarity between cell types - {target_dataset}')
fig2.tight_layout()

# Prepare data for barplot: melt the DataFrame to long format for seaborn
df_sim_reset = df_sim.reset_index().rename(columns={'index': 'Model'})
df_sim_melted = df_sim_reset.melt(id_vars='Model', value_vars=['RNA', 'ATAC'], 
                                  var_name='Modality', value_name='Mean Cosine Similarity')

# Create a barplot comparing teachers and students for RNA and ATAC separately
plt.figure(figsize=(8, 5))
sns.barplot(data=df_sim_melted, x='Modality', y='Mean Cosine Similarity', hue='Model')
plt.title(f'Similarity between {celltypes_list[0]} and {celltypes_list[1]}')
plt.ylabel('Mean Cosine Similarity')
plt.xlabel('Modality')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% align embeddings with procrustes for unified UMAP
from eclare.post_hoc_utils import plot_umap_embeddings, create_celltype_palette

color_map_ct = create_celltype_palette(student_rna_labels, student_atac_labels, plot_color_palette=False)
celltypes_list=['Astrocytes', 'OPCs']

from scipy.linalg import orthogonal_procrustes

def center(X): return X - X.mean(axis=0)

def align_embeddings(ref, others_rna, others_atac):

    ref = ref.cpu().detach().numpy()
    ref = center(ref)
    
    aligned_others_rna = []
    for emb in others_rna:
        emb = emb.cpu().detach().numpy()
        cand = center(emb)
        R, _ = orthogonal_procrustes(cand, ref)
        aligned_others_rna.append(cand.dot(R))

    aligned_others_atac = []
    for emb in others_atac:
        emb = emb.cpu().detach().numpy()
        cand = center(emb)
        R, _ = orthogonal_procrustes(cand, ref)
        aligned_others_atac.append(cand.dot(R))

    return ref, aligned_others_rna, aligned_others_atac

others_rna = \
    [all_kd_clip_rna_latents[source_dataset] for source_dataset in source_datasets] + \
    [all_teachers_rna_latents[source_dataset] for source_dataset in source_datasets]

others_atac = \
    [all_kd_clip_atac_latents[source_dataset] for source_dataset in source_datasets] + \
    [all_teachers_atac_latents[source_dataset] for source_dataset in source_datasets]

ref, aligned_others_rna, aligned_others_atac = align_embeddings(student_rna_latents, others_rna, others_atac)

all_rna_latents = np.concatenate([ref, *aligned_others_rna], axis=0)
all_atac_latents = np.concatenate([ref, *aligned_others_atac], axis=0)

assert student_rna_labels == teacher_rna_labels
all_rna_labels = np.tile(student_rna_labels, 5)

assert student_atac_labels == teacher_atac_labels
all_atac_labels = np.tile(student_atac_labels, 5)

all_rna_conditions = \
    ['eclare_student'] * len(student_rna_labels) + \
    ['kd_clip_student'] * len(student_rna_labels) + \
    ['kd_clip_student'] * len(student_rna_labels) + \
    ['teacher'] * len(teacher_rna_labels) + \
    ['teacher'] * len(teacher_rna_labels)

all_atac_conditions = \
    ['eclare_student'] * len(student_atac_labels) + \
    ['kd_clip_student'] * len(student_atac_labels) + \
    ['kd_clip_student'] * len(student_atac_labels) + \
    ['teacher'] * len(teacher_atac_labels) + \
    ['teacher'] * len(teacher_atac_labels)

'''
all_rna_latents = np.concatenate([teacher_rna_latents.cpu().detach().numpy(), kd_clip_rna_latents.cpu().detach().numpy(), student_rna_latents.cpu().detach().numpy()], axis=0)
all_atac_latents = np.concatenate([teacher_atac_latents.cpu().detach().numpy(), kd_clip_atac_latents.cpu().detach().numpy(), student_atac_latents.cpu().detach().numpy()], axis=0)
all_rna_labels = np.concatenate([teacher_rna_labels, teacher_rna_labels, student_rna_labels], axis=0)
all_atac_labels = np.concatenate([teacher_atac_labels, teacher_atac_labels, student_atac_labels], axis=0)
all_rna_conditions = ['teacher'] * len(teacher_rna_labels) + ['kd_clip_student'] * len(teacher_rna_labels) + ['eclare_student'] * len(student_rna_labels)
'''

umap_embedding, umap_figure, _ = plot_umap_embeddings(all_rna_latents, all_atac_latents, all_rna_labels, all_atac_labels, all_rna_conditions, all_rna_conditions, color_map_ct, celltypes_list=celltypes_list)

_, umap_figure, _ = plot_umap_embeddings(teacher_rna_latents.cpu().detach().numpy(), teacher_atac_latents.cpu().detach().numpy(), teacher_rna_labels, teacher_atac_labels, ['nan'] * len(teacher_rna_labels), ['nan'] * len(teacher_atac_labels), color_map_ct, celltypes_list=celltypes_list, umap_embedding=umap_embedding)
_, umap_figure, _ = plot_umap_embeddings(kd_clip_rna_latents.cpu().detach().numpy(), kd_clip_atac_latents.cpu().detach().numpy(), teacher_rna_labels, teacher_atac_labels, ['nan'] * len(teacher_rna_labels), ['nan'] * len(teacher_atac_labels), color_map_ct, celltypes_list=celltypes_list, umap_embedding=umap_embedding)
_, umap_figure, _ = plot_umap_embeddings(student_rna_latents.cpu().detach().numpy(), student_atac_latents.cpu().detach().numpy(), student_rna_labels, student_atac_labels, ['nan'] * len(student_rna_labels), ['nan'] * len(student_atac_labels), color_map_ct, celltypes_list=celltypes_list, umap_embedding=umap_embedding)

#%% cell-type level losses
from eclare.losses_and_distances_utils import clip_loss, clip_loss_split_by_ct

# Student
student_loss_df = clip_loss(None, student_atac_labels, student_rna_labels, student_atac_latents, student_rna_latents, temperature=1, do_ct=True)
student_split_by_ct_loss_df = clip_loss_split_by_ct(student_atac_latents, student_rna_latents, student_atac_labels, student_rna_labels, temperature=1)

# Teacher 1 (original)
teacher1_loss_df = clip_loss(None, teacher_atac_labels, teacher_rna_labels, teacher_atac_latents, teacher_rna_latents, temperature=1, do_ct=True)
teacher1_split_by_ct_loss_df = clip_loss_split_by_ct(teacher_atac_latents, teacher_rna_latents, teacher_atac_labels, teacher_rna_labels, temperature=1)

# Teacher 2 (kd_clip)
teacher2_loss_df = clip_loss(None, teacher_atac_labels, teacher_rna_labels, kd_clip_atac_latents, kd_clip_rna_latents, temperature=1, do_ct=True)
teacher2_split_by_ct_loss_df = clip_loss_split_by_ct(kd_clip_atac_latents, kd_clip_rna_latents, teacher_atac_labels, teacher_rna_labels, temperature=1)

# Combine all losses
teacher1_label = f"teacher  ({source_datasets[0]})"
teacher2_label = f"teacher ({source_datasets[1]})"
teacher1_loss_atac_label = f"teacher_loss_atac ({source_datasets[0]})"
teacher1_loss_rna_label = f"teacher_loss_rna ({source_datasets[0]})"
teacher2_loss_atac_label = f"teacher_loss_atac ({source_datasets[1]})"
teacher2_loss_rna_label = f"teacher_loss_rna ({source_datasets[1]})"

all_loss_df = pd.concat([
    pd.concat(student_loss_df[-2:], axis=1),
    pd.concat(teacher1_loss_df[-2:], axis=1),
    pd.concat(teacher2_loss_df[-2:], axis=1)
], axis=1)
all_loss_df.columns = [
    'student_loss_atac', 'student_loss_rna',
    teacher1_loss_atac_label, teacher1_loss_rna_label,
    teacher2_loss_atac_label, teacher2_loss_rna_label
]

all_loss_df['student_loss'] = 0.5 * (all_loss_df['student_loss_atac'] + all_loss_df['student_loss_rna'])
all_loss_df[teacher1_label] = 0.5 * (all_loss_df[teacher1_loss_atac_label] + all_loss_df[teacher1_loss_rna_label])
all_loss_df[teacher2_label] = 0.5 * (all_loss_df[teacher2_loss_atac_label] + all_loss_df[teacher2_loss_rna_label])

ALL_student_loss = all_loss_df.loc['ALL', 'student_loss']
ALL_teacher1_loss = all_loss_df.loc['ALL', teacher1_label]
ALL_teacher2_loss = all_loss_df.loc['ALL', teacher2_label]

# Plot as grouped barplot
fig, ax = plt.subplots(figsize=(14, 6))
all_loss_df[['student_loss', teacher1_label, teacher2_label]].plot(kind='bar', ax=ax)
ax.set_ylabel('Mean Loss')
ax.set_xlabel('Cell Type')
ax.set_title('Mean Student and Teacher CLIP Losses by Cell Type')
ax.legend(title='Loss Type', loc='lower left')

ax.axhline(ALL_student_loss, color='blue', linestyle=':', linewidth=2, label='Mean Student Loss (All)')
ax.axhline(ALL_teacher1_loss, color='orange', linestyle=':', linewidth=2, label=f'Mean Teacher1 Loss (All) [{source_datasets[0]}]')
ax.axhline(ALL_teacher2_loss, color='green', linestyle=':', linewidth=2, label=f'Mean Teacher2 Loss (All) [{source_datasets[1]}]')

plt.tight_layout()
plt.show()
