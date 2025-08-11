#%% set env variables
from eclare import set_env_variables
set_env_variables(config_path='../config')

#%%
import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns

# Set matplotlib to use a thread-safe backend
import matplotlib
matplotlib.use('inline')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
import pickle

from eclare.post_hoc_utils import \
    extract_target_source_replicate, metric_boxplots, get_next_version_dir, load_model_and_metadata, \
    download_mlflow_runs

from eclare.setup_utils import teachers_setup, return_setup_func_from_dataset
from eclare.data_utils import fetch_data_from_loader_light

def mean_cosine_similarities(x, y, x_celltypes, y_celltypes):

    similarity = torch.matmul(x, y.T).cpu().detach().numpy()

    x_celltypes = np.array(x_celltypes)
    y_celltypes = np.array(y_celltypes)

    df_rna = pd.DataFrame({
        'celltype_x': np.repeat(x_celltypes, len(x_celltypes)),
        'celltype_y': np.tile(y_celltypes, len(y_celltypes)),
        'similarity': similarity.flatten(),
    })

    mean_cosine_similarity_by_label = df_rna.groupby(['celltype_x', 'celltype_y'])['similarity'].mean().unstack(fill_value=float('nan'))
    
    return mean_cosine_similarity_by_label

def get_cmap(name: str = "light_pubugn") -> LinearSegmentedColormap:
    """
    Return a PuBuGn-inspired colormap where:
      - 0 is near-white cream,
      - 0.01 is pale lavender,
      - values ramp through blue to teal-green.
    """
    stops = [
        (0.00, "#fffff0"),  # very light cream
        (0.05, "#ece2f0"),  # pale lavender
        (0.20, "#a6bddb"),  # light blue
        (0.50, "#3690c0"),  # mid blue
        (0.75, "#1c9099"),  # teal
        (1.00, "#016c59"),  # deep green-teal
    ]
    return LinearSegmentedColormap.from_list(name, stops)

## approximate number of cells to subsample
subsample = 5000

cmap = get_cmap('light_pubugn')

cuda_available = torch.cuda.is_available()
n_cudas = torch.cuda.device_count()
device = torch.device(f'cuda:{n_cudas - 1}') if cuda_available else 'cpu'

## Create dict for methods and job_ids
methods_id_dict = {
    'clip': '09114308',
    'kd_clip': '09140533',
    'eclare': ['10165959'],
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

#%% get BICCN similarity matrices

sim_mat_path = os.path.join(os.environ['DATAPATH'], 'biccn_dend_to_matrix.pkl')
with open(sim_mat_path, 'rb') as f:
    sim_mat_dict, celltype_map_dict = pickle.load(f)

# Compute the mean of the DataFrames in sim_mat_dict, keeping the DataFrame format
sim_mat_mean_df = pd.concat(sim_mat_dict.values()).groupby(level=0).mean()

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
source_datasets = ['DLPFC_Ma', 'PFC_Zhu']

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


# %% Setup student
model_uri_paths_str = f"clip_*{methods_id_dict['clip']}/{target_dataset}/**/{best_eclare}/model_uri.txt"
model_uri_paths = glob(os.path.join(os.environ['OUTPATH'], model_uri_paths_str))

## Get student loaders
student_setup_func = return_setup_func_from_dataset(target_dataset)
#genes_by_peaks_str = genes_by_peaks_df.loc[target_dataset, 'MDD']

args = SimpleNamespace(
    source_dataset=target_dataset,
    target_dataset=None,
    genes_by_peaks_str='17987_by_127358',
    ignore_sources=[None],
    source_dataset_embedder=None,
    batch_size=1000,
    total_epochs=0,
)

student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_total_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_total_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
    student_setup_func(args, return_type='loaders')

#%% compare with baselines

from scglue.genomics import Bed, window_graph
from scglue.data import get_gene_annotation
import anndata
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import scanpy as sc

import muon as mu
import mojitoo
import episcanpy as epi

## get target dataset
rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = \
    student_setup_func(args, return_type='data', hvg_only=True)

## convert peaks to gene activity scores
gene_ad = anndata.AnnData(var=pd.DataFrame(index=rna.var_names.to_list()))
get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')
gene_ad = gene_ad[:,gene_ad.var['chromStart'].notna()]
rna = rna[:,gene_ad.var_names]

peak_coords = atac.var['interval'].str.split('[:-]', expand=True).rename(columns={0: 'chrom', 1: 'chromStart', 2: 'chromEnd'}).astype({'chrom': 'category', 'chromStart': 'int32', 'chromEnd': 'int32'})

genes_bed = Bed(gene_ad.var.assign(name=gene_ad.var_names)).expand(2e3, 0)
peaks_bed = Bed(peak_coords.assign(name=atac.var_names))

graph = window_graph(
    genes_bed,
    peaks_bed,
    window_size=0, attr_fn=lambda l, r, d: {"weight": 1.0, "sign": 1}
)

biadj = nx.algorithms.bipartite.biadjacency_matrix(graph, gene_ad.var.index, peak_coords.index)
atac2rna = anndata.AnnData(X=atac.X @ biadj.T, obs=atac.obs, var=rna.var, uns=atac.uns)

## run scanpy PCA pipeline to RNA and ATAC2RNA
sc.tl.pca(rna, svd_solver='arpack')
sc.tl.pca(atac2rna, svd_solver='arpack')

## plot umap of PCA embeddings
sc.pp.neighbors(rna)
sc.tl.umap(rna)
sc.pl.umap(rna, color=cell_group)

sc.pp.neighbors(atac2rna)
sc.tl.umap(atac2rna)
sc.pl.umap(atac2rna, color=cell_group)

## subsample data
n_cells = rna.shape[0]
n_splits = np.ceil(n_cells / subsample).astype(int)
skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
_, cells_idx = next(skf.split(np.zeros_like(rna.obs[cell_group].to_list()), rna.obs[cell_group].to_list()))

x_celltypes = rna[cells_idx].obs[cell_group].to_list()
y_celltypes = atac2rna[cells_idx].obs[cell_group].to_list()

x = torch.from_numpy(rna[cells_idx].obsm['X_pca']).to(device=device)
y = torch.from_numpy(atac2rna[cells_idx].obsm['X_pca']).to(device=device)

x = torch.nn.functional.normalize(x, dim=1)
y = torch.nn.functional.normalize(y, dim=1)

## get mean cosine similarity between cell types
pca_sim_rna = mean_cosine_similarities(x, x, x_celltypes, x_celltypes)
pca_sim_atac = mean_cosine_similarities(y, y, y_celltypes, y_celltypes)
pca_sim_rna_atac = mean_cosine_similarities(x, y, x_celltypes, y_celltypes)

## MOFA
mdata = mu.MuData({'rna': rna[cells_idx], 'atac': atac2rna[cells_idx]})
mu.tl.mofa(mdata)
x_mofa = mdata.obsm['X_mofa']
x_mofa = torch.from_numpy(x_mofa).to(device=device)
x_mofa = torch.nn.functional.normalize(x_mofa, dim=1)
mofa_sim_rna_atac = mean_cosine_similarities(x_mofa, x_mofa, x_celltypes, x_celltypes)
sns.heatmap(mofa_sim_rna_atac, cmap='PuBuGn'); plt.title('MOFA')

## MOJITOO
def mojitoo_process_data(rna, atac):
    ## Preprocess ATAC data (MOJITOO documentation)
    epi.pp.cal_var(atac)
    epi.pp.select_var_feature(atac, nb_features=5000)
    epi.tl.tfidf(atac)
    epi.tl.lsi(atac, n_components=50)

    ## Preprocess RNA data (MOJITOO documentation)
    sc.pp.normalize_total(rna, target_sum=1e4)
    #sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, svd_solver='arpack')

    ## Create MuData object
    mdata = mu.MuData({"atac": atac, "rna": rna})
    mdata.obsm["pca"] = rna.obsm["X_pca"]
    mdata.obsm["lsi"] = atac.obsm["X_lsi"]

    return mdata

mdata = mojitoo_process_data(rna, atac2rna)
mdata = mdata[cells_idx].copy()
mojitoo.mojitoo(mdata, reduction_list=["pca", "lsi"],  dims_list=(range(50), range(1,50)),reduction_name='mojitoo', overwrite=True)
x_mojitoo = mdata.obsm['mojitoo']
x_mojitoo = torch.from_numpy(x_mojitoo).to(device=device)
x_mojitoo = torch.nn.functional.normalize(x_mojitoo, dim=1)
mojitoo_sim_rna_atac = mean_cosine_similarities(x_mojitoo, x_mojitoo, x_celltypes, x_celltypes)
sns.heatmap(mojitoo_sim_rna_atac, cmap=cmap); plt.title('MOJITOO')

#%% extract student latents for analysis

## project data through student model
student_rna_cells, student_rna_labels, student_rna_batches = fetch_data_from_loader_light(student_rna_valid_loader, subsample=subsample, shuffle=False)
student_atac_cells, student_atac_labels, student_atac_batches = fetch_data_from_loader_light(student_atac_valid_loader, subsample=subsample, shuffle=False)

student_rna_latents, _ = eclare_student_model(student_rna_cells.to(device=device), modality=0)
student_atac_latents, _ = eclare_student_model(student_atac_cells.to(device=device), modality=1)

## init dict to save specific similarity combinations
celltypes_list = ['OPCs', 'Astrocytes']
df_sim = pd.DataFrame(index=['student', f'teacher ({source_datasets[0]})', f'teacher ({source_datasets[1]})'], columns=['RNA', 'ATAC'])

## get mean cosine similarity between teachers and student
eclare_sim_rna = mean_cosine_similarities(student_rna_latents, student_rna_latents, student_rna_labels, student_rna_labels)
eclare_sim_atac = mean_cosine_similarities(student_atac_latents, student_atac_latents, student_atac_labels, student_atac_labels)
eclare_sim_rna_atac = mean_cosine_similarities(student_rna_latents, student_atac_latents, student_rna_labels, student_atac_labels)

df_sim.loc['student', 'RNA'] = eclare_sim_rna.loc[celltypes_list[0], celltypes_list[1]]
df_sim.loc['student', 'ATAC'] = eclare_sim_atac.loc[celltypes_list[0], celltypes_list[1]]
df_sim.loc['student', 'RNA_ATAC'] = eclare_sim_rna_atac.loc[celltypes_list[0], celltypes_list[1]]


#%% ECLARE & teachers plot

## Setup teachers
if not ('datasets' in locals() or 'datasets' in globals()):

    args = SimpleNamespace(
        source_dataset=None,
        target_dataset=target_dataset,
        genes_by_peaks_str=None,
        ignore_sources=[None],
        source_dataset_embedder=None,
        batch_size=1000,
        total_epochs=0,
    )
    datasets, models, teacher_rna_train_loaders, teacher_atac_train_loaders, teacher_rna_valid_loaders, teacher_atac_valid_loaders = \
        teachers_setup(model_uri_paths, args, device)

## init figures
fig1, ax1 = plt.subplots(3, len(source_datasets) + 1, figsize=(4*(len(source_datasets) + 1), 12))

sns.heatmap(eclare_sim_rna, cmap=cmap, ax=ax1[0, 0]); ax1[0, 0].set_xlabel(''); ax1[0, 0].set_ylabel('')
sns.heatmap(eclare_sim_atac, cmap=cmap, ax=ax1[1, 0]); ax1[1, 0].set_xlabel(''); ax1[1, 0].set_ylabel('')
sns.heatmap(eclare_sim_rna_atac, cmap=cmap, ax=ax1[2, 0]); ax1[2, 0].set_xlabel(''); ax1[2, 0].set_ylabel('')
ax1[0, 0].set_title(f'Student: {target_dataset}'); ax1[0, 0].set_ylabel('RNA'); ax1[1, 0].set_ylabel('ATAC'); ax1[2, 0].set_ylabel('RNA & ATAC')

## get data & latents
all_teachers_rna_latents = {}
all_teachers_atac_latents = {}

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
    mean_cosine_similarity_by_label_rna = mean_cosine_similarities(teacher_rna_latents, teacher_rna_latents, teacher_rna_labels, teacher_rna_labels)
    mean_cosine_similarity_by_label_atac = mean_cosine_similarities(teacher_atac_latents, teacher_atac_latents, teacher_atac_labels, teacher_atac_labels)
    mean_cosine_similarity_by_label_rna_atac = mean_cosine_similarities(teacher_rna_latents, teacher_atac_latents, teacher_rna_labels, teacher_atac_labels)

    df_sim.loc[f'teacher ({source_dataset})', 'RNA'] = mean_cosine_similarity_by_label_rna.loc[celltypes_list[0], celltypes_list[1]]
    df_sim.loc[f'teacher ({source_dataset})', 'ATAC'] = mean_cosine_similarity_by_label_atac.loc[celltypes_list[0], celltypes_list[1]]
    df_sim.loc[f'teacher ({source_dataset})', 'RNA_ATAC'] = mean_cosine_similarity_by_label_rna_atac.loc[celltypes_list[0], celltypes_list[1]]

    sns.heatmap(mean_cosine_similarity_by_label_rna, cmap=cmap, ax=ax1[0, s + 1]); ax1[0, s + 1].set_xlabel(''); ax1[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, cmap=cmap, ax=ax1[1, s + 1]); ax1[1, s + 1].set_xlabel(''); ax1[1, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_rna_atac, cmap=cmap, ax=ax1[2, s + 1]); ax1[2, s + 1].set_xlabel(''); ax1[2, s + 1].set_ylabel('')
    ax1[0, s + 1].set_title(f'Teacher: {source_dataset}')

fig1.suptitle(f'Mean cosine similarity between cell types - {target_dataset}')
fig1.tight_layout()

## erase from device data used to produce plots
del rna_valid_loader, atac_valid_loader, model, teacher_rna_cells, teacher_atac_cells, teacher_rna_latents, teacher_atac_latents, teacher_rna_labels, teacher_atac_labels, teacher_rna_batches, teacher_atac_batches
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#%% Get source datasets

genes_by_peaks_path = os.path.join(os.environ['DATAPATH'], 'genes_by_peaks_full_str.csv')
genes_by_peaks_df = pd.read_csv(genes_by_peaks_path, index_col=0)

celltype_proportions_dict = {}

print(f'Processing {target_dataset}...')

setup_func = return_setup_func_from_dataset(target_dataset)
genes_by_peaks_str = '17987_by_127358'

args = SimpleNamespace(
    source_dataset=target_dataset,
    target_dataset=None,
    genes_by_peaks_str=genes_by_peaks_str,
)

rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = \
    setup_func(args, return_type='data')

celltype_proportions_rna = rna.obs[cell_group].value_counts(normalize=True)
celltype_proportions_atac = atac.obs[cell_group].value_counts(normalize=True)
assert celltype_proportions_rna.equals(celltype_proportions_atac)

celltype_proportions_dict[target_dataset] = celltype_proportions_rna

del rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath, celltype_proportions_rna, celltype_proportions_atac

for source_dataset in source_datasets:

    print(f'Processing {source_dataset}...')

    setup_func = return_setup_func_from_dataset(source_dataset)
    genes_by_peaks_str = genes_by_peaks_df.loc[source_dataset, target_dataset]

    args = SimpleNamespace(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        genes_by_peaks_str=genes_by_peaks_str,
    )

    rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath = \
        setup_func(args, return_type='data')

    celltype_proportions_rna = rna.obs[cell_group].value_counts(normalize=True)
    celltype_proportions_atac = atac.obs[cell_group].value_counts(normalize=True)
    assert celltype_proportions_rna.equals(celltype_proportions_atac)

    celltype_proportions_dict[source_dataset] = celltype_proportions_rna

    del rna, atac, cell_group, genes_to_peaks_binary_mask, genes_peaks_dict, atac_datapath, rna_datapath, celltype_proportions_rna, celltype_proportions_atac

## draw pie charts of celltype proportions
# Reference classes:
# ['Excitatory', 'Oligodendrocytes', 'Astrocytes', 'Inhibitory', 'OPCs', 'Microglia', 'Endothelial', 'Pericytes']

mapper_set1 = {
    'L2_3_IT': 'Excitatory',       # IT = intratelencephalic pyramidal neurons (layers 2/3) → excitatory
    'L3_5_IT_1': 'Excitatory',     # IT pyramidal subclasses (layers 3–5) → excitatory
    'L3_5_IT_2': 'Excitatory',     # same rationale as above
    'L3_5_IT_3': 'Excitatory',     # same rationale as above
    'L6_IT_1': 'Excitatory',       # layer 6 IT pyramidal → excitatory
    'L6_IT_2': 'Excitatory',       # layer 6 IT pyramidal → excitatory
    'L6_CT': 'Excitatory',         # corticothalamic pyramidal → excitatory
    'L6b': 'Excitatory',           # layer 6b pyramidal-like neurons → excitatory
    'L5_6_NP': 'Excitatory',       # near-projecting pyramidal subclass → excitatory
    'L5_PT': 'Excitatory',         # pyramidal tract (subcortical-projecting) → excitatory

    'VIP': 'Inhibitory',           # VIP is a canonical interneuron marker → inhibitory
    'SST': 'Inhibitory',           # somatostatin interneuron class → inhibitory
    'SST_NPY': 'Inhibitory',       # SST/NPY interneuron subclass → inhibitory
    'PVALB': 'Inhibitory',         # parvalbumin basket/chandelier lineage → inhibitory
    'PVALB_CHC': 'Inhibitory',     # chandelier cells (PVALB+) → inhibitory
    'LAMP5_RELN': 'Inhibitory',    # LAMP5/RELN neurogliaform/ivy interneurons → inhibitory
    'LAMP5_LHX6': 'Inhibitory',    # LAMP5 lineage interneuron variant → inhibitory
    'TH': 'Inhibitory',            # cortical TH+ (dopaminergic-like) interneurons → inhibitory
    'ADARB2': 'Inhibitory',        # CGE-related interneuron marker; map to inhibitory (flip if your dataset uses ADARB2 for excitatory L2 IT)

    'Oligo': 'Oligodendrocytes',   # short for oligodendrocytes
    'Astro': 'Astrocytes',         # short for astrocytes
    'OPC': 'OPCs',                 # oligodendrocyte precursor cells
    'Micro': 'Microglia',          # shorthand for microglia
    'immune': 'Microglia',         # brain-resident immune cluster typically microglia/macrophages → map to Microglia

    'Endo': 'Endothelial',         # endothelial cell shorthand

    'PC': 'Pericytes',             # pericyte shorthand → mural/perivascular class
    'SMC': 'Pericytes',            # smooth muscle cells grouped under mural/perivascular → pericytes bucket
    'VLMC': 'Pericytes',           # vascular leptomeningeal cells; coarse-grained as mural/perivascular → pericytes
}

mapper_set2 = {
    'Astrocytes': 'Astrocytes',    # identical label → same class
    'Oligodendrocytes': 'Oligodendrocytes',  # identical label → same class
    'OPC': 'OPCs',                 # oligodendrocyte precursor cells
    'Microglia': 'Microglia',      # identical label → same class

    'EN': 'Excitatory',            # EN = excitatory neurons → excitatory
    'IPC': 'Excitatory',           # intermediate progenitor cells (glutamatergic lineage) → excitatory bucket

    'IN-MGE': 'Inhibitory',        # MGE-derived interneurons (PVALB/SST lineages) → inhibitory
    'IN-CGE': 'Inhibitory',        # CGE-derived interneurons (VIP/LAMP5) → inhibitory

    'Endothelial': 'Endothelial',  # identical label → same class

    'Pericytes': 'Pericytes',      # identical label → same class
    'VSMC': 'Pericytes',           # vascular smooth muscle cells; mural/perivascular → pericytes bucket

    'RG': 'Astrocytes',            # radial glia; coarse harmonization with astroglial lineage
}

celltype_mapper = {**mapper_set1, **mapper_set2}
plot_source_datasets = [target_dataset] + [source_dataset for source_dataset in source_datasets if source_dataset != 'Midbrain_Adams']

fig1, ax1 = plt.subplots(1, len(plot_source_datasets), figsize=(10*len(plot_source_datasets), 10))
fig2, ax2 = plt.subplots(1, len(plot_source_datasets), figsize=(10*len(plot_source_datasets), 10))

for s, source_dataset in enumerate(plot_source_datasets):

    celltype_proportions = celltype_proportions_dict[source_dataset]
    celltype_proportions.plot.pie(autopct='%1.1f%%', ax=ax1[s])
    ax1[s].set_title(source_dataset)

    celltype_proportions_mapped = celltype_proportions.rename(celltype_mapper)
    celltype_proportions_mapped = celltype_proportions_mapped.groupby(celltype_proportions_mapped.index).sum()
    celltype_proportions_mapped.plot.pie(autopct='%1.1f%%', ax=ax2[s])
    ax2[s].set_title(source_dataset)


#%% ECLARE & KD-CLIP students plot

## init figures
fig2, ax2 = plt.subplots(3, len(source_datasets) + 1, figsize=(4*(len(source_datasets) + 1), 12))

sns.heatmap(eclare_sim_rna, cmap=cmap, ax=ax2[0, 0]); ax2[0, 0].set_xlabel(''); ax2[0, 0].set_ylabel('')
sns.heatmap(eclare_sim_atac, cmap=cmap, ax=ax2[1, 0]); ax2[1, 0].set_xlabel(''); ax2[1, 0].set_ylabel('')
sns.heatmap(eclare_sim_rna_atac, cmap=cmap, ax=ax2[2, 0]); ax2[2, 0].set_xlabel(''); ax2[2, 0].set_ylabel('')
ax2[0, 0].set_title(f'Student: {target_dataset}'); ax2[0, 0].set_ylabel('RNA'); ax2[1, 0].set_ylabel('ATAC'); ax2[2, 0].set_ylabel('RNA & ATAC')

## get data & latents
#all_kd_clip_rna_latents = {}
#all_kd_clip_atac_latents = {}

all_kd_sim_rna = {}
all_kd_sim_atac = {}
all_kd_sim_rna_atac = {}

for s, source_dataset in enumerate(source_datasets):

    ## project data through KD_CLIP student model to get latents
    kd_clip_rna_latents, _ = kd_clip_student_models[source_dataset](student_rna_cells.to(device=device), modality=0)
    kd_clip_atac_latents, _ = kd_clip_student_models[source_dataset](student_atac_cells.to(device=device), modality=1)

    #all_kd_clip_rna_latents[source_dataset] = kd_clip_rna_latents
    #all_kd_clip_atac_latents[source_dataset] = kd_clip_atac_latents

    mean_cosine_similarity_by_label_rna = mean_cosine_similarities(kd_clip_rna_latents, kd_clip_rna_latents, student_rna_labels, student_rna_labels)
    mean_cosine_similarity_by_label_atac = mean_cosine_similarities(kd_clip_atac_latents, kd_clip_atac_latents, student_atac_labels, student_atac_labels)
    mean_cosine_similarity_by_label_rna_atac = mean_cosine_similarities(kd_clip_rna_latents, kd_clip_atac_latents, student_rna_labels, student_atac_labels)

    all_kd_sim_rna_atac[source_dataset] = mean_cosine_similarity_by_label_rna_atac

    sns.heatmap(mean_cosine_similarity_by_label_rna, ax=ax2[0, s + 1], cmap=cmap); ax2[0, s + 1].set_xlabel(''); ax2[0, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_atac, ax=ax2[1, s + 1], cmap=cmap); ax2[1, s + 1].set_xlabel(''); ax2[1, s + 1].set_ylabel('')
    sns.heatmap(mean_cosine_similarity_by_label_rna_atac, ax=ax2[2, s + 1], cmap=cmap); ax2[2, s + 1].set_xlabel(''); ax2[2, s + 1].set_ylabel('')
    ax2[0, s + 1].set_title(f'Student (KD_CLIP): {source_dataset}')


fig2.suptitle(f'Mean cosine similarity between cell types - {target_dataset}')
fig2.tight_layout()

#%% Barplot of Spearman correlation between inferred similarities and BICCN reference
from scipy.stats import spearmanr

mean_kd_sim_rna_atac = pd.concat(all_kd_sim_rna_atac.values()).groupby(level=0).mean()

spearman_df = pd.DataFrame(index=['ECLARE', 'mean KD-CLIP', 'MOFA+', 'MOJITOO'], columns=['full', 'triangular'])

reference_sim = sim_mat_mean_df.values.flatten()

eclare_sim = eclare_sim_rna_atac.values.flatten()
mean_kd_clip_sim = mean_kd_sim_rna_atac.values.flatten()
mofa_sim = mofa_sim_rna_atac.values.flatten()
mojitoo_sim = mojitoo_sim_rna_atac.values.flatten()

spearman_df.loc['ECLARE', 'full'] = spearmanr(reference_sim, eclare_sim)[0]
spearman_df.loc['mean KD-CLIP', 'full'] = spearmanr(reference_sim, mean_kd_clip_sim)[0]
spearman_df.loc['MOFA+', 'full'] = spearmanr(reference_sim, mofa_sim)[0]
spearman_df.loc['MOJITOO', 'full'] = spearmanr(reference_sim, mojitoo_sim)[0]

triu_idxs = np.triu(np.ones(len(sim_mat_mean_df)), k=1).astype(bool)

reference_sim_triangular = sim_mat_mean_df.values[triu_idxs]
eclare_sim_triangular = 0.5 * (eclare_sim_rna_atac.values[triu_idxs] + eclare_sim_rna_atac.T.values[triu_idxs]) # since RNA-ATAC not symmetric
mean_kd_clip_sim_triangular = 0.5 * (mean_kd_sim_rna_atac.values[triu_idxs] + mean_kd_sim_rna_atac.T.values[triu_idxs]) # since RNA-ATAC not symmetric
mofa_sim_triangular = mofa_sim_rna_atac.values[triu_idxs]
mojitoo_sim_triangular = mojitoo_sim_rna_atac.values[triu_idxs]

spearman_df.loc['ECLARE', 'triangular'] = spearmanr(reference_sim_triangular, eclare_sim_triangular)[0]
spearman_df.loc['mean KD-CLIP', 'triangular'] = spearmanr(reference_sim_triangular, mean_kd_clip_sim_triangular)[0]
spearman_df.loc['MOFA+', 'triangular'] = spearmanr(reference_sim_triangular, mofa_sim_triangular)[0]
spearman_df.loc['MOJITOO', 'triangular'] = spearmanr(reference_sim_triangular, mojitoo_sim_triangular)[0]

## plot spearman correlations in barplot
spearman_df.T.plot.bar(figsize=(6, 5)); plt.ylabel('Spearman correlation')
plt.xlabel('portion of the similarity matrix')
plt.ylabel('Spearman correlation')
plt.title('Spearman correlation between inferred cell type similarities and BICCN reference')
plt.xticks(rotation=0)
plt.tight_layout()

#%% barplot

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

#%% load source datasets

for source_dataset in source_datasets:

    source_setup_func = return_setup_func_from_dataset(source_dataset)
    genes_by_peaks_str = genes_by_peaks_df.loc[source_dataset, target_dataset[0]]

    args = SimpleNamespace(
        source_dataset=source_dataset,
        target_dataset=target_dataset[0],
        genes_by_peaks_str=genes_by_peaks_str,
    )

    ## get data loaders
    rna, atac, cell_group, _, _, _, _ = source_setup_func(args, return_type='data')

    olig2_x = rna[:,rna.var_names == 'OLIG2'].X.toarray().flatten()
    gfap_x = rna[:,rna.var_names == 'GFAP'].X.toarray().flatten()
    X_df = pd.DataFrame({'OLIG2': olig2_x, 'GFAP': gfap_x, 'celltype': rna.obs[cell_group].values})
    
    X_trunc_df = X_df[X_df['celltype'].isin(['Astro','OPC'])].copy()
    X_trunc_df['celltype'] = X_trunc_df['celltype'].astype('category')
    X_trunc_df['celltype'] = X_trunc_df['celltype'].cat.remove_unused_categories()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='celltype', y='OLIG2', data=X_trunc_df, inner='box')
    plt.title(f"OLIG2 Expression by Cell Type in {source_dataset}")
    plt.xlabel("Cell Type")
    plt.ylabel("OLIG2 Expression")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='celltype', y='GFAP', data=X_trunc_df, inner='box')
    plt.title(f"GFAP Expression by Cell Type in {source_dataset}")
    plt.xlabel("Cell Type")
    plt.ylabel("GFAP Expression")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

umap_embedding, umap_figure, rna_atac_df_umap = plot_umap_embeddings(student_rna_latents.cpu().detach().numpy(), student_atac_latents.cpu().detach().numpy(), student_rna_labels, student_atac_labels, ['nan'] * len(student_rna_labels), ['nan'] * len(student_atac_labels), color_map_ct)
rna_atac_df_umap.drop(columns=['condition'], inplace=True)

olig2_flag = student_rna_valid_loader.dataset.adatas[0].var_names == 'OLIG2'
gfap_flag = student_rna_valid_loader.dataset.adatas[0].var_names == 'GFAP'

olig2_expr = student_rna_cells[:, olig2_flag].detach().cpu().numpy().flatten()
gfap_expr = student_rna_cells[:, gfap_flag].detach().cpu().numpy().flatten()

rna_df_umap = rna_atac_df_umap[rna_atac_df_umap['modality'] == 'RNA'].copy()
rna_df_umap['OLIG2_expr'] = np.log1p(olig2_expr)
rna_df_umap['GFAP_expr'] = np.log1p(gfap_expr)


# Plot UMAP embeddings colored by OLIG2, GFAP expression, and celltype in subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Determine shared colorbar limits for expression
vmin = min(rna_df_umap['OLIG2_expr'].min(), rna_df_umap['GFAP_expr'].min())
vmax = max(rna_df_umap['OLIG2_expr'].max(), rna_df_umap['GFAP_expr'].max())

# OLIG2 subplot
sc_olig2 = axes[0].scatter(
    rna_df_umap['umap_1'], rna_df_umap['umap_2'],
    c=rna_df_umap['OLIG2_expr'], cmap='magma', s=4, alpha=0.1, vmin=vmin, vmax=vmax
)
alpha_olig2 = (rna_df_umap['OLIG2_expr']/rna_df_umap['OLIG2_expr'].max())
sc_olig2 = axes[0].scatter(
    rna_df_umap['umap_1'], rna_df_umap['umap_2'],
    c=rna_df_umap['OLIG2_expr'], cmap='magma', s=25, alpha=alpha_olig2, vmin=vmin, vmax=vmax
)
axes[0].set_title("UMAP: RNA cells colored by OLIG2 expression")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# GFAP subplot
sc_gfap = axes[1].scatter(
    rna_df_umap['umap_1'], rna_df_umap['umap_2'],
    c=rna_df_umap['GFAP_expr'], cmap='magma', s=4, alpha=0.1, vmin=vmin, vmax=vmax
)
alpha_gfap = (rna_df_umap['GFAP_expr']/rna_df_umap['GFAP_expr'].max())
sc_gfap = axes[1].scatter(
    rna_df_umap['umap_1'], rna_df_umap['umap_2'],
    c=rna_df_umap['GFAP_expr'], cmap='magma', s=25, alpha=alpha_gfap, vmin=vmin, vmax=vmax
)
axes[1].set_title("UMAP: RNA cells colored by GFAP expression")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

# Celltype subplot
import matplotlib
celltypes = rna_df_umap['celltypes'].astype(str)
unique_celltypes = celltypes.unique()
# Use tab20 or tab10 if few celltypes, otherwise fallback to 'hsv'
if len(unique_celltypes) <= 10:
    cmap = plt.get_cmap('tab10')
elif len(unique_celltypes) <= 20:
    cmap = plt.get_cmap('tab20')
else:
    cmap = plt.get_cmap('hsv', len(unique_celltypes))
celltype_colors = {ct: cmap(i) for i, ct in enumerate(unique_celltypes)}
colors = celltypes.map(celltype_colors)

sc_celltype = axes[2].scatter(
    rna_df_umap['umap_1'], rna_df_umap['umap_2'],
    c=colors, s=25, alpha=0.8
)
axes[2].set_title("UMAP: RNA cells colored by celltype")
axes[2].set_xlabel("UMAP 1")
axes[2].set_ylabel("UMAP 2")

# Add colorbars to each expression subplot with same limits
cbar_olig2 = fig.colorbar(sc_olig2, ax=axes[0], fraction=0.046, pad=0.04)
cbar_olig2.set_label('Expression (log1p)')
cbar_gfap = fig.colorbar(sc_gfap, ax=axes[1], fraction=0.046, pad=0.04)
cbar_gfap.set_label('Expression (log1p)')

# Add legend for celltype subplot
handles = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=ct,
                                   markerfacecolor=celltype_colors[ct], markersize=8)
           for ct in unique_celltypes]
axes[2].legend(handles=handles, title='Celltype', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

