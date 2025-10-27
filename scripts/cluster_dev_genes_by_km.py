import scanpy as sc
import anndata
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore
from sklearn.cluster import KMeans
from pygam import GAM
from pygam.terms import s

from esda.moran import Moran
from libpysal.weights import WSP

sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
from eclare import set_env_variables
set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')

## load EN-only data from PFC Zhu
pfc_zhu_rna_EN = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'rna', 'pfc_zhu_rna_EN_ordinal.h5ad'))

## create graph from ordinal latents
sc.pp.neighbors(pfc_zhu_rna_EN, use_rep='X_ordinal_latents', key_added='X_ordinal_latents_neighbors', n_neighbors=30, n_pcs=10)
sc.tl.leiden(pfc_zhu_rna_EN, resolution=0.6, random_state=0, neighbors_key='X_ordinal_latents_neighbors')
sc.tl.umap(pfc_zhu_rna_EN, neighbors_key='X_ordinal_latents_neighbors')
sc.pl.umap(pfc_zhu_rna_EN, color=['Cell type', 'ordinal_pseudotime', 'leiden'], wspace=0.3, ncols=3)

## compute Moran's I for each gene
W = WSP(pfc_zhu_rna_EN.obsp['X_ordinal_latents_neighbors_connectivities'].tocsr())
W_dense = W.to_W()

# Convert to dense array once to avoid repeated conversion overhead
gene_expr = pfc_zhu_rna_EN.X.toarray() if hasattr(pfc_zhu_rna_EN.X, 'toarray') else pfc_zhu_rna_EN.X

# Initialize Ray - will use all available CPU cores by default
ray.init(num_cpus=40)

@ray.remote
def compute_moran_for_gene(g, gene, X, W_dense, permutations=0):
    """Compute Moran's I for a single gene."""
    # X is now dense, so just index directly
    x = X[:, g].flatten() if hasattr(X[:, g], 'flatten') else X[:, g].ravel()
    moran = Moran(x, W_dense, two_tailed=True, permutations=permutations)
    p_type = 'p_sim' if permutations > 0 else 'p_norm'
    moran_series = pd.Series(moran.__dict__, name=gene).loc[['I', p_type]]
    return moran_series

# Put data in Ray's object store so all workers can access it efficiently
X_ref = ray.put(gene_expr)
W_ref = ray.put(W_dense)

# Parallel computation with Ray
print(f"Computing Moran's I for {len(pfc_zhu_rna_EN.var_names)} genes with Ray...")
futures = [
    compute_moran_for_gene.remote(g, gene, X_ref, W_ref, permutations=1000)
    for g, gene in enumerate(pfc_zhu_rna_EN.var_names)
]

# Get results with progress bar
moran_series_list = []
for future in tqdm(futures, desc="Processing genes"):
    moran_series_list.append(ray.get(future))

moran_df = pd.concat(moran_series_list, axis=1)

# Add q-values (FDR-corrected p-values) using Benjamini-Hochberg method
p_type = moran_df.index[-1]
p_values = moran_df.loc[p_type].values
reject, q_values, _, _ = multipletests(p_values, method='fdr_bh')
moran_df.loc['q_value'] = q_values

# Filter for significant genes (q < 0.01)
km_genes = moran_df.loc[:, moran_df.loc['q_value'] < 0.01].columns.tolist()
print(f"Found {len(km_genes)} significant genes (q < 0.01) for trajectory analysis")

# Extract expression data for significant genes only
gene_expr_sig = gene_expr[:, pfc_zhu_rna_EN.var_names.isin(km_genes)]

## smooth gene expression by ordinal pseudotime using NB-GAM (similar to tradeSeq)
@ray.remote
def fit_gam_smoother_for_gene(g, gene_name, X, pseudotime, n_knots=9, n_grid=200, 
                               distribution='gamma', link='log'):
    """
    Fit a GAM smoother to gene expression data along pseudotime (parallelized version).
    
    Similar to tradeSeq's fitGAM + predictSmooth approach:
    - Uses spline basis functions with specified knots
    - Fits a GAM with appropriate distribution (gamma as approximation to NB for continuous data)
    - Returns smoothed predictions over pseudotime grid
    
    Parameters:
    -----------
    g : int
        Gene index
    gene_name : str
        Gene name
    X : array-like
        Full gene expression matrix
    pseudotime : array-like
        Pseudotime values for each cell
    n_knots : int
        Number of knots for spline basis (default 9, as in paper)
    n_grid : int
        Number of points for prediction grid
    distribution : str
        Distribution family ('gamma', 'normal', or 'poisson')
    link : str
        Link function ('log', 'identity')
    
    Returns:
    --------
    gene_name : str
        Gene name
    smooth_pseudotime : array
        Pseudotime grid points
    smooth_expr : array
        Smoothed expression values
    """
    from pygam import GAM
    from pygam.terms import s
    import numpy as np
    
    # Extract gene expression
    y = X[:, g].flatten() if hasattr(X[:, g], 'flatten') else X[:, g].ravel()
    
    # Add small offset to avoid zeros for log link
    y_offset = y + 0.1 if distribution in ['gamma', 'poisson'] else y
    
    # Fit GAM with spline term
    # s(0) means fit a smooth spline on the first (and only) feature
    gam = GAM(s(0, n_splines=n_knots), distribution=distribution, link=link)
    
    try:
        gam.fit(pseudotime.reshape(-1, 1), y_offset)
        
        # Generate prediction grid
        pt_min, pt_max = pseudotime.min(), pseudotime.max()
        smooth_pseudotime = np.linspace(pt_min, pt_max, n_grid)
        
        # Predict smoothed values
        smooth_expr = gam.predict(smooth_pseudotime.reshape(-1, 1))
        
        # Remove offset if added
        if distribution in ['gamma', 'poisson']:
            smooth_expr = np.maximum(smooth_expr - 0.1, 0)
            
    except Exception as e:
        # If GAM fitting fails, fall back to linear interpolation
        sort_idx = np.argsort(pseudotime)
        smooth_pseudotime = pseudotime[sort_idx]
        smooth_expr = y[sort_idx]
        # Interpolate to regular grid
        pt_min, pt_max = pseudotime.min(), pseudotime.max()
        grid_pseudotime = np.linspace(pt_min, pt_max, n_grid)
        smooth_expr = np.interp(grid_pseudotime, smooth_pseudotime, smooth_expr)
        smooth_pseudotime = grid_pseudotime
    
    return gene_name, smooth_pseudotime, smooth_expr

# Get pseudotime and prepare for smoothing and put data in Ray's object store for efficient parallel access
pseudotime = pfc_zhu_rna_EN.obs['ordinal_pseudotime'].values
pseudotime_ref = ray.put(pseudotime)
X_ref_sig = ray.put(gene_expr_sig)

print(f"Fitting GAM smoothers for {len(km_genes)} significant genes with Ray...")
gam_futures = [
    fit_gam_smoother_for_gene.remote(g, gene, X_ref_sig, pseudotime_ref, 
                                     n_knots=9, n_grid=200)
    for g, gene in enumerate(km_genes)
]

# Get results with progress bar
smooth_results = {}
smooth_pseudotime_grid = None

for future in tqdm(gam_futures, desc="Smoothing genes"):
    gene_name, smooth_pt, smooth_expr = ray.get(future)
    smooth_results[gene_name] = smooth_expr
    if smooth_pseudotime_grid is None:
        smooth_pseudotime_grid = smooth_pt

# Create DataFrame with smoothed expression
smooth_gene_expr_df = pd.DataFrame(smooth_results, index=smooth_pseudotime_grid)
smooth_gene_expr_df.index.name = 'pseudotime'

# Shutdown Ray to free resources
ray.shutdown()

# Apply z-score normalization to smoothed trajectories
smooth_gene_expr_zscore = smooth_gene_expr_df.apply(zscore, axis=0)

# Perform K-means clustering on z-scored smooth trajectories
print("Performing K-means clustering on smoothed trajectories...")
n_clusters = 4
km = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
km.fit(smooth_gene_expr_zscore.T.values)

# Create cluster labels
km_labels = pd.Series(km.labels_, index=km_genes)
km_map = {i: f'cluster_{i+1}' for i in range(n_clusters)}
km_labels_named = km_labels.map(km_map)

print(f"Genes per cluster: {km_labels.value_counts().sort_index().to_dict()}")

# Compute mean trajectory per cluster
smooth_gene_expr_by_cluster = smooth_gene_expr_zscore.copy()
smooth_gene_expr_by_cluster.rename(columns=km_labels.to_dict(), inplace=True)
smooth_gene_expr_by_cluster.columns.name = 'cluster'
mean_trajectories = smooth_gene_expr_by_cluster.groupby(level=0, axis=1).mean()

# Plot mean trajectories
max_idx_dict = {}
ax = mean_trajectories.plot(linewidth=2, alpha=0.5)
for cluster, trajectory in mean_trajectories.items():
    max_idx = trajectory.idxmax()
    max_val = trajectory.max()
    ax.annotate('*', xy=(max_idx, max_val), fontsize=12, color=f'C{cluster}')
    max_idx_dict[cluster] = max_idx

## Relabel clusters based on location of maximum expression along pseudotime
new_labels_mapper = 'km' + pd.Series(max_idx_dict).rank().astype(int).astype(str)
mean_trajectories_relabelled = mean_trajectories.rename(columns=new_labels_mapper).sort_index(axis=1)
km_labels_relabelled = km_labels.replace(new_labels_mapper).rename('clusters')
genes_per_cluster = km_labels_relabelled.groupby(km_labels_relabelled).apply(lambda x: x.index.unique().tolist())

## Compare learned clusters with clusters of GWAS hits
zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
gwas_hits = pd.read_excel(zhu_supp_tables, sheet_name='Table S12', header=2)
gwas_kms = gwas_hits.drop_duplicates(subset='Target gene name').set_index('Target gene name').dropna(subset=['km']).get('km')

## Make gene-cluster mappings categorical
km_labels_relabelled = km_labels_relabelled.astype(pd.CategoricalDtype(categories=new_labels_mapper.sort_values().values, ordered=True))
gwas_kms = gwas_kms.astype(pd.CategoricalDtype(categories=new_labels_mapper.sort_values().values, ordered=True))

## Compare gene-cluster mappings and compute MAE
mae_dict = {}
for gene in gwas_kms.index:
    try:
        mae = np.abs(gwas_kms.cat.codes.loc[gene] - km_labels_relabelled.cat.codes.loc[gene])
    except:
        mae = np.nan
    mae_dict[gene] = mae

mae_df = pd.Series(mae_dict)
mae_value = np.nanmean(mae_df)
print(f"MAE value: {mae_value:.2f}")

matches_df = mae_df.value_counts(dropna=False)
print(matches_df)

## write gene-clusters mapping to file to use as gene-sets
genes_per_cluster.to_csv(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv'))
print(f"Written gene-clusters mapping to {os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv')}")