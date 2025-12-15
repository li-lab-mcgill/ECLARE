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
import ast

from esda.moran import Moran
from libpysal.weights import WSP

sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
from eclare import set_env_variables
set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')

## load EN-only data from PFC Zhu
pfc_zhu_rna_EN = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'rna', 'pfc_zhu_rna_EN_ordinal.h5ad'))

pfc_zhu_atac_full = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'atac', 'PFC_Zhu_atac_raw.h5ad'), backed='r')
pfc_zhu_atac_EN = pfc_zhu_atac_full[pfc_zhu_atac_full.obs['Cell type'].isin(pfc_zhu_rna_EN.obs['Cell type'].unique())].to_memory()

## create graph from ordinal latents
sc.pp.neighbors(pfc_zhu_rna_EN, use_rep='X_ordinal_latents', key_added='X_ordinal_latents_neighbors', n_neighbors=30, n_pcs=10)
sc.tl.leiden(pfc_zhu_rna_EN, resolution=0.6, random_state=0, neighbors_key='X_ordinal_latents_neighbors')
sc.tl.umap(pfc_zhu_rna_EN, neighbors_key='X_ordinal_latents_neighbors')
sc.pl.umap(pfc_zhu_rna_EN, color=['Cell type', 'ordinal_pseudotime', 'leiden'], wspace=0.3, ncols=3)


# Convert raw to AnnData (copies X, var from raw; obs from parent)
mdd_rna_scaled_sub = anndata.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', f'mdd_rna_scaled_sub_15582.h5ad'))
mdd_rna_raw = mdd_rna_scaled_sub.raw.to_adata()

# Copy embeddings and graphs from processed object
mdd_rna_raw.obsm = dict(mdd_rna_scaled_sub.obsm)
mdd_rna_raw.obsp = dict(mdd_rna_scaled_sub.obsp)
mdd_rna_raw.uns = mdd_rna_scaled_sub.uns.copy()

sc.pp.normalize_total(mdd_rna_raw, target_sum=1e4, exclude_highly_expressed=False)
sc.pp.log1p(mdd_rna_raw)

## Define Ray remote functions for parallel computation (must be global)
@ray.remote
def compute_moran_for_gene(g, gene, X, W_dense, permutations=0):
    """Compute Moran's I for a single gene."""
    from esda.moran import Moran
    import pandas as pd
    # X is now dense, so just index directly
    x = X[:, g].flatten() if hasattr(X[:, g], 'flatten') else X[:, g].ravel()
    moran = Moran(x, W_dense, two_tailed=True, permutations=permutations)
    p_type = 'p_sim' if permutations > 0 else 'p_norm'
    moran_series = pd.Series(moran.__dict__, name=gene).loc[['I', p_type]]
    return moran_series

@ray.remote
def fit_gam_smoother_for_gene(g, gene_name, X, pseudotime, n_knots=9, n_grid=200, 
                               distribution='gamma', link='log'):
    """
    Fit a GAM smoother to gene expression data along pseudotime (parallelized version).
    
    Similar to tradeSeq's fitGAM + predictSmooth approach:
    - Uses spline basis functions with specified knots
    - Fits a GAM with appropriate distribution (gamma as approximation to NB for continuous data)
    - Returns smoothed predictions over pseudotime grid
    """
    from pygam import GAM
    from pygam.terms import s
    import numpy as np
    
    # Extract gene expression
    y = X[:, g].flatten() if hasattr(X[:, g], 'flatten') else X[:, g].ravel()
    
    # Add small offset to avoid zeros for log link
    y_offset = y + 0.1 if distribution in ['gamma', 'poisson'] else y
    
    # Fit GAM with spline term
    gam = GAM(s(0, n_splines=n_knots, spline_order=3))
    
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

def cluster_genes_by_trajectory(adata, 
                                 connectivity_key='X_ordinal_latents_neighbors_connectivities',
                                 pseudotime_col='ordinal_pseudotime',
                                 n_clusters=4,
                                 q_threshold=0.01,
                                 permutations=1000,
                                 n_knots=9,
                                 n_grid=200,
                                 dataset_name='dataset'):
    """
    Cluster genes based on their expression trajectories along pseudotime.
    
    This function:
    1. Computes Moran's I for each gene to identify spatially autocorrelated genes
    2. Filters for significant genes (q < q_threshold)
    3. Smooths gene expression using GAM along pseudotime
    4. Performs K-means clustering on z-scored smoothed trajectories
    5. Relabels clusters based on peak expression timing
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with gene expression
    connectivity_key : str
        Key in adata.obsp for the connectivity matrix
    pseudotime_col : str
        Column name in adata.obs for pseudotime values
    n_clusters : int
        Number of clusters for K-means
    q_threshold : float
        FDR threshold for significant genes
    permutations : int
        Number of permutations for Moran's I
    n_knots : int
        Number of knots for GAM splines
    n_grid : int
        Number of grid points for smoothed predictions
    dataset_name : str
        Name of the dataset for logging
    
    Returns:
    --------
    dict with:
        - 'moran_df': DataFrame with Moran's I statistics
        - 'km_genes': list of significant genes
        - 'smooth_gene_expr_df': DataFrame with smoothed expression
        - 'km_labels_relabelled': Series with cluster labels
        - 'genes_per_cluster': Series with genes per cluster
        - 'mean_trajectories': DataFrame with mean trajectory per cluster
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    ## Step 1: Compute Moran's I for each gene
    print(f"Computing Moran's I for {len(adata.var_names)} genes...")
    W = WSP(adata.obsp[connectivity_key].tocsr())
    W_dense = W.to_W()
    
    # Convert to dense array once to avoid repeated conversion overhead
    gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Put data in Ray's object store
    X_ref = ray.put(gene_expr)
    W_ref = ray.put(W_dense)
    
    # Parallel computation with Ray
    futures = [
        compute_moran_for_gene.remote(g, gene, X_ref, W_ref, permutations=permutations)
        for g, gene in enumerate(adata.var_names)
    ]
    
    # Get results with progress bar
    moran_series_list = []
    for future in tqdm(futures, desc=f"[{dataset_name}] Processing genes"):
        moran_series_list.append(ray.get(future))
    
    moran_df = pd.concat(moran_series_list, axis=1)
    
    # Add q-values (FDR-corrected p-values) using Benjamini-Hochberg method
    p_type = moran_df.index[-1]
    p_values = moran_df.loc[p_type].values
    reject, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    moran_df.loc['q_value'] = q_values
    
    # Filter for significant genes
    km_genes = moran_df.loc[:, moran_df.loc['q_value'] < q_threshold].columns.tolist()
    print(f"[{dataset_name}] Found {len(km_genes)} significant genes (q < {q_threshold}) for trajectory analysis")
    
    if len(km_genes) == 0:
        print(f"[{dataset_name}] No significant genes found. Returning None.")
        return None
    
    # Extract expression data for significant genes only
    gene_expr_sig = gene_expr[:, adata.var_names.isin(km_genes)]
    
    ## Step 2: Smooth gene expression using GAM
    print(f"[{dataset_name}] Fitting GAM smoothers for {len(km_genes)} significant genes...")
    pseudotime = adata.obs[pseudotime_col].values
    gene_expr_sig_df = pd.DataFrame(gene_expr_sig, index=pseudotime, columns=km_genes)
    gene_expr_sig_df.sort_index(inplace=True)
    
    # Extract pseudotime and expression data from sorted dataframe
    pseudotime_sorted = gene_expr_sig_df.index.values
    gene_expr_sig_sorted = gene_expr_sig_df.values
    
    pseudotime_ref = ray.put(pseudotime_sorted)
    X_ref_sig = ray.put(gene_expr_sig_sorted)
    
    gam_futures = [
        fit_gam_smoother_for_gene.remote(g, gene, X_ref_sig, pseudotime_ref, 
                                         n_knots=n_knots, n_grid=n_grid)
        for g, gene in enumerate(km_genes)
    ]
    
    # Get results with progress bar
    smooth_results = {}
    smooth_pseudotime_grid = None
    
    for future in tqdm(gam_futures, desc=f"[{dataset_name}] Smoothing genes"):
        gene_name, smooth_pt, smooth_expr = ray.get(future)
        smooth_results[gene_name] = smooth_expr
        if smooth_pseudotime_grid is None:
            smooth_pseudotime_grid = smooth_pt
    
    # Create DataFrame with smoothed expression
    smooth_gene_expr_df = pd.DataFrame(smooth_results, index=smooth_pseudotime_grid)
    smooth_gene_expr_df.index.name = 'pseudotime'
    
    ## Step 3: K-means clustering on z-scored smooth trajectories
    print(f"[{dataset_name}] Performing K-means clustering...")
    smooth_gene_expr_zscore = smooth_gene_expr_df.apply(zscore, axis=0)
    
    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    km.fit(smooth_gene_expr_zscore.T.values)
    
    # Create cluster labels
    km_labels = pd.Series(km.labels_, index=km_genes)
    
    print(f"[{dataset_name}] Genes per cluster: {km_labels.value_counts().sort_index().to_dict()}")
    
    # Compute mean trajectory per cluster
    smooth_gene_expr_by_cluster = smooth_gene_expr_zscore.copy()
    smooth_gene_expr_by_cluster.rename(columns=km_labels.to_dict(), inplace=True)
    smooth_gene_expr_by_cluster.columns.name = 'cluster'
    mean_trajectories = smooth_gene_expr_by_cluster.groupby(level=0, axis=1).mean()
    
    # Plot mean trajectories
    max_idx_dict = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_trajectories.plot(ax=ax, linewidth=2, alpha=0.5)
    for cluster, trajectory in mean_trajectories.items():
        max_idx = trajectory.idxmax()
        max_val = trajectory.max()
        ax.annotate('*', xy=(max_idx, max_val), fontsize=12, color=f'C{cluster}')
        max_idx_dict[cluster] = max_idx
    ax.set_title(f'{dataset_name}: Mean trajectories by cluster')
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Z-scored expression')
    plt.tight_layout()
    plt.show()
    
    ## Step 4: Relabel clusters based on location of maximum expression along pseudotime
    new_labels_mapper = 'km' + pd.Series(max_idx_dict).rank().astype(int).astype(str)
    mean_trajectories_relabelled = mean_trajectories.rename(columns=new_labels_mapper).sort_index(axis=1)
    km_labels_relabelled = km_labels.replace(new_labels_mapper).rename('clusters')
    genes_per_cluster = km_labels_relabelled.groupby(km_labels_relabelled).apply(lambda x: x.index.unique().tolist())
    
    print(f"[{dataset_name}] Clustering complete!")
    
    return {
        'moran_df': moran_df,
        'km_genes': km_genes,
        'smooth_gene_expr_df': smooth_gene_expr_df,
        'smooth_gene_expr_zscore': smooth_gene_expr_zscore,
        'km_labels_relabelled': km_labels_relabelled,
        'genes_per_cluster': genes_per_cluster,
        'mean_trajectories': mean_trajectories_relabelled,
        'new_labels_mapper': new_labels_mapper,
        'pseudotime': pseudotime,
        'gene_expr_sig': gene_expr_sig,
    }

# Initialize Ray - will use all available CPU cores by default
ray.init(num_cpus=40)

# Run trajectory clustering on both datasets
results_pfc_zhu = cluster_genes_by_trajectory(
    pfc_zhu_rna_EN,
    connectivity_key='X_ordinal_latents_neighbors_connectivities',
    pseudotime_col='ordinal_pseudotime',
    n_clusters=4,
    dataset_name='pfc_zhu_rna_EN'
)

results_mdd = cluster_genes_by_trajectory(
    mdd_rna_raw,
    connectivity_key='X_ordinal_latents_neighbors_connectivities',
    pseudotime_col='ordinal_pseudotime',
    n_clusters=4,
    dataset_name='mdd_rna_raw'
)

# Shutdown Ray to free resources
ray.shutdown()

#
from skimage.exposure import match_histograms
mdd_rna_ages = mdd_rna_raw.obs['Age'].values
mdd_ordinal_pseudotime = mdd_rna_raw.obs['ordinal_pseudotime'].values
mdd_ordinal_pseudotime_matched = match_histograms(mdd_ordinal_pseudotime, mdd_rna_ages)
mdd_ordinal_pseudotime_mapper = pd.Series(mdd_ordinal_pseudotime_matched, index=mdd_ordinal_pseudotime)


def devmdd_figS2(results_pfc_zhu, results_mdd, map_to_age=False, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'devmdd_figS2.pdf')):
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    results_pfc_zhu['mean_trajectories'].plot(ax=ax[0], linewidth=3, alpha=0.5)

    if map_to_age:
        # Map MDD mean trajectories index to closest pseudotime in mapper
        mdd_mean_traj = results_mdd['mean_trajectories']
        mapper_index = mdd_ordinal_pseudotime_mapper.index.values
        
        # Find closest mapper index for each trajectory index
        mapped_indices = []
        for idx_val in mdd_mean_traj.index:
            closest_idx = mapper_index[np.argmin(np.abs(mapper_index - idx_val))]
            mapped_indices.append(np.unique(mdd_ordinal_pseudotime_mapper.loc[closest_idx]).item())
        
        # Create new dataframe with mapped age indices
        mdd_mean_traj_mapped = mdd_mean_traj.copy()
        mdd_mean_traj_mapped.index = mapped_indices
        mdd_mean_traj_mapped.plot(ax=ax[1], linewidth=3, alpha=0.5)
    else:
        results_mdd['mean_trajectories'].plot(ax=ax[1], linewidth=3, alpha=0.5)

    ax[0].set_title('PFC-Zhu')
    ax[1].set_title('MDD')
    ax[0].set_xlabel('Pseudotime')
    ax[1].set_xlabel('Pseudotime')
    ax[0].set_ylabel('Z-scored expression')
    ax[1].set_ylabel('Z-scored expression')
    plt.tight_layout()
    
    print(f"Saving figure to {manuscript_figpath}")
    fig.savefig(manuscript_figpath)
    plt.close(fig)

# Extract results from pfc_zhu for downstream analysis
def enrichr_by_km(results, gene_sets):
    from gseapy import dotplot as gp_dotplot
    from gseapy import enrichr

    genes_per_cluster = results['genes_per_cluster']

    for km, km_genes in genes_per_cluster.items():
        hits = list(km_genes)
        enr = enrichr(
            gene_list=hits,
            gene_sets=gene_sets
        )
        enr_res = enr.results

        if gene_sets == 'ChEA_2022':
            enr_res = enr_res.loc[enr_res['Term'].str.contains('Human')]
            enr_res.loc[:, 'Term'] = enr_res['Term'].str.split(' ').str[0]

        g = gp_dotplot(enr_res, title=f'{km} - {len(hits)} genes', size=10, top_term=20)

    return g

#enrichr_by_km(results_pfc_zhu, tf_tg_links)
#enrichr_by_km(results_mdd, tf_tg_links)

g_pfc_zhu = enrichr_by_km(results_pfc_zhu, 'ChEA_2022')
g_mdd = enrichr_by_km(results_mdd, 'ChEA_2022')

## write gene-clusters mapping to file to use as gene-sets
#genes_per_cluster.to_csv(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv'))
#print(f"Written gene-clusters mapping to {os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv')}")

import pickle
with open(os.path.join(os.environ['OUTPATH'], 'gene_clusters_results.pkl'), 'wb') as f:
    pickle.dump(results_pfc_zhu, f)
    pickle.dump(results_mdd, f)
print(f"Written gene-clusters results to {os.path.join(os.environ['OUTPATH'], 'gene_clusters_results.pkl')}")

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

from scipy.sparse import csr_matrix
def build_mask_from_intervals(var_names, intervals_np):
    """
    Returns: (mask_bool, n_provided, n_matched)
    """
    idx = {p: i for i, p in enumerate(var_names)}
    hits = [idx[s] for s in map(str, intervals_np) if s in idx]
    mask = np.zeros(len(var_names), dtype=bool)
    if hits:
        mask[np.asarray(hits, dtype=int)] = True
    return mask, len(intervals_np), len(hits)

def make_peakset_annotation(var_names, sets_dict):
    """
    sets_dict: {"setname": np.array([...interval strings...]), ...}
    Returns: (CSR n_peaks × n_sets, kept_names[list])
    """
    cols = []
    names = []
    for name, ivals in sets_dict.items():
        mask, n_in, n_hit = build_mask_from_intervals(var_names, ivals)
        if n_hit == 0:
            print(f"[warn] {name}: 0/{n_in} intervals matched var_names — skipping")
            continue
        cols.append(mask.astype(np.uint8)[:, None])
        names.append(name)
    if not cols:
        raise ValueError("No intervals matched var_names for any set.")
    M = np.hstack(cols)  # small dense then CSR
    return csr_matrix(M), names

def deviations_to_df(dev_adata, row_name="sample", col_name="annotation"):
    """
    Convert the AnnData returned by compute_deviations into a tidy DataFrame (samples × annotations).
    """
    df = pd.DataFrame(
        dev_adata.X,
        index=getattr(dev_adata, "obs_names", None),
        columns=getattr(dev_adata, "var_names", None),
    )
    df.index.name = row_name
    df.columns.name = col_name
    return df

def km_cluster_enrichments(output_dir = os.path.join(os.environ['OUTPATH'], 'km_cluster_enrichments')):

    os.makedirs(output_dir, exist_ok=True)

    from pyjaspar import jaspardb
    import pychromvar as pc
    from gseapy import enrichr
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    zhu_supp_tables = os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'adg3754_Tables_S1_to_S14.xlsx')
    peak_gene_links = pd.read_excel(zhu_supp_tables, sheet_name='Table S7', header=2)
    peak_gene_links['peak'] = peak_gene_links['peak'].str.split(':|-').apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}")

    ## get gene-cluster mappings
    km_gene_sets = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'PFC_Zhu', 'gene_clusters_mapping.csv'), index_col=0, header=0)
    km_gene_sets = km_gene_sets['clusters.1'].to_dict()
    km_gene_sets = {k: ast.literal_eval(v) for k, v in km_gene_sets.items()} # or else, just a single string that looks like a list of genes

    km_gene_sets_df = pd.DataFrame.from_dict(km_gene_sets, orient='index').unstack().reset_index().set_index(0).drop(columns=['level_0']).rename(columns={'level_1': 'km'})
    peak_gene_links = peak_gene_links.merge(km_gene_sets_df, left_on='gene', right_index=True, how='left')
    linked_peaks_in_km = peak_gene_links.groupby('km')['peak'].unique().to_dict()


    ## pseudobulk ATAC data
    celltype_donor = pfc_zhu_atac_EN.obs[['Cell type','Donor ID']].apply(lambda x: f"{x[0]}_{x[1]}", axis=1)
    pseudobulk_groups = pd.get_dummies(celltype_donor, sparse=False).astype(int)
    pseudobulk_weights = pseudobulk_groups.sum(axis=0) / pseudobulk_groups.sum().sum()
    pseudobulk_matrix = pfc_zhu_atac_EN.X.T.dot(pseudobulk_groups.values)
    pseudobulk_matrix = pseudobulk_matrix.T
    pfc_zhu_atac_pseudobulked = anndata.AnnData(X=pseudobulk_matrix, obs=pseudobulk_groups.columns.to_frame(), var=pfc_zhu_atac_EN.var)
    pfc_zhu_atac_pseudobulked.obs['n_cells'] = pseudobulk_groups.sum(axis=0)
    pfc_zhu_atac_pseudobulked = pfc_zhu_atac_pseudobulked[pfc_zhu_atac_pseudobulked.obs['n_cells'] > 50]

    ## subset peaks to reduce memory usage (if enabled)
    subset_peaks = False

    if subset_peaks:
        peaks = peak_gene_links['peak'].unique()
        keep_peaks = \
            pfc_zhu_atac_pseudobulked.var_names[pfc_zhu_atac_pseudobulked.var_names.isin(peaks)].tolist() + \
            pfc_zhu_atac_pseudobulked.var_names.to_series().sample(120000).tolist()
        keep_peaks = np.unique(keep_peaks)

        pfc_zhu_atac_pseudobulked_sub = pfc_zhu_atac_pseudobulked[:,pfc_zhu_atac_pseudobulked.var_names.isin(keep_peaks)]
        adata = pfc_zhu_atac_pseudobulked_sub.copy()
    else:
        adata = pfc_zhu_atac_pseudobulked.copy()

    #pc.get_genome("hg38", output_dir="./")
    pc.add_peak_seq(adata, genome_file=os.path.join(os.environ['DATAPATH'], 'hg38.fa'), delimiter=':|-')
    pc.add_gc_bias(adata)
    pc.get_bg_peaks(adata) # ~2.5 minutes

    ## get motifs
    jdb_obj = jaspardb(release='JASPAR2020')
    motifs = jdb_obj.fetch_motifs(
        collection = 'CORE',
        tax_group = ['vertebrates'])

    pc.match_motif(adata, motifs=motifs)

    #orig_M = csr_matrix(adata.varm["motif_match"])
    orig_M = adata.varm["motif_match"]
    orig_names = np.asarray(adata.uns["motif_name"]).copy()

    motif_backup = orig_M.copy()
    name_backup  = orig_names.copy()

    dfs_per_set = []
    n_matching_peaks_per_set = {}
    for set_name, ivals in linked_peaks_in_km.items():

        print(f"Processing {set_name} (n={len(ivals)})")

        # 1) build mask for this set
        mask, n_in, n_hit = build_mask_from_intervals(adata.var_names, ivals)
        if n_hit == 0:
            print(f"[warn] {set_name}: 0/{n_in} intervals matched var_names — skipping")
            continue

        # 2) intersect motif annotations with this set
        #M_subset = csr_matrix(orig_M.multiply(mask[:, None]))      # zero peaks outside the set
        M_subset = orig_M * mask[:, None]

        # drop motifs with no peaks in this set
        keep = np.asarray(M_subset.sum(axis=0)).ravel() > 0
        if keep.sum() == 0:
            print(f"[warn] {set_name}: no motifs overlap the set — skipping")
            continue
        M_subset = M_subset[:, keep]
        names_subset = orig_names[keep].astype(object)

        # 3) swap in, compute, restore
        adata.varm["motif_match"] = M_subset
        adata.uns["motif_name"]   = names_subset

        dev_this = pc.compute_deviations(adata, n_jobs=10)

        # tidy DF; tag columns with the set name to keep them distinct
        df_this = deviations_to_df(dev_this, col_name="motif")
        df_this.columns = [f"{m}__in__{set_name}" for m in df_this.columns]
        dfs_per_set.append(df_this)

        overlap = pd.Series(M_subset.sum(axis=0), index=df_this.columns, name='Overlap').astype(str) + f"/{len(M_subset)}"
        n_matching_peaks_per_set[set_name] = overlap

    # restore original motif table
    adata.varm["motif_match"] = motif_backup
    adata.uns["motif_name"]   = name_backup

    # Join across sets (samples × (motif×set))
    df_dev_motif_in_sets = pd.concat(dfs_per_set, axis=1).sort_index(axis=1)
    w = pseudobulk_weights.values[:,None]
    #assert df_dev_motif_in_sets.index.equals(pseudobulk_weights.index)

    km_devs_dict = {}
    all_motif_names = []
    for km in km_gene_sets.keys():

        km_dev = df_dev_motif_in_sets.loc[:, df_dev_motif_in_sets.columns.str.contains(f"__in__{km}")]
        km_dev.columns = km_dev.columns.str.split('__in__').str[0]
        all_motif_names.extend(km_dev.columns.tolist())

        Z = km_dev.values
        #stouffer_Z = np.sum(Z * w, axis=0) / np.sqrt(np.sum(w**2))
        stouffer_Z = np.sum(Z, axis=0) / np.sqrt(len(Z))
        p = norm.sf(abs(stouffer_Z)) * 2  # Two-tailed p-value
        reject, q, _, _ = multipletests(p, method='fdr_bh')

        km_dev_adata = anndata.AnnData(X=km_dev.values, var=km_dev.columns.to_frame(), obs=km_dev.index.to_frame())
        km_dev_adata.var['q_value'] = q
        km_dev_adata.var['p_value'] = p
        km_dev_adata.var['stouffer_Z'] = stouffer_Z
        km_dev_adata.var['reject'] = reject

        km_devs_dict[km] = km_dev_adata

    ## get motif information
    all_motif_names = pd.Series(all_motif_names).unique().tolist()
    all_motifs = [jdb_obj.fetch_motif_by_id('.'.join(m.split('.')[:2])) for m in all_motif_names]
    all_motif_df = pd.DataFrame({
        "motif_id": [m.matrix_id for m in all_motifs],
        "name": [m.name for m in all_motifs],
        "tf_class": [m.tf_class for m in all_motifs],
        "species": [m.species for m in all_motifs],
    }, index=all_motif_names)
    motifs_name_mapper = all_motif_df['name'].to_dict()

    ## concatenate all km devs
    km_devs_df_list = []
    for km, km_dev_adata in km_devs_dict.items():

        if (km_dev_adata.var['q_value']<0.05).any():
            plot_motif_deviations(km_dev_adata, km, n_matching_peaks_per_set[km], outdir=output_dir)
        else:
            print(f"No significant motifs found for {km}")

        df = km_dev_adata.var['reject']
        df.index = pd.MultiIndex.from_product([[km], df.index], names=['km', 'motif'])
        km_devs_df_list.append(df)

    ## concatenate all km devs
    km_devs_df = pd.concat(km_devs_df_list)
    km_devs_df.fillna(False, inplace=True)
    km_devs_df = km_devs_df.unstack()
    km_devs_df.columns = km_devs_df.columns.map(motifs_name_mapper)
    hits_per_km_atac = km_devs_df.apply(lambda x: x.index[x.fillna(False).values].tolist(), axis=1).to_dict()

    ## EnrichR analysis
    motifs_overlap_dict = {}
    hits_per_km_rna = {}
    for km, km_genes in km_gene_sets.items():

        enr = enrichr(
            gene_list=list(km_genes),
            gene_sets="ChEA_2022",
            cutoff=1.0,
        )
        enr_res = enr.results
        enr_sig = enr_res[enr_res['Adjusted P-value'] < 0.05]

        plot_enrichr_dotplot(enr_sig, km, outdir=output_dir)

        tfs_sig = enr_sig['Term'].str.split(' ').str[0].unique()
        hits_per_km_rna[km] = tfs_sig.tolist()

        motifs_overlap = set(hits_per_km_atac[km]) & set(tfs_sig)
        motifs_overlap_dict[km] = list(motifs_overlap)


    ## save results to csv
    pd.DataFrame.from_dict(hits_per_km_rna, orient='index').to_csv(os.path.join(output_dir, 'hits_per_km_rna.csv'))
    pd.DataFrame.from_dict(hits_per_km_atac, orient='index').to_csv(os.path.join(output_dir, 'hits_per_km_atac.csv'))
    pd.DataFrame.from_dict(motifs_overlap_dict, orient='index').to_csv(os.path.join(output_dir, 'motifs_overlap.csv'))

    ## enrichment plots
    from gseapy import dotplot as gp_dotplot

    def plot_enrichr_dotplot(enr_res, km, outdir=None):

        # dotplot
        ofname = None if outdir is None else os.path.join(outdir, f'enrichr_dotplot_{km}.svg')

        fig, ax = plt.subplots(figsize=(2,6))
        fig = gp_dotplot(enr_res,
                column='Combined Score',
                x='-log10(fdr)',
                title=f"TF enrichment in {km} (EnrichR)",
                cmap=plt.cm.winter,
                size=12,
                top_term=10,
                show_ring=False,
                ax=ax,
                ofname=ofname)

        # Remove plt.show() - it's not thread-safe
        if ofname is not None:
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()

    def plot_motif_deviations(km_dev_adata, km, overlap, outdir=None):

        ofname = None if outdir is None else os.path.join(outdir, f'pychromvar_dotplot_{km}.svg')

        if overlap.index.str.contains('__in__').any():
            overlap.index = overlap.index.str.split('__in__').str[0]

        overlap = overlap.str.split('/').str[0] + '/100'
        km_dev_adata.var = km_dev_adata.var.merge(overlap, left_index=True, right_index=True)

        df_like_enr_res = km_dev_adata.var.rename(columns={0: 'motif', 'q_value': 'FDR q-val'}).copy()
        df_like_enr_res['neglog10q'] = -np.log10(np.clip(df_like_enr_res['FDR q-val'].values, 1e-300, 1))
        df_like_enr_res_sig = df_like_enr_res[df_like_enr_res['FDR q-val'] < 0.05]

        fig, ax = plt.subplots(figsize=(6,6))
        fig = gp_dotplot(df_like_enr_res_sig,
                column='FDR q-val',
                x='Stouffer Z',
                y='motif',
                title=f"TF enrichment in {km} (pychromVAR)",
                top_term=15,
                cmap=plt.cm.winter,
                size=20,
                size_legend="blabla",
                show_ring=False,
                ax=ax,
                ofname=ofname)

        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("# matched\npeaks")
            

        if ofname is not None:
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()
