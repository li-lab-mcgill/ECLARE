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
