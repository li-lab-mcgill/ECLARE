import anndata
import torch
import numpy as np
from scipy.sparse import csr_matrix
from pandas import DataFrame

from scanpy.tl import leiden
from scanpy.pp import neighbors

from scib_metrics.nearest_neighbors import jax_approx_min_k
from scib_metrics import ilisi_knn, nmi_ari_cluster_labels_leiden, silhouette_label

from jax import jit
import jax.numpy as jnp
from jax.lax import top_k as lax_top_k
from functools import partial

from eclare.losses_and_distances_utils import clip_loss, clip_loss_split_by_ct
from eclare.data_utils import fetch_data_from_loader_light
from eclare.losses_and_distances_utils import cosine_distance

def get_metrics(model, rna_valid_loader, atac_valid_loader, device, paired=True):

    ## get data & latents
    rna_cells, rna_labels, rna_batches = fetch_data_from_loader_light(rna_valid_loader)
    atac_cells, atac_labels, atac_batches = fetch_data_from_loader_light(atac_valid_loader)
    rna_latents, _ = model(rna_cells.to(device=device), modality=0)
    atac_latents, _ = model(atac_cells.to(device=device), modality=1)

    ## concatenate latents
    rna_atac_latents    = torch.cat((rna_latents, atac_latents), dim=0)
    rna_atac_labels     = np.hstack([rna_labels, atac_labels])
    rna_atac_batches    = np.hstack([rna_batches, atac_batches])
    rna_atac_modalities = np.hstack([ np.repeat('rna', len(rna_latents)) , np.repeat('atac', len(atac_latents)) ])

    ## get unpaired metrics
    metrics = unpaired_metrics(rna_atac_latents, rna_atac_labels, rna_atac_modalities, rna_atac_batches)

    ## get paired metrics (if appropriate)
    if paired:
        foscttm_score = foscttm_moscot(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy()).mean().item()
        one_minus_foscttm_score = 1 - foscttm_score # the higher the better
        metrics['1-foscttm'] = one_minus_foscttm_score

    return metrics

def unpaired_metrics(latents, labels, modalities, batches, k=30):

    ## get neighbors object & initialize metrics dict
    neighbors = jax_approx_min_k(latents.detach().cpu(), k)
    unpaired_metrics = {}

    ## bioconservation
    nmi_ari_dict = nmi_ari_cluster_labels_leiden(neighbors, labels, optimize_resolution=True)
    silhouette_celltype = silhouette_label(latents.detach().cpu().numpy(), labels, rescale=True)

    ## multimodal & batch integration
    multimodal_ilisi = ilisi_knn(neighbors, modalities, scale=True)
    batches_ilisi = ilisi_knn(neighbors, batches, scale=True)

    ## update nmi_ari_dict to include other metrics
    unpaired_metrics.update({
        'nmi': nmi_ari_dict['nmi'],
        'ari': nmi_ari_dict['ari'],
        'silhouette_celltype': silhouette_celltype,
        'multimodal_ilisi': multimodal_ilisi,
        'batches_ilisi': batches_ilisi
    })

    return unpaired_metrics

def align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes, paired=True, is_latents=False):
            
    if not is_latents:

        ## get RNA latents
        rna_latents, rna_genes  = model(rna_cells, modality='rna', task='align')

        ## get ATAC latents
        atac_latents, atac_genes  = model(atac_cells, modality='atac', task='align')

    elif is_latents:
        rna_latents, atac_latents = rna_cells, atac_cells
        rna_genes, atac_genes = None, None

    ## concatenated modalities and celltypes arrays
    rna_atac_latents = torch.vstack([rna_latents, atac_latents])
    rna_atac_latents_norm = torch.nn.functional.normalize(rna_atac_latents)
    modalities = np.hstack([ np.repeat('rna', len(rna_latents)) , np.repeat('atac', len(atac_latents)) ])
    celltypes  = np.hstack([ rna_celltypes , atac_celltypes ])

    ## jitted metrics for paired data
    if paired:
        foscttm_score_full = foscttm_moscot(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy())
        foscttm_score = foscttm_score_full.mean()
        foscttm_score_ct = DataFrame(np.asarray(foscttm_score_full), index=rna_celltypes).groupby(level=0).mean()

        accuracy_full = topk_accuracy(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy(), 1)
        accuracy_top5_full = topk_accuracy(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy(), 5)

        accuracy = accuracy_full.mean()
        accuracy_top5 = accuracy_top5_full.mean()

        ## cell type-level accuracy
        accuracy_ct = DataFrame(np.asarray(accuracy_full), index=atac_celltypes).groupby(level=0).mean()
        accuracy_top5_ct = DataFrame(np.asarray(accuracy_top5_full), index=rna_celltypes).groupby(level=0).mean()

        ## CLIP losses
        clip_loss_atac, clip_loss_rna, clip_loss_atac_ct, clip_loss_rna_ct = clip_loss(None, atac_celltypes, rna_celltypes, atac_latents, rna_latents, temperature=1, do_ct=True, censor_same_celltype=False)
        clip_loss_ = ((clip_loss_atac + clip_loss_rna) / 2).item()
        clip_loss_ct = ((clip_loss_atac_ct + clip_loss_rna_ct) / 2)

        #clip_loss_censored_atac, clip_loss_censored_rna = clip_loss(atac_latents, rna_latents, atac_celltypes, rna_celltypes, temperature=1, censor_same_celltype=True)
        #clip_loss_censored = ((clip_loss_censored_atac + clip_loss_censored_rna) / 2).item()
        clip_loss_censored = None

        clip_loss_ct_split = clip_loss_split_by_ct(atac_latents, rna_latents, atac_celltypes, rna_celltypes, temperature=1)

    elif not paired:
        foscttm_score   = None
        foscttm_score_ct = None
        rank_score_ = None
        accuracy        = None
        accuracy_top5   = None
        accuracy_ct     = None
        accuracy_top5_ct = None
        clip_loss_ = None
        clip_loss_censored = None
        clip_loss_ct = None
        clip_loss_ct_split  = None

    ## compute concetration of the inner product towards the off-diagonal elements
    if (rna_genes is not None) and (atac_genes is not None):

        gene_by_gene = 1 - cosine_distance(rna_genes.T, atac_genes.T, detach=False).detach().cpu().numpy()
        gene_by_gene[np.isnan(gene_by_gene)] = 0
        A = np.sum(gene_by_gene**2)**(0.5)
        D = np.sum(np.diag(gene_by_gene)**2)**(0.5)
        diag_concentration = D / A
        diag_concentration_minimizer = -np.log10(diag_concentration)

    else:
        diag_concentration_minimizer = None

    atac_and_rna_latents_df = anndata.concat([anndata.AnnData(atac_latents.detach().cpu().numpy(), obs={'modality':'atac'}), anndata.AnnData(rna_latents.detach().cpu().numpy(), {'modality':'rna'})])
    atac_and_rna_latents_df.obs['modality'] = atac_and_rna_latents_df.obs['modality'].astype('category')
    neighbors(atac_and_rna_latents_df, n_neighbors=30, use_rep='X')
    leiden(atac_and_rna_latents_df)

    asw_ct = silhouette_label(atac_and_rna_latents_df.X, atac_and_rna_latents_df.obs['leiden'], rescale=True, chunk_size=256)
    #asw_mod = silhouette_batch(atac_and_rna_latents_df.X, atac_and_rna_latents_df.obs['leiden'], modalities, rescale=True, chunk_size=256)

    rank_score_ = iLISI(rna_latents.detach().cpu(), atac_latents.detach().cpu(), modalities)

    similarity = torch.matmul(rna_atac_latents_norm,rna_atac_latents_norm.T)
    similarity = torch.clamp(similarity, min=-1, max=1)

    #similarity = torch.nn.functional.cosine_similarity(rna_atac_latents[None,:,:], rna_atac_latents[:,None,:], dim=-1)

    distance = 1 - similarity
    distance = distance.detach().cpu().numpy()
    np.fill_diagonal(distance, np.inf)

    ilisis = 0
    clisis = 0
    nmi = 0
    ari = 0
    #n_neighbors_array = np.logspace(np.log10(10), np.log10(subsample//2), 5).astype(int)

    n_neighbors_dict = {
        'ilisi': 30,
        'nmi_ari': 30
    }
    
    for metric, n_neighbors in n_neighbors_dict.items():

        distance_ = distance.copy()
        not_nns = np.argpartition(distance_, n_neighbors, axis=1)[:,n_neighbors:]
        nns = np.argpartition(distance_, n_neighbors, axis=1)[:,:n_neighbors]

        for n, (not_nns_, nns_) in enumerate(zip(not_nns, nns)):
            distance_[n][not_nns_] = 0
            distance_[n][nns_] += 1e-7

        np.fill_diagonal(distance_, 0)
        distance_ = csr_matrix(distance_)

        #sc.pp.neighbors(atac_and_rna_latents, n_neighbors=n_neighbors, metric='cosine')
        #distance_ = atac_and_rna_latents.obsp['distances']

        ilisis_itr, nmi_itr, ari_itr = 0, 0, 0

        if metric == 'ilisi':
            ilisis_itr = ilisi_knn(distance_, modalities, scale=True)  # long

        elif metric == 'nmi_ari':
            nmi_itr, ari_itr = nmi_ari_cluster_labels_leiden(distance_, celltypes, optimize_resolution=True).values()

        ilisis += ilisis_itr#/len(n_neighbors_array)
        nmi += nmi_itr#/len(n_neighbors_array)
        ari += ari_itr#/len(n_neighbors_array)

    return ilisis, asw_ct, nmi, ari, diag_concentration_minimizer, foscttm_score, rank_score_, accuracy, accuracy_top5, clip_loss_, clip_loss_censored,\
            foscttm_score_ct, accuracy_ct, accuracy_top5_ct, clip_loss_ct, clip_loss_ct_split


def cdist(x, y, metric='cosine'):
    """Compute the pairwise distance matrix between each row of x and y."""
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    y2 = jnp.sum(y**2, axis=1, keepdims=True)
    xy = jnp.dot(x, y.T)

    if metric == 'euclidean':
        return jnp.sqrt(x2 - 2*xy + y2.T)
    elif metric == 'cosine':
        return 1 - (xy / (jnp.sqrt(x2) * jnp.sqrt(y2.T)))
    elif metric == 'inner':
        return -xy

@jit
def closest_cells(mod1, mod2, Gs, P=None, device='cpu'):

    '''
    can confirm that code works by:
    - setting: mapped_X_onto_proj_Y = proj_Y.clone()
    - checking that y_closest_to_x_post_ot and x_closest_to_y_post_ot all 0's
    '''
    # Pairwise distance before linear transformation P, before optimal transport
    foscttm_pre_p = foscttm_moscot(mod1, mod2)
    
    # Pairwise distance after linear transformation P, before optimal transport
    if P is not None:
        mod2 = jnp.matmul(mod2, P)
    foscttm_pre_ot = foscttm_moscot(mod1, mod2)
    
    # Pairwise distance after optimal transport 
    mapped_mod1_onto_mod2 = jnp.matmul(Gs, mod2)
    foscttm_post_ot = foscttm_moscot(mapped_mod1_onto_mod2, mod2)

    return foscttm_pre_p, foscttm_pre_ot, foscttm_post_ot


## both foscttm functions below require that cells x and y match, or else would need to handle cell identifiers
@jit
def foscttm(
    x: jnp.array,
    y: jnp.array,
) -> float:
    c = cdist(x, y)
    y_closest_to_x = jnp.argsort(c, axis=1)
    x_closest_to_y = jnp.argsort(c, axis=0)

    where_y_closest_to_x = jnp.where(y_closest_to_x == jnp.arange(y_closest_to_x.shape[0])[:, jnp.newaxis])[1]
    where_x_closest_to_y = jnp.where(x_closest_to_y == jnp.arange(x_closest_to_y.shape[1])[jnp.newaxis, :])[0]
    
    foscttm = 0.5 * ((where_y_closest_to_x.astype(jnp.float32).mean() / len(y)) + (where_x_closest_to_y.astype(jnp.float32).mean() / len(x)))
    return foscttm.item()

@jit
def foscttm_moscot( # from https://moscot.readthedocs.io/en/latest/notebooks/tutorials/600_tutorial_translation.html#define-utility-functions
    x: jnp.array,
    y: jnp.array,
) -> float:
    c = cdist(x, y)
    foscttm_x = (c < jnp.expand_dims(jnp.diag(c), axis=1)).mean(axis=1)
    foscttm_y = (c < jnp.expand_dims(jnp.diag(c), axis=0)).mean(axis=0)
    foscttm_full = (foscttm_x + foscttm_y) / 2 #jnp.mean(foscttm_x + foscttm_y) / 2
    return foscttm_full

@jit
def pair_alignment_accuracy(
    x: jnp.array,
    y: jnp.array
) -> float:
    c = cdist(x, y)
    accuracy_x = ((c < jnp.expand_dims(jnp.diag(c), axis=1)).mean(axis=1) == 0).mean()
    accuracy_y = ((c < jnp.expand_dims(jnp.diag(c), axis=0)).mean(axis=0) == 0).mean()
    accuracy = (accuracy_x + accuracy_y) / 2
    return accuracy

@partial(jit, static_argnums=(2,))
def topk_accuracy(
    x: jnp.array,
    y: jnp.array,
    K: int = 1
) -> float:
    """
    Check if true match is in top K closest cells
    """
    c = cdist(x, y)
    y_closest_to_x = lax_top_k((1-c), K)[1]
    x_closest_to_y = lax_top_k((1-c).T, K)[1].T

    topk_accuracy_x = (y_closest_to_x == jnp.arange(y_closest_to_x.shape[0])[:, jnp.newaxis]).any(axis=1)
    topk_accuracy_y = (x_closest_to_y == jnp.arange(x_closest_to_y.shape[1])[jnp.newaxis, :]).any(axis=0)

    topk_accuracy = (topk_accuracy_x + topk_accuracy_y) / 2
    return topk_accuracy

@partial(jit, static_argnums=(2,3))
def topk_accuracy_csls(
    x: jnp.array,
    y: jnp.array,
    K_acc: int = 5,
    K_csls: int = 10
) -> float:
    """
    Check if true match is in top K closest cells
    """
    csls = csls_matrix(x, y, K=K_csls)
    y_closest_to_x = lax_top_k(csls, K_acc)[1]
    x_closest_to_y = lax_top_k(csls.T, K_acc)[1].T

    topk_accuracy_x = (y_closest_to_x == jnp.arange(y_closest_to_x.shape[0])[:, jnp.newaxis]).any(axis=1)
    topk_accuracy_y = (x_closest_to_y == jnp.arange(x_closest_to_y.shape[1])[jnp.newaxis, :]).any(axis=0)

    topk_accuracy = (topk_accuracy_x.mean() + topk_accuracy_y.mean()) / 2
    return topk_accuracy


@jit
def trace_metric(
    x: jnp.array,
    y: jnp.array,
) -> float:
    c = cdist(x, y)
    geom = pointcloud.PointCloud(x, y, cost_matrix=None, cost_fn=None, epsilon = c.mean())
    Gs = linear.solve(geom, initializer='default').matrix
    test_trace_metric = (1 - Gs.trace()) * len(Gs)
    return test_trace_metric

@jit
def pairwise_cosine_similarity(x, y):
    # Normalize the vectors along the last dimension (feature axis)
    x_norm = x / jnp.sqrt(jnp.sum(x**2, axis=1, keepdims=True))
    y_norm = y / jnp.sqrt(jnp.sum(y**2, axis=1, keepdims=True))
    
    # Compute the cosine similarity
    similarity = jnp.dot(x_norm, y_norm.T)
    similarity = jnp.where(jnp.isnan(similarity), 0, similarity)
    
    return similarity

@partial(jit, static_argnums=(2,))
def csls_matrix(
    x: jnp.array,
    y: jnp.array,
    K: int = 10,
) -> float:
    cosine_sims = pairwise_cosine_similarity(x, y)
    R0 = jnp.sort(cosine_sims, axis=0)[-K:,].mean(axis=0, keepdims=True)
    R1 = jnp.sort(cosine_sims, axis=1)[:,-K:].mean(axis=1, keepdims=True)
    R = R0 + R1     # not sure if should transpose R such that axes align with x and y
    csls = (2 * cosine_sims) - R
    return csls

@partial(jit, static_argnums=(2,))
def foscttm_csls(
    x: jnp.array,
    y: jnp.array,
    K: int = 100
) -> float:
    csls = csls_matrix(x, y, K=K)
    foscttm_csls_x = (csls > jnp.expand_dims(jnp.diag(csls), axis=1)).mean(axis=1)
    foscttm_csls_y = (csls > jnp.expand_dims(jnp.diag(csls), axis=0)).mean(axis=0)
    foscttm_csls = jnp.mean(foscttm_csls_x + foscttm_csls_y) / 2
    return foscttm_csls


def compute_mdd_eval_metrics(model, mdd_rna_loader, mdd_atac_loader, device, mdd_eval_method='subsample'):

    if mdd_eval_method == 'subsample':
        rna_cells, rna_celltypes, atac_cells, atac_celltypes, _, _ = fetch_data_from_loaders(mdd_rna_loader, mdd_atac_loader, paired=False, subsample=2000)
        ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, _, acc, acc_top5, _, _, _, _, _, _, _ = align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes, paired=False)
        
    elif mdd_eval_method == 'loaders':

        all_ilisis, all_clisis, all_nmi, all_ari, all_diag_concentration_minimizer, all_foscttm_score, all_acc, all_acc_top5 = [], [], [], [], [], [], [], []

        for (rna_cells, rna_celltypes), (atac_cells, atac_celltypes) in zip(mdd_rna_loader, mdd_atac_loader):
            rna_cells = rna_cells.to_dense().squeeze(1).to(device=device)
            atac_cells = atac_cells.to_dense().squeeze(1).to(device=device)

            ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, acc, acc_top5, _, _, _, _, _, _, _ = align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes)

            all_ilisis.append(ilisis)
            all_clisis.append(clisis)
            all_nmi.append(nmi)
            all_ari.append(ari)
            all_diag_concentration_minimizer.append(diag_concentration_minimizer)
            all_foscttm_score.append(foscttm_score)
            all_acc.append(acc)
            all_acc_top5.append(acc_top5)

        ilisis = torch.mean(torch.stack(all_ilisis)).item()
        clisis = torch.mean(torch.stack(all_clisis)).item()
        nmi = torch.mean(torch.stack(all_nmi)).item()
        ari = torch.mean(torch.stack(all_ari)).item()
        diag_concentration_minimizer = torch.mean(torch.stack(all_diag_concentration_minimizer)).item()
        foscttm_score = torch.mean(torch.stack(all_foscttm_score)).item()
        acc = torch.mean(torch.stack(all_acc)).item()
        acc_top5 = torch.mean(torch.stack(all_acc_top5)).item()
    
    ## Save MDD evaluation metrics to csv
    return ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, acc, acc_top5

def rank_score(x, y, labels, bound='lower'):

    xy = torch.vstack([x, y])
    rna_atac_latents_norm = torch.nn.functional.normalize(xy)
    similarity = torch.matmul(rna_atac_latents_norm,rna_atac_latents_norm.T)
    similarity = torch.clamp(similarity, min=-1, max=1)
    distance = 1 - similarity

    sort_idx = np.argsort(distance, 1)
    labels_argsort = labels[sort_idx]
    labels_match = (labels_argsort[:,0][:,None] == labels_argsort[:,1:]).astype(int)

    ranks = np.arange(labels_match.shape[1])
    ranks_cumsum = ranks.cumsum()
    labels_match_ranks = labels_match * ranks[None]

    labels_match_freq = labels_match.sum(1)
    rank_score_lower_bound = (labels_match_freq * (labels_match_freq+1)) / 2
    rank_score_upper_bound = ranks_cumsum[-1] - ranks_cumsum[-labels_match_freq]

    rank_score_ = labels_match_ranks.sum(1)
    #mean_rank_score = rank_score_.mean()

    if bound == 'lower':
        rank_score_weighted = (rank_score_upper_bound - rank_score_) / (rank_score_upper_bound - rank_score_lower_bound)
        mean_rank_score_weighted = rank_score_weighted.mean()

    elif bound == 'upper':
        rank_score_weighted = rank_score_ / rank_score_upper_bound
        rank_score_weighted = 1 - rank_score_weighted
        mean_rank_score_weighted = rank_score_weighted.mean()
        
    return mean_rank_score_weighted     # higher the better

def rank_gap_score(x, y, labels):

    xy = torch.vstack([x, y])
    rna_atac_latents_norm = torch.nn.functional.normalize(xy)
    similarity = torch.matmul(rna_atac_latents_norm,rna_atac_latents_norm.T)
    similarity = torch.clamp(similarity, min=-1, max=1)
    distance = 1 - similarity

    sort_idx = np.argsort(distance, 1)
    labels_argsort = labels[sort_idx]
    labels_match = (labels_argsort[:,0][:,None] == labels_argsort[:,1:]).astype(int)

    ranks = np.arange(labels_match.shape[1])
    labels_match_ranks = labels_match * ranks[None]
    labels_not_match_ranks = (1 - labels_match) * ranks[None]

    rank_gap = np.abs(labels_not_match_ranks.mean(1) - labels_match_ranks.mean(1))

    labels_match_freq = labels_match.sum(1)
    rank_gap_upper_bound = ranks[-1] - labels_match_freq
    adjusted_rank_gap = rank_gap / rank_gap_upper_bound

    mean_adjusted_rank_gap = adjusted_rank_gap.mean()

    return mean_adjusted_rank_gap.item()     # higher the better

def ndcg(x, y, labels):

    xy = torch.vstack([x, y])
    rna_atac_latents_norm = torch.nn.functional.normalize(xy)
    similarity = torch.matmul(rna_atac_latents_norm,rna_atac_latents_norm.T)
    similarity = torch.clamp(similarity, min=-1, max=1)
    distance = 1 - similarity

    sort_idx = np.argsort(distance, 1)
    labels_argsort = labels[sort_idx]
    labels_match = (labels_argsort[:,0][:,None] == labels_argsort[:,1:]).astype(int)

    cumulative_gain = 2**labels_match - 1  # identical to labels match if relevance scores are binary
    discount = torch.log2(torch.arange(2, labels_match.shape[1]+2).float())
    discounted_cumulative_gain = (cumulative_gain / discount[None]).sum(1)

    labels_match_freq = labels_match.sum(1)
    inverse_discount_cumsum = discount.pow(-1).cumsum(0)
    ideal_discounted_cumulative_gain = inverse_discount_cumsum[labels_match_freq] # works if relevance score is binary, such that all non-zero relevance scores are equal to 1

    normalized_discounted_cumulative_gain = discounted_cumulative_gain / ideal_discounted_cumulative_gain
    mean_ndcg = normalized_discounted_cumulative_gain.mean()
    return mean_ndcg.item()

def iLISI(x, y, labels, K=30):

    xy = torch.vstack([x, y])
    rna_atac_latents_norm = torch.nn.functional.normalize(xy)
    similarity = torch.matmul(rna_atac_latents_norm,rna_atac_latents_norm.T)
    similarity = torch.clamp(similarity, min=-1, max=1)
    distance = 1 - similarity

    sort_idx = torch.argsort(distance, 1)
    labels_argsort = labels[sort_idx]
    labels_match = (labels_argsort[:,0][:,None] == labels_argsort[:,1:]).astype(int)
    labels_match_top_K = labels_match[:,:K]

    labels_match_counts = np.apply_along_axis(lambda x: np.resize( np.unique(x, return_counts=True)[1] , 2 ), 1, labels_match_top_K)  # use np.resize() to pad with 0's if only one unique label
    labels_match_probs = labels_match_counts / K
    all_same_label = (labels_match_probs == 1).all(1)
    iLISI_ = 1/(labels_match_probs**2).sum(1)

    B = len(np.unique(labels))
    iLISI_scaled = (iLISI_ - 1) / (B - 1)
    iLISI_scaled[all_same_label] = 0  # no integration if all same label

    median_iLISI_scaled = np.median(iLISI_scaled)
    mean_iLISI_scaled = np.mean(iLISI_scaled)

    return mean_iLISI_scaled
