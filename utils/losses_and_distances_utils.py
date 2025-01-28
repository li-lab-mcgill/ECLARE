import torch
import numpy as np
from pandas import DataFrame, factorize

def cosine_distance(x, y, norm=True, detach=True):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)

    similarity = torch.matmul(x_norm, y_norm.transpose(0, 1))
    distance = (1 - similarity)

    if detach:
        distance = distance.detach().cpu().numpy()
    return distance

def euclidean_distance(x, y, detach=True):
    distance = torch.cdist(x, y, p=2)
    if detach:
        distance = distance.detach().cpu().numpy()
    return distance

def mean_similarity(distance, off_diag=False):
    similarity = 1 - distance
    if off_diag:
        similarity = similarity[~np.eye(similarity.shape[0],dtype=bool)]
    return similarity.mean()

def compute_mmd_loss(atac_genes, rna_genes, bandwidth_parameter):
    distance_euclidean = euclidean_distance(atac_genes, rna_genes, detach=False) # dimensions ATAC x RNA
    distance_atac_euclidean = euclidean_distance(atac_genes, atac_genes, detach=False)
    distance_rna_euclidean = euclidean_distance(rna_genes, rna_genes, detach=False)

    ## Get median distance from concatenation or stacking of all three types of distance
    all_distances = np.hstack([distance_euclidean.flatten().detach().cpu().numpy(), distance_atac_euclidean.flatten().detach().cpu().numpy(), distance_rna_euclidean.flatten().detach().cpu().numpy()])
    median_distance = np.median(all_distances).item()
    #min_distance = min(i for i in all_distances if i > 0)  # minimum value above 0, LONG computation
    #sigma = np.max([median_distance, min_distance]) # ensure non
    sigma = median_distance
    gamma = bandwidth_parameter / (2 * sigma**2)

    rbf = rbf_from_distance(distance_euclidean, gamma=gamma)
    rbf_atac = rbf_from_distance(distance_atac_euclidean, gamma=gamma, off_diag=True)
    rbf_rna = rbf_from_distance(distance_rna_euclidean, gamma=gamma, off_diag=True)

    mmd_loss = rbf_atac.mean() + rbf_rna.mean() - 2*rbf.mean()
    #mmd_loss = mean_similarity(distance_atac, off_diag=True) + mean_similarity(distance_rna, off_diag=True) - 2*mean_similarity(distance)  # replace RBF kernel with cosine dist with cosine similarity ("cosine similarity kernel"), 1-dist to get back similarity
    return mmd_loss

def rbf_from_distance(distances, gamma=1, off_diag=False):
    rbf = (-gamma * distances**2).exp()
    if off_diag:
        rbf = rbf[~np.eye(rbf.shape[0],dtype=bool)]
    return rbf

def clip_loss(logits, atac_celltypes=None, rna_celltypes=None, atac_latents=None, rna_latents=None, temperature=1, do_ct=False, censor_same_celltype=False, return_logits=False, reduce=True):
    
    if (atac_latents is not None) and (rna_latents is not None) and (logits is None):
        atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
        rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
        logits = torch.matmul(atac_latents, rna_latents.T) * np.exp(temperature)

    if return_logits:
        logits_distance = np.exp(temperature) - logits
        return logits, logits_distance

    if censor_same_celltype:
        atac_labels, atac_uniques = factorize(atac_celltypes, sort=True)
        rna_labels, rna_uniques = factorize(rna_celltypes, sort=True)
        assert (atac_uniques == rna_uniques).all()

        censor_mat = (atac_labels[:, None] == rna_labels[None, :])
        np.fill_diagonal(censor_mat, False) ## still need the matching pair to be included in logits computation
        logits[censor_mat] = -np.inf

    # Symmetric loss function
    n = logits.shape[0]
    labels = torch.arange(n).to(logits.device)

    loss_atac_full = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    loss_rna_full = torch.nn.functional.cross_entropy(logits.T, labels, reduction='none')

    if (not do_ct) and (not reduce):
        return loss_atac_full, loss_rna_full

    elif (not do_ct) and reduce:
        loss_atac = loss_atac_full.mean()
        loss_rna  = loss_rna_full.mean()
        return loss_atac, loss_rna

    ## cell type-level CLIP loss
    elif do_ct and reduce:
        loss_atac = loss_atac_full.mean()
        loss_rna  = loss_rna_full.mean()
        
        loss_atac_ct = DataFrame(loss_atac_full.detach().cpu().numpy(), index=atac_celltypes).groupby(level=0).mean()
        loss_rna_ct = DataFrame(loss_rna_full.detach().cpu().numpy(), index=rna_celltypes).groupby(level=0).mean()

        loss_atac_ct.loc[len(loss_atac_ct)] = loss_atac.item()
        loss_rna_ct.loc[len(loss_rna_ct)] = loss_rna.item()

        loss_atac_ct = loss_atac_ct.rename(index={loss_atac_ct.index[-1]: 'ALL'})
        loss_rna_ct = loss_rna_ct.rename(index={loss_rna_ct.index[-1]: 'ALL'})

        return loss_atac, loss_rna, loss_atac_ct, loss_rna_ct

def clip_loss_split_by_ct(atac_latents, rna_latents, atac_celltypes, rna_celltypes, temperature=1):
    
    all_celltypes = np.unique( list(atac_celltypes) + list(rna_celltypes) )
    loss_dict = {}

    for celltype in all_celltypes:
        atac_latents_ct = atac_latents[atac_celltypes == celltype]
        rna_latents_ct = rna_latents[rna_celltypes == celltype]

        rna_celltypes_ct = [celltype] * len(rna_latents_ct)
        atac_celltypes_ct = [celltype] * len(atac_latents_ct)

        loss_atac, loss_rna = clip_loss(None, atac_celltypes_ct, rna_celltypes_ct, atac_latents_ct, rna_latents_ct, temperature=temperature, do_ct=False)
        loss_dict[celltype] = ((loss_atac + loss_rna)/2).item()

    loss_df = DataFrame.from_dict(loss_dict, orient='index')
    return loss_df

