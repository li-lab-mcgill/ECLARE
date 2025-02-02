import numpy as np
import torch
import warnings


def find_mnn_idxs(rna_genes, atac_genes, distance, n_neighbors=30):

    rna_nearest_to_atac = np.argpartition(distance, n_neighbors, axis=-1)[:,:n_neighbors]
    atac_nearest_to_rna = np.argpartition(distance.T, n_neighbors, axis=-1)[:,:n_neighbors]
    
    ## Find MNNs of RNA cells, which are ATAC cells
    atac_nearest_to_rna_mnns_idx = np.zeros(len(atac_nearest_to_rna))
    for i in range(len(atac_nearest_to_rna)):
        nn_i = atac_nearest_to_rna[i]
        mnn_i = nn_i[(rna_nearest_to_atac[nn_i] == i).any(1)]  # in NNs of cell i, find cells that have cell i as mutual NN
        if len(mnn_i) > 0:
            #mnn_idx = np.random.choice(mnn_i)
            mnn_idx = mnn_i[distance.T[i, mnn_i].argmax()]  # select MNN with largest distance to anchor
        else:
            mnn_idx = np.random.choice(nn_i)
        atac_nearest_to_rna_mnns_idx[i] = mnn_idx

    ## Find MNNs of ATAC cells, which are RNA cells
    rna_nearest_to_atac_mnns_idx = np.zeros(len(rna_nearest_to_atac))
    for i in range(len(rna_nearest_to_atac)):
        nn_i = rna_nearest_to_atac[i]
        mnn_i = nn_i[(atac_nearest_to_rna[nn_i] == i).any(1)]
        if len(mnn_i) > 0:
            #mnn_idx = np.random.choice(mnn_i)
            mnn_idx = mnn_i[distance[i, mnn_i].argmax()]  # select MNN with largest distance to anchor
        else:
            mnn_idx = np.random.choice(nn_i)
        rna_nearest_to_atac_mnns_idx[i] = mnn_idx

    return rna_nearest_to_atac_mnns_idx, atac_nearest_to_rna_mnns_idx


def get_triplet_loss(atac_latents, rna_latents, atac_celltypes, rna_celltypes, triplet_type='mnn', loss_fn=None):
    
    if triplet_type == 'mnn':
        ## Compute cosine distance
        #similarity = torch.nn.functional.cosine_similarity(rna_latents[None,:,:], atac_latents[:,None,:], dim=-1)
        distance = cosine_distance(atac_latents, rna_latents, detach=False) # dimensions ATAC x RNA
        #distance_atac = cosine_distance(atac_latents, atac_latents, detach=False)
        #distance_rna = cosine_distance(rna_latents, rna_latents, detach=False)

        ## Create MNN triplet indices
        with torch.no_grad():
            distance = distance.detach().cpu().numpy()
            rna_nearest_to_atac_mnns_idx, atac_nearest_to_rna_mnns_idx = find_mnn_idxs(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy(), distance, n_neighbors=30)

        ## Extract MNNs
        atac_latents_pos = atac_latents[atac_nearest_to_rna_mnns_idx]
        rna_latents_pos = rna_latents[rna_nearest_to_atac_mnns_idx]

        ## ATAC to RNA triplet
        atac_random_idxs = np.random.randint(0, atac_latents.size(0), (atac_latents.size(0),))
        atac_latents_neg   = atac_latents[atac_random_idxs]

        ## RNA to ATAC triplet        
        rna_random_idxs = np.random.randint(0, rna_latents.size(0), (rna_latents.size(0),))
        rna_latents_neg   = rna_latents[rna_random_idxs]
    
    elif triplet_type == 'cell-type':

        with torch.no_grad():

            distance = cosine_distance(atac_latents, rna_latents, detach=False) # dimensions ATAC x RNA
            same_celltypes_matrix = np.asarray(atac_celltypes)[:,None] == np.asarray(rna_celltypes)[None,:]  # dimensions ATAC x RNA
            atac_cells_with_match = same_celltypes_matrix.any(1)
            rna_cells_with_match = same_celltypes_matrix.any(0)

            rna_latents_pos_idxs, rna_latents_neg_idxs, atac_latents_pos_idxs, atac_latents_neg_idxs = [], [], [], []
            
            for d, distance_row in enumerate(distance.clone()):
                #if same_celltypes_matrix[d].any(): ## although most do, some ATAC cells have no RNA cells of the same celltype. should be resolved with larger batch size
                distance_row[~same_celltypes_matrix[d]] = -np.inf
                rna_latents_pos_idxs.append(distance_row.argmax().item())

                atac_latents_neg_idxs_candidates = np.where(~same_celltypes_matrix[:,same_celltypes_matrix[d]].any(1))[0]  # will still output candidates even if no match
                atac_latents_neg_idxs.append(np.random.choice(atac_latents_neg_idxs_candidates, size=1).item())

            for d, distance_col in enumerate(distance.T.clone()):
                #if same_celltypes_matrix[:,d].any(): ## although most do, some RNA cells have no ATAC cells of the same celltype. should be resolved with larger batch size
                distance_col[~same_celltypes_matrix[:,d]] = -np.inf
                atac_latents_pos_idxs.append(distance_col.argmax().item())

                rna_latents_neg_idxs_candidates = np.where(~same_celltypes_matrix[same_celltypes_matrix[:,d]].any(0))[0]  # will still output candidates even if no match
                rna_latents_neg_idxs.append(np.random.choice(rna_latents_neg_idxs_candidates, size=1).item())

            rna_latents_pos_idxs = np.asarray(rna_latents_pos_idxs)
            rna_latents_neg_idxs = np.asarray(rna_latents_neg_idxs)
            atac_latents_pos_idxs = np.asarray(atac_latents_pos_idxs)
            atac_latents_neg_idxs = np.asarray(atac_latents_neg_idxs)

            ## confirm proper triplet formation
            proper_triplets = \
            (np.asarray(rna_celltypes) == np.asarray(atac_celltypes)[atac_latents_pos_idxs])[rna_cells_with_match].all() and \
            (np.asarray(atac_celltypes) == np.asarray(rna_celltypes)[rna_latents_pos_idxs])[atac_cells_with_match].all() and \
            ~(np.asarray(rna_celltypes) == np.asarray(rna_celltypes)[rna_latents_neg_idxs]).any() and \
            ~(np.asarray(atac_celltypes) == np.asarray(atac_celltypes)[atac_latents_neg_idxs]).any()
            if not proper_triplets:
                warnings.warn('Improper triplet formation')

            ## sample triplets
            atac_latents_pos = atac_latents[atac_latents_pos_idxs]
            rna_latents_pos = rna_latents[rna_latents_pos_idxs]
            atac_latents_neg = atac_latents[atac_latents_neg_idxs]
            rna_latents_neg = rna_latents[rna_latents_neg_idxs]

    if (triplet_type=='mnn') or (triplet_type=='cell-type'):
        ## Compute ATAC triplet loss
        loss_atac = loss_fn(atac_latents, rna_latents_pos, atac_latents_neg)
        loss_atac = loss_atac.mean()
        
        ## Compute RNA triplet loss
        loss_rna = loss_fn(rna_latents, atac_latents_pos, rna_latents_neg)
        loss_rna = loss_rna.mean()

    
    return loss_atac, loss_rna