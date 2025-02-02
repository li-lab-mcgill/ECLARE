#%%
from post_hoc_utils import *
from models import load_CLIP_model
from losses_and_distances_utils import clip_loss

import torch
import numpy as np
import ot
import argparse
import os
import pickle as pkl
from scipy.sparse import csr_matrix

import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    default_outdir = '/home/dmannk/scratch'
elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    default_outdir = '/Users/dmannk/cisformer/outputs'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get peak-gene links')
    parser.add_argument('--slurm_id', type=str, default='34887192', help='SLURM ID from which to load the model')
    parser.add_argument('--feature', type=str, default=None)
    args = parser.parse_args()
    #args, _ = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Load the model and data
    model_path = os.path.join(default_outdir, 'triplet_align_'+args.slurm_id, 'model.pt')
    model, source_rna, source_atac, mdd_rna, mdd_atac = get_model_and_data(model_path)

    ## Find unique cell types that are present in both RNA and ATAC data
    unique_celltypes = set(mdd_rna.obs['ClustersMapped'].unique()) & set(mdd_atac.obs['ClustersMapped'].unique())

    ## Find subjects that have both RNA and ATAC data, and split into control and MDD subjects
    overlapping_subjects = set(mdd_rna.obs['OriginalSub'].unique()) & set(mdd_atac.obs['BrainID'].unique())

    control_subjects_rna = set(mdd_rna.obs['OriginalSub'][mdd_rna.obs['Condition'] == 'Control'].unique()) & set(overlapping_subjects)
    mdd_subjects_rna     = set(mdd_rna.obs['OriginalSub'][mdd_rna.obs['Condition'] == 'Case'].unique())    & set(overlapping_subjects)
    control_subjects_atac = set(mdd_atac.obs['BrainID'][mdd_atac.obs['condition'] == 'Control'].unique()) & set(overlapping_subjects)
    mdd_subjects_atac     = set(mdd_atac.obs['BrainID'][mdd_atac.obs['condition'] == 'Case'].unique())    & set(overlapping_subjects)

    assert control_subjects_rna == control_subjects_atac;   control_subjects = control_subjects_rna = control_subjects_atac
    assert mdd_subjects_rna == mdd_subjects_atac;           mdd_subjects = mdd_subjects_rna = mdd_subjects_atac


    ## Initialize dictionaries to store peak-gene links and alignment rates
    all_peak_gene_links = {}
    all_align_rates = {}

    for celltype in unique_celltypes:
        for condition in ['Control', 'Case']:
            all_peak_gene_links[(celltype, condition)] = np.zeros([mdd_rna.n_vars, mdd_atac.n_vars])

    def get_coupling(model, rna_subject, atac_subject):
        rna_latents, atac_latents = get_latents(model, rna_subject, atac_subject)
        logits, logits_distance = clip_loss(atac_latents, rna_latents, None, None, return_logits=True)

        a = torch.ones(logits.shape[0]) / logits.shape[0]
        b = torch.ones(logits.shape[1]) / logits.shape[1]
        coupling = ot.lp.emd(a, b, logits_distance)

        return coupling

    def peak_gene_links(subject, celltype):

        rna_subject = mdd_rna[ (mdd_rna.obs['OriginalSub']==subject) & (mdd_rna.obs['ClustersMapped']==celltype) ]
        atac_subject = mdd_atac[ (mdd_atac.obs['BrainID'] == subject) & (mdd_atac.obs['ClustersMapped'] == celltype) ]

        n_rna_nuclei = len(rna_subject)
        n_atac_nuclei = len(atac_subject)
        print(f'For subject {subject} & cell type {celltype}, {n_rna_nuclei} RNA nuclei and {n_atac_nuclei} ATAC nuclei')

        ## get latents and compute pairwise cosine similarity
        coupling = get_coupling(model, rna_subject, atac_subject)

        if n_rna_nuclei > n_atac_nuclei:
            matches = coupling.argmax(0)
            best_pairs = torch.stack([matches, torch.arange(n_rna_nuclei)]).T
            atac_subject = atac_subject[best_pairs[:, 0].numpy()]

        elif n_rna_nuclei < n_atac_nuclei:
            matches = coupling.argmax(1)
            best_pairs = torch.stack([torch.arange(n_atac_nuclei), matches]).T
            rna_subject = rna_subject[best_pairs[:, 1].numpy()]

        elif n_rna_nuclei == n_atac_nuclei: # unlikely if MDD
            best_pairs = torch.stack(torch.where(coupling > 0)).T

        ''' for rectangular matrices, will ignore excessive rows/columns
        matches = sp.optimize.linear_sum_assignment(logits_distance)
        matches = np.stack(matches).T
        '''

        ## obtain couplings again on sorted data, to verify how well OT realigns the nuclei
        coupling = get_coupling(model, rna_subject, atac_subject)
        rna_matches = coupling.argmax(0)
        atac_matches = coupling.argmax(1)
        rna_align_rate = (rna_matches == torch.arange(len(rna_matches))).sum() / len(rna_matches)
        atac_align_rate = (atac_matches == torch.arange(len(atac_matches))).sum() / len(atac_matches)
        align_rate = ((rna_align_rate + atac_align_rate) / 2).item()
        print(f'alignment rate: {align_rate:.2f}')

        ## obtain peak-gene links
        rna_subject_X = torch.from_numpy(rna_subject.X.toarray()).float().T
        atac_subject_X = torch.from_numpy(atac_subject.X.toarray()).float().T

        logits, logits_distance = clip_loss(rna_subject_X, atac_subject_X, None, None, return_logits=True)

        return logits.numpy(), align_rate


        '''
        rna_norms = np.sqrt(rna_subject_X.multiply(rna_subject_X).sum(axis=1)).A1
        atac_norms = np.sqrt(atac_subject_X.multiply(atac_subject_X).sum(axis=1)).A1

        norms = np.outer(rna_norms, atac_norms)
        norms[norms == 0] = 1
        cosine_similarity = rna_subject_X.T @ atac_subject_X
        '''

        '''
        # Apply Spectral Biclustering
        n_clusters = 3  # Set the number of clusters
        biclustering = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
        biclustering.fit(logits)

        # Rearrange the rows and columns of the matrix
        row_order = np.argsort(biclustering.row_labels_)
        column_order = np.argsort(biclustering.column_labels_)
        peak_gene_links = logits[row_order, :][:, column_order]
        '''

    for c, celltype in enumerate(unique_celltypes):
        for s, subject in enumerate(list(overlapping_subjects)):

            print(f'cell type {c+1} / {len(unique_celltypes)} - subject {s+1} / {len(overlapping_subjects)}')

            peak_gene_links_itr, align_rate = peak_gene_links(subject, celltype)

            if subject in control_subjects:
                all_peak_gene_links[(celltype, 'Control')] += peak_gene_links_itr / len(control_subjects)
            elif subject in mdd_subjects:
                all_peak_gene_links[(celltype, 'Case')] += peak_gene_links_itr / len(mdd_subjects)

            all_align_rates[(celltype, subject)] = align_rate

            print('\n')

    ## Save all peak-gene links as sparse csr matrices
    for celltype in unique_celltypes:
        for condition in ['Control', 'Case']:
            all_peak_gene_links[(celltype, condition)] = csr_matrix(all_peak_gene_links[(celltype, condition)])

    
    ## Temporarly enable write permission of output directory
    outdir = os.path.join(default_outdir, 'triplet_align_'+args.slurm_id)
    os.system(f'chmod -R 705 {outdir}')

    # Save the dictionary to a pickle file
    with open( os.path.join(outdir, 'peak_gene_dict.pkl') , 'wb') as f:
        pkl.dump([all_peak_gene_links, all_align_rates], f)

    ## Remove write permission of output directory
    os.system(f'chmod -R -w {outdir}')

    print('Done!')
