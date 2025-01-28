# module load git-lfs/2.11.0

import argparse
from anndata import read_h5ad
import os
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from pybedtools import BedTool
from Bio import SeqIO
import pandas as pd
import torch

from models import get_hyena_dna_model_and_tokenizer

from setup_utils import \
    CAtlas_Tabula_Sapiens_setup, mdd_setup, pbmc_multiome_setup, splatter_sim_setup, toy_simulation_setup, Roussos_cerebral_cortex_setup, retain_feature_overlap, snMultiome_388_human_brains_setup, snMultiome_388_human_brains_one_subject_setup

import socket
hostname = socket.gethostname()

def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], -x[2]))

def top_k_columns_per_row(sorted_tuples, num_rows, K):
    # Initialize the matrix to store the top K column indices for each row
    top_k_matrix = np.full((num_rows, K), -1, dtype=int)  # Fill with -1 or any invalid index

    # Initialize a counter to track how many columns we've added for each row
    row_counter = np.zeros(num_rows, dtype=int)

    for row, col, value in sorted_tuples:
        if row_counter[row] < K:
            top_k_matrix[row, row_counter[row]] = col
            row_counter[row] += 1
    
    return top_k_matrix

def extract_hyena_dna_embeddings(atac, model, tokenizer, hg38_fa_path):

    ## create BedTool from extracted sequences
    atac_bed_df = pd.DataFrame(list(atac.var.index.str.split(':|-', expand=True).values), columns=['chrom','start','end'])
    atac_bed_df = atac_bed_df.copy()
    atac_bed_df['name'] = atac.var.index
    atac_bed = BedTool.from_dataframe(atac_bed_df)

    ## extract sequences from the ATAC data, based on reference genome
    sequences = atac_bed.sequence(fi=os.path.join(hg38_fa_path,'hg38.fa'))

    # Initialize an empty list to hold the sequences
    sequence_list = []
    tok_sequences = []

    # Read the sequences from the temporary FASTA file
    with torch.inference_mode():
        with open(sequences.seqfn) as seq_file:
            for record in SeqIO.parse(seq_file, "fasta"):
                #sequence_list.append(str(record.seq.upper()))
                sequence = str(record.seq.upper())

                tok_seq = tokenizer(sequence)
                tok_seq = tok_seq["input_ids"]  # grab ids
                tok_sequences.append(tok_seq)

        tok_sequences = np.vstack(tok_sequences)
        tok_sequences = torch.LongTensor(tok_sequences).to(device)

        ## loop through the sequences and get the embeddings, process in batches each of size 1/N
        N = 25
        batch_size = int(len(tok_sequences)/N)
        embeddings = []
        for i in range(0, len(tok_sequences), batch_size ):
            batch = tok_sequences[i:i+batch_size]
            embeddings_batch = model(batch).mean(-1) # mean pooling
            embeddings_batch = embeddings_batch.detach().cpu().numpy()
            embeddings.append(embeddings_batch)
        
    embeddings = np.vstack(embeddings)

    ## extract counts data and sort in descending order
    #x = atac.X.toarray()
    #x_argsort = np.argsort(x, axis=1)
    #x_argsort = np.flip(x_argsort, axis=1)
    #x_sorted = np.take_along_axis(x, x_argsort, axis=1)

    ## extract top K peaks
    #K = 100
    #x_argsort = x_argsort[:,:K]
    #x_sorted = x_sorted[:,:K]

    K = 100
    sorted_tuples = sort_coo(atac.X.tocoo()) # for MDD: ~15 minutes, peaks @ around 85 Gb
    x_argsort = top_k_columns_per_row(sorted_tuples, len(atac), K)# for MDD: another 15 minutes

    ## extract HyenaDNA embeddings of top K peaks & assign to ATAC data obsm
    topK_embeddings = embeddings[x_argsort, :] # peaks @ 97.7 Gb
    topK_embeddings = topK_embeddings.mean(1)  # mean-pooling (again), across top K peaks
    atac.obsm['hyena-dna'] = topK_embeddings

    return atac

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    nam_path = '/home/dmannk/projects/def-liyue/dmannk/neural-additive-models-pt'
    hyenadna_path = '/home/dmannk/projects/def-liyue/dmannk/hyena-dna'
    root_datapath = '/home/dmannk/projects/def-liyue/dmannk/data'
    hg38_fa_path = '/home/dmannk/projects/def-liyue/dmannk/data'

elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    nam_path = '/Users/dmannk/cisformer/neural-additive-models-pt'
    hyenadna_path = '/Users/dmannk/cisformer/hyena-dna'
    root_datapath = '/Users/dmannk/cisformer/workspace'
    hg38_fa_path = '/Users/dmannk/cisformer/workspace'

## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CAtlas_celltyping')
    parser.add_argument('--dataset', type=str, default='pbmc_multiome',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--atac_datapath', type=str, default='/Users/dmannk/cisformer/workspace',
                        help='path to ATAC data')
    parser.add_argument('--genes_by_peaks_str', type=str, default=None,
                        help='indicator of peaks to genes mapping to skip processing')
    args = parser.parse_args()

    ## define datapaths for MDD data
    mdd_rna_datapath = mdd_atac_datapath = os.path.join(root_datapath, 'mdd_data')

    ## set setup func for CLIP dataset
    if args.dataset == 'CAtlas_Tabula_Sapiens':
        setup_func = CAtlas_Tabula_Sapiens_setup

    elif args.dataset == 'pbmc_multiome':
        setup_func = pbmc_multiome_setup
        rna_datapath = atac_datapath = os.path.join(root_datapath, '10x_pbmc')
        RNA_file = "pbmcMultiome_rna.h5ad"
        ATAC_file = "pbmcMultiome_atac.h5ad"

    elif args.dataset == 'roussos':
        setup_func = Roussos_cerebral_cortex_setup
        datapath = os.path.join(root_datapath, 'Roussos_lab')
        rna_datapath = os.path.join(datapath, 'rna')
        atac_datapath = os.path.join(datapath, 'atac')

    elif args.dataset == '388_human_brains':
        setup_func = snMultiome_388_human_brains_setup
        datapath = rna_datapath = atac_datapath = os.path.join(root_datapath, '388_human_brains')

    elif args.dataset == '388_human_brains_one_subject':
        setup_func = snMultiome_388_human_brains_one_subject_setup
        subject = 'RT00391N'
        datapath = rna_datapath = atac_datapath = os.path.join(root_datapath, '388_human_brains', subject)

    ## get HyenaDNA model and sequence tokenizer
    model, tokenizer = get_hyena_dna_model_and_tokenizer()
    model.to(device)
    model.eval()

    ## extract data
    print('Extracting source data')
    if args.dataset == 'merged_roussos_pbmc_multiome':
        rna_datapath = atac_datapath = os.path.join(root_datapath, 'merged_data')
        source_rna = read_h5ad(os.path.join(root_datapath, 'merged_data', 'rna_merged_roussos_pbmc_multiome.h5ad'))
        source_atac = read_h5ad(os.path.join(root_datapath, 'merged_data', 'atac_merged_roussos_pbmc_multiome.h5ad'))
        source_cell_group = 'Cell type'
    else:
        source_rna, source_atac, source_cell_group, _, _, source_atac_fullpath, source_rna_fullpath = setup_func(args, hvg_only=True, protein_coding_only=True, pretrain=None, return_type='data')

    ## extract HyenaDNA embeddings for source ATAC data, and write to disk
    source_atac = extract_hyena_dna_embeddings(source_atac, model, tokenizer, hg38_fa_path)
    source_atac.write(source_atac_fullpath)

    ## Load MDD data, then extract HyenaDNA embeddings for MDD ATAC data and write to disk
    print('Extracting MDD data')
    mdd_rna, mdd_atac, mdd_cell_group, mdd_genes_to_peaks_binary_mask, mdd_genes_peaks_dict, mdd_atac_fullpath, mdd_rna_fullpath = mdd_setup(args, pretrain=None, return_type='data')
    mdd_atac = extract_hyena_dna_embeddings(mdd_atac, model, tokenizer, hg38_fa_path)
    mdd_atac.write(mdd_atac_fullpath)

    print('done')