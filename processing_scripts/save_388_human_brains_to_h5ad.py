from scanpy import read_mtx
from scanpy.external.pp import harmony_integrate, mnn_correct
from scanpy.pp import pca as sc_pca
from scanpy.pp import combat
import scanpy as sc
from muon import atac as ac

from glob import glob
import os
import numpy as np
import gzip
from pybedtools.bedtool import BedTool
import argparse
from anndata import concat as adata_concat

from setup_utils import get_protein_coding_genes

## Parse arguments
parser = argparse.ArgumentParser(description='Save 388 human brains data to h5ad')
parser.add_argument('--datapath', type=str, default='/home/dmannk/projects/def-liyue/dmannk/data/388_human_brains', help='Path to the data')
parser.add_argument('--save_type', type=str, default='concat', help='Whether to save the data as separate h5ad files or as a single concatenated h5ad file')
parser.add_argument('--batch_correction', type=str, default=None)
args = parser.parse_args()

datapath = args.datapath

filenames = glob(os.path.join(datapath, 'GSM*gz'))
subject_ids = np.unique([os.path.basename(f).split('_')[1] for f in filenames])

## Read gene list and peakset
gene_list_file = os.path.join(datapath, 'GSE261983_RNA_gene_list.txt.gz')
with gzip.open(gene_list_file, 'rt') as f: gene_list = f.read().split('\n')

peakset_file = os.path.join(datapath, 'GSE261983_ATAC_peakset.bed.gz')
peakset_bed = BedTool(peakset_file)
intervals_strings = [f'{interval.chrom}:{interval.start}-{interval.stop}' for interval in peakset_bed.intervals]
peakset_df = peakset_bed.to_dataframe()
peakset_df.index = intervals_strings

if args.save_type == 'concat':
    rna_datasets, atac_datasets, retained_subject_ids = [], [], []
    

## Loop through subjects
for subject_id in subject_ids:

    print(f'Processing {subject_id}...')

    ## Read RNA data and barcodes
    rna_data_file = glob(os.path.join(datapath, f'*{subject_id}_RNA_matrix.mtx.gz')).pop()
    rna = read_mtx(rna_data_file)
    rna = rna.T

    if rna.n_obs == 0:
        print(f'No observations in RNA data for {subject_id}, skipping...')
        continue


    ## Read ATAC data and barcodes
    atac_data_file = glob(os.path.join(datapath, f'*{subject_id}_ATAC_matrix.mtx.gz')).pop()
    atac = read_mtx(atac_data_file)
    atac = atac.T

    if atac.n_obs == 0:
        print(f'No observations in ATAC data for {subject_id}, skipping...')
        continue

    ## Set obs and var
    rna_barcode_file = glob(os.path.join(datapath, f'*{subject_id}_RNA_barcodes.txt.gz')).pop()
    with gzip.open(rna_barcode_file, 'rt') as f: rna_barcodes = f.read().split('\n')

    atac_barcode_file = glob(os.path.join(datapath, f'*{subject_id}_ATAC_barcodes.txt.gz')).pop()
    with gzip.open(atac_barcode_file, 'rt') as f: atac_barcodes = f.read().split('\n')

    rna.obs_names = rna_barcodes[:-1] # last element of barcodes empty '' string
    atac.obs_names = atac_barcodes[:-1] # last element of barcodes empty '' string

    rna.var_names = gene_list[:-1] # last element of gene_list empty '' string
    atac.var = peakset_df # last element of gene_list empty '' string

    ## Keep overlapping cells (obs)
    overlapping_obs = set(rna.obs_names) & set(atac.obs_names)
    rna = rna[ rna.obs_names.isin(overlapping_obs) ].copy()
    atac = atac[ atac.obs_names.isin(overlapping_obs) ].copy()

    ## Check if number of observations match between RNA and ATAC datasets
    if rna.n_obs != atac.n_obs:
        print(f'Number of observations do not match between RNA and ATAC datasets for {subject_id}, skipping...')
        continue

    ## save or append data
    if args.save_type == 'individual':
        ## Set var and obs names
        print(f'Saving {subject_id}...')

        ## Create directory for individual subject if it doesn't exist
        if not os.path.exists(os.path.join(datapath, subject_id)):
            os.makedirs(os.path.join(datapath, subject_id))
        
        rna.write_h5ad(os.path.join(datapath, subject_id, f'{subject_id}_rna.h5ad'))
        atac.write_h5ad(os.path.join(datapath, subject_id, f'{subject_id}_atac.h5ad'))

    elif args.save_type == 'concat':
        rna_datasets.append(rna)
        atac_datasets.append(atac)
        retained_subject_ids.append(subject_id)


## concatenate datasets and save
if args.save_type == 'concat':

    if args.batch_correction == 'mnn':
        mnn_correct(*rna_datasets, do_concatenate=True, batch_key='subject', batch_categories=retained_subject_ids)

    rna_datasets = adata_concat(rna_datasets, join='outer', merge='same', label='subject', axis=0)
    atac_datasets = adata_concat(atac_datasets, join='outer', merge='same', label='subject', axis=0)

    subject_id_mapper = dict(zip(np.arange(len(retained_subject_ids)), retained_subject_ids))
    rna_datasets.obs['subject'] = rna_datasets.obs['subject'].astype(int).replace(subject_id_mapper)
    atac_datasets.obs['subject'] = atac_datasets.obs['subject'].astype(int).replace(subject_id_mapper)


    if args.batch_correction == 'harmony':
        sc_pca(rna_datasets, zero_center=False)
        sc_pca(atac_datasets, zero_center=False)
        harmony_integrate(rna_datasets, key='subject')
        harmony_integrate(atac_datasets, key='subject')

    elif args.batch_correction == 'combat':
        ## Subset to protein-coding genes
        print('Subsetting to protein-coding genes...')
        rna = get_protein_coding_genes(rna)

        ## Subset to variable features
        print('Subsetting to variable features...')
        ac.pp.tfidf(atac, scale_factor=1e4)
        sc.pp.normalize_total(atac, target_sum=1e5, exclude_highly_expressed=False)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, n_top_genes=200000) # sc.pl.highly_variable_genes(atac)
        atac = atac[:, atac.var['highly_variable'].astype(bool)].to_memory()

        sc.pp.normalize_total(rna, target_sum=1e5, exclude_highly_expressed=False)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, n_top_genes=10000) # sc.pl.highly_variable_genes(rna)
        rna = rna[:, rna.var['highly_variable'].astype(bool)].to_memory()
        
        ## Combat
        print('Batch correcting wiht Combat...')
        combat(rna_datasets, key='subject')
        #combat(atac_datasets, key='subject')
        rna_datasets.X = (rna_datasets.X - rna_datasets.X.min()) / (rna_datasets.X.max() - rna_datasets.X.min())
        atac_datasets.X = (atac_datasets.X - atac_datasets.X.min()) / (atac_datasets.X.max() - atac_datasets.X.min())

    rna_datasets.write_h5ad(os.path.join(datapath, 'rna.h5ad'))
    atac_datasets.write_h5ad(os.path.join(datapath, 'atac.h5ad'))


print('Done!')



