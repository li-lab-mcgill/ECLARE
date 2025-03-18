import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
from shutil import copy2

#from eclare import merged_dataset_setup, return_setup_func_from_dataset, run_CLIP, study_summary
from eclare import return_setup_func_from_dataset, study_summary
from eclare.run_utils import run_spatial_CLIP

if __name__ == "__main__":

    parser = ArgumentParser(description='Spatial CLIP')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', None),
                        help='output directory')
    parser.add_argument('--source_dataset', type=str, default='AD_Anderson_et_al',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--target_dataset', type=str, default='mdd',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--atac_datapath', type=str, default='/Users/dmannk/cisformer/workspace',
                        help='path to ATAC data')
    parser.add_argument('--rna_datapath', type=str, default='/Users/dmannk/cisformer/workspace',
                        help='path to RNA data')
    parser.add_argument('--genes_by_peaks_str', type=str, default=None,
                        help='indicator of peaks to genes mapping to skip processing')
    parser.add_argument('--cell_ontology_file', type=str, default='Cell_ontology.tsv',
                        help='Cell ontology file')  
    parser.add_argument('--ABC_dummies_file', type=str, default='CRE_to_genes_matrix_tabula_sapiens_HVG_global_max_max.h5ad',
                        help='ABC dummies file')
    parser.add_argument('--CAtlas_celltype_annotation_file', type=str, default='CAtlas_celltype_annotation.xlsx',
                        help='CAtlas celltype annotation file')
    parser.add_argument('--triplet_type', type=str, default='clip',
                        help='type of triplets to use')
    parser.add_argument('--num_units', type=int, default=128, metavar='U', nargs="+",
                        help='number of hidden units in encoders')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L', nargs="+",
                        help='learning rate')
    parser.add_argument('--total_epochs', type=int, default=2, metavar='E',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1000, metavar='B',
                        help='size of mini-batch')
    parser.add_argument('--feature', type=str, default='None specified', metavar='F',
                        help='Distinctive feature for current job')
    parser.add_argument('--tune_hyperparameters', action='store_true', default=False,
                        help='tune hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=1, metavar='R',
                        help='number of trials used for hyperparameter search')
    parser.add_argument('--CRE_to_genes_file', type=str, default='atac_sl_celltype_predictor_arguments.pkl',
                        help='Projection matrix to map cCREs to genes')
    parser.add_argument('--ATAC_file', type=str, default='atac_filt_hvp.h5ad',
                        help='ATAC data')
    parser.add_argument('--RNA_file', type=str, default='TabulaSapiens_hvg_genes_aligned_atac.h5ad',
                        help='RNA data')
    parser.add_argument('--not_adult_only', action='store_true', default=False,
                        help='flag to indicate whether only adult cells are considered')
    parser.add_argument('--valid_freq', type=int, default=1, metavar='V',
                        help='number of epochs after which performance evaluated on validation set')
    parser.add_argument('--tune_id', type=str, default=None,
                        help='ID of job for Optuna hyperparameter tuning')
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Flag to activate DDP for multi-GPU training')
    parser.add_argument('--subsample', type=int, default=-1,
                        help='number of samples to keep')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Flag to enable autoencoder pretraining')
    parser.add_argument('--use_warmstart', action='store_true', default=False,
                        help='Flag to use best result from warmstart')
    parser.add_argument('--use_tune', action='store_true', default=False,
                        help='Flag to use best result from tune')

    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    args = parser.parse_args()
    #args, _ = parser.parse_known_args()
    
    if torch.cuda.is_available():
        print('CUDA available')
        device   = 'cuda'
        num_gpus = torch.cuda.device_count()
    else:
        print('CUDA not available, set default to CPU')
        device   = 'cpu'

    ## Check number of cpus (does not work in interactive SALLOC environment)
    cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Allocated CPUs: {cpus_per_task}")

    ## set target dataset as env variable so easier to access in setup functions
    os.environ['target_dataset'] = args.target_dataset

    ## SOURCE dataset setup function
    source_setup_func = return_setup_func_from_dataset(args.source_dataset)

    ## TARGET dataset setup function
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)
    
    ## get data loaders
    train_loader, train_num_batches, train_n_batches_str_length, train_n_epochs_str_length, valid_loader, valid_num_batches, valid_n_batches_str_length, valid_n_epochs_str_length, n_genes, valid_idx = \
        source_setup_func(args.batch_size, args.total_epochs, return_type='loaders', dataset=args.source_dataset)
    
    ## missing overlapping_subjects argument if target is MDD (False by default)
    #_, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
    #    target_setup_func(args, return_type='loaders', dataset=args.target_dataset)
    target_rna_valid_loader = None
            
    ## Run training loops
    model = run_spatial_CLIP(
        args=args,
        rna_train_loader=train_loader,
        rna_valid_loader=valid_loader,
        train_num_batches=train_num_batches,
        train_n_batches_str_length=train_n_batches_str_length,
        train_n_epochs_str_length=train_n_epochs_str_length,
        valid_num_batches=valid_num_batches,
        valid_n_batches_str_length=valid_n_batches_str_length,
        valid_n_epochs_str_length=valid_n_epochs_str_length,
        target_rna_loader=target_rna_valid_loader,
        do_align_train=True,
        do_align_valid=True,
        do_align_train_eval=False,
        outdir=args.outdir
    )

    ## print output directory
    print('\n', args.outdir)

# %%
