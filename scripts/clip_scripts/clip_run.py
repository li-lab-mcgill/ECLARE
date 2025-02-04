
import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
from shutil import copy2

#from eclare import merged_dataset_setup, return_setup_func_from_dataset, run_CLIP, study_summary
from eclare import merged_dataset_setup, return_setup_func_from_dataset, study_summary
from eclare.run_utils import run_CLIP

if __name__ == "__main__":

    parser = ArgumentParser(description='CAtlas_celltyping')
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

    os.environ['target_dataset'] = args.target_dataset
    
    if torch.cuda.is_available():
        print('CUDA available')
        device   = 'cuda'
        num_gpus = torch.cuda.device_count()
    else:
        print('CUDA not available, set default to CPU')
        device   = 'cpu'

    ## Check number of cpus (does not work in interactive SALLOC environment)
    cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus_per_task:
        cpus_per_task = int(cpus_per_task)
    else:
        cpus_per_task = 1  # Default to 1 if not set

    print(f"Allocated CPUs: {cpus_per_task}")


    ## Setup
    align_setup_completed = False

    ## SOURCE dataset setup function
    if 'merged' in args.source_dataset:  # in process_datasets.py, difference between merged & imputed_merged. not here
        #args.source_dataset = args.source_dataset.replace('merged_', '')  # or else, not able to find proper subdirectory and data due to how subdirectory saved in merge_datasets.py
        #args.source_dataset = args.source_dataset.replace('imputed_', '')
        source_setup_func = merged_dataset_setup

    else:
        source_setup_func = return_setup_func_from_dataset(args.source_dataset)


    ## TARGET dataset setup function
    if 'merged' in args.target_dataset:
        #args.target_dataset = args.target_dataset.replace('merged_', '')
        #args.target_dataset = args.target_dataset.replace('imputed_', '')
        target_setup_func = merged_dataset_setup

    else:
        target_setup_func = return_setup_func_from_dataset(args.target_dataset)
    

    ## get data loaders
    rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
        source_setup_func(args, pretrain=None, return_type='loaders', dataset=args.source_dataset)
    
    if (args.source_dataset == 'roussos') and (args.target_dataset == 'mdd'):
        _, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
            target_setup_func(args, pretrain=None, return_type='loaders', dataset=args.target_dataset, overlapping_subjects_only=True)
    else:
        _, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
            target_setup_func(args, pretrain=None, return_type='loaders', dataset=args.target_dataset)
            
    ## Run training loops
    if (not args.tune_hyperparameters) and (args.tune_id is None):

        tuned_hyperparameters = {}

        model = run_CLIP(None, args, genes_to_peaks_binary_mask,
                               rna_train_loader, atac_train_loader, rna_valid_loader, atac_valid_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length,\
                               target_atac_valid_loader, target_rna_valid_loader, \
                               tuned_hyperparameters, outdir=args.outdir, \
                                do_pretrain_train=False, do_pretrain_valid=False, do_align_train=True, do_align_valid=True)
                         
    elif (not args.tune_hyperparameters) and (args.tune_id is not None):

        tune_id = str(args.tune_id)
        tune_job = 'clip_tune_' + tune_id
        tune_job_path = os.path.join(os.environ.get('OUTPATH'), tune_job)

        if args.use_tune:
            trial_df_path = os.path.join(tune_job_path, 'tune_optuna_trials.csv')
            trial_df = pd.read_csv(trial_df_path, index_col=0)
        elif args.use_warmstart:
            trial_df_path = os.path.join(tune_job_path, 'warmstart_optuna_trials.csv')
            trial_df = pd.read_csv(trial_df_path, index_col=0)
        else:
            print('warmstart vs tune not specified, default to concat of tune and warmstart trials')

            trial_df_path_warmstart = os.path.join(tune_job_path, 'warmstart_optuna_trials.csv')
            trial_df_warmstart = pd.read_csv(trial_df_path_warmstart, index_col=0)

            trial_df_path_tune      = os.path.join(tune_job_path, 'tune_optuna_trials.csv')
            trial_df_tune = pd.read_csv(trial_df_path_tune, index_col=0)

            trial_df = pd.concat([trial_df_warmstart, trial_df_tune], axis=0)
        
        tuned_hyperparameters = dict(trial_df.iloc[trial_df['value'].argmin()])

        model = run_CLIP(None, args, genes_to_peaks_binary_mask,
                               rna_train_loader, atac_train_loader, rna_valid_loader, atac_valid_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length,\
                               target_atac_valid_loader, target_rna_valid_loader, \
                               tuned_hyperparameters, outdir=args.outdir, \
                                do_pretrain_train=False, do_pretrain_valid=False, do_align_train=True, do_align_valid=True)


        try:
            copy2(os.path.join(tune_job_path, 'tune_optuna_trials.csv'), args.outdir)
        except FileNotFoundError:
            pass

        try:
            copy2(os.path.join(tune_job_path, 'warmstart_optuna_trials.csv'), args.outdir)
        except FileNotFoundError:
            pass


    elif args.tune_hyperparameters:

        tuned_hyperparameters = None

        run_CLIP_with_args = lambda trial: run_CLIP(trial, args, genes_to_peaks_binary_mask,
                                                                        rna_train_loader, atac_train_loader, rna_valid_loader, atac_valid_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, target_atac_valid_loader, target_rna_valid_loader, tuned_hyperparameters, outdir=args.outdir, do_pretrain_train=True, do_pretrain_valid=True, do_align_train=False, do_align_valid=False)
                                                                        

        ## pretrain + warm-up
        print('pretrain / warm-start')

        #pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        pruner = optuna.pruners.NopPruner()
        warmstart = optuna.create_study(direction="minimize", pruner=pruner, study_name='tune-align', sampler=optuna.samplers.RandomSampler())

        warmstart.optimize(run_CLIP_with_args, n_trials=args.n_trials, timeout=None)

        if args.n_trials > 0:
            study_summary(warmstart)
            warmstart_df = warmstart.trials_dataframe(attrs=('value', 'params', 'user_attrs')) ## extract dataframe containing data of hyperparameter tuning
            warmstart_df.to_csv(args.outdir + '/warmstart_optuna_trials.csv')

        ## tune
        print('tune')
                
        #pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        pruner = optuna.pruners.NopPruner()
        tune = optuna.create_study(direction="minimize", pruner=pruner, study_name=warmstart.study_name, sampler=optuna.samplers.TPESampler(), load_if_exists=True)
        
        tune.optimize(run_CLIP_with_args, n_trials=int(args.n_trials / 4) + 1, timeout=None)

        if args.n_trials > 0:
            study_summary(tune)
            tune_df = tune.trials_dataframe(attrs=('value', 'params', 'user_attrs'))
            tune_df.to_csv(args.outdir + '/tune_optuna_trials.csv')


    ## print output directory
    print('\n', args.outdir)

# %%
