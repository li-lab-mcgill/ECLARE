
import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
from shutil import copy2
import mlflow

from eclare import return_setup_func_from_dataset
from eclare.tune_utils import Optuna_propose_hyperparameters, champion_callback
from eclare.run_utils import run_CLIP, get_or_create_experiment
from eclare.models import get_clip_hparams

def tune_CLIP(args, experiment_id):

    suggested_hyperparameters = get_clip_hparams()

    def run_CLIP_wrapper(trial, run_args):
        with mlflow.start_run(experiment_id=experiment_id, run_name=args.feature if args.feature else f'Run {trial.number}', nested=True):

            params = Optuna_propose_hyperparameters(trial, suggested_hyperparameters=suggested_hyperparameters)
            run_args['trial'] = trial

            mlflow.log_params(params)
            _, nmi_ari_score = run_CLIP(**run_args, params=params)

            return nmi_ari_score

    ## create study and run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            consider_prior=False,  # not recommended when sampling from categorical variables
            n_startup_trials=0,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Don't prune until this many trials have completed
            n_warmup_steps=20,   # Don't prune until this many steps in each trial
            interval_steps=1,     # Check for pruning every this many steps
        )
    )
    Optuna_objective = lambda trial: run_CLIP_wrapper(trial, run_args)
    study.optimize(Optuna_objective, n_trials=args.n_trials, callbacks=[champion_callback])

    ## log best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({"best_nmi_ari_score": study.best_trial.value})

    ## log metadata
    mlflow.set_tags(tags={
        'suggested_hyperparameters': suggested_hyperparameters
    })

    return study.best_params
    


if __name__ == "__main__":

    parser = ArgumentParser(description='CLIP')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', None),
                        help='output directory')
    parser.add_argument('--source_dataset', type=str, default='AD_Anderson_et_al',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--target_dataset', type=str, default='mdd',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--genes_by_peaks_str', type=str, default='9918_by_43840',
                        help='indicator of peaks to genes mapping to skip processing')
    parser.add_argument('--total_epochs', type=int, default=2, metavar='E',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1000, metavar='B',
                        help='size of mini-batch')
    parser.add_argument('--feature', type=str, default=None, metavar='F',
                        help='Distinctive feature for current job')
    parser.add_argument('--tune_hyperparameters', action='store_true', default=False,
                        help='tune hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=1, metavar='R',
                        help='number of trials used for hyperparameter search')
    parser.add_argument('--tune_id', type=str, default=None,
                        help='ID of job for Optuna hyperparameter tuning')
    parser.add_argument('--subsample', type=int, default=-1,
                        help='number of samples to keep')

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
    rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
        source_setup_func(args, return_type='loaders', dataset=args.source_dataset)
    
    ## missing overlapping_subjects argument if target is MDD (False by default)
    #_, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
    #    target_setup_func(args, return_type='loaders', dataset=args.target_dataset)
    
    run_args = {
        'args': args,
        'rna_train_loader': rna_train_loader,
        'rna_valid_loader': rna_valid_loader,
        'atac_train_loader': atac_train_loader,
        'atac_valid_loader': atac_valid_loader,
    }

    ## get or create mlflow experiment
    experiment_id = get_or_create_experiment('CLIP')
    mlflow.set_experiment(experiment_id)
            
    ## Run training loops
    if (not args.tune_hyperparameters):

        tuned_hyperparameters = {}
        model = run_CLIP(**run_args, params=tuned_hyperparameters)

    else:

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        with mlflow.start_run(experiment_id=experiment_id, run_name=args.feature if args.feature else 'Hyperparameter tuning'):

            best_params = tune_CLIP(args, experiment_id)

            ## run best model
            run_args['trial'] = None
            run_args['args'].total_epochs = 100
            model, _ = run_CLIP(**run_args, params=best_params)

            ## infer signature
            x = torch.from_numpy(rna_valid_loader.dataset.adatas[0].X[0].toarray()).to(device=device, dtype=torch.float32).detach()
            signature = mlflow.models.signature.infer_signature(x.cpu().numpy(), model(x, 0)[0].detach().cpu().numpy()) ## TODO: check if can have multiple output signatures

            ## script model
            script_model = torch.jit.script(model)

            ## log model and signature
            mlflow.pytorch.log_model(script_model, "best_model", signature=signature)
            model_uri = mlflow.get_artifact_uri("best_model")


    ## print output directory
    print('\n', args.outdir)

# %%
