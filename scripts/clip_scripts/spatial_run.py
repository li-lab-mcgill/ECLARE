import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
import mlflow
from shutil import copy2

#from eclare import merged_dataset_setup, return_setup_func_from_dataset, run_CLIP, study_summary
from eclare import return_setup_func_from_dataset
from eclare.run_utils import run_spatial_CLIP, get_or_create_experiment
from eclare.tune_utils import Optuna_propose_hyperparameters, champion_callback
from eclare.models import get_hparams

if __name__ == "__main__":

    parser = ArgumentParser(description='Spatial CLIP')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', None),
                        help='output directory')
    parser.add_argument('--source_dataset', type=str, default='AD_Anderson_et_al',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--target_dataset', type=str, default='mdd',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_multiome, pbmc_multiome_setup, splatter_sim, toy_simulation')
    parser.add_argument('--total_epochs', type=int, default=2, metavar='E',
                        help='number of epochs for training')
    parser.add_argument('--feature', type=str, default=None, metavar='F',
                        help='Distinctive feature for current job')
    parser.add_argument('--tune_hyperparameters', action='store_true', default=False,
                        help='tune hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=1, metavar='R',
                        help='number of trials used for hyperparameter search')

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

    run_args = {
        'args': args,
        'rna_train_loader': train_loader,
        'rna_valid_loader': valid_loader,
        'train_num_batches': train_num_batches,
        'train_n_batches_str_length': train_n_batches_str_length,
        'train_n_epochs_str_length': train_n_epochs_str_length,
        'valid_num_batches': valid_num_batches,
        'valid_n_batches_str_length': valid_n_batches_str_length,
        'valid_n_epochs_str_length': valid_n_epochs_str_length,
        'target_rna_loader': target_rna_valid_loader,
        'do_align_train': True,
        'do_align_valid': True,
        'do_align_train_eval': False,
        'outdir': args.outdir
    }

    ## get or create experiment
    experiment_id = get_or_create_experiment('Spatial CLIP')
    mlflow.set_experiment(experiment_id)

    if args.tune_hyperparameters:

        # override Optuna's default logging to ERROR only
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        with mlflow.start_run(experiment_id=experiment_id, run_name=args.feature if args.feature else 'Hyperparameter tuning'):

            default_hyperparameters, proposed_hyperparameters = get_hparams()

            def run_spatial_CLIP_wrapper(trial, run_args):
                with mlflow.start_run(experiment_id=experiment_id, run_name=args.feature if args.feature else f'Hyperparameter tuning - run {trial.number}', nested=True):
                    params = Optuna_propose_hyperparameters(trial, proposed_hyperparameters=proposed_hyperparameters, default_hyperparameters_keys=default_hyperparameters.keys())
                    _, valid_loss = run_spatial_CLIP(**run_args, params=params)
                    return valid_loss

            ## create study and run optimization
            study = optuna.create_study(direction='minimize')
            Optuna_objective = lambda trial: run_spatial_CLIP_wrapper(trial, run_args)
            study.optimize(Optuna_objective, n_trials=args.n_trials, callbacks=[champion_callback])

            ## log best trial
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"best_valid_loss": study.best_trial.value})

            ## log metadata
            mlflow.set_tags(tags={
                'default_hyperparameters': default_hyperparameters,
                'proposed_hyperparameters': proposed_hyperparameters
            })

            ## run best model
            model, _ = run_spatial_CLIP(**run_args, params=study.best_params)

            ## infer signature
            x = torch.from_numpy(valid_loader.dataset.adatas[0].X[0].toarray()).to(device=device, dtype=torch.float32).detach()
            signature = mlflow.models.signature.infer_signature(x.cpu().numpy(), model(x).detach().cpu().numpy())

            ## script model
            script_model = torch.jit.script(model)

            ## log model and signature
            mlflow.pytorch.log_model(script_model, "best_model", signature=signature)
            model_uri = mlflow.get_artifact_uri("best_model")


    else:
        tuned_hyperparameters = {}
        run_spatial_CLIP(**run_args, params=tuned_hyperparameters)
            

    ## print output directory
    print('\n', args.outdir)

# %%
