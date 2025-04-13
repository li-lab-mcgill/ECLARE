
import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
from shutil import copy2
import socket

import mlflow
from mlflow import get_artifact_uri, MlflowClient
from mlflow.pytorch import log_model
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec, ParamSchema, ParamSpec

from eclare import return_setup_func_from_dataset
from eclare.tune_utils import tune_CLIP
from eclare.run_utils import run_CLIP, get_or_create_experiment
from eclare.models import get_clip_hparams
from eclare.post_hoc_utils import plot_umap_embeddings, create_celltype_palette, get_latents


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
    parser.add_argument('--batch_size', type=int, default=800, metavar='B',
                        help='size of mini-batch')
    parser.add_argument('--feature', type=str, default=None, metavar='F',
                        help='Distinctive feature for current job')
    parser.add_argument('--tune_hyperparameters', action='store_true', default=False,
                        help='tune hyperparameters using Optuna')
    parser.add_argument('--metric_to_optimize', type=str, default='1-foscttm', metavar='M',
                        help='metric to optimize')
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

    ## get data loaders
    rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
        source_setup_func(args, return_type='loaders', dataset=args.source_dataset)
    
    ## TARGET dataset setup function
    target_setup_func = return_setup_func_from_dataset(args.target_dataset)

    ## missing overlapping_subjects argument if target is MDD (False by default)
    _, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
        target_setup_func(args, return_type='loaders', dataset=args.target_dataset)
    
    run_args = {
        'args': args,
        'rna_train_loader': rna_train_loader,
        'rna_valid_loader': rna_valid_loader,
        'atac_train_loader': atac_train_loader,
        'atac_valid_loader': atac_valid_loader,
        'target_rna_valid_loader': target_rna_valid_loader,
        'target_atac_valid_loader': target_atac_valid_loader,
    }

    ## get clip_job_id from outdir
    clip_job_id = args.outdir.split('/')[-4].split('_')[-1]

    ## get or create mlflow experiment
    if args.target_dataset == 'MDD':
        experiment = get_or_create_experiment(f'clip_mdd_{clip_job_id}')
    else:
        experiment = get_or_create_experiment(f'clip_{clip_job_id}')
        
    experiment_id = experiment.experiment_id
    experiment_name = experiment.name
    mlflow.set_experiment(experiment_name)

    # Get or create experiment type run
    client = MlflowClient()
    experiment_type = 'CLIP'
    exp_type_run_name = f'{experiment_type}_{clip_job_id}'
    exp_type_filter = f"tags.mlflow.runName = '{exp_type_run_name}'"
    exp_type_runs = client.search_runs(experiment_ids=[experiment_id], filter_string=exp_type_filter)
    
    if not exp_type_runs:
        print(f"Creating new experiment type run: {exp_type_run_name}")
        exp_type_run = client.create_run(experiment_id, run_name=exp_type_run_name)
        client.set_tag(exp_type_run.info.run_id, "experiment_type", experiment_type)
    else:
        print(f"Reusing existing experiment type run: {exp_type_run_name}")
        exp_type_run = exp_type_runs[0]  # Take the first (is the most recent?)

    if args.feature:
        run_name = args.feature
    else:
        run_name = 'Hyperparameter tuning' if args.tune_hyperparameters else 'Training'

    with mlflow.start_run(run_id=exp_type_run.info.run_id):
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

            mlflow.set_tag("outdir", args.outdir)
            mlflow.set_tag("hostname", socket.gethostname())

            hyperparameters = get_clip_hparams()
            default_hyperparameters = {k: hyperparameters[k]['default'] for k in hyperparameters}

            ## Run training loops
            if (not args.tune_hyperparameters):

                model, _ = run_CLIP(**run_args, params=default_hyperparameters)
                model_str = "trained_model"

                ## UMAP: source dataset
                rna_latents, atac_latents = get_latents(model, rna_valid_loader.dataset.adatas[0], atac_valid_loader.dataset.adatas[0], return_tensor=False)

                rna_celltypes = rna_valid_loader.dataset.adatas[0].obs['cell_type'].values
                atac_celltypes = atac_valid_loader.dataset.adatas[0].obs['cell_type'].values
                color_map_ct = create_celltype_palette(rna_celltypes.categories, atac_celltypes.categories, plot_color_palette=False)

                rna_condition = ['nan'] * len(rna_celltypes)
                atac_condition = ['nan'] * len(atac_celltypes)

                umap_embedding, umap_figure = plot_umap_embeddings(rna_latents, atac_latents, rna_celltypes, atac_celltypes, rna_condition, atac_condition, color_map_ct, umap_embedding=None)
                umap_figure.suptitle(f"source dataset: {args.source_dataset}", fontsize=14, y=0.98)
                umap_figure.tight_layout()
                umap_figure.savefig(os.path.join(args.outdir, 'source_umap_embeddings.png'))
                mlflow.log_figure(umap_figure, 'source_umap_embeddings.png')

                ## UMAP: target dataset
                rna_latents, atac_latents = get_latents(model, target_rna_valid_loader.dataset.adatas[0], target_atac_valid_loader.dataset.adatas[0], return_tensor=False)

                rna_latents = model(rna_cells.to(device=device), 0)[0].detach().cpu().numpy()   
                atac_latents = model(atac_cells.to(device=device), 1)[0].detach().cpu().numpy()

                rna_celltypes = target_rna_valid_loader.dataset.adatas[0].obs['cell_type'].values
                atac_celltypes = target_atac_valid_loader.dataset.adatas[0].obs['cell_type'].values
                color_map_ct = create_celltype_palette(rna_celltypes.categories, atac_celltypes.categories, plot_color_palette=False)

                rna_condition = ['nan'] * len(rna_celltypes)
                atac_condition = ['nan'] * len(atac_celltypes)
                
                umap_embedding, umap_figure = plot_umap_embeddings(rna_latents, atac_latents, rna_celltypes, atac_celltypes, rna_condition, atac_condition, color_map_ct, umap_embedding=None)
                umap_figure.suptitle(f"target dataset: {args.target_dataset}", fontsize=14, y=0.98)
                umap_figure.tight_layout()
                umap_figure.savefig(os.path.join(args.outdir, 'target_umap_embeddings.png'))
                mlflow.log_figure(umap_figure, 'target_umap_embeddings.png')
    
            elif args.tune_hyperparameters:

                optuna.logging.set_verbosity(optuna.logging.ERROR)

                best_params = tune_CLIP(args, experiment_id)

                ## run best model
                run_args['trial'] = None
                run_args['args'].total_epochs = 100
                model, _ = run_CLIP(**run_args, params=best_params)
                model_str = "best_model"

            ## infer signature
            x = rna_valid_loader.dataset.adatas[0].X[0].toarray()
            params = best_params if args.tune_hyperparameters else default_hyperparameters
            num_units = best_params['num_units'] if args.tune_hyperparameters else default_hyperparameters['num_units']
            
            #signature = mlflow.models.signature.infer_signature(x.cpu().numpy(), model(x, 0)[0].detach().cpu().numpy()) ## TODO: check if can have multiple output signatures
            #param_specs_list = [ParamSpec(name=k, dtype=v.dtype, default=default_hyperparameters[k]) for k, v in params.items()]

            signature = ModelSignature(
                inputs=Schema([TensorSpec(name="cell", type=x.dtype, shape=(-1, None))]),
                outputs=Schema([
                        TensorSpec(name="latent", type=x.dtype, shape=(-1, num_units)),
                        TensorSpec(name="recon", type=x.dtype, shape=(-1, None)),
                    ]),
                #params=ParamSchema(param_specs_list)
            )

            ## script model
            script_model = torch.jit.script(model)

            ## create metadata dict
            metadata = {
                "modalities": ["rna", "atac"],
                "modality_feature_sizes": [n_genes, n_peaks],
                "genes_by_peaks_str": args.genes_by_peaks_str,
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
            }

            ## log model with mlflow.pytorch.log_model
            log_model(script_model,
                model_str,
                signature=signature,
                metadata=metadata)
                #input_example=x,

            ## save both model uri and file uri
            file_uri = get_artifact_uri(model_str)
            run_id = mlflow.active_run().info.run_id
            runs_uri = f"runs:/{run_id}/{model_str}"

            with open(os.path.join(args.outdir, 'model_uri.txt'), 'w') as f:
                f.write(f"{runs_uri}\n")
                f.write(f"{file_uri}\n")

            ## print output directory
            print('\n', args.outdir)

# %%
