import os
from argparse import ArgumentParser
import torch
import pandas as pd
import optuna
from copy import deepcopy
import socket

import mlflow
from mlflow import get_artifact_uri, MlflowClient
from mlflow.pytorch import log_model
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec, ParamSchema, ParamSpec

from eclare.tune_utils import tune_CLIP
from eclare.run_utils import run_CLIP, get_or_create_experiment
from eclare.models import get_clip_hparams
from eclare.setup_utils import return_setup_func_from_dataset
from eclare.post_hoc_utils import plot_umap_embeddings, create_celltype_palette
from eclare.data_utils import fetch_data_from_loader_light


if __name__ == "__main__":

    parser = ArgumentParser(description='CLIP')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', None),
                        help='output directory')
    parser.add_argument('--source_dataset', type=str, default='DLPFC_Anderson',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_10x, pbmc_10x_setup, splatter_sim, toy_simulation')
    parser.add_argument('--target_dataset', type=str, default='MDD',
                        help='current options: CAtlas_Tabula_Sapiens, pbmc_10x, pbmc_10x_setup, splatter_sim, toy_simulation')
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
    parser.add_argument('--metric_to_optimize', type=str, default='compound_metric', metavar='M',
                        help='metric to optimize')
    parser.add_argument('--n_trials', type=int, default=1, metavar='R',
                        help='number of trials used for hyperparameter search')
    parser.add_argument('--tune_id', type=str, default=None,
                        help='ID of job for Optuna hyperparameter tuning')
    parser.add_argument('--job_id', type=str, default=None,
                        help='Job ID for experiment naming')
    parser.add_argument('--valid_subsample', type=int, default=2000,
                        help='number of samples to keep')
    parser.add_argument('--keep_group', type=str, default=None,
                        help='group of samples to keep')

    args = parser.parse_args()
    #args, _ = parser.parse_known_args()
    
    if torch.cuda.is_available():
        print('CUDA available')
        num_gpus = torch.cuda.device_count()
        device   = torch.device(f'cuda:{num_gpus-1}')
        torch.cuda.set_device(device.index)  # default device used by e.g. AnnLoader
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

    if args.keep_group is not None:

        ## get data loaders, with keep_group argument
        rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
            source_setup_func(args, return_type='loaders', keep_group=[args.keep_group])

        args_tmp = deepcopy(args)
        args_tmp.target_dataset = None

        _, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
            target_setup_func(args_tmp, return_type='loaders', keep_group=['']) # keep_group=[''] to keep all subjects

    else:
        ## get data loaders
        rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
            source_setup_func(args, return_type='loaders')

        ## missing overlapping_subjects argument if target is MDD (False by default)
        _, _, _, _, _, target_rna_valid_loader, target_atac_valid_loader, target_atac_valid_num_batches, target_atac_valid_n_batches_str_length, target_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, _ =\
            target_setup_func(args, return_type='loaders')
    
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
    clip_job_id = args.job_id

    ## get or create mlflow experiment
    if args.target_dataset in ['MDD', 'Cortex_Velmeshev']:
        experiment = get_or_create_experiment(f'clip_{args.target_dataset.lower()}_{clip_job_id}')
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

            hyperparameters = get_clip_hparams(context='teacher')
            default_hyperparameters = {k: hyperparameters[k]['default'] for k in hyperparameters}

            ## Run training loops
            if (not args.tune_hyperparameters):

                model, _ = run_CLIP(**run_args, params=default_hyperparameters, device=device)
                model_str = "trained_model"
    
            elif args.tune_hyperparameters:

                optuna.logging.set_verbosity(optuna.logging.ERROR)

                best_params = tune_CLIP(args, experiment_id, run_args)

                ## run best model
                run_args['trial'] = None
                run_args['args'].total_epochs = 100
                model, _ = run_CLIP(**run_args, params=best_params, device=device)
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
                "source_dataset": args.keep_group if args.keep_group is not None else args.source_dataset,
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

            ## create umap embeddings for source dataset
            rna_cells, rna_labels, rna_batches = fetch_data_from_loader_light(rna_valid_loader, subsample=args.valid_subsample, shuffle=False)
            atac_cells, atac_labels, atac_batches = fetch_data_from_loader_light(atac_valid_loader, subsample=args.valid_subsample, shuffle=False)

            #rna_latents = model(rna_cells.to(model.device) if hasattr(model, 'device') else rna_cells.to('cuda' if torch.cuda.is_available() else 'cpu'), modality=0, normalize=0)[0].detach().cpu().numpy()
            #atac_latents = model(atac_cells.to(model.device) if hasattr(model, 'device') else atac_cells.to('cuda' if torch.cuda.is_available() else 'cpu'), modality=1, normalize=0)[0].detach().cpu().numpy()
            rna_latents, _ = model(rna_cells.to(device), modality=0)
            atac_latents, _ = model(atac_cells.to(device), modality=1)

            # Use logits for UMAP embeddings
            rna_latents = rna_latents.detach().cpu().numpy()
            atac_latents = atac_latents.detach().cpu().numpy()

            color_map_ct = create_celltype_palette(rna_labels, atac_labels, plot_color_palette=False)

            rna_condition = ['nan'] * len(rna_labels)
            atac_condition = ['nan'] * len(atac_labels)

            umap_embedding, umap_figure, _ = plot_umap_embeddings(
                rna_latents, atac_latents, rna_labels, atac_labels, rna_condition, atac_condition, color_map_ct, umap_embedding=None
            )
            umap_figure.suptitle(f'UMAP embeddings of CLIP model (source: {args.source_dataset}) based on {args.valid_subsample} cells')
            umap_figure.savefig(os.path.join(args.outdir, 'source_umap_embeddings.png'))
            mlflow.log_figure(umap_figure, 'source_umap_embeddings.png')

            ## print output directory
            print('\n', args.outdir)

# %%
