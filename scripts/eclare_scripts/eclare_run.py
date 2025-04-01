#%%
import os
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from glob import glob

import mlflow
from mlflow import get_artifact_uri, MlflowClient
from mlflow.pytorch import log_model
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec, ParamSchema, ParamSpec

from eclare import \
    CLIP, Knowledge_distillation_fn, return_setup_func_from_dataset, teachers_setup, save_latents, fetch_data_from_loaders

from eclare.models import get_clip_hparams
from eclare.run_utils import get_metrics, get_or_create_experiment
from eclare.tune_utils import Optuna_propose_hyperparameters, champion_callback

import optuna
from optuna import Trial, TrialPruned

def tune_ECLARE(args, experiment_name):
    suggested_hyperparameters = get_clip_hparams()

    def run_CLIP_wrapper(trial, run_args):
        with mlflow.start_run(experiment_id=experiment_name, run_name=f'Trial {trial.number}', nested=True):

            params = Optuna_propose_hyperparameters(trial, suggested_hyperparameters=suggested_hyperparameters)
            run_args['trial'] = trial

            mlflow.log_params(params)
            _, metric_to_optimize = run_ECLARE(**run_args, params=params)

            return metric_to_optimize

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
    mlflow.log_metrics({f"best_{args.metric_to_optimize}": study.best_trial.value})

    ## log metadata
    mlflow.set_tags(tags={
        'suggested_hyperparameters': suggested_hyperparameters
    })

    return study.best_params

if __name__ == "__main__":

    parser = ArgumentParser(description='')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', ''),
                        help='output directory')
    parser.add_argument('--clip_job_id', type=str, default=None,
                        help='Job ID of CLIP training')
    parser.add_argument('--total_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--loss_type', type=str, default='knowledge_distillation',
                        help='type of loss to use for training')
    parser.add_argument('--loop_order', type=str, default='batches_first',
                        help='order of loops in training')
    parser.add_argument('--save_latents', action='store_true',
                        help='save latents during training')
    parser.add_argument('--genes_by_peaks_str', type=str, default='10112_by_56354', ## aligned with MDD data
                        help='genes by peaks string')
    parser.add_argument('--source_dataset_embedder', action='store_true', default=False,
                        help='use a dataset embedder')
    parser.add_argument('--distil_lambda', type=float, default=0.1,
                        help='lambda value for MobileCLIP loss')
    parser.add_argument('--valid_subsample', type=int, default=2000,
                        help='number of nuclei to subsample for validation')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='source dataset')
    parser.add_argument('--target_dataset', type=str, default=None,
                        help='target dataset')
    parser.add_argument('--replicate_idx', type=int, default=0,
                        help='replicate index')
    parser.add_argument('--feature', type=str, default=None,
                        help='feature to run')
    parser.add_argument('--tune_hyperparameters', action='store_true', default=False,
                        help='tune hyperparameters')
    args = parser.parse_args()
    #args = parser.parse_known_args()[0]

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

    ## extract data
    outpath = os.environ.get('OUTPATH', args.outdir)

    print('Extracting data')

    target_dataset_og = args.target_dataset
    replicate_idx = str(args.replicate_idx)

    if args.source_dataset is not None:
        model_uri_paths_str = f'clip_*{args.clip_job_id}/{args.target_dataset}/{args.source_dataset}/{replicate_idx}/model_uri.txt'
    else:
        model_uri_paths_str = f'clip_*{args.clip_job_id}/{target_dataset_og}/**/{replicate_idx}/model_uri.txt'

    model_uri_paths = glob(os.path.join(outpath, model_uri_paths_str))
    assert len(model_uri_paths) > 0, f'Model URI path not found @ {model_uri_paths_str}'

    ##Get student loaders
    args_tmp = deepcopy(args)
    args_tmp.source_dataset = args.target_dataset
    args_tmp.target_dataset = 'MDD'

    student_setup_func = return_setup_func_from_dataset(args.target_dataset)
    student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_total_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_total_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
        student_setup_func(args_tmp, pretrain=None, return_type='loaders')
    
    ## Setup teachers
    datasets, models, target_rna_train_loaders, target_atac_train_loaders, target_rna_valid_loaders, target_atac_valid_loaders = \
        teachers_setup(model_uri_paths, device, args)
    
    from optuna import Trial, TrialPruned
    from argparse import Namespace
    def eclare_pass(
        student_rna_iterator,
        student_atac_iterator,
        target_rna_iterators,
        target_atac_iterators,
        student_model,
        optimizer,
        knowledge_distillation_fn,
        loop_order
        ):

        # Initialize iterators for each data loader
        target_rna_iterators = {dataset: iter(loader) for dataset, loader in target_rna_iterators.items()}
        target_atac_iterators = {dataset: iter(loader) for dataset, loader in target_atac_iterators.items()}

        student_rna_iterator = iter(student_rna_iterator)
        student_atac_iterator = iter(student_atac_iterator)

        # Determine the number of batches per dataset
        num_batches_per_dataset = {
            dataset: min(
                len(target_rna_iterators[dataset]), len(target_atac_iterators[dataset]), len(student_rna_iterator), len(student_atac_iterator)
            )
            for dataset in datasets
        }

        # Determine the global minimum number of batches
        if knowledge_distillation_fn.paired:
            num_batches = min(num_batches_per_dataset.values())
        else:
            num_batches = min(num_batches_per_dataset.values()) - 1  # if not paired, unlikely that last batches have same size. so better to trim by one batch

        # Define outer and inner iterables based on loop_order
        if loop_order == 'datasets_first':
            outer_iterable = datasets
            inner_iterable = range(num_batches)

        elif loop_order == 'batches_first':
            outer_iterable = range(num_batches)
            inner_iterable = datasets


        # Initialize dictionaries to accumulate losses
        epoch_total_losses, epoch_align_loss, epoch_distil_loss = [{dataset: 0.0 for dataset in datasets+[target_dataset_og]} for _ in range(3)]

        for outer in tqdm(outer_iterable):

            ## Extract student data once only for each outer loop iteration, not at every inner loop iteration
            if loop_order == 'batches_first':
                try:
                    student_rna_dat = next(student_rna_iterator)
                    student_atac_dat = next(student_atac_iterator)
                except StopIteration:
                    # If any iterator runs out of data, continue to the next iteration
                    continue

            ## Project student RNA data (target)
            student_rna_cells = student_rna_dat.X.float().to(device)
            student_rna_latents, _ = student_model(student_rna_cells, modality=0)
            student_rna_celltypes = student_rna_dat.obs['cell_type'].to_list()

            ## Project student ATAC data (target)
            student_atac_cells = student_atac_dat.X.float().to(device)
            student_atac_latents, _ = student_model(student_atac_cells, modality=1)
            student_atac_celltypes = student_atac_dat.obs['cell_type'].to_list()

            ## Initialize list of dataset distil losses
            distil_losses, distil_losses_T = [], []
            align_losses, align_losses_T = [], []
            offsets, offsets_T = [], []

            for inner in (tqdm(inner_iterable, leave=False) if loop_order == 'datasets_first' else inner_iterable):
                if loop_order == 'datasets_first':
                    dataset = outer
                    batch_idx = inner
                else:
                    dataset = inner
                    batch_idx = outer

                # Retrieve the next batch from each iterator
                try:
                    target_rna_dat = next(target_rna_iterators[dataset])
                    target_atac_dat = next(target_atac_iterators[dataset])

                except StopIteration:
                    # If any iterator runs out of data, continue to the next iteration
                    continue

                # Load the model for that dataset
                model = models[dataset]

                # Project target RNA data
                target_rna_cells = target_rna_dat.X.float().to(device)
                target_rna_latents, _ = model(target_rna_cells, modality=0)
                target_rna_celltypes = target_rna_dat.obs['cell_type'].to_list()

                # Project target ATAC data
                target_atac_cells = target_atac_dat.X.float().to(device)
                target_atac_latents, _ = model(target_atac_cells, modality=1)
                target_atac_celltypes = target_atac_dat.obs['cell_type'].to_list()

                ## Ensure that the target latents are detached
                target_rna_latents = target_rna_latents.detach()
                target_atac_latents = target_atac_latents.detach()

                assert (student_rna_dat.obs_names == target_rna_dat.obs_names).all()
                assert (student_atac_dat.obs_names == target_atac_dat.obs_names).all()

                ## compute teacher losses
                distil_loss, distil_loss_T, align_loss_scaled, align_loss_T_scaled, offset, offset_T = \
                    knowledge_distillation_fn(student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, 'teacher')

                distil_losses.append(distil_loss)
                align_losses.append(align_loss_scaled)
                distil_losses_T.append(distil_loss_T)
                align_losses_T.append(align_loss_T_scaled)
                offsets.append(offset)
                offsets_T.append(offset_T)

                ## get "aggregated" losses for distillation and alignment
                distil_loss = 0.5 * (distil_loss + distil_loss_T)
                align_loss_scaled = 0.5 * (align_loss_scaled + align_loss_T_scaled)

                ## get total loss
                lambd = args.distil_lambda
                total_loss = (lambd * distil_loss) + ((1-lambd) * align_loss_scaled)

                # Accumulate scalar loss values for logging
                epoch_distil_loss[dataset] += distil_loss.mean().item()
                epoch_align_loss[dataset] += align_loss_scaled.mean().item()
                epoch_total_losses[dataset] += total_loss.mean().item()


            ## Compute mean distillation loss
            distil_losses = torch.stack(distil_losses)
            align_losses = torch.stack(align_losses)
            distil_losses_T = torch.stack(distil_losses_T)
            align_losses_T = torch.stack(align_losses_T)
            offsets = torch.stack(offsets)
            offsets_T = torch.stack(offsets_T)

            mean_distil_loss, _ = \
                knowledge_distillation_fn.distil_loss_weighting( distil_losses, distil_losses_T, (offsets - align_losses), (offsets_T - align_losses_T))

            ## Get student align loss
            _, _, align_loss_scaled, _, _, _ = \
                knowledge_distillation_fn(student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, 'student')

            ## Compute total loss as convex combination of CLIP loss and average distillation loss
            total_loss = (args.distil_lambda * mean_distil_loss) + ((1-args.distil_lambda) * align_loss_scaled)
            epoch_distil_loss[target_dataset_og] += mean_distil_loss.item()
            epoch_total_losses[target_dataset_og] += total_loss.item()

            if (optimizer is not None):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            

        # Prepare metrics dictionary for MLflow logging
        metrics_dict = {}
        
        # Add metrics
        for dataset in datasets+[target_dataset_og]:
            metrics_dict[f'total_loss_{dataset}']   = epoch_total_losses[dataset] / num_batches
            metrics_dict[f'distil_loss_{dataset}']  = epoch_distil_loss[dataset] / num_batches
            metrics_dict[f'align_loss_{dataset}']   = epoch_align_loss[dataset] / num_batches # could ignore non-target datasets since constant value across epochs
                
        return metrics_dict


    def run_ECLARE(
        args: Namespace,
        student_rna_train_loader,
        student_rna_valid_loader,
        student_atac_train_loader,
        student_atac_valid_loader,
        target_rna_train_loaders,
        target_atac_train_loaders,
        target_rna_valid_loaders,
        target_atac_valid_loaders,
        trial: Trial = None,
        params: dict = {},
        device: str = 'cpu',
        ):

        ## Get number of genes and peaks from genes_by_peaks_str
        n_genes = int(args.genes_by_peaks_str.split('_')[0])
        n_peaks = int(args.genes_by_peaks_str.split('_')[-1])

        # Instantiate student model with optimized parameters
        student_model = CLIP(n_peaks=n_peaks, n_genes=n_genes, **params).to(device=device)
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=0.01)

        # Instantiate the knowledge distillation loss function
        paired = (args.target_dataset != 'MDD')
        knowledge_distillation_fn = Knowledge_distillation_fn(device=device, student_temperature=1, target_temperature=1, paired=paired, weigh_distil_by_align_type='none')

        # Define loop_order parameter
        loop_order = args.loop_order  # 'datasets_first' or 'batches_first'

        # Log model parameters with MLflow
        mlflow.log_params(params)
        mlflow.log_param("paired", paired)
        mlflow.log_param("loop_order", loop_order)
        mlflow.log_param("n_genes", n_genes)
        mlflow.log_param("n_peaks", n_peaks)

        # Define loop_order parameter
        loop_order = args.loop_order  # 'datasets_first' or 'batches_first'

        ## Get metrics -- valid dataset
        with torch.inference_mode():

            student_model.eval()

            valid_losses = eclare_pass(
                student_rna_valid_loader,
                student_atac_valid_loader,
                target_rna_valid_loaders,
                target_atac_valid_loaders,
                student_model,
                None,
                knowledge_distillation_fn,
                loop_order
            )

            ## get metrics
            metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device)
            metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})

            # Log all metrics at once with MLflow
            mlflow.log_metrics(metrics, step=0)


        print('Iterating over epochs, batches & datasets')
        for epoch in range(args.total_epochs):

            ## Set gradients to zero, but why??
            optimizer.zero_grad()

            # Start the loops -- training data
            student_model.train()

            train_losses = eclare_pass(
                student_rna_train_loader,
                student_atac_train_loader,
                target_rna_train_loaders,
                target_atac_train_loaders,
                student_model,
                optimizer,
                knowledge_distillation_fn,
                loop_order
            )

            with torch.inference_mode():

                student_model.eval()

                valid_losses = eclare_pass(
                    student_rna_valid_loader,
                    student_atac_valid_loader,
                    target_rna_valid_loaders,
                    target_atac_valid_loaders,
                    student_model,
                    None,
                    knowledge_distillation_fn,
                    loop_order
                )

            # Add performance metrics from get_metrics
            metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device)
            metrics.update({f'train_{k}': v for k, v in train_losses.items() if ~np.isnan(v)})
            metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})
            
            # Log all metrics at once with MLflow
            mlflow.log_metrics(metrics, step=epoch+1)

            ## early stopping with Optuna pruner
            if trial is not None:
                metric_to_optimize = metrics.get('nmi_ari', 0)  # Default to nmi_ari if not specified
                trial.report(metric_to_optimize, step=epoch+1)
                if trial.should_prune():
                    raise TrialPruned()

        return student_model, metrics
    
    
    run_args = {
        'args': args,
        'student_rna_train_loader': student_rna_train_loader,
        'student_rna_valid_loader': student_rna_valid_loader,
        'student_atac_train_loader': student_atac_train_loader,
        'student_atac_valid_loader': student_atac_valid_loader,
        'target_rna_train_loaders': target_rna_train_loaders,
        'target_atac_train_loaders': target_atac_train_loaders,
        'target_rna_valid_loaders': target_rna_valid_loaders,
        'target_atac_valid_loaders': target_atac_valid_loaders,
    }

    ## get or create mlflow experiment
    experiment = get_or_create_experiment(f'clip_{args.clip_job_id}')
    experiment_name = experiment.name
    experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    # 1. Get or create experiment type run
    experiment_type = 'ECLARE' if args.source_dataset is None else 'KD_CLIP'
    exp_type_run_name = f'{experiment_type}_{args.clip_job_id}'
    exp_type_filter = f"tags.mlflow.runName = '{exp_type_run_name}'"
    exp_type_runs = client.search_runs(experiment_ids=[experiment_id], filter_string=exp_type_filter)
    
    if not exp_type_runs:
        print(f"Creating new experiment type run: {exp_type_run_name}")
        exp_type_run = client.create_run(experiment_id, run_name=exp_type_run_name)
        client.set_tag(exp_type_run.info.run_id, "experiment_type", experiment_type)
    else:
        print(f"Reusing existing experiment type run: {exp_type_run_name}")
        exp_type_run = exp_type_runs[0]  # Take the first (is the most recent?)

    # 2. Get or create data run with more specific filtering
    data_run_name = f'{args.source_dataset}_to_{args.target_dataset}' if args.source_dataset is not None else args.target_dataset
    data_run_filter = (
        f"tags.mlflow.runName = '{data_run_name}' and "
        f"tags.mlflow.parentRunId = '{exp_type_run.info.run_id}'"
    )
    data_runs = client.search_runs(experiment_ids=[experiment_id], filter_string=data_run_filter)

    if not data_runs:
        print(f"Creating new data run: {data_run_name}")
        data_run = client.create_run(
            experiment_id,
            run_name=data_run_name,
            tags={
                "mlflow.parentRunId": exp_type_run.info.run_id,
                "data_run_name": data_run_name
            }
        )
    else:
        print(f"Reusing existing data run: {data_run_name}")
        data_run = data_runs[0]  # Take the first RUNNING run

    # 3. Create new replicate run
    replicate_name = str(args.replicate_idx)
    replicate_run = client.create_run(
        experiment_id,
        run_name=replicate_name,
        tags={
            "mlflow.parentRunId": data_run.info.run_id,
            "replicate_idx": replicate_name,
            "outdir": args.outdir
        }
    )
    print(f"Created new replicate run: {replicate_name}")

    # Debug information
    print(f"\nRun hierarchy:")
    print(f"└── {exp_type_run_name} (ID: {exp_type_run.info.run_id})")
    print(f"    └── {data_run_name} (ID: {data_run.info.run_id})")
    print(f"        └── replicate_{replicate_name} (ID: {replicate_run.info.run_id})")

    # Use the runs
    with mlflow.start_run(run_id=exp_type_run.info.run_id):
        with mlflow.start_run(run_id=data_run.info.run_id, nested=True):
            with mlflow.start_run(run_id=replicate_run.info.run_id, nested=True):
                print(f"Running replicate {args.replicate_idx} for {data_run_name}")

                hyperparameters = get_clip_hparams()
                default_hyperparameters = {k: hyperparameters[k]['default'] for k in hyperparameters}

                if not args.tune_hyperparameters:

                    student_model, metrics_dict = run_ECLARE(**run_args, params=default_hyperparameters, device=device)
                    model_str = "trained_model"

                else:
                    optuna.logging.set_verbosity(optuna.logging.ERROR)
                    best_params = tune_ECLARE(args, experiment_name)

                    ## run best model
                    run_args['trial'] = None
                    run_args['args'].total_epochs = 100
                    model, _ = run_ECLARE(**run_args, params=best_params, device=device)
                    model_str = "best_model"

                ## infer signature
                x = student_rna_valid_loader.dataset.adatas[0].X[0].toarray()
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
                script_model = torch.jit.script(student_model)

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