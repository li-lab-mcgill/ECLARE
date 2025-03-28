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
from mlflow import get_artifact_uri
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

def tune_ECLARE(args, experiment_id):
    suggested_hyperparameters = get_clip_hparams()

    def run_CLIP_wrapper(trial, run_args):
        with mlflow.start_run(experiment_id=experiment_id, run_name=f'Trial {trial.number}', nested=True):

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

    if args.target_dataset is not None:
        target_dataset_og = args.target_dataset
        replicate_idx = str(args.replicate_idx)

        if args.source_dataset is not None:
            NotImplementedError('Need to specify source dataset')
        else:
            model_uri_paths = glob(os.path.join(outpath, f'clip_*{args.clip_job_id}/{target_dataset_og}/**/{replicate_idx}/model_uri.txt'))
            assert len(model_uri_paths) > 0, f'Model URI path not found for {target_dataset_og} ({replicate_idx})'
    else:
        raise ValueError(f'Need to specify target dataset')

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

        if paired:
            student_foscttm_per_ct_valid = pd.DataFrame()
            teacher_foscttm_per_ct_valid = pd.DataFrame(columns=datasets)

        # Define loop_order parameter
        loop_order = args.loop_order  # 'datasets_first' or 'batches_first'

        print('Iterating over epochs, batches & datasets')
        for epoch in range(args.total_epochs):

            # Initialize dictionaries to accumulate losses
            epoch_losses, epoch_align_loss, epoch_distil_loss = [{dataset: 0.0 for dataset in datasets+[target_dataset_og]} for _ in range(3)]
            epoch_losses_valid, epoch_align_losses_valid, epoch_distil_losses_valid = {}, {}, {}
            epoch_ilisis, epoch_clisis, epoch_nmi, epoch_ari, epoch_foscttm, epoch_rank_score = {}, {}, {}, {}, {}, {}

            if args.save_latents:
                all_rna_latents_valid, all_atac_latents_valid = [], []
                all_rna_celltypes_valid, all_atac_celltypes_valid = [], []

            ## Get metrics -- valid dataset
            with torch.inference_mode():

                distil_losses_valid, distil_losses_T_valid = [], []
                align_losses_valid, align_losses_T_valid = [], []
                offsets_valid, offsets_T_valid = [], []

                ## Get student data and latents
                student_rna_cells_valid, student_rna_celltypes_valid, student_atac_cells_valid, student_atac_celltypes_valid, rna_nuclei_idx_valid, atac_nuclei_idxs_valid =\
                    fetch_data_from_loaders(student_rna_valid_loader, student_atac_valid_loader, paired=paired, subsample=args.valid_subsample) # reuse same nuclei indices for all student and teacher datasets, or else distillation loss not sensible

                student_rna_latents_valid, _ = student_model(student_rna_cells_valid, modality=0)
                student_atac_latents_valid, _ = student_model(student_atac_cells_valid, modality=1)

                ## normalize latents (already being normalized in loss_fn)
                student_rna_latents_valid = torch.nn.functional.normalize(student_rna_latents_valid, p=2, dim=1)
                student_atac_latents_valid = torch.nn.functional.normalize(student_atac_latents_valid, p=2, dim=1)    
                
                ## loop through teachers
                for dataset in datasets:

                    dataset_embedding = None

                    model = models[dataset]
                    target_rna_valid_loader = target_rna_valid_loaders[dataset]
                    target_atac_valid_loader = target_atac_valid_loaders[dataset]

                    target_rna_cells_valid, target_rna_celltypes_valid, target_atac_cells_valid, target_atac_celltypes_valid, _, _        =\
                        fetch_data_from_loaders(target_rna_valid_loader, target_atac_valid_loader, paired=paired, subsample=args.valid_subsample, rna_cells_idx=rna_nuclei_idx_valid, atac_cells_idx=atac_nuclei_idxs_valid)

                    target_rna_latents_valid, _ = model(target_rna_cells_valid, modality=0)
                    target_atac_latents_valid, _ = model(target_atac_cells_valid, modality=1)

                    ## compute teacher losses, in principle no need to recompute align losses for teachers since fixed
                    distil_loss_valid, distil_loss_valid_T, align_loss_valid, align_loss_T_valid, offset_valid, offset_T_valid =\
                        knowledge_distillation_fn(student_rna_latents_valid, student_atac_latents_valid, target_rna_latents_valid, target_atac_latents_valid, 'teacher')

                    ## update losses for teacher
                    align_losses_valid.append(align_loss_valid)
                    align_losses_T_valid.append(align_loss_valid)
                    distil_losses_valid.append(distil_loss_valid)
                    distil_losses_T_valid.append(distil_loss_valid)
                    offsets_valid.append(offset_valid)
                    offsets_T_valid.append(offset_T_valid)
                    
                    ## get "aggregated" losses for distillation and alignment
                    distil_loss_valid = 0.5 * (distil_loss_valid + distil_loss_valid_T)
                    align_loss_valid = 0.5 * (align_loss_valid + align_loss_T_valid)

                    epoch_distil_losses_valid[dataset] = distil_loss_valid.mean().item()
                    epoch_align_losses_valid[dataset] = align_loss_valid.mean().item()
                    epoch_losses_valid[dataset] = args.distil_lambda*epoch_distil_losses_valid[dataset] + (1-args.distil_lambda)*epoch_align_losses_valid[dataset]

                    if epoch == 0:
                        metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device)

                ## Compute mean distillation loss
                distil_losses_valid     = torch.stack(distil_losses_valid)
                distil_losses_T_valid   = torch.stack(distil_losses_T_valid)
                align_losses_valid      = torch.stack(align_losses_valid)
                align_losses_T_valid    = torch.stack(align_losses_T_valid)
                offsets_valid           = torch.stack(offsets_valid)
                offsets_T_valid         = torch.stack(offsets_T_valid)

                ## Compute distillation loss weighted by alignment loss, if more than one teacher
                mean_distil_loss_valid, _ = knowledge_distillation_fn.distil_loss_weighting( distil_losses_valid, distil_losses_T_valid, (offsets_valid - align_losses_valid), (offsets_T_valid - align_losses_T_valid))

                ## Compute student alignment loss
                _, _, align_loss_scaled_valid, _, _, _ = knowledge_distillation_fn(student_rna_latents_valid, student_atac_latents_valid, target_rna_latents_valid, target_atac_latents_valid, 'student')

                ## Set align loss scale if first epoch
                if epoch == 0:
                    align_loss_scale = (mean_distil_loss_valid / align_loss_scaled_valid).detach().cpu().numpy().item()
                    knowledge_distillation_fn.align_loss_scale = align_loss_scale
                    print(f'Align loss scale: {align_loss_scale}')
                    mlflow.log_param("align_loss_scale", align_loss_scale)

                    ## Retroactively scale teacher & student CLIP losses
                    epoch_align_losses_valid    = {dataset: align_loss_scale * epoch_align_losses_valid[dataset] for dataset in datasets} # not sure if updating correctly
                    epoch_losses_valid          = {dataset: args.distil_lambda * epoch_distil_losses_valid[dataset] + (1-args.distil_lambda) * epoch_align_losses_valid[dataset] for dataset in datasets}
                    align_loss_scaled_valid     = align_loss_scale * align_loss_scaled_valid


                ## Compute total loss as convex combination of CLIP loss and average distillation loss
                total_loss_valid = (args.distil_lambda * mean_distil_loss_valid) + ((1-args.distil_lambda) * align_loss_scaled_valid)

                ## Get metrics for student
                metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device)

                ## Log validation losses for student
                epoch_align_losses_valid[target_dataset_og] = align_loss_scaled_valid.item()
                epoch_distil_losses_valid[target_dataset_og] = mean_distil_loss_valid.item()
                epoch_losses_valid[target_dataset_og] = total_loss_valid.item()

                if args.save_latents:
                    save_latents(student_rna_latents_valid, student_atac_latents_valid, student_rna_celltypes_valid, student_atac_celltypes_valid, epoch, args.total_epochs, args.outdir)

            ## Set gradients to zero
            optimizer.zero_grad()

            # Initialize iterators for each data loader
            target_rna_train_iterators = {dataset: iter(loader) for dataset, loader in target_rna_train_loaders.items()}
            target_atac_train_iterators = {dataset: iter(loader) for dataset, loader in target_atac_train_loaders.items()}

            student_rna_train_iterator = iter(student_rna_train_loader)
            student_atac_train_iterator = iter(student_atac_train_loader)

            # Determine the number of batches per dataset
            num_batches_per_dataset = {
                dataset: min(
                    len(target_rna_train_loaders[dataset]), len(target_atac_train_loaders[dataset]), len(student_rna_train_loader), len(student_atac_train_loader)
                )
                for dataset in datasets
            }

            # Determine the global minimum number of batches
            if paired:
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

            # Start the loops -- training data
            for outer in tqdm(outer_iterable, desc=f"Epoch {epoch+1}"):

                ## Extract student data once only for each outer loop iteration, not at every inner loop iteration
                if loop_order == 'batches_first':
                    try:
                        student_rna_dat = next(student_rna_train_iterator)
                        student_atac_dat = next(student_atac_train_iterator)
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
                        target_rna_dat = next(target_rna_train_iterators[dataset])
                        target_atac_dat = next(target_atac_train_iterators[dataset])

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

                    dataset_embedding = None

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

                    # Accumulate scalar loss values for logging
                    epoch_distil_loss[dataset] += distil_loss.mean().item()
                    epoch_align_loss[dataset] += align_loss_scaled.mean().item()
                    epoch_losses[dataset] += args.distil_lambda*distil_loss.mean().item() + (1-args.distil_lambda)*align_loss_scaled.mean().item()


                ## Compute mean distillation loss
                distil_losses = torch.stack(distil_losses)
                align_losses = torch.stack(align_losses)
                distil_losses_T = torch.stack(distil_losses_T)
                align_losses_T = torch.stack(align_losses_T)
                offsets = torch.stack(offsets)
                offsets_T = torch.stack(offsets_T)

                mean_distil_loss, _ = knowledge_distillation_fn.distil_loss_weighting( distil_losses, distil_losses_T, (offsets - align_losses), (offsets_T - align_losses_T))

                ## Get student align loss
                _, _, align_loss_scaled, _, _, _ = knowledge_distillation_fn(student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, 'student')

                ## Compute total loss as convex combination of CLIP loss and average distillation loss
                total_loss = (args.distil_lambda * mean_distil_loss) + ((1-args.distil_lambda) * align_loss_scaled)
                total_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                
                epoch_distil_loss[target_dataset_og] += mean_distil_loss.item()
                epoch_losses[target_dataset_og] += total_loss.item()

            # Prepare metrics dictionary for MLflow logging
            metrics_dict = {}
            
            # Add training metrics
            for dataset in datasets+[target_dataset_og]:
                metrics_dict[f'train_loss_{dataset}'] = epoch_losses[dataset] / num_batches
                metrics_dict[f'train_distil_loss_{dataset}'] = epoch_distil_loss[dataset] / num_batches
            
            # Add validation metrics
            metrics_dict[f'valid_loss_{target_dataset_og}'] = epoch_losses_valid[target_dataset_og]
            metrics_dict[f'valid_align_loss_{target_dataset_og}'] = epoch_align_losses_valid[target_dataset_og]
            metrics_dict[f'valid_distil_loss_{target_dataset_og}'] = epoch_distil_losses_valid[target_dataset_og]
            
            # Add performance metrics from get_metrics
            metrics_dict.update(metrics)
            
            # Log all metrics at once with MLflow
            mlflow.log_metrics(metrics_dict, step=epoch+1)

            ## early stopping with Optuna pruner
            if trial is not None:
                metric_to_optimize = metrics.get('nmi_ari', 0)  # Default to nmi_ari if not specified
                trial.report(metric_to_optimize, step=epoch+1)
                if trial.should_prune():
                    raise TrialPruned()

        return student_model, metrics_dict
    
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
    experiment_id = get_or_create_experiment('ECLARE')
    mlflow.set_experiment(experiment_id)

    if args.feature:
        run_name = args.feature
    else:
        run_name = 'Hyperparameter tuning' if args.tune_hyperparameters else 'Training'

    ## run experiment
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):

        mlflow.set_tag("outdir", args.outdir)

        hyperparameters = get_clip_hparams()
        default_hyperparameters = {k: hyperparameters[k]['default'] for k in hyperparameters}

        if (not args.tune_hyperparameters):
            student_model, metrics_dict = run_ECLARE(**run_args, params=default_hyperparameters, device=device)
            model_str = "trained_model"

        else:

            optuna.logging.set_verbosity(optuna.logging.ERROR)

            best_params = tune_ECLARE(args, experiment_id)

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