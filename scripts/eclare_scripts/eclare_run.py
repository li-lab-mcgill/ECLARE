#%%
import os
from copy import deepcopy
from argparse import ArgumentParser
import torch
from glob import glob
import optuna
import socket

import mlflow
from mlflow import get_artifact_uri, MlflowClient
from mlflow.pytorch import log_model
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec

from eclare.setup_utils import return_setup_func_from_dataset, teachers_setup
from eclare.models import get_clip_hparams
from eclare.run_utils import get_or_create_experiment, run_ECLARE
from eclare.tune_utils import tune_ECLARE
from eclare.post_hoc_utils import plot_umap_embeddings, create_celltype_palette, get_latents
from eclare.data_utils import fetch_data_from_loader_light


if __name__ == "__main__":

    parser = ArgumentParser(description='')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', ''),
                        help='output directory')
    parser.add_argument('--clip_job_id', type=str, default=None,
                        help='Job ID of CLIP training')
    parser.add_argument('--experiment_job_id', type=str, default=None,
                        help='Job ID of experiment')
    parser.add_argument('--total_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=800,
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
    parser.add_argument('--n_trials', type=int, default=10, metavar='N',
                        help='number of trials for hyperparameter tuning')
    parser.add_argument('--metric_to_optimize', type=str, default='compound_metric', metavar='M',
                        help='metric to optimize during hyperparameter tuning')
    parser.add_argument('--ignore_sources', nargs='+', type=str, default=[None],
                        help='List of sources to ignore')
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

    ## get model uri paths based on experiment type

    # kd-clip
    if (args.source_dataset is not None) and (args.target_dataset != 'MDD'):
        model_uri_paths_str = f'clip_*{args.clip_job_id}/{args.target_dataset}/{args.source_dataset}/{replicate_idx}/model_uri.txt'

    # eclare
    elif (args.source_dataset is None) and (args.target_dataset != 'MDD'):
        model_uri_paths_str = f'clip_*{args.clip_job_id}/{target_dataset_og}/**/{replicate_idx}/model_uri.txt'

    # kd-clip mdd
    elif (args.source_dataset is not None) and (args.target_dataset == 'MDD'):
        model_uri_paths_str = f'clip_mdd_*{args.clip_job_id}/{args.target_dataset}/{args.source_dataset}/{replicate_idx}/model_uri.txt'

    # eclare mdd
    elif (args.source_dataset is None) and (args.target_dataset == 'MDD'):
        model_uri_paths_str = f'clip_mdd_*{args.clip_job_id}/{target_dataset_og}/**/{replicate_idx}/model_uri.txt'


    model_uri_paths = glob(os.path.join(outpath, model_uri_paths_str))
    assert len(model_uri_paths) > 0, f'Model URI path not found @ {model_uri_paths_str}'

    ##Get student loaders
    args_tmp = deepcopy(args)
    args_tmp.source_dataset = args.target_dataset

    if target_dataset_og == 'DLPFC_Anderson':
        args_tmp.target_dataset = None  # could be any dataset, specified to skip processing (or do further zero-shot tasks)
    else:
        args_tmp.target_dataset = 'MDD'

    student_setup_func = return_setup_func_from_dataset(args.target_dataset)
    student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_total_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_total_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
        student_setup_func(args_tmp, return_type='loaders')
    
    ## Setup teachers
    datasets, models, teacher_rna_train_loaders, teacher_atac_train_loaders, teacher_rna_valid_loaders, teacher_atac_valid_loaders = \
        teachers_setup(model_uri_paths, args, device)
    

    run_args = {
        'args': args,
        'student_rna_train_loader': student_rna_train_loader,
        'student_rna_valid_loader': student_rna_valid_loader,
        'student_atac_train_loader': student_atac_train_loader,
        'student_atac_valid_loader': student_atac_valid_loader,
        'teacher_rna_train_loaders': teacher_rna_train_loaders,
        'teacher_atac_train_loaders': teacher_atac_train_loaders,
        'teacher_rna_valid_loaders': teacher_rna_valid_loaders,
        'teacher_atac_valid_loaders': teacher_atac_valid_loaders,
        'teacher_models': models,
    }

    ## get or create mlflow experiment
    if args.target_dataset == 'MDD':
        experiment = get_or_create_experiment(f'clip_mdd_{args.clip_job_id}')
    else:
        experiment = get_or_create_experiment(f'clip_{args.clip_job_id}')

    experiment_name = experiment.name
    experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    # 1. Get or create experiment type run
    experiment_type = 'ECLARE' if args.source_dataset is None else 'KD_CLIP'
    exp_type_run_name = f'{experiment_type}_{args.experiment_job_id}'
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
    if args.source_dataset is not None:
        data_run_and_replicate_name = f'{args.source_dataset}-to-{args.target_dataset}-{args.replicate_idx}'
    elif (args.source_dataset is None) and (args.ignore_sources is not None):
        data_run_and_replicate_name = f'{args.target_dataset}-IGNORE-{"_".join(args.ignore_sources)}-{args.replicate_idx}'
    else:
        data_run_and_replicate_name = f'{args.target_dataset}-{args.replicate_idx}'

    data_run_filter = (
        f"tags.mlflow.runName = '{data_run_and_replicate_name}' and "
        f"tags.mlflow.parentRunId = '{exp_type_run.info.run_id}'"
    )
    data_runs = client.search_runs(experiment_ids=[experiment_id], filter_string=data_run_filter)

    if not data_runs:
        print(f"Creating new data run: {data_run_and_replicate_name}")
        data_run = client.create_run(
            experiment_id,
            run_name=data_run_and_replicate_name,
            tags={
                "mlflow.parentRunId": exp_type_run.info.run_id,
                "data_run_name": data_run_and_replicate_name
            }
        )
    else:
        print(f"Reusing existing data run: {data_run_and_replicate_name}")
        data_run = data_runs[0]  # Take the first RUNNING run

    # Use the runs
    with mlflow.start_run(run_id=exp_type_run.info.run_id):
        with mlflow.start_run(run_id=data_run.info.run_id, nested=True):
            print(f"Running {data_run_and_replicate_name}")

            mlflow.set_tag("ignore_sources", args.ignore_sources)
            mlflow.set_tag("outdir", args.outdir)
            mlflow.set_tag("hostname", socket.gethostname())

            hyperparameters = get_clip_hparams(context='student')
            default_hyperparameters = {k: hyperparameters[k]['default'] for k in hyperparameters}

            if not args.tune_hyperparameters:

                student_model, metrics_dict = run_ECLARE(**run_args, params=default_hyperparameters, device=device)
                model_str = "trained_model"

            else:
                optuna.logging.set_verbosity(optuna.logging.ERROR)
                best_params = tune_ECLARE(args, experiment_id, run_args, device)

                ## run best model
                run_args['trial'] = None
                run_args['args'].total_epochs = 100
                student_model, _ = run_ECLARE(**run_args, params=best_params, device=device)
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

            ## create umap embeddings
            student_rna_cells, student_rna_labels, student_rna_batches = fetch_data_from_loader_light(student_rna_valid_loader, subsample=args.valid_subsample, shuffle=False)
            student_atac_cells, student_atac_labels, student_atac_batches = fetch_data_from_loader_light(student_atac_valid_loader, subsample=args.valid_subsample, shuffle=False)

            rna_latents = student_model(student_rna_cells.to(device), modality=0)[0].detach().cpu().numpy()
            atac_latents = student_model(student_atac_cells.to(device), modality=1)[0].detach().cpu().numpy()

            color_map_ct = create_celltype_palette(student_rna_labels, student_atac_labels, plot_color_palette=False)

            rna_condition = ['nan'] * len(student_rna_labels)
            atac_condition = ['nan'] * len(student_atac_labels)
            '''
            student_rna_valid_adata = student_rna_valid_loader.dataset.adatas[0]
            student_atac_valid_adata = student_atac_valid_loader.dataset.adatas[0]

            if args.target_dataset == 'MDD':
                student_rna_valid_adata = student_rna_valid_adata[::2].copy()
                student_atac_valid_adata = student_atac_valid_adata[::2].copy()

            rna_latents, atac_latents = get_latents(student_model, student_rna_valid_adata, student_atac_valid_adata, return_tensor=False)

            rna_celltypes = student_rna_valid_adata.obs['cell_type'].values
            atac_celltypes = student_atac_valid_adata.obs['cell_type'].values
            color_map_ct = create_celltype_palette(rna_celltypes.categories, atac_celltypes.categories, plot_color_palette=False)

            rna_condition = ['nan'] * len(rna_celltypes)
            atac_condition = ['nan'] * len(atac_celltypes)
            '''

            ## save umap embeddings
            umap_embedding, umap_figure, _ = plot_umap_embeddings(rna_latents, atac_latents, student_rna_labels, student_atac_labels, rna_condition, atac_condition, color_map_ct, umap_embedding=None)
            umap_figure.suptitle(f'UMAP embeddings of ECLARE model based on {args.valid_subsample} cells')
            umap_figure.savefig(os.path.join(args.outdir, 'umap_embeddings.png'))
            mlflow.log_figure(umap_figure, 'umap_embeddings.png')

    ## print output directory
    print('\n', args.outdir)
    

# %%