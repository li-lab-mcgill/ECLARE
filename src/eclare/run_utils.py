from argparse import Namespace
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from optuna import TrialPruned, Trial
import mlflow
import subprocess
from scipy.spatial.distance import squareform

from eclare.models import CLIP, SpatialCLIP, get_clip_hparams, get_spatial_clip_hparams
from eclare.losses_and_distances_utils import clip_loss, spatial_clip_loss, rbf_from_distance, Knowledge_distillation_fn
from eclare.eval_utils import get_metrics
from eclare.data_utils import fetch_data_from_loader_light

def run_CLIP(
    args: Namespace,
    rna_train_loader,
    rna_valid_loader,
    atac_train_loader,
    atac_valid_loader,
    target_rna_valid_loader,
    target_atac_valid_loader,
    trial: Trial = None,
    params: dict = {},
    ):

    ## setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paired_target = (args.target_dataset != 'MDD')
    n_genes = rna_train_loader.dataset.shape[1]
    n_peaks = atac_train_loader.dataset.shape[1]

    model = CLIP(n_peaks=n_peaks, n_genes=n_genes, **params).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## log initial valid loss
    with torch.inference_mode():
        model.eval()
        valid_losses = clip_pass(rna_valid_loader, atac_valid_loader, model, None)
        
        ## source performance metrics
        metrics = {}
        source_metrics = get_metrics(model, rna_valid_loader, atac_valid_loader, device)
        metrics.update({f'source_{k}': v for k, v in source_metrics.items() if ~np.isnan(v)})
        metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})

        ## target performance metrics
        target_metrics = get_metrics(model, target_rna_valid_loader, target_atac_valid_loader, device, paired = paired_target)
        metrics.update(target_metrics)

        ## log metrics
        mlflow.log_metrics(metrics, step=0)
        

    ## Train model
    for epoch in (epochs_pbar := tqdm(range(args.total_epochs))):
        epochs_pbar.set_description('EPOCHS')

        ## optimize
        model.train()
        train_losses = clip_pass(rna_train_loader, atac_train_loader, model, optimizer)

        ## evaluate
        with torch.inference_mode():
            model.eval()
            valid_losses = clip_pass(rna_valid_loader, atac_valid_loader, model, None)


        ## source performance metrics
        metrics = {}
        source_metrics = get_metrics(model, rna_valid_loader, atac_valid_loader, device)
        metrics.update({f'source_{k}': v for k, v in source_metrics.items() if ~np.isnan(v)})
        metrics.update({f'train_{k}': v for k, v in train_losses.items() if ~np.isnan(v)})
        metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})

        ## target performance metrics
        target_metrics = get_metrics(model, target_rna_valid_loader, target_atac_valid_loader, device, paired = paired_target)
        metrics.update(target_metrics)
        
        ## log metrics
        mlflow.log_metrics(metrics, step=epoch+1)

        ## get metric to optimize
        metric_to_optimize = metrics.get(args.metric_to_optimize, None)

        ## early stopping with Optuna pruner
        if trial is not None:
            trial.report(metric_to_optimize, step=epoch+1)
            if trial.should_prune():
                raise TrialPruned()

    return model, metric_to_optimize


def run_spatial_CLIP(
    args: Namespace,
    rna_train_loader,
    rna_valid_loader,
    trial: Trial = None,
    params: dict = {},
    ):

    ## setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_genes = rna_train_loader.dataset.shape[1]

    ## initialize model
    model = SpatialCLIP(n_genes=n_genes, **params).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    ## add decoder loss function to loaders - can inspect with 'import inspect; print(inspect.getsource(trial.params['decoder_loss']))'
    rna_train_loader.decoder_loss = params['decoder_loss']
    rna_valid_loader.decoder_loss = params['decoder_loss']

    ## log initial valid loss
    with torch.inference_mode():
        model.eval()
        valid_losses = spatial_pass(rna_valid_loader, model, device, None)
        mlflow.log_metrics({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)}, step=0)

    ## train model
    for epoch in (epochs_pbar := tqdm(range(args.total_epochs))):
        epochs_pbar.set_description('EPOCHS')

        ## optimize
        model.train()
        train_losses = spatial_pass(rna_train_loader, model, device, optimizer)

        ## evaluate
        with torch.inference_mode():
            model.eval()
            valid_losses = spatial_pass(rna_valid_loader, model, device, None)

        ## get clustering performance
        rna_cells, rna_labels = fetch_data_from_loader_light(rna_valid_loader, label_key='spatialLIBD')
        rna_latents, _ = model(rna_cells.to(device=device))
        nmi, ari = align_metrics_light(rna_latents, rna_labels)
        nmi_ari_score = 0.5 * (nmi + ari)

        # Combine all metrics into a single dictionary
        metrics = {'nmi': nmi, 'ari': ari}
        metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})
        metrics.update({f'train_{k}': v for k, v in train_losses.items() if ~np.isnan(v)})
        
        # Log all metrics at once
        mlflow.log_metrics(metrics, step=epoch+1)

        ## early stopping with Optuna pruner
        if trial is not None:
            trial.report(nmi_ari_score, step=epoch+1)
            if trial.should_prune():
                raise TrialPruned()

    return model, nmi_ari_score


def spatial_pass(rna_loader, model, device, optimizer):

    epoch_losses = {'sp_loss': [], 'recn_loss': [], 'tot_loss': []}
    
    for rna_dat in (align_itr_pbar := tqdm(rna_loader)):

        ## project RNA data
        rna_cells = rna_dat.X.float() # already float32
        #rna_celltypes = rna_dat.obs['cell_type'].tolist()

        rna_cells.requires_grad_()
        rna_latents, rna_recon = model(rna_cells)

        ## get spatial coordinates of spots
        array_row = rna_dat.obs['array_row']
        array_col = rna_dat.obs['array_col']
        array_coords = torch.stack((array_row, array_col), dim=1).float()

        ## get spatial adjacency matrix
        Ds_flat = nn.functional.pdist(array_coords, p=2)
        gamma = 1 / (2 * torch.median(Ds_flat)**2)  # better to derive median from full distance matrix
        As_flat = rbf_from_distance(Ds_flat, gamma=gamma)
        As = squareform(As_flat.cpu().numpy())
        As = torch.from_numpy(As).to(device=device).detach()  # zero along diagonal

        ## get logits and apply temperature
        rna_latents = torch.nn.functional.normalize(rna_latents, dim=1)
        logits = torch.matmul(rna_latents, rna_latents.T)
        logits = logits * np.exp(1/model.temperature)

        ## handle diagonal entries
        As[torch.eye(As.shape[0], dtype=torch.bool)] = -torch.inf
        logits[torch.eye(logits.shape[0], dtype=torch.bool)] = -torch.inf

        ## get spatial loss
        spatial_loss_rows, spatial_loss_cols = spatial_clip_loss(logits, As)
        loss = 0.5 * (spatial_loss_rows + spatial_loss_cols)
        epoch_losses['sp_loss'].append(loss.item())

        ## get reconstruction loss
        if rna_loader.decoder_loss is not None:
            reconstruction_loss = rna_loader.decoder_loss(rna_recon, rna_cells)
            loss = loss + (0.01 * reconstruction_loss) # make sure that roughly on same scale as spatial loss

            ## update other epoch losses
            epoch_losses['recn_loss'].append(reconstruction_loss.item())
            epoch_losses['tot_loss'].append(loss.item())

        ## backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## update progress bar
        align_itr_pbar.set_description(f"{'VALID' if optimizer is None else 'TRAIN'} itr -- SPATIAL (loss: {loss.item():.4f})")

    ## replace epoch losses with mean
    epoch_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    return epoch_losses

def clip_pass(rna_loader, atac_loader, model, optimizer):

    epoch_losses = {'clip_loss_rna': [], 'clip_loss_atac': [], 'recon_loss_rna': [], 'recon_loss_atac': [], 'tot_loss': []}
        
    for rna_dat, atac_dat in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):

        ## check if cells loaded in matching pairs
        #assert (rna_dat.obs_names == atac_dat.obs_names).all()
        
        ## get cells/nuclei and their cell types
        rna_cells = rna_dat.X.float() # already float32
        atac_cells = atac_dat.X.float()

        rna_cells.requires_grad_()
        atac_cells.requires_grad_()

        rna_celltypes = rna_dat.obs['cell_type'].tolist()
        atac_celltypes = atac_dat.obs['cell_type'].tolist()

        ## project cells/nuclei
        rna_latents, rna_recon = model(rna_cells, modality=0)
        atac_latents, atac_recon = model(atac_cells, modality=1)

        ## Align losses
        clip_loss_atac, clip_loss_rna = clip_loss(None,
                                                atac_latents=atac_latents,
                                                rna_latents=rna_latents,
                                                atac_celltypes=atac_celltypes,
                                                rna_celltypes=rna_celltypes,
                                                temperature=model.temperature)

        loss = 0.5 * (clip_loss_atac + clip_loss_rna)
        epoch_losses['clip_loss_rna'].append(clip_loss_rna.item())
        epoch_losses['clip_loss_atac'].append(clip_loss_atac.item())

        ## Reconstruction losses
        if model.decoder_loss:
            recon_loss_rna = model.decoder_loss(rna_recon, rna_cells)
            recon_loss_atac = model.decoder_loss(atac_recon, atac_cells)
            recon_loss = 0.5 * (recon_loss_rna + recon_loss_atac)
            loss = loss + (0.01 * recon_loss) ## Total loss. Sum losses, but more memory efficient to call backward on each loss separately (not implemented here yet)

            epoch_losses['recon_loss_rna'].append(recon_loss_rna.item())
            epoch_losses['recon_loss_atac'].append(recon_loss_atac.item())
            epoch_losses['tot_loss'].append(loss.item())

        if (optimizer is not None):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        align_itr_pbar.set_description(f"{'VALID' if optimizer is None else 'TRAIN'} itr -- CLIP (loss: {loss.item():.4f})")

    ## replace epoch losses with mean
    epoch_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    return epoch_losses


def save_latents(student_rna_latents_valid, student_atac_latents_valid, student_rna_celltypes_valid, student_atac_celltypes_valid, epoch, n_epochs, outdir):
    ## in reality, only a single batch of valid latents per epoch, so no need to accumulate

    n = 4  # set to -1 to inactivate epoch-level save latents
    save_latents_ckpts_epochs = (np.linspace(0,1,n+1) * n_epochs).astype(int)
    save_latents_ckpts_epochs[-1] = save_latents_ckpts_epochs[-1] - 1 # ensure that last checkpoint is the last epoch

    if np.isin(epoch, save_latents_ckpts_epochs).item():

        n_epochs_str_length = len(str(n_epochs - 1))
        epoch_str = str(epoch).zfill(n_epochs_str_length)

        filename = f'latents_valid_epoch_{epoch_str}.npz'
            
        all_rna_latents_valid = student_rna_latents_valid.detach().cpu().numpy(); all_atac_latents_valid = student_atac_latents_valid.detach().cpu().numpy()
        all_rna_celltypes_valid = student_rna_celltypes_valid; all_atac_celltypes_valid = student_atac_celltypes_valid

        filepath = os.path.join(outdir, filename)
        np.savez_compressed(filepath, rna=all_rna_latents_valid, atac=all_atac_latents_valid, rna_celltypes=all_rna_celltypes_valid, atac_celltypes=all_atac_celltypes_valid)

def get_or_create_experiment(experiment_name):
  """
  Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

  This function checks if an experiment with the given name exists within MLflow.
  If it does, the function returns its ID. If not, it creates a new experiment
  with the provided name and returns its ID.

  Parameters:
  - experiment_name (str): Name of the MLflow experiment.

  Returns:
  - str: ID of the existing or newly created MLflow experiment.
  """

  if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment
  else:
      experiment_id = mlflow.create_experiment(experiment_name)
      experiment = mlflow.get_experiment(experiment_id)
      return experiment

def fully_delete_experiment(experiment_name):

    mlflow.set_tracking_uri('http://localhost:5000') # raises error if set to path to local mlruns directory
    #mlflow.set_tracking_uri('file://' + os.path.abspath('mlruns'))
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else experiment_name

    print(f"Soft deleting experiment {experiment_id}...")
    client.delete_experiment(experiment_id)

    # Perform garbage collection immediately after
    print("Performing garbage collection...")
    subprocess.run(
        ["mlflow", "gc", "--backend-store-uri", mlflow.get_tracking_uri()],
        check=True
    )

    print(f"✅ Fully deleted experiment '{experiment_name}'.")


def eclare_pass(
    target_dataset_og,
    student_rna_iterator,
    student_atac_iterator,
    teacher_rna_iterators,
    teacher_atac_iterators,
    student_model,
    models,
    lambd,
    optimizer,
    knowledge_distillation_fn,
    loop_order
    ):

    # Extract datasets from iterators
    datasets = list(teacher_rna_iterators.keys())

    # Initialize iterators for each data loader
    teacher_rna_iterators = {dataset: iter(loader) for dataset, loader in teacher_rna_iterators.items()}
    teacher_atac_iterators = {dataset: iter(loader) for dataset, loader in teacher_atac_iterators.items()}

    student_rna_iterator = iter(student_rna_iterator)
    student_atac_iterator = iter(student_atac_iterator)

    # Determine the number of batches per dataset
    num_batches_per_dataset = {
        dataset: min(
            len(teacher_rna_iterators[dataset]), len(teacher_atac_iterators[dataset]), len(student_rna_iterator), len(student_atac_iterator)
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

    # Extract device from model
    device = next(student_model.parameters()).device

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

        ## Project student RNA data (teacher)
        student_rna_cells = student_rna_dat.X.float().to(device)
        student_rna_latents, _ = student_model(student_rna_cells, modality=0)

        ## Project student ATAC data (teacher)
        student_atac_cells = student_atac_dat.X.float().to(device)
        student_atac_latents, _ = student_model(student_atac_cells, modality=1)

        ## Initialize list of dataset distil losses
        distil_losses, distil_losses_T = [], []
        align_losses, align_losses_T = [], []
        offsets, offsets_T = [], []

        for inner in (tqdm(inner_iterable, leave=False) if loop_order == 'datasets_first' else inner_iterable):
            dataset = outer if loop_order == 'datasets_first' else inner

            # Retrieve the next batch from each iterator
            try:
                teacher_rna_dat = next(teacher_rna_iterators[dataset])
                teacher_atac_dat = next(teacher_atac_iterators[dataset])

            except StopIteration:
                # If any iterator runs out of data, continue to the next iteration
                continue

            # Load the model for that dataset
            model = models[dataset]

            # Project teacher RNA data
            teacher_rna_cells = teacher_rna_dat.X.float().to(device)
            teacher_rna_latents, _ = model(teacher_rna_cells, modality=0)

            # Project teacher ATAC data
            teacher_atac_cells = teacher_atac_dat.X.float().to(device)
            teacher_atac_latents, _ = model(teacher_atac_cells, modality=1)

            ## Ensure that the teacher latents are detached
            teacher_rna_latents = teacher_rna_latents.detach()
            teacher_atac_latents = teacher_atac_latents.detach()

            assert (student_rna_dat.obs_names == teacher_rna_dat.obs_names).all()
            assert (student_atac_dat.obs_names == teacher_atac_dat.obs_names).all()

            ## compute teacher losses
            distil_loss, distil_loss_T, align_loss_scaled, align_loss_T_scaled, offset, offset_T = \
                knowledge_distillation_fn(student_rna_latents, student_atac_latents, teacher_rna_latents, teacher_atac_latents, 'teacher')

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
            total_loss = (lambd * distil_loss) + ((1-lambd) * align_loss_scaled)

            # Accumulate scalar loss values for logging
            epoch_total_losses[dataset]     += total_loss.mean().item()
            epoch_distil_loss[dataset]      += distil_loss.mean().item()
            epoch_align_loss[dataset]       += align_loss_scaled.mean().item()


        ## Stack teacher distillation losses
        distil_losses = torch.stack(distil_losses)
        align_losses = torch.stack(align_losses)
        distil_losses_T = torch.stack(distil_losses_T)
        align_losses_T = torch.stack(align_losses_T)
        offsets = torch.stack(offsets)
        offsets_T = torch.stack(offsets_T)

        ## Compute distillation loss weighted by alignment loss, if more than one teacher
        mean_distil_loss, _ = \
            knowledge_distillation_fn.distil_loss_weighting( distil_losses, distil_losses_T, (offsets - align_losses), (offsets_T - align_losses_T))

        ## Get student align loss
        _, _, align_loss_scaled, _, _, _ = \
            knowledge_distillation_fn(student_rna_latents, student_atac_latents, teacher_rna_latents, teacher_atac_latents, 'student')

        ## Compute total loss as convex combination of CLIP loss and average distillation loss
        total_loss = (lambd * mean_distil_loss) + ((1-lambd) * align_loss_scaled)

        ## Save losses
        epoch_total_losses[target_dataset_og]   += total_loss.item()
        epoch_distil_loss[target_dataset_og]    += mean_distil_loss.item()
        epoch_align_loss[target_dataset_og]     += align_loss_scaled.item()

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
        metrics_dict[f'align_loss_{dataset}']   = epoch_align_loss[dataset] / num_batches # could ignore non-teacher datasets since constant value across epochs
            
    return metrics_dict


def run_ECLARE(
    args: Namespace,
    student_rna_train_loader,
    student_rna_valid_loader,
    student_atac_train_loader,
    student_atac_valid_loader,
    teacher_rna_train_loaders,
    teacher_atac_train_loaders,
    teacher_rna_valid_loaders,
    teacher_atac_valid_loaders,
    teacher_models,
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
    knowledge_distillation_fn = Knowledge_distillation_fn(device=device, student_temperature=1, teacher_temperature=1, paired=paired, weigh_distil_by_align_type='batch')

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

        #student_model.eval()

        valid_losses = eclare_pass(
            args.target_dataset,
            student_rna_valid_loader,
            student_atac_valid_loader,
            teacher_rna_valid_loaders,
            teacher_atac_valid_loaders,
            student_model,
            teacher_models,
            args.distil_lambda,
            None,
            knowledge_distillation_fn,
            loop_order
        )

        ## Update alignment loss scale
        target_distil_loss = valid_losses['distil_loss_'+args.target_dataset]
        target_align_loss = valid_losses['align_loss_'+args.target_dataset]
        align_loss_scale = (target_distil_loss / target_align_loss)
        knowledge_distillation_fn.align_loss_scale = align_loss_scale

        ## Retroactively scale teacher & student CLIP losses
        datasets = list(teacher_rna_valid_loaders.keys())
        for dataset in datasets+[args.target_dataset]:
            valid_losses['align_loss_'+dataset] = valid_losses['align_loss_'+dataset] * align_loss_scale
            valid_losses['total_loss_'+dataset] = (args.distil_lambda * valid_losses['distil_loss_'+dataset]) + ((1-args.distil_lambda) * valid_losses['align_loss_'+dataset])

        ## get metrics
        metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device, paired=paired)
        metrics.update({f'valid_{k}': v for k, v in valid_losses.items() if ~np.isnan(v)})

        # Log all metrics at once with MLflow
        mlflow.log_metrics(metrics, step=0)



    print('Iterating over epochs, batches & datasets')
    for epoch in range(args.total_epochs):

        # Start the loops -- training data
        student_model.train()

        train_losses = eclare_pass(
            args.target_dataset,
            student_rna_train_loader,
            student_atac_train_loader,
            teacher_rna_train_loaders,
            teacher_atac_train_loaders,
            student_model,
            teacher_models,
            args.distil_lambda,
            optimizer,
            knowledge_distillation_fn,
            loop_order
        )

        with torch.inference_mode():

            #student_model.eval()

            valid_losses = eclare_pass(
                args.target_dataset,
                student_rna_valid_loader,
                student_atac_valid_loader,
                teacher_rna_valid_loaders,
                teacher_atac_valid_loaders,
                student_model,
                teacher_models,
                args.distil_lambda,
                None,
                knowledge_distillation_fn,
                loop_order
            )

        # Add performance metrics from get_metrics
        metrics = get_metrics(student_model, student_rna_valid_loader, student_atac_valid_loader, device, paired=paired)
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
    
    