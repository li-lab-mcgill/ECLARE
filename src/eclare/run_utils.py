from argparse import Namespace
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from optuna import TrialPruned, Trial
import mlflow
from scipy.spatial.distance import squareform

from eclare.models import CLIP, SpatialCLIP
from eclare.losses_and_distances_utils import spatial_clip_loss, rbf_from_distance
from eclare.eval_utils import align_metrics, align_metrics_light, compute_mdd_eval_metrics
from eclare.data_utils import fetch_data_from_loaders, fetch_data_from_loader_light
from eclare.eval_utils import foscttm_moscot

def run_CLIP(trial,
    args: Namespace,
    genes_to_peaks_binary_mask,
    rna_train_loader,
    atac_train_loader,
    rna_valid_loader,
    atac_valid_loader,
    atac_train_num_batches,
    atac_train_n_batches_str_length,
    atac_train_n_epochs_str_length,
    atac_valid_num_batches,
    atac_valid_n_batches_str_length,
    atac_valid_n_epochs_str_length,
    target_atac_loader,
    target_rna_loader,
    tuned_hyperparameters: dict = {},
    do_pretrain_train: bool = True,
    do_pretrain_valid: bool = True,
    do_pretrain_train_eval: bool = False,
    do_align_train: bool = True,
    do_align_valid: bool = True,
    do_align_train_eval: bool = False,
    outdir: str = None,
    ):


    ## Model setup
    if (not args.tune_hyperparameters):
        print('parameters already defined')
        trial = None
        
        default_hyperparameters = {'params_pretrain': True if np.isin(args.source_dataset, ['388_human_brains', '388_human_brains_one_subject']).item() else False,
                                'params_margin': 1.,
                                'params_mmd_scale': 1.,
                                'params_bandwidth_parameter': 1.,
                                'temperature': 1.}
        
        pretrain    = tuned_hyperparameters.get('params_pretrain', default_hyperparameters['params_pretrain'])  # default should already be set before calling run_CLIP
        margin      = tuned_hyperparameters.get('params_margin', default_hyperparameters['params_margin'])
        mmd_scale   = tuned_hyperparameters.get('params_mmd_scale', default_hyperparameters['params_mmd_scale'])
        bandwidth_parameter = tuned_hyperparameters.get('params_bandwidth_parameter', default_hyperparameters['params_bandwidth_parameter'])
        temperature = tuned_hyperparameters.get('temperature', default_hyperparameters['temperature'])

    elif args.tune_hyperparameters: # tune hyperparameters
        pretrain    = True #trial.suggest_categorical("pretrain", [True, False])
        margin      = 0 #trial.suggest_float("margin", 0., 2.)
        mmd_scale   = 0 #trial.suggest_float("mmd_scale", 0.1, 10000., log=True)
        bandwidth_parameter = 0 #trial.suggest_float("bandwidth_parameter", 0.01, 10000., log=True)
        temperature = 0#trial.suggest_float("temperature", 0.1, 100., log=True)


    #if pretrain:
    pretrain_loss_fn_str = 'poisson' #trial.suggest_categorical("pretrain_loss_fn", ['poisson', 'poisson_log', 'mse', 'bce']) # poisson seems to be only that produces sensible validation curves (despite mse giving rise to good incorrect fraction scores...)
    pretrain_loss_fn_reduction = 'mean' #trial.suggest_categorical("pretrain_loss_fn_reduction", ['sum', 'mean']) # no evidence that sum reduction helps

    if pretrain_loss_fn_str == 'poisson':
        rna_pretrain_loss_fn = torch.nn.PoissonNLLLoss(log_input = False, reduction=pretrain_loss_fn_reduction)
        atac_pretrain_loss_fn = torch.nn.PoissonNLLLoss(log_input = False, reduction=pretrain_loss_fn_reduction)
    elif pretrain_loss_fn_str == 'poisson_log':
        pretrain_loss_fn = torch.nn.PoissonNLLLoss(log_input = True, reduction=pretrain_loss_fn_reduction)
    elif pretrain_loss_fn_str == 'mse':
        pretrain_loss_fn = torch.nn.MSELoss(reduction=pretrain_loss_fn_reduction)
    elif pretrain_loss_fn_str == 'bce':
        pretrain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction=pretrain_loss_fn_reduction)

    rna_recon_loss_fn = rna_pretrain_loss_fn
    atac_recon_loss_fn = atac_pretrain_loss_fn

    ## Initialize model
    if hasattr(atac_valid_loader.dataset, 'dataset') and hasattr(rna_valid_loader.dataset, 'dataset'): # True if use "standard" loaders
        n_peaks = atac_valid_loader.dataset.dataset.X.shape[1]
        n_genes = rna_valid_loader.dataset.dataset.X.shape[1]

        rna_valid_idx = list(rna_valid_loader.dataset.indices)
        atac_valid_idx = list(atac_valid_loader.dataset.indices)
        
    else:   # AnnLoaders
        n_peaks = atac_train_loader.dataset.shape[1]
        n_genes = rna_train_loader.dataset.shape[1]

        rna_valid_idx = rna_valid_loader.dataset.obs['level_0'].tolist()
        atac_valid_idx = atac_valid_loader.dataset.obs['level_0'].tolist()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args_dict = {
        'n_peaks': n_peaks,
        'n_genes': n_genes,
        'args': args,
        'device': device,
        'nam_type': 'few-to-one',
        'genes_to_peaks_binary_mask': genes_to_peaks_binary_mask,
        'pretrain': pretrain,
        'tuned_hyperparameters': tuned_hyperparameters,
        'rna_valid_idx': rna_valid_idx,
        'atac_valid_idx': atac_valid_idx,
    }

    batch_classifier = None

    model = CLIP(**model_args_dict, trial=trial).to(device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## Initialize clip loss
    align_loss_fn = clip_loss
    
    align_loss_fn.bandwidth_parameter = bandwidth_parameter
    align_loss_fn.temperature = temperature

    batch_classification_loss_fn = nn.CrossEntropyLoss() if args.source_dataset == '388_human_brains' else None
    
    if args.tune_hyperparameters:
        trial.set_user_attr('num_params', num_params)


    optimizer_align                = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_batch_loss_minimizer = None #torch.optim.AdamW(batch_classifier.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_batch_loss_maximizer = None #torch.optim.AdamW(model.rna_to_core.parameters(), lr=0.001, weight_decay=0.01)
    scheduler_align           = None


    ## Train model

    for epoch in (epochs_pbar := tqdm(range(args.total_epochs))):
        epochs_pbar.set_description('EPOCHS')

        if (((epoch+1) % args.valid_freq) == 0) or (epoch == 0):
            with torch.no_grad():

                model.eval()

                ## Valid align
                if do_align_valid:
                    align_valid_loss, triplet_losses_atac, triplet_losses_rna,\
                        align_metrics_dict, target_align_metrics_dict, clip_loss_, clip_loss_censored,\
                        acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split = \
                        align_and_reconstruction_pass(rna_valid_loader, atac_valid_loader, \
                                                        target_rna_loader, target_atac_loader, \
                                                        model, batch_classifier, device, None, None, None, scheduler_align,\
                                                        align_loss_fn, rna_recon_loss_fn, atac_recon_loss_fn, batch_classification_loss_fn,\
                                                        atac_valid_num_batches, args.triplet_type, args.tune_hyperparameters, save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_valid_n_batches_str_length, n_epochs_str_length=atac_valid_n_epochs_str_length)
                    
                    if not args.tune_hyperparameters:
                        ## initialize cell-type level metrics
                        if epoch == 0:
                            valid_metrics_df = pd.DataFrame.from_dict(align_metrics_dict, orient='index').T
                            valid_target_metrics_df = pd.DataFrame.from_dict(target_align_metrics_dict, orient='index').T

                            valid_acc_ct = acc_ct
                            valid_acc_top5_ct = acc_top5_ct

                            valid_clip_loss_ct = clip_loss_ct
                            valid_clip_loss_ct_split = clip_loss_ct_split

                        ## update cell-type level metrics
                        elif epoch > 0:
                            row_df = pd.DataFrame.from_dict(align_metrics_dict, orient='index').T
                            row_df.index = [epoch]
                            valid_metrics_df = pd.concat([valid_metrics_df, row_df])

                            row_df = pd.DataFrame.from_dict(target_align_metrics_dict, orient='index').T
                            row_df.index = [epoch]
                            valid_target_metrics_df = pd.concat([valid_target_metrics_df, row_df])

                            valid_acc_ct = valid_acc_ct.merge(acc_ct.rename(columns={0:epoch}), how='left', left_index=True, right_index=True)
                            valid_acc_top5_ct = valid_acc_top5_ct.merge(acc_top5_ct.rename(columns={0:epoch}), how='left', left_index=True, right_index=True)

                            valid_clip_loss_ct = valid_clip_loss_ct.merge(clip_loss_ct.rename(columns={0:epoch}), how='left', left_index=True, right_index=True)
                            valid_clip_loss_ct_split = valid_clip_loss_ct_split.merge(clip_loss_ct_split.rename(columns={0:epoch}), how='left', left_index=True, right_index=True)

                    elif args.tune_hyperparameters:
                        ## add per-epoch loss to trial logger
                        trial.report(align_valid_loss, step=epoch)
                        epoch_str = str(epoch).zfill(atac_valid_n_epochs_str_length)

                        trial.set_user_attr(f"epoch_{epoch_str}_align_loss", align_valid_loss)
                        trial.set_user_attr(f"epoch_{epoch_str}_triplet_loss_atac", triplet_losses_atac)
                        trial.set_user_attr(f"epoch_{epoch_str}_triplet_loss_rna", triplet_losses_rna)
                        trial.set_user_attr(f"epoch_{epoch_str}_mmd_loss", mmd_losses)
                        trial.set_user_attr(f"epoch_{epoch_str}_foscttm", align_metrics_dict["foscttm"])

                        ## early stopping -- default pruner is Median pruner
                        if trial.should_prune():
                            raise TrialPruned()


                ## Train align (eval)
                if do_align_train_eval:
                    _ = align_and_reconstruction_pass(rna_train_loader, atac_train_loader,\
                                                    target_rna_loader, target_atac_loader,\
                                                    model, batch_classifier, device, None, None, None, None,\
                                                    align_loss_fn, rna_recon_loss_fn, atac_recon_loss_fn, batch_classification_loss_fn,\
                                                    atac_train_num_batches, args.triplet_type, args.tune_hyperparameters, eval=True, save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_train_n_batches_str_length, n_epochs_str_length=atac_train_n_epochs_str_length)

                model.train()

        ## Train align
        if do_align_train:
            _ = align_and_reconstruction_pass(rna_train_loader, atac_train_loader,\
                                            target_rna_loader, target_atac_loader,\
                                            model, batch_classifier, device, optimizer_align, optimizer_batch_loss_minimizer, optimizer_batch_loss_maximizer, scheduler_align,\
                                            align_loss_fn, rna_recon_loss_fn, atac_recon_loss_fn, batch_classification_loss_fn,\
                                            atac_train_num_batches, args.triplet_type, args.tune_hyperparameters, save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_train_n_batches_str_length, n_epochs_str_length=atac_train_n_epochs_str_length)
            

    if not args.tune_hyperparameters and (do_align_train or do_align_valid):

        model.eval()
        model_args_dict['model_state_dict'] = model.state_dict()
        torch.save(model_args_dict, os.path.join(outdir,'model.pt'))

        ## Save metrics on valid data
        valid_metrics_df.to_csv(os.path.join(outdir, 'metrics.csv'))
        valid_target_metrics_df.to_csv(os.path.join(outdir, 'target_metrics.csv'))

        valid_acc_ct.T.to_csv(os.path.join(outdir, 'valid_acc_ct.csv'))
        valid_acc_top5_ct.T.to_csv(os.path.join(outdir, 'valid_acc_top5_ct.csv'))

        valid_clip_loss_ct.T.to_csv(os.path.join(outdir, 'valid_clip_loss_ct.csv'))
        valid_clip_loss_ct_split.T.to_csv(os.path.join(outdir, 'valid_clip_loss_ct_split.csv'))

        ## Save last epoch for restricted set of metrics to align with other methods, currently missing ASW for all methods
        valid_metrics_df.iloc[-1].loc[['ilisis','asw_ct','nmi','ari','foscttm_score']].to_csv(os.path.join(outdir, 'clip_metrics_source_valid.csv'))
        valid_target_metrics_df.iloc[-1].loc[['ilisis','asw_ct','nmi','ari','foscttm_score']].to_csv(os.path.join(outdir, 'clip_metrics_target_valid.csv'))

        return model

    elif args.tune_hyperparameters:

        trial_df = pd.DataFrame.from_dict(trial.user_attrs, orient='index')
        min_foscttm = trial_df[trial_df.index.str.contains('foscttm')].min().item()

        if do_align_valid and (not args.tune_hyperparameters):
            optuna_obj = np.subtract(1, valid_metrics_df['all_align_valid_clisis']).min()

        elif do_align_valid and args.tune_hyperparameters:
            optuna_obj = min_foscttm

        elif do_pretrain_valid and (not do_align_valid):
            optuna_obj = np.min(all_pretrain_valid_loss_df['valid_rna'] + all_pretrain_valid_loss_df['valid_atac'])

        return optuna_obj

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

    ## log initial valid loss
    model.eval()
    with torch.inference_mode():
        valid_loss = spatial_pass(rna_valid_loader, model, device, None)
    mlflow.log_metrics({'valid_loss': valid_loss}, step=0)

    ## train model
    for epoch in (epochs_pbar := tqdm(range(args.total_epochs))):
        epochs_pbar.set_description('EPOCHS')

        ## optimize
        model.train()
        train_loss = spatial_pass(rna_train_loader, model, device, optimizer)

        ## evaluate
        model.eval()
        with torch.inference_mode():
            valid_loss = spatial_pass(rna_valid_loader, model, device, None)

        ## get clustering performance
        rna_cells, rna_labels = fetch_data_from_loader_light(rna_valid_loader, label_key='spatialLIBD')
        rna_latents = model(rna_cells.to(device=device))
        nmi, ari = align_metrics_light(rna_latents, rna_labels)
        nmi_ari_score = 0.5 * (nmi + ari)

        mlflow.log_metrics({'valid_loss': valid_loss, 'train_loss': train_loss, 'nmi': nmi, 'ari': ari}, step=epoch+1)

        ## early stopping with Optuna pruner
        if trial is not None:
            trial.report(nmi_ari_score, step=epoch+1)
            if trial.should_prune():
                raise TrialPruned()

    return model, nmi_ari_score

def spatial_pass(rna_loader, model, device, optimizer):

    epoch_loss = []
    
    for rna_dat in (align_itr_pbar := tqdm(rna_loader)):

        ## project RNA data
        rna_cells = rna_dat.X.float() # already float32
        #rna_celltypes = rna_dat.obs['cell_type'].tolist()

        rna_cells.requires_grad_()
        rna_latents = model(rna_cells)

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

        ## get loss
        loss_rows, loss_cols = spatial_clip_loss(logits, As)
        loss = 0.5 * (loss_rows + loss_cols)

        ## update epoch loss
        epoch_loss.append(loss.item())

        ## backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## update progress bar
        align_itr_pbar.set_description(f"{'VALID' if optimizer is None else 'TRAIN'} itr -- SPATIAL (loss: {loss.item():.4f})")

    return np.mean(epoch_loss)

def align_and_reconstruction_pass(rna_loader,
                                atac_loader,
                                target_rna_loader,
                                target_atac_loader,
                                model, batch_classifier,
                                device, optimizer, optimizer_batch_loss_minimizer, optimizer_batch_loss_maximizer, scheduler,\
                                loss_fn, rna_recon_loss_fn, atac_recon_loss_fn, batch_classification_loss_fn,\
                                num_batches, triplet_type, tune_hyperparameters, eval=False, save_latents=False, **kwargs):

    align_losses, align_losses_atac, align_losses_rna = 0, 0, 0
    recon_losses, recon_losses_atac, recon_losses_rna = 0, 0, 0

    if save_latents:
        all_rna_latents, all_atac_latents = [], []
        all_rna_celltypes, all_atac_celltypes = [], []
        all_rna_batch_labels, all_atac_batch_labels = [], []
        
    for rna_dat, atac_dat in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):

        if (optimizer is not None):
            align_itr_pbar.set_description('TRAIN itr -- ALIGN & RECON')
        elif eval:
            align_itr_pbar.set_description('TRAIN EVAL itr -- ALIGN & RECON')
        else:
            align_itr_pbar.set_description('VALID itr -- ALIGN & RECON')

        ## check if cells loaded in matching pairs
        #assert (rna_dat.obs_names == atac_dat.obs_names).all()
        
        ## project RNA data
        rna_cells = rna_dat.X.float() # already float32
        rna_celltypes = rna_dat.obs['cell_type'].tolist()
        rna_batch_labels = np.zeros(len(rna_dat)) # rna_dat.obs['subject'].tolist()

        rna_cells.requires_grad_()
        rna_cells_recon, rna_latents = model(rna_cells, modality='rna', task='pretrain')

        ## project ATAC data
        atac_cells = atac_dat.X.float()
        atac_celltypes = atac_dat.obs['cell_type'].tolist()
        atac_batch_labels = np.zeros(len(atac_dat)) # atac_dat.obs['subject'].tolist()

        atac_cells.requires_grad_()
        atac_cells_recon, atac_latents = model(atac_cells, modality='atac', task='pretrain')

        if save_latents:
            all_rna_latents.append(rna_latents.detach().cpu().numpy())
            all_atac_latents.append(atac_latents.detach().cpu().numpy())
            all_rna_celltypes.append(rna_celltypes)
            all_atac_celltypes.append(atac_celltypes)
            all_rna_batch_labels.append(rna_batch_labels)
            all_atac_batch_labels.append(atac_batch_labels)

        ## Align losses
        temperature = loss_fn.temperature
        align_loss_atac, align_loss_rna = loss_fn(None, atac_latents=atac_latents, rna_latents=rna_latents, atac_celltypes=atac_celltypes, rna_celltypes=rna_celltypes, temperature=temperature)

        align_loss = 0.5 * (align_loss_atac + align_loss_rna)

        ## Reconstruction losses
        recon_loss_rna = rna_recon_loss_fn(rna_cells_recon, rna_cells)
        recon_loss_atac = atac_recon_loss_fn(atac_cells_recon, atac_cells)
        recon_loss = 0.5 * (recon_loss_rna + recon_loss_atac)
        recon_weight = 0.0
            
        ## Total loss. Sum losses, but more memory efficient to call backward on each loss separately (not implemented here yet)
        loss = align_loss + (recon_weight * recon_loss)

        if (optimizer is not None) and (not eval):
            ## CLIP loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Save losses
        align_losses_atac += (align_loss_atac/num_batches).detach().cpu().numpy()
        align_losses_rna += (align_loss_rna/num_batches).detach().cpu().numpy()
        align_losses += (align_loss/num_batches).detach().cpu().numpy()

        recon_losses_atac += (recon_loss_atac/num_batches).detach().cpu().numpy()
        recon_losses_rna += (recon_loss_rna/num_batches).detach().cpu().numpy()
        recon_losses += (recon_loss/num_batches).detach().cpu().numpy()

        ## Save latents
        n = 0  # set to -1 to inactivate batch-level save latents
        save_latents_ckpts_batches = (np.linspace(0,1,n+1) * num_batches).astype(int)


    if save_latents:

        n = 4  # set to -1 to inactivate epoch-level save latents
        save_latents_ckpts_epochs = (np.linspace(0,1,n+1) * kwargs['total_epochs']).astype(int)
        save_latents_ckpts_epochs[-1] = save_latents_ckpts_epochs[-1] - 1 # ensure that last checkpoint is the last epoch

        if np.isin(kwargs['epoch'], save_latents_ckpts_epochs).item():

            epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])

            if optimizer is not None:
                filename = f'latents_train_epoch_{epoch}.npz'
            elif eval:
                filename = f'latents_train_eval_epoch_{epoch}.npz'
            else:
                filename = f'latents_valid_epoch_{epoch}.npz'
                
                ## Target latents when valid
                with torch.no_grad():
                    rna_cells, rna_celltypes, atac_cells, atac_celltypes, _, _ = fetch_data_from_loaders(target_rna_loader, target_atac_loader, paired=False, subsample=2000)

                    rna_latents, _ = model(rna_cells, modality='rna', task='align')
                    atac_latents, _ = model(atac_cells, modality='atac', task='align')

                    filepath = os.path.join(os.path.join(kwargs['outdir'], f'target_{filename}'))
                    np.savez_compressed(filepath, rna=rna_latents.detach().cpu().numpy(), atac=atac_latents.detach().cpu().numpy(), rna_celltypes=rna_celltypes, atac_celltypes=atac_celltypes)

                
            all_rna_latents = np.vstack(all_rna_latents); all_atac_latents = np.vstack(all_atac_latents)
            all_rna_celltypes = np.hstack(all_rna_celltypes); all_atac_celltypes = np.hstack(all_atac_celltypes)
            all_rna_batch_labels = np.hstack(all_rna_batch_labels); all_atac_batch_labels = np.hstack(all_atac_batch_labels)

            filepath = os.path.join(os.path.join(kwargs['outdir'], filename))
            np.savez_compressed(filepath, rna=all_rna_latents, atac=all_atac_latents, rna_celltypes=all_rna_celltypes, atac_celltypes=all_atac_celltypes, rna_batches=all_rna_batch_labels, atac_batches=all_atac_batch_labels)


    ## update scheduler based on validation loss after each epoch to change learning rate
    if (scheduler is not None) and (not eval) and (optimizer is None):
        scheduler.step(align_losses.mean().item())

    ## compute metrics, if valid or eval data
    if (optimizer is None) and (not tune_hyperparameters):

        with torch.no_grad():
            rna_cells, rna_celltypes, atac_cells, atac_celltypes, _, _ = fetch_data_from_loaders(rna_loader, atac_loader)
            ilisis, asw_ct, nmi, ari, diag_concentration_minimizer, foscttm_score, _, acc, acc_top5, clip_loss_, clip_loss_censored, \
                foscttm_score_ct, acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split = align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes) # should be valid loaders
            
            align_metrics_dict = {'align_loss_rna': align_losses_rna, 'align_loss_atac': align_losses_atac, \
                                'recon_loss_rna': recon_losses_rna, 'recon_loss_atac': recon_losses_atac, \
                                'clip_loss': clip_loss_, 'clip_loss_censored': clip_loss_censored, \
                                'ilisis':ilisis, 'asw_ct':asw_ct, 'nmi':nmi, 'ari':ari, 'diag_concentration_minimizer':diag_concentration_minimizer, 'foscttm_score':foscttm_score, 'acc':acc, 'acc_top5':acc_top5}

            ## evaluate on target data (e.g. MDD)
            if os.environ['target_dataset'] == 'mdd':
                ilisis, asw_ct, nmi, ari, diag_concentration_minimizer, foscttm_score, acc, acc_top5 = compute_mdd_eval_metrics(model, target_rna_loader, target_atac_loader, device, mdd_eval_method='subsample')
            else:
                rna_cells, rna_celltypes, atac_cells, atac_celltypes, _, _ = fetch_data_from_loaders(target_rna_loader, target_atac_loader)
                ilisis, asw_ct, nmi, ari, diag_concentration_minimizer, foscttm_score, _, acc, acc_top5, clip_loss_, clip_loss_censored, \
                    foscttm_score_ct, acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split = align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes)

            target_align_metrics_dict = {'ilisis':ilisis, 'asw_ct':asw_ct, 'nmi':nmi, 'ari':ari, 'diag_concentration_minimizer':diag_concentration_minimizer, 'foscttm_score':foscttm_score, 'acc':acc, 'acc_top5':acc_top5}

    elif (optimizer is None) and tune_hyperparameters:
        foscttm_score = foscttm_moscot(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy())
        align_metrics_dict = {'foscttm_score':foscttm_score.item()}
        target_align_metrics_dict = acc_ct = acc_top5_ct = clip_loss_ct = clip_loss_ct_split = clip_loss_ = clip_loss_censored = None

    else:
        align_metrics_dict = target_align_metrics_dict = acc_ct = acc_top5_ct = clip_loss_ct = clip_loss_ct_split = clip_loss_ = clip_loss_censored = None

    return align_losses, align_losses_atac, align_losses_rna,\
        align_metrics_dict, target_align_metrics_dict, clip_loss_, clip_loss_censored,\
        acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split

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
      return experiment.experiment_id
  else:
      return mlflow.create_experiment(experiment_name)