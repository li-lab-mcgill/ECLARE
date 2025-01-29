from argparse import Namespace
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from optuna import TrialPruned

def run_scTripletgrate(trial,
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

    print('Activate anomaly detection')
    torch.autograd.set_detect_anomaly(True)

    ## Model setup
    if (not args.tune_hyperparameters) and (args.slurm_id is not None):
        print('parameters already defined')
        trial = None
        
        #for key in tuned_hyperparameters:
        #    if 'params_' in key:
        #        new_variable_name = key.replace('params_', '')
        #        globals()[new_variable_name] = tuned_hyperparameters[key]  # discouraged

        #columns_align    = pd.MultiIndex.from_product([['align train','align valid','align eval'], ['incorrect fraction score','loss','ari']])
        #indices = pd.MultiIndex.from_product([np.arange(args.total_epochs), ['pulmonary epithelial', 'mast', 'b lymphocyte', 'pnc-derived', 'endothelial', 'mural', 'ductal', 'acinar', 'hepatocyte', 'ionic mesenchymal', 'keratinocyte', 'stromal smooth muscle', 'luminal epithelial', 'myeloid / macrophage', 'myoepithelial', 'gastrointestinal epithelial', 'mesenchymal']], names=['epoch','celltype'])
        #celltype_results_df = pd.DataFrame( 0, index = indices, columns = columns_align)

        default_hyperparameters = {'params_pretrain': True if np.isin(args.source_dataset, ['388_human_brains', '388_human_brains_one_subject']).item() else False,
                                'params_margin': 1.,
                                'params_mmd_scale': 1.,
                                'params_bandwidth_parameter': 1.,
                                'temperature': 1.}
        
        pretrain    = tuned_hyperparameters.get('params_pretrain', default_hyperparameters['params_pretrain'])  # default should already be set before calling run_scTripletgrate
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

    if args.source_dataset == '388_human_brains':
        print('creating batch classifier')

        num_units = 256
        dropout_p = 0.3

        if hasattr(atac_valid_loader.dataset, 'dataset'):
            num_batches = len(np.unique(rna_train_loader.dataset.dataset.batches))
        else:
            num_batches = rna_train_loader.dataset.obs['subject'].nunique()  # 'subject' keyword currently only good for 388_human_brains data

        model_args_dict['num_experimental_batches'] = num_batches

        batch_classifier = nn.Sequential(
            nn.Linear(num_units, 2*num_batches), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(2*num_batches, num_batches)
        )
    else:
        batch_classifier = None

    #model = scTripletgrate(n_peaks, n_genes, args, device,\
    #                                         nam_type='few-to-one', genes_to_peaks_binary_mask=genes_to_peaks_binary_mask, pretrain=pretrain, tuned_hyperparameters=tuned_hyperparameters).to(device=device)
    model = scTripletgrate(**model_args_dict, trial=trial).to(device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## Initialize triplet loss with margin hyperparameter
    if np.isin(args.triplet_type, ['mnn','cell-type']).item():
        align_loss_fn = (
            nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y, dim=-1),
            margin=margin, reduction='none')) # margin=1.0 as default
        
    elif np.isin(args.triplet_type, ['clip']).item():
        align_loss_fn = clip_loss
    
    align_loss_fn.bandwidth_parameter = bandwidth_parameter
    align_loss_fn.temperature = temperature

    batch_classification_loss_fn = nn.CrossEntropyLoss() if args.source_dataset == '388_human_brains' else None
    
    if args.tune_hyperparameters:
        trial.set_user_attr('num_params', num_params)

    ## create separate optimizers for align loss and batch correction
    #align_params = model.parameters() #[ param for name, param in model.named_parameters() if not name.startswith("batch_classifier") ] # isolate parameters not part of batch_classifier
    #batch_classifier_params = batch_classifier.parameters()

    optimizer_align                = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_batch_loss_minimizer = None #torch.optim.AdamW(batch_classifier.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_batch_loss_maximizer = None #torch.optim.AdamW(model.rna_to_core.parameters(), lr=0.001, weight_decay=0.01)
    scheduler_align           = None

    '''
    if pretrain:
        
        optimizer_pretrain_unimodal = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        optimizer_pretrain_bimodal  = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        all_pretrain_valid_loss_rna = []
        all_pretrain_train_loss_rna = []
        all_pretrain_valid_loss_atac = []
        all_pretrain_train_loss_atac = []
        all_pretrain_valid_loss = []
        all_pretrain_eval_loss = []

        for epoch in (epochs_pbar := tqdm(range(args.total_epochs))):
            epochs_pbar.set_description('EPOCHS')

            if (((epoch+1) % args.valid_freq) == 0) or (epoch == 0):
                with torch.no_grad():
                    model.eval()

                    ## Valid pretrain
                    if do_pretrain_valid:
                        pretrain_valid_loss_rna, pretrain_valid_loss_atac = pretrain_pass(rna_valid_loader, atac_valid_loader, model, None, None, rna_pretrain_loss_fn, atac_pretrain_loss_fn, atac_valid_num_batches,
                                                                            save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_valid_n_batches_str_length, n_epochs_str_length=atac_valid_n_epochs_str_length)
                        
                        all_pretrain_valid_loss_rna.append(pretrain_valid_loss_rna)
                        all_pretrain_valid_loss_atac.append(pretrain_valid_loss_atac)

                        if args.tune_hyperparameters:
                            ## add per-epoch loss to trial logger
                            #trial.report(pretrain_valid_loss, step=epoch)  #UserWarning: The reported value is ignored because this `step` 0 is already reported.
                            epoch_str = str(epoch).zfill(len(str(args.total_epochs)))
                            trial.set_user_attr(f"epoch_{epoch_str}_pretrain_loss_rna", pretrain_valid_loss_rna)
                            trial.set_user_attr(f"epoch_{epoch_str}_pretrain_loss_atac", pretrain_valid_loss_atac)

                            ## early stopping -- default pruner is Median pruner
                            if trial.should_prune():
                                raise TrialPruned()

                    ## Train pretrain (eval)
                    if do_pretrain_train_eval:
                        pretrain_eval_loss = pretrain_pass(rna_valid_loader, atac_valid_loader, model, None, None, rna_pretrain_loss_fn, atac_pretrain_loss_fn, atac_train_num_batches,
                                                           eval=True, save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_train_n_batches_str_length, n_epochs_str_length=atac_train_n_epochs_str_length)
                        all_pretrain_eval_loss.append(pretrain_eval_loss)

                    model.train()

            ## Train pretrain
            if do_pretrain_train:
                pretrain_train_loss_rna, pretrain_train_loss_atac = pretrain_pass(rna_train_loader, atac_train_loader, model, optimizer_pretrain_unimodal, optimizer_pretrain_bimodal, rna_pretrain_loss_fn, atac_pretrain_loss_fn, atac_train_num_batches,
                                                    save_latents=False if args.tune_hyperparameters else True, outdir=outdir, epoch=epoch, total_epochs=args.total_epochs, n_batches_str_length=atac_train_n_batches_str_length, n_epochs_str_length=atac_train_n_epochs_str_length)
                
                all_pretrain_train_loss_rna.append(pretrain_train_loss_rna)
                all_pretrain_train_loss_atac.append(pretrain_train_loss_atac)

        if not args.tune_hyperparameters:
            model.eval()
            model_args_dict['model_state_dict'] = model.state_dict()
            torch.save(model_args_dict, os.path.join(outdir,'model.pt'))

            ## Save all valid and train pretrain losses in one dataframe
            all_pretrain_valid_loss_df = pd.DataFrame({'valid_rna':all_pretrain_valid_loss_rna, 'valid_atac':all_pretrain_valid_loss_atac, 'train_rna':all_pretrain_train_loss_rna, 'train_atac':all_pretrain_train_loss_atac})
            all_pretrain_valid_loss_df.to_csv(os.path.join(outdir, 'pretrain_losses.csv'))

            if not (do_align_train or do_align_valid):
                return model
    '''

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


def mdd_loss_per_ct(atac_genes, rna_genes, atac_celltypes, rna_celltypes):

    rna_celltypes_dummies = pd.get_dummies(rna_celltypes).reindex(sorted(np.unique(rna_celltypes)), axis=1)
    atac_celltypes_dummies = pd.get_dummies(atac_celltypes).reindex(sorted(np.unique(atac_celltypes)), axis=1)
    common_celltypes = set(rna_celltypes).intersection(set(atac_celltypes))

    mmd_loss = 0
    for celltype in common_celltypes:

        rna_genes_ct = rna_genes[ rna_celltypes_dummies[celltype] ]
        atac_genes_ct = atac_genes[ atac_celltypes_dummies[celltype] ]

        ## zero-pad, if required
        padding_rows = np.abs(len(rna_genes_ct) - len(atac_genes_ct))
        if padding_rows != 0:
            if len(rna_genes_ct) < len(atac_genes_ct):
                rna_genes_ct = nn.functional.pad(rna_genes_ct, (0, 0, 0, padding_rows), mode='constant', value=0)
            elif len(atac_genes_ct) < len(rna_genes_ct):
                atac_genes_ct = nn.functional.pad(atac_genes_ct, (0, 0, 0, padding_rows), mode='constant', value=0)


        rna_genes_sorted = torch.sort(rna_genes_ct, dim=0).values
        atac_genes_sorted = torch.sort(atac_genes_ct, dim=0).values

        mmd_loss_ct = mmd_loss_fn(atac_genes_sorted, rna_genes_sorted)

        #gene_by_gene = torch.matmul(rna_genes_sorted.T, atac_genes_sorted)
        #gene_by_gene = gene_by_gene / torch.linalg.norm(gene_by_gene)
        #mmd_loss_ct = torch.log2(gene_by_gene.sum()) - torch.log2(gene_by_gene.trace())

        mmd_loss = mmd_loss + (mmd_loss_ct/len(common_celltypes))
        return mmd_loss

def get_batch_minmax(latents, batch_labels):
    batch_prediction_rna = batch_classifier(rna_latents.clone())
    batch_prediction_atac = batch_classifier(atac_latents.clone())
    batch_prediction = torch.mean(torch.stack([batch_prediction_rna, batch_prediction_atac]), dim=0)

    batch_loss_minimizer = batch_classification_loss(batch_prediction, batch_labels)
    batch_loss_maximizer = -1 * batch_loss_minimizer.clone()
    return batch_loss_minimizer, batch_loss_maximizer
        
def pretrain_pass_unimodal(loader, modality, model, optimizer, loss_fn, num_batches, eval=False, save_latents=False, **kwargs):

    if save_latents:
        all_rna_genes, all_atac_genes = [], []
        all_rna_celltypes, all_atac_celltypes = [], []

    pretrain_loss = 0
    for features, celltypes in (pretrain_itr_pbar := tqdm( loader , total=int(num_batches-1) )):  # kernel dumped

        if (optimizer is not None):
            pretrain_itr_pbar.set_description('TRAIN itr -- PRETRAIN')
        elif eval:
            pretrain_itr_pbar.set_description('TRAIN EVAL itr -- PRETRAIN')
        else:
            pretrain_itr_pbar.set_description('VALID itr -- PRETRAIN')

        features = features.to_dense().squeeze(1).to(device=device)

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            features = (features > 0).float()  # can binarize and use BCE loss

        features.requires_grad_()
        features_recon, latents = model(features, modality=modality, task='pretrain')  # output has requires_grad=False...why?

        if save_latents:
            if modality == 'rna':
                all_rna_genes.append(latents.detach().cpu().numpy())
                all_rna_celltypes.append(celltypes)
            elif modality == 'atac':
                all_atac_genes.append(latents.detach().cpu().numpy())
                all_atac_celltypes.append(celltypes)

        ## reconstruction loss
        loss = loss_fn(features_recon, features)
        pretrain_loss += (loss/num_batches).detach().cpu().numpy()

        if (optimizer is not None) and (not eval):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n=0
        if save_latents=='itr' and np.isin( pretrain_itr_pbar.n , (np.linspace(0,1,n+1) * num_batches).astype(int) ).item(): # save latents at each 1/nth iteration
            epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])
            itr   = str(pretrain_itr_pbar.n).zfill(kwargs['n_batches_str_length'])

            if optimizer is not None:
                filename = f'latents_train_pretrain_epoch_{epoch}_itr_{itr}.npz'
            elif eval:
                filename = f'latents_train_eval_pretrain_epoch_{epoch}_itr_{itr}.npz'
            else:
                filename = f'latents_valid_pretrain_epoch_{epoch}_itr_{itr}.npz'

            filepath = os.path.join(os.path.join(kwargs['outdir'], filename))
            np.savez_compressed(filepath, latents=latents.detach().cpu().numpy(), loss=loss.detach().cpu().numpy(), celltypes=celltypes)

    if save_latents:
        epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])

        if optimizer is not None:
            filename = f'latents_train_pretrain_epoch_{epoch}.npz'
        elif eval:
            filename = f'latents_train_eval_pretrain_epoch_{epoch}.npz'
        else:
            filename = f'latents_valid_pretrain_epoch_{epoch}.npz'

        all_rna_genes = np.vstack(all_rna_genes); all_atac_genes = np.vstack(all_atac_genes)
        all_rna_celltypes = np.hstack(all_rna_celltypes); all_atac_celltypes = np.hstack(all_atac_celltypes)        

        filepath = os.path.join(os.path.join(kwargs['outdir'], filename))
        if modality == 'rna':
            np.savez_compressed(filepath, latents=all_rna_genes, celltypes=all_rna_celltypes)
        elif modality == 'atac':
            np.savez_compressed(filepath, latents=all_atac_genes, celltypes=all_atac_celltypes)
        
    return pretrain_loss


def pretrain_pass(rna_loader, atac_loader, model, optimizer_pretrain_unimodal, optimizer_pretrain_bimodal, rna_loss_fn, atac_loss_fn, num_batches,
                  do_recon_loss=True, do_unimodal_ortho_loss=False, do_bimodal_ortho_loss=False, eval=False, save_latents=False, **kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if save_latents:
        all_rna_latents, all_atac_latents = [], []
        all_rna_celltypes, all_atac_celltypes = [], []
        all_rna_batch_labels, all_atac_batch_labels = [], []

    rna_ortho_losses, atac_ortho_losses = 0, 0
    rna_recon_losses, atac_recon_losses = 0, 0
    bimodal_ortho_losses = 0
    bimodal_losses = 0

    #torch.autograd.set_detect_anomaly(True)
        
    #for (rna_genes, rna_celltypes, rna_batch_labels), (atac_peaks, atac_celltypes, atac_batch_labels) in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):
    for rna_dat, atac_dat in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):

        if (optimizer_pretrain_unimodal is not None) and (optimizer_pretrain_bimodal is not None):
            align_itr_pbar.set_description('TRAIN itr -- PRETRAIN')
        elif eval:
            align_itr_pbar.set_description('TRAIN EVAL itr -- PRETRAIN')
        else:
            align_itr_pbar.set_description('VALID itr -- PRETRAIN')

        ## project RNA data and compute losses
        #rna_genes = rna_genes.to_dense().squeeze(1).to(device=device)
        rna_genes = rna_dat.X.float() # already float32
        rna_celltypes = rna_dat.obs['cell_type']#.tolist()
        rna_batch_labels = np.zeros(len(rna_dat))# rna_dat.obs['subject']#.tolist()

        rna_genes.requires_grad_()
        rna_genes_recon, rna_latents = model(rna_genes, modality='rna', task='pretrain')
        #rna_latents = rna_latents - rna_latents.mean(dim=0, keepdim=True)  # center latents

        ## project ATAC data and compute losses
        #atac_peaks = atac_peaks.to_dense().squeeze(1).to(device=device)
        atac_peaks = atac_dat.X.float()
        atac_celltypes = atac_dat.obs['cell_type']#.tolist()
        atac_batch_labels = np.zeros(len(atac_dat))# atac_dat.obs['subject']#.tolist()

        atac_peaks.requires_grad_()
        atac_peaks_recon, atac_latents = model(atac_peaks, modality='atac', task='pretrain')
        #atac_latents = atac_latents - atac_latents.mean(dim=0, keepdim=True)  # center latents

        if do_recon_loss:
            rna_recon_loss = rna_loss_fn(rna_genes_recon, rna_genes); rna_recon_losses += (rna_recon_loss/num_batches).detach().cpu().numpy()
            atac_recon_loss = atac_loss_fn(atac_peaks_recon, atac_peaks); atac_recon_losses += (atac_recon_loss/num_batches).detach().cpu().numpy()
        else:
            rna_recon_loss = atac_recon_loss = 0


        '''
        if do_unimodal_ortho_loss:
            rna_latents_gram = torch.matmul(rna_latents.T, rna_latents)
            rna_ortho_loss = rna_latents_gram.triu().pow(2).mean()
            rna_ortho_loss /= 10
            rna_ortho_losses += (rna_ortho_loss/num_batches).detach().cpu().numpy()
        else:
            rna_ortho_loss = 0

        if do_unimodal_ortho_loss and not do_bimodal_ortho_loss:
            rna_loss = rna_recon_loss + rna_ortho_loss

            if (optimizer_pretrain_unimodal is not None) and (not eval):
                optimizer_pretrain_unimodal.zero_grad()
                rna_loss.backward()
                optimizer_pretrain_unimodal.step()
        '''


        if optimizer_pretrain_bimodal is not None:
            optimizer_pretrain_bimodal.zero_grad()
            rna_recon_loss.backward()
            atac_recon_loss.backward()
            optimizer_pretrain_bimodal.step()

        '''
        if do_unimodal_ortho_loss:
            atac_latents_gram = torch.matmul(atac_latents.T, atac_latents)
            atac_ortho_loss = atac_latents_gram.triu().pow(2).mean()
            atac_ortho_loss /= 10
            atac_ortho_losses += (atac_ortho_loss/num_batches).detach().cpu().numpy()
        else:
            atac_ortho_loss = 0

        if do_unimodal_ortho_loss and not do_bimodal_ortho_loss:
            atac_loss = atac_recon_loss + atac_ortho_loss

            if (optimizer_pretrain_unimodal is not None) and (not eval):
                optimizer_pretrain_unimodal.zero_grad()            
                atac_loss.backward(retain_graph=True)
                optimizer_pretrain_unimodal.step()

        ## bimodal ortho loss
        if do_bimodal_ortho_loss:

            #bimodal_latents_gram = torch.matmul(rna_latents.T, atac_latents)
            #bimodal_ortho_loss_offdiag = bimodal_latents_gram.triu().pow(2).mean()
            #bimodal_ortho_loss_trace = bimodal_latents_gram.diag().mean()
            #bimodal_ortho_loss = bimodal_ortho_loss_offdiag - bimodal_ortho_loss_trace

            bimodal_ortho_loss = SamplesLoss(loss="gaussian", blur=1)(rna_latents, atac_latents)

            bimodal_ortho_losses += (bimodal_ortho_loss/num_batches).detach().cpu().numpy()

            #prop_epochs = kwargs['epoch'] / (kwargs['total_epochs'] - 1)
            sched = 0 #(1 - np.cos( prop_epochs * np.pi )) / 2

            bimodal_loss = (sched*bimodal_ortho_loss + 0.5*rna_ortho_loss + 0.5*atac_ortho_loss) + (rna_recon_loss + atac_recon_loss)
            bimodal_losses += (bimodal_loss/num_batches).detach().cpu().numpy()

            if (optimizer_pretrain_bimodal is not None) and (not eval):
                optimizer_pretrain_bimodal.zero_grad()
                bimodal_loss.backward()
                optimizer_pretrain_bimodal.step()

        else:
            bimodal_ortho_loss = 0

        '''

        ## save latents
        if save_latents:
            all_rna_latents.append(rna_latents.detach().cpu().numpy())
            all_atac_latents.append(atac_latents.detach().cpu().numpy())
            all_rna_celltypes.append(rna_celltypes)
            all_atac_celltypes.append(atac_celltypes)
            all_rna_batch_labels.append(rna_batch_labels)
            all_atac_batch_labels.append(atac_batch_labels)
            #all_rna_batch_labels.append(rna_batch_labels.detach().cpu().numpy())
            #all_atac_batch_labels.append(atac_batch_labels.detach().cpu().numpy())

    
    ## set checkpoints for saving latents
    n = 4  # set to -1 to inactivate batch-level save latents
    save_latents_ckpts_epochs = (np.linspace(0,1,n+1) * kwargs['total_epochs']).astype(int)
    save_latents_ckpts_epochs[-1] = save_latents_ckpts_epochs[-1] - 1

    if save_latents and np.isin( kwargs['epoch'] , save_latents_ckpts_epochs ).item(): # save latents at each 1/nth iteration
        epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])

        if (optimizer_pretrain_unimodal is not None) and (optimizer_pretrain_bimodal is not None):
            filename = f'latents_train_pretrain_epoch_{epoch}.npz'
        elif eval:
            filename = f'latents_train_eval_pretrain_epoch_{epoch}.npz'
        else:
            filename = f'latents_valid_pretrain_epoch_{epoch}.npz'

        all_rna_latents = np.vstack(all_rna_latents); all_atac_latents = np.vstack(all_atac_latents)
        all_rna_celltypes = np.hstack(all_rna_celltypes); all_atac_celltypes = np.hstack(all_atac_celltypes)
        all_rna_batch_labels = np.hstack(all_rna_batch_labels); all_atac_batch_labels = np.hstack(all_atac_batch_labels)

        filepath = os.path.join(os.path.join(kwargs['outdir'], filename))

        np.savez_compressed(filepath, rna=all_rna_latents, atac=all_atac_latents, \
                            rna_celltypes=all_rna_celltypes, atac_celltypes=all_atac_celltypes, \
                            rna_recon_loss=rna_recon_losses, atac_recon_loss=atac_recon_losses, \
                            rna_ortho_loss=rna_ortho_losses, atac_ortho_loss=atac_ortho_losses, \
                            bimodal_ortho_loss=bimodal_ortho_losses,\
                            rna_batches=all_rna_batch_labels, atac_batches=all_atac_batch_labels)


    if do_bimodal_ortho_loss:
        return bimodal_losses
    
    elif do_unimodal_ortho_loss and not do_bimodal_ortho_loss:
        return rna_ortho_losses + atac_ortho_losses
    
    elif do_recon_loss:
        return rna_recon_losses, atac_recon_losses


def bimodal_align_pass(rna_loader,
                       atac_loader,
                       target_rna_loader,
                       target_atac_loader,
                       model, batch_classifier,
                       device, optimizer, optimizer_batch_loss_minimizer, optimizer_batch_loss_maximizer, scheduler, loss_fn, mmd_loss_fn, batch_classification_loss, num_batches, triplet_type, tune_hyperparameters, eval=False, save_latents=False, **kwargs):

    align_losses, align_losses_atac, align_losses_rna = 0, 0, 0
    #losses_atac_ct = pd.DataFrame(0, index = np.arange(1), columns = list(atac_loader.dataset.dataset.y.categories) + list(['ALL']))
    #losses_rna_ct = pd.DataFrame(0, index = np.arange(1), columns = list(rna_loader.dataset.dataset.y.categories) + list(['ALL']))

    if save_latents:
        all_rna_latents, all_atac_latents = [], []
        all_rna_celltypes, all_atac_celltypes = [], []
        all_rna_batch_labels, all_atac_batch_labels = [], []
        
    #for (rna_cells, rna_celltypes, rna_batch_labels), (atac_cells, atac_celltypes, atac_batch_labels) in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):
    for rna_dat, atac_dat in (align_itr_pbar := tqdm( zip(rna_loader, atac_loader))):

        if (optimizer is not None):
            align_itr_pbar.set_description('TRAIN itr -- ALIGN')
        elif eval:
            align_itr_pbar.set_description('TRAIN EVAL itr -- ALIGN')
        else:
            align_itr_pbar.set_description('VALID itr -- ALIGN')

        ## check if cells loaded in matching pairs
        #assert (rna_dat.obs_names == atac_dat.obs_names).all()
        
        ## project RNA data
        #rna_cells = rna_cells.to_dense().squeeze(1).to(device=device)
        rna_cells = rna_dat.X.float() # already float32
        rna_celltypes = rna_dat.obs['cell_type'].tolist()
        rna_batch_labels = np.zeros(len(rna_dat)) # rna_dat.obs['subject'].tolist()

        rna_cells.requires_grad_()
        rna_latents, rna_genes = model(rna_cells, modality='rna', task='align')

        ## project ATAC data
        #atac_cells = atac_cells.to_dense().squeeze(1).to(device=device)
        atac_cells = atac_dat.X.float()
        atac_celltypes = atac_dat.obs['cell_type'].tolist()
        atac_batch_labels = np.zeros(len(atac_dat)) # atac_dat.obs['subject'].tolist()

        atac_cells.requires_grad_()
        atac_latents, atac_genes = model(atac_cells, modality='atac', task='align')

        ## obtain single set of batch labels
        #assert np.all(rna_batch_labels == atac_batch_labels)
        #batch_labels = torch.nn.functional.one_hot(rna_batch_labels).float() if rna_batch_labels is not None else None

        if save_latents:
            all_rna_latents.append(rna_latents.detach().cpu().numpy())
            all_atac_latents.append(atac_latents.detach().cpu().numpy())
            all_rna_celltypes.append(rna_celltypes)
            all_atac_celltypes.append(atac_celltypes)
            #all_rna_batch_labels.append(rna_batch_labels)
            #all_atac_batch_labels.append(atac_batch_labels)

        if triplet_type != 'clip':
            loss_atac, loss_rna = get_triplet_loss(atac_latents, rna_latents, atac_celltypes, rna_celltypes, triplet_type='mnn', loss_fn=None)
        elif triplet_type == 'clip':
            temperature = loss_fn.temperature
            loss_atac, loss_rna = loss_fn(atac_latents, rna_latents, atac_celltypes, rna_celltypes, temperature=temperature)

        ## Compute MMD loss
        atac_genes = None
        if atac_genes is not None:

            rna_genes = nn.functional.normalize(rna_genes, dim=0)
            atac_genes = nn.functional.normalize(atac_genes, dim=0)
            #atac_genes = -atac_genes # flip to recreate cosine disimilarity


            
            #target_loss = target_loss_per_ct(atac_genes, rna_genes, atac_celltypes, rna_celltypes)
            

        ## Sum losses - more memory efficient to call backward on each loss separately (not implemented here yet)

        loss = (loss_atac + loss_rna)

        align_losses_atac += (loss_atac/num_batches).detach().cpu().numpy()
        align_losses_rna += (loss_rna/num_batches).detach().cpu().numpy()
        align_losses += (loss/num_batches).detach().cpu().numpy()


        if (optimizer is not None) and (not eval):
            ## CLIP loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            '''
            ## batch loss minimizer
            optimizer_batch_loss_minimizer.zero_grad()
            batch_loss_minimizer.backward(retain_graph=True)
            optimizer_batch_loss_minimizer.step()

            ## batch loss maximizer
            optimizer_batch_loss_maximizer.zero_grad()
            batch_loss_maximizer.backward(retain_graph=True)
            optimizer_batch_loss_maximizer.step()
            '''

        #else:
        #    atac_and_rna_latents = anndata.concat([anndata.AnnData(atac_latents.detach().cpu().numpy(), obs={'modality':'atac'}), anndata.AnnData(rna_latents.detach().cpu().numpy(), {'modality':'rna'})])
        #    atac_and_rna_latents.obs['modality'] = atac_and_rna_latents.obs['modality'].astype('category')
        #    ilisi = ilisi_graph(atac_and_rna_latents, batch_key='modality', type_='full')
        #    ilisis += ilisi / num_batches

        n = 0  # set to -1 to inactivate batch-level save latents
        save_latents_ckpts_batches = (np.linspace(0,1,n+1) * num_batches).astype(int)

        if save_latents=='itr' and np.isin( align_itr_pbar.n , save_latents_ckpts_batches ).item(): # save latents at each 1/nth iteration

            epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])
            itr   = str(align_itr_pbar.n).zfill(kwargs['n_batches_str_length'])

            if optimizer is not None:
                filename = f'latents_train_epoch_{epoch}_itr_{itr}.npz'
            elif eval:
                filename = f'latents_train_eval_epoch_{epoch}_itr_{itr}.npz'
            else:
                filename = f'latents_valid_epoch_{epoch}_itr_{itr}.npz'

            filepath = os.path.join(os.path.join(kwargs['outdir'], filename))

            rna_ = rna_latents.detach().cpu().numpy()
            atac_ = atac_latents.detach().cpu().numpy()
            loss_ = loss.detach().cpu().numpy()
            loss_rna_ = loss_rna.detach().cpu().numpy()
            loss_atac_ = loss_atac.detach().cpu().numpy()
            mmd_loss_ = mmd_loss.detach().cpu().numpy()
            np.savez_compressed(filepath, rna=rna_, atac=atac_, loss=loss_, loss_rna=loss_rna_, loss_atac=loss_atac_, mmd_loss=mmd_loss_, rna_celltypes=rna_celltypes, atac_celltypes=atac_celltypes)


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
                
                ## MDD latents when valid
                with torch.no_grad():
                    rna_cells, rna_celltypes, atac_cells, atac_celltypes = fetch_data_from_loaders(target_rna_loader, target_atac_loader, paired=False, subsample=2000)

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
            rna_cells, rna_celltypes, atac_cells, atac_celltypes = fetch_data_from_loaders(rna_loader, atac_loader)
            ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, acc, acc_top5, clip_loss_, clip_loss_censored, \
                foscttm_score_ct, acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split = align_metrics(model, rna_cells, rna_celltypes, atac_cells, atac_celltypes) # should be valid loaders
            
            align_metrics_dict = {'loss_rna':loss_rna.item(), 'loss_atac':loss_atac.item(), \
                                  'clip_loss': clip_loss_, 'clip_loss_censored': clip_loss_censored, \
                                  'ilisi':ilisis, 'clisi':clisis, 'nmi':nmi, 'ari':ari, 'diag_concentration_minimizer':diag_concentration_minimizer, 'foscttm':foscttm_score, 'acc':acc, 'acc_top5':acc_top5}

            ## evaluate on MDD data
            ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, acc, acc_top5 = compute_mdd_eval_metrics(model, target_rna_loader, target_atac_loader, device, mdd_eval_method='subsample')
            target_align_metrics_dict = {'ilisi':ilisis, 'clisi':clisis, 'nmi':nmi, 'ari':ari, 'diag_concentration_minimizer':diag_concentration_minimizer, 'foscttm':foscttm_score, 'acc':acc, 'acc_top5':acc_top5}

    elif (optimizer is None) and tune_hyperparameters:
        foscttm_score = foscttm_moscot(rna_latents.detach().cpu().numpy(), atac_latents.detach().cpu().numpy())
        align_metrics_dict = {'foscttm':foscttm_score.item()}
        target_align_metrics_dict = acc_ct = acc_top5_ct = clip_loss_ct = clip_loss_ct_split = clip_loss_ = clip_loss_censored = None

    else:
        align_metrics_dict = target_align_metrics_dict = acc_ct = acc_top5_ct = clip_loss_ct = clip_loss_ct_split = clip_loss_ = clip_loss_censored = None

    return align_losses, align_losses_atac, align_losses_rna, mmd_losses, align_metrics_dict, target_align_metrics_dict, clip_loss_, clip_loss_censored,\
        acc_ct, acc_top5_ct, clip_loss_ct, clip_loss_ct_split



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

        ## obtain single set of batch labels
        #assert np.all(rna_batch_labels == atac_batch_labels)
        #batch_labels = torch.nn.functional.one_hot(rna_batch_labels).float() if rna_batch_labels is not None else None

        if save_latents:
            all_rna_latents.append(rna_latents.detach().cpu().numpy())
            all_atac_latents.append(atac_latents.detach().cpu().numpy())
            all_rna_celltypes.append(rna_celltypes)
            all_atac_celltypes.append(atac_celltypes)
            all_rna_batch_labels.append(rna_batch_labels)
            all_atac_batch_labels.append(atac_batch_labels)

        ## Align losses
        if triplet_type != 'clip':
            loss_atac, loss_rna = get_triplet_loss(atac_latents, rna_latents, atac_celltypes, rna_celltypes, triplet_type='mnn', loss_fn=None)
        elif triplet_type == 'clip':
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

        #else:
        #    atac_and_rna_latents = anndata.concat([anndata.AnnData(atac_latents.detach().cpu().numpy(), obs={'modality':'atac'}), anndata.AnnData(rna_latents.detach().cpu().numpy(), {'modality':'rna'})])
        #    atac_and_rna_latents.obs['modality'] = atac_and_rna_latents.obs['modality'].astype('category')
        #    ilisi = ilisi_graph(atac_and_rna_latents, batch_key='modality', type_='full')
        #    ilisis += ilisi / num_batches

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

        if save_latents=='itr' and np.isin( align_itr_pbar.n , save_latents_ckpts_batches ).item(): # save latents at each 1/nth iteration

            epoch = str(kwargs['epoch']).zfill(kwargs['n_epochs_str_length'])
            itr   = str(align_itr_pbar.n).zfill(kwargs['n_batches_str_length'])

            if optimizer is not None:
                filename = f'latents_train_epoch_{epoch}_itr_{itr}.npz'
            elif eval:
                filename = f'latents_train_eval_epoch_{epoch}_itr_{itr}.npz'
            else:
                filename = f'latents_valid_epoch_{epoch}_itr_{itr}.npz'

            filepath = os.path.join(os.path.join(kwargs['outdir'], filename))

            rna_ = rna_latents.detach().cpu().numpy()
            atac_ = atac_latents.detach().cpu().numpy()
            loss_ = loss.detach().cpu().numpy()
            loss_rna_ = loss_rna.detach().cpu().numpy()
            loss_atac_ = loss_atac.detach().cpu().numpy()
            mmd_loss_ = mmd_loss.detach().cpu().numpy()
            np.savez_compressed(filepath, rna=rna_, atac=atac_, loss=loss_, loss_rna=loss_rna_, loss_atac=loss_atac_, mmd_loss=mmd_loss_, rna_celltypes=rna_celltypes, atac_celltypes=atac_celltypes)


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


def save_latents(student_rna_latents_valid, student_atac_latents_valid, student_rna_celltypes_valid, all_atac_celltypes_valid, epoch, n_epochs, outdir):
    ## in reality, only a single batch of valid latents per epoch, so no need to accumulate

    n = 4  # set to -1 to inactivate epoch-level save latents
    save_latents_ckpts_epochs = (np.linspace(0,1,n+1) * n_epochs).astype(int)
    save_latents_ckpts_epochs[-1] = save_latents_ckpts_epochs[-1] - 1 # ensure that last checkpoint is the last epoch

    if np.isin(epoch, save_latents_ckpts_epochs).item():

        n_epochs_str_length = len(str(args.n_epochs - 1))
        epoch_str = str(epoch).zfill(n_epochs_str_length)

        filename = f'latents_valid_epoch_{epoch_str}.npz'
            
        all_rna_latents_valid = student_rna_latents_valid.detach().cpu().numpy(); all_atac_latents_valid = student_atac_latents_valid.detach().cpu().numpy()
        all_rna_celltypes_valid = student_rna_celltypes_valid; all_atac_celltypes_valid = student_atac_celltypes_valid

        filepath = os.path.join(outdir, filename)
        np.savez_compressed(filepath, rna=all_rna_latents_valid, atac=all_atac_latents_valid, rna_celltypes=all_rna_celltypes_valid, atac_celltypes=all_atac_celltypes_valid)
