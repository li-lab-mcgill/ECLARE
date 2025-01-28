#%%
import sys
import os

from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from ot import solve as ot_solve
from glob import glob

from ott.geometry.geometry import Geometry
import jax.numpy as jnp

from eclare import \
    scTripletgrate, load_scTripletgrate_model, return_setup_func_from_dataset, mdd_setup, align_metrics, fetch_data_from_loaders, clip_loss

class Knowledge_distillation_fn(torch.nn.Module):

    def __init__(self, device='cpu', kd_loss='KL', paired=True, student_temperature=1, target_temperature=1, weigh_distil_by_align_type='none'):
        super(Knowledge_distillation_fn, self).__init__()
        self.device = device
        self.kd_loss = kd_loss
        self.paired = paired
        self.align_loss_scale = 1 # to be updated later in script
        self.weigh_distil_by_align_type = weigh_distil_by_align_type

        ## lower temperature = lower entropy, and vice versa
        self.student_temperature = torch.tensor(student_temperature, requires_grad=False).to(device)
        self.target_temperature = torch.tensor(target_temperature, requires_grad=False).to(device)

        ## set distillation loss function    
        if kd_loss == 'CE':
            self.distil_loss_fn = self.ce_forward
        elif kd_loss == 'KL':
            self.distil_loss_fn = self.kl_forward

        ## set alignment loss function for target latents
        if not paired:
            #self.geom_loss_fn = SamplesLoss(loss='gaussian', blur=0.05)
            #self.align_loss_forward = self.geom_loss_forward
            self.align_loss_forward = self.ot_clip_loss_forward
            #self.solve_fn = jax.jit(linear.solve)
        else:
            self.align_loss_forward = self.ot_clip_loss_forward # for now, use OT-CLIP in all cases

        self.all_teacher_ot_plans = []
        self.all_teacher_ot_values = []

    ## forward functions for different KD losses
    def kl_forward(self, student_logits, target_logits):
        student_log_softmax = F.log_softmax(student_logits, dim=1)
        target_log_softmax = F.log_softmax(target_logits, dim=1)
        loss = torch.nn.functional.kl_div(student_log_softmax, target_log_softmax.detach(), reduction='none', log_target=True)

        student_log_softmax_T = F.log_softmax(student_logits.T, dim=1)
        target_log_softmax_T = F.log_softmax(target_logits.T, dim=1)
        loss_T = torch.nn.functional.kl_div(student_log_softmax_T, target_log_softmax_T.detach(), reduction='none', log_target=True)

        ## sum over "target" nuclei, end up with a loss per "student" nucleus, whereas batchmean will still average across all nuclei
        loss = loss.sum(1)
        loss_T = loss_T.sum(1)

        return loss, loss_T

    def ce_forward(self, student_logits, target_logits):
        target_softmax = F.softmax(target_logits, dim=1)
        loss = torch.nn.functional.cross_entropy(student_logits, target_softmax.detach(), weight=None)

        target_softmax_T = F.softmax(target_logits.T, dim=1)
        loss_T = torch.nn.functional.cross_entropy(student_logits.T, target_softmax_T.detach(), weight=None)
        return loss, loss_T

    ## forward functions for different alignment losses
    def clip_loss_forward(self, teacher_logits, student_logits, reduce=False):

        if teacher_logits is not None:
            logits = teacher_logits
        elif student_logits is not None:
            logits = student_logits
        else:
            raise ValueError('One of teacher_logits or student_logits must be provided')
        
        clip_loss_atac, clip_loss_rna = clip_loss(logits, None, None, temperature=self.student_temperature, return_logits=False, reduce=reduce) # for now, using self.student_temperature for both student and target
        #clip_loss_ = 0.5 * (clip_loss_rna + clip_loss_atac)
        return clip_loss_atac, clip_loss_rna 
    
    def geom_loss_forward(self, atac_latents, rna_latents):
        geom_loss = self.geom_loss_fn(atac_latents, rna_latents)
        return geom_loss
    
    def ot_clip_loss_forward(self, teacher_logits, student_logits=None, solver='pot'):

        # student loss
        if (student_logits is not None):

            ## obtain teacher weights
            ot_clip_loss_weights = (1 - torch.stack(self.all_teacher_ot_values)).softmax(dim=0).to(device=self.device)  # in principle, would also have values & weights for plan_T
            ot_clip_loss = torch.zeros(len(student_logits), device=self.device)
            ot_clip_loss_T = torch.zeros(len(student_logits), device=self.device)

            ## obtain weighted align loss per teacher
            for t, teacher_ot_plan in enumerate(self.all_teacher_ot_plans):
                labels = torch.argmax(teacher_ot_plan, dim=1)
                labels_T = torch.argmax(teacher_ot_plan.T, dim=1)

                ot_clip_loss = torch.nn.functional.cross_entropy(student_logits, labels, reduction='none')
                ot_clip_loss_T = torch.nn.functional.cross_entropy(student_logits.T, labels_T, reduction='none')

                ot_clip_loss = ot_clip_loss + (ot_clip_loss_weights[t] * ot_clip_loss)
                ot_clip_loss_T = ot_clip_loss_T + (ot_clip_loss_weights[t] * ot_clip_loss_T)

            self.all_teacher_ot_plans = []
            self.all_teacher_ot_values = []


        # teacher loss
        else:
            teacher_cost = 1 - (teacher_logits / torch.exp(1/self.target_temperature))

            if solver == 'pot':
                ot_res = ot_solve(teacher_cost)
                plan = ot_res.plan
                plan_T = plan.T
                #plan_T = ot_solve(teacher_cost.T).plan  # empirically, plan_T != plan.T
                value = ot_res.value_linear

            elif solver == 'jax-ot':
                geom = Geometry(cost_matrix = jnp.asarray(teacher_cost.cpu()))
                geom_T = Geometry(cost_matrix = jnp.asarray(teacher_cost_T.cpu()))

                ot_output = self.solve_fn(geom)
                ot_output_T = self.solve_fn(geom_T)

                plan = torch.from_numpy(np.asarray(ot_output.matrix)).to(device=device)
                plan_T = torch.from_numpy(np.asarray(ot_output_T.matrix)).to(device=device)

            self.all_teacher_ot_plans.append(plan)
            self.all_teacher_ot_values.append(value)

            labels = torch.argmax(plan, dim=1)
            labels_T = torch.argmax(plan_T, dim=1)  # plan_T != plan.T

            ot_clip_loss = torch.nn.functional.cross_entropy(teacher_logits, labels, reduction='none')
            ot_clip_loss_T = torch.nn.functional.cross_entropy(teacher_logits.T, labels_T, reduction='none')

        return ot_clip_loss, ot_clip_loss_T


    def forward(self, student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, teacher_or_student, dataset_embedding=None):

        ## beware, since adjusting target latents contradicts the teacher-student paradigm
        #target_rna_latents = target_rna_latents + dataset_embedding
        #target_atac_latents = target_atac_latents + dataset_embedding
        #student_rna_latents = student_rna_latents + dataset_embedding
        #student_atac_latents = student_atac_latents + dataset_embedding

        ## already normalized during clip loss, but need to normalize before to be consistent with Concerto
        target_rna_latents = torch.nn.functional.normalize(target_rna_latents, p=2, dim=1)
        target_atac_latents = torch.nn.functional.normalize(target_atac_latents, p=2, dim=1)
        student_rna_latents = torch.nn.functional.normalize(student_rna_latents, p=2, dim=1)
        student_atac_latents = torch.nn.functional.normalize(student_atac_latents, p=2, dim=1)

        ## get logits
        student_logits = torch.matmul(student_atac_latents, student_rna_latents.T) * torch.exp(1/self.student_temperature)
        target_logits = torch.matmul(target_atac_latents, target_rna_latents.T) * torch.exp(1/self.target_temperature)

        ## get distillation loss
        distil_loss, distil_loss_T = self.distil_loss_fn(student_logits, target_logits)

        ## get alignment loss
        if teacher_or_student == 'student':
            align_loss, align_loss_T = self.align_loss_forward(None, student_logits)
            align_loss = 0.5 * (align_loss + align_loss_T).mean()
            offset = student_logits.exp().sum(1).log().mean()
            offset_T = student_logits.T.exp().sum(1).log().mean()
            align_loss_T_scaled = None

        elif teacher_or_student == 'teacher':
            align_loss, align_loss_T = self.align_loss_forward(target_logits, None)
            offset = target_logits.exp().sum(1).log()
            offset_T = target_logits.T.exp().sum(1).log()
            align_loss_T_scaled = self.align_loss_scale * align_loss_T

        ## scale alignment loss
        align_loss_scaled = self.align_loss_scale * align_loss

        offset_scaled = self.align_loss_scale * offset
        offset_T_scaled = self.align_loss_scale * offset_T

        return distil_loss, distil_loss_T, align_loss_scaled, align_loss_T_scaled, offset_scaled, offset_T_scaled
    
    
    def distil_loss_weighting(self, distil_losses, distil_losses_T, align_losses_scaled_offset, align_losses_T_scaled_offset):

        ## for 'batch' or 'sample', gets overwritten if its 'none'
        align_losses_weights = torch.softmax(distil_losses, dim=0)
        align_losses_T_weights = torch.softmax(distil_losses_T, dim=0)

        if self.weigh_distil_by_align_type == 'none': # in reality, no need to create uniform weights, but leads to losses on more similar scales than other align types

            ## overwrite weights to obtain uniform weights
            align_losses_weights = align_losses_T_weights = torch.ones_like(align_losses_weights) / len(align_losses_weights)

            distil_loss = (distil_losses * align_losses_weights).sum(0)                         # teacher-based weighting (pointwise)
            distil_loss_T = (distil_losses_T * align_losses_T_weights).sum(0)                   # teacher-based weighting (pointwise)
            distil_loss = 0.5 * (distil_loss + distil_loss_T).mean()  

        elif self.weigh_distil_by_align_type == 'batch':# or (not self.paired and self.weigh_distil_by_align_type != 'none'): # for MMD align loss, cannot do 'sample', so default to 'batch'
            distil_loss = (distil_losses * align_losses_weights.unsqueeze(1)).sum(0)            # teacher-based weighting (broadcasted)
            distil_loss_T = (distil_losses_T * align_losses_T_weights.unsqueeze(1)).sum(0)      # teacher-based weighting (broadcasted)
            distil_loss = 0.5 * (distil_loss + distil_loss_T).mean()                            # sample-based averaging

        elif self.weigh_distil_by_align_type == 'sample':
            distil_loss = (distil_losses * align_losses_weights).sum(0)                         # teacher-based weighting (pointwise)
            distil_loss_T = (distil_losses_T * align_losses_T_weights).sum(0)                   # teacher-based weighting (pointwise)
            distil_loss = 0.5 * (distil_loss + distil_loss_T).mean()                            # sample-based averaging

        distil_loss         = 0.5 * torch.stack([distil_losses, distil_losses_T]).sum(0).mean()  # close to 'batchmean' reduction of KL divergence, but not identical
        align_loss_scaled   = 0.5 * torch.stack([align_losses_scaled_offset, align_losses_T_scaled_offset]).sum(0).mean()

        return distil_loss, align_loss_scaled


def teachers_setup(model_paths):
    datasets = []
    models = {}
    target_rna_train_loaders = {}
    target_atac_train_loaders = {}
    target_rna_valid_loaders = {}
    target_atac_valid_loaders = {}
    
    for m, model_path in enumerate(model_paths):

        print(model_path)

        ## Load the model
        model, model_args_dict = load_scTripletgrate_model(model_path, device=device)

        ## Determine the dataset
        dataset = model_args_dict['args'].source_dataset
        source_setup_func = return_setup_func_from_dataset(dataset)
        target_setup_func = return_setup_func_from_dataset(model_args_dict['args'].target_dataset)
        genes_by_peaks_str = model_args_dict['args'].genes_by_peaks_str

        print(dataset)
        datasets.append(dataset)

        ## Load the data loaders
        #rna_train_loader, atac_train_loader, atac_train_num_batches, atac_train_n_batches_str_length, atac_train_n_epochs_str_length, rna_valid_loader, atac_valid_loader, atac_valid_num_batches, atac_valid_n_batches_str_length, atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask = \
        #    source_setup_func(model_args_dict['args'], pretrain=None, return_type='loaders', dataset=dataset)
        
        overlapping_subjects_only = False #True if args.dataset == 'roussos' else False
        target_rna_train_loader, target_atac_train_loader, _, _, _, target_rna_valid_loader, target_atac_valid_loader, _, _, _, _, _, _, _, _ =\
            target_setup_func(model_args_dict['args'], pretrain=None, return_type='loaders')
        
        if args.train_encoders:
            model.train()

        models[dataset] = model

        if args.source_dataset_embedder:
            dataset_idx_dict[dataset] = m

        target_rna_train_loaders[dataset] = target_rna_train_loader
        target_atac_train_loaders[dataset] = target_atac_train_loader
        target_rna_valid_loaders[dataset] = target_rna_valid_loader
        target_atac_valid_loaders[dataset] = target_atac_valid_loader

    return datasets, models, target_rna_train_loaders, target_atac_train_loaders, target_rna_valid_loaders, target_atac_valid_loaders

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
        

def ct_losses(distil_losses_valid, distil_losses_T_valid, align_losses_valid, align_losses_T_valid, target_atac_celltypes_valid, target_rna_celltypes_valid, datasets, rank=False, stack=True):

    distil_losses_atac_valid_df = pd.DataFrame(distil_losses_valid.cpu().detach().numpy().T, columns=datasets)
    distil_losses_atac_valid_df['cell-types'] = target_atac_celltypes_valid  # should be same cell-types across teachers and students since batches are synced
    distil_losses_atac_per_ct_valid = distil_losses_atac_valid_df.groupby('cell-types').mean()

    distil_losses_rna_valid_df = pd.DataFrame(distil_losses_T_valid.cpu().detach().numpy().T, columns=datasets)
    distil_losses_rna_valid_df['cell-types'] = target_rna_celltypes_valid  # should be same cell-types across teachers and students since batches are synced
    distil_losses_rna_per_ct_valid = distil_losses_rna_valid_df.groupby('cell-types').mean()

    align_losses_atac_valid_df = pd.DataFrame(align_losses_valid.cpu().detach().numpy().T, columns=datasets)
    align_losses_atac_valid_df['cell-types'] = target_atac_celltypes_valid  # should be same cell-types across teachers and students since batches are synced
    align_losses_atac_per_ct_valid = align_losses_atac_valid_df.groupby('cell-types').mean()

    align_losses_rna_valid_df = pd.DataFrame(align_losses_T_valid.cpu().detach().numpy().T, columns=datasets)
    align_losses_rna_valid_df['cell-types'] = target_rna_celltypes_valid  # should be same cell-types across teachers and students since batches are synced
    align_losses_rna_per_ct_valid = align_losses_rna_valid_df.groupby('cell-types').mean()

    if rank:
        distil_losses_atac_per_ct_valid = distil_losses_atac_per_ct_valid.rank(axis=1)
        distil_losses_rna_per_ct_valid = distil_losses_rna_per_ct_valid.rank(axis=1)
        align_losses_atac_per_ct_valid = align_losses_atac_per_ct_valid.rank(axis=1)
        align_losses_rna_per_ct_valid = align_losses_rna_per_ct_valid.rank(axis=1)

    if stack:
        distil_losses_atac_per_ct_valid = distil_losses_atac_per_ct_valid.stack()
        distil_losses_rna_per_ct_valid = distil_losses_rna_per_ct_valid.stack()
        align_losses_atac_per_ct_valid = align_losses_atac_per_ct_valid.stack()
        align_losses_rna_per_ct_valid = align_losses_rna_per_ct_valid.stack()

    return distil_losses_atac_per_ct_valid, distil_losses_rna_per_ct_valid, align_losses_atac_per_ct_valid, align_losses_rna_per_ct_valid



if __name__ == "__main__":

    parser = ArgumentParser(description='')
    parser.add_argument('--outdir', type=str, default=default_outdir,
                        help='output directory')
    parser.add_argument('--slurm_job_ids', default=['37119283'], #['34887192', '34870779', '35175388', '35168470'], # 35318790 (SEA-AD) error when doing data[valid_idx].copy()
                        help='list of datasets to merge')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--loss_type', type=str, default='knowledge_distillation',
                        help='type of loss to use for training')
    parser.add_argument('--train_encoders', action='store_true',
                        help='train the encoders during training (name starting with letter f returns error)')
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
    parser.add_argument('--valid_subsample', type=int, default=5000,
                        help='number of nuclei to subsample for validation')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='source dataset')
    parser.add_argument('--target_dataset', type=str, default=None,
                        help='target dataset')
    parser.add_argument('--replicate_idx', type=int, default=0,
                        help='replicate index')
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
    print('Extracting data')
    slurm_job_ids = args.slurm_job_ids if isinstance(args.slurm_job_ids, list) else [args.slurm_job_ids]
    if len(slurm_job_ids) > 1:
        model_paths = [os.path.join(outpath, f'triplet_align_{slurm_job_id}', 'model.pt') for slurm_job_id in slurm_job_ids]

    else:
        if args.target_dataset is not None:
            target_dataset_og = args.target_dataset
            replicate_idx = str(args.replicate_idx)

            if args.source_dataset is not None:
                source_dataset_og = args.source_dataset
                model_paths = glob(os.path.join(outpath, f'clip_*{slurm_job_ids[0]}/{target_dataset_og}/{source_dataset_og}/{replicate_idx}/model.pt'))
            else:
                model_paths = glob(os.path.join(outpath, f'clip_*{slurm_job_ids[0]}/{target_dataset_og}/**/{replicate_idx}/model.pt'))

            if len(model_paths) == 0:
                model_paths = glob(os.path.join(outpath, f'clip_*{slurm_job_ids[0]}/{target_dataset_og}/{source_dataset_og}/{replicate_idx}/model.pt'))
        
        else:  # for older job, one less sub-directory deep
            model_paths = glob(os.path.join(outpath, f'triplet_align_{slurm_job_ids[0]}/**/model.pt'))

    #model_paths = [model_paths[0], model_paths[1]]

    ## Instantiate student model args dict from one of the source datasets, later overwrite with target dataset
    _, student_model_args_dict = load_scTripletgrate_model(model_paths[0], device=device)
    target_dataset_og = student_model_args_dict['args'].target_dataset
    student_setup_func = return_setup_func_from_dataset(target_dataset_og)

    ## Get number of genes and peaks from genes_by_peaks_str
    n_genes = int(args.genes_by_peaks_str.split('_')[0])
    n_peaks = int(args.genes_by_peaks_str.split('_')[-1])

    ## Overwrite
    if target_dataset_og == 'mdd':
        student_model_args_dict['args'].source_dataset = 'mdd'
        student_model_args_dict['args'].target_dataset = None
    else:
        student_model_args_dict['args'].source_dataset = target_dataset_og
        student_model_args_dict['args'].target_dataset = 'mdd'

    student_model_args_dict['args'].genes_by_peaks_str = args.genes_by_peaks_str
    student_model_args_dict['n_genes'] = n_genes
    student_model_args_dict['n_peaks'] = n_peaks
    student_model_args_dict['tuned_hyperparameters']['params_num_layers'] = 2
    student_model_args_dict['pretrain'] = student_model_args_dict['rna_valid_idx']  = student_model_args_dict['atac_valid_idx'] = None
    #student_model_args_dict['args'].genes_by_peaks_str = None  # setting genes_by_peaks to None creates large processing overhead, can just use one of the preset configs for now

    student_rna_train_loader, student_atac_train_loader, student_atac_train_num_batches, student_atac_train_n_batches_str_length, student_atac_train_n_epochs_str_length, student_rna_valid_loader, student_atac_valid_loader, student_atac_valid_num_batches, student_atac_valid_n_batches_str_length, student_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
        student_setup_func(student_model_args_dict['args'], pretrain=None, return_type='loaders')
    
    mdd_rna_train_loader, mdd_atac_train_loader, mdd_atac_train_num_batches, mdd_atac_train_n_batches_str_length, mdd_atac_train_n_epochs_str_length, mdd_rna_valid_loader, mdd_atac_valid_loader, mdd_atac_valid_num_batches, mdd_atac_valid_n_batches_str_length, mdd_atac_valid_n_epochs_str_length, n_peaks, n_genes, atac_valid_idx, rna_valid_idx, genes_to_peaks_binary_mask =\
        mdd_setup(student_model_args_dict['args'], overlapping_subjects_only=False, pretrain=None, return_type='loaders')

    ## Create student model
    student_model = scTripletgrate(**student_model_args_dict, trial=None).to(device=device)

    ## Create source dataset embedder
    if args.source_dataset_embedder:
        n_units = student_model.rna_to_core[0].out_features
        dataset_idx_dict = {}

        source_dataset_embedder = torch.nn.Sequential(
            torch.nn.Embedding(len(model_paths), n_units, device=device),
            torch.nn.ReLU()
        ).to(device)
    
    ## Setup teachers
    datasets, models, target_rna_train_loaders, target_atac_train_loaders, target_rna_valid_loaders, target_atac_valid_loaders = \
        teachers_setup(model_paths)


    # Define the optimized parameters
    optimized_parameters = list(student_model.parameters())

    if args.train_encoders:
        optimized_parameters += [param for model in models.values() for param in model.parameters()]
    if args.source_dataset_embedder:
        optimized_parameters += list(source_dataset_embedder.parameters())

    optimizer = torch.optim.AdamW(optimized_parameters, lr=1e-3, weight_decay=0.01)

    # Instantiate the knowledge distillation loss function
    paired = (student_model_args_dict['args'].source_dataset != 'mdd')
    knowledge_distillation_fn = Knowledge_distillation_fn(device=device, kd_loss='KL', student_temperature=1, target_temperature=1, paired=paired, weigh_distil_by_align_type='none')

    # Initialize main logger DataFrame
    columns = ['Epoch']
    log_df = pd.DataFrame(columns=columns)

    ## Initialize cell-type logger DataFrames
    all_distil_losses_atac_per_ct_valid = pd.DataFrame()
    all_distil_losses_rna_per_ct_valid = pd.DataFrame()
    all_align_losses_atac_per_ct_valid = pd.DataFrame()
    all_align_losses_rna_per_ct_valid = pd.DataFrame()

    if paired:
        student_foscttm_per_ct_valid = pd.DataFrame()
        teacher_foscttm_per_ct_valid = pd.DataFrame(columns=datasets)

    # Define loop_order parameter
    loop_order = args.loop_order  # 'datasets_first' or 'batches_first'

    print('Iterating over epochs, batches & datasets')
    for epoch in range(args.n_epochs):

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

            if args.source_dataset_embedder:
                source_dataset_embedder.eval()
            #student_model.eval()

            ## Get student data and latents
            student_rna_cells_valid, student_rna_celltypes_valid, student_atac_cells_valid, student_atac_celltypes_valid, rna_nuclei_idx_valid, atac_nuclei_idxs_valid    = fetch_data_from_loaders(student_rna_valid_loader, student_atac_valid_loader, paired=paired, subsample=args.valid_subsample) # reuse same nuclei indices for all student and teacher datasets, or else distillation loss not sensible
            student_rna_latents_valid   = student_model(student_rna_cells_valid, modality='rna', task='align')[0]
            student_atac_latents_valid  = student_model(student_atac_cells_valid, modality='atac', task='align')[0]

            ## Get MDD data and latents
            mdd_rna_cells_valid, mdd_rna_celltypes_valid, mdd_atac_cells_valid, mdd_atac_celltypes_valid, _, _        = fetch_data_from_loaders(mdd_rna_valid_loader, mdd_atac_valid_loader, paired=False, subsample=args.valid_subsample)
            mdd_rna_latents_valid   = student_model(mdd_rna_cells_valid, modality='rna', task='align')[0]
            mdd_atac_latents_valid  = student_model(mdd_atac_cells_valid, modality='atac', task='align')[0]

            ## normalize latents (already being normalized in loss_fn)
            student_rna_latents_valid = torch.nn.functional.normalize(student_rna_latents_valid, p=2, dim=1)
            student_atac_latents_valid = torch.nn.functional.normalize(student_atac_latents_valid, p=2, dim=1)    

            mdd_rna_latents_valid = torch.nn.functional.normalize(mdd_rna_latents_valid, p=2, dim=1)
            mdd_atac_latents_valid = torch.nn.functional.normalize(mdd_atac_latents_valid, p=2, dim=1)
            
            ## loop through teachers
            for dataset in datasets:

                ## Obtain dataset embedding
                if args.source_dataset_embedder:
                    #dataset = np.random.choice(datasets)  # randomly select one of the datasets, but then would also have to select the target dataset accordingly
                    dataset_idx = dataset_idx_dict[dataset]
                    dataset_idx = torch.LongTensor([dataset_idx]).to(device)
                    dataset_embedding = source_dataset_embedder(dataset_idx)
                else:
                    dataset_embedding = None

                model = models[dataset]

                target_rna_valid_loader = target_rna_valid_loaders[dataset]
                target_atac_valid_loader = target_atac_valid_loaders[dataset]
                target_rna_cells_valid, target_rna_celltypes_valid, target_atac_cells_valid, target_atac_celltypes_valid, _, _        = fetch_data_from_loaders(target_rna_valid_loader, target_atac_valid_loader, paired=paired, subsample=args.valid_subsample, rna_cells_idx=rna_nuclei_idx_valid, atac_cells_idx=atac_nuclei_idxs_valid)

                target_rna_latents_valid    = model(target_rna_cells_valid, modality='rna', task='align')[0].detach()
                target_atac_latents_valid   = model(target_atac_cells_valid, modality='atac', task='align')[0].detach()

                distil_loss_valid, distil_loss_valid_T, align_loss_valid, align_loss_T_valid, offset_valid, offset_T_valid = knowledge_distillation_fn(student_rna_latents_valid, student_atac_latents_valid, target_rna_latents_valid, target_atac_latents_valid, 'teacher')

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
                    ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, rank_score_, acc, acc_top5, clip_loss_, clip_loss_censored, \
                    foscttm_score_ct, accuracy_ct, accuracy_top5_ct, clip_loss_ct, clip_loss_ct_split = \
                        align_metrics(model, target_rna_latents_valid, target_rna_celltypes_valid, target_atac_latents_valid, target_atac_celltypes_valid, paired=paired, is_latents=True)
                    epoch_ilisis[dataset], epoch_clisis[dataset], epoch_nmi[dataset], epoch_ari[dataset] = ilisis, clisis, nmi, ari
                    epoch_rank_score[dataset] = rank_score_

                    if paired:
                        epoch_foscttm[dataset] = foscttm_score.item()
                        teacher_foscttm_per_ct_valid[dataset] = foscttm_score_ct


            ## Compute mean distillation loss
            distil_losses_valid     = torch.stack(distil_losses_valid)
            distil_losses_T_valid   = torch.stack(distil_losses_T_valid)
            align_losses_valid      = torch.stack(align_losses_valid)
            align_losses_T_valid    = torch.stack(align_losses_T_valid)
            offsets_valid           = torch.stack(offsets_valid)
            offsets_T_valid         = torch.stack(offsets_T_valid)
            
            ## Compute losses per cell type
            distil_losses_atac_per_ct_valid, distil_losses_rna_per_ct_valid, align_losses_atac_per_ct_valid, align_losses_rna_per_ct_valid = ct_losses(distil_losses_valid, distil_losses_T_valid, align_losses_valid, align_losses_T_valid, target_atac_celltypes_valid, target_rna_celltypes_valid, datasets)
            distil_losses_atac_per_ct_valid.name = distil_losses_rna_per_ct_valid.name = align_losses_atac_per_ct_valid.name = align_losses_rna_per_ct_valid.name = f'{epoch+1}'
            all_distil_losses_atac_per_ct_valid = pd.concat([all_distil_losses_atac_per_ct_valid, distil_losses_atac_per_ct_valid], axis=1)
            all_distil_losses_rna_per_ct_valid  = pd.concat([all_distil_losses_rna_per_ct_valid, distil_losses_rna_per_ct_valid], axis=1)
            all_align_losses_atac_per_ct_valid  = pd.concat([all_align_losses_atac_per_ct_valid, align_losses_atac_per_ct_valid], axis=1)
            all_align_losses_rna_per_ct_valid   = pd.concat([all_align_losses_rna_per_ct_valid, align_losses_rna_per_ct_valid], axis=1)

            ## Compute distillation loss weighted by alignment loss, if more than one teacher
            mean_distil_loss_valid, _ = knowledge_distillation_fn.distil_loss_weighting( distil_losses_valid, distil_losses_T_valid, (offsets_valid - align_losses_valid), (offsets_T_valid - align_losses_T_valid))

            ## Compute student alignment loss
            _, _, align_loss_scaled_valid, _, _, _ = knowledge_distillation_fn(student_rna_latents_valid, student_atac_latents_valid, target_rna_latents_valid, target_atac_latents_valid, 'student')

            ## Set align loss scale if first epoch
            if epoch == 0:
                align_loss_scale = (mean_distil_loss_valid / align_loss_scaled_valid).detach().cpu().numpy().item()
                knowledge_distillation_fn.align_loss_scale = align_loss_scale
                print(f'Align loss scale: {align_loss_scale}')

                ## Retroactively scale teacher & student CLIP losses
                epoch_align_losses_valid    = {dataset: align_loss_scale * epoch_align_losses_valid[dataset] for dataset in datasets} # not sure if updating correctly
                epoch_losses_valid          = {dataset: args.distil_lambda * epoch_distil_losses_valid[dataset] + (1-args.distil_lambda) * epoch_align_losses_valid[dataset] for dataset in datasets}
                align_loss_scaled_valid     = align_loss_scale * align_loss_scaled_valid

                all_align_losses_atac_per_ct_valid.iloc[:,0] *= align_loss_scale
                all_align_losses_rna_per_ct_valid.iloc[:,0] *= align_loss_scale

            ## Compute total loss as convex combination of CLIP loss and average distillation loss
            total_loss_valid = (args.distil_lambda * mean_distil_loss_valid) + ((1-args.distil_lambda) * align_loss_scaled_valid)

            ## Get metrics for student
            ilisis, clisis, nmi, ari, diag_concentration_minimizer, foscttm_score, rank_score_, acc, acc_top5, clip_loss_, clip_loss_censored, \
            foscttm_score_ct, accuracy_ct, accuracy_top5_ct, clip_loss_ct, clip_loss_ct_split = \
                align_metrics(student_model, student_rna_latents_valid, student_rna_celltypes_valid, student_atac_latents_valid, student_atac_celltypes_valid, paired=paired, is_latents=True)
            

            ## Log validation losses for student
            epoch_align_losses_valid[target_dataset_og] = align_loss_scaled_valid.item()
            epoch_distil_losses_valid[target_dataset_og] = mean_distil_loss_valid.item()
            epoch_losses_valid[target_dataset_og] = total_loss_valid.item()

            epoch_ilisis[target_dataset_og], epoch_clisis[target_dataset_og], epoch_nmi[target_dataset_og], epoch_ari[target_dataset_og] = ilisis, clisis, nmi, ari
            epoch_rank_score[target_dataset_og] = rank_score_

            if paired:
                foscttm_score_ct.name = f'{epoch+1}'
                student_foscttm_per_ct_valid = pd.concat([student_foscttm_per_ct_valid, foscttm_score_ct], axis=1)
                epoch_foscttm[target_dataset_og] = foscttm_score.item()
            
                ## Get metrics for MDD data
                ilisis_mdd, clisis_mdd, nmi_mdd, ari_mdd, diag_concentration_minimizer_mdd, foscttm_score_mdd, rank_score_mdd, acc_mdd, acc_top5_mdd, _, _, _, _, _, _, _ = \
                    align_metrics(student_model, mdd_rna_latents_valid, mdd_rna_celltypes_valid, mdd_atac_latents_valid, mdd_atac_celltypes_valid, paired=False, is_latents=True)

                epoch_ilisis['MDD'], epoch_clisis['MDD'], epoch_nmi['MDD'], epoch_ari['MDD'] = ilisis_mdd, clisis_mdd, nmi_mdd, ari_mdd
                epoch_rank_score['MDD'] = rank_score_mdd


            if args.save_latents:
                save_latents(student_rna_latents_valid, student_atac_latents_valid, student_rna_celltypes_valid, student_atac_celltypes_valid, epoch, args.n_epochs, args.outdir)

            #student_model.train()
            if args.source_dataset_embedder:
                source_dataset_embedder.train()

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
            student_rna_latents, _ = student_model(student_rna_cells, modality='rna', task='align')
            student_rna_celltypes = student_rna_dat.obs['cell_type'].to_list()

            ## Project student ATAC data (target)
            student_atac_cells = student_atac_dat.X.float().to(device)
            student_atac_latents, _ = student_model(student_atac_cells, modality='atac', task='align')
            student_atac_celltypes = student_atac_dat.obs['cell_type'].to_list()

            ## Initialize lit of dataset distil losses
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
                target_rna_latents, _ = model(target_rna_cells, modality='rna', task='align')
                target_rna_celltypes = target_rna_dat.obs['cell_type'].to_list()

                # Project target ATAC data
                target_atac_cells = target_atac_dat.X.float().to(device)
                target_atac_latents, _ = model(target_atac_cells, modality='atac', task='align')
                target_atac_celltypes = target_atac_dat.obs['cell_type'].to_list()

                ## Ensure that the target latents are detached
                target_rna_latents = target_rna_latents.detach()
                target_atac_latents = target_atac_latents.detach()

                ## Obtain dataset embedding
                if args.source_dataset_embedder:
                    #dataset = np.random.choice(datasets)  # randomly select one of the datasets, but then would also have to select the target dataset accordingly
                    dataset_idx = dataset_idx_dict[dataset]
                    dataset_idx = torch.LongTensor([dataset_idx]).to(device)
                    dataset_embedding = source_dataset_embedder(dataset_idx)
                else:
                    dataset_embedding = None

                assert (student_rna_dat.obs_names == target_rna_dat.obs_names).all()
                assert (student_atac_dat.obs_names == target_atac_dat.obs_names).all()

                ## compute teacher losses
                distil_loss, distil_loss_T, align_loss_scaled, align_loss_T_scaled, offset, offset_T = knowledge_distillation_fn(student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, 'teacher')

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
            
            epoch_align_loss[target_dataset_og] += align_loss_scaled.item()
            epoch_distil_loss[target_dataset_og] += mean_distil_loss.item()
            epoch_losses[target_dataset_og] += total_loss.item()


        # Log the average loss per dataset
        row = {'Epoch': epoch + 1}
        for dataset in datasets+[target_dataset_og]:

            if dataset != 'MDD':
                row[f'Loss-{dataset}'] = epoch_losses[dataset] / num_batches
                row[f'Loss_align-{dataset}'] = epoch_align_loss[dataset] / num_batches
                row[f'Loss_distil-{dataset}'] = epoch_distil_loss[dataset] / num_batches

                row[f'Loss_valid-{dataset}'] = epoch_losses_valid[dataset]
                row[f'Loss_align_valid-{dataset}'] = epoch_align_losses_valid[dataset]
                row[f'Loss_distil_valid-{dataset}'] = epoch_distil_losses_valid[dataset]
            
            if (epoch == 0) or np.isin(dataset, [target_dataset_og]).item():
                row[f'iLISI-{dataset}'] = epoch_ilisis[dataset]
                row[f'cLISI-{dataset}'] = epoch_clisis[dataset]
                row[f'NMI-{dataset}'] = epoch_nmi[dataset]
                row[f'ARI-{dataset}'] = epoch_ari[dataset]

                if paired:
                    row[f'FOSCTTM-{dataset}'] = epoch_foscttm[dataset]
                    row[f'Rank_score-{dataset}'] = epoch_rank_score[dataset]

        if paired:
            row[f'iLISI-MDD'] = epoch_ilisis['MDD']
            row[f'cLISI-MDD'] = epoch_clisis['MDD']
            row[f'NMI-MDD'] = epoch_nmi['MDD']
            row[f'ARI-MDD'] = epoch_ari['MDD']
            row[f'Rank_score-MDD'] = epoch_rank_score['MDD']
            
        # Append the row to the DataFrame
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)


    # Define the CSV file path
    csv_file_path = os.path.join(args.outdir, 'training_log.csv')
    log_df.to_csv(csv_file_path, index=False)
    print(f"Training log saved to {csv_file_path}")

    ## Extract important metrics from last epoch for target dataset adn save separately
    core_metrics = ['iLISI', 'cLISI', 'NMI', 'ARI', 'FOSCTTM']
    core_metrics_target = [f'{metric}-{target_dataset_og}' for metric in core_metrics if log_df.columns.str.contains(metric).any()]
    last_epoch_log_df = \
        log_df.loc[:,log_df.columns.str.contains(target_dataset_og)].loc[:,
        core_metrics_target].iloc[-1]
    
    last_epoch_log_df = last_epoch_log_df.rename(index={f'{col}': f'{col.split("-")[0].lower()}' for col in last_epoch_log_df.index})
    last_epoch_log_df = last_epoch_log_df.rename(index={'ilisi': 'ilisis', 'clisi': 'clisis', 'foscttm': 'foscttm_score'})

    if len(datasets) > 1:
        last_epoch_log_df.to_csv(os.path.join(args.outdir, 'scMultiCLIP_metrics_target_valid.csv'))
    else:
        last_epoch_log_df.to_csv(os.path.join(args.outdir, 'kd_clip_metrics_target_valid.csv'))

    ## Save cell-type losses to CSV
    all_distil_losses_atac_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'all_distil_losses_atac_per_ct_valid.csv'))
    all_distil_losses_rna_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'all_distil_losses_rna_per_ct_valid.csv'))
    all_align_losses_atac_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'all_align_losses_atac_per_ct_valid.csv'))
    all_align_losses_rna_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'all_align_losses_rna_per_ct_valid.csv'))

    if paired:
        student_foscttm_per_ct_valid.columns = np.arange(1, args.n_epochs+1)
        student_foscttm_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'student_foscttm_per_ct_valid.csv'))
        teacher_foscttm_per_ct_valid.T.to_csv(os.path.join(args.outdir, 'teacher_foscttm_per_ct_valid.csv'))

    ## Save student model
    student_model_args_dict = {
        'n_peaks': n_peaks,
        'n_genes': n_genes,
        'args': args,
        'device': device,
        'nam_type': 'few-to-one',
        'genes_to_peaks_binary_mask': genes_to_peaks_binary_mask,
        'pretrain': False,
        'tuned_hyperparameters': student_model_args_dict['tuned_hyperparameters'],
        'rna_valid_idx': rna_valid_idx,
        'atac_valid_idx': atac_valid_idx,
    }

    student_model.eval()
    student_model_args_dict['model_state_dict'] = student_model.state_dict()
    torch.save(student_model_args_dict, os.path.join(args.outdir,'student_model.pt'))

    print('done!')

# %%
