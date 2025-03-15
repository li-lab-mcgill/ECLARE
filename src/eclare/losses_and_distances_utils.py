import torch
import numpy as np
from pandas import DataFrame, factorize
import pandas as pd
import torch.nn.functional as F
from ot import solve as ot_solve

def cosine_distance(x, y, norm=True, detach=True):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)

    similarity = torch.matmul(x_norm, y_norm.transpose(0, 1))
    distance = (1 - similarity)

    if detach:
        distance = distance.detach().cpu().numpy()
    return distance

def euclidean_distance(x, y, detach=True):
    distance = torch.cdist(x, y, p=2)
    if detach:
        distance = distance.detach().cpu().numpy()
    return distance

def mean_similarity(distance, off_diag=False):
    similarity = 1 - distance
    if off_diag:
        similarity = similarity[~np.eye(similarity.shape[0],dtype=bool)]
    return similarity.mean()

def compute_mmd_loss(atac_genes, rna_genes, bandwidth_parameter):
    distance_euclidean = euclidean_distance(atac_genes, rna_genes, detach=False) # dimensions ATAC x RNA
    distance_atac_euclidean = euclidean_distance(atac_genes, atac_genes, detach=False)
    distance_rna_euclidean = euclidean_distance(rna_genes, rna_genes, detach=False)

    ## Get median distance from concatenation or stacking of all three types of distance
    all_distances = np.hstack([distance_euclidean.flatten().detach().cpu().numpy(), distance_atac_euclidean.flatten().detach().cpu().numpy(), distance_rna_euclidean.flatten().detach().cpu().numpy()])
    median_distance = np.median(all_distances).item()
    #min_distance = min(i for i in all_distances if i > 0)  # minimum value above 0, LONG computation
    #sigma = np.max([median_distance, min_distance]) # ensure non
    sigma = median_distance
    gamma = bandwidth_parameter / (2 * sigma**2)

    rbf = rbf_from_distance(distance_euclidean, gamma=gamma)
    rbf_atac = rbf_from_distance(distance_atac_euclidean, gamma=gamma, off_diag=True)
    rbf_rna = rbf_from_distance(distance_rna_euclidean, gamma=gamma, off_diag=True)

    mmd_loss = rbf_atac.mean() + rbf_rna.mean() - 2*rbf.mean()
    #mmd_loss = mean_similarity(distance_atac, off_diag=True) + mean_similarity(distance_rna, off_diag=True) - 2*mean_similarity(distance)  # replace RBF kernel with cosine dist with cosine similarity ("cosine similarity kernel"), 1-dist to get back similarity
    return mmd_loss

def rbf_from_distance(distances, gamma=1, off_diag=False):
    rbf = (-gamma * distances**2).exp()
    if off_diag:
        rbf = rbf[~np.eye(rbf.shape[0],dtype=bool)]
    return rbf

def clip_loss(logits, atac_celltypes=None, rna_celltypes=None, atac_latents=None, rna_latents=None, temperature=1, do_ct=False, censor_same_celltype=False, return_logits=False, reduce=True):
    
    if (atac_latents is not None) and (rna_latents is not None) and (logits is None):
        atac_latents = torch.nn.functional.normalize(atac_latents, p=2, dim=1)
        rna_latents = torch.nn.functional.normalize(rna_latents, p=2, dim=1)
        logits = torch.matmul(atac_latents, rna_latents.T) * np.exp(temperature)

    if return_logits:
        logits_distance = np.exp(temperature) - logits
        return logits, logits_distance

    if censor_same_celltype:
        atac_labels, atac_uniques = factorize(atac_celltypes, sort=True)
        rna_labels, rna_uniques = factorize(rna_celltypes, sort=True)
        assert (atac_uniques == rna_uniques).all()

        censor_mat = (atac_labels[:, None] == rna_labels[None, :])
        np.fill_diagonal(censor_mat, False) ## still need the matching pair to be included in logits computation
        logits[censor_mat] = -np.inf

    # Symmetric loss function
    n = logits.shape[0]
    labels = torch.arange(n).to(logits.device)

    loss_atac_full = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    loss_rna_full = torch.nn.functional.cross_entropy(logits.T, labels, reduction='none')

    if (not do_ct) and (not reduce):
        return loss_atac_full, loss_rna_full

    elif (not do_ct) and reduce:
        loss_atac = loss_atac_full.mean()
        loss_rna  = loss_rna_full.mean()
        return loss_atac, loss_rna

    ## cell type-level CLIP loss
    elif do_ct and reduce:
        loss_atac = loss_atac_full.mean()
        loss_rna  = loss_rna_full.mean()
        
        loss_atac_ct = DataFrame(loss_atac_full.detach().cpu().numpy(), index=atac_celltypes).groupby(level=0).mean()
        loss_rna_ct = DataFrame(loss_rna_full.detach().cpu().numpy(), index=rna_celltypes).groupby(level=0).mean()

        loss_atac_ct.loc[len(loss_atac_ct)] = loss_atac.item()
        loss_rna_ct.loc[len(loss_rna_ct)] = loss_rna.item()

        loss_atac_ct = loss_atac_ct.rename(index={loss_atac_ct.index[-1]: 'ALL'})
        loss_rna_ct = loss_rna_ct.rename(index={loss_rna_ct.index[-1]: 'ALL'})

        return loss_atac, loss_rna, loss_atac_ct, loss_rna_ct
    
def spatial_clip_loss(logits, spatial_adj, temperature=1):

    labels_rows = F.softmax(spatial_adj, dim=1)
    labels_cols = F.softmax(spatial_adj, dim=0)

    loss_rows = torch.nn.functional.cross_entropy(logits, labels_rows)
    loss_cols = torch.nn.functional.cross_entropy(logits.T, labels_cols)

    return loss_rows, loss_cols

def clip_loss_split_by_ct(atac_latents, rna_latents, atac_celltypes, rna_celltypes, temperature=1):
    
    all_celltypes = np.unique( list(atac_celltypes) + list(rna_celltypes) )
    loss_dict = {}

    for celltype in all_celltypes:
        atac_latents_ct = atac_latents[atac_celltypes == celltype]
        rna_latents_ct = rna_latents[rna_celltypes == celltype]

        rna_celltypes_ct = [celltype] * len(rna_latents_ct)
        atac_celltypes_ct = [celltype] * len(atac_latents_ct)

        loss_atac, loss_rna = clip_loss(None, atac_celltypes_ct, rna_celltypes_ct, atac_latents_ct, rna_latents_ct, temperature=temperature, do_ct=False)
        loss_dict[celltype] = ((loss_atac + loss_rna)/2).item()

    loss_df = DataFrame.from_dict(loss_dict, orient='index')
    return loss_df


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
