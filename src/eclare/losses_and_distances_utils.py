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
    
def spatial_clip_loss(logits, spatial_adj):

    ## row-wise loss
    logits_log_softmax = F.log_softmax(logits, dim=1)
    spatial_adj_log_softmax = F.log_softmax(spatial_adj, dim=1)
    loss = torch.nn.functional.kl_div(logits_log_softmax, spatial_adj_log_softmax.detach(), reduction='none', log_target=True)

    ## column-wise loss
    logits_log_softmax_T = F.log_softmax(logits.T, dim=1)
    spatial_adj_log_softmax_T = F.log_softmax(spatial_adj.T, dim=1)
    loss_T = torch.nn.functional.kl_div(logits_log_softmax_T, spatial_adj_log_softmax_T.detach(), reduction='none', log_target=True)

    ## zero-out diagonal entries
    loss[torch.eye(loss.shape[0], dtype=torch.bool)] = 0
    loss_T[torch.eye(loss_T.shape[0], dtype=torch.bool)] = 0

    ## apply batchmean reduction
    loss = loss.sum(1).mean()
    loss_T = loss_T.sum(1).mean()

    return loss, loss_T

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

    def __init__(self, device='cpu', paired=True, student_temperature=1, teacher_temperature=1, weigh_distil_by_align_type='none'):
        super(Knowledge_distillation_fn, self).__init__()
        self.device = device
        self.paired = paired
        self.align_loss_scale = 1 # to be updated later in script
        self.weigh_distil_by_align_type = weigh_distil_by_align_type

        ## lower temperature = lower entropy, and vice versa
        self.student_temperature = torch.tensor(student_temperature, requires_grad=False).to(device)
        self.teacher_temperature = torch.tensor(teacher_temperature, requires_grad=False).to(device)

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
    
    def ot_clip_loss_forward(self, teacher_logits, student_logits=None):

        # student loss
        if (student_logits is not None):

            ## obtain teacher weights - without temperature scaling, teacher weights very uniform
            ot_clip_loss_weights = (1 - torch.stack(self.all_teacher_ot_values)).softmax(dim=0).to(device=self.device)  # in principle, would also have values & weights for plan_T
            ot_clip_loss = torch.zeros(len(student_logits), device=self.device)
            ot_clip_loss_T = torch.zeros(len(student_logits), device=self.device)

            ## obtain weighted align loss per teacher
            for t, teacher_ot_plan in enumerate(self.all_teacher_ot_plans):

                labels = torch.argmax(teacher_ot_plan, dim=1)
                labels_T = torch.argmax(teacher_ot_plan.T, dim=1)

                ot_clip_loss = torch.nn.functional.cross_entropy(student_logits, labels, reduction='none')
                ot_clip_loss_T = torch.nn.functional.cross_entropy(student_logits.T, labels_T, reduction='none')

                ## apply teacher weight
                ot_clip_loss = ot_clip_loss + (ot_clip_loss_weights[t] * ot_clip_loss)
                ot_clip_loss_T = ot_clip_loss_T + (ot_clip_loss_weights[t] * ot_clip_loss_T)

            ## reset teacher plans and values, update in next forward pass
            self.all_teacher_ot_plans = []
            self.all_teacher_ot_values = []


        # teacher loss
        else:
            teacher_cost = 1 - (teacher_logits / torch.exp(1/self.teacher_temperature))

            ot_res = ot_solve(teacher_cost)
            plan = ot_res.plan
            plan_T = plan.T
            #plan_T = ot_solve(teacher_cost.T).plan  # empirically, plan_T != plan.T
            value = ot_res.value_linear

            self.all_teacher_ot_plans.append(plan)
            self.all_teacher_ot_values.append(value)

            labels = torch.argmax(plan, dim=1)
            labels_T = torch.argmax(plan_T, dim=1)  # plan_T != plan.T

            ot_clip_loss = torch.nn.functional.cross_entropy(teacher_logits, labels, reduction='none')
            ot_clip_loss_T = torch.nn.functional.cross_entropy(teacher_logits.T, labels_T, reduction='none')

        return ot_clip_loss, ot_clip_loss_T


    def forward(self, student_rna_latents, student_atac_latents, target_rna_latents, target_atac_latents, teacher_or_student, dataset_embedding=None):

        ## normalize latents
        target_rna_latents = torch.nn.functional.normalize(target_rna_latents, p=2, dim=1)
        target_atac_latents = torch.nn.functional.normalize(target_atac_latents, p=2, dim=1)
        student_rna_latents = torch.nn.functional.normalize(student_rna_latents, p=2, dim=1)
        student_atac_latents = torch.nn.functional.normalize(student_atac_latents, p=2, dim=1)

        ## get logits
        student_logits = torch.matmul(student_atac_latents, student_rna_latents.T) * torch.exp(1/self.student_temperature)
        target_logits = torch.matmul(target_atac_latents, target_rna_latents.T) * torch.exp(1/self.teacher_temperature)

        ## get distillation loss
        distil_loss, distil_loss_T = self.kl_forward(student_logits, target_logits)

        ## get alignment loss
        if teacher_or_student == 'student':
            align_loss, align_loss_T = self.ot_clip_loss_forward(None, student_logits)
            align_loss = 0.5 * (align_loss + align_loss_T).mean()
            offset = student_logits.exp().sum(1).log().mean()
            offset_T = student_logits.T.exp().sum(1).log().mean()
            align_loss_T_scaled = None

        elif teacher_or_student == 'teacher':
            align_loss, align_loss_T = self.ot_clip_loss_forward(target_logits, None)
            offset = target_logits.exp().sum(1).log()       # dim: batch size
            offset_T = target_logits.T.exp().sum(1).log()   # dim: batch size
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