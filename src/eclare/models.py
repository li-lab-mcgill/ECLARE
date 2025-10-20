import torch
import torch.nn as nn
from optuna.distributions import CategoricalDistribution, FloatDistribution
from coral_pytorch.layers import CoralLayer

import numpy as np
from copy import deepcopy
import glob

import os

from eclare.data_utils import PrintableLambda

class MCDropout(nn.Dropout):
    def forward(self, input):
        # Always apply dropout, regardless of model.training
        return nn.functional.dropout(input, self.p, training=True, inplace=self.inplace)

class CLIP(nn.Module):

    inv_softplus = lambda x, beta=1.0: 1/beta * np.log(np.exp(beta*x) - 1)

    HPARAMS = {
        'num_units': {
            'suggest_distribution': CategoricalDistribution(choices=[128, 256, 512]),
            'default': 64
        },
        '_teacher_num_layers': {
            'suggest_distribution': CategoricalDistribution(choices=[1, 2]),
            'default': 1
        },
        '_student_num_layers': {
            'suggest_distribution': CategoricalDistribution(choices=[1, 2]),
            'default': 3
        },
        'dropout_p': {
            'suggest_distribution': FloatDistribution(low=0.1, high=0.9),
            'default': 0.3
        },
        'teacher_temperature': {
            'suggest_distribution': FloatDistribution(low=inv_softplus(0.01), high=inv_softplus(5)),
            'default': inv_softplus(1.)
        },
        'student_temperature': {
            'suggest_distribution': FloatDistribution(low=inv_softplus(0.01), high=inv_softplus(5)),
            'default': inv_softplus(1.)
        },
        'weights_temperature': {
            'suggest_distribution': FloatDistribution(low=inv_softplus(0.01), high=inv_softplus(5)),
            'default': inv_softplus(0.005)
        },
        'distil_lambda': {
            'suggest_distribution': FloatDistribution(low=0.0, high=1.0),
            'default': 0.006
        },
        'decoder_loss': {
            'suggest_distribution': CategoricalDistribution(
                choices=[
                    None, 
                    PrintableLambda(lambda x, y: nn.MSELoss()(x, y), name='MSELoss'), 
                    PrintableLambda(lambda x, y: nn.PoissonNLLLoss(log_input=False, reduction='mean')(x, y), name='PoissonNLLLoss')
                ]),
            'default': None
        }
    }
    
    def __init__(self, n_peaks, n_genes, paired=True, **hparams):
        super().__init__()

        ## whether data for forward pass is paired
        self.paired = paired

        ## hyperparameters
        self.decoder_loss   = hparams['decoder_loss']
        num_units           = hparams['num_units']
        num_layers          = hparams['num_layers']
        dropout_p           = hparams['dropout_p']

        ## encoders
        rna_encoder     = [nn.Linear(n_genes, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + (num_layers-1) * [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)]
        atac_encoder    = [nn.Linear(n_peaks, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + (num_layers-1) * [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)]

        self.rna_to_core = nn.Sequential(*rna_encoder)
        self.atac_to_core = nn.Sequential(*atac_encoder)

        ## decoders
        if self.decoder_loss is not None:
            rna_decoder     = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_genes), nn.ReLU()]
            atac_decoder    = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_peaks), nn.ReLU()]
            
            self.core_to_rna = nn.Sequential(*rna_decoder)
            self.core_to_atac = nn.Sequential(*atac_decoder)

    @classmethod
    def get_hparams(cls, context='student', key=None):

        hparams = cls.HPARAMS[key].copy() if key else cls.HPARAMS.copy()

        context_hparam_key = f'_{context}_num_layers'
        if context_hparam_key in hparams.keys():
            hparams['num_layers'] = hparams[context_hparam_key].copy()
            del hparams[f'_student_num_layers'], hparams[f'_teacher_num_layers'] # don't need neither of these anymore
            
        return hparams

    def forward(self, x, modality: int, normalize: int = 1):

        '''
        modality: 0 for rna, 1 for atac. encode with int to enable model scriptability with torch.jit
        '''
    
        if modality == 0:

            latent = self.rna_to_core(x)

            if normalize==1:
                latent = torch.nn.functional.normalize(latent, p=2.0, dim=1)

            if hasattr(self, 'core_to_rna'):
                recon = self.core_to_rna(latent)
                return latent, recon
            else:
                return latent, None
        
        elif modality == 1:

            latent = self.atac_to_core(x)

            if normalize==1:
                latent = torch.nn.functional.normalize(latent, p=2.0, dim=1)

            if hasattr(self, 'core_to_atac'):
                recon = self.core_to_atac(latent)
                return latent, recon
            else:
                return latent, None
            
class ORDINAL(CLIP):
    def __init__(self, n_peaks, n_genes, ordinal_classes_df, shared_coral_layer=False, **hparams):
        super().__init__(n_peaks, n_genes, **hparams)
        num_units = hparams.get('num_units', 256)

        # Add ordinal_layer for both modalities
        self.ordinal_classes = ordinal_classes_df.index.tolist()
        num_classes = len(self.ordinal_classes)

        ''' get importance weights for ordinal loss
        eps = 1e-6
        rna_p = (ordinal_classes_df['rna_n_cells'].cumsum()[:-1] / ordinal_classes_df['rna_n_cells'].sum()).fillna(0).values
        atac_p = (ordinal_classes_df['atac_n_cells'].cumsum()[:-1] / ordinal_classes_df['atac_n_cells'].sum()).fillna(0).values

        rna_ordinal_weights = 1.0 / (rna_p * (1-rna_p) + eps)
        atac_ordinal_weights = 1.0 / (atac_p * (1-atac_p) + eps)
        atac_ordinal_weights[atac_p==0] = 0

        rna_ordinal_weights = rna_ordinal_weights / rna_ordinal_weights.sum()
        atac_ordinal_weights = atac_ordinal_weights / atac_ordinal_weights.sum()

        self.rna_ordinal_weights = torch.from_numpy(rna_ordinal_weights).float()
        self.atac_ordinal_weights = torch.from_numpy(atac_ordinal_weights).float()
        '''
        
        if shared_coral_layer:
            self.ordinal_layer = CoralLayer(size_in=num_units, num_classes=num_classes)
            self.ordinal_layer_rna = self.ordinal_layer
            self.ordinal_layer_atac = self.ordinal_layer
        else:
            self.ordinal_layer_rna = CoralLayer(size_in=num_units, num_classes=num_classes)
            self.ordinal_layer_atac = CoralLayer(size_in=num_units, num_classes=num_classes)

    def forward(self, x, modality: int, normalize: int = 0):
        '''
        modality: 0 for rna, 1 for atac. encode with int to enable model scriptability with torch.jit
        '''
        if modality == 0:
            latent = self.rna_to_core(x)
            if normalize == 1:
                latent = torch.nn.functional.normalize(latent, p=2.0, dim=1)
            logits = self.ordinal_layer_rna(latent)

        elif modality == 1:
            latent = self.atac_to_core(x)
            if normalize == 1:
                latent = torch.nn.functional.normalize(latent, p=2.0, dim=1)
            logits = self.ordinal_layer_atac(latent)

        else:
            # Handle invalid modality values for JIT compatibility
            # For JIT compatibility, we need to define logits in all branches
            # Since this is an error case, we'll use a dummy tensor that matches the expected shape
            logits = torch.zeros(x.shape[0], 6, device=x.device, dtype=x.dtype)
            raise ValueError(f"Invalid modality: {modality}. Must be 0 (RNA) or 1 (ATAC)")

        probas = torch.sigmoid(logits)

        return logits, probas, latent

    def forward_probas(self, x, modality: int, normalize: int = 0):

        if modality == 0:
            ordinal_coral_prebias = self.ordinal_layer_rna.coral_weights(x)

        elif modality == 1:
            ordinal_coral_prebias = self.ordinal_layer_atac.coral_weights(x)

        ordinal_pt = torch.sigmoid(ordinal_coral_prebias).flatten().detach().cpu().numpy()
        return ordinal_pt


# Expose the method separately as a standalone function
get_clip_hparams = CLIP.get_hparams

class SpatialCLIP(nn.Module):

    HPARAMS = {
        'num_units': {
            'suggest_distribution': CategoricalDistribution(choices=[128, 256, 512]),
            'default': 256
        },
        'num_layers': {
            'suggest_distribution': CategoricalDistribution(choices=[1, 2]),
            'default': 2
        },
        'dropout_p': {
            'suggest_distribution': FloatDistribution(low=0.1, high=0.9),
            'default': 0.2
        },
        'temperature': {
            'suggest_distribution': FloatDistribution(low=0.01, high=5),
            'default': 1
        },
        'decoder_loss': {
            'suggest_distribution': CategoricalDistribution(
                choices=[
                    None, 
                    PrintableLambda(lambda x, y: nn.MSELoss()(x, y), name='MSELoss'), 
                    PrintableLambda(lambda x, y: nn.PoissonNLLLoss(log_input=False, reduction='mean')(x, y), name='PoissonNLLLoss')
                ]),
            'default': None
        }
    }

    def __init__(self, n_genes, **hparams):
        super().__init__()

        self.n_genes = n_genes

        num_units           = hparams['num_units']
        num_layers          = hparams['num_layers']
        dropout_p           = hparams['dropout_p']
        decoder_loss        = hparams['decoder_loss']

        ## create encoder
        rna_encoder = [nn.Linear(n_genes, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] \
            + [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1)
        
        self.rna_to_core = nn.Sequential(*rna_encoder)

        ## create decoder, if decoder_loss is not None
        if decoder_loss is not None:
            self.core_to_rna = nn.Sequential(
                nn.Linear(num_units, n_genes),
                nn.ReLU()
            )

    @classmethod
    def get_hparams(cls, key=None):
        return cls.HPARAMS[key] if key else cls.HPARAMS
    
    def forward(self, x):
        ## encode
        latent = self.rna_to_core(x)

        ## decode, if decoder_loss is not None
        if hasattr(self, 'core_to_rna'):
            recon = self.core_to_rna(latent)
            return latent, recon
        else:
            return latent, None
    
# Expose the method separately as a standalone function
get_spatial_clip_hparams = SpatialCLIP.get_hparams

def load_CLIP_model(model_path, device, **kwargs):

    model_args_dict = torch.load(model_path, map_location=device)
    model_args_dict['device'] = device

    if 'tune_hyperparameters' in kwargs:
        model_args_dict['args'].tune_hyperparameters = kwargs['tune_hyperparameters']
    else:
        model_args_dict['args'].tune_hyperparameters = False

    model = CLIP(**model_args_dict).to(device=device)
    model.load_state_dict(model_args_dict['model_state_dict'])
    model.eval()
    
    return model, model_args_dict


def load_CLIP_and_ECLARE_model(student_model_path, best_multiclip_idx, device='cpu', genes_by_peaks_str='6816_by_55284'):
    student_model_args_dict = torch.load(student_model_path, map_location=device)

    clip_job_id = student_model_args_dict['args'].clip_job_id
    model_paths = glob(os.path.join(os.environ['OUTPATH'], f'clip_{clip_job_id}/PFC_Zhu/**/{best_multiclip_idx}/model.pt'))

    teacher_models = {}
    for model_path in model_paths:  
        teacher_model, teacher_clip_model_args_dict = load_CLIP_model(model_path, device=device)
        teacher_models[teacher_model.args.source_dataset] = teacher_model.eval()

    # student copies from last teacher model, makes no difference
    student_clip_model_args_dict = deepcopy(teacher_clip_model_args_dict)

    student_clip_model_args_dict['args'].source_dataset = 'PFC_Zhu'
    student_clip_model_args_dict['args'].target_dataset = 'mdd'

    student_clip_model_args_dict['args'].genes_by_peaks_str = genes_by_peaks_str
    student_clip_model_args_dict['n_genes'] = int(genes_by_peaks_str.split('_')[0])
    student_clip_model_args_dict['n_peaks'] = int(genes_by_peaks_str.split('_')[-1])
    student_clip_model_args_dict['tuned_hyperparameters']['params_num_layers'] = 2
    student_clip_model_args_dict['pretrain'] = student_clip_model_args_dict['rna_valid_idx']  = student_clip_model_args_dict['atac_valid_idx'] = None

    student_model = CLIP(**student_clip_model_args_dict, trial=None)
    student_model.load_state_dict(student_model_args_dict['model_state_dict'])
    student_model.eval()

    return teacher_models, student_model