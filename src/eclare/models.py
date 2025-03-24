import torch
import torch.nn as nn
from optuna.distributions import CategoricalDistribution, FloatDistribution

import numpy as np
from copy import deepcopy
import glob

import sys
import os

from eclare.data_utils import PrintableLambda

class CLIP(nn.Module):

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
    
    def __init__(self, n_peaks, n_genes, **hparams):
        super().__init__()
        
        self.temperature    = hparams['temperature']
        self.decoder_loss        = hparams['decoder_loss']
        num_units           = hparams['num_units']
        num_layers          = hparams['num_layers']
        dropout_p           = hparams['dropout_p']

        ## encoders
        rna_encoder = [nn.Linear(n_genes, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1)
        atac_encoder = [nn.Linear(n_peaks, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1)

        self.rna_to_core = nn.Sequential(*rna_encoder)
        self.atac_to_core = nn.Sequential(*atac_encoder)

        ## decoders
        if self.decoder_loss is not None:
            rna_decoder = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_genes), nn.ReLU(), nn.Dropout(p=dropout_p)]
            atac_decoder = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_peaks), nn.ReLU(), nn.Dropout(p=dropout_p)]
            
            self.core_to_rna = nn.Sequential(*rna_decoder)
            self.core_to_atac = nn.Sequential(*atac_decoder)

    @classmethod
    def get_hparams(cls, key=None):
        return cls.HPARAMS[key] if key else cls.HPARAMS

    def forward(self, x, modality: int):

        '''
        modality: 0 for rna, 1 for atac. encode with int to enable model scriptability with torch.jit
        '''
    
        if modality == 0:
            latent = self.rna_to_core(x)

            if hasattr(self, 'core_to_rna'):
                recon = self.core_to_rna(latent)
                return latent, recon
            else:
                return latent, None
        
        elif modality == 1:
            latent = self.atac_to_core(x)

            if hasattr(self, 'core_to_atac'):
                recon = self.core_to_atac(latent)
                return latent, recon
            else:
                return latent, None
            
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

        self.temperature    = hparams['temperature']
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