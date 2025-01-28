import torch
import torch.nn as nn

import numpy as np
#from sparselinear import SparseLinear

import sys
import os
sys.path.insert(0, os.environ['nampath'])

class scTripletgrate(nn.Module):
    def __init__(self, n_peaks, n_genes, args, device, trial=None, **kwargs):
        super().__init__()

        ## save args as attribute
        self.args = args      

        default_hyperparameters = {
            'params_layer_norm': False,
            'params_dropout_p': 0.3,
            'params_num_layers': 1,
            'params_num_units': 256,
            'params_sparse_linear_layer': False,
            'params_shared_weights_encoder_decoder': False,
            'params_skip_rna': True,
            'params_num_topics': 20,
            'params_embedding_dim': 200,
            'params_layer_norm': False
        }
        
        ## set parameters for ATAC-to-RNA projection, based on whether hyperparameters are tuned or not
        if not args.tune_hyperparameters:


            tuned_hyperparameters = kwargs['tuned_hyperparameters']

            self.layer_norm                 = tuned_hyperparameters.get('params_layer_norm', default_hyperparameters['params_layer_norm'])
            dropout_p                       = tuned_hyperparameters.get('params_dropout_p', default_hyperparameters['params_dropout_p'])
            num_layers                      = tuned_hyperparameters.get('params_num_layers', default_hyperparameters['params_num_layers'])
            num_units                       = int(tuned_hyperparameters.get('params_num_units', default_hyperparameters['params_num_units']))
            sparse_linear_layer             = tuned_hyperparameters.get('params_sparse_linear_layer', default_hyperparameters['params_sparse_linear_layer'])
            shared_weights_encoder_decoder  = tuned_hyperparameters.get('params_shared_weights_encoder_decoder', default_hyperparameters['params_shared_weights_encoder_decoder'])
            skip_rna                        = tuned_hyperparameters.get('params_skip_rna', default_hyperparameters['params_skip_rna'])
            num_topics                      = tuned_hyperparameters.get('params_num_topics', default_hyperparameters['params_num_topics'])
            embedding_dim                   = tuned_hyperparameters.get('params_embedding_dim', default_hyperparameters['params_embedding_dim'])

        elif args.tune_hyperparameters:

            dropout_p  = default_hyperparameters['params_dropout_p'] #trial.suggest_float("dropout_p", 0.1, 0.9)
            sparse_linear_layer = default_hyperparameters['params_sparse_linear_layer'] #trial.suggest_categorical("sparse_linear_layer", [True, False]) #no evidence that sparse linear layer improves performance (although drastic drop in n_params, from 46316869 to 53249)

            if kwargs['pretrain']:
                shared_weights_encoder_decoder = trial.suggest_categorical("shared_weights_encoder_decoder", [True, False])
                self.layer_norm = default_hyperparameters['params_layer_norm'] # trial.suggest_categorical("layer_norm", [True, False])

            num_layers = default_hyperparameters['params_num_layers'] #trial.suggest_int("num_layers", 0, 1)
            embedding_dim = default_hyperparameters['params_embedding_dim'] #trial.suggest_int("embedding_dim", 50, 500)

            if num_layers == 1:
                num_units = default_hyperparameters['params_num_units'] #trial.suggest_int("num_units", 50, 500)
                num_topics = default_hyperparameters['params_num_topics'] #trial.suggest_int("num_topics", 50, 500)
            elif num_layers == 0:
                num_units = n_genes

            skip_rna = default_hyperparameters['params_skip_rna'] #trial.suggest_categorical("skip_rna", [True, False])

            
        ## batch norm for RNA input data
        self.layer_norm_rna_genes  = nn.BatchNorm1d(n_genes, affine=True).to(device=device) #nn.LayerNorm([n_genes], elementwise_affine=False).to(device=device)
        self.layer_norm_rna_core   = nn.BatchNorm1d(num_units, affine=False).to(device=device) #nn.LayerNorm([num_units], elementwise_affine=False).to(device=device)

        self.layer_norm_atac_genes = nn.BatchNorm1d(n_genes, affine=True).to(device=device) #nn.LayerNorm([n_genes], elementwise_affine=False).to(device=device)
        self.layer_norm_atac_core  = nn.BatchNorm1d(num_units, affine=False).to(device=device) #nn.LayerNorm([num_units], elementwise_affine=False).to(device=device)

        genes_to_peaks_binary_mask = kwargs.get('genes_to_peaks_binary_mask', None)
        genes_to_peaks_binary_mask_array = torch.from_numpy(np.stack(genes_to_peaks_binary_mask.nonzero()))
        genes_to_peaks_binary_mask_array = genes_to_peaks_binary_mask_array.long()
        
        ## ATAC-to-RNA projection (and vice versa)
        if (sparse_linear_layer) and (not skip_rna):

            self.atac_to_rna = nn.Sequential(
                SparseLinear( n_peaks , n_genes , bias = False, connectivity = genes_to_peaks_binary_mask_array), nn.ReLU(), nn.Dropout(p=0)
            )
            
            if kwargs.get('nam_type', None)  == 'one-to-one':
                self.atac_to_rna.append(NeuralAdditiveModel(n_peaks, 2, (3,), forward_type='multi', nam_type='one-to-one', connectivity=genes_to_peaks_binary_mask_array))
                self.atac_to_rna.append(nn.ReLU())

            self.rna_to_atac = nn.Sequential(
                SparseLinear( n_genes , n_peaks , connectivity = torch.flip(genes_to_peaks_binary_mask_array, dims=(0,))), nn.ReLU(), nn.Dropout(p=0)
            )

        elif (kwargs.get('nam_type', None) == 'few-to-one') and (not sparse_linear_layer) and (not skip_rna):

            self.atac_to_rna = nn.Sequential(
                NeuralAdditiveModel(n_peaks, 2, (3,), output_size=n_genes, forward_type='multi', nam_type='few-to-one', connectivity=genes_to_peaks_binary_mask_array),
                nn.ReLU()
            )
            self.rna_to_atac = nn.Sequential(
                nn.Linear( n_genes , n_peaks ), nn.ReLU(), nn.Dropout(p=np.sqrt(dropout_p))
            )

        elif (not sparse_linear_layer) and (not skip_rna):
            self.atac_to_rna = nn.Sequential(
                nn.Linear( n_peaks , n_genes ), nn.ReLU(), nn.Dropout(p=np.sqrt(dropout_p)),
                #nn.Linear( 256 , n_genes ), nn.ReLU(), nn.Dropout(p=np.sqrt(dropout_p))
            )
            self.rna_to_atac = nn.Sequential(
                nn.Linear( n_genes , n_peaks ), nn.ReLU(), nn.Dropout(p=np.sqrt(dropout_p))
            )

        ## RNA-to-core and core-to-RNA
        #self.core1_to_core2 = nn.Linear(8*num_units, num_units)    
        
        ## RNA encoder and decoder
        rna_encoder = [nn.Linear(n_genes, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1)
        rna_decoder = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_genes), nn.ReLU(), nn.Dropout(p=dropout_p)]

        self.rna_to_core = nn.Sequential(*rna_encoder)
        self.core_to_rna = nn.Sequential(*rna_decoder)

        # ETM decoder
        '''
        self.rna_to_core = nn.Sequential(
            nn.Linear(n_genes, num_units), nn.ReLU(), nn.Dropout(p=dropout_p),
        )
        self.topic_embedding = nn.Linear(num_units, embedding_dim)
        self.gene_embedding = nn.Linear(embedding_dim, n_genes)
        self.core_to_rna = nn.Sequential(
            nn.Softmax(dim=1),
            self.topic_embedding,
            self.gene_embedding,
            nn.ReLU()
        )
        '''

        ## ATAC encoder and decoder - based on boolean variable 'skip_rna', create an encoder that skips the rna layer and projects directly to the core layer if skip_rna is True
        atac_encoder = [nn.Linear(n_peaks, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] + [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1)
        atac_decoder = [nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout_p)] * (num_layers-1) + [nn.Linear(num_units, n_peaks), nn.ReLU(), nn.Dropout(p=dropout_p)]

        if skip_rna and num_topics:
            self.atac_to_core = nn.Sequential(*atac_encoder)
            self.core_to_atac = nn.Sequential(*atac_decoder)
        elif not skip_rna:
            self.atac_to_core = nn.Sequential(
                self.atac_to_rna,
                self.rna_to_core
            )
            self.peak_embedding = nn.Linear(embedding_dim, n_peaks)
            self.core_to_atac = nn.Sequential(
                nn.Softmax(dim=1),
                self.topic_embedding,
                self.peak_embedding,
                nn.ReLU()
            )

        ## Share weights - not sure why don't need to transpose weights to ensure that dimensions match
        #if args.pretrain:
        if shared_weights_encoder_decoder:

            if isinstance(self.atac_to_core, nn.Sequential) and isinstance(self.core_to_atac, nn.Sequential):
                self.core_to_atac[0].weight.data = self.atac_to_core[3].weight.data.T
                self.core_to_atac[3].weight.data = self.atac_to_core[0].weight.data.T

                #self.core_to_atac[0].bias.data = self.atac_to_core[0].bias.data
                
            else:
                self.core_to_atac.weight.data = self.atac_to_core.weight.data.T

            if skip_rna:

                if isinstance(self.rna_to_core, nn.Sequential) and isinstance(self.core_to_rna, nn.Sequential):
                    self.core_to_rna[0].weight.data = self.rna_to_core[3].weight.data.T
                    self.core_to_rna[3].weight.data = self.rna_to_core[0].weight.data.T

                    #self.core_to_rna[3].bias = self.rna_to_core[0].bias

                else:
                    self.core_to_rna.weight.data = self.rna_to_core.weight.data.T
        
        ## create copy of encoder so can run forward on both align and pretrain tasks
        #self.atac_to_core_mlp_with_hidden = self.atac_to_core_mlp
        #self.atac_to_core_mlp_decoder_with_hidden = self.atac_to_core_mlp_decoder

        ## define classifier to predict batch label (entropy maximizer as loss for batch correction)
        '''
        if args.dataset == '388_human_brains':

            print('creating batch classifier')
            num_batches = kwargs['num_experimental_batches']

            self.batch_classifier = nn.Sequential(
                nn.Linear(num_units, 2*num_batches), nn.ReLU(), nn.Dropout(p=dropout_p),
                nn.Linear(2*num_batches, num_batches)
            )
        '''
        

        print('model checkpoint')


    def forward(self, x, modality='rna', task='align'):
    
        if modality == 'rna':
            #genes = self.layer_norm_rna_genes(x)
            genes = x
            core = self.rna_to_core(genes)

            if task == 'pretrain':
                if self.layer_norm:
                    core = self.layer_norm_rna_core(core)

                #core = core - core.mean(dim=0, keepdim=True)  # center core layer
                core = torch.nn.functional.normalize(core)  # normalize core layer

                x_recon = self.core_to_rna(core)
                return x_recon, core

            elif task == 'align':
                ## normalize RNA genes to ensure proper alignment with ATAC genes
                #core = self.layer_norm_rna_core(core)
                #core = torch.nn.functional.normalize(core)  # normalize core layer
                return core, genes
        
        elif modality == 'atac':

            if task == 'pretrain':

                core = self.atac_to_core(x)

                if self.layer_norm:
                    core = self.layer_norm_atac_core(core)

                #core = core - core.mean(dim=0, keepdim=True)  # center core layer
                core = torch.nn.functional.normalize(core)  # normalize core layer
            
                x_recon = self.core_to_atac(core)
                return x_recon, core
            
            elif task == 'align':
                ## normalize ATAC genes to ensure proper alignment with RNA genes
                #core = torch.nn.functional.normalize(core)  # normalize core layer
                if hasattr(self, "atac_to_rna"):
                    genes = self.atac_to_rna(x)
                    #genes = self.layer_norm_atac_genes(genes)
                    core = self.rna_to_core(genes)
                else:
                    genes = None
                    core = self.atac_to_core(x)

                return core, genes


def load_scTripletgrate_model(model_path, device, **kwargs):

    model_args_dict = torch.load(model_path, map_location=device)
    model_args_dict['device'] = device

    if 'tune_hyperparameters' in kwargs:
        model_args_dict['args'].tune_hyperparameters = kwargs['tune_hyperparameters']

    model = scTripletgrate(**model_args_dict).to(device=device)
    model.load_state_dict(model_args_dict['model_state_dict'])
    model.eval()
    
    return model, model_args_dict