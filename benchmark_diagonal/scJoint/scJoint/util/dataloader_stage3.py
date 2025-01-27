import glob
import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import random
import csv
import scipy.sparse
import h5py

from config import Config
from util.h5_reader import H5_Reader

from anndata import AnnData
from anndata.experimental import AnnLoader
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(1)

def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
    return np.loadtxt(label_file)


def npz_reader(file_name):
    print('load npz matrix:', file_name)
    data = scipy.sparse.load_npz(file_name)
    
    return data
    

def read_from_file(data_path, label_path = None, protien_path = None):
    data_path = os.path.join(os.path.realpath('.'), data_path)

    labels = None
    input_size, input_size_protein = 0, 0
    
    data_reader = npz_reader(data_path) 
    protein_reader = None
    if label_path is not None:        
        label_path = os.path.join(os.path.realpath('.'), label_path)    
        labels = load_labels(label_path)
    if protien_path is not None:
        protien_path = os.path.join(os.path.realpath('.'), protien_path)    
        protein_reader = npz_reader(protien_path)
        
    return data_reader, labels, protein_reader

def collate_fn_binarize_with_labels(batch):
    data = batch.X
    label = batch.obs['labels']
    data = (data>0).float()
    return data, label

def collate_fn_binarize(batch):
    data = batch.X
    data = (data>0).float()
    return data


class Dataloader(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None):
        self.train = train        
        self.data_reader, self.labels, self.protein_reader = data_reader, labels, protein_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        
        self.input_size_protein = None
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]

    def __getitem__(self, index):
        if self.train:
            # get atac data            
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype('float')  # binarize data
            
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            in_label = self.labels[rand_idx]
 
            return in_data, in_label

        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(np.float)  # binarize data

            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[index]
 
            return in_data, in_label

    def __len__(self):
        return self.sample_num
                
              
class DataloaderWithoutLabel(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None):
        self.train = train
        self.data_reader, self.labels, self.protein_reader = data_reader, labels, protein_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        
        self.input_size_protein = None
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]
            
            
    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype('float')  # binarize data
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype('float')  # binarize data
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

    def __len__(self):
        return self.sample_num


class PrepareDataloader():
    def __init__(self, config, idx_dict):
        self.config = config
        # hardware constraint
        # hardware constraint
        num_workers = self.config.threads - 1
        if num_workers < 0:
            num_workers = 0
        print('num_workers:', num_workers)
        kwargs = {'num_workers': num_workers, 'pin_memory': False} # 0: one thread, 1: two threads ...
        
        self.rna_train_idx = idx_dict['rna_train']
        self.rna_valid_idx = idx_dict['rna_valid']
        self.atac_train_idx = idx_dict['atac_train']
        self.atac_valid_idx = idx_dict['atac_valid']
        
        # load RNA
        train_rna_loaders = []
        test_rna_loaders = []

        for rna_path, label_path in zip(config.rna_paths, config.rna_labels):  
            data_reader, labels, _ = read_from_file(rna_path, label_path)

            rna_train_idx = self.rna_train_idx
            rna_valid_idx = self.rna_valid_idx

            # train loader 
            #trainset = Dataloader(True, data_reader, labels)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size= config.batch_size, shuffle=True, **kwargs)                        
            data_reader_ad_train = AnnData(data_reader[rna_train_idx], obs={'labels': labels[rna_train_idx]})
            trainloader = AnnLoader(data_reader_ad_train, use_cuda=torch.cuda.is_available(), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_binarize_with_labels)
            train_rna_loaders.append(trainloader)
            
            # test loader 
            #trainset = Dataloader(False, data_reader, labels)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=False, **kwargs)                        
            data_reader_ad_test = AnnData(data_reader[rna_valid_idx])
            testloader = AnnLoader(data_reader_ad_test, use_cuda=torch.cuda.is_available(), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_binarize)
            test_rna_loaders.append(testloader)
                
                
        # load ATAC
        train_atac_loaders = []
        test_atac_loaders = []
        self.num_of_atac = 0

        for i, atac_path in enumerate(config.atac_paths):  
            pseudo_label_path = os.environ['outdir'] + os.path.basename(config.atac_paths[i]).split('.')[0] + '_knn_predictions.txt'
            data_reader, labels, _ = read_from_file(atac_path, pseudo_label_path) # labels for trainset only from training stage 1

            # train loader
            #trainset = DataloaderWithoutLabel(True, data_reader)
            self.num_of_atac += data_reader.shape[0]

            atac_train_idx = self.atac_train_idx
            atac_valid_idx = self.atac_valid_idx
            
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)                        
            data_reader_ad_train = AnnData(data_reader[atac_train_idx], obs={'labels': labels})
            trainloader = AnnLoader(data_reader_ad_train, use_cuda=torch.cuda.is_available(), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_binarize_with_labels)
            train_atac_loaders.append(trainloader)
            
            # test loader
            #testset = DataloaderWithoutLabel(False, data_reader)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=False, **kwargs)                        
            data_reader_ad_test = AnnData(data_reader[atac_valid_idx])
            testloader = AnnLoader(data_reader_ad_test, use_cuda=torch.cuda.is_available(), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_binarize)
            test_atac_loaders.append(testloader)
                                           
  
        self.train_rna_loaders = train_rna_loaders
        self.test_rna_loaders = test_rna_loaders
        self.train_atac_loaders = train_atac_loaders
        self.test_atac_loaders = test_atac_loaders
                    
        
    def getloader(self):
        return self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, len(self.train_atac_loaders[0])


if __name__ == "__main__":
    config = Config()
    rna_data = Dataloader(True, config.rna_paths[0], config.rna_labels[0])
    #print 'rna data:', rna_data.input_size, rna_data.input_size_protein, len(rna_data.data)
    
    atac_data = DataloaderWithoutLabel(True, config.atac_paths[0])
    #print 'atac data:', atac_data.input_size, atac_data.input_size_protein, len(atac_data.data)
    
    
    train_rna_loaders, test_rna_loaders, train_atac_loaders, test_atac_loaders = PrepareDataloader(config).getloader()
    print(len(train_rna_loaders), len(test_atac_loaders))
    
    print(len(train_rna_loaders[1]), len(train_atac_loaders[0]))
