from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
import os
import anndata
#from anndata.experimental import AnnLoader
from scipy.sparse import issparse
import h5py

from eclare.custom_annloader import CustomAnnLoader as AnnLoader

class SparseData(Dataset):
    def __init__(self, X, celltypes, batches=None):
        self.X = X
        self.y = celltypes
        self.batches = batches

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_coo = self.X[idx, :].tocoo()
        ind = torch.LongTensor(np.array([x_coo.row, x_coo.col]))  ## np.array() avoids warning about transforming list of arrays into tensor being very slow
        data = torch.FloatTensor(x_coo.data)
        out_data = torch.sparse.FloatTensor(ind, data, list(x_coo.shape))

        celltypes = self.y[idx]
        batches = self.batches[idx] if self.batches is not None else None

        return out_data, celltypes, batches
    
class DenseData(Dataset):
    def __init__(self, X, celltypes, batches=None):
        self.X = X
        self.y = celltypes
        self.batches = batches

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        data = torch.FloatTensor(self.X[idx, :])
        celltypes = self.y[idx]
        batches = self.batches[idx] if self.batches is not None else None
        return data, celltypes, batches

class StratifiedBatchSampler:
    """Stratified batch sampling -- https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        if n_batches > 1:
            self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        else:
            print('WARNING: only 1 batch, creates issues with StratifiedShuffleSplit \n')
            self.skf = StratifiedShuffleSplit(n_splits=n_batches)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()

        for train_idx, test_idx in self.skf.split(self.X, self.y):
            if self.skf.n_splits > 1:
                yield test_idx
            else:
                yield np.hstack([train_idx, test_idx])


    def __len__(self):
        return len(self.y)
    

def merge_major_cell_group(atac, rna, args):
            
    atac_cell_ontology = pd.read_csv(os.path.join(args.atac_datapath, args.cell_ontology_file), delimiter='\t')

    cell_ontology = atac_cell_ontology.drop(columns='Cell Ontology ID').rename(
        columns={'closest Cell Ontology term(s)':'cell ontology', 'Cell type':'cell type'})

    ## Import celltype annotation, associating major cell-types (aka 'Cluster ID') with sub-cell-type
    celltype_annotation = pd.read_excel(os.path.join(args.atac_datapath, args.CAtlas_celltype_annotation_file))
    celltype_annotation['Cluster ID'] = celltype_annotation['Cluster ID'].fillna(method='ffill')

    ## Rename some of annotations to ensure proper merging between cell ontology and major cell groups
    is_fetal = celltype_annotation['Life stage'].fillna(method='ffill') == 'Fetal tissue samples'
    celltype_annotation['Subcluster Annotation'].loc[(celltype_annotation['Subcluster Annotation'] == 'Lymphatic Endothelial Cell') & is_fetal] = 'Fetal Lymphatic Endothelial Cell'

    celltype_annotation['Subcluster Annotation'] = celltype_annotation['Subcluster Annotation'].astype('category').cat.rename_categories({
        'Naïve T cell': 'Naive T cell',
        'Pancreatic Delta / Gamma cell': 'Pancreatic Delta,Gamma cell',
        'Gastric Neuroendocrine cell': 'Gastric Neuroendocrine Cell',
        'Alverolar Type 2 / Immune': 'Alverolar Type 2,Immune',
        'Macrophage (General / Alveolar)': 'Macrophage (General,Alveolar)',
        'CNS / Enteric Neuron': 'CNS,Enteric Neuron',
        'Ductal Cell': 'Ductal Cell (Pancreatic)',
        'Colonic Goblet Cell ': 'Colonic Goblet Cell',
        'Fetal Alveolar Capillary Endothelial Cell': 'Fetal Alveolar Endothelial Cell',
        #'Lymphatic Endothelial Cell': 'Fetal Lymphatic Endothelial Cell', #duplicate adult and fetal "Lymphatic Endothelial Cell" annotation
        'Fetal Enterocyte 1 ': 'Fetal Enterocyte 1',
        'Fetal Schwann Cell (General)': 'Fetal Schwann Cell',
        'Fetal Oligodendrocyte Progenitor': 'Fetal Oligodendrocyte Progenitor 2', #both return false: (cell_ontology['cell type'] == 'Fetal Oligodendrocyte Progenitor').any() and (cell_ontology['cell type'] == 'Fetal Oligodendrocyte Progenitor 1').any()
        'Fetal Syncitiotrophoblast/Cytotrophoblast/Trophoblast Giant': 'Fetal Syncitiotrophoblast,Cytotrophoblast,Trophoblast Giant',
        'Fetal Cholangiocyte': 'Fetal Cholangiocytes',
        'Fetal Parietal/Chief Cell': 'Fetal Parietal,Chief Cell',
        'Fetal Ventricular Cardioyocyte': 'Fetal Ventricular Cardiomyocyte'
    })

    ## Merge cell ontology with sub-cell-type
    cell_ontology = cell_ontology.merge(celltype_annotation[['Cluster ID','Subcluster Annotation']], left_on='cell type', right_on='Subcluster Annotation', how='left')
    cell_ontology = cell_ontology.drop(columns='Subcluster Annotation')
    cell_ontology = cell_ontology.drop_duplicates()  # or else, len 223 rather than 222
    cell_ontology = cell_ontology.rename(columns={'Cluster ID':'major cell group'})

    ## Clean-up ATAC
    atac.obs['cell type'] = atac.obs['cell type'].cat.rename_categories({'Fetal Ventricular Cardioyocyte': 'Fetal Ventricular Cardiomyocyte'})
    atac.obs = atac.obs.reset_index().merge(cell_ontology, on='cell type', how='left').set_index('barcodes').copy()
    atac.obs['major cell group'] = atac.obs['major cell group'].str.lower() ## set to lowercase, so easier overlap with RNA

    ## merge RNA cell-types from the Cell Ontology with ATAC major cell groups
    rna.obs = rna.obs.reset_index().merge(cell_ontology[['cell ontology','major cell group']].drop_duplicates(subset='cell ontology'), left_on = 'cell_ontology_class', right_on = 'cell ontology', how='left').set_index('cell_id')
    rna.obs['major cell group'] = rna.obs['major cell group'].str.lower()

    ## set to lowercase, so easier to merge
    rna.obs['cell_ontology_class'] = rna.obs['cell_ontology_class'].str.lower()
    cell_ontology['cell ontology'] = cell_ontology['cell ontology'].str.lower()

    ## either drop RNA cells with nan annotations, or replace nan annotations by Cell Ontology labels
    handle_rna_nan = 'cell compartment'

    if handle_rna_nan == 'drop':
        rna = rna[rna.obs['major cell group'].notna()]
    elif handle_rna_nan == 'cell ontology':
        rna.obs['major cell group'] = rna.obs['major cell group'].fillna(rna.obs['cell_ontology_class'])  ## replace unsuccesfully merged major cell groups by cell onto
    elif handle_rna_nan == 'cell compartment':
        rna.obs['major cell group'] = rna.obs['major cell group'].fillna(rna.obs['compartment'])  ## replace unsuccesfully merged major cell groups by cell compartment

    atac_onto = atac.obs['major cell group'].unique()  ## need to change names of variables, so not so dependent on cell ontology annotation
    rna_onto  = rna.obs['major cell group'].unique()

    onto_overlap   = list( set(rna_onto)  & set(atac_onto)   )
    onto_union     = list( set(rna_onto)  | set(atac_onto)   )
    onto_rna_only  = list( set(rna_onto)  - set(onto_overlap))
    onto_atac_only = list( set(atac_onto) - set(onto_overlap))

    return atac, rna, onto_overlap


def create_loaders(
        data: anndata.AnnData,
        dataset: str,
        batch_size: int,
        total_epochs: int,
        cell_group_key: str='major cell group',
        batch_key: str='batch',
        valid_only: bool=False,
        standard: bool=False,
        stratified: bool=True):
    
    celltypes = data.obs[cell_group_key].values

    if batch_key in data.obs.columns:
        batches = data.obs[batch_key].values
        batches = pd.factorize(batches)[0]
    else:
        batches = None

    train_len = (int(0.8*len(data)))
    valid_len = int(0.2 * train_len)
    test_len = int(len(data) - (train_len + valid_len))

    ## reduce data size to keep CPU RAM memory below 62G
    if dataset == 'mdd':
        train_len = int(train_len * 0.5)
        valid_len = int(valid_len * 0.5)

    train_valid_random_state = int(os.environ.get('RANDOM_STATE', 42))
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=train_valid_random_state).split(X=np.empty(len(data)), y=celltypes))


    if standard:

        if dataset in ['pbmc_multiome', 'roussos', '388_human_brains_one_subject', '388_human_brains']:
            data = DenseData(data.X.toarray(), celltypes, batches)
        else:
            data = SparseData(data.X, celltypes, batches)

        if not valid_only:
        ## create train loader
            train_data = torch.utils.data.Subset(data, train_idx)
            train_num_batches = np.ceil(len(train_data) / batch_size)

            #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)
            train_sampler = StratifiedBatchSampler(celltypes[train_idx], batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(train_data, num_workers=0, batch_sampler=train_sampler, shuffle=False)

            train_num_batches = np.ceil(len(train_data) / batch_size)
            train_n_batches_str_length = len(str(int(train_num_batches)))
            train_n_epochs_str_length = len(str(int(total_epochs)))
        
        elif valid_only:
            train_data = train_num_batches = train_sampler = train_loader = train_n_batches_str_length = train_n_epochs_str_length = None

        ## create valid loader
        valid_data = torch.utils.data.Subset(data, valid_idx)
        valid_num_batches = np.ceil(len(valid_data) / batch_size)

        #valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
        if stratified:
            valid_sampler = StratifiedBatchSampler(celltypes[valid_idx], batch_size=batch_size, shuffle=False)
            valid_loader = DataLoader(valid_data, num_workers=0, batch_sampler=valid_sampler, shuffle=False)
        else:
            valid_loader = DataLoader(valid_data, num_workers=0, batch_sampler=None, batch_size=batch_size, shuffle=False)

    elif not standard:

        data.obs = data.obs.reset_index().reset_index().set_index('index')  # ensure to register indices in terms of integers, to be saved later
        data.obs = data.obs.rename(columns={cell_group_key:'cell_type'})

        valid_data = data[valid_idx].copy()

        if stratified:
            valid_sampler = StratifiedBatchSampler(celltypes[valid_idx], batch_size=batch_size, shuffle=False)
            valid_loader_indices = [(batch_indices,) for batch_indices in valid_sampler]
            valid_loader_indices = np.hstack(valid_loader_indices).squeeze()   # stack, so can sensibly use batch_size argument for AnnLoader

            #weights = 1 / data.obs['cell_type'].value_counts(normalize=True).sort_index().values
            #valid_sampler = WeightedRandomSampler(weights, len(valid_data), replacement=True)

            valid_loader = AnnLoader(valid_data, use_cuda=torch.cuda.is_available(), batch_size=batch_size, shuffle=False, indices=valid_loader_indices)
        else:
            valid_loader = AnnLoader(valid_data, use_cuda=torch.cuda.is_available(), batch_size=batch_size, shuffle=False)

        if not valid_only:
            train_data = data[train_idx].copy()
            train_sampler = StratifiedBatchSampler(celltypes[train_idx], batch_size=batch_size, shuffle=False)
            train_loader_indices = [(batch_indices,) for batch_indices in train_sampler]
            train_loader_indices = np.hstack(train_loader_indices).squeeze()   # stack, so can sensibly use batch_size argument for AnnLoader

            #weights = 1 / data.obs['cell_type'].value_counts(normalize=True).sort_index().values
            #train_sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)

            train_loader = AnnLoader(train_data, use_cuda=torch.cuda.is_available(), batch_size=batch_size, shuffle=False, indices=train_loader_indices)

            train_num_batches = np.ceil(len(train_data) / batch_size)
            train_n_batches_str_length = len(str(int(train_num_batches)))
            train_n_epochs_str_length = len(str(int(total_epochs)))

        elif valid_only:
            train_data = train_loader = train_num_batches = train_n_batches_str_length = train_n_epochs_str_length = None

    valid_num_batches = np.ceil(len(valid_data) / batch_size)
    valid_n_batches_str_length = len(str(int(valid_num_batches)))
    valid_n_epochs_str_length = len(str(int(total_epochs)))

    return train_loader, valid_loader, valid_idx, train_num_batches, valid_num_batches, train_n_batches_str_length, valid_n_batches_str_length, train_n_epochs_str_length, valid_n_epochs_str_length

def fetch_data_from_loader_light(loader, subsample=2000, label_key='cell_type'):

    n_cells = loader.dataset.shape[0]
    n_splits = np.ceil(n_cells / subsample).astype(int)

    ## if more cells than subsample, use stratified k-fold to sample
    if n_cells > subsample:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        _, cells_idx = next(skf.split(np.zeros_like(loader.dataset.obs[label_key].values), loader.dataset.obs[label_key].values))
    else:
        cells_idx = np.arange(n_cells)

    if issparse(loader.dataset.adatas[0].X):
        cells  = torch.tensor(loader.dataset.adatas[0].X[cells_idx].toarray() , dtype=torch.float32)
    else:
        cells  = torch.tensor(loader.dataset.adatas[0].X[cells_idx] , dtype=torch.float32)

    labels = loader.dataset.obs[label_key].values[cells_idx]
    return cells, labels

def fetch_data_from_loaders(rna_loader, atac_loader, paired=True, subsample=2000, rna_cells_idx=None, atac_cells_idx=None, label_key='cell_type'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hasattr(atac_loader.dataset, 'dataset') and hasattr(rna_loader.dataset, 'dataset'): # True if use "standard" loaders
        total_rna_cells = rna_loader.dataset.dataset.X.shape[0]
        total_atac_cells = atac_loader.dataset.dataset.X.shape[0]

        rna_celltypes = rna_loader.dataset.dataset.y
        atac_celltypes = atac_loader.dataset.dataset.y

    else:   # AnnLoaders
        total_rna_cells = rna_loader.dataset.shape[0]
        total_atac_cells = atac_loader.dataset.shape[0]

        rna_celltypes = rna_loader.dataset.obs[label_key].values
        atac_celltypes = atac_loader.dataset.obs[label_key].values


    if paired and (rna_cells_idx is None) and (atac_cells_idx is None):
        assert total_rna_cells == total_atac_cells, 'RNA and ATAC datasets have different number of cells'
        assert (rna_celltypes == atac_celltypes).all(), 'RNA and ATAC datasets have different celltypes'

        n_splits = np.ceil(total_rna_cells / subsample).astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        _, rna_cells_idx = next(skf.split(np.zeros_like(rna_celltypes), rna_celltypes))
        atac_cells_idx = rna_cells_idx.copy()

    elif (not paired) and (rna_cells_idx is None) and (atac_cells_idx is None):

        n_splits = np.ceil(total_rna_cells / subsample).astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        _, rna_cells_idx = next(skf.split(np.zeros_like(rna_celltypes), rna_celltypes))

        n_splits = np.ceil(total_atac_cells / subsample).astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        _, atac_cells_idx = next(skf.split(np.zeros_like(atac_celltypes), atac_celltypes))

        ## ensure same number of cells in both datasets
        delta_n = len(rna_cells_idx) - len(atac_cells_idx)
        if delta_n > 0:
            rna_cells_idx = rna_cells_idx[:-delta_n]
        elif delta_n < 0:
            atac_cells_idx = atac_cells_idx[:delta_n]


    if hasattr(atac_loader.dataset, 'dataset') and hasattr(rna_loader.dataset, 'dataset'):
        ## RNA
        if issparse(rna_loader.dataset.dataset.X):
            rna_cells  = torch.tensor(rna_loader.dataset.dataset.X[rna_cells_idx].toarray() , device=device , dtype=torch.float32)
        else:
            rna_cells  = torch.tensor(rna_loader.dataset.dataset.X[rna_cells_idx] , device=device , dtype=torch.float32)

        rna_celltypes = rna_celltypes[rna_cells_idx]

        ## ATAC
        if issparse(atac_loader.dataset.dataset.X):
            atac_cells  = torch.tensor(atac_loader.dataset.dataset.X[atac_cells_idx].toarray() , device=device , dtype=torch.float32)
        else:
            atac_cells  = torch.tensor(atac_loader.dataset.dataset.X[atac_cells_idx] , device=device , dtype=torch.float32)

        atac_celltypes = atac_celltypes[atac_cells_idx]

    
    else: # AnnLoader

        ## RNA
        if issparse(rna_loader.dataset.adatas[0].X):
            rna_cells  = torch.tensor(rna_loader.dataset.adatas[0].X[rna_cells_idx].toarray() , device=device , dtype=torch.float32)
        else:
            rna_cells  = torch.tensor(rna_loader.dataset.adatas[0].X[rna_cells_idx] , device=device , dtype=torch.float32)

        rna_celltypes = rna_celltypes[rna_cells_idx]

        ## ATAC
        if issparse(atac_loader.dataset.adatas[0].X):
            atac_cells  = torch.tensor(atac_loader.dataset.adatas[0].X[atac_cells_idx].toarray() , device=device , dtype=torch.float32)
        else:
            atac_cells  = torch.tensor(atac_loader.dataset.adatas[0].X[atac_cells_idx] , device=device , dtype=torch.float32)

        atac_celltypes = atac_celltypes[atac_cells_idx]


    return rna_cells, rna_celltypes, atac_cells, atac_celltypes, rna_cells_idx, atac_cells_idx



def keep_CREs_and_adult_only(atac, args, check_adult_only=True):
    ## Keep only CREs that are in the CRE_to_genes_matrix
    dummies = anndata.read_h5ad(os.path.join(args.atac_datapath, args.ABC_dummies_file )).to_df()
    CRE_to_genes_matrix = atac.var.index.to_frame().merge(dummies, left_index=True, right_index=True, how='inner').fillna(0).drop(columns='Feature_ID', inplace=False)
    keep_CRE = atac.var.index.isin(CRE_to_genes_matrix.index)
    atac = atac[:, keep_CRE ]

    ## Extract indices of non-zero scores for sparse linear layer
    CRE_to_genes_mapper = torch.stack(torch.where(torch.tensor(CRE_to_genes_matrix.values) > 0))
    CRE_to_genes_mapper = CRE_to_genes_mapper.long().flip(0)

    ## Keep only adult cells
    if check_adult_only and not args.not_adult_only:
        atac = atac[atac.obs['Life stage'] == 'Adult']

    return atac, CRE_to_genes_mapper


def get_memory_usage_h5ad(file_path):
    with h5py.File(file_path, "r") as f:
        def print_memory_usage(group, prefix=""):
            total_memory = 0
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    memory = item.size * item.dtype.itemsize
                    print(f"{prefix}/{key}: {memory / (1024**3):.2f} GB")
                    total_memory += memory
                elif isinstance(item, h5py.Group):
                    memory = print_memory_usage(item, prefix=f"{prefix}/{key}")
                    total_memory += memory
            return total_memory

        total_usage = print_memory_usage(f)
        print(f"Total Memory Usage: {total_usage / (1024**3):.2f} GB")


class PrintableLambda:
    def __init__(self, func, name="CustomLambda"):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<PrintableLambda: {self.name}>"