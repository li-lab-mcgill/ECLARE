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
from scipy.stats import linregress
import h5py
from warnings import warn
from pybedtools import BedTool
from glob import glob
from tqdm import tqdm

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
        cell_group_key: str,
        batch_key: str,
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
    if dataset == 'MDD':
        warn('Reducing MDD data size to keep CPU RAM memory below 62G')
        train_len = int(train_len * 0.5)
        valid_len = int(valid_len * 0.5)

    train_valid_random_state = int(os.environ.get('RANDOM_STATE', 42))
    train_idx, valid_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_len, test_size=valid_len, random_state=train_valid_random_state).split(X=np.empty(len(data)), y=celltypes))


    if standard:

        if dataset in ['pbmc_10x', 'PFC_Zhu', '388_human_brains_one_subject', '388_human_brains']:
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
        data.obs = data.obs.rename(columns={cell_group_key:'cell_type', batch_key:'batch'})

        valid_data = data[valid_idx].copy()

        if stratified:
            valid_sampler = StratifiedBatchSampler(celltypes[valid_idx], batch_size=batch_size, shuffle=False)
            valid_loader_indices = [(batch_indices,) for batch_indices in valid_sampler]
            valid_loader_indices = np.hstack(valid_loader_indices).squeeze()   # stack, so can sensibly use batch_size argument for AnnLoader

            valid_loader = AnnLoader(valid_data, use_cuda=torch.cuda.is_available(), batch_size=batch_size, shuffle=False, indices=valid_loader_indices)
        else:
            valid_loader = AnnLoader(valid_data, use_cuda=torch.cuda.is_available(), batch_size=batch_size, shuffle=False)

        if not valid_only:
            train_data = data[train_idx].copy()
            train_sampler = StratifiedBatchSampler(celltypes[train_idx], batch_size=batch_size, shuffle=False)
            train_loader_indices = [(batch_indices,) for batch_indices in train_sampler]
            train_loader_indices = np.hstack(train_loader_indices).squeeze()   # stack, so can sensibly use batch_size argument for AnnLoader

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

def fetch_data_from_loader_light(loader, subsample=2000, label_key='cell_type', batch_key='batch'):

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

    labels = loader.dataset.obs[label_key].values[cells_idx].to_list()
    batches = loader.dataset.obs[batch_key].values[cells_idx].to_list() if batch_key in loader.dataset.obs.columns else None

    return cells, labels, batches

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



def get_gene_peak_links(cre_gene_links_3d, cre_gene_links_crispr, cre_gene_links_eqtl):

    def load_links(filepath, psychscreen_cre_ensembl, chunksize=10000):
        filtered_rows = []

        for chunk in pd.read_csv(filepath, chunksize=chunksize, delimiter='\t', header=None):

            is_protein_coding = chunk.iloc[:,3] == "protein_coding"
            is_psychscreen = chunk.iloc[:,0].isin(psychscreen_cre_ensembl)

            filtered_chunk = chunk[is_protein_coding & is_psychscreen]
            filtered_rows.append(filtered_chunk)

        df = pd.concat(filtered_rows)
        return df

    psychscreen_bed_path = os.path.join(os.environ['datapath'], 'brainSCOPE', 'adult_bCREs.bed')
    cre_gene_links_3d_path = os.path.join(os.environ['datapath'], 'SCREEN', 'Human-Gene-Links', 'V4-hg38.Gene-Links.3D-Chromatin.txt')
    cre_gene_links_crispr_path = os.path.join(os.environ['datapath'], 'SCREEN', 'Human-Gene-Links', 'V4-hg38.Gene-Links.CRISPR.txt')
    cre_gene_links_eqtl_path = os.path.join(os.environ['datapath'], 'SCREEN', 'Human-Gene-Links', 'V4-hg38.Gene-Links.eQTLs.txt')

    psychscreen_bed = BedTool(psychscreen_bed_path) # PsychSCREEN
    psychscreen_bed_df = psychscreen_bed.to_dataframe()
    psychscreen_cre_ensembl = psychscreen_bed_df['score']
    #bed = BedTool('/Users/dmannk/Downloads/All.celltypes.Union.PeakCalls.bed') # BrainSCOPE

    cre_gene_links_3d = load_links('/Users/dmannk/Downloads/Human-Gene-Links/V4-hg38.Gene-Links.3D-Chromatin.txt', psychscreen_cre_ensembl) # 10119573 cCREs (tbnk)
    cre_gene_links_crispr = load_links('/Users/dmannk/Downloads/Human-Gene-Links/V4-hg38.Gene-Links.CRISPR.txt', psychscreen_cre_ensembl) # 1918 cCREs
    cre_gene_links_eqtl = load_links('/Users/dmannk/Downloads/Human-Gene-Links/V4-hg38.Gene-Links.eQTLs.txt', psychscreen_cre_ensembl) # 194344 cCREs

    ## Format as matrices
    cre_gene_links_3d_matrix = pd.crosstab(cre_gene_links_3d[0], cre_gene_links_3d[1])
    cre_gene_links_crispr_matrix = pd.crosstab(cre_gene_links_crispr[0], cre_gene_links_crispr[1])
    cre_gene_links_eqtl_matrix = pd.crosstab(cre_gene_links_eqtl[0], cre_gene_links_eqtl[1])

    ## example where multiple hits for same gene-peak pair
    cre_gene_links_crispr[(cre_gene_links_crispr[0]=="EH38E2911701") & (cre_gene_links_crispr[1]=="ENSG00000108179")]


def get_unified_grns(grn_path):

    grn_files = glob(os.path.join(grn_path,'*GRN.txt'))

    ## load all GRNs and concatenate
    grns = []
    for grn_file in grn_files:
        grn_df = pd.read_csv(grn_file, delimiter='\t')
        grns.append(grn_df)

    grn_df = pd.concat(grns)

    ## only keep GRNs inferred from scGRNom method
    grn_df_clean = grn_df[grn_df['method'].isin(['scGRNom', 'ATAC'])] # GRNs for which TF is matched to JASPAR

    ## clean GRNs
    grn_df_clean = grn_df_clean[~grn_df_clean['edgeWeight'].isna()] # remove rows with missing values, or else will corrupt mean
    mean_grn_df = grn_df_clean.groupby(['TF','enhancer','TG'])[['edgeWeight','Correlation']].mean() # take average across all cell types
    mean_grn_df.reset_index(inplace=True)

    return mean_grn_df

def filter_mean_grn(mean_grn_df, mdd_rna, mdd_atac, deg_genes=None):

    ## get peaks from GRNs
    peaks_df = pd.DataFrame(index=mdd_atac.var_names.str.split(':|-', expand=True)).reset_index()
    peaks_bedtool = BedTool.from_dataframe(peaks_df)

    unique_peaks = pd.Series(mean_grn_df['enhancer'].unique())
    grns_peaks_df = pd.DataFrame(unique_peaks.str.split(':|-', expand=True))
    grns_peaks_bedtool = BedTool.from_dataframe(grns_peaks_df)

    # Get indices of ATAC peaks that overlap with GRN peaks
    grn_peaks_in_data = peaks_bedtool.intersect(grns_peaks_bedtool, wa=True, wb=True)
    grn_peaks_in_data_df = grn_peaks_in_data.to_dataframe()

    # Get names of peaks to create mapper
    atac_peak_names = grn_peaks_in_data_df.iloc[:,:3].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
    grn_peak_names = grn_peaks_in_data_df.iloc[:,3:].astype(str).apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1)
    peaks_names_mapper = dict(zip(atac_peak_names, grn_peak_names)) # not necessarly one-to-one map, can have many-to-one

    if len(np.unique(list(peaks_names_mapper.keys()))) == len(peaks_names_mapper):
        print(f'all peaks from ATAC data are unique in mapper')
    if len(np.unique(list(peaks_names_mapper.values()))) == len(peaks_names_mapper):
        print(f'all peaks from GRN are unique in mapper')

    ## map ATAC peaks to create separate GRN peaks in ATAC data
    atac_peaks_mapped_to_grn = pd.Series(mdd_atac.var_names).map(peaks_names_mapper)
    mdd_atac.var['GRN_peak_interval'] = atac_peaks_mapped_to_grn.values

    ## keep only peaks that have a GRN peak interval
    peaks_indices = mdd_atac.var['GRN_peak_interval'].notna()
    mdd_atac = mdd_atac[:, peaks_indices]

    ## keep only GRNs that have a peak in the ATAC data
    mean_grn_df = mean_grn_df[mean_grn_df['enhancer'].isin(mdd_atac.var['GRN_peak_interval'])]

    ## remove GRNs for which TG or TF is not in the data
    mean_grn_df = mean_grn_df[(mean_grn_df['TG'].isin(mdd_rna.var_names)) & (mean_grn_df['TF'].isin(mdd_rna.var_names))]

    ## get genes from GRNs
    genes = mdd_rna.var_names
    is_target_gene = genes.isin(mean_grn_df['TG'])
    if deg_genes is not None:
        is_target_gene = is_target_gene & genes.isin(deg_genes)
    is_tf = genes.isin(mean_grn_df['TF'])
    is_both = is_target_gene & is_tf

    ## keep only genes that are target genes or transcription factors
    mdd_rna.var['is_target_gene'] = is_target_gene
    mdd_rna.var['is_tf'] = is_tf
    genes_indices = is_target_gene | is_tf
    mdd_rna = mdd_rna[:, genes_indices]

    ## add unified GRN diffusion scores
    #diffusion_scores_path = os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'Unified_GRN_diffusion.txt')
    #diffusion_scores = pd.read_csv(diffusion_scores_path, delimiter='\t')
    #unified_diffusions_scores = diffusion_scores[['TF','TG','Unified Score']] # also have cell-type specific scores
    #mean_grn_df = mean_grn_df.merge(unified_diffusions_scores, on=['TF', 'TG'], how='left')

    ## create mappers that can be used to track indices of genes and peaks in ATAC and RNA data
    data_gene_idx_mapper = dict(zip(mdd_rna.var['features'], np.arange(len(mdd_rna.var['features']))))
    data_peak_idx_mapper = dict(zip(mdd_atac.var['GRN_peak_interval'], np.arange(len(mdd_atac.var['GRN_peak_interval']))))

    ## apply mappers to track indices
    mean_grn_df['TF_idx_in_data'] = mean_grn_df['TF'].map(data_gene_idx_mapper)
    mean_grn_df['TG_idx_in_data'] = mean_grn_df['TG'].map(data_gene_idx_mapper)
    mean_grn_df['enhancer_idx_in_data'] = mean_grn_df['enhancer'].map(data_peak_idx_mapper)
    mean_grn_df.loc[:, 'enhancer_idx_in_data'] = mean_grn_df['enhancer_idx_in_data'].astype(int).values
    
    return mean_grn_df, mdd_rna, mdd_atac

def get_scompreg_loglikelihood_full(mean_grn_df, X_rna, X_atac, overlapping_target_genes, overlapping_tfs, column_name='log_gaussian_likelihood'):

    mean_grn_df_grouped = mean_grn_df.groupby('TG')

    tfrps = {}
    tg_expressions = {}

    for gene in tqdm(overlapping_target_genes):

        mean_grn_df_gene = mean_grn_df_grouped.get_group(gene)
        mean_grn_df_gene = mean_grn_df_gene[mean_grn_df_gene['TF'].isin(overlapping_tfs)]
        n_linked_peaks = mean_grn_df_gene['enhancer'].nunique()

        '''
        ## terms that encapsulate effects of both TF-peak (B) and peak-gene (I) interactions as per sc-compReg
        BI_diffusions = mean_grn_df_gene['Unified Score'] / mean_grn_df_gene['Unified Score'].sum() # global, for weighing TFRPs across TFs
        BI_enhancers = mean_grn_df_gene['edgeWeight']                                               # enhancer-specific
        BI_sign = np.sign(mean_grn_df_gene['Correlation'])                                          # sign of correlation between TF and TG
        BI = BI_enhancers * BI_diffusions * BI_sign
        '''

        ## sample peak accessibilities
        peak_idxs = mean_grn_df_gene['enhancer_idx_in_data'].astype(int).values
        peak_expressions = X_atac[:, peak_idxs]

        ## sample TF expressions
        tf_idxs = mean_grn_df_gene['TF_idx_in_data'].astype(int).values
        tf_expressions = X_rna[:, tf_idxs]

        ## sample target gene expression
        tg_idx = mean_grn_df_gene['TG_idx_in_data'].astype(int).unique().item() # should be unique
        tg_expression = X_rna[:, tg_idx]

        ## compute peak-tg correlations
        peak_tg_expressions = np.concatenate([peak_expressions.argsort(0), tg_expression.argsort(0)[:,None]], axis=1)
        peak_tg_correlations = np.corrcoef(peak_tg_expressions.T)[:-1, -1]
        peak_tg_correlations = peak_tg_correlations[None, :]

        dist_weight = mean_grn_df_gene['weight'].values[None, :]
        B = mean_grn_df_gene['motif_score_norm'].values[None, :]
        I = peak_tg_correlations * dist_weight
        BI = B * I
        BI = BI / BI.sum(axis=1, keepdims=True)

        #BI_sign = np.sign(mean_grn_df_gene['Correlation']).values[None, :]                                          # sign of correlation between TF and TG
        #BI = BI * BI_sign

        tfrp = tf_expressions * peak_expressions * BI

        tfrps[gene] = pd.DataFrame(tfrp.T, index=mean_grn_df_gene.index)
        tg_expressions[gene] = tg_expression

        ## plot tfrp vs tg_expression for each TF-peak pair
        #tfrp_tg_df = pd.DataFrame(np.hstack([tfrp, tg_expression[:,None]]), columns=[f'tfrp_{i}' for i in range(tfrp.shape[1])] + ['tg_expression'])
        #sns.pairplot(tfrp_tg_df, x_vars=[f'tfrp_{i}' for i in range(tfrp.shape[1])], y_vars=['tg_expression'])

        log_gaussian_likelihood = []
        for tfrp_ in tfrp.T:
            try:
                ## compute slope and intercept of linear regression - tg_expression is sparse (so is tfrp)
                linregress_res = linregress(tg_expression, tfrp_) # if unpack directly, returns only 5 of the 6 outputs..
            except:
                print(f'{gene} has no variance in tg_expression')
                tfrp_predictions = np.ones_like(tg_expression) * np.nan
                log_gaussian_likelihood_ = np.array([np.nan])
                slope = np.nan
                intercept = np.nan
                std_err = np.nan
                intercept_stderr = np.nan
            else:
                slope, intercept, r_value, p_value, std_err, intercept_stderr = (linregress_res.slope, linregress_res.intercept, linregress_res.rvalue, linregress_res.pvalue, linregress_res.stderr, linregress_res.intercept_stderr)
                tfrp_predictions = slope * tg_expression + intercept

                ## compute residuals and variance
                n = len(tfrp_)
                sq_residuals = (tfrp_ - tfrp_predictions)**2
                var = sq_residuals.sum() / n
                log_gaussian_likelihood_ = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()

            log_gaussian_likelihood.append(log_gaussian_likelihood_)

        log_gaussian_likelihood = np.array(log_gaussian_likelihood)
        mean_grn_df.loc[mean_grn_df_gene.index, column_name] = log_gaussian_likelihood

    return mean_grn_df, tfrps, tg_expressions

        


def get_scompreg_loglikelihood(mean_grn_df, X_rna, X_atac, overlapping_target_genes, overlapping_tfs):

    ## prototype on one gene
    #overlapping_target_genes = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_target_gene']].index.values
    #overlapping_tfs = mdd_rna_sampled_group.var[mdd_rna_sampled_group.var['is_tf']].index.values
    #X_rna = mdd_rna_sampled_group.X.toarray()
    #X_atac = mdd_atac_sampled_group.X.toarray()

    mean_grn_df_grouped = mean_grn_df.groupby('TG')

    def _scompreg_loglikelihood(gene, tfrp_aggregation='sum'):

        #mean_grn_df_gene = mean_grn_df[mean_grn_df['TG'] == gene].set_index('TG')
        mean_grn_df_gene = mean_grn_df_grouped.get_group(gene)
        mean_grn_df_gene = mean_grn_df_gene[mean_grn_df_gene['TF'].isin(overlapping_tfs)]
        n_linked_peaks = mean_grn_df_gene['enhancer'].nunique()

        '''
        ## terms that encapsulate effects of both TF-peak (B) and peak-gene (I) interactions as per sc-compReg
        BI_diffusions = mean_grn_df_gene['Unified Score'] / mean_grn_df_gene['Unified Score'].sum() # global, for weighing TFRPs across TFs
        BI_enhancers = mean_grn_df_gene['edgeWeight']                                               # enhancer-specific
        BI_sign = np.sign(mean_grn_df_gene['Correlation'])                                          # sign of correlation between TF and TG
        BI = BI_enhancers * BI_diffusions * BI_sign
        '''

        ## sample peak accessibilities
        peak_idxs = mean_grn_df_gene['enhancer_idx_in_data'].astype(int).values
        peak_expressions = X_atac[:, peak_idxs]

        ## sample TF expressions
        tf_idxs = mean_grn_df_gene['TF_idx_in_data'].astype(int).values
        tf_expressions = X_rna[:, tf_idxs]

        ## sample target gene expression
        tg_idx = mean_grn_df_gene['TG_idx_in_data'].astype(int).unique().item() # should be unique
        tg_expression = X_rna[:, tg_idx]

        ## compute peak-tg correlations
        peak_tg_expressions = np.concatenate([peak_expressions.argsort(0), tg_expression.argsort(0)[:,None]], axis=1)
        peak_tg_correlations = np.corrcoef(peak_tg_expressions.T)[:-1, -1]
        peak_tg_correlations = peak_tg_correlations[None, :]

        dist_weight = mean_grn_df_gene['weight'].values[None, :]
        B = mean_grn_df_gene['motif_score_norm'].values[None, :]
        I = peak_tg_correlations * dist_weight
        BI = B * I
        BI = BI / BI.sum(axis=1, keepdims=True)

        #BI_sign = np.sign(mean_grn_df_gene['Correlation']).values[None, :]                                          # sign of correlation between TF and TG
        #BI = BI * BI_sign

        tfrp = tf_expressions * peak_expressions * BI


        ## compute tfrp - note that peak_expressions are sparse, leading to sparse tfrp

        if tfrp_aggregation == 'sum':

            #BI = (BI_enhancers * BI_diffusions * BI_sign).values[None]

            #BI = (BI_enhancers * BI_sign).values[None]
            #tfrp = tf_expressions * peak_expressions * BI #* peak_tg_correlations * abc_scores

            tfrp = tfrp.sum(axis=1) # sum across TF-enhancer pairs to obtain single TFRP per gene

            try:
                ## compute slope and intercept of linear regression - tg_expression is sparse (so is tfrp)
                linregress_res = linregress(tg_expression, tfrp) # if unpack directly, returns only 5 of the 6 outputs..
            except:
                print(f'{gene} has no variance in tg_expression')
                tfrp_predictions = np.ones_like(tg_expression) * np.nan
                log_gaussian_likelihood = np.array([np.nan])
                slope = np.nan
                intercept = np.nan
                std_err = np.nan
                intercept_stderr = np.nan
            else:
                slope, intercept, r_value, p_value, std_err, intercept_stderr = (linregress_res.slope, linregress_res.intercept, linregress_res.rvalue, linregress_res.pvalue, linregress_res.stderr, linregress_res.intercept_stderr)
                tfrp_predictions = slope * tg_expression + intercept

                ## compute residuals and variance
                n = len(tfrp)
                sq_residuals = (tfrp - tfrp_predictions)**2
                var = sq_residuals.sum() / n
                log_gaussian_likelihood = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()

        elif tfrp_aggregation == 'factorized':

            BI = BI_enhancers
            tfrp = tf_expressions * peak_expressions * BI

            log_gaussian_likelihood = 0

            for tfrp_idx in range(tfrp.shape[1]):
                tfrp_ = tfrp[:, tfrp_idx]
                try:
                    ## compute slope and intercept of linear regression - tg_expression is sparse (so is tfrp)
                    linregress_res = linregress(tg_expression, tfrp_) # if unpack directly, returns only 5 of the 6 outputs..
                except:
                    print(f'{gene} has no variance in tg_expression')
                    tfrp_predictions = np.ones_like(tg_expression) * np.nan
                    log_gaussian_likelihood = np.array([np.nan])
                    slope = np.nan
                    intercept = np.nan
                    std_err = np.nan
                    intercept_stderr = np.nan
                else:
                    slope, intercept, r_value, p_value, std_err, intercept_stderr = (linregress_res.slope, linregress_res.intercept, linregress_res.rvalue, linregress_res.pvalue, linregress_res.stderr, linregress_res.intercept_stderr)
                    tfrp_predictions = slope * tg_expression + intercept

                    ## compute residuals and variance
                    n = len(tfrp)
                    sq_residuals = (tfrp - tfrp_predictions)**2
                    var = sq_residuals.sum() / n
                    log_gaussian_likelihood_ = -n/2 * np.log(2*np.pi*var) - 1/(2*var) * sq_residuals.sum()
                    log_gaussian_likelihood += log_gaussian_likelihood_

        elif tfrp_aggregation == 'kendall':
            for tfrp_idx in range(tfrp.shape[1]):
                tfrp_ = tfrp[:, tfrp_idx]
                kendall_tau, kendall_p = kendalltau(tg_expression, tfrp_)
                log_gaussian_likelihood += kendall_tau


        return log_gaussian_likelihood, tg_expression, tfrp, tfrp_predictions, slope, intercept, std_err, intercept_stderr

    log_gaussian_likelihoods    = {}
    slopes                      = {}
    intercepts                  = {}
    std_errs                    = {}
    intercept_stderrs           = {}
    tg_expressions              = pd.DataFrame(columns=overlapping_target_genes)
    tfrps                       = pd.DataFrame(columns=overlapping_target_genes)
    tfrp_predictions            = pd.DataFrame(columns=overlapping_target_genes)

    for gene in tqdm(overlapping_target_genes):

        log_gaussian_likelihood, tg_expression, tfrp, tfrp_prediction, slope, intercept, std_err, intercept_stderr = _scompreg_loglikelihood(gene)

        log_gaussian_likelihood = log_gaussian_likelihood.item()
        log_gaussian_likelihoods[gene] = log_gaussian_likelihood

        slopes[gene] = slope
        intercepts[gene] = intercept
        std_errs[gene] = std_err
        intercept_stderrs[gene] = intercept_stderr

        tg_expressions.loc[:,gene]      = tg_expression
        tfrps.loc[:,gene]               = tfrp
        tfrp_predictions.loc[:,gene]    = tfrp_prediction

    return log_gaussian_likelihoods, tg_expressions, tfrps, tfrp_predictions, slopes, intercepts, std_errs, intercept_stderrs


    #%load_ext line_profiler
    #%lprun -f process_gene process_gene(gene)