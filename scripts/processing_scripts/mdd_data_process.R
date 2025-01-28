library(Seurat)
library(SeuratDisk)

## define paths to data
mdd_datapath <- '/home/dmannk/projects/def-liyue/dmannk/data/mdd_data'

rna_datapath <- file.path(mdd_datapath, '1_enhanced_harmonized_object.Rds')
atac_broad_datapath <- file.path(mdd_datapath, 'broadpeaks_normalized_seuObj.rds')

## load data
rna_obj <- readRDS(file = rna_datapath)
atac_broad_obj <- readRDS(file = atac_broad_datapath)   # requires about 40 Gb
#rna_scale_data <- rna_obj@assays$RNA@scale.data

# Save the original data and scaled data in separate assays
rna_obj[['RNA']] <- CreateAssayObject(data = rna_obj@assays$RNA@data)

# Keep overlapping subjects only
''' PYTHON CODE
rna_subj_ids = rna.obs['OriginalSub'].tolist()
atac_subj_ids = atac.obs['BrainID'].tolist()
subj_ids = set(rna_subj_ids) & set(atac_subj_ids) # check for overlapping subject IDs

keep_rna = rna.obs['OriginalSub'].isin(subj_ids)
keep_atac = atac.obs['BrainID'].isin(subj_ids)

rna = rna[keep_rna].to_memory()
atac = atac[keep_atac].to_memory()
'''

## convert to h5
rna_h5_datapath <- file.path(mdd_datapath, "mdd_rna.h5Seurat")
SaveH5Seurat(rna_obj, overwrite = TRUE, filename = rna_h5_datapath)
Convert(rna_h5_datapath, overwrite = TRUE, dest = "h5ad")

atac_broad_h5_datapath <- file.path(mdd_datapath, "mdd_atac_broad.h5Seurat")
SaveH5Seurat(atac_broad_obj, overwrite = TRUE, filename = atac_broad_h5_datapath)
Convert(atac_broad_h5_datapath, overwrite = TRUE, dest = "h5ad")

# Function to strip a Seurat object of its numerical data
strip_seurat_object <- function(seurat_obj) {
  seurat_stripped <- seurat_obj
  
  # Remove large numerical slots: assays, graphs, neighbors, reductions, commands
  seurat_stripped@assays <- list()      # Remove assays (expression data)
  seurat_stripped@graphs <- list()      # Remove graphs
  seurat_stripped@neighbors <- list()   # Remove neighbors
  seurat_stripped@reductions <- list()  # Remove dimensionality reductions
  seurat_stripped@commands <- list()    # Remove command history
  
  return(seurat_stripped)
}

# Apply the function to the RNA object
rna_stripped <- strip_seurat_object(rna_obj)

# Apply the function to the ATAC object
atac_broad_stripped <- strip_seurat_object(atac_broad_obj)

# Save both stripped Seurat objects to RDS files
saveRDS(rna_stripped, file = file.path(mdd_datapath, '1_enhanced_harmonized_object_metadata.Rds'))
saveRDS(atac_broad_stripped, file = file.path(mdd_datapath, 'broadpeaks_normalized_seuObj_metadata.rds'))

# access metadata via object@meta.data$<field>