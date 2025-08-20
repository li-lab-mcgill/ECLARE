library(Seurat)
library(SeuratDisk)

## define paths to data
datapath <- '/Users/dmannk/cisformer/data/PFC_V1_Wang/snMultiome_atlas_Seurat_object.rds'

## load data
obj <- readRDS(file = datapath)
