# StdEnv/2023 gcc/12.3 r-bundle-bioconductor/3.18

library(Seurat)
library(SeuratDisk)

datapath <- "/home/dmannk/projects/def-liyue/dmannk/data/Roussos_lab"
atac_datapath <- file.path(datapath, 'atac', 'GSE204682%5Fcount%5Fmatrix.RDS')
rna_datapath <- file.path(datapath, 'rna', 'GSE204683%5Fcount%5Fmatrix.RDS')

atac <- readRDS(file = atac_datapath)
rna <- readRDS(file = rna_datapath)

## convert to h5
atac_h5_datapath <- file.path(atac_datapath, "roussos_atac.h5Seurat")
SaveH5Seurat(atac, overwrite = TRUE, filename = atac_h5_datapath)
Convert(atac_h5_datapath, overwrite = TRUE, dest = "h5ad")

rna_h5_datapath <- file.path(mdd_datapath, "mdd_rna.h5Seurat")
SaveH5Seurat(rna_obj, overwrite = TRUE, filename = rna_h5_datapath)
Convert(rna_h5_datapath, overwrite = TRUE, dest = "h5ad")
