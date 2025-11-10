#!/usr/bin/env Rscript

library(Seurat)

# Load the object
DATAPATH <- "/home/mcb/users/dmannk/scMultiCLIP/data/mdd_data"
rds_path <- file.path(DATAPATH, "ArchR_Subcluster_by_peaks_updated.rds")
bed_path <- file.path(DATAPATH, "ArchR_Subcluster_peaks_narrowPeaks.bed")

proj <- readRDS(rds_path)

# Check available assays
print(Assays(proj))

# Pick the ATAC assay (adjust name if needed)
assay_name <- if ("peaks" %in% Assays(proj)) "peaks" else "ATAC"

# Extract the ranges
peak_ranges <- proj@assays[[assay_name]]@ranges

# Convert to BED data.frame
df <- data.frame(
  chr   = as.character(GenomicRanges::seqnames(peak_ranges)),
  start = GenomicRanges::start(peak_ranges) - 1L,  # BED = 0-based
  end   = GenomicRanges::end(peak_ranges)
)

# Optional: name & score
df$name  <- paste0("peak_", seq_len(nrow(df)))
df$score <- 1

# Write as BED
write.table(
  df,
  file = bed_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

cat("✅ Exported", nrow(df), "peaks to", bed_path, "\n")