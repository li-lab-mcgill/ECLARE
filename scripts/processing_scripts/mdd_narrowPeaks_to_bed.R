#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(Matrix)
  library(zellkonverter)
  library(GenomicRanges)
  library(IRanges)
  library(SummarizedExperiment)
})

# ---- Paths ----
DATAPATH <- "/home/mcb/users/dmannk/scMultiCLIP/data/mdd_data"
h5ad_path <- file.path(DATAPATH, "mdd_atac_broad_sub_14814.h5ad")
#rds_path <- file.path(DATAPATH, "ArchR_Subcluster_by_peaks_updated.rds")
celltype_key <- "most_common_cluster"

# We'll write many files: prefix + group_label + ".narrowPeak"
out_prefix <- file.path(DATAPATH, "pseudobulked_narrowPeaks", "mdd_atac_broad_sub_14814")

# ---- Load object ----
# readH5AD returns a SingleCellExperiment, need to extract data and create Seurat object
sce <- readH5AD(h5ad_path)
cat("Loaded SingleCellExperiment with", ncol(sce), "cells and", nrow(sce), "features\n")

# Use the only assay present: "X"
count_mat <- SummarizedExperiment::assay(sce, "X")
cat("Using assay 'X' with", nrow(count_mat), "features and", ncol(count_mat), "cells\n")

# Extract peak names (should be in chr:start-end format)
peak_names <- rownames(sce)
cat("Parsing", length(peak_names), "peak coordinates from var_names...\n")

# Parse chr:start-end format
peak_parts <- strsplit(peak_names, "[:-]")
if (any(sapply(peak_parts, length) != 3)) {
  stop("Peak names are not in expected format 'chr:start-end'. ",
       "First few names: ", paste(head(peak_names, 3), collapse = ", "))
}

chrom <- sapply(peak_parts, function(x) x[1])
start <- as.integer(sapply(peak_parts, function(x) x[2]))
end <- as.integer(sapply(peak_parts, function(x) x[3]))

# Create GenomicRanges
peak_ranges_gr <- GRanges(
  seqnames = chrom,
  ranges = IRanges(start = start, end = end)
)
names(peak_ranges_gr) <- peak_names

cat("Created GenomicRanges with", length(peak_ranges_gr), "peaks\n")

# Extract metadata
meta <- as.data.frame(colData(sce))
rownames(meta) <- colnames(sce)

# Create ChromatinAssay
cat("Creating ChromatinAssay...\n")
chrom_assay <- CreateChromatinAssay(
  counts = count_mat,
  ranges = peak_ranges_gr
)

# Create Seurat object with ChromatinAssay
cat("Creating Seurat object...\n")
proj <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = meta
)

atac_assay <- "peaks"
cat("Created Seurat object with ChromatinAssay '", atac_assay, "'\n")

# ---- 1) Get peak ranges (shared across all groups) ----
peak_ranges <- granges(proj[[atac_assay]])

chrom  <- as.character(GenomicRanges::seqnames(peak_ranges))
start0 <- GenomicRanges::start(peak_ranges) - 1L  # 0-based
end1   <- GenomicRanges::end(peak_ranges)         # 1-based

name   <- paste0("peak_", seq_along(peak_ranges))
strand <- rep(".", length(peak_ranges))

peak_width <- end1 - (start0 + 1L)
summit_off <- floor(peak_width / 2)
summit_off[peak_width <= 0] <- 0L

# ---- 2) Count matrix (peaks x cells) ----
count_mat <- GetAssayData(proj, assay = atac_assay, slot = "counts")

# ---- 3) Define groups: all combinations of sex / celltype_key / condition ----
# Extract metadata from Seurat object (already added during object creation)
meta <- proj@meta.data

# Define required columns for grouping
required_cols <- c("sex", celltype_key, "condition")
# Check if columns exist (handle case variations)
if (!"sex" %in% colnames(meta)) {
  if ("Sex" %in% colnames(meta)) {
    meta$sex <- meta$Sex
  } else {
    stop("Neither 'sex' nor 'Sex' column found in metadata")
  }
}
if (!"condition" %in% colnames(meta)) {
  if ("Condition" %in% colnames(meta)) {
    meta$condition <- meta$Condition
  } else {
    stop("Neither 'condition' nor 'Condition' column found in metadata")
  }
}
if (!celltype_key %in% colnames(meta)) {
  stop(paste0(celltype_key, " column not found in metadata. Available columns: "), 
       paste(colnames(meta), collapse = ", "))
}

# Create group dataframe
group_df <- meta[, required_cols, drop = FALSE]
# Ensure rownames match cell names
rownames(group_df) <- rownames(meta)

group_df$group_label <- apply(group_df, 1, function(x) {
  paste0(
    "sex_", x["sex"],
    "_Cluster_", x[celltype_key],
    "_cond_", x["condition"]
  )
})
# sanitize to be file-system safe
group_df$group_label <- gsub("[^A-Za-z0-9._-]", "_", group_df$group_label)

group_levels <- unique(group_df$group_label)
cat("Found", length(group_levels), "groups\n")

# ---- 4) Loop over groups, pseudo-bulk + narrowPeak for each ----
for (g in group_levels) {
  cells_in_group <- rownames(group_df)[group_df$group_label == g]
  if (length(cells_in_group) == 0) {
    next
  }

  cat("Processing group:", g, "with", length(cells_in_group), "cells\n")

  cell_idx <- which(colnames(count_mat) %in% cells_in_group)
  if (length(cell_idx) == 0) {
    cat("  -> No matching cells in count matrix, skipping.\n")
    next
  }

  group_mat <- count_mat[, cell_idx, drop = FALSE]

  # --- PSEUDOBULK + CPM NORMALIZATION ---
  peak_counts <- Matrix::rowSums(group_mat)   # raw counts per peak in this group
  lib_size    <- sum(peak_counts)

  if (lib_size == 0) {
    # no counts at all; avoid division by zero
    cpm <- rep(0, length(peak_counts))
  } else {
    cpm <- 1e6 * peak_counts / lib_size      # counts per million
  }

  # signalValue (col 7): log10(CPM + tiny offset)
  signal_value <- log10(cpm + 1e-3)

  # --- FIXED SCALING FOR SCORE (no per-group min–max) ---
  # Map signalValue to [0, 1000] in a consistent, monotonic way
  # Here: score ≈ 100 * signalValue, clipped to [0, 1000]
  score_raw <- round(signal_value * 100)
  score_raw[score_raw < 0]    <- 0
  score_raw[score_raw > 1000] <- 1000
  score <- as.integer(score_raw)

  # pValue / qValue unknown -> -1 (MACS2 convention for "no value")
  pValue <- rep(-1, length(signal_value))
  qValue <- rep(-1, length(signal_value))

  narrowPeak_df <- data.frame(
    chrom       = chrom,
    chromStart  = start0,
    chromEnd    = end1,
    name        = name,
    score       = score,
    strand      = strand,
    signalValue = signal_value,
    pValue      = pValue,
    qValue      = qValue,
    peak        = summit_off,
    stringsAsFactors = FALSE
  )

  out_file <- paste0(out_prefix, ".", g, ".narrowPeak")

  write.table(
    narrowPeak_df,
    file      = out_file,
    sep       = "\t",
    quote     = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )

  cat("  ✅ Wrote", nrow(narrowPeak_df), "peaks to", out_file, "\n")
}

cat("Done.\n")
