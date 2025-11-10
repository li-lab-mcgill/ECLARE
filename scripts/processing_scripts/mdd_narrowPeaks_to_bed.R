#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(Matrix)
})

# ---- Paths ----
DATAPATH <- "/home/mcb/users/dmannk/scMultiCLIP/data/mdd_data"
rds_path <- file.path(DATAPATH, "ArchR_Subcluster_by_peaks_updated.rds")
# We'll write many files: prefix + group_label + ".narrowPeak"
out_prefix <- file.path(DATAPATH, "ArchR_Subcluster")

# ---- Load object ----
proj <- readRDS(rds_path)

meta <- proj@meta.data
required_cols <- c("sex", "ClustersMapped", "condition")
missing_cols <- setdiff(required_cols, colnames(meta))
if (length(missing_cols) > 0) {
  stop("Missing required metadata columns in proj@meta.data: ",
       paste(missing_cols, collapse = ", "))
}

# ---- Find ATAC / peaks assay ----
assays <- Assays(proj)
cat("Available assays:", paste(assays, collapse = ", "), "\n")

if ("peaks" %in% assays) {
  atac_assay <- "peaks"
} else if ("ATAC" %in% assays) {
  atac_assay <- "ATAC"
} else {
  atac_assay <- DefaultAssay(proj)
}

cat("Using assay:", atac_assay, "\n")

if (!inherits(proj[[atac_assay]], "ChromatinAssay")) {
  stop("Assay '", atac_assay, "' is not a ChromatinAssay. ",
       "Check which assay contains your ATAC peaks.")
}

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

# ---- 3) Define groups: all combinations of sex / ClustersMapped / condition ----
group_df <- meta[, required_cols, drop = FALSE]

group_df$group_label <- apply(group_df, 1, function(x) {
  paste0(
    "sex_", x["sex"],
    "_Cluster_", x["ClustersMapped"],
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
