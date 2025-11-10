library(ArchR)
library(GenomicRanges)

proj <- readRDS("/Users/dmannk/cisformer/data/mdd_data/ArchR_Subcluster_by_peaks_updated.rds")   # or loadArchRProject()

# Get peak set (usually from the ATAC / PeakMatrix)
peakGR <- getPeakSet(proj, useMatrix = "PeakMatrix")  # adjust matrix name if different

# Convert to a BED-like data.frame
df <- data.frame(
  chr   = as.character(GenomicRanges::seqnames(peakGR)),
  start = GenomicRanges::start(peakGR) - 1,  # BED is 0-based start
  end   = GenomicRanges::end(peakGR)
)

# Optional: add name and score columns if you have them
m <- as.data.frame(mcols(peakGR))

if ("name" %in% colnames(m)) {
  df$name <- m$name
} else {
  df$name <- paste0("peak_", seq_len(nrow(df)))
}

if ("score" %in% colnames(m)) {
  df$score <- m$score
} else if ("Log2FC" %in% colnames(m)) {
  # or any quantitative column you like
  df$score <- m$Log2FC
} else {
  df$score <- 1  # fallback: binary peaks
}

# Write as plain BED-like file (pyGenomeTracks will happily read this)
write.table(
  df,
  file = "archr_narrow_peaks.bed",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)
