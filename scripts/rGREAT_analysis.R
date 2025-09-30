library(rGREAT)
library(GSEABase)
library(GenomicRanges)

outdir      <- '/Users/dmannk/cisformer/outputs/enrichment_analyses_16103846_41/'
datapath    <- '/Users/dmannk/cisformer/data/'
gmt_dir     <- paste0(datapath, 'brain_gmt_cortical.gmt')
bed_dir     <- paste0(outdir, 'peak_bed_files')
out_rds     <- paste0(outdir, "all_sex_celltype_GREAT_results.rds")    # where to save the results

csv_dir     <- paste0(outdir, 'great_csv_outputs')
dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)

sexes = c("female", "male")
celltypes = c('Ast', 'End', 'ExN', 'InN', 'Mic', 'OPC', 'Oli')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare your custom gene‐sets (Entrez)
# ─────────────────────────────────────────────────────────────────────────────
# read in your GMT (gene symbols)
gmt     <- getGmt(gmt_dir)
symList <- geneIds(gmt)

# map SYMBOL → ENTREZID
entList <- lapply(symList, function(syms) {
    ids <- mapIds(org.Hs.eg.db,
                  keys      = unique(syms),
                  column    = "ENTREZID",
                  keytype   = "SYMBOL",
                  multiVals = "first")
    ids <- na.omit(unname(ids))
    as.character(ids)
})

# ─────────────────────────────────────────────────────────────────────────────
# 3. Loop through all .bed files, run GREAT, collect results
# ─────────────────────────────────────────────────────────────────────────────
for (sex in sexes) {
    for (ct in celltypes) {
        # construct path to your BED
        bed_file <- file.path(bed_dir, paste0(sex, "_", ct, "_peaks.bed"))
        if (!file.exists(bed_file)) next  # skip if missing

        # read & make GRanges
        peaks_df <- read.table(bed_file, header = FALSE,
                               col.names = c("chr","start","end"))
        gr <- GRanges(peaks_df$chr, IRanges(peaks_df$start, peaks_df$end))

        # run GREAT
        job      <- great(gr, entList, "hg38")
        res_list <- getEnrichmentTables(job)

        out_csv   <- file.path(csv_dir,
                               paste0(sex, "_", ct, ".csv"))

        write.csv(res_list,
                  file      = out_csv,
                  row.names = FALSE,
                  quote     = TRUE)
    }
}
