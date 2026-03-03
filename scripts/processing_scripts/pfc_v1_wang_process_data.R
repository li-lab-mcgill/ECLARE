suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(SingleCellExperiment)
  library(zellkonverter)
})

## =========================
## CONFIG
## =========================
input_rds <- "/home/mcb/users/dmannk/scMultiCLIP/data/PFC_V1_Wang/MERFISH_Seurat__object.rds"
output_dir <- "/home/mcb/users/dmannk/scMultiCLIP/data/PFC_V1_Wang/merfish"

assay_name <- NULL          # Uses DefaultAssay(obj) when NULL
sample_col <- NULL          # Auto-detects when NULL
coord_cols <- NULL          # Auto-detects when NULL; otherwise c("x_col", "y_col")
overwrite <- TRUE
write_manifest <- TRUE

stopf <- function(fmt, ...) {
  stop(sprintf(fmt, ...), call. = FALSE)
}

sanitize_sample_id <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x == ""] <- "sample_1"
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x[x == ""] <- "sample_1"
  x
}

match_existing_col <- function(colnames_vec, candidates) {
  low <- tolower(colnames_vec)
  for (cand in candidates) {
    idx <- which(low == tolower(cand))
    if (length(idx) > 0) {
      return(colnames_vec[idx[1]])
    }
  }
  NULL
}

find_best_coord_pair <- function(df, candidates) {
  if (nrow(df) == 0 || ncol(df) == 0) {
    return(NULL)
  }

  best_pair <- NULL
  best_score <- -1L

  for (pair in candidates) {
    if (length(pair) != 2) next
    if (!all(pair %in% colnames(df))) next

    x <- suppressWarnings(as.numeric(df[[pair[1]]]))
    y <- suppressWarnings(as.numeric(df[[pair[2]]]))
    score <- sum(is.finite(x) & is.finite(y))

    if (score > best_score) {
      best_score <- score
      best_pair <- pair
    }
  }

  if (is.null(best_pair) || best_score <= 0) {
    return(NULL)
  }

  best_pair
}

get_counts_matrix <- function(seurat_obj, assay) {
  mat <- tryCatch(
    SeuratObject::GetAssayData(seurat_obj, assay = assay, slot = "counts"),
    error = function(e) NULL
  )

  if (is.null(mat) || length(mat) == 0) {
    mat <- tryCatch(
      SeuratObject::GetAssayData(seurat_obj, assay = assay, layer = "counts"),
      error = function(e) NULL
    )
  }

  if (is.null(mat) || length(mat) == 0) {
    stopf("Raw counts were not found in assay '%s'.", assay)
  }

  if (nrow(mat) == 0 || ncol(mat) == 0) {
    stopf("Counts matrix in assay '%s' is empty.", assay)
  }

  if (!inherits(mat, "dgCMatrix")) {
    mat <- as(mat, "dgCMatrix")
  }

  mat
}

extract_image_coordinates <- function(seurat_obj) {
  image_names <- tryCatch(Images(seurat_obj), error = function(e) character(0))
  if (length(image_names) == 0) {
    return(NULL)
  }

  coord_candidates <- list(
    c("x", "y"),
    c("X", "Y"),
    c("imagecol", "imagerow"),
    c("col", "row"),
    c("xcoord", "ycoord"),
    c("center_x", "center_y"),
    c("x_centroid", "y_centroid"),
    c("coord_x", "coord_y")
  )

  coord_list <- list()

  for (img in image_names) {
    img_coords <- tryCatch(
      GetTissueCoordinates(seurat_obj, image = img),
      error = function(e) NULL
    )

    if (is.null(img_coords) || nrow(img_coords) == 0) next
    img_coords <- as.data.frame(img_coords)

    cell_ids_for_coords <- NULL
    if ("cell" %in% colnames(img_coords)) {
      cell_ids_for_coords <- as.character(img_coords$cell)
    } else if (!is.null(rownames(img_coords))) {
      cell_ids_for_coords <- rownames(img_coords)
    }
    if (is.null(cell_ids_for_coords)) next

    pair <- find_best_coord_pair(img_coords, coord_candidates)
    if (is.null(pair)) next

    x <- suppressWarnings(as.numeric(img_coords[[pair[1]]]))
    y <- suppressWarnings(as.numeric(img_coords[[pair[2]]]))

    coord_df <- data.frame(
      cell_id = cell_ids_for_coords,
      x = x,
      y = y,
      image_name = img,
      stringsAsFactors = FALSE
    )

    keep <- !is.na(coord_df$cell_id) & nzchar(coord_df$cell_id)
    coord_list[[img]] <- coord_df[keep, , drop = FALSE]
  }

  if (length(coord_list) == 0) {
    return(NULL)
  }

  coords <- do.call(rbind, coord_list)
  coords <- coords[!duplicated(coords$cell_id), , drop = FALSE]
  rownames(coords) <- coords$cell_id
  coords
}

resolve_metadata_coords <- function(md, coord_cols = NULL) {
  coord_candidates <- list(
    c("x", "y"),
    c("X", "Y"),
    c("xcoord", "ycoord"),
    c("center_x", "center_y"),
    c("x_centroid", "y_centroid"),
    c("coord_x", "coord_y"),
    c("imagecol", "imagerow"),
    c("col", "row")
  )

  if (!is.null(coord_cols)) {
    if (!is.character(coord_cols) || length(coord_cols) != 2) {
      stopf("coord_cols must be NULL or a character vector of length 2.")
    }
    if (!all(coord_cols %in% colnames(md))) {
      stopf(
        "Configured coord_cols (%s, %s) not found in metadata.",
        coord_cols[1], coord_cols[2]
      )
    }
    pair <- coord_cols
    source <- "metadata_config"
  } else {
    pair <- find_best_coord_pair(md, coord_candidates)
    if (is.null(pair)) {
      return(NULL)
    }
    source <- "metadata_auto"
  }

  x <- suppressWarnings(as.numeric(md[[pair[1]]]))
  y <- suppressWarnings(as.numeric(md[[pair[2]]]))

  out <- data.frame(
    cell_id = rownames(md),
    x = x,
    y = y,
    stringsAsFactors = FALSE
  )
  rownames(out) <- out$cell_id

  list(coords = out, source = source, x_col = pair[1], y_col = pair[2])
}

if (!file.exists(input_rds)) {
  stopf("Input RDS does not exist: %s", input_rds)
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

cat(sprintf("Loading MERFISH Seurat object: %s\n", input_rds))
obj <- readRDS(input_rds)

if (!inherits(obj, "Seurat")) {
  stopf("Input object must be a Seurat object. Found class: %s", paste(class(obj), collapse = ", "))
}

if (ncol(obj) == 0 || nrow(obj) == 0) {
  stopf("Seurat object is empty: %d cells, %d features.", ncol(obj), nrow(obj))
}

available_assays <- SeuratObject::Assays(obj)
if (is.null(assay_name)) {
  if ("Vizgen" %in% available_assays) {
    assay_name <- "Vizgen"
  } else {
    assay_name <- DefaultAssay(obj)
  }
}
if (!(assay_name %in% available_assays)) {
  stopf(
    "Assay '%s' not found. Available assays: %s",
    assay_name,
    paste(available_assays, collapse = ", ")
  )
}

cell_ids <- colnames(obj)
md <- obj@meta.data
if (!all(cell_ids %in% rownames(md))) {
  stopf("Cell IDs in Seurat object and metadata do not align.")
}
md <- md[cell_ids, , drop = FALSE]

cat(sprintf("Using assay: %s\n", assay_name))

image_coords <- extract_image_coordinates(obj)
metadata_coord_info <- NULL
coord_info <- NULL

if (is.null(coord_cols)) {
  metadata_coord_info <- tryCatch(
    resolve_metadata_coords(md, coord_cols = NULL),
    error = function(e) NULL
  )
} else {
  metadata_coord_info <- resolve_metadata_coords(md, coord_cols = coord_cols)
}

if (!is.null(image_coords)) {
  aligned <- data.frame(
    cell_id = cell_ids,
    x = NA_real_,
    y = NA_real_,
    image_name = NA_character_,
    stringsAsFactors = FALSE
  )
  rownames(aligned) <- cell_ids

  common_cells <- intersect(cell_ids, image_coords$cell_id)
  if (length(common_cells) > 0) {
    aligned[common_cells, c("x", "y", "image_name")] <- image_coords[common_cells, c("x", "y", "image_name")]

    source <- "images"
    x_col <- "x_from_image"
    y_col <- "y_from_image"

    missing <- !(is.finite(aligned$x) & is.finite(aligned$y))
    if (any(missing) && !is.null(metadata_coord_info)) {
      fill_cells <- rownames(aligned)[missing]
      aligned[fill_cells, c("x", "y")] <- metadata_coord_info$coords[fill_cells, c("x", "y"), drop = FALSE]
      source <- sprintf("images+%s", metadata_coord_info$source)
      x_col <- sprintf("x_from_image_or_%s", metadata_coord_info$x_col)
      y_col <- sprintf("y_from_image_or_%s", metadata_coord_info$y_col)
    }

    coord_info <- list(coords = aligned, source = source, x_col = x_col, y_col = y_col)
  }
}

if (is.null(coord_info)) {
  coord_info <- metadata_coord_info
}

if (is.null(coord_info)) {
  stopf("No spatial coordinates found from images or metadata.")
}

coords <- coord_info$coords[cell_ids, , drop = FALSE]
finite_coords <- is.finite(coords$x) & is.finite(coords$y)
if (!all(finite_coords)) {
  stopf(
    "Found missing/non-finite spatial coordinates for %d of %d cells.",
    sum(!finite_coords),
    length(finite_coords)
  )
}

sample_key <- NULL
sample_values <- NULL

if (!is.null(sample_col)) {
  if (!(sample_col %in% colnames(md))) {
    stopf("Configured sample_col '%s' is not in metadata.", sample_col)
  }
  sample_key <- sample_col
  sample_values <- as.character(md[[sample_col]])
} else {
  # Upstream MERFISH processing script uses Sample_ID and fov annotations.
  sample_candidates <- c(
    "Sample_ID", "sample_id", "sample", "orig.ident", "fov", "FOV",
    "section", "slice", "library_id", "replicate", "subject"
  )
  detected_sample_col <- match_existing_col(colnames(md), sample_candidates)
  if (!is.null(detected_sample_col)) {
    sample_key <- detected_sample_col
    sample_values <- as.character(md[[detected_sample_col]])
  } else if (coord_info$source == "images" && "image_name" %in% colnames(coords)) {
    sample_key <- "image_name"
    sample_values <- as.character(coords$image_name)
  } else {
    sample_key <- "constant"
    sample_values <- rep("sample_1", length(cell_ids))
  }
}

sample_values[is.na(sample_values) | !nzchar(sample_values)] <- "sample_1"
names(sample_values) <- cell_ids

counts <- get_counts_matrix(obj, assay_name)
if (!all(cell_ids %in% colnames(counts))) {
  stopf("Counts matrix columns do not align with Seurat cell IDs.")
}
counts <- counts[, cell_ids, drop = FALSE]

sample_levels <- sort(unique(sample_values))
cat(sprintf("Coordinate source: %s (%s, %s)\n", coord_info$source, coord_info$x_col, coord_info$y_col))
cat(sprintf("Sample key: %s\n", sample_key))
cat(sprintf("Exporting %d sample(s) to %s\n", length(sample_levels), output_dir))

write_args_supported <- names(formals(zellkonverter::writeH5AD))
manifest_rows <- vector("list", length(sample_levels))

for (i in seq_along(sample_levels)) {
  sample_id <- sample_levels[i]
  sample_cells <- names(sample_values)[sample_values == sample_id]

  sample_counts <- counts[, sample_cells, drop = FALSE]
  sample_md <- md[sample_cells, , drop = FALSE]
  sample_md$sample_id_export <- sample_id

  sample_coords <- coords[sample_cells, c("x", "y"), drop = FALSE]
  if (!all(is.finite(sample_coords$x) & is.finite(sample_coords$y))) {
    stopf("Non-finite coordinates detected in sample '%s'.", sample_id)
  }

  row_data <- S4Vectors::DataFrame(
    feature_name = rownames(sample_counts),
    row.names = rownames(sample_counts)
  )

  sce <- SingleCellExperiment(
    assays = list(counts = sample_counts),
    colData = S4Vectors::DataFrame(sample_md),
    rowData = row_data
  )

  reducedDim(sce, "spatial") <- as.matrix(sample_coords)

  safe_id <- sanitize_sample_id(sample_id)
  out_file <- file.path(output_dir, sprintf("merfish_%s.h5ad", safe_id))

  if (file.exists(out_file) && !isTRUE(overwrite)) {
    stopf("Output already exists and overwrite=FALSE: %s", out_file)
  }

  write_args <- list(sce = sce, file = out_file)
  if ("X_name" %in% write_args_supported) {
    write_args$X_name <- "counts"
  }

  do.call(zellkonverter::writeH5AD, write_args)

  manifest_rows[[i]] <- data.frame(
    sample_id = sample_id,
    sample_id_sanitized = safe_id,
    file = out_file,
    n_cells = ncol(sce),
    n_genes = nrow(sce),
    assay = assay_name,
    sample_key = sample_key,
    coord_source = coord_info$source,
    x_col = coord_info$x_col,
    y_col = coord_info$y_col,
    stringsAsFactors = FALSE
  )

  cat(sprintf("[%d/%d] Wrote %s (%d cells x %d genes)\n", i, length(sample_levels), out_file, ncol(sce), nrow(sce)))
}

manifest <- do.call(rbind, manifest_rows)

if (isTRUE(write_manifest)) {
  manifest_file <- file.path(output_dir, "merfish_manifest.csv")
  write.csv(manifest, file = manifest_file, row.names = FALSE)
  cat(sprintf("Wrote manifest: %s\n", manifest_file))
}

cat(sprintf(
  "Done. Exported %d samples, %d total cells, %d genes.\n",
  nrow(manifest),
  sum(manifest$n_cells),
  if (nrow(manifest) > 0) manifest$n_genes[1] else 0
))
