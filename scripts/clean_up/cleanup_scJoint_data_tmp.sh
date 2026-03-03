#!/bin/bash
# One-liner to delete all safe-to-delete files in scJoint_data_tmp
# Keeps only scJoint_latents.h5ad files

OUTPATH="${OUTPATH:-/home/mcb/users/dmannk/scMultiCLIP/outputs}"
SCJOINT_DIR="${OUTPATH}/scJoint_data_tmp"

# Delete all files in root directory
find "$SCJOINT_DIR" -maxdepth 1 -type f -delete

# Delete all intermediate files in subdirectories (*_embeddings.txt, *_predictions.txt, etc.) but keep scJoint_latents.h5ad
find "$SCJOINT_DIR" -mindepth 2 -type f ! -name "scJoint_latents.h5ad" -delete

# Remove empty directories
find "$SCJOINT_DIR" -type d -empty -delete

echo "Cleanup complete. Only scJoint_latents.h5ad files remain."
