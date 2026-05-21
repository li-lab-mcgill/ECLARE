#!/usr/bin/env bash
# Move all dataset files to $DATAPATH/<DATASET_NAME> (no subdirs).
# Run from anywhere; set DATAPATH first, e.g.:
#   export DATAPATH=/path/to/your/data
set -e
DP="${DATAPATH:?Set DATAPATH}"

# --- MDD ---
# Uses $DATAPATH/mdd_data (no rename to MDD). Already flat — no move.

# --- PFC_Zhu ---
# Files under PFC_Zhu/atac/ and PFC_Zhu/rna/ → target PFC_Zhu/
if [[ -d "$DP/PFC_Zhu/atac" ]]; then
  mv -n "$DP/PFC_Zhu/atac"/* "$DP/PFC_Zhu/" 2>/dev/null || true
  rmdir "$DP/PFC_Zhu/atac" 2>/dev/null || true
fi
if [[ -d "$DP/PFC_Zhu/rna" ]]; then
  mv -n "$DP/PFC_Zhu/rna"/* "$DP/PFC_Zhu/" 2>/dev/null || true
  rmdir "$DP/PFC_Zhu/rna" 2>/dev/null || true
fi

# --- DLPFC_Anderson ---
# Currently under DLPFC_Anderson/snMultiome/ → target DLPFC_Anderson/
if [[ -d "$DP/DLPFC_Anderson/snMultiome" ]]; then
  mv -n "$DP/DLPFC_Anderson/snMultiome"/* "$DP/DLPFC_Anderson/" 2>/dev/null || true
  rmdir "$DP/DLPFC_Anderson/snMultiome" 2>/dev/null || true
fi

# --- Midbrain_Adams ---
# Already at $DATAPATH/Midbrain_Adams — no move.

# --- DLPFC_Ma ---
# Already at $DATAPATH/DLPFC_Ma — no move.

# --- spatialLIBD ---
# Already at $DATAPATH/spatialLIBD — no move.

# --- pbmc_10x ---
# Already at $DATAPATH/pbmc_10x — no move.

# --- mouse_brain_10x ---
# Already at $DATAPATH/mouse_brain_10x — no move.

# --- PFC_V1_Wang ---
# Files under PFC_V1_Wang/atac/ and PFC_V1_Wang/rna/ → target PFC_V1_Wang/
if [[ -d "$DP/PFC_V1_Wang/atac" ]]; then
  mv -n "$DP/PFC_V1_Wang/atac"/* "$DP/PFC_V1_Wang/" 2>/dev/null || true
  rmdir "$DP/PFC_V1_Wang/atac" 2>/dev/null || true
fi
if [[ -d "$DP/PFC_V1_Wang/rna" ]]; then
  mv -n "$DP/PFC_V1_Wang/rna"/* "$DP/PFC_V1_Wang/" 2>/dev/null || true
  rmdir "$DP/PFC_V1_Wang/rna" 2>/dev/null || true
fi

# --- Cortex_Velmeshev ---
# Files under Cortex_Velmeshev/atac/ and Cortex_Velmeshev/rna/ → target Cortex_Velmeshev/
if [[ -d "$DP/Cortex_Velmeshev/atac" ]]; then
  mv -n "$DP/Cortex_Velmeshev/atac"/* "$DP/Cortex_Velmeshev/" 2>/dev/null || true
  rmdir "$DP/Cortex_Velmeshev/atac" 2>/dev/null || true
fi
if [[ -d "$DP/Cortex_Velmeshev/rna" ]]; then
  mv -n "$DP/Cortex_Velmeshev/rna"/* "$DP/Cortex_Velmeshev/" 2>/dev/null || true
  rmdir "$DP/Cortex_Velmeshev/rna" 2>/dev/null || true
fi

echo "Done. All dataset files are under \$DATAPATH/<DATASET_NAME>."
