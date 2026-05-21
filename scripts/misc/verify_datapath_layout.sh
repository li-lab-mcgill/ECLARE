#!/usr/bin/env bash
# Verify dataset files are under $DATAPATH/<DATASET_NAME> (no atac/, rna/, snMultiome subdirs).
# Run with: DATAPATH=/your/data bash scripts/verify_datapath_layout.sh
set -e
DP="${DATAPATH:?Set DATAPATH}"

ok=0
fail=0

check_flat() {
  local dir="$1"
  local name="$2"
  if [[ ! -d "$dir" ]]; then
    echo "[SKIP] $name: $dir not found"
    return
  fi
  local bad=0
  [[ -d "$dir/atac" ]] && [[ -n "$(ls -A "$dir/atac" 2>/dev/null)" ]] && { echo "[FAIL] $name: $dir/atac/ still has content"; bad=1; }
  [[ -d "$dir/snMultiome" ]] && [[ -n "$(ls -A "$dir/snMultiome" 2>/dev/null)" ]] && { echo "[FAIL] $name: $dir/snMultiome/ still has content"; bad=1; }
  [[ -d "$dir/rna" ]] && [[ -n "$(ls -A "$dir/rna" 2>/dev/null)" ]] && { echo "[FAIL] $name: $dir/rna/ still has content"; bad=1; }
  if [[ $bad -eq 1 ]]; then
    ((++fail))
  else
    count=$(find "$dir" -maxdepth 1 -type f \( -name '*.h5ad' -o -name '*.h5' -o -name '*.npz' -o -name '*.csv' -o -name '*.tsv*' \) 2>/dev/null | wc -l)
    echo "[OK]   $name: $dir is flat (${count} data file(s) in root)"
    ((++ok))
  fi
}

echo "Checking layout under DP=$DP"
echo "---"

check_flat "$DP/PFC_Zhu"           "PFC_Zhu"
check_flat "$DP/DLPFC_Anderson"    "DLPFC_Anderson"
check_flat "$DP/PFC_V1_Wang"       "PFC_V1_Wang"
check_flat "$DP/Cortex_Velmeshev"  "Cortex_Velmeshev"

# Optional: ensure mdd_data exists and is flat (no atac/rna subdirs)
if [[ -d "$DP/mdd_data" ]]; then
  if [[ -d "$DP/mdd_data/atac" ]] || [[ -d "$DP/mdd_data/rna" ]]; then
    echo "[FAIL] mdd_data: still has atac/ or rna/ subdirs"
    ((++fail))
  else
    echo "[OK]   mdd_data: flat"
    ((++ok))
  fi
fi

echo "---"
echo "Result: $ok ok, $fail failure(s)"
[[ $fail -eq 0 ]]
