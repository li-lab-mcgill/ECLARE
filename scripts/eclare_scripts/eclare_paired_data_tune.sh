#!/bin/bash -l
# Launch Optuna hyperparameter tuning with one worker per GPU (up to K),
# coordinating trials via a shared Optuna study/storage.

set -euo pipefail

### --- USER SETTINGS (override by exporting env vars before running) ---
K="${K:-4}"                         # Max concurrent GPU workers
GLOBAL_TRIALS="${GLOBAL_TRIALS:-24}"# Total COMPLETE trials across all workers
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}" # Training epochs for final run in each trial
BATCH_SIZE="${BATCH_SIZE:-800}"     # Batch size
FEATURE="${FEATURE:-}"              # Optional feature flag to pass to Python (leave empty to omit)
CLIP_JOB_ID="${CLIP_JOB_ID:-09114308}"

# Datasets (often a single target)
target_datasets=("DLPFC_Anderson")

# Sources to ignore (space-separated array)
ignore_sources=("Midbrain_Adams")

# Required environment variables (project paths)
: "${ECLARE_ROOT:?Set ECLARE_ROOT to your project root}"
: "${OUTPATH:?Set OUTPATH to your output directory base}"
: "${DATAPATH:?Set DATAPATH to your data directory base}"

csv_file="${DATAPATH}/genes_by_peaks_str.csv"

### --- Conda env + project root ---
# If conda isn't on PATH (non-interactive), source the init script.
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
fi

conda activate eclare_env
cd "$ECLARE_ROOT"

### --- Job staging directory (for reproducibility & outputs) ---
JOB_ID="$(date +%d%H%M%S)"   # very small chance of collision
TMPDIR="${OUTPATH}/eclare_${JOB_ID}"
mkdir -p "$TMPDIR"

mkdir -p "$TMPDIR/scripts_snapshot"
cp ./scripts/eclare_scripts/eclare_run.py ./scripts/eclare_scripts/eclare_paired_data_tune.sh "$TMPDIR/scripts_snapshot" 2>/dev/null || true

### --- Helper: detect idle GPUs (lightweight heuristic) ---
is_gpu_idle() {
  local gpu_id="$1"
  local utilization
  local memory_used
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id")
  memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id")

  local utilization_threshold=10   # %
  local memory_threshold=500       # MB

  if [ "$utilization" -lt "$utilization_threshold" ] && [ "$memory_used" -lt "$memory_threshold" ]; then
    return 0
  else
    return 1
  fi
}

### --- Helper: extract genes_by_peaks_str from CSV ---
extract_genes_by_peaks_str() {
  local csv="$1"
  local source_dataset="$2"
  local target_dataset="$3"

  local val
  val=$(awk -F',' -v source="$source_dataset" -v target="$target_dataset" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == target) target_idx = i
      }
    }
    $1 == source && target_idx { print $(target_idx) }
  ' "$csv")

  if [ -z "$val" ]; then
    echo "Warning: No genes_by_peaks_str for source='$source_dataset', target='$target_dataset' in $csv" >&2
    # Basic CRLF header check
    header_check=$(awk -F',' 'NR==1 { for (i=1; i<=NF; i++) if ($i ~ /^\]/) print $i }' "$csv")
    if [ -n "$header_check" ]; then
      echo "Detected malformed header field(s): $header_check" >&2
      echo "This may be caused by Windows-style carriage returns (\r). Consider:" >&2
      echo "  sed -i 's/\r\$//' \"$csv\"" >&2
      echo "or: dos2unix \"$csv\"" >&2
    fi
    return 1
  fi

  echo "$val"
  return 0
}

### --- GPU discovery ---
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found; GPUs required."; exit 1; }

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
idle_gpus=()
for ((i=0; i<num_gpus; i++)); do
  if is_gpu_idle "$i"; then idle_gpus+=("$i"); fi
done
echo "Idle GPUs detected: ${idle_gpus[*]:-none}"
if [ "${#idle_gpus[@]}" -eq 0 ]; then
  echo "No idle GPUs available. Exiting."
  exit 1
fi

### --- Threading hygiene for multi-GPU multi-process runs ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

### --- Per-target-dataset launch ---
for target_dataset in "${target_datasets[@]}"; do
  echo "Preparing target dataset: $target_dataset"

  # Get genes_by_peaks_str for (source=target_dataset, target=MDD), mirroring your previous script
  if ! genes_by_peaks_str=$(extract_genes_by_peaks_str "$csv_file" "$target_dataset" "MDD"); then
    echo "Skipping $target_dataset due to missing genes_by_peaks_str."
    continue
  fi
  echo "genes_by_peaks_str for $target_dataset -> MDD: $genes_by_peaks_str"

  # Optuna study/storage (one study shared by all workers for this target)
  OPTUNA_STORAGE="sqlite:///${ECLARE_ROOT}/optuna.db"  # local SQLite; switch to Postgres/MySQL if scaling up
  OPTUNA_STUDY="eclare_tuning_${target_dataset}_${CLIP_JOB_ID}"

  # Use up to K idle GPUs
  mapfile -t use_gpus < <(printf "%s\n" "${idle_gpus[@]}" | head -n "$K")
  if [ "${#use_gpus[@]}" -eq 0 ]; then
    echo "No GPUs selected for $target_dataset. Skipping."
    continue
  fi
  echo "Launching workers on GPUs: ${use_gpus[*]}"

  # Launch one worker per GPU
  pids=()
  for gid in "${use_gpus[@]}"; do
    outdir="${TMPDIR}/${target_dataset}/gpu_${gid}"
    mkdir -p "$outdir"

    # Build arg list (include --ignore_sources as multiple args; include --feature only if set)
    args=(
      --outdir "$outdir"
      --replicate_idx "$gid"
      --clip_job_id "$CLIP_JOB_ID"
      --experiment_job_id "$JOB_ID"
      --target_dataset "$target_dataset"
      --genes_by_peaks_str "$genes_by_peaks_str"
      --total_epochs "$TOTAL_EPOCHS"
      --batch_size "$BATCH_SIZE"
      --tune_hyperparameters
      --global_n_trials "$GLOBAL_TRIALS"
      --study_name "$OPTUNA_STUDY"
      --storage "$OPTUNA_STORAGE"
      --timeout 0
    )
    for s in "${ignore_sources[@]}"; do
      args+=(--ignore_sources "$s")
    done
    if [ -n "$FEATURE" ]; then
      args+=(--feature "$FEATURE")
    fi

    echo ">> GPU $gid → ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py ${args[*]}"
    CUDA_VISIBLE_DEVICES="$gid" \
      python "${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py" "${args[@]}" &
    pids+=("$!")
  done

  # Wait for this target's workers to finish
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Completed target dataset: $target_dataset"
done

echo "All workers finished. Outputs in: $TMPDIR"
