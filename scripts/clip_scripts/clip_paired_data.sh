#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/clip_${JOB_ID}
TMPDIR=${OUTPATH}/clip_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/clip_scripts/clip_run.py ./scripts/clip_scripts/clip_paired_data.sh $TMPDIR
 
## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
  
## Set path to CSV file containing genes-by-peaks strings to match datasets
csv_file=${DATAPATH}/genes_by_peaks_full_str.csv
 
## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

source_datasets=("PFC_Zhu" "DLPFC_Ma" "Midbrain_Adams")
target_datasets=("DLPFC_Anderson")

## Define number of parallel tasks to run (replace with desired number of cores)
#N_CORES=6 # only relevant for multi-replicate tasks
N_REPLICATES=1

## Define random state
RANDOM=42
random_states=()
for i in $(seq 0 $((N_REPLICATES - 1))); do
    random_states+=($RANDOM)
done
 
## Define total number of epochs
total_epochs=100

## Create a temporary file to store all the commands we want to run
commands_file=$(mktemp)

# Function to check if a GPU is idle
is_gpu_idle() {
    local gpu_id=$1
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    
    # Define thresholds for idle state
    local utilization_threshold=10
    local memory_threshold=500 # in MB

    if [ "$utilization" -lt "$utilization_threshold" ] && [ "$memory_used" -lt "$memory_threshold" ]; then
        return 0 # GPU is idle
    else
        return 1 # GPU is not idle
    fi
}

# Get the number of GPUs
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Find idle GPUs
idle_gpus=()
for ((i=0; i<num_gpus; i++)); do
    if is_gpu_idle $i; then
        idle_gpus+=($i)
    fi
done

echo "Idle GPUs: ${idle_gpus[@]}"

# Function to run a task on a specific GPU
run_clip_task_on_gpu() {
    local gpu_id=$1
    local target_dataset=$2
    local source_dataset=$3
    local task_idx=$4
    local random_state=$5
    local genes_by_peaks_str=$6
    local feature=$7

    echo "Running '${feature}' on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/clip_scripts/clip_run.py \
    --outdir $TMPDIR/$target_dataset/$source_dataset/$task_idx \
    --target_dataset=$target_dataset \
    --source_dataset=$source_dataset \
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=800 \
    --feature="${feature}" \
    --metric_to_optimize="1-foscttm" &
    #--tune_hyperparameters \
    #--n_trials=3 &
}

# Function to extract genes_by_peaks_str from CSV file
extract_genes_by_peaks_str() {
    local csv_file=$1
    local source_dataset=$2
    local target_dataset=$3
    
    local genes_by_peaks_str=$(awk -F',' -v source="$source_dataset" -v target="$target_dataset" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == target) target_idx = i
            }
        }
        $1 == source {
            print $(target_idx)
        }
    ' "$csv_file")
    
    ## Check if the value was successfully extracted
    if [ -z "$genes_by_peaks_str" ]; then
        echo "Warning: No value found for source=$source_dataset, target=$target_dataset"

        ## Check for possible malformed header with extra bracket (e.g., ]mdd)
        header_check=$(awk -F',' 'NR==1 { for (i=1; i<=NF; i++) if ($i ~ /^\]/) print $i }' "$csv_file")
        if [ -n "$header_check" ]; then
            echo "Detected malformed header field(s): $header_check"
            echo "This may be caused by Windows-style carriage returns (\r)."
            echo "Suggested fix: Clean the CSV using the following command:"
            echo "    sed -i 's/\r\$//' \"$csv_file\""
            echo "Or preprocess with: dos2unix \"$csv_file\""
        fi
        
        return 1
    fi
    
    echo "$genes_by_peaks_str"
    return 0
}

## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('clip_${JOB_ID}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'CLIP_${JOB_ID}'
client.create_run(experiment_id, run_name=run_name)
"

## Train CLIP from source datasets

## Outer loop: iterate over datasets as the target_dataset
target_datasets_idx=0
for target_dataset in "${target_datasets[@]}"; do
    
    ## Middle loop: iterate over datasets as the source_dataset
    source_dataset_idx=0
    for source_dataset in "${source_datasets[@]}"; do
 
        # Skip the case where source and target datasets are the same
        if [ "$source_dataset" != "$target_dataset" ]; then

            ## Extract the value of `genes_by_peaks_str` for the current source and target
            genes_by_peaks_str=$(extract_genes_by_peaks_str $csv_file $source_dataset $target_dataset)

            ## Check if extraction was successful
            if [ $? -ne 0 ]; then
                continue
            fi

            ## Inner loop: iterate over task replicates (one per GPU)
            for task_idx in $(seq 0 $((N_REPLICATES-1))); do

                random_state=${random_states[$task_idx]}
                feature="${source_dataset}-to-${target_dataset}-${task_idx}"
 
                ## Make new sub-sub-directory for source dataset
                mkdir -p $TMPDIR/$target_dataset/$source_dataset/$task_idx
                
                # Assign task to an idle GPU, ensuring both source_dataset_idx and task_idx are used for load balancing
                gpu_id=${idle_gpus[$(((source_dataset_idx * N_REPLICATES + task_idx) % ${#idle_gpus[@]}))]}

                # Run CLIP task on idle GPU
                run_clip_task_on_gpu $gpu_id $target_dataset $source_dataset $task_idx $random_state $genes_by_peaks_str $feature
            done
                
        fi
        source_dataset_idx=$((source_dataset_idx + 1))
    done

    wait # Wait for all tasks for this loop to complete before moving to the next one

    target_datasets_idx=$((target_datasets_idx + 1))
done

cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR