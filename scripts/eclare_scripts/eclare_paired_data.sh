#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/eclare_${JOB_ID}
TMPDIR=${OUTPATH}/eclare_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/eclare_scripts/eclare_run.py ./scripts/eclare_scripts/eclare_paired_data.sh $TMPDIR
 
## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
  
## Set path to CSV file containing genes-by-peaks strings to match datasets
csv_file=${DATAPATH}/genes_by_peaks_str.csv
 
## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

## Reverse the order of datasets to have pbmc_multiome and mouse_brain_multiome first
datasets=($(for i in $(seq $((${#datasets[@]} - 1)) -1 0); do echo "${datasets[$i]}"; done))


## Define number of parallel tasks to run (replace with desired number of cores)
N_CORES=6
N_REPLICATES=1

## Define random state
RANDOM=42
random_states=()
for i in $(seq 0 $((N_CORES - 1))); do
    random_states+=($RANDOM)
done
 
## Define total number of epochs
clip_job_id='02202535'
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
run_eclare_task_on_gpu() {
    local clip_job_id=$1
    local gpu_id=$2
    local target_dataset=$3
    local task_idx=$4
    local random_state=$5
    local genes_by_peaks_str=$6
    local feature=$7

    echo "Running ${source_dataset} to ${target_dataset} (task $task_idx) on GPU $gpu_id"

    # for paired data, do not specify source_dataset to ensure that all datasets serve as sources
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py \
    --outdir $TMPDIR/$target_dataset/$source_dataset/$task_idx \
    --replicate_idx=$task_idx \
    --clip_job_id=$clip_job_id \
    --target_dataset=$target_dataset \
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=500 \
    --feature="'$feature'" \
    --distil_lambda=0.1 &
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
experiment = get_or_create_experiment('clip_${clip_job_id}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'ECLARE_${clip_job_id}'
client.create_run(experiment_id, run_name=run_name)
"

## Outer loop: iterate over datasets as the target_dataset
target_datasets_idx=0
for target_dataset in "${datasets[@]}"; do

    ## Extract the value of `genes_by_peaks_str` for the current source and target
    genes_by_peaks_str=$(extract_genes_by_peaks_str "$csv_file" "$target_dataset" "PFC_Zhu")
    
    ## Check if extraction was successful
    if [ $? -ne 0 ]; then
        continue
    fi

    for task_idx in $(seq 0 $((N_REPLICATES-1))); do

        random_state=${random_states[$task_idx]}
        feature="${target_dataset}-${task_idx}"

        ## Make new sub-sub-directory for source dataset
        mkdir -p $TMPDIR/$target_dataset/$task_idx
        
        # Assign task to an idle GPU
        gpu_id=${idle_gpus[$((target_datasets_idx % ${#idle_gpus[@]}))]}
        run_eclare_task_on_gpu $clip_job_id $gpu_id $target_dataset $task_idx $random_state $genes_by_peaks_str $feature
    done

    # Wait for all tasks for this target dataset to complete before moving to the next one
    #wait

    target_datasets_idx=$((target_datasets_idx + 1))
done


cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR