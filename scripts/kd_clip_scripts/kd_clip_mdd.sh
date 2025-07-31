#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/kd_clip_mdd_${JOB_ID}
TMPDIR=${OUTPATH}/kd_clip_mdd_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/eclare_scripts/eclare_run.py ./scripts/kd_clip_scripts/kd_clip_mdd.sh $TMPDIR
 
## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
  
## Set path to CSV file containing genes-by-peaks strings to match datasets
csv_file=${DATAPATH}/genes_by_peaks_str.csv
 
## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

source_datasets=("pbmc_10x" "mouse_brain_10x")

## Preset target dataset
target_dataset="MDD"

## Define total number of epochs
clip_job_id='16204608'
total_epochs=100

## Define number of parallel tasks to run (replace with desired number of cores)
N_CORES=3
N_REPLICATES=3

## Define random state
RANDOM=42
random_states=()
for i in $(seq 0 $((N_REPLICATES - 1))); do
    random_states+=($RANDOM)
done

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
    local experiment_job_id=$2
    local gpu_id=$3
    local source_dataset=$4
    local target_dataset=$5
    local task_idx=$6
    local random_state=$7
    local feature=$8

    echo "Running ${source_dataset} to ${target_dataset} (task $task_idx) on GPU $gpu_id"

    # for paired data, do not specify source_dataset to ensure that all datasets serve as sources
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py \
    --outdir $TMPDIR/$target_dataset/$source_dataset/$task_idx \
    --replicate_idx=$task_idx \
    --clip_job_id=$clip_job_id \
    --experiment_job_id=$experiment_job_id \
    --source_dataset=$source_dataset \
    --target_dataset=$target_dataset \
    --genes_by_peaks_str='17563_by_100000' \
    --total_epochs=$total_epochs \
    --batch_size=800 \
    --feature="$feature" \
    --distil_lambda=0.1 #&  # Add & to run in background

    # Increment job counter
    ((current_jobs++))
    
    # If we've reached N_CORES jobs, wait for one to finish
    if [ $current_jobs -ge $N_CORES ]; then
        wait -n  # Wait for any background job to finish
        ((current_jobs--))
    fi
}

# Initialize job counter
current_jobs=0

## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('clip_mdd_${clip_job_id}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'KD_CLIP_${JOB_ID}'
client.create_run(experiment_id, run_name=run_name)
"

## Middle loop: iterate over datasets as the source_dataset
source_datasets_idx=0
for source_dataset in "${source_datasets[@]}"; do

    # Skip the case where source and target datasets are the same
    if [ "$source_dataset" != "$target_dataset" ]; then
        
        ## Check if extraction was successful
        if [ $? -ne 0 ]; then
            continue
        fi

        ## Inner loop: iterate over task indices
        for task_idx in $(seq 0 $((N_REPLICATES-1))); do

            random_state=${random_states[$task_idx]}
            feature="${source_dataset}-to-${target_dataset}-${task_idx}"

            ## Make new sub-sub-directory for source dataset
            mkdir -p $TMPDIR/$target_dataset/$source_dataset/$task_idx
            
            # Assign task to an idle GPU
            gpu_id=${idle_gpus[$((source_datasets_idx % ${#idle_gpus[@]}))]}

            # Run ECLARE task on idle GPU
            run_eclare_task_on_gpu $clip_job_id $JOB_ID $gpu_id $source_dataset $target_dataset $task_idx $random_state $feature
        done

    fi
    source_datasets_idx=$((source_datasets_idx + 1))
done

# Wait for any remaining jobs to finish
wait

cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR