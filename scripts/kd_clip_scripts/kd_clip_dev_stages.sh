#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/kd_clip_${JOB_ID}
TMPDIR=${OUTPATH}/kd_clip_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/eclare_scripts/eclare_run.py ./scripts/kd_clip_scripts/kd_clip_dev_stages.sh $TMPDIR

## Define target datasets
#target_dataset=("PFC_Zhu")
#source_datasets=("EaFet" "LaFet" "Inf" "Child" "Adol" "Adult")
#genes_by_peaks_str=("9832_by_70751")
#clip_job_id='18175628'
#ordinal_job_id='21125926'

target_dataset=("Cortex_Velmeshev")
target_dataset_lowercase=$(echo "${target_dataset}" | tr '[:upper:]' '[:lower:]')
genes_by_peaks_str=("9584_by_66620")
source_datasets=("FirstTrim_PFC_V1_Wang" "SecTrim_PFC_V1_Wang" "ThirdTrim_PFC_V1_Wang" "Inf_PFC_V1_Wang" "Adol_PFC_V1_Wang")
clip_job_id='16155618'
ordinal_job_id=None  # not really needed for KD_CLIP, since no teacher weights, although weights still logged

## Define JOB IDs and total number of epochs
total_epochs=10

## Define number of parallel tasks to run (replace with desired number of cores)
#N_CORES=6
N_REPLICATES=1

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
    local ordinal_job_id=$2
    local experiment_job_id=$3
    local gpu_id=$4
    local source_dataset=$5
    local target_dataset=$6
    local task_idx=$7
    local random_state=$8
    local genes_by_peaks_str=$9
    local feature=$10
    
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
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=800 \
    --feature="'$feature'" \
    &
    #--tune_hyperparameters \
    #--total_epochs=10 \
    #--n_trials=100 \ &
}


## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('clip_${target_dataset_lowercase}_${clip_job_id}')
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

    for task_idx in $(seq 0 $((N_REPLICATES-1))); do

        random_state=${random_states[$task_idx]}
        feature="${source_dataset}-to-${target_dataset}-${task_idx}"

        ## Make new sub-sub-directory for source dataset
        mkdir -p $TMPDIR/$target_dataset/$source_dataset/$task_idx
        
        # Assign task to an idle GPU
        gpu_id=${idle_gpus[$(((source_datasets_idx * N_REPLICATES + task_idx) % ${#idle_gpus[@]}))]}
        
        run_eclare_task_on_gpu $clip_job_id $ordinal_job_id $JOB_ID $gpu_id $source_dataset $target_dataset $task_idx $random_state $genes_by_peaks_str $feature
    done

    source_datasets_idx=$((source_datasets_idx + 1))
done


# Wait for all tasks for this target dataset to complete before moving to the next one
#wait


cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR