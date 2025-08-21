#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/eclare_${JOB_ID}
TMPDIR=${OUTPATH}/eclare_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/eclare_scripts/eclare_run.py ./scripts/eclare_scripts/eclare_dev_stages.sh $TMPDIR

## Define target datasets
#target_dataset=("PFC_Zhu")
#genes_by_peaks_str=("9832_by_70751")

target_dataset=("PFC_V1_Wang")
genes_by_peaks_str=("9914_by_63404")

## Define JOB IDs and total number of epochs
clip_job_id='20194800'
ordinal_job_id='20180433'  # not really needed for KD_CLIP, since no teacher weights, although weights still logged
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
    local target_dataset=$5
    local task_idx=$6
    local random_state=$7
    local genes_by_peaks_str=$8
    local feature=$9
    
    echo "Running ${source_dataset} to ${target_dataset} (task $task_idx) on GPU $gpu_id"

    # for paired data, do not specify source_dataset to ensure that all datasets serve as sources
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py \
    --outdir $TMPDIR/$target_dataset/$source_dataset/$task_idx \
    --replicate_idx=$task_idx \
    --clip_job_id=$clip_job_id \
    --ordinal_job_id=$ordinal_job_id \
    --experiment_job_id=$experiment_job_id \
    --target_dataset=$target_dataset \
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=500 \
    --feature="'$feature'"
    #--tune_hyperparameters \
    #--total_epochs=10 \
    #--n_trials=100 \ &
}


## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('clip_${clip_job_id}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'ECLARE_${JOB_ID}'
client.create_run(experiment_id, run_name=run_name)
"
    
for task_idx in $(seq 0 $((N_REPLICATES-1))); do

    random_state=${random_states[$task_idx]}
    feature="${target_dataset}-${task_idx}"

    ## Make new sub-sub-directory for source dataset
    mkdir -p $TMPDIR/$target_dataset/$task_idx
    
    # Assign task to an idle GPU
    gpu_id=${idle_gpus[-1]}
    
    run_eclare_task_on_gpu $clip_job_id $ordinal_job_id $JOB_ID $gpu_id $target_dataset $task_idx $random_state $genes_by_peaks_str $feature
done

# Wait for all tasks for this target dataset to complete before moving to the next one
#wait


cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR