#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/clip_${JOB_ID}
TMPDIR=${OUTPATH}/clip_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/clip_scripts/clip_run.py ./scripts/clip_scripts/clip_dev_stages.sh $TMPDIR

#dev_stages=("EaFet" "LaFet" "Inf" "Child" "Adol" "Adult")
#dataset=("PFC_Zhu")
#genes_by_peaks_str=("9832_by_70751")

dev_stages=("FirstTrim" "SecTrim" "ThirdTrim" "Inf" "Adol")
dataset=("PFC_V1_Wang")
genes_by_peaks_str=("9914_by_63404")


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
total_epochs=10

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
    local dataset=$2
    local dev_stage=$3
    local task_idx=$4
    local random_state=$5
    local genes_by_peaks_str=$6
    local feature=$7

    echo "Running '${feature}' on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/clip_scripts/clip_run.py \
    --outdir $TMPDIR/$dataset/$dev_stage/$task_idx \
    --source_dataset=$dataset \
    --target_dataset=$dataset \
    --keep_group=$dev_stage \
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=800 \
    --feature="${feature}" \
    --metric_to_optimize="1-foscttm" \
    --job_id=$JOB_ID
    #--tune_hyperparameters \
    #--n_trials=3 &
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

## iterate over datasets as the source_dataset
dev_stage_idx=0
for dev_stage in "${dev_stages[@]}"; do

    ## Inner loop: iterate over task replicates (one per GPU)
    for task_idx in $(seq 0 $((N_REPLICATES-1))); do

        random_state=${random_states[$task_idx]}
        feature="${dataset}_${dev_stage}-${task_idx}"

        ## Make new sub-sub-directory for source dataset
        mkdir -p $TMPDIR/$dataset/$dev_stage/$task_idx
        
        # Assign task to an idle GPU, ensuring both source_dataset_idx and task_idx are used for load balancing
        gpu_id=${idle_gpus[$(((dev_stage_idx * N_REPLICATES + task_idx) % ${#idle_gpus[@]}))]}

        # Run CLIP task on idle GPU
        run_clip_task_on_gpu $gpu_id $dataset $dev_stage $task_idx $random_state $genes_by_peaks_str $feature
    done

    dev_stage_idx=$((dev_stage_idx + 1))

done

wait # Wait for all tasks to end

cp $commands_file $TMPDIR
