#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/ordinal_${JOB_ID}
TMPDIR=${OUTPATH}/ordinal_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/ordinal_scripts/ordinal_run.py ./scripts/ordinal_scripts/ordinal.sh $TMPDIR
 
## Define total number of epochs
total_epochs=100

source_dataset="PFC_V1_Wang"
genes_by_peaks_str="9914_by_63404"
#source_dataset="PFC_Zhu"
#genes_by_peaks_str="9832_by_70751"

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

## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('ordinal_${JOB_ID}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'ORDINAL_${JOB_ID}'
client.create_run(experiment_id, run_name=run_name)
"

## Make new sub-sub-directory for source dataset
mkdir -p $TMPDIR

# Assign task to an idle GPU, ensuring both source_dataset_idx and task_idx are used for load balancing
gpu_id=${idle_gpus[-1]}

echo "Running 'ordinal' on GPU $gpu_id"
CUDA_VISIBLE_DEVICES=$gpu_id \
python ${ECLARE_ROOT}/scripts/ordinal_scripts/ordinal_run.py \
--outdir $TMPDIR \
--total_epochs=$total_epochs \
--source_dataset=$source_dataset \
--genes_by_peaks_str=$genes_by_peaks_str \
--feature=$source_dataset \
--job_id $JOB_ID