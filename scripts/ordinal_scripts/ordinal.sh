#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT

# Parse command line arguments
usage() {
    echo "Usage: $0 [-g GPU_ID] [-h]"
    echo "  -g GPU_ID      Specific GPU ID to use (e.g., '0', '1', '2')"
    echo "  -h             Show this help message"
    exit 1
}

# Default values
gpu_id_arg=""

# Parse arguments
while getopts "g:h" opt; do
    case $opt in
        g) gpu_id_arg="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    esac
done
 
## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/ordinal_${JOB_ID}
TMPDIR=${OUTPATH}/ordinal_${JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp ./scripts/ordinal_scripts/ordinal_run.py ./scripts/ordinal_scripts/ordinal.sh $TMPDIR
 
## Define total number of epochs
total_epochs=50

source_dataset="PFC_Zhu"
target_dataset="MDD"
genes_by_peaks_str="6816_by_55284"

#source_dataset="Cortex_Velmeshev"
#genes_by_peaks_str="9584_by_66620"

#source_dataset="PFC_V1_Wang"
#genes_by_peaks_str="9914_by_63404"

#source_dataset="PFC_V1_Wang"
#target_dataset="Cortex_Velmeshev"
#genes_by_peaks_str="6124_by_19914"

#source_dataset="PFC_Zhu"
#genes_by_peaks_str="9832_by_70751"

#source_dataset="PFC_Zhu"
#target_dataset="Cortex_Velmeshev"
#genes_by_peaks_str="6033_by_16249"

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

# GPU selection logic
if [ -n "$gpu_id_arg" ]; then
    # Use the provided GPU ID
    echo "Using specified GPU: $gpu_id_arg"
    gpu_id=$gpu_id_arg
    
    # Validate that the GPU exists
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$gpu_id" -ge "$num_gpus" ] || [ "$gpu_id" -lt 0 ]; then
        echo "Error: GPU ID $gpu_id is out of range. Available GPUs: 0 to $((num_gpus-1))"
        exit 1
    fi
    
    # Check if the specified GPU is idle (optional warning)
    if ! is_gpu_idle $gpu_id; then
        echo "Warning: Specified GPU $gpu_id is not idle. Proceeding anyway..."
    fi
else
    # Automatic idle GPU detection (original logic)
    echo "No GPU specified, detecting idle GPUs automatically..."
    
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
    
    # Check if any idle GPUs were found
    if [ ${#idle_gpus[@]} -eq 0 ]; then
        echo "Error: No idle GPUs found. Please specify a GPU manually with -g option."
        exit 1
    fi
    
    # Assign task to an idle GPU
    gpu_id=${idle_gpus[-1]}
    echo "Selected idle GPU: $gpu_id"
fi

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

echo "Running 'ordinal' on GPU $gpu_id"
CUDA_VISIBLE_DEVICES=$gpu_id \
python ${ECLARE_ROOT}/scripts/ordinal_scripts/ordinal_run.py \
--outdir $TMPDIR \
--total_epochs=$total_epochs \
--source_dataset=$source_dataset \
--target_dataset=$target_dataset \
--source_or_target="source" \
--genes_by_peaks_str=$genes_by_peaks_str \
--feature=$source_dataset \
--job_id $JOB_ID