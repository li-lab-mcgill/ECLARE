#!/bin/bash -l
 
conda activate eclare_env
cd $ECLARE_ROOT

# Parse command line arguments
usage() {
    echo "Usage: $0 [-g GPU_IDS] [-c CPU_CORES] [-h]"
    echo "  -g GPU_IDS     Comma-separated list of GPU IDs to use (e.g., '0,1,2')"
    echo "  -c CPU_CORES   Comma-separated list of CPU core IDs to use (e.g., '0,1,2')"
    echo "  -h             Show this help message"
    exit 1
}

# Default values
gpu_ids=""

# Parse arguments
while getopts "g:h" opt; do
    case $opt in
        g) gpu_ids="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$gpu_ids" ]; then
    echo "Error: GPU IDs (-g) must be specified"
    usage
fi

# Convert comma-separated string to array
IFS=',' read -ra gpu_array <<< "$gpu_ids"
echo "Using GPUs: ${gpu_array[@]}"
 
## Copy scripts to sub-directory for reproducibility
#cp ./scripts/eclare_scripts/eclare_run.py ./scripts/eclare_scripts/eclare_mdd.sh $TMPDIR
 
## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
  
## Set path to CSV file containing genes-by-peaks strings to match datasets
csv_file=${DATAPATH}/genes_by_peaks_str.csv
 
## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

source_datasets=("PFC_V1_Wang" "PFC_Zhu")

## Preset target dataset
target_dataset="Cortex_Velmeshev"
genes_by_peaks_str='9584_by_66620'
clip_job_id='25165730'
#ordinal_job_id='25195201'
#eclare_job_id='26214511'

#clip_job_id='30153403'
#target_dataset="MDD"
#genes_by_peaks_str='17563_by_100000'

## Define total number of epochs
total_epochs=10
target_dataset_lowercase=$(echo "${target_dataset}" | tr '[:upper:]' '[:lower:]')

## Make new sub-directory for current job ID and assign to "TMPDIR" variable
JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
mkdir -p ${OUTPATH}/eclare_${target_dataset_lowercase}_${JOB_ID}
TMPDIR=${OUTPATH}/eclare_${target_dataset_lowercase}_${JOB_ID}

## Define number of parallel tasks to run (replace with desired number of cores)
N_CORES=1
N_REPLICATES=1

## Define random state
RANDOM=42
random_states=()
for i in $(seq 0 $((N_CORES - 1))); do
    random_states+=($RANDOM)
done


## Create a temporary file to store all the commands we want to run
commands_file=$(mktemp)

# Function to run a task on a specific GPU
run_eclare_task_on_gpu() {
    local clip_job_id=$1
    local experiment_job_id=$2
    local gpu_id=$3
    local target_dataset=$4
    local task_idx=$5
    local random_state=$6
    local feature=$7

    echo "Running ${source_dataset} to ${target_dataset} (task $task_idx) on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py \
    --outdir $TMPDIR/$target_dataset/$source_dataset/$task_idx \
    --replicate_idx=$task_idx \
    --clip_job_id=$clip_job_id \
    --experiment_job_id=$experiment_job_id \
    --target_dataset=$target_dataset \
    --genes_by_peaks_str=$genes_by_peaks_str \
    --total_epochs=$total_epochs \
    --batch_size=800 \
    --feature="'$feature'" &
    #--ordinal_job_id=$ordinal_job_id \
    #--eclare_job_id=$eclare_job_id \
    #--tune_hyperparameters \
    #--args.total_epochs=10 \
    #--n_trials=3 &
}

## Create experiment ID (or detect if it already exists)
python -c "
from src.eclare.run_utils import get_or_create_experiment; 
experiment = get_or_create_experiment('clip_${target_dataset_lowercase}_${clip_job_id}')
experiment_id = experiment.experiment_id
print(experiment_id)

from mlflow import MlflowClient
client = MlflowClient()
run_name = 'ECLARE_${JOB_ID}'
client.create_run(experiment_id, run_name=run_name)
"

## Outer loop: iterate over datasets as the target_dataset

## Check if extraction was successful
if [ $? -ne 0 ]; then
    continue
fi

for task_idx in $(seq 0 $((N_REPLICATES-1))); do
    random_state=${random_states[$task_idx]}
    feature="${target_dataset}-${task_idx}"

    ## Make new sub-sub-directory for source dataset
    mkdir -p $TMPDIR/$target_dataset/$task_idx
    
    # Assign task to a GPU (cycling through available GPUs)
    gpu_id=${gpu_array[$((task_idx % ${#gpu_array[@]}))]}
    
    run_eclare_task_on_gpu $clip_job_id $JOB_ID $gpu_id $target_dataset $task_idx $random_state $feature
done

# Wait for all tasks for this target dataset to complete before moving to the next one
#wait



cp $commands_file $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR 