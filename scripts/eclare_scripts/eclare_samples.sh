#!/bin/bash

# Create a unique job ID
JOB_ID=$(date +%d%H%M%S)
echo "Job ID: eclare_$JOB_ID"

# Create a temporary directory for the job
rm -rf ${OUTPATH}/eclare_${JOB_ID} && mkdir ${OUTPATH}/eclare_${JOB_ID}
TMPDIR=${OUTPATH}/eclare_${JOB_ID}

# Redirect output to a log file
exec > >(tee -a ${TMPDIR}/log.txt) 2>&1

# Copy scripts to the temporary directory
cp ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_samples.sh $TMPDIR

# Read datasets from a CSV file
csv_file=${DATAPATH}/'genes_by_peaks_str_samples.csv'
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

# Define random states
N=1
RANDOM=42
random_states=()
for ((i = 0; i < N; i++)); do
    random_states+=($RANDOM)
done

# Define total number of epochs
total_epochs=$1
clip_job_id=$2

echo "Total epochs: $total_epochs"
echo "CLIP job ID: $clip_job_id"

for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="

    # Prepare arguments for xargs
    for ((i = 0; i < N; i++)); do

        random_state=${random_states[$i]}
        export RANDOM_STATE=$random_state
        echo "=== Random state: $random_state ==="

        ## Make new sub-directory for target dataset
        mkdir -p $TMPDIR/$target_dataset/${i}

        ## Perform celltyping analysis on Tabula Sapiens RNA data
        python ${ECLARE_ROOT}/scripts/eclare_scripts/eclare_run.py \
        --outdir=$TMPDIR/$target_dataset/${i} \
        --n_epochs=$total_epochs \
        --clip_job_id=$clip_job_id \
        --target_dataset=$target_dataset \
        --genes_by_peaks_str='6816_by_55284' \
        --distil_lambda=0.1

    done
done
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

echo "Job complete."
