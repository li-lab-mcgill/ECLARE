#!/bin/bash

JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
echo "Job ID: clip_$JOB_ID"

## Make new sub-directory for current SLURM job ID and assign to "TMPDIR" variable
rm -rf ${OUTPATH}/clip_${JOB_ID} && mkdir ${OUTPATH}/clip_${JOB_ID}
TMPDIR=${OUTPATH}/clip_${JOB_ID}

exec > >(tee -a ${TMPDIR}/log.txt) 2>&1
 
## Copy scripts to sub-directory for reproducibility
cp ${ECLARE_ROOT}/scripts/clip_scripts/clip_run.py ${ECLARE_ROOT}/scripts/clip_scripts/clip_samples.sh $TMPDIR
 
## Read the first column and first row to get dataset names
csv_file=${DATAPATH}/'genes_by_peaks_str_samples.csv'
source_datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))
target_datasets="PFC_Zhu"

## Print datasets for verification
echo "Source datasets: ${source_datasets[@]}"
echo "Target datasets: ${target_datasets[@]}"

## Define random state
N=1         # number of train/test replicates to run, set to 3 to replicate manuscript results
RANDOM=42
random_states=()
for ((i = 0; i < N; i++)); do
    random_states+=($RANDOM)
done

## Define total number of epochs
total_epochs=$1
 
## -- Train CLIP models from source datasets -- ##

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${target_datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
    
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${source_datasets[@]}"; do
 
        feature="Align nuclei from $source_dataset data to $target_dataset data."
        echo "~~ $feature ~~"

        ## Extract the values of `genes_by_peaks_str` for the current source and target
        genes_by_peaks_str=$(awk -F',' -v source="$source_dataset" -v target="$target_dataset" '
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    if ($i == target) target_idx = i
                }
            }
            $1 == source {
                split($target_idx, arr, ",")
                print arr[2]  # Extract only the second part after splitting by comma
            }
        ' "$csv_file")

        # Sanitize the genes_by_peaks_str to remove any carriage return or newline characters
        genes_by_peaks_str=$(echo "$genes_by_peaks_str" | tr -d '\r\n')

        ## Check if the value was successfully extracted
        if [ -z "$genes_by_peaks_str" ]; then
            echo "Warning: No value found for source=$source_dataset, target=$target_dataset"
            continue
        fi

        echo "Genes by peaks string: $genes_by_peaks_str"

        # Prepare arguments for xargs
        for ((i = 0; i < N; i++)); do

            random_state=${random_states[$i]}
            export RANDOM_STATE=$random_state
            echo "=== Random state: $random_state ==="

            # Make new sub-sub-directory for source dataset
            mkdir -p $TMPDIR/$target_dataset/$source_dataset/${i}

            # Run the python script
            python ${ECLARE_ROOT}/scripts/clip_scripts/clip_run.py \
                --outdir $TMPDIR/$target_dataset/$source_dataset/${i} \
                --source_dataset=$source_dataset \
                --target_dataset=$target_dataset \
                --genes_by_peaks_str=$genes_by_peaks_str \
                --total_epochs=$total_epochs
        done
        
    done
done
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

echo "Job complete."
