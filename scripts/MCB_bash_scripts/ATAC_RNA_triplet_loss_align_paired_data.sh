JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
echo "Job ID: clip_$JOB_ID"

## Make new sub-directory for current SLURM job ID and assign to "TMPDIR" variable
rm -rf /home/mcb/users/dmannk/scMultiCLIP/outputs/clip_${JOB_ID} && mkdir /home/mcb/users/dmannk/scMultiCLIP/outputs/clip_${JOB_ID}
TMPDIR=/home/mcb/users/dmannk/scMultiCLIP/outputs/clip_${JOB_ID}

exec > >(tee -a ${TMPDIR}/log.txt) 2>&1

conda init
conda activate torch_env_py39
cd /home/mcb/users/dmannk/scMultiCLIP/CLARE
 
## Copy scripts to sub-directory for reproducibility
cp ATAC_RNA_triplet_loss_align.py MCB_bash_scripts/ATAC_RNA_triplet_loss_align_paired_data.sh $TMPDIR
 
## Read the first column of the CSV to get dataset names (excludes MDD)
csv_file='/home/mcb/users/dmannk/scMultiCLIP/data/genes_by_peaks_str.csv'
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

## Define random state
N=3         # number of jobs to run in parallel
RANDOM=42
random_states=()
for ((i = 0; i < N; i++)); do
    random_states+=($RANDOM)
done

## Detect idle GPUs
idle_cuda_devices=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '$2 < 100 && $3 == 0 {print $1}')

## Define total number of epochs
total_epochs=2
n_trials=2
## Train CLARE from source datasets

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
    
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${datasets[@]}"; do
 
        # Skip the case where source and target datasets are the same
        if [ "$source_dataset" != "$target_dataset" ]; then
            feature="$source_dataset data to $target_dataset data."
            echo "~~ $feature ~~"
 
            ## Extract the value of `genes_by_peaks_str` for the current source and target
            genes_by_peaks_str=$(awk -F',' -v source="$source_dataset" -v target="$target_dataset" '
                NR == 1 {
                    for (i = 1; i <= NF; i++) {
                        if ($i == target) target_idx = i
                    }
                }
                $1 == source {
                    print $target_idx
                }
            ' "$csv_file")
 
            ## Check if the value was successfully extracted
            if [ -z "$genes_by_peaks_str" ]; then
                echo "Warning: No value found for source=$source_dataset, target=$target_dataset"
                continue
            fi
 
            export RANDOM_STATES="${random_states[*]}"
            export TMPDIR
            export target_dataset
            export source_dataset
            export genes_by_peaks_str
            export total_epochs
            export idle_cuda_devices
            export feature
            export n_trials

            # Prepare arguments for xargs
            seq 0 $((N - 1)) | xargs -n1 -P$N -I{} bash -c '
                i={}
                random_state=$(echo $RANDOM_STATES | cut -d" " -f$((i + 1)))
                export RANDOM_STATE=$random_state
                echo "=== Random state: $random_state ==="

                # Make new sub-sub-directory for source dataset
                mkdir -p $TMPDIR/$target_dataset/$source_dataset/${i}

                ## Cast as array
                idle_array=($idle_cuda_devices)

                ## echo all arguments
                echo "=== All arguments: ==="
                echo "feature: $feature"
                echo "n_trials: $n_trials"
                echo "target_dataset: $target_dataset"
                echo "source_dataset: $source_dataset"
                echo "genes_by_peaks_str: $genes_by_peaks_str"
                echo "total_epochs: $total_epochs"

                # Run the python script
                CUDA_VISIBLE_DEVICES=${idle_array[${i}]} \
                python ${ECLARE_ROOT}/scripts/clip_scripts/clip_run.py \
                    --outdir $TMPDIR/$target_dataset/$source_dataset/${i} \
                    --source_dataset=$source_dataset \
                    --target_dataset=$target_dataset \
                    --genes_by_peaks_str=$genes_by_peaks_str \
                    --total_epochs=$total_epochs \
                    --feature='${feature}' \
                    --tune_hyperparameters \
                    --n_trials=$n_trials
            '

        fi
    done
done
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

echo "Job complete."
