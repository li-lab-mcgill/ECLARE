JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
echo "Job ID: scMulticlip_mdd_$JOB_ID"

## Make new sub-directory for current SLURM job ID and assign to "TMPDIR" variable
rm -rf /home/mcb/users/dmannk/scMultiCLIP/outputs/scMulticlip_mdd_${JOB_ID} && mkdir /home/mcb/users/dmannk/scMultiCLIP/outputs/scMulticlip_mdd_${JOB_ID}
TMPDIR=/home/mcb/users/dmannk/scMultiCLIP/outputs/scMulticlip_mdd_${JOB_ID}

exec > >(tee -a ${TMPDIR}/log.txt) 2>&1

conda init
conda activate torch_env_py39
cd /home/mcb/users/dmannk/scMultiCLIP/CLARE
 
## Copy scripts to sub-directory for reproducibility
cp weighted_ensemble_learning.py MCB_bash_scripts/weighted_ensemble_learning_multi_teacher_paired_data.sh $TMPDIR

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

## Read the first column of the CSV to get dataset names (excludes MDD)
target_dataset="mdd"

## Define total number of epochs
total_epochs=100

echo "=== Target dataset: $target_dataset ==="

    export RANDOM_STATES="${random_states[*]}"
    export TMPDIR
    export target_dataset
    export source_dataset
    export genes_by_peaks_str
    export total_epochs
    export idle_cuda_devices

    # Prepare arguments for xargs
    #for i in $(seq 0 $((N - 1))); do
    seq 0 $((N - 1)) | xargs -n1 -P$N -I{} bash -c '
        i={}
        random_state=$(echo $RANDOM_STATES | cut -d" " -f$((i + 1)))
        export RANDOM_STATE=$random_state
        echo "=== Random state: $random_state ==="

        ## Make new sub-directory for target dataset
        mkdir -p $TMPDIR/$target_dataset/${i}

        ## Cast as array
        idle_array=($idle_cuda_devices)

        ## Perform celltyping analysis on Tabula Sapiens RNA data
        CUDA_VISIBLE_DEVICES=${idle_array[i]} python weighted_ensemble_learning.py \
        --outdir=$TMPDIR/$target_dataset/${i} \
        --n_epochs=$total_epochs \
        --save_latents \
        --slurm_job_ids='15155920' \
        --target_dataset=$target_dataset \
        --replicate_idx=${i} \
        --genes_by_peaks_str='17563_by_100000' \
        --distil_lambda=0.1
    '
    #done
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

echo "Job complete."
