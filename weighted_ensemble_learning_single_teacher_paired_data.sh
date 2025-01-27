#!/bin/bash -l
#SBATCH --job-name=weighted_ensemble_learning_single_teacher_paired_data
#SBATCH --account=ctb-liyue
#SBATCH --time=1:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=62G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL
 
#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE
 
mkdir /home/dmannk/scratch/kd_clip_paired_${SLURM_JOB_ID}
TMPDIR=/home/dmannk/scratch/kd_clip_paired_${SLURM_JOB_ID}

## Read the first column of the CSV to get dataset names (excludes MDD)
csv_file='/home/dmannk/projects/def-liyue/dmannk/data/genes_by_peaks_str.csv'
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="

    ## Make new sub-directory for target dataset (overwrite if exists)
    rm -rf $TMPDIR/$target_dataset && mkdir $TMPDIR/$target_dataset
 
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${datasets[@]}"; do

        # Skip the case where source and target datasets are the same
        if [ "$source_dataset" != "$target_dataset" ]; then
            feature="Align nuclei from $source_dataset data to $target_dataset data."
            echo "~~ $feature ~~"

            ## Extract the value of `genes_by_peaks_str` for aligning the target dataset to MDD (target thus becomes source)
            genes_by_peaks_str_mdd=$(awk -F',' -v source="$target_dataset" -v target="mdd" '
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
            if [ -z "$genes_by_peaks_str_mdd" ]; then
                echo "Warning: No value found for source=$target_dataset, target=mdd"
                continue
            fi

            ## Make new sub-directory for target dataset
            mkdir $TMPDIR/$target_dataset/$source_dataset

            ## Perform celltyping analysis on Tabula Sapiens RNA data
            srun python weighted_ensemble_learning.py --n_epochs=50 --outdir=$TMPDIR/$target_dataset/$source_dataset \
            --slurm_job_ids='38959699' \
            --source_dataset=$source_dataset \
            --target_dataset=$target_dataset \
            --genes_by_peaks_str=$genes_by_peaks_str_mdd \
            --distil_lambda=0.1

        fi
    done
done

 
## Move SLURM log file to sub-directory
mv slurm-${SLURM_JOB_ID}.out $TMPDIR
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

<< FEATURE

L=0.1. Log MDD even if MDD not target. Logger for foscttm global and by cell-type, with teacher references. OT-CLIP on paired data. Use slurm_id and genes_by_peaks_str from commit 57a01d9 for paired target data

FEATURE
