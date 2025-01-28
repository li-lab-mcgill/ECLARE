#!/bin/bash -l
#SBATCH --job-name=weighted_ensemble_learning_multi_teacher_paired_data
#SBATCH --account=ctb-liyue
#SBATCH --time=1:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=31G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL
 
#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE
 
mkdir /home/dmannk/scratch/scMulticlip_paired_${SLURM_JOB_ID}
TMPDIR=/home/dmannk/scratch/scMulticlip_paired_${SLURM_JOB_ID}
csv_file='/home/dmannk/projects/def-liyue/dmannk/data/genes_by_peaks_str.csv'

## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
 
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
    mkdir $TMPDIR/$target_dataset

    ## Perform celltyping analysis on Tabula Sapiens RNA data
    srun python weighted_ensemble_learning.py --n_epochs=50 --save_latents --outdir=$TMPDIR/$target_dataset \
    --slurm_job_ids='38959699' \
    --target_dataset=$target_dataset \
    --genes_by_peaks_str=$genes_by_peaks_str_mdd \
    --distil_lambda=0.1
done

 
## Move SLURM log file to sub-directory
mv slurm-${SLURM_JOB_ID}.out $TMPDIR
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR

<< FEATURE

L=0.1. Log MDD even if MDD not target. Logger for foscttm global and by cell-type, with teacher references. OT-CLIP on paired data. Use slurm_id and genes_by_peaks_str from commit 57a01d9 for paired target data

FEATURE
