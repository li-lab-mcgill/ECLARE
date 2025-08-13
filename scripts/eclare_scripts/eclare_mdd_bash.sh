#!/bin/bash -l
#SBATCH --job-name=weighted_ensemble_learning_multi_teacher_mdd
#SBATCH --account=ctb-liyue
#SBATCH --time=4:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=124G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL
 
#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE
 
mkdir /home/dmannk/scratch/scMulticlip_mdd_${SLURM_JOB_ID}
TMPDIR=/home/dmannk/scratch/scMulticlip_mdd_${SLURM_JOB_ID}
csv_file='/home/dmannk/projects/def-liyue/dmannk/data/genes_by_peaks_str.csv'

## Read the first column of the CSV to get dataset names (excludes MDD)
target_dataset="mdd"

echo "=== Target dataset: $target_dataset ==="
mkdir $TMPDIR/$target_dataset

## Perform celltyping analysis on Tabula Sapiens RNA data
srun python weighted_ensemble_learning.py --n_epochs=50 --save_latents --outdir=$TMPDIR/$target_dataset \
--clip_job_id='39309250' \
--target_dataset=$target_dataset \
--ignore_sources "PFC_Zhu" "DLPFC_Anderson" "DLPFC_Ma" "Midbrain_Adams" \
--genes_by_peaks_str='17563_by_100000' \
--total_epochs=$total_epochs \
--batch_size=800 \
--feature="'$feature'"

 
## Move SLURM log file to sub-directory
mv slurm-${SLURM_JOB_ID}.out $TMPDIR
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR
