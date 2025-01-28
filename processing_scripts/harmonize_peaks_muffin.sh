#!/bin/bash -l
#SBATCH --job-name=harmonize_peaks
#SBATCH --account=ctb-liyue
#SBATCH --time=0-5:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=156G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

## Send data to local scratch, more efficient I/O
#mkdir -p $SCRATCH/PD_all_files
#cp /home/dmannk/projects/def-liyue/dmannk/data/genome_annot/hg38.chrom.sizes.sorted $SCRATCH # $SLURM_TMPDIR
#cp /home/dmannk/projects/def-liyue/dmannk/data/PD_Adams_et_al/GSE193688_RAW.tar $SCRATCH # $SLURM_TMPDIR
#tar -xvf ${SCRATCH}/GSE193688_RAW.tar -C ${SCRATCH}/PD_all_files

## run script
srun python -u harmonize_peaks_muffin.py