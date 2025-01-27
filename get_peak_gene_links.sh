#!/bin/bash -l
#SBATCH --job-name=get_peak_gene_links
#SBATCH --account=ctb-liyue
#SBATCH --time=0-1:30:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=187G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

## Perform celltyping analysis on Tabula Sapiens RNA data
srun python get_peak_gene_links.py \
--feature='Get peak-gene links from MDD data, trained on AD data' \
--slurm_id='34887192'