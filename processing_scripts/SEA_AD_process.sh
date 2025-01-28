#!/bin/bash -l
#SBATCH --job-name=sea_ad_data_process
#SBATCH --account=def-liyue
#SBATCH --time=0-0:45:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

srun python SEA_AD_process.py