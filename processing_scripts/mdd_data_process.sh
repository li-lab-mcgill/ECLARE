#!/bin/bash -l
#SBATCH --job-name=mdd_data_process
#SBATCH --account=def-liyue
#SBATCH --time=0-1:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023 gcc/12.3 r/4.4.0 r-bundle-bioconductor/3.18
Rscript mdd_data_process.R