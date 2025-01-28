#!/bin/bash -l
#SBATCH --job-name=388_human_brains_to_h5ad
#SBATCH --account=def-liyue
#SBATCH --time=0-1:30:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=30G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/scTripletgrate

srun python save_388_human_brains_to_h5ad.py --batch_correction None --save_type 'individual'