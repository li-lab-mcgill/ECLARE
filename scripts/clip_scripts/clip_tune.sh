#!/bin/bash -l
#SBATCH --job-name=clip_tune
#SBATCH --account=ctb-liyue
#SBATCH --time=0-6:30:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/ECLARE

## Make new sub-directory for current SLURM job ID and assign to "TMPDIR" variable
mkdir /home/dmannk/scratch/clip_tune_${SLURM_JOB_ID}
TMPDIR=/home/dmannk/scratch/clip_tune_${SLURM_JOB_ID}

## Copy scripts to sub-directory for reproducibility
cp clip_run.py clip_tune.sh $TMPDIR

## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"  
echo "r$SLURM_NODEID Launching python script"


## Perform celltyping analysis on Tabula Sapiens RNA data
srun python clip_run.py --outdir $TMPDIR \
--init_method=tcp://$MASTER_ADDR:3456 \
--world_size=$SLURM_NTASKS \
--feature='388 human brains, pretraining only with hyperparameter tuning' \
--dataset='388_human_brains' \
--genes_by_peaks_str='17716_by_85028' \
--triplet_type='clip' \
--total_epochs=40 \
--batch_size=500 \
--n_trials=4 \
--tune_hyperparameters \

## Move SLURM log file to sub-directory
mv slurm-${SLURM_JOB_ID}.out $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR
