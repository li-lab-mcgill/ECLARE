#!/bin/bash -l
#SBATCH --job-name=process_datasets_mdd
#SBATCH --account=ctb-liyue
#SBATCH --time=1:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=249G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

## Align to MDD
target_dataset='MDD'
echo "=== Target dataset: $target_dataset ==="

## Define paired datasets to align to MDD
datasets=("mouse_brain_multiome" "pbmc_multiome")  # exclude SEA-AD

for source_dataset in "${datasets[@]}"; do

    feature="Align nuclei from $source_dataset data to $target_dataset data."
    echo "~~ $feature ~~"
    
    #srun python process_datasets.py \
    python process_datasets.py \
    --feature="$feature" \
    --source_dataset="$source_dataset" \
    --target_dataset="$target_dataset"
done

<< IGNORE
## MDD data only
feature="Process $target_dataset data, without alignment."
echo "~~ $feature ~~"
srun python process_datasets.py \
--feature=$feature \
--target_dataset=$target_dataset

## SEA-AD
source_dataset='SEA-AD'
feature="Align nuclei from $source_dataset data to $target_dataset data."
echo "~~ $feature ~~"
srun python process_datasets.py \
--feature=$feature \
--source_dataset=$source_dataset \
--target_dataset=$target_dataset
IGNORE