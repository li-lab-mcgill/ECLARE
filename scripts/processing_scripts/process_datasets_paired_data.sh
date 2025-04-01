#!/bin/bash -l
#SBATCH --job-name=process_datasets_paired_data
#SBATCH --account=ctb-liyue
#SBATCH --time=4:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

## Define all paired datasets
datasets=("PFC_Zhu" "DLPFC_Anderson" "DLPFC_Ma" "Midbrain_Adams" "multiome_mouse_brain" "pbmc_multiome")  # exclude SEA-AD

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
    
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${datasets[@]}"; do
        # Skip the case where source and target datasets are the same
        if [ "$source_dataset" != "$target_dataset" ]; then
            feature="Align nuclei from $source_dataset data to $target_dataset data."
            echo "~~ $feature ~~"
            
            srun python process_datasets.py \
            --feature="$feature" \
            --source_dataset="$source_dataset" \
            --target_dataset="$target_dataset"
        fi
    done
done