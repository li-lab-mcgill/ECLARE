#!/bin/bash -l
#SBATCH --job-name=process_datasets_paired_data
#SBATCH --account=ctb-liyue
#SBATCH --time=2:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

#source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
conda activate eclare_env
cd $ECLARE_ROOT

## Define all paired datasets
datasets=("DLPFC_Anderson" "DLPFC_Ma" "mouse_brain_10x")

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
    
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${datasets[@]}"; do

        # Skip the case where source and target datasets are the same
        if [ "$source_dataset" != "$target_dataset" ]; then
        
            feature="Align nuclei from $source_dataset data to $target_dataset data."
            echo -e "\n\n ~~ $feature ~~ \n\n"
            
            #srun python process_datasets.py \
            python ${ECLARE_ROOT}/scripts/processing_scripts/process_datasets.py \
            --feature="$feature" \
            --source_dataset="$source_dataset" \
            --target_dataset="$target_dataset"
        fi
    done
done