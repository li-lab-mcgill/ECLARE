#!/bin/bash -l
#SBATCH --job-name=clip_mdd
#SBATCH --account=ctb-liyue
#SBATCH --time=0-1:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --mem=61G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL
 
#source /home/dmannk/projects/def-liyue/dmannk/torch_env/bin/activate
source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/ECLARE
 
## Make new sub-directory for current SLURM job ID and assign to "TMPDIR" variable
mkdir /home/dmannk/scratch/clip_mdd_${SLURM_JOB_ID}
TMPDIR=/home/dmannk/scratch/clip_mdd_${SLURM_JOB_ID}
 
## Copy scripts to sub-directory for reproducibility
cp clip_run.py clip_mdd.sh $TMPDIR
 
## https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
 
echo "r$SLURM_NODEID master: $MASTER_ADDR"  
echo "r$SLURM_NODEID Launching python script"
 
## Define target dataset
 
csv_file='/home/dmannk/projects/def-liyue/dmannk/data/genes_by_peaks_str.csv'
 
## Read the first column of the CSV to get dataset names (excludes MDD)
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))
 
total_epochs=2

target_dataset="mdd"

echo "=== Target dataset: $target_dataset ==="
mkdir $TMPDIR/$target_dataset
 
## Inner loop: iterate over datasets as the source_dataset
for source_dataset in "${datasets[@]}"; do

    # Skip the case where source and target datasets are the same (shouldn't need for this MDD scripts specifically)
    if [ "$source_dataset" != "$target_dataset" ]; then
        feature="Align nuclei from $source_dataset data to $target_dataset data."
        echo "~~ $feature ~~"

        ## Extract the value of `genes_by_peaks_str` for the current source and target
        genes_by_peaks_str=$(awk -F',' -v source="$source_dataset" -v target="$target_dataset" '
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    if ($i == target) target_idx = i
                }
            }
            $1 == source {
                print $target_idx
            }
        ' "$csv_file")

        ## Check if the value was successfully extracted
        if [ -z "$genes_by_peaks_str" ]; then
            echo "Warning: No value found for source=$source_dataset, target=$target_dataset"
            continue
        fi

        ## Make new sub-sub-directory for source dataset
        mkdir $TMPDIR/$target_dataset/$source_dataset
        
        ## Run the python script
        srun clip_run.py --outdir $TMPDIR/$target_dataset/$source_dataset \
        --source_dataset=$source_dataset \
        --target_dataset=$target_dataset \
        --genes_by_peaks_str=$genes_by_peaks_str \
        --total_epochs=$total_epochs
    fi
done
 
## Move SLURM log file to sub-directory
mv slurm-${SLURM_JOB_ID}.out $TMPDIR
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR