#!/bin/bash -l
#SBATCH --job-name=glue_vertical_parallel
#SBATCH --account=ctb-liyue
#SBATCH --time=2:30:0
#SBATCH --mem=50G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

cd /home/dmannk/projects/def-liyue/dmannk/CLARE/benchmark_vertical/glue/
source ${PWD}/glue_env/bin/activate

TMPDIR=/home/dmannk/scratch/glue_${SLURM_ARRAY_JOB_ID}
mkdir -p $TMPDIR

## Read the first column of the CSV to get dataset names (excludes MDD)
csv_file='/home/dmannk/projects/def-liyue/dmannk/data/genes_by_peaks_str.csv'
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

## Use SLURM_ARRAY_TASK_ID to get the specific target dataset
target_dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
echo "=== Target dataset: $target_dataset ==="

## Make new sub-directory for target dataset (overwrite if exists)
rm -rf $TMPDIR/$target_dataset && mkdir $TMPDIR/$target_dataset

## Iterate over source datasets
for source_dataset in "${datasets[@]}"; do
    # Skip the case where source and target datasets are the same
    if [ "$source_dataset" != "$target_dataset" ]; then
        feature="Align nuclei from $source_dataset data to $target_dataset data."
        echo "~~ $feature ~~"

        ## Extract the value of `genes_by_peaks_str`
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

        srun python ${PWD}/glue_run.py --outdir $TMPDIR/$target_dataset/$source_dataset \
        --source_dataset="$source_dataset" \
        --target_dataset="$target_dataset"
    fi
done

## Move SLURM log file to sub-directory
mv slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $TMPDIR

## Remove write permission from sub-directory and its files to prevent accidental corruption
#chmod -R -w $TMPDIR
