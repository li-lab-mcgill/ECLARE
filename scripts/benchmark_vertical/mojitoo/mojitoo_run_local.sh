JOB_ID=$(date +%d%H%M%S)  # very small chance of collision
echo "Job ID: mojitoo_$JOB_ID"

 
rm -rf /Users/dmannk/cisformer/outputs/mojitoo_${JOB_ID} && mkdir /Users/dmannk/cisformer/outputs/mojitoo_${JOB_ID}
TMPDIR=/Users/dmannk/cisformer/outputs/mojitoo_${JOB_ID}


cd /Users/dmannk/cisformer/CLARE
source ${PWD}/benchmark_vertical/mojitoo/mojitoo_env/bin/activate

## Read the first column of the CSV to get dataset names (excludes MDD)
csv_file='/Users/dmannk/cisformer/data/genes_by_peaks_str.csv'
datasets=($(awk -F',' '{if (NR > 1) print $1}' "$csv_file"))

## Define random state
N=3         # number of jobs to run in parallel
RANDOM=42
random_states=()
for ((i = 0; i < N; i++)); do
    random_states+=($RANDOM)
done

## Outer loop: iterate over datasets as the target_dataset
for target_dataset in "${datasets[@]}"; do
    echo "=== Target dataset: $target_dataset ==="
    
    ## Inner loop: iterate over datasets as the source_dataset
    for source_dataset in "${datasets[@]}"; do

        # Skip the case where source and target datasets are the same
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

            for ((i = 0; i < N; i++)); do

                random_state=$(echo $RANDOM_STATES | cut -d" " -f$((i + 1)))
                export RANDOM_STATE=$random_state
                echo "=== Random state: $random_state ==="
            
                ## Make new sub-sub-directory for source dataset
                mkdir -p $TMPDIR/$target_dataset/$source_dataset/${i}

                python ${PWD}/benchmark_vertical/mojitoo/mojitoo_run.py \
                --outdir $TMPDIR/$target_dataset/$source_dataset/${i} \
                --source_dataset=$source_dataset \
                --target_dataset=$target_dataset \
                --genes_by_peaks_str=$genes_by_peaks_str
            done

        fi
    done
done
 
## Remove write permission from sub-directory and its files to prevent accidental corruption
chmod -R -w $TMPDIR