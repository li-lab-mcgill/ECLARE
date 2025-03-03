import os
import subprocess
import datetime
import shutil
import csv
import argparse
from pathlib import Path
from scripts.eclare_scripts.eclare_run import eclare_run  # Import the function from clip_run.py
import yaml  # Add this import at the top of the file
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='ECLARE command line tool')
    parser.add_argument('command', type=str, choices=['eclare', 'clip', 'kd_clip'],
                        help='Command to execute')
    parser.add_argument('--outdir', type=str, default=os.environ.get('OUTPATH', None),
                        help='output directory')
    parser.add_argument('--clip_job_id', type=str, default=None,
                        help='Job ID of CLIP training')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--loss_type', type=str, default='knowledge_distillation',
                        help='type of loss to use for training')
    parser.add_argument('--train_encoders', action='store_true',
                        help='train the encoders during training (name starting with letter f returns error)')
    parser.add_argument('--loop_order', type=str, default='batches_first',
                        help='order of loops in training')
    parser.add_argument('--save_latents', action='store_true',
                        help='save latents during training')
    parser.add_argument('--genes_by_peaks_str', type=str, default='10112_by_56354', ## aligned with MDD data
                        help='genes by peaks string')
    parser.add_argument('--source_dataset_embedder', action='store_true', default=False,
                        help='use a dataset embedder')
    parser.add_argument('--distil_lambda', type=float, default=0.1,
                        help='lambda value for MobileCLIP loss')
    parser.add_argument('--valid_subsample', type=int, default=5000,
                        help='number of nuclei to subsample for validation')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='source dataset')
    parser.add_argument('--target_dataset', type=str, default=None,
                        help='target dataset')
    parser.add_argument('--replicate_idx', type=int, default=0,
                        help='replicate index')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='Path to the YAML file containing environment variables')
    return parser

def create_job_id():
    return datetime.datetime.now().strftime("%d%H%M%S")

def set_environment_variables(yaml_path):
    with open(yaml_path, 'r') as file:
        env_vars = yaml.safe_load(file)
    
    if not env_vars:
        raise ValueError("No environment variables found in the YAML file.")
    
    for key, value in env_vars.items():
        os.environ[key] = value

def create_temp_directory(outpath, job_id):
    tmpdir = Path(outpath) / f'clip_{job_id}'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    return tmpdir

def read_datasets(csv_file):
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        return [row[0] for row in reader]

def run_eclare(args):
    # Logic for running eclare
    print("Running eclare...")
    job_id = create_job_id()
    print(f"Job ID: clip_{job_id}")

    eclare_root = os.environ['ECLARE_ROOT']
    outpath = os.environ['OUTPATH']
    datapath = os.environ['DATAPATH']

    tmpdir = create_temp_directory(outpath, job_id)
    log_file = tmpdir / 'log.txt'

    csv_file = Path(datapath) / 'genes_by_peaks_str_samples.csv'
    source_datasets = read_datasets(csv_file)
    target_datasets = ["PFC_Zhu"]

    print(f"Source datasets: {source_datasets}")
    print(f"Target datasets: {target_datasets}")

    N = 1
    random_states = [42] * N  # Fixed random state for reproducibility

    total_epochs = 10  # Example value, replace with actual

    for target_dataset in target_datasets:
        print(f"=== Target dataset: {target_dataset} ===")
        for source_dataset in source_datasets:
            feature = f"Align nuclei from {source_dataset} data to {target_dataset} data."
            print(f"~~ {feature} ~~")

            genes_by_peaks_str = subprocess.check_output(
                f"awk -F',' -v source='{source_dataset}' -v target='{target_dataset}' 'NR == 1 {{for (i = 1; i <= NF; i++) {{if ($i == target) target_idx = i}}}} $1 == source {{split($target_idx, arr, \",\"); print arr[2]}}' {csv_file}",
                shell=True
            ).decode().strip()

            if not genes_by_peaks_str:
                print(f"Warning: No value found for source={source_dataset}, target={target_dataset}")
                continue

            print(f"Genes by peaks string: {genes_by_peaks_str}")

            for i in range(N):
                random_state = random_states[i]
                os.environ['RANDOM_STATE'] = str(random_state)
                print(f"=== Random state: {random_state} ===")

                output_dir = tmpdir / target_dataset / source_dataset / str(i)
                output_dir.mkdir(parents=True, exist_ok=True)

                eclare_run(
                    outdir=str(output_dir),
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    genes_by_peaks_str=genes_by_peaks_str,
                    total_epochs=total_epochs
                )

    # Remove write permission from sub-directory and its files to prevent accidental corruption
    for root, dirs, files in os.walk(tmpdir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)

    print("Job complete.")

def run_clip(args):
    # Logic for running clip
    print("Running clip...")
    # ... existing code for clip ...

def run_kd_clip(args):
    # Logic for running kd_clip
    print("Running kd_clip...")
    # ... existing code for kd_clip ...

def main():
    args = get_parser().parse_args()
    set_environment_variables(args.yaml_path)

    if args.command == 'eclare':
        run_eclare(args)
    elif args.command == 'clip':
        run_clip(args)
    elif args.command == 'kd_clip':
        run_kd_clip(args)
    else:
        print(f"Unknown command: {args.command}")

    print("Job complete.")

if __name__ == "__main__":
    main()