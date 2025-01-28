#!/bin/bash -l
#SBATCH --job-name=impute_merged_data
#SBATCH --account=ctb-liyue
#SBATCH --time=0-0:30:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --mem=16G
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=ALL

source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/scalex_env/bin/activate
#source /home/dmannk/projects/def-liyue/dmannk/tmp/envs/torch_env_py39/bin/activate
cd /home/dmannk/projects/def-liyue/dmannk/CLARE

## Impute merged data
n_gpu=1 #$SLURM_GPUS_PER_TASK

## Specify path to data based on modality
modality=$1

if [ $modality == 'atac' ]
then
    echo "ATAC data"
    filepath='/home/dmannk/projects/def-liyue/dmannk/data/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc/atac_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'

    ## Check if pretrain model is provided
    if [ -z "$2" ]
    then  # no model provided
        echo "No SCALE model provided - training new model"
        srun SCALE.py -d $filepath --impute --lr=0.00001 --max_iter=20000 --verbose --min_cells=0 --min_peaks=0 --outdir='/home/dmannk/scratch/' \
            --gpu=$n_gpu
    else # model provided
        echo "SCALE model provided - straight to imputation"
        pretrain=$2
        srun SCALE.py -d $filepath --impute --lr=0.00001 --max_iter=20000 --verbose --min_cells=0 --min_peaks=0 --outdir='/home/dmannk/scratch/' \
            --pretrain=$pretrain --gpu=$n_gpu --batch_size=1024
    fi
elif [ $modality == 'rna' ]
then
    echo "RNA data"
    #filepath='/home/dmannk/projects/def-liyue/dmannk/data/merged_data/roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc/rna_merged_roussos_AD_Anderson_et_al_PD_Adams_et_al_human_dlpfc.h5ad'
    srun python impute_merged_data.py

else
    echo "Invalid modality. Please choose 'atac' or 'rna'."
    exit 1
fi

