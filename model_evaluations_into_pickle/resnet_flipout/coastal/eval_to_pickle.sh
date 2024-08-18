#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/coastal/eval_to-pickle_30JUL-%j.txt    
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-3-6


. /etc/profile

module load lang/miniconda3/23.1.0

source activate vi_flipout_clone

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib





python3 -u /data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/coastal/eval_multiple_runs_to_pickle.py \

