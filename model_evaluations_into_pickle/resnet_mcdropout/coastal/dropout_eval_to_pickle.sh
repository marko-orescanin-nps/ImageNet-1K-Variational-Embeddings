#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_mcdropout/coastal/droppout_eval_to-pickle_18JUL-%j.txt    
#SBATCH --partition=kraken


. /etc/profile

module load lang/miniconda3/23.1.0

source activate vi_flipout

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib





python3 -u /data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_mcdropout/coastal/eval_multiple_to_pickle.py \

