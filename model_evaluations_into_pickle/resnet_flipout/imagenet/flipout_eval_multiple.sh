#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --time=48:00:00
#SBATCH --output=/data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_flipout/imagenet/flipout_eval_19JUL-%j.txt    
#SBATCH --partition=kraken

. /etc/profile

module load lang/miniconda3/23.1.0

source activate vi_flipout

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib



python3 -u /data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_flipout/imagenet/flipout_eval_multiple_to_pickle.py \



