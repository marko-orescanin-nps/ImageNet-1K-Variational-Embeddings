#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastal_proj_code/orange_cable/vi_flipout/eval_multiple_to_pickle_10_20_cablenoimages_flipout-%j.txt    
#SBATCH --partition=kraken
#SBATCH --mail-type=END
#SBATCH --mail-user=lucian.rombado@nps.edu

#ABOUT:

. /etc/profile

module load lang/miniconda3/23.1.0

source activate vi_flipout

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib





python3 -u /data/kraken/coastal_project/coastal_proj_code/orange_cable/vi_flipout/eval_multiple_runs_to_pickle.py \

