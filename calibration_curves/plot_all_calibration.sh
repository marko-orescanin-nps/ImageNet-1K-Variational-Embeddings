#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastal_uncertainty/calibration_curves/plot_all_calibration_8aug-%j.txt    
#SBATCH --partition=kraken

#ABOUT:

. /etc/profile

module load lang/miniconda3/23.1.0

source activate thesis_work

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib



python3 -u /data/kraken/coastal_project/coastal_uncertainty/calibration_curves/plot_calibration_curves.py \