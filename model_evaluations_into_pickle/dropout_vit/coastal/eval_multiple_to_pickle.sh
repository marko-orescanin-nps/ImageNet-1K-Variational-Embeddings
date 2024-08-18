#!/bin/bash
#SBATCH --job-name=rombado_ViT_dropout
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --time=20:00:00
#SBATCH --output=/data/kraken/coastal_project/coastal_proj_code/dropout_vit/eval_multiple_to_pickle_10_20_mcropout_3AUG_ckpt28-%j.txt    
#SBATCH --partition=kraken
#SBATCH --mail-type=END
#SBATCH --mail-user=lucian.rombado@nps.edu

#ABOUT:

. /etc/profile

module load lang/miniconda3/23.1.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib



source activate ViT2



python3 -u /data/kraken/coastal_project/coastal_proj_code/dropout_vit/eval_multiple_to_pickle.py \


