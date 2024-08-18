#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastaluncertainty/trainers/resnet_deterministic/training_code/logs/test_10jun.txt  
#SBATCH --partition=kraken
#SBATCH --mail-type=END
#SBATCH --mail-user=lucian.rombado@nps.edu

#ABOUT: This script runs resnet50 with 2 additional embedded layer of 80 neurons added. It unfreezes the last 3 layers for fine tuning.

. /etc/profile

module load lang/miniconda3/23.1.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

source activate thesis_work



python3 -u task.py \
--model_dir="/data/kraken/coastal_project/coastaluncertainty/trainers/resnet_deterministic/training_code/models/resnet50_$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="resnet50" \
--test_dir="/data/cs4321/HW1/test" \
--train_dir="/data/cs4321/HW1/train" \
--val_dir="/data/cs4321/HW1/validation" \
--input_shape_x=299 \
--input_shape_y=299 \
--num_classes=8 \
--base_learning_rate=0.001 \
--min_learning_rate=0.00001 \
--num_epochs=1 \
--batch_size=10 \
--eval_metrics="accuracy" \
--loss_type="categorical_crossentropy" \
--optimizer="SGD" \
--trainable="True" \
--output_drop="True" \
--embedded_drop="True" \
--pool_layer="avg2d" \
--train_layers=4 \
--max_hidden_layers=5 \
--min_hidden_layers=0 \
--neurons=80 \
--data_augmentation="True" \
--callback_list="checkpoint, csv_log, early_stopping"
