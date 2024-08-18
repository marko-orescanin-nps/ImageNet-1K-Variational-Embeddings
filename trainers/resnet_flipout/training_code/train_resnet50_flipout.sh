#!/bin/bash
#SBATCH --job-name=rombado_resnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/data/kraken/coastal_project/coastaluncertainty/trainers/resnet_flipout/logs/test_flipout_18jul.txt  
#SBATCH --partition=kraken


. /etc/profile

module load lang/miniconda3/23.1.0

source activate vi_flipout_clone



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib





python3 -u /data/kraken/coastal_project/coastaluncertainty/trainers/resnet_flipout/training_code/task.py \
--model_dir="/data/kraken/coastal_project/coastaluncertainty/trainers/resnet_flipout/models/resnet50_vi_flipout_$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="resnet50_flipout" \
--test_dir="/data/cs4321/HW1/test" \
--train_dir="/data/cs4321/HW1/train" \
--val_dir="/data/cs4321/HW1/validation" \
--input_shape_x=224 \
--input_shape_y=224 \
--num_classes=8 \
--base_learning_rate=0.001 \
--min_learning_rate=0.00001 \
--num_epochs=500 \
--batch_size=10 \
--eval_metrics="accuracy" \
--activation_fn="relu" \
--loss_type="categorical_crossentropy" \
--optimizer="adam" \
--trainable="True" \
--output_drop="True" \
--embedded_drop="True" \
--dropout_rate=0.05 \
--embedded_drop_rate=0.20 \
--filterwise_dropout="True" \
--pool_layer="avg2d" \
--train_layers=20 \
--max_hidden_layers=5 \
--min_hidden_layers=0 \
--neurons=1024 \
--data_augmentation="True" \
--callback_list="checkpoint, csv_log, early_stopping" \
--checkpoint_path="/data/kraken/coastal_project/coastaluncertainty/imagenet_checkpoints/vi_flipout/checkpoint-29"
