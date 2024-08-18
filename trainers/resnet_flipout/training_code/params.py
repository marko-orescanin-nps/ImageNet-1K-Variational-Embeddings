# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Created by marko.orescanin@nps.edu on 7/21/20

"""params.py

This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.

"""
#from scipy.constants import golden, pi
import tensorflow as tf
import numpy as np
import argparse
import os
import datetime
#import yaml


def make_argparser():
    parser = argparse.ArgumentParser(
        description="Arguments to run training for GestureRecognitionModeling"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--val_dir", type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet50",
        help="the type of model to use.",
    )
    parser.add_argument(
        "--regression", type=str2bool, default=False, help="is it a regression model"
    )
    parser.add_argument(
        "--input_shape_x",
        type=int,
        default=28,
        help="The input shape of model to use. It adjust the images used to readily connect the model",
    )
    parser.add_argument(
        "--input_shape_y",
        type=int,
        default=28,
        help="The input shape of model to use. It adjust the images used to readily connect the model",
    )
    parser.add_argument(
        "--continue_training",
        type=str2bool,
        default=False,
        help="continue training from a checkpoint",
    )
    parser.add_argument(
        "--trainable",
        type=str2bool,
        default=False,
        help="Set if the transfered model is trainable",
    )
    parser.add_argument(
        "--train_layers",
        type=int,
        default=1,
        help="The number of layers from the output to the input to be trainable",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="path to the checkpoint to continue training",
    )
    parser.add_argument(
        "--predict",
        type=str2bool,
        default=False,
        help="predict from a checkpoint, use checkpoint flag to pass a model",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="the type of model to use. allowed inputs are fully_connected and cnn",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="the type of model to use. allowed inputs are fully_connected and cnn",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="the type of model to use. allowed inputs are fully_connected and cnn",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="specify the optimizer for the model",
    )
    parser.add_argument(
        "--callback_list", type=str, default=None, help="the callbacks to be added"
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="relu",
        help="specify the hidden layer activation function for the model",
    )
    parser.add_argument(
        "--pool_layer",
        type=str,
        default="avg2d",
        help="Select the base model pooling layer: avg2d/flat",
    )
    parser.add_argument(
        "--min_hidden_layers",
        type=int,
        default=0,
        help="specify the minimun number of hidden layer, if min and max number of hidden layers are the same, only 1 serie is generated",
    )
    parser.add_argument(
        "--max_hidden_layers",
        type=int,
        default=0,
        help="specify the maximum number of hidden layer, if min and max number of hidden layers are the same, only 1 serie is generated",
    )
    parser.add_argument(
        "--output_drop", type=str2bool, default=True, help="Add an droput layer before the output layer"
    )
    parser.add_argument(
        "--embedded_drop", type=str2bool, default=True, help="Add an droput layer before the output layer"
    )
    parser.add_argument(
        "--embedded_drop_rate", type=float, default=True, help="Specify the dropout rate used before the output layer"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.05, help="Specify dropoutrate"
    )
    parser.add_argument(
        "--filterwise_dropout", type=str2bool, default=True, help="Filterwise dropout"
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=16,
        help="specify the number of neuron per hidden layer",
    )
    parser.add_argument(
        "--base_learning_rate",
        type=float,
        default=0.001,
        help="specify the base learning rate for the specified optimizer for the model",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.0001,
        help="specify the base learning rate for the specified optimizer for the model",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="categorical_crossentropy",
        help=" loss type: Options [categorical_crossentropy | binary_crossentropy]",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default=None,
        help="a list of the metrics of interest, seperated by commas",
    )
    # multi-gpu training arguments
    parser.add_argument(
        "--mgpu_run", type=str2bool, default=False, help="multi gpu run"
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpu's on the machine, juno is 2",
    )
    # multi-processing arguments
    parser.add_argument(
        "--use_multiprocessing",
        type=str2bool,
        default=True,
        help="specifys weather to use use_multiprocessing in .fit_genrator method ",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="number of CPU's, for my machine 6 workers, for Juno 18",
    )
    # data augmentation arguments
    parser.add_argument(
        "--data_augmentation",
        type=str2bool,
        default=False,
        help="set to True to run data augmentation",
    )
    return parser.parse_args()


# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1"):
        return True
    elif v.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_hparams():
    """any preprocessing, special handling of the hparams object"""

    parser = make_argparser()
    #print(parser)

    return parser


def save_hparams(hparams):
    path_ = os.path.join(hparams.model_dir, "params.txt")
    hparams_ = vars(hparams)
    with open(path_, "w") as f:
        for arg in hparams_:
            print(arg, ":", hparams_[arg])
            f.write(arg + ":" + str(hparams_[arg]) + "\n")

    # path_ = os.path.join(hparams.model_dir, "params.yml")
    # with open(path_, "w") as f:
    #     yaml.dump(
    #         hparams_, f, default_flow_style=False
    #     )  