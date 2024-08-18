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

import tensorflow as tf


def get_pipelined_data(hparams):

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        hparams.train_dir,
        label_mode="categorical",
        shuffle=True,
        batch_size=hparams.batch_size,
        image_size=(hparams.input_shape_x, hparams.input_shape_y),
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        hparams.val_dir,
        label_mode="categorical",
        shuffle=True,
        batch_size=hparams.batch_size,
        image_size=(hparams.input_shape_x, hparams.input_shape_y),
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        hparams.test_dir,
        label_mode="categorical",
        shuffle=True,
        batch_size=1,
        image_size=(hparams.input_shape_x, hparams.input_shape_y),
    )

    return train_dataset, validation_dataset, test_dataset
