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
import os
import shutil

import tensorflow as tf
import params
import models
import optimizers
import pickle
import callbacks
import data_class
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from keras.models import load_model

def main():

    hparams = params.get_hparams()
    #Make a directory to store model artifacts 

    if not os.path.exists(hparams.model_dir):
        os.mkdir(hparams.model_dir)

    params.save_hparams(hparams)

    
    print("-----------HPARAMS------------")
    print(hparams)
    print("-----------HPARAMS------------")

    # Import dataset 
    train_data, val_data, test_data = data_class.get_pipelined_data(hparams)

    model, base_model = models.create_model(hparams)

    model.compile(
        optimizer=optimizers.get_optimizer(hparams),
        loss=hparams.loss_type,
        metrics=[hparams.eval_metrics],
    )

    print("Model summary before fine tuning: " + str(model.summary()))

    model.fit(
            train_data,
            epochs=hparams.num_epochs,
            validation_data=val_data,
            callbacks=callbacks.make_callbacks(hparams),
        )
    
    print("-----------EVALUATION BEFORE FINE TUNING------------")
    
    model.evaluate(test_data)

    if hparams.trainable:
        print("")
        print("")
        print("FINE TUNING TRAINING")
        print("------------------------------------------")

       
        #Unfreeze layers for fine tuning
        models.unfreeze_layers(model, base_model, hparams.train_layers)
        print(model.summary())
       
        #Reduce learning rate for fine tuning
        hparams.base_learning_rate = hparams.base_learning_rate / 10.0

        print("")
        print("Current learning rate: ", hparams.base_learning_rate)
        print("")

        #Recompile model, this updates the new learning rate 
        model.compile(
                        optimizer=optimizers.get_optimizer(hparams),
                        loss=hparams.loss_type,
                        metrics=[hparams.eval_metrics],
                    )
        #Train with the fine tuning layers
        model.fit(
                    train_data,
                    epochs=hparams.num_epochs,
                    validation_data=val_data,
                    callbacks=callbacks.make_callbacks(hparams),
                    )
        #Evaluate results

        print("-----------EVALUATION AFTER FINE TUNING------------")

        model.evaluate(test_data)



if __name__ == "__main__":
    main()