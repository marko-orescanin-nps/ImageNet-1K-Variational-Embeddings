#The purpose of this file is the evaluate the loaded model and then save the predcition/y_label pairs to a dictionary
#in a pickle dump. This pickle file can then be used to more efficiently load evaluation data to produce plots more quickly. 

import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
from resnet50VI_ub import resnet50_variational, BatchNormalization, Conv2DFlipout
from model_metrics import BNN_predict
from model_metrics import multiclass_metrics
from sklearn.metrics import accuracy_score

ITERATIONS = 10
MC_SAMPLES = 20
pickle_save_path = f'/data/kraken/coastal_project/coastal_alaska/vi_flipout/vi_flipout_alaska_{ITERATIONS}_{MC_SAMPLES}.pickle'



def evaluate_model_to_pickle(model, test_data):
    print(f"Performing {ITERATIONS} rounds of {MC_SAMPLES} Monte Carlo samples.")

    print('-'*30)
    print('EVALUATING MODEL: ', model)

    results_list = []

    #Run this whole process 10 times 
    for i in range(ITERATIONS):
        print(f'Iteration {i + 1}')

        pred_list = []
        y_test = []
        y_test = np.concatenate([y for x,y in test_data], axis=0)
        class_labels = ["CoastalCliffs", "CoastalRocky", "CoastalWaterWay", "Dunes", "ManMadeStructures",
                            "SaltMarshes", "SandyBeaches","TidalFlats"]     

        #Run MC Samples
        for _ in range(MC_SAMPLES):
            pred_list.append(model.predict(test_data, verbose=1))
            
        predict_probs = np.stack(pred_list, axis=0)

        iteration_results = {
            'preds': predict_probs,
            'trueLabels': y_test
        }
        
        results_list.append(iteration_results)  # Append results to the list
    if not os.path.exists(pickle_save_path):
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(results_list, f) 





IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
NUM_CLASSES = 1000
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
NEURONS = 1024
ACTIVATION_FN = 'relu'

base_model = resnet50_variational(
          input_shape=(224,224,3),
          num_classes=NUM_CLASSES, 
          prior_stddev=None,
          dataset_size=APPROX_IMAGENET_TRAIN_IMAGES,
          stddev_mean_init=0.00097618,
          stddev_stddev_init=0.41994,
          tied_mean_prior=True,)

new_model = tf.keras.models.Model(inputs = base_model.layers[0].input, outputs = base_model.layers[-2].output) 

#Dense
embed_layer_1 = tf.keras.layers.Dense(units=NEURONS, activation=ACTIVATION_FN)(new_model.output)
#Dense
embed_layer_2 = tf.keras.layers.Dense(units=NEURONS, activation=ACTIVATION_FN)(embed_layer_1)

classification_head = tf.keras.layers.Dense(8, activation='softmax', name="class_head")(embed_layer_2)

built_model = tf.keras.models.Model(inputs=new_model.input, outputs=classification_head)


checkpoint = tf.train.Checkpoint(model=built_model)  

#Restore the trained VI Flipout model
checkpoint.restore('/data/kraken/coastal_project/coastal_proj_code/resnet_VI_flipout/models/resnet50_vi_flipout_500_properweights_2023-10-09_20-51-02/checkpoint77')

model = tf.keras.models.Model(inputs = checkpoint.model.layers[0].input, outputs = checkpoint.model.layers[-1].output) 
model.load_weights('/data/kraken/coastal_project/coastal_proj_code/resnet_VI_flipout/models/resnet50_vi_flipout_500_properweights_2023-10-09_20-51-02/checkpoint77')

test_data = tf.keras.utils.image_dataset_from_directory(
    '/data/kraken/coastal_project/coastal_alaska/coastal_alaska_resized',
    label_mode='categorical',
    labels="inferred",
    shuffle=False,
    batch_size=1,
    image_size=(224,224))


evaluate_model_to_pickle(model, test_data)