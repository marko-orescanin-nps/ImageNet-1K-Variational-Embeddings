#The purpose of this file is the evaluate the loaded model and then save the predcition/y_label pairs to a dictionary
#in a pickle dump. This pickle file can then be used to more efficiently load evaluation data to produce plots more quickly. 

import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
import cv2 
import matplotlib.gridspec as gridspec
import params
from vit_keras import vit
import tensorflow_addons as tfa
ITERATIONS = 10
MC_SAMPLES = 20
pickle_save_path = f'/data/kraken/coastal_project/coastal_proj_code/dropout_vit/pickle_dump/vit_MCDropout_{ITERATIONS}_{MC_SAMPLES}_3AUG.pickle'


def evaluate_model_to_pickle(model, test_data):
    print(f"Performing {ITERATIONS} rounds of {MC_SAMPLES} Monte Carlo samples.")
    print('-'*30)
    print('EVALUATING MODEL: ', model)
    results_list = []
    #Run this whole process 10 times 
    for i in range(ITERATIONS):
        print(f'Iteration {i + 1}')
        # images = []
        pred_list = []
        y_test = []
        y_test = np.concatenate([y for x,y in test_data], axis=0)
        # images = np.concatenate([x for x,y in test_data], axis=0)
        class_labels = ["CoastalCliffs", "CoastalRocky", "CoastalWaterWay", "Dunes", "ManMadeStructures",
                            "SaltMarshes", "SandyBeaches","TidalFlats"]       
        

        #Run 20 model predictions
        for _ in range(MC_SAMPLES):
            pred_list.append(model.predict(test_data))
            
        predict_probs = np.stack(pred_list, axis=0)

        
        iteration_results = {
        # 'images': images,
        'preds': predict_probs,
        'trueLabels': y_test
        }
        results_list.append(iteration_results)  # Append results to the list

    if not os.path.exists(pickle_save_path):
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(results_list, f)
    




# model = tf.keras.models.load_model('/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/models/resnet50_mcdropout_unfreezeFront15_01SEPT_2023-09-01_11-16-10_BEST_01SEPT/checkpoint81-0.32.h5')
test_data = tf.keras.utils.image_dataset_from_directory(
    '/data/cs4321/HW1/test',
    label_mode='categorical',
    shuffle=False,
    batch_size=1,
    image_size=(224,224))


vit_base_model = vit.vit_b16(
        image_size = 224,
        weights = "imagenet21k",
        pretrained = True,
        include_top = False,
        pretrained_top = False)
    
# preprocess_input = vit.preprocess_inputs
input_layer = tf.keras.Input(shape=(224,224, 3))
x = tf.keras.layers.RandomFlip("horizontal")(input_layer)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomBrightness(factor=0.2)(x)
x = tf.keras.layers.RandomContrast(factor=0.2)(x)
x = tf.keras.layers.Lambda(tf.keras.applications.imagenet_utils.preprocess_input, arguments={'data_format': None, 'mode': 'tf'})(x)
x = vit_base_model(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation=tfa.activations.gelu)(x)
x = tf.keras.layers.Dropout(0.2, name='embed_dropout_1')(x, training = True)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, activation=tfa.activations.gelu)(x)
x = tf.keras.layers.Dropout(0.2, name='embed_dropout_2')(x, training = True)
x = tf.keras.layers.Dense(32, activation=tfa.activations.gelu)(x)
x = tf.keras.layers.Dropout(0.2, name='embed_dropout_3')(x, training=True)
output_layer = tf.keras.layers.Dense(8, activation='softmax', name='classification_head')(x)

built_model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name = 'vision_transformer')
built_model.summary()

checkpoint = tf.train.Checkpoint(model=built_model)  




#Restore the trained VI Flipout model
checkpoint.restore('/data/kraken/coastal_project/coastal_proj_code/dropout_vit/models/vit_dropout_2024-08-02_14-34-23/checkpoint28')



model_mcdropout = tf.keras.models.Model(inputs = checkpoint.model.layers[0].input, outputs = checkpoint.model.layers[-1].output) 
# model_mcdropout = tf.keras.models.Model(inputs = checkpoint.model.layers[0].input, outputs = checkpoint.model.layers.output[-1]) 
model_mcdropout.load_weights('/data/kraken/coastal_project/coastal_proj_code/dropout_vit/models/vit_dropout_2024-08-02_14-34-23/checkpoint28')



evaluate_model_to_pickle(model_mcdropout, test_data)