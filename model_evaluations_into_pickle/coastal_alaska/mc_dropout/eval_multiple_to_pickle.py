#The purpose of this file is the evaluate the loaded model and then save the predcition/y_label pairs to a dictionary
#in a pickle dump. This pickle file can then be used to more efficiently load evaluation data to produce plots more quickly. 

import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
ITERATIONS = 1
MC_SAMPLES = 2
pickle_save_path = f'/data/kraken/coastal_project/coastal_alaska/alaska_mcdrop_{ITERATIONS}_{MC_SAMPLES}_images.pickle'


def evaluate_model_to_pickle(model, test_data):
    print(f"Performing {ITERATIONS} rounds of {MC_SAMPLES} Monte Carlo samples.")
    print('-'*30)
    print('EVALUATING MODEL: ', model)
    results_list = []
    #Run this whole process 10 times 
    for i in range(ITERATIONS):
        print(f'Iteration {i + 1}')
        images = []
        pred_list = []
        y_test = []
        y_test = np.concatenate([y for x,y in test_data], axis=0)
        images = np.concatenate([x for x,y in test_data], axis=0)
        class_labels = ["CoastalCliffs", "CoastalRocky", "CoastalWaterWay", "Dunes", "ManMadeStructures",
                            "SaltMarshes", "SandyBeaches","TidalFlats"]       
        

        #Run 50 model predictions
        for _ in range(MC_SAMPLES):
            pred_list.append(model.predict(test_data, verbose=1))
            
        predict_probs = np.stack(pred_list, axis=0)

        
        iteration_results = {
        'images': images,
        'preds': predict_probs,
        'trueLabels': y_test
        }
        results_list.append(iteration_results)  # Append results to the list

    if not os.path.exists(pickle_save_path):
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(results_list, f)
    




model = tf.keras.models.load_model('/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/models/resnet50_mcdropout_unfreezeFront15_01SEPT_2023-09-01_11-16-10_BEST_01SEPT/checkpoint81-0.32.h5')
test_data = tf.keras.utils.image_dataset_from_directory(
    '/data/kraken/coastal_project/coastal_proj_code/coastal_alaska/coastal_alaska_resized',
    label_mode='categorical',
    labels = "inferred",
    shuffle=False,
    batch_size=1,
    image_size=(224,224))


evaluate_model_to_pickle(model, test_data)