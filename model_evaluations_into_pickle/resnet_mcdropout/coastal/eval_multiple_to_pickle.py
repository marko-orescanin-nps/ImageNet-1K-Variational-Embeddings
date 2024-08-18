'''The purpose of this file is the evaluate the loaded model and then save the predcition/y_label pairs to a dictionary
in a pickle dump. This pickle file can then be used to more efficiently load evaluation data to produce plots more quickly. 
'''
import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
ITERATIONS = 10
MC_SAMPLES = 20
pickle_save_path = f'/data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_mcdropout/coastal/pickle_dump/coastal_mcdrop_{ITERATIONS}_{MC_SAMPLES}_19JUL.pickle'


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
        

        #Run 20 model predictions
        for _ in range(MC_SAMPLES):
            pred_list.append(model.predict(test_data, verbose=1))
            
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
    

model = tf.keras.models.load_model('/data/kraken/coastal_project/coastaluncertainty/trainers/resnet_mcdropout/models/mcdropout_best/checkpoint81-0.32.h5')
test_data = tf.keras.utils.image_dataset_from_directory(
    '/data/cs4321/HW1/test',
    label_mode='categorical',
    shuffle=False,
    batch_size=1,
    image_size=(224,224))


evaluate_model_to_pickle(model, test_data)