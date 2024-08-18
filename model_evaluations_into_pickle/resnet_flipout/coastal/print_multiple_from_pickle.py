'''The purpose of this file is to load model from Pickle file to get predictions and corresponding y labels
and to then run the Bayesian model metrics on this loaded data and print out the metrics.'''

import tensorflow as tf 
import numpy as np
import pandas as pd
import pickle
from model_metrics import BNN_predict
from sklearn.metrics import accuracy_score
import pylab
pylab.rcParams['figure.figsize'] = (12.0, 6.0)
import seaborn as sns
from resnet50VI_ub import resnet50_variational
from model_metrics import BNN_predict
from sklearn.metrics import accuracy_score
pickle_save_path = '/data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_flipout/coastal/pickle_dump/vi_flipout_10_20_test11jun.pickle'



def evaluate_model(model, test_data):
    acc_cal_combine = []
    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        #Grab the list of dictionaries from the pickle file
        results_list = pickle.load(f)
    aleatoric_list = []
    epistemic_list = []
    total_entropy_list = []
    
    for index, iteration_results in enumerate(results_list, start=1):
        print(f'Grabbing Dictionary # {index}:')
        
        # Access 'preds' and 'trueLabels' from the current dictionary
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels']

        preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(8, predict_probs, y_test_labels,'multi_class')

        y_pred = np.argmax(preds_mc, axis=1)
        y_true = np.argmax(y_test_labels, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        print('ACCURACY:', accuracy)
        # print('NLL:', nll_mc)
        print('VARIANCE', np.mean(var))
        print('TOTAL_ENTROPY:', np.mean(normal_entropy))
        print('ALEATORIC UNCERTAINTY (depeweg):', np.mean(aleatoric_depeweg))
        print('EPISTEMIC UNCERTAINTY (depeweg):', np.mean(epistemic_depeweg))

        aleatoric_list.append(np.mean(aleatoric_depeweg))
        epistemic_list.append(np.mean(epistemic_depeweg))
        total_entropy_list.append(np.mean(entropy_mc))
    print("Aleatoric List: ", aleatoric_list)
    print("Epistemic List: ", epistemic_list)
    print("Total Entropy List: ", total_entropy_list)

        
    df = pd.DataFrame(acc_cal_combine)
    print("total dataframe: ", df)
    mean = df.mean(axis=0) 
    print("total dataframe mean(): ", mean)
    std = df.std(axis=0) 
    print("total dataframe std(): ", std)

    


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
    '/data/cs4321/HW1/test',
    label_mode='categorical',
    shuffle=False,
    batch_size=1,
    image_size=(224,224))



evaluate_model(model, test_data)