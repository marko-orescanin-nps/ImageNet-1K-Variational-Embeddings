#The purpose of this file is to load model predictions and corresponding y labels from the pickle file
#and to then run the Bayesian model metrics on this loaded data.

import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
import gc 
from model_metrics import BNN_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import pylab
from resnet50VI_ub import *
from PIL import Image
pylab.rcParams['figure.figsize'] = (12.0, 6.0)
import seaborn as sns

ITERATIONS = 10
MC_SAMPLES = 20


pickle_save_path_mcdrop_imagenet = f'/data/kraken/coastal_project/coastaluncertainty/model_evaluations_into_pickle/resnet_flipout/imagenet/pickle_dump/imagenet_flipout_11JUNE_10_20.pickle'

def get_metrics_from_pickle(pickle_save_path, imgnet_flag):

    acc_cal_combine = []

    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        results_list = pickle.load(f)
    aleatoric_list = []
    epistemic_list = []
    total_entropy_list = []

    for index, iteration_results in enumerate(results_list, start=1):
        print(f'Grabbing Dictionary # {index}:')
        
        # Access 'preds' and 'trueLabels' from the current dictionary
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels']
        if imgnet_flag == True:
            y_test_labels = np.expand_dims(y_test_labels, 1)
    
        preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(1000, predict_probs, y_test_labels,'multi_class')


        y_pred = np.argmax(preds_mc, axis=1)
        if imgnet_flag == True:
            
            y_true = y_test_labels
        else:
            y_true = np.argmax(y_test_labels, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        print('ACCURACY:', accuracy)
        print('NLL:', nll_mc)
        print('VARIANCE', np.mean(var))
        print('NORMAL_ENTROPY:', np.mean(normal_entropy))
        print('ALEATORIC UNCERTAINTY:', np.mean(aleatoric_depeweg))
        print('EPISTEMIC UNCERTAINTY:', np.mean(epistemic_depeweg))
        aleatoric_list.append(np.mean(aleatoric_depeweg))
        epistemic_list.append(np.mean(epistemic_depeweg))
        total_entropy_list.append(np.mean(entropy_mc))
    print("Aleatoric List: ", aleatoric_list)
    print("Epistemic List: ", epistemic_list)
    print("Total Entropy List: ", total_entropy_list)


    
get_metrics_from_pickle(pickle_save_path_mcdrop_imagenet, imgnet_flag=True)