#The purpose of this file is to load model predictions and corresponding y labels from the picke file
#and to then run the Bayesian model metrics on this loaded data.

import tensorflow as tf 
import os
import sys
import numpy as np
import pandas as pd
import pickle
from model_metrics import BNN_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import pylab
pylab.rcParams['figure.figsize'] = (12.0, 6.0)
import seaborn as sns
pickle_save_path = '/data/kraken/coastal_project/coastal_alaska/vi_flipout/vi_flipout_alaska_10_20.pickle'



def evaluate_model():
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

    
        preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(8, predict_probs, y_test_labels,'multi_class')
        y_pred = np.argmax(preds_mc, axis=1)
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
    df = pd.DataFrame(acc_cal_combine)
    print("total dataframe: ", df)
    mean = df.mean(axis=0) 
    print("total dataframe mean(): ", mean)
    std = df.std(axis=0) 
    print("total dataframe std(): ", std)

    



evaluate_model()