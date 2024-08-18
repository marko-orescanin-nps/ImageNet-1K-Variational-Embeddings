'''
The purpose of this file is to produce Classification Reports for the various models with 100% of the data and with 80% (after the most uncertain 20% is removed).
Model predictions are loaded in from pickle files.
'''
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
import math
from sklearn.metrics import classification_report

#Coastal Pickle Files
pickle_save_path_mcdrop_coastal = '/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/trainer/plots/pickle_dump/mcdropout_10_20.pickle'
pickle_save_path_viflipout_coastal = '/data/kraken/coastal_project/coastal_proj_code/resnet_VI_flipout/trainer/plots/pickle_dump/vi_flipout_10_20.pickle'
# pickle_save_path_deterministic_coastal = '/data/kraken/coastal_project/coastal_proj_code/resnet_deterministic/trainer/plots/pickle_dump/deterministic_10_20.pickle'
# pickle_save_path_ViT_coastal = '/data/kraken/coastal_project/coastal_proj_code/dropout_vit/pickle_dump/vit_dropout_good_ckpt_10_20.pickle'

# #ImageNet Pickle Files 
# pickle_save_path_mcdrop_imagenet = '/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/imagenet/pickle_dump/imagenet_mcdrop_multiple_10_20.pickle'
# pickle_save_path_viflipout_imagenet = '/data/kraken/coastal_project/coastal_proj_code/resnet_VI_flipout/imagenet/pickle_dump/imagenet_flipout_10_20.pickle'
# pickle_save_path_deterministic_imagenet = '/data/kraken/coastal_project/coastal_proj_code/resnet_deterministic/imagenet/pickle_dump/imagenet_deterministic_dropoff_10_20.pickle'
# imagenet_pickle_list = [pickle_save_path_mcdrop_imagenet, pickle_save_path_viflipout_imagenet, pickle_save_path_deterministic_imagenet]

def get_metrics_from_pickle(pickle_save_path, imgnet_flag):
    acc_cal_combine = []
    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        results_list = pickle.load(f)

    for index, iteration_results in enumerate(results_list, start=1):
        print(f'Grabbing Dictionary # {index}:')
        
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels']
        if imgnet_flag == True:
            y_test_labels = np.expand_dims(y_test_labels, 1)
        if imgnet_flag:
            preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(1000, predict_probs, y_test_labels,'multi_class')
        else:
            preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(8, predict_probs, y_test_labels,'multi_class')

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
        print('CLASSIFICATION REPORT:\n', classification_report(y_true, y_pred))
       
        acc_cal_noe = []
        one_pct = int(y_true.shape[0] * .01) 
        pred_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_pred), key = lambda x: x[0], reverse=True)]
        true_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_true), key = lambda x: x[0], reverse=True)]
        
        for p in range(100):
            tl = true_labels_sort[p*one_pct:]
            pl = pred_labels_sort[p*one_pct:]
            accuracy=accuracy_score(tl,pl)
            acc_cal_noe.append(accuracy)


        acc_cal_noe.reverse()
        acc_cal_combine.append(acc_cal_noe)


        cutoff_index = int(0.2 * len(normal_entropy))
        remaining_indices = np.argsort(normal_entropy)[:-cutoff_index]

        y_true_remaining = y_true[remaining_indices]
        y_pred_remaining = y_pred[remaining_indices]


        print('CLASSIFICATION REPORT AFTER REMOVING 20% MOST UNCERTAIN:\n', classification_report(y_true_remaining, y_pred_remaining))
        print('picke data used: ', pickle_save_path)
        
        
    df = pd.DataFrame(acc_cal_combine)
    mean = df.mean(axis=0) 
    std = df.std(axis=0) 
    index = mean.index

    ci = 1.96 * df.std(axis=0) / math.sqrt(10)
    error = ci
    

    return index, mean, error

# index_mcdrop_coastal, mean_mcdrop_coastal, error_mcdrop_coastal = get_metrics_from_pickle(pickle_save_path_mcdrop_coastal, imgnet_flag=False)
index_viflipout_coastal, mean_viflipout_coastal, error_viflipout_coastal = get_metrics_from_pickle(pickle_save_path_viflipout_coastal, imgnet_flag=False)
