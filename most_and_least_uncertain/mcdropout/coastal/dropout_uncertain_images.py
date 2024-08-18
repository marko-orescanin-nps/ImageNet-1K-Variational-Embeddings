'''
The purpose of this file is to display the most and least uncertain images.
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
from PIL import Image
pylab.rcParams['figure.figsize'] = (12.0, 6.0)
import seaborn as sns


pickle_save_path_mcdrop_coastal = '/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/trainer/plots/pickle_dump/coastal_mcdrop_1_20_imagetest.pickle'

def get_metrics_from_pickle(pickle_save_path, imgnet_flag=False): 
    acc_cal_combine = []
    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        results_list = pickle.load(f)

    for index, iteration_results in enumerate(results_list, start=1):
        print(f'Grabbing Dictionary # {index}:')
        images = iteration_results['images']
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels']
        if imgnet_flag == True:
            y_test_labels = np.expand_dims(y_test_labels, 1)

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

        acc_cal_noe = []
        one_pct = int(y_true.shape[0] * .01) #this is 8
        pred_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_pred), key = lambda x: x[0], reverse=True)]
        true_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_true), key = lambda x: x[0], reverse=True)]
        images_sort =      [v for _,v in sorted(zip(normal_entropy,images), key = lambda x: x[0], reverse=True)]
       
        #for most uncertain~~~~~~~~~~~~~~~~~~
        # corresponding_true_labels = true_labels_sort[1:11]
        # corresponding_predicted_labels = pred_labels_sort[1:11]
        # images_to_analyze = images_sort[1:11]
        # images_to_analyze = np.array(images_to_analyze)
        # images_to_analyze = images_to_analyze.astype(np.uint8)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #for least uncertain ~~~~~~~~~~~~
        corresponding_true_labels = true_labels_sort[789:]
        corresponding_predicted_labels = pred_labels_sort[789:]
        images_to_analyze = images_sort[789:]
        images_to_analyze = np.array(images_to_analyze)
        images_to_analyze = images_to_analyze.astype(np.uint8)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for p in range(100):
            tl = true_labels_sort[p*one_pct:]
            pl = pred_labels_sort[p*one_pct:]
            accuracy=accuracy_score(tl,pl)
            acc_cal_noe.append(accuracy)

        acc_cal_noe.reverse()
        acc_cal_combine.append(acc_cal_noe)
        
    df = pd.DataFrame(acc_cal_combine)
    print("total dataframe: ", df)
    mean = df.mean(axis=0) 
    print("total dataframe mean(): ", mean)
    std = df.std(axis=0) 
    print("total dataframe std(): ", std)

    index = mean.index
    print("index: ", index)

    error = std 
    labels_arr=np.array(["CoastalCliffs", "CoastalRocky", "CoastalWaterWay", "Dunes", "ManMadeStructures",
                         "SaltMarshes", "SandyBeaches","TidalFlats"])
    fig, axes = plt.subplots(2, 5, figsize=(12,8))
    for i in range(2):  # Rows
        for j in range(5):  # Columns
            axes[i, j].imshow(images_to_analyze[i * 5 + j])
            axes[i, j].set_xlabel(f'True: {labels_arr[corresponding_true_labels[i * 5 + j]]} \n Pred: {labels_arr[corresponding_predicted_labels[i * 5 + j]]}',rotation=0)
            axes[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.1, hspace=0.0005)

    plt.suptitle("10 Least Uncertain Coastal Images (total entropy)", fontsize=30)
    plt.savefig(f"/data/kraken/coastal_project/coastaluncertainty/analyze_uncertainty/mcdropout/coastal/22may_again_least_uncertain_images.png")
    plt.tight_layout()



    return index, mean, error


index_mcdrop, mean_mcdrop, error_mcdrop = get_metrics_from_pickle(pickle_save_path_mcdrop_coastal)
