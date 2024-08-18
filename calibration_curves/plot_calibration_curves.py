#The purpose of this file is to take in pickle data from ResNet50 deterministic, ResNet50 MC Dropout, and ResNet50 VI Flipout 
#and plot the three calibration curves on a single plot. 

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

#Coastal Pickle Files

pickle_save_path_mcdrop_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_mcdropout/coastal/pickle_dump/mcdropout_10_20.pickle'

pickle_save_path_viflipout_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/coastal/pickle_dump/vi_flipout_10_20.pickle'

pickle_save_path_deterministic_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_deterministic/coastal/pickle_dump/coastal_resnet_deterministic_10_20.pickle'

pickle_save_path_ViT_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/dropout_vit/coastal/pickle_dump/vit_MCDropout_10_20_3AUG.pickle'
#'/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/dropout_vit/pickle_dump/vit_dropout_good_ckpt_10_20.pickle'

#ImageNet Pickle Files 
pickle_save_path_mcdrop_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_mcdropout/imagenet/pickle_dump/imagenet_mcdrop_multiple_10_20.pickle'

pickle_save_path_viflipout_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/imagenet/pickle_dump/imagenet_flipout_10_20.pickle'

pickle_save_path_deterministic_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_deterministic/imagenet/pickle_dump/imagenet_deterministic_10_20.pickle'


imagenet_pickle_list = [pickle_save_path_mcdrop_imagenet, pickle_save_path_viflipout_imagenet, pickle_save_path_deterministic_imagenet]

def get_metrics_from_pickle(pickle_save_path, imgnet_flag, name):
    acc_cal_combine = []
    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        results_list = pickle.load(f)
    mean_of_MNLLs = []
    mnll_list = []
    

    for index, iteration_results in enumerate(results_list, start=1):
        print(f'Grabbing Dictionary # {index}:')
        
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels'] #for imagenet, these are indexes [22,54,etc], for coastal its one hot
        # if imgnet_flag == True:
        #     y_test_labels = np.expand_dims(y_test_labels, 1)


        if imgnet_flag:
            y_test_labels_copy = y_test_labels.astype(int)
            y_test_labels_onehot = np.zeros((y_test_labels_copy.size, y_test_labels_copy.max()+1), dtype=int)
            y_test_labels_onehot[np.arange(y_test_labels_copy.size),y_test_labels_copy] = 1
            preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(1000, predict_probs, y_test_labels_onehot,'multi_class')
        else:
            preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(8, predict_probs, y_test_labels,'multi_class')
        
        y_pred = np.argmax(preds_mc, axis=1)
        if imgnet_flag == True:
            y_true = y_test_labels
        else:
            y_true = np.argmax(y_test_labels, axis=1)
            
        accuracy = accuracy_score(y_true, y_pred)
        
        # print('ACCURACY:', accuracy)
        # print('NLL:', nll_mc)
        # print('VARIANCE', np.mean(var))
        # print('NORMAL_ENTROPY:', np.mean(normal_entropy))
        # print('ALEATORIC UNCERTAINTY:', np.mean(aleatoric_depeweg))
        # print('EPISTEMIC UNCERTAINTY:', np.mean(epistemic_depeweg))

        acc_cal_noe = []
        one_pct = int(y_true.shape[0] * .01) #this is 8

        #this results in more incorrect first --> most correct at the end 
        #high to low entropy
        pred_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_pred), key = lambda x: x[0], reverse=True)]
        true_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_true), key = lambda x: x[0], reverse=True)]
        
        for p in range(100):
            tl = true_labels_sort[p*one_pct:]
            pl = pred_labels_sort[p*one_pct:]
            accuracy=accuracy_score(tl,pl)
            acc_cal_noe.append(accuracy)


        acc_cal_noe.reverse()
        #this is the list of lists
        acc_cal_combine.append(acc_cal_noe)

        #Calculate MNLL across the 20 predictions, ignore NaNs
        mnll = np.nanmean(nll_mc)

        #Append the MCLL across the 20 predictions to the MNLL list
        mnll_list.append(mnll)
    
    mnlls_df = pd.DataFrame(mnll_list)
    mnlls_mean = mnlls_df.mean(axis=0)
    mnlls_std = mnlls_df.std(axis=0)
    mnlls_ci = 1.96 * mnlls_std / math.sqrt(10)
    print(f"*** MNLL Mean across 10 rounds of 20 for {name}: {mnlls_mean}")
    print(f"*** MNLL CI across 10 rounds of 20 for {name}: {mnlls_ci}")






        
    df = pd.DataFrame(acc_cal_combine)
    print("total dataframe: ", df)
    mean = df.mean(axis=0) 
    print("total dataframe mean(): ", mean)
    std = df.std(axis=0) 
    print("picke file being evaluated: ", pickle_save_path)
    print("total dataframe std(): ", std)

    index = mean.index
    print("index: ", index)

    ci = 1.96 * std / math.sqrt(10)
    error = ci
    print("error: ", error)

    return index, mean, error

def plot_metrics(index, mean, error, label, imgnet_flag):
    lower_plot = mean - error 
    upper_plot = mean + error 

    if imgnet_flag:
        ax.plot(index, mean, marker='s', markevery=10,label=label)
    else:
        ax.plot(index, mean, marker='o', markevery=10,label=label)


    ax.plot(index, lower_plot, color='tab:blue', alpha=0.1)
    ax.plot(index, upper_plot, color='tab:blue', alpha=0.1)
    ax.fill_between(index, lower_plot, upper_plot, alpha=0.2)
    ax.set_xlabel('Retained Data (Sorted by Increasing Normal Entropy)', fontsize = 18)
    ax.set_ylabel('Accuracy', fontsize = 18)     
    ax.grid()
    ax.legend() 
    plt.xlim([50,100])
    plt.title("Top-1 Accuracy (10 runs of 20 Monte Carlo Samples)", fontsize=18)

    
    

#Get the information from the coastal pickle files 
print("hi")
index_mcdrop_coastal, mean_mcdrop_coastal, error_mcdrop_coastal = get_metrics_from_pickle(pickle_save_path_mcdrop_coastal, imgnet_flag=False, name = "MC Dropout Coastal")
index_viflipout_coastal, mean_viflipout_coastal, error_viflipout_coastal = get_metrics_from_pickle(pickle_save_path_viflipout_coastal, imgnet_flag=False, name = "VI Flipout Coastal")
index_deterministic_coastal, mean_deterministic_coastal, error_deterministic_coastal = get_metrics_from_pickle(pickle_save_path_deterministic_coastal, imgnet_flag=False, name = "Deterministic Coastal")
index_ViT_coastal, mean_ViT_coastal, error_ViT_coastal = get_metrics_from_pickle(pickle_save_path_ViT_coastal, imgnet_flag=False, name = "MC Dropout ViT Coastal")

#Get the information from the Imagenet pickle files 
index_mcdrop_imagenet, mean_mcdrop_imagenet, error_mcdrop_imagenet = get_metrics_from_pickle(pickle_save_path_mcdrop_imagenet, imgnet_flag=True, name = "MC Dropout ImageNet")
index_viflipout_imagenet, mean_viflipout_imagenet, error_viflipout_imagenet = get_metrics_from_pickle(pickle_save_path_viflipout_imagenet, imgnet_flag=True, name = "VI Flipout ImageNet")
# index_deterministic_imagenet, mean_deterministic_imagenet, error_deterministic_imagenet = get_metrics_from_pickle(pickle_save_path_deterministic_imagenet, imgnet_flag=True)

fig, ax = plt.subplots(figsize=(9,5))
#Plot multiple lines on same plot
plot_metrics(index_mcdrop_coastal, mean_mcdrop_coastal, error_mcdrop_coastal,label="ResNet50 MC Dropout Coastal", imgnet_flag=True)
plot_metrics(index_viflipout_coastal, mean_viflipout_coastal, error_viflipout_coastal, label="ResNet50 VI Flipout Coastal", imgnet_flag=True)
plot_metrics(index_ViT_coastal, mean_ViT_coastal, error_ViT_coastal, label="ViT MC Dropout Coastal", imgnet_flag=True)


plot_metrics(index_mcdrop_imagenet, mean_mcdrop_imagenet, error_mcdrop_imagenet,label="ResNet50 MC Dropout ImageNet", imgnet_flag=False)
plot_metrics(index_viflipout_imagenet, mean_viflipout_imagenet, error_viflipout_imagenet, label="ResNet50 VI Flipout ImageNet", imgnet_flag=False)
# plot_metrics(index_deterministic_imagenet, mean_deterministic_imagenet, error_deterministic_imagenet, label="Deterministic ImageNet", imgnet_flag=False)
plt.savefig(f"/data/kraken/coastal_project/coastal_uncertainty/calibration_curves/plot_calibrations_8aug.png")
plt.close()