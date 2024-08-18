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
pylab.rcParams['figure.figsize'] = (12.0, 6.0)
import seaborn as sns
from imagenet_dict import IMAGENET_CLASS_INT_TO_STR
import cv2

#ImageNet Pickle Files 
pickle_save_path_mcdrop_imagenet = '/data/kraken/coastal_project/coastal_proj_code/resnet_MCdropout/imagenet/pickle_dump/imagenet_mcdrop_1_1_imagetest.pickle'
def get_metrics_from_pickle(pickle_save_path, imgnet_flag): 
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
            #pass
        
        preds_mc, entropy_mc, nll_mc, pred_std_mc, var, normal_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg = BNN_predict(1000, predict_probs, y_test_labels,'multi_class')

        y_pred = np.argmax(preds_mc, axis=1)
        if imgnet_flag == True:
            y_true = y_test_labels
        else:
            y_true = np.argmax(y_test_labels, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        acc_cal_noe = []
        one_pct = int(y_true.shape[0] * .01) #this is 8
        images = np.squeeze(images, axis=0)
        y_true = np.squeeze(y_true, axis=1)
        pred_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_pred), key = lambda x: x[0], reverse=True)]
        true_labels_sort = [v for _,v in sorted(zip(normal_entropy,y_true), key = lambda x: x[0], reverse=True)]
        images_sort =      [v for _,v in sorted(zip(normal_entropy,images), key = lambda x: x[0], reverse=True)]
        #we want to grab the first few images from this array image_sort 

        #Most Uncertain~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # corresponding_true_labels = true_labels_sort[1:11]
        # corresponding_predicted_labels = pred_labels_sort[1:11]
        # images_to_analyze = images_sort[1:11]
        # images_to_analyze = np.array(images_to_analyze)
        # images_to_analyze = images_to_analyze.astype(np.uint8)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #Least Uncertain 50000 ~~~~~~~~~~~~~~~~~~~~~~~
        corresponding_true_labels = true_labels_sort[49989:]
        corresponding_predicted_labels = pred_labels_sort[49989:]
        images_to_analyze = images_sort[49989:]
        images_to_analyze = np.array(images_to_analyze)
        images_to_analyze = images_to_analyze.astype(np.uint8)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        
        for p in range(100):
            tl = true_labels_sort[p*one_pct:]
            pl = pred_labels_sort[p*one_pct:]
            accuracy=accuracy_score(tl,pl)
            acc_cal_noe.append(accuracy)

        acc_cal_noe.reverse()
        #this is the list of lists
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
    fig, axes = plt.subplots(2, 5, figsize=(12,8))
    for i in range(2):  # Rows
        for j in range(5):  # Columns  
            resized_image = cv2.resize(images_to_analyze[i * 5 + j], (150,150))
            axes[i, j].imshow(resized_image)
            axes[i, j].set_xlabel(f'True: {IMAGENET_CLASS_INT_TO_STR[corresponding_true_labels[i * 5 + j]]} \n Pred: {IMAGENET_CLASS_INT_TO_STR[corresponding_predicted_labels[i * 5 + j]]}')
            axes[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.1, hspace=0.0005)

    plt.suptitle("10 Least Uncertain ImageNet Images (total entropy)", fontsize=30)
    plt.savefig(f"/data/kraken/coastal_project/coastaluncertainty/analyze_uncertainty/mcdropout/imagenet/22may_least_uncertain_images.png")

    return index, mean, error


#Get the information from the Imagenet pickle files 
index_mcdrop_imagenet, mean_mcdrop_imagenet, error_mcdrop_imagenet = get_metrics_from_pickle(pickle_save_path_mcdrop_imagenet, imgnet_flag=True)
