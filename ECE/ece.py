import numpy as np
from sklearn.metrics import accuracy_score



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
import tensorflow_probability as tfp



#Coastal Pickle Files

pickle_save_path_mcdrop_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_mcdropout/coastal/pickle_dump/mcdropout_10_20.pickle'

pickle_save_path_viflipout_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/coastal/pickle_dump/vi_flipout_10_20.pickle'

pickle_save_path_deterministic_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_deterministic/coastal/pickle_dump/coastal_resnet_deterministic_10_20.pickle'

pickle_save_path_ViT_coastal = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/dropout_vit/pickle_dump/vit_MCDropout_10_20_3AUG.pickle'

#ImageNet Pickle Files 
pickle_save_path_mcdrop_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_mcdropout/imagenet/pickle_dump/imagenet_mcdrop_multiple_10_20.pickle'

pickle_save_path_viflipout_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_flipout/imagenet/pickle_dump/imagenet_flipout_10_20.pickle'

# pickle_save_path_deterministic_imagenet = '/data/kraken/coastal_project/coastal_uncertainty/model_evaluations_into_pickle/resnet_deterministic/imagenet/pickle_dump/imagenet_deterministic_10_20.pickle'


#Code from: https://colab.research.google.com/github/majapavlo/medium/blob/main/ece_medium.ipynb
def expected_calibration_error(samples, true_labels, M=15):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece



def get_ece_from_pickle(pickle_save_path, imgnet_flag, name):
    '''
    This function iterates through 10 rounds of 20 MC samples in each pickle file passed in. ECE is calcualted for each of the 20 samples, 10 times. Means
    are printed out at the bottom of the function. 
    '''
    
    #Load from pickle file
    with open(pickle_save_path, 'rb') as f:
        results_list = pickle.load(f)
        ece_mean_list = []
        accuracy_mean_list = []

    for index, iteration_results in enumerate(results_list, start=1):
        #Looping through ten rounds

        ece_list = []
        accuracy_list = []
        
        print(f'Grabbing Dictionary # {index}:')    
        
        predict_probs = iteration_results['preds']
        y_test_labels = iteration_results['trueLabels'] #This is the same across all 10 rounds (it's just the true labels)

        #Loop through the 20 MC samples across the 20 run ensemble

        for i in range (20):
            probs = predict_probs[i]
            
            if imgnet_flag:
               
                y_true_ece = (y_test_labels.flatten()).astype(int) #Special formatting due to imagenet array structure
                ece = expected_calibration_error(probs, y_true_ece)
                y_pred = np.argmax(probs, axis=1)
                accuracy = accuracy_score(y_test_labels, y_pred)

            else:
                y_test = np.argmax(y_test_labels, axis=1) #gets the index (1,2,3,4,etc.) from [0,0,0,1,...]
                ece = expected_calibration_error(probs, y_test)
                y_pred = np.argmax(probs, axis=1)
                accuracy = accuracy_score(np.argmax(y_test_labels, axis=1), y_pred)

            ece_list.append(ece)
            accuracy_list.append(accuracy)

        ece_mean = np.mean(np.array(ece_list)) #Take the mean across the 20
        print(f"Mean ECE across 1 run of 20 samples for {name}: {ece_mean}")

        accuracy_mean = np.mean(np.array(accuracy_list)) #Take the mean across the 20
        print(f"Mean Accuracy across 1 run of 20 samples for {name}: {accuracy_mean}")

        ece_mean_list.append(ece_mean)
        accuracy_mean_list.append(accuracy_mean)

    ece_df = pd.DataFrame(ece_mean_list)
    ece_std = ece_df.std(axis=0) 

    accuracy_df = pd.DataFrame(accuracy_mean_list)
    accuracy_std = accuracy_df.std(axis=0)

    accuracy_mean = accuracy_df.mean(axis=0) 
    print("***Accuracy Total Mean: ", accuracy_mean)

    ece_mean = ece_df.mean(axis=0)
    print("***ECE Total Mean: ", ece_mean)

    accuracy_mean = accuracy_df.mean(axis=0) 

    print("***Accuracy Total Std: ", accuracy_std)
    print("***ECE Total Std: ", ece_std)

    accuracy_ci = 1.96 * accuracy_std / math.sqrt(10)
    ece_ci = 1.96 * ece_std / math.sqrt(10)

    print("***Accuracy Total Confidence Interval: ", accuracy_ci)
    print("***ECE Total Confidence Interval: ", ece_ci)


#Get the information from the coastal pickle files 
# get_ece_from_pickle(pickle_save_path_mcdrop_coastal, imgnet_flag=False, name = "MC Dropout Coastal")
# get_ece_from_pickle(pickle_save_path_viflipout_coastal, imgnet_flag=False, name = "VI Flipout Coastal")
# get_ece_from_pickle(pickle_save_path_deterministic_coastal, imgnet_flag=False, name = "Deterministic Coastal")
get_ece_from_pickle(pickle_save_path_ViT_coastal, imgnet_flag=False, name = "VIT Dropout Coastal")

#Get the information from the Imagenet pickle files 
get_ece_from_pickle(pickle_save_path_mcdrop_imagenet, imgnet_flag=True, name = "MC Dropout ImageNet")
get_ece_from_pickle(pickle_save_path_viflipout_imagenet, imgnet_flag=True, name = "VI Flipout ImageNet")



