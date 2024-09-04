# CoastalUncertainty


## Name
Uncertainty aware classification of aerial coastal imagery using probabilistic convolutional neural networks and vision transformers

## Description
This repository contains all project source code as well as a Jupyter Notebook walk-through of different visualizations. This repository is the source code for the IEEE Access paper titled "Uncertainty-Aware Aerial Coastal Imagery Pattern Recognition Through Transfer Learning with ImageNet-1K Variational Embeddings".

## Roadmap
The datasets, model checkpoints, and some pickle evaluation files for this project are available for download at https://nps.app.box.com/folder/260201116211 due to GitHub storage limits.

# trainers
    Contains code for training the ResNet50 Deterministic, MC Dropout, and VI Flipout models. There is a bash script in each folder that was used to run the training process. The train .sh file launches the Python code task.py and also contains all of the customizable hyperparameters for each training. The provided task.py files are the main files for performing transfer learning from ImageNet to the coastal dataset. The task.py files load model weights from the provided ImageNet checkpoint and then perform training on the coastal dataset. There is an option to perform either fixed-feature training (where all layers except final classification head remain frozen while transfer learning occurs on the coastal dataset), or fine-tuning training (where a specified number of layers are unfrozen and allowed to train on coastal data). This is controlled by the "training" hyperparameter in the train.sh files.  
    To run the transfer learning, run the files such as train_mcdropout.sh or train_resnet50_flipout.sh. This will save model checkpoints throguhout the training process and output training logs (a .txt file) to view incremental training progress. The model checkpoints produced can be loaded to produce the various visualizations depicted in the paper. 
model_evaluations_into_pickle/
    Provides code for evaluating various metrics on test data and storing this data into pickle files for visualizations. This folder is broken into various subfolders depending on which dataset the model to be evaluated was trained on. For example, to run the evaluations for MC Dropout on the coastal dataset and generate the metrics data to be put in a pickle file, run the dropout_eval_to_pickle.sh, which calls the file eval_multiple_to_pickle.py. This file loads the MC Dropout model checkpoint that was trained by performing transfer learning from ImageNet to the coastal dataset and runs an evaluation on the coastal test set while producing uncertainty metrics such as epistemic uncertainty, aleatoric uncertainty,etc. 
classification_reports/
    Code for sklearn classification report
calibration_curves/
    Contains code to generate the calibration curves across the various models
grad_cam/ 
    Contains code to run Grad Cam heatmap visualizations
confusion_matrix/
    Contains code for confusion matrix generation
uncertainty_bar_plots/
    Contains code for epistemic uncertainty bar plots across different datasets
ECE/
    Contains code for generating the Expected Calibration Error for the various models
most_and_least_uncertain/
    Contains code for generating the ten most uncertain and ten least uncertain images from a given dataset, as shown in the paper appendix
follow_along_notebook/
    proj_notebook.ipynb contains code to load the test dataset, checkpoints, and then run some key visualizations on the data along with explanations.

## Setup Instructions
Due to Tensorflow compatibility issues, we had to use a few different conda environments for various parts of the code. The exact setup of these different environments are all included in the setup_files folder of this repo. The thesis_work environment should function for all code dealing with MC Dropout and deterministic models. The thesis_work environment is also used to run the Jupyter Notebook walk through. The VI_flipout environment should function for all code dealing with flipout model training and evaluation. The ViT2 model should function for all code dealing with the Vision Transformers. 


## Basic Pipeline

1. The first step is to train the models on the coastal dataset. This was done by loading the ImageNet checkpoint (availabled on Box), performing
fixed feature training on the coastal dataset where the majority of the model weights were frozen. Then, the model weights were unfrozen
and allowed to train on the coastal dataset. The task.py file in each folder of trainer/ is the main driver code for training each model. 
2. With a trained model (checkpoints available on Box), you can then run model evaluations and calculate uncertainty metrics for visualizations.
Using the trained checkpoints, we can the metric calculations and then output the pertinent data into pickle files. These evaluation files are in the folder model_evaluations_into_pickle. To generate a new pickle file with evaluation metrics, the files titled "eval_multiple_to_pickle.py" can be run using a loaded checkpoint. The pickle files with evaluation metrics already captured for the coastal, alaska, and orange cable are all available on Box. These are ready to use for visualziations, without having to run any eval code. 
3. The code for all of the visualizations in the paper is available in this repo. Notable visualizations from the paper include the calibration curves, uncertainty bar plots, and grad-cam heat map figures. 
4. Beyond this, the follow_along_notebook folder contains a simple Jupyter notebook that contains code with a model evaluation and some key visualizations.

## Cite

L. Rombado, M. Orescanin and M. Orescanin, "Uncertainty-Aware Aerial Coastal Imagery Pattern Recognition Through Transfer Learning with ImageNet-1K Variational Embeddings," in IEEE Access, doi: 10.1109/ACCESS.2024.3451373. 




