# CoastalUncertainty


## Name
Uncertainty aware classification of aerial coastal imagery using probabilistic convolutional neural networks and vision transformers

## Description
This repository contains all project source code as well as some Jupyter Notebook walk-throughs of different visualizations. This repository is the source code for the IEEE Access paper (link here).

## Roadmap
The dataset and model checkpoints for this project are available for download at https://nps.app.box.com/folder/260201116211 due to GitHub storage limits.

trainer/
    Contains code for training the ResNet50 Deterministic, MC Dropout, and VI Flipout models
calibration_curves/
    Contains code to generate the calibration curves across the various models
grad_cam/ 
    Contains code to run Grad Cam heatmap visualizations
confusion_matrix/
    Contains code for confusion matrix generation
paper_bar_plots/
    Contains code for epistemic uncertainty bar plots across different datasets
proj_notebook.ipynb contains code to load the test dataset, checkpoints, and then run some key visualizations on the data along with explanations.

## Setup Instructions
Due to Tensorflow compatibility issues, we had to use a few different conda environments for various parts of the code. The exact setup of these different environments are all included in the setup_files folder of this repo. The thesis_work environment should function for all code dealing with MC Dropout and deterministic models. The thesis_work environment is also used to run the Jupyter Notebook walk through. The VI_flipout environment should function for all code dealing with flipout model training and evaluation. The ViT2 model should function for all code dealing with the Vision Transformers. 


## Basic Pipeline

1. The first step is to train the models on the coastal dataset. This was done by loading the ImageNet checkpoint (availabled on Box), performing
fixed feature training on the coastal dataset where the majority of the model weights were frozen. Then, the model weights were unfrozen
and allowed to train on the coastal dataset. The task.py file in each folder of trainer/ is the main driver code for training each model. 
2. With a trained model (checkpoints available on Box), you can then run model evaluations and calculate uncertainty metrics for visualizations.
Using the trained checkpoints, we can the metric calculations and then output the pertinent data into pickle files. The pickle files are all available on Box. 
3. The code for all of the visualizations in the paper is available in this repo. 
4. Beyond this, the follow_along_notebook folder contains a simple Jupyter notebook that contains code with a model evaluation and some key visualizations.


