## Project Overview

This project is focused on ship detection using satellite imagery. The code consists of utility functions (utils.py), a model training script (train.py), and a prediction script (predict.py).

## File Descriptions

# 1. utils.py
This file contains utility functions for data preprocessing, image augmentation, and mask encoding/decoding. Key functions include:

Directory Paths: Paths for training and test image directories are specified.
Data Loading: Reading image names and ship segmentations from CSV file.
Mask Functions: Functions for run-length encoding and decoding of masks.
Data Balancing: Balancing the training set by sampling from each group.
Data Generators: Functions to generate batches of augmented training images and masks.

# 2. train.py
This script defines and trains a U-Net style convolutional neural network for ship segmentation. Key components include:

Model Architecture: U-Net style design with encoder and decoder layers.
Data Loading: Loading training and validation data using data generators.
Model Compilation: Compiling the model with the Adam optimizer and a custom Dice score, binary_accuracy metric.
Model Training: Training the model with specified callbacks and generators.
Model Saving: Saving the best weights and the trained model.
Training History Plots: Plotting and saving the training history (loss and binary accuracy).

# 3. predict.py
This script loads the trained model and performs predictions on test data. Key functionalities include:

Model Loading: Loading the trained segmentation model.
Prediction: Generating predictions on validation data.
Visualization: Plotting and saving predictions for a subset of test images.
