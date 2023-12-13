# PolypDetectionSeparableUNET

PolypDetectionUNET is a repository for polyp detection using the U-Net architecture implemented in PyTorch. It includes Python scripts for handling the dataset, defining the U-Net model, training the model, and utility functions.

## Table of Contents
- [Introduction](#introduction)
- [Files](#files)
  - [dataset.py](#datasetpy)
  - [model.py](#modelpy)
  - [train.py](#trainpy)
  - [utils.py](#utilspy)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

PolypDetectionUNET is designed for the detection of polyps using the U-Net architecture, a popular convolutional neural network for image segmentation tasks. The repository is organized into four Python files: `dataset.py`, `model.py`, `train.py`, and `utils.py`. Each file serves a specific purpose in the pipeline.

## Files

### dataset.py

The `dataset.py` file contains code for handling the dataset used for training and evaluation. It may include data loading, preprocessing, and any other necessary steps for preparing the dataset.

### model.py

The `model.py` file defines the U-Net model architecture for polyp detection using PyTorch. This file contains the neural network architecture, layers, and any additional components required for the model.

### train.py

The `train.py` file is responsible for training the U-Net model on the polyp dataset. It includes code for setting up training parameters, loading the dataset, and executing the training loop.

### utils.py

The `utils.py` file provides utility functions that may be used across different components of the project. These functions can include metrics calculation, image processing, or any other helper functions.

