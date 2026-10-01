hychung1/ML_EfficientNet-B0_Fruit_Recognition

# Fruit Recognition Using EfficientNet-B0

A machine learning school project that uses the EfficientNet-B0 convolutional neural network to classify images of fruits.

## Project Overview

The goal of this project is to build an image-classification model capable of recognizing different types of fruit from images. The project uses transfer learning with EfficientNet-B0, a lightweight convolutional neural network designed to achieve strong image-classification performance with relatively few computational resources.

This project was created for educational purposes as part of a machine learning school project.

## Objectives

- Learn how image-classification models work
- Apply transfer learning to a custom image dataset
- Train an EfficientNet-B0 model to recognize different fruit classes
- Evaluate the model using validation and test images
- Practice organizing and documenting a machine learning project

## Model

This project uses **EfficientNet-B0** as the image-classification model.

EfficientNet models use compound scaling to balance network depth, width, and image resolution. EfficientNet-B0 is the baseline version of the EfficientNet family and is commonly used for efficient image-classification tasks.

The model was adapted for this project by replacing or modifying its final classification layer to match the number of fruit categories in the dataset.

## Dataset

You can get on kaggle.com
-> Link: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

The dataset consists of labeled fruit images organized by category.

Expected directory structure:

```text
dataset/
├── train/
│   ├── fruit\_class\_1/
│   ├── fruit\_class\_2/
│   └── ...
├── validation/
│   ├── fruit\_class\_1/
│   ├── fruit\_class\_2/
│   └── ...
└── test/
    ├── fruit\_class\_1/
    ├── fruit\_class\_2/
    └── ...
