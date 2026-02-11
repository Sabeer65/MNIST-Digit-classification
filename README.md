# MNIST Digit Classification using Convolutional Neural Network (CNN)

This repository contains a deep learning project that implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset. The goal of this project is to understand and apply core computer vision and deep learning concepts using a well-known benchmark dataset.

## Project Overview

The MNIST dataset consists of 70,000 grayscale images (28x28 pixels) of handwritten digits. It is widely used for learning and benchmarking image classification models.

This project demonstrates:

- Loading and exploring image datasets
- Preprocessing and normalizing pixel values
- Designing and training a CNN using TensorFlow/Keras
- Evaluating model performance on unseen test data
- Making predictions on sample digit images

All implementation is done inside a Jupyter Notebook to clearly show each step of the workflow.

### Key Features

- **Multi-Class Image Classification:** Classifies digits from 0 to 9.
- **CNN Architecture:** Uses convolutional and pooling layers for spatial feature extraction.
- **Training & Evaluation:** Tracks accuracy and loss metrics during training.
- **Prediction Testing:** Allows inference on custom digit images.
- **Fully Local Execution:** Runs entirely on a local machine without external services.

## Repository Structure

- `models/` – Optional directory for saving trained model weights.
- `MNIST.ipynb` – Main notebook containing preprocessing, model architecture, training, and evaluation.
- `MNIST_digit.png` – Sample digit image used for testing predictions.
- `README.md` – Project documentation.

## Installation

```bash
git clone https://github.com/Sabeer65/MNIST-Digit-classification.git
cd MNIST-Digit-classification
pip install tensorflow numpy matplotlib
```

## Usage

### Training

1. Launch Jupyter Notebook:

```bash
jupyter notebook MNIST.ipynb
```

2. Run all cells sequentially.
3. The dataset is automatically loaded from Keras.
4. The model trains and displays performance metrics.

### Inference

After training:

- Evaluate model accuracy on test data.
- Run prediction cells to classify sample or custom digit images.

## Model Architecture

The CNN model includes:

- Convolutional Layers for detecting spatial features
- ReLU Activation functions
- MaxPooling Layers for dimensionality reduction
- Flatten Layer to prepare data for dense layers
- Fully Connected (Dense) Layers
- Softmax Output Layer (10 classes)

## Results

The model achieves high accuracy on the MNIST test dataset depending on training configuration such as number of epochs and batch size. Performance curves (accuracy and loss) are visualized within the notebook.

## Learning Outcomes

Through this project, key concepts practiced include:

- Understanding CNN architecture components
- Handling image datasets
- Preventing overfitting using proper preprocessing
- Evaluating classification performance

## License

This project is open-source and intended for educational and learning purposes.
