# Image Classification using Convolutional Neural Network (CNN)

This repository contains an image classification project implemented using a Convolutional Neural Network (CNN) in Python. The project demonstrates how deep learning models can automatically learn visual features from images and classify handwritten digits from the MNIST dataset.

---

## Overview

Image classification is a fundamental problem in computer vision where an input image is assigned a label.

In this project, a CNN is used to extract spatial features such as edges, curves, and digit patterns from 28x28 grayscale images in the MNIST dataset. These learned features are then used to classify digits from 0 to 9. The project is implemented using a Jupyter Notebook for experimentation, visualization, and step-by-step model development.

---

## Technologies Used

Python 3  
TensorFlow / Keras  
NumPy  
Matplotlib  
Jupyter Notebook  

---

## Dataset

This project uses the MNIST handwritten digit dataset.

The MNIST dataset consists of:

- 60,000 training images  
- 10,000 testing images  
- 28x28 grayscale digit images  
- 10 output classes (digits 0–9)

The dataset is automatically loaded using TensorFlow/Keras inside the notebook, so no manual download is required.

---

## Project Structure

MNIST-Digit-classification/  
  MNIST.ipynb  
  MNIST_digit.png  
  models/  
  README.md  

MNIST.ipynb contains the complete workflow including dataset loading, preprocessing, CNN model building, training, evaluation, and prediction.  

MNIST_digit.png is a sample digit image used for visualization or testing predictions.  

models/ directory can be used to store saved model weights or trained models.

---

## Installation

Clone the repository and move into the project directory.

git clone https://github.com/Sabeer65/MNIST-Digit-classification.git  
cd MNIST-Digit-classification  

Install the required dependencies.

pip install tensorflow numpy matplotlib  

---

## Usage

Launch Jupyter Notebook and open the main notebook.

jupyter notebook MNIST.ipynb  

Run the notebook cells sequentially to:

- Load the MNIST dataset  
- Preprocess and normalize image data  
- Build the CNN model  
- Train the model  
- Evaluate performance  
- Test predictions on sample images  

---

## Model Training

The CNN model includes:

- Convolutional layers for feature extraction  
- Activation functions (ReLU)  
- Pooling layers for dimensionality reduction  
- Flatten layer to convert feature maps into vectors  
- Fully connected (Dense) layers for classification  
- Softmax output layer for multi-class digit prediction  

Training parameters such as epochs, batch size, optimizer, and learning rate can be adjusted inside the notebook.

---

## Testing and Prediction

After training, the model is evaluated using the test dataset.

The notebook also demonstrates how to:

- Predict individual digit images  
- Visualize model predictions  
- Display predicted vs actual labels  

---

## Results

The CNN model achieves high accuracy on the MNIST dataset depending on the architecture and hyperparameters used.

Training and validation accuracy and loss values are visualized using plots generated during model training.

---

## Limitations

Model performance depends on architecture depth and training configuration.  
Overfitting may occur if the model is too complex.  
The project is notebook-based and does not include deployment.  

---

## Future Improvements

Experiment with deeper CNN architectures.  
Add dropout and batch normalization.  
Save and load trained model weights.  
Add confusion matrix and classification report.  
Convert the notebook into a deployable web application.  
