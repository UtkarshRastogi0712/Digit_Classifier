# Digit_Classifier
Multiclass classification of handwritten digits from the MNIST digit dataset using a neural network

The project employs a Neural Network with one hidden layer to predict a 28px by 28px handwritten digit from the MNIST digit dataset (downloaded from: https://www.kaggle.com/oddrationale/mnist-in-csv ). It has been written in python and only uses numpy for linear algebra mathematical functions and does not employ any learning libraries.

Update (19/07/2022) : A simple CNN implementation has also been added.

The overall dataset has two CSV files. Namely the training set with 60000 rows of grey levels of 784px (28x28) size-normalized and centered images of handwritten digits and test set with 10000 rows of the same.
The neural network has 784 nodes in the input layer, 10 nodes in the hidden layer and 10 nodes in the output layer(for each of the 10 digits). It uses a ReLU activation function in the hidden layer and softmax activation function in the output layer and assumes a cross entroy cost function for the sake of simplified back propagation.

The model has been tested with various learning rates (eg~ 0.01, 0.005, 0.001) and for a multitude of iterations (eg~ 10000, 50000, 100000, 200000) and has an accuracy percentage between 87% to 92% 
