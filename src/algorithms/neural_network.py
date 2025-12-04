"""
Neural Network implementation.
Redes Neurais - Algorithm for deep learning and complex pattern recognition.
"""

import numpy as np  # Will be used when implementing the algorithm


class NeuralNetwork:
    """
    Neural Network algorithm.
    
    This class will implement a neural network for classification and regression tasks.
    To be implemented in future iterations.
    
    Attributes:
    -----------
    layers_ : list
        List of layer sizes (to be implemented).
    weights_ : list
        List of weight matrices (to be implemented).
    biases_ : list
        List of bias vectors (to be implemented).
    """
    
    def __init__(self, hidden_layers=(100,), activation='relu'):
        """
        Initialize Neural Network model.
        
        Parameters:
        -----------
        hidden_layers : tuple, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer.
        activation : str, default='relu'
            Activation function for the hidden layers.
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.layers_ = None
        self.weights_ = None
        self.biases_ = None
    
    def fit(self, X, y, epochs=100, learning_rate=0.01):
        """
        Fit the neural network model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        epochs : int, default=100
            Number of training epochs.
        learning_rate : float, default=0.01
            Learning rate for gradient descent.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # To be implemented
        pass
    
    def predict(self, X):
        """
        Predict using the neural network.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted values or class labels.
        """
        # To be implemented
        pass
    
    def score(self, X, y):
        """
        Return the score of the prediction.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True values for X.
            
        Returns:
        --------
        score : float
            Score of the prediction.
        """
        # To be implemented
        pass
