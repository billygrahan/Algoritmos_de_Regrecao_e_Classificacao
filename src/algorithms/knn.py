"""
K-Nearest Neighbors (KNN) implementation.
KNN - Algorithm for classification and regression based on nearest neighbors.
"""

import numpy as np


class KNN:
    """
    K-Nearest Neighbors algorithm.
    
    This class will implement KNN for classification and regression tasks.
    To be implemented in future iterations.
    
    Attributes:
    -----------
    k : int
        Number of neighbors to use (to be implemented).
    X_train_ : array-like
        Training data (to be implemented).
    y_train_ : array-like
        Training labels (to be implemented).
    """
    
    def __init__(self, k=5):
        """
        Initialize KNN model.
        
        Parameters:
        -----------
        k : int, default=5
            Number of neighbors to use.
        """
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X, y):
        """
        Fit the KNN model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # To be implemented
        pass
    
    def predict(self, X):
        """
        Predict class labels or values for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels or values.
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
