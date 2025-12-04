"""
Linear Regression implementation.
Regressão Linear - Algorithm for predicting continuous values.
"""

import numpy as np  # Will be used when implementing the algorithm


class LinearRegression:
    """
    Linear Regression algorithm.
    
    This class will implement linear regression for predicting continuous values.
    To be implemented in future iterations.
    
    Attributes:
    -----------
    coefficients_ : array-like
        Coefficients of the linear model (to be implemented).
    intercept_ : float
        Intercept of the linear model (to be implemented).
    """
    
    def __init__(self):
        """Initialize Linear Regression model."""
        self.coefficients_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
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
        Predict using the linear model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        # To be implemented
        pass
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True values for X.
            
        Returns:
        --------
        score : float
            R^2 score.
        """
        # To be implemented
        pass
