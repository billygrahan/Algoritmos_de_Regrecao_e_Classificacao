"""
Machine Learning Algorithms Module.

This module provides implementations of various ML algorithms:
- Linear Regression (Regressão Linear)
- Logistic Regression (Regressão Logística)
- K-Nearest Neighbors (KNN)
- Neural Networks (Redes Neurais)
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .knn import KNN
from .neural_network import NeuralNetwork

__all__ = ['LinearRegression', 'LogisticRegression', 'KNN', 'NeuralNetwork']
