"""
Algoritmos de Regressão e Classificação

This package provides implementations of various machine learning algorithms
including Linear Regression, Logistic Regression, KNN, and Neural Networks,
along with utilities for loading and processing datasets.
"""

from . import algorithms
from . import data

__version__ = '0.1.0'
__all__ = ['algorithms', 'data']
