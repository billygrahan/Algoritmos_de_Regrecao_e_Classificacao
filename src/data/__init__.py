"""
Data loading module for ML algorithms.
Provides easy access to California Housing and Breast Cancer datasets.
"""

from .datasets import load_california_housing, load_breast_cancer

__all__ = ['load_california_housing', 'load_breast_cancer']
