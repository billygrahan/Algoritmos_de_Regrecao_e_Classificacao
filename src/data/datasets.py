"""
Dataset loaders for California Housing and Breast Cancer datasets.
Uses scikit-learn datasets and returns them as pandas DataFrames.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer as sk_load_breast_cancer


def load_california_housing(return_X_y=False, as_frame=True):
    """
    Load the California Housing dataset from scikit-learn.
    
    Note: This dataset requires internet connection for first-time download.
    If the dataset cannot be fetched, it will attempt to use cached version
    or provide helpful error message.
    
    Parameters:
    -----------
    return_X_y : bool, default=False
        If True, returns (data, target) instead of a dictionary.
    as_frame : bool, default=True
        If True, returns data as pandas DataFrame/Series.
        
    Returns:
    --------
    data : dict or tuple
        Dictionary with 'data' and 'target' keys containing pandas DataFrames,
        or tuple of (X, y) if return_X_y=True.
        
    Examples:
    ---------
    >>> data = load_california_housing()
    >>> X, y = load_california_housing(return_X_y=True)
    
    Raises:
    -------
    RuntimeError
        If the dataset cannot be loaded and is not cached.
    """
    try:
        dataset = fetch_california_housing(as_frame=as_frame)
    except Exception as e:
        error_msg = (
            "Failed to load California Housing dataset. "
            "This dataset requires internet connection for first-time download. "
            f"Error: {str(e)}\n\n"
            "Please ensure you have internet connectivity, or the dataset is already cached."
        )
        raise RuntimeError(error_msg) from e
    
    if return_X_y:
        return dataset.data, dataset.target
    
    return {
        'data': dataset.data,
        'target': dataset.target,
        'feature_names': dataset.feature_names,
        'DESCR': dataset.DESCR
    }


def load_breast_cancer(return_X_y=False, as_frame=True):
    """
    Load the Breast Cancer Wisconsin dataset from scikit-learn.
    
    Parameters:
    -----------
    return_X_y : bool, default=False
        If True, returns (data, target) instead of a dictionary.
    as_frame : bool, default=True
        If True, returns data as pandas DataFrame/Series.
        
    Returns:
    --------
    data : dict or tuple
        Dictionary with 'data' and 'target' keys containing pandas DataFrames,
        or tuple of (X, y) if return_X_y=True.
        
    Examples:
    ---------
    >>> data = load_breast_cancer()
    >>> X, y = load_breast_cancer(return_X_y=True)
    """
    dataset = sk_load_breast_cancer(as_frame=as_frame)
    
    if return_X_y:
        return dataset.data, dataset.target
    
    return {
        'data': dataset.data,
        'target': dataset.target,
        'feature_names': dataset.feature_names,
        'target_names': dataset.target_names,
        'DESCR': dataset.DESCR
    }
