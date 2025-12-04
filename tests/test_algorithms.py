"""
Tests for algorithm implementations.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from algorithms import LinearRegression, LogisticRegression, KNN, NeuralNetwork


class TestLinearRegression:
    """Tests for Linear Regression algorithm."""
    
    def test_initialization(self):
        """Test that Linear Regression can be initialized."""
        model = LinearRegression()
        assert model is not None
        assert model.coefficients_ is None
        assert model.intercept_ is None


class TestLogisticRegression:
    """Tests for Logistic Regression algorithm."""
    
    def test_initialization(self):
        """Test that Logistic Regression can be initialized."""
        model = LogisticRegression()
        assert model is not None
        assert model.coefficients_ is None
        assert model.intercept_ is None
        assert model.classes_ is None


class TestKNN:
    """Tests for KNN algorithm."""
    
    def test_initialization(self):
        """Test that KNN can be initialized."""
        model = KNN()
        assert model is not None
        assert model.k == 5
        assert model.X_train_ is None
        assert model.y_train_ is None
    
    def test_initialization_custom_k(self):
        """Test that KNN can be initialized with custom k."""
        model = KNN(k=3)
        assert model.k == 3


class TestNeuralNetwork:
    """Tests for Neural Network algorithm."""
    
    def test_initialization(self):
        """Test that Neural Network can be initialized."""
        model = NeuralNetwork()
        assert model is not None
        assert model.hidden_layers == (100,)
        assert model.activation == 'relu'
        assert model.weights_ is None
        assert model.biases_ is None
    
    def test_initialization_custom_layers(self):
        """Test that Neural Network can be initialized with custom layers."""
        model = NeuralNetwork(hidden_layers=(50, 25), activation='sigmoid')
        assert model.hidden_layers == (50, 25)
        assert model.activation == 'sigmoid'
