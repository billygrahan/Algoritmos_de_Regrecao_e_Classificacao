"""
Tests for dataset loading functionality.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import load_california_housing, load_breast_cancer


class TestCaliforniaHousing:
    """Tests for California Housing dataset loader."""
    
    def test_load_requires_internet(self):
        """Test that loading without internet raises appropriate error."""
        # This test will pass if RuntimeError is raised when internet is not available
        # Or if the dataset loads successfully (if internet is available or cached)
        try:
            data = load_california_housing()
            # If we get here, dataset loaded successfully
            assert 'data' in data
            assert 'target' in data
            assert 'feature_names' in data
            assert data['data'].shape[0] > 0
            assert data['data'].shape[1] > 0
        except RuntimeError as e:
            # Expected when internet is not available and dataset not cached
            assert "Failed to load California Housing dataset" in str(e)
    
    def test_load_return_X_y_or_error(self):
        """Test loading with return_X_y=True or expect error."""
        try:
            X, y = load_california_housing(return_X_y=True)
            assert X.shape[0] == y.shape[0]
            assert X.shape[0] > 0
            assert X.shape[1] > 0
            # California Housing should have 8 features
            assert X.shape[1] == 8
            # Should have 20640 samples
            assert X.shape[0] == 20640
        except RuntimeError:
            # Expected when dataset cannot be loaded
            pytest.skip("California Housing dataset requires internet connection")


class TestBreastCancer:
    """Tests for Breast Cancer dataset loader."""
    
    def test_load_default(self):
        """Test loading with default parameters."""
        data = load_breast_cancer()
        assert 'data' in data
        assert 'target' in data
        assert 'feature_names' in data
        assert 'target_names' in data
        assert data['data'].shape[0] > 0
        assert data['data'].shape[1] > 0
    
    def test_load_return_X_y(self):
        """Test loading with return_X_y=True."""
        X, y = load_breast_cancer(return_X_y=True)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0
        assert X.shape[1] > 0
    
    def test_data_shape(self):
        """Test that dataset has expected properties."""
        X, y = load_breast_cancer(return_X_y=True)
        # Breast Cancer should have 30 features
        assert X.shape[1] == 30
        # Should have 569 samples
        assert X.shape[0] == 569
    
    def test_binary_classification(self):
        """Test that target has only two classes."""
        data = load_breast_cancer()
        assert len(data['target_names']) == 2
