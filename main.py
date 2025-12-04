"""
Main entry point for the ML Algorithms project.

This script demonstrates the project structure and how to use the datasets
and algorithm implementations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_california_housing, load_breast_cancer
from algorithms import LinearRegression, LogisticRegression, KNN, NeuralNetwork


def main():
    """Main function to demonstrate the project structure."""
    print("=" * 60)
    print("Algoritmos de Regressão e Classificação")
    print("=" * 60)
    
    # Load datasets
    print("\n📊 Loading Datasets...")
    
    # California Housing Dataset (for regression)
    print("\n1. California Housing Dataset (Regression)")
    try:
        california_data = load_california_housing()
        X_calif, y_calif = california_data['data'], california_data['target']
        print(f"   - Features: {X_calif.shape[1]}")
        print(f"   - Samples: {X_calif.shape[0]}")
        print(f"   - Feature names: {list(california_data['feature_names'])}")
    except RuntimeError as e:
        print(f"   ⚠️  Could not load California Housing dataset")
        print(f"   Reason: Requires internet connection for first download")
        print(f"   Note: Dataset will be available after first successful download")
    
    # Breast Cancer Dataset (for classification)
    print("\n2. Breast Cancer Dataset (Classification)")
    try:
        cancer_data = load_breast_cancer()
        X_cancer, y_cancer = cancer_data['data'], cancer_data['target']
        print(f"   - Features: {X_cancer.shape[1]}")
        print(f"   - Samples: {X_cancer.shape[0]}")
        print(f"   - Classes: {list(cancer_data['target_names'])}")
    except Exception as e:
        print(f"   ⚠️  Could not load Breast Cancer dataset: {e}")
    
    # Display algorithm implementations
    print("\n🤖 Available Algorithms:")
    print("\n1. Linear Regression (Regressão Linear)")
    lr = LinearRegression()
    print(f"   - Class: {lr.__class__.__name__}")
    print(f"   - Status: Ready for implementation")
    
    print("\n2. Logistic Regression (Regressão Logística)")
    logr = LogisticRegression()
    print(f"   - Class: {logr.__class__.__name__}")
    print(f"   - Status: Ready for implementation")
    
    print("\n3. K-Nearest Neighbors (KNN)")
    knn = KNN(k=5)
    print(f"   - Class: {knn.__class__.__name__}")
    print(f"   - Status: Ready for implementation")
    
    print("\n4. Neural Network (Redes Neurais)")
    nn = NeuralNetwork(hidden_layers=(100,))
    print(f"   - Class: {nn.__class__.__name__}")
    print(f"   - Status: Ready for implementation")
    
    print("\n" + "=" * 60)
    print("✅ Project structure is ready!")
    print("📝 Algorithms are ready to be implemented.")
    print("=" * 60)


if __name__ == "__main__":
    main()
