from typing import Tuple

import numpy as np

from schemas.hyperparameter_validation import RFParams
from schemas.evaluation_metrics import EvaluationMetrics
from implementation.random_forest_classifier import RandomForestMnistClassifier

class MnistClassifier:
    """
    High-level wrapper for MNIST classifiers.  
    Provides a unified interface for training, prediction, and evaluation
    regardless of the underlying algorithm.

    Attributes:
        classifier: An instance of a specific MNIST classifier (e.g., RandomForestMnistClassifier).
    """

    def __init__(self, algorithm: str, hyperparams: RFParams = None):
        """
        Initialize the MNIST classifier with the chosen algorithm.

        Args:
            algorithm (str): The algorithm to use. Supported values:
                - "rf": Random Forest
            hyperparams (RFParams, optional): Hyperparameters specific to the chosen algorithm.
                Defaults to None, in which case default hyperparameters are used.

        Raises:
            ValueError: If an unknown algorithm string is provided.
        """
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier(hyperparams)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the selected MNIST classifier on the training dataset.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation feature data for evaluation after training.
            y_val (np.ndarray, optional): Validation labels.

        Notes:
            If validation data is provided, validation metrics are printed during training.
        """
        self.classifier.train(X_train, y_train, X_val, y_val)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given test data using the selected classifier.

        Args:
            X_test (np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted class labels.
        """        
        return self.classifier.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[EvaluationMetrics, np.array]:
        """
        Evaluate the classifier on test data and compute performance metrics.

        Args:
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): True test labels.

        Returns:
            Tuple[EvaluationMetrics, np.ndarray]:
                - EvaluationMetrics object containing accuracy, precision, recall, f1-score, and confusion matrix.
                - Numpy array of predicted labels.
        """
        return self.classifier.evaluate(X_test, y_test)