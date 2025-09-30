from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from schemas.hyperparameter_validation import RFParams
from schemas.evaluation_metrics import EvaluationMetrics
from interface.mnist_classifier_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    A wrapper around scikit-learn's RandomForestClassifier for MNIST digit classification.
    Implements the MnistClassifierInterface with train and predict methods.

    Attributes:
        model (RandomForestClassifier): 
            The underlying scikit-learn RandomForest model.
    """

    def __init__(self, params: RFParams = None):
        """
        Initialize the RandomForestMnistClassifier with given parameters.

        Args:
            params (RFParams, optional): 
                Random forest hyperparameters defined in RFParams schema. 
                If None, default RFParams are used.
        """
        if params is None:
            params = RFParams()
        self.model = RandomForestClassifier(**params.model_dump())

    def train(self, X_train, y_train, X_val=None, y_val=None) -> None:
        """
        Train the random forest classifier on the training dataset.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation feature data for evaluation after training.
            y_val (np.ndarray, optional): Validation labels.

        Notes:
            Prints validation metrics if validation data is provided.
        """
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_metrics, _ = self.evaluate(X_val, y_val)
            print("Validation metrics:\n")
            for field, value in val_metrics.model_dump().items():
                if value is not None:
                    # If it's a confusion matrix, print it
                    if isinstance(value, np.ndarray):
                        print(f"{field}: \n{value}")
                    else:
                        print(f"{field}: {value:.4f}" if isinstance(value, float) else f"{field}: {value}")

    def predict(self, X_test) -> np.ndarray:
        """
        Generate predictions for the given test data.

        Args:
            X_test (np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test) -> Tuple[EvaluationMetrics, np.array]:
        """
        Evaluate the model on test data and compute performance metrics.

        Args:
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): True test labels.

        Returns:
            Tuple[EvaluationMetrics, np.ndarray]:
                - EvaluationMetrics object containing accuracy, precision, recall, f1-score, and confusion matrix.
                - Numpy array of predicted labels.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm
        ), y_pred
    