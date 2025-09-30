from abc import ABC, abstractmethod

import numpy as np

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the model using training data.
        Optionally, use validation data for early stopping or tuning.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels for the given test data."""
        pass
