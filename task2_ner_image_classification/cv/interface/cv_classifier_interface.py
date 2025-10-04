from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from task2_ner_image_classification.schemas.cv_schemas import EvaluationMetrics

class ClassifierInterfaceCV(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: np.ndarray = None, 
              y_val: np.ndarray = None) -> tf.keras.callbacks.History:
        """
        Train the model using training data.
        Optionally, use validation data for early stopping or tuning.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels for the given test data."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[EvaluationMetrics, np.ndarray]:
        """Evaluate the model on the given test data."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to the given path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from the given path."""
        pass
        
    @abstractmethod
    def plot_history(self, history: Optional[tf.keras.callbacks.History] = None) -> None:
        """Plot training and validation accuracy and loss per epoch."""
