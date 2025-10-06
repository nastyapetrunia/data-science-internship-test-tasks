from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from task2_ner_image_classification.schemas.cv_schemas import EvaluationMetrics

class ClassifierBaseCV(ABC):
    @abstractmethod
    def _build_model(self):
        """Build and compile the model."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Return a string representation of the model."""
        pass

    def train(self, train_ds, val_ds=None) -> tf.keras.callbacks.History:
        """
        Train the CNN on training data.
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Cannot train: model was loaded without params")
        
        early_stop = EarlyStopping(
            monitor='val_loss',  
            patience=5,         
            restore_best_weights=True 
        )
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.params.epochs,
            verbose=1, 
            callbacks=[early_stop]
        )
        self.recent_history = history
        return history

    def predict(self, X_test: Union[np.ndarray, tf.data.Dataset]) -> np.ndarray:
        """
        Predict labels for test data.
        """
        y_pred_probs = self.model.predict(X_test, verbose=0)
        return np.argmax(y_pred_probs, axis=1)

    def evaluate(
        self,
        X_test: Union[np.ndarray, tf.data.Dataset],
        y_test: Optional[np.ndarray] = None
    ) -> tuple[EvaluationMetrics, np.ndarray]:
        """
        Evaluate the model and return metrics + predictions.
        
        Works with both NumPy arrays (X_test, y_test) and tf.data.Dataset.

        Args:
            X_test: np.ndarray of test data or tf.data.Dataset yielding (x, y)
            y_test: np.ndarray of true labels (only needed for NumPy arrays)

        Returns:
            metrics: EvaluationMetrics
            y_pred: Predicted labels
        """
        y_true_all = []
        y_pred_all = []

        if isinstance(X_test, tf.data.Dataset):
            for x_batch, y_batch in tqdm(X_test):
                y_pred_batch = self.model.predict(x_batch)
                y_pred_labels = np.argmax(y_pred_batch, axis=1)
                y_pred_all.extend(y_pred_labels)
                y_true_all.extend(y_batch.numpy())
        else:
            y_pred_probs = self.model.predict(X_test)
            y_pred_all = np.argmax(y_pred_probs, axis=1)
            y_true_all = y_test

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_true_all, y_pred_all),
            precision=precision_score(y_true_all, y_pred_all, average="macro"),
            recall=recall_score(y_true_all, y_pred_all, average="macro"),
            f1_score=f1_score(y_true_all, y_pred_all, average="macro"),
            confusion_matrix=confusion_matrix(y_true_all, y_pred_all)
        )

        return metrics, y_pred_all
        
    def save(self, path):
        """Save the model to the given path."""
        self.model.save(path)
    
    def _load(self, path):
        """Load the model from the given path."""
        try:
            self.model = tf.keras.models.load_model(path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def plot_history(self, history: Optional[tf.keras.callbacks.History] = None) -> None:
        """
        Plot training and validation accuracy and loss per epoch.

        Args:
            history: The History object returned by the train() method (tf.keras.callbacks.History).
        """
        history = history or getattr(self, "recent_history", None)
        if history is None:
            print("No recent history available. Train the model first.")
            return
        
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title("Accuracy per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train_loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='val_loss')
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()