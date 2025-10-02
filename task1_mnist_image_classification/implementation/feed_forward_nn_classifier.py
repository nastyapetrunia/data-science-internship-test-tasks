from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from schemas.hyperparameter_validation import NNParams
from schemas.evaluation_metrics import EvaluationMetrics
from interface.mnist_classifier_interface import MnistClassifierInterface


class FeedForwardNNMnistClassifier(MnistClassifierInterface):
    """
    Feed-Forward Neural Network (FFNN) implementation for MNIST classification.
    Uses fully connected layers with optional dropout as specified in NNParams.
    """

    def __init__(self, params: NNParams = None):
        if params is None:
            params = NNParams()
        self.params = params
        self.model = self._build_model()
        self.recent_history = None

    def _build_model(self):
        """
        Build a feed-forward neural network model based on the hyperparameters.
        Ensures reproducibility by setting TensorFlow random seed if specified.
        """
        if self.params.random_state is not None:
            tf.random.set_seed(self.params.random_state)

        model = Sequential()
        model.add(Input(shape=self.params.input_shape))
        model.add(Flatten())

        for units, dropout in self.params.hidden_units:
            model.add(Dense(units, activation="relu"))
            if dropout is not None:
                model.add(Dropout(dropout))

        model.add(Dense(10, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=self.params.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray = None, 
        y_val: np.ndarray = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model on training data with optional validation set.
        """
        early_stop = EarlyStopping(
            monitor='val_loss',  
            patience=5,         
            restore_best_weights=True 
        )
        
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            verbose=1,
            callbacks=[early_stop]
        )
        self.recent_history = history
        return history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.
        """
        y_pred_probs = self.model.predict(X_test)
        return np.argmax(y_pred_probs, axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[EvaluationMetrics, np.ndarray]:
        """
        Evaluate the model on test data and return metrics.
        """
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm
        ), y_pred
    
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
