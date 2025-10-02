from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from schemas.hyperparameter_validation import CNNParams
from schemas.evaluation_metrics import EvaluationMetrics
from interface.mnist_classifier_interface import MnistClassifierInterface


class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self, params: CNNParams = None):
        if params is None:
            params = CNNParams()
        self.params = params
        self.model = self._build_model()
        self.recent_history = None

    def _build_model(self):
        """
        Build a Convolutional Neural Network model based on the hyperparameters.
        """
        if self.params.random_state is not None:
            tf.random.set_seed(self.params.random_state)

        model = Sequential()
        model.add(Input(shape=self.params.input_shape))
        
        for conv_cfg in self.params.conv_layers:
            model.add(Conv2D(filters=conv_cfg.filters, kernel_size=conv_cfg.kernel_size, activation="relu", padding="same"))
            if conv_cfg.pool_size is not None:
                model.add(MaxPooling2D(pool_size=conv_cfg.pool_size))
            if conv_cfg.dropout is not None:
                model.add(Dropout(conv_cfg.dropout))
        
        model.add(Flatten())
        
        for units, dropout in self.params.dense_units:
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

    def train(self, X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray = None, 
              y_val: np.ndarray = None) -> tf.keras.callbacks.History:
        """
        Train the CNN on training data.
        """
        early_stop = EarlyStopping(
            monitor='val_loss',  
            patience=5,         
            restore_best_weights=True 
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            verbose=1, 
            callbacks=[early_stop]
        )
        self.recent_history = history
        return history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data.
        """
        y_pred_probs = self.model.predict(X_test)
        return np.argmax(y_pred_probs, axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[EvaluationMetrics, np.ndarray]:
        """
        Evaluate model on test data and return evaluation metrics + predictions.
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
