from typing import Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D

from task2_ner_image_classification.schemas.cv_schemas import CNNParams
from task2_ner_image_classification.cv.interface.cv_classifier_base import ClassifierBaseCV


class CNNClassifier(ClassifierBaseCV):
    def __init__(self, params: CNNParams = None, load_from_path: Optional[str] = None):
        if load_from_path:
            self._load(path=load_from_path)
        else:
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
        
        model.add(Dense(self.params.num_classes, activation="softmax"))
        
        model.compile(
            optimizer=Adam(learning_rate=self.params.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def to_string(self) -> str:
        return "cnn_classifier"
