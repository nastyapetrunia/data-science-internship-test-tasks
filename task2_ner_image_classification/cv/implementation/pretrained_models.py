from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from task2_ner_image_classification.schemas.cv_schemas import TransferLearningParams
from task2_ner_image_classification.cv.interface.cv_classifier_base import ClassifierBaseCV


class TransferLearningBase(ClassifierBaseCV):
    """
    Transfer learning classifier that builds a model based on TransferLearningParams.
    """

    def __init__(self, params: TransferLearningParams = None, load_from_path: Optional[str] = None):
        if load_from_path:
            self._load(path=load_from_path)
        else:
            self.params = params
            self.model = self._build_model()
        self.recent_history = None

    def _get_base_model(self, weights="imagenet", include_top=False) -> tf.keras.Model:
        """
        Map the Literal base_model name to the corresponding Keras application.
        """
        base_models = {
            "resnet50": tf.keras.applications.ResNet50,
            "resnet101": tf.keras.applications.ResNet101,
            "efficientnetb0": tf.keras.applications.EfficientNetB0,
            "mobilenetv2": tf.keras.applications.MobileNetV2,
            "vgg16": tf.keras.applications.VGG16,
        }

        if self.params.base_model not in base_models:
            raise ValueError(f"Unsupported base model: {self.params.base_model}")
        
        return base_models[self.params.base_model](
            weights=weights,
            include_top=include_top,
            input_shape=self.params.input_shape
        )

    def _build_model(self):
        """Build the full transfer learning model based on params."""
        if self.params.random_state is not None:
            tf.random.set_seed(self.params.random_state)
        
        base_model = self._get_base_model()
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.params.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        for units, dropout in self.params.dense_units:
            x = layers.Dense(units, activation="relu")(x)
            if dropout:
                x = layers.Dropout(dropout)(x)

        outputs = layers.Dense(self.params.num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs, name=f"{self.params.base_model}_classifier")

        if self.params.fine_tune_at is not None:
            for layer in model.layers[:self.params.fine_tune_at]:
                layer.trainable = False
            for layer in model.layers[self.params.fine_tune_at:]:
                layer.trainable = True

        optimizer = optimizers.Adam(learning_rate=self.params.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
    
    def to_string(self) -> str:
        return f"{self.params.base_model}_classifier"


class ResNet50Classifier(TransferLearningBase):
    """ResNet50-based classifier with default TransferLearningParams."""
    def __init__(self, params: TransferLearningParams):
        params = params.model_copy(update={"base_model": "resnet50"})
        super().__init__(params)


class ResNet101Classifier(TransferLearningBase):
    """ResNet101-based classifier with default TransferLearningParams."""
    def __init__(self, params: TransferLearningParams):
        params = params.model_copy(update={"base_model": "resnet101"})
        super().__init__(params)


class EfficientNetB0Classifier(TransferLearningBase):
    """EfficientNetB0-based classifier with default TransferLearningParams."""
    def __init__(self, params: TransferLearningParams):
        params = params.model_copy(update={"base_model": "efficientnetb0"})
        super().__init__(params)


class MobileNetV2Classifier(TransferLearningBase):
    """MobileNetV2-based classifier with default TransferLearningParams."""
    def __init__(self, params: TransferLearningParams):
        params = params.model_copy(update={"base_model": "mobilenetv2"})
        super().__init__(params)


class VGG16Classifier(TransferLearningBase):
    """VGG16-based classifier with default TransferLearningParams."""
    def __init__(self, params: TransferLearningParams):
        params = params.model_copy(update={"base_model": "vgg16"})
        super().__init__(params)

pretrained_mapping = {
    "efficientnetb0": EfficientNetB0Classifier,
    "resnet50": ResNet50Classifier,
    "resnet101": ResNet101Classifier,
    "mobilenetv2": MobileNetV2Classifier,
    "vgg16": VGG16Classifier,
}
