from typing import Optional, Tuple, List, Literal

import numpy as np
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """
    Container for evaluation metrics of a classifier.

    Attributes:
        accuracy (float): Overall accuracy of the model.
        precision (Optional[float]): Precision score. Default is None.
        recall (Optional[float]): Recall score. Default is None.
        f1_score (Optional[float]): F1 score. Default is None.
        loss (Optional[float]): Optional loss value (if applicable). Default is None.
        confusion_matrix (Optional[np.ndarray]): Confusion matrix as a NumPy array. Default is None.

    Notes:
        - `arbitrary_types_allowed` is enabled in Config to allow storing NumPy arrays.
        - Any metric not computed can be left as None.
    """
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

class ConvLayerConfig(BaseModel):
    """
    Configuration for a single convolutional layer in a CNN.

    Attributes:
        filters (int): Number of filters (output channels) in the convolutional layer.
        kernel_size (Tuple[int, int]): Height and width of the convolutional kernel.
        dropout (Optional[float]): Dropout rate applied after this layer. If None, no dropout is applied.
        pool_size (Optional[Tuple[int, int]]): Size of the max pooling window. If None, no pooling is applied.
    """
    filters: int
    kernel_size: Tuple[int, int]
    dropout: Optional[float] = None
    pool_size: Optional[Tuple[int, int]] = None

class CNNParams(BaseModel):
    """
    Hyperparameters for the Convolutional Neural Network (CNN) classifier.

    Attributes:
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of input images (height, width, channels). Default is (28, 28, 1).
        conv_layers (List[ConvLayerConfig]): List of convolutional layer configurations.
            Each ConvLayerConfig defines:
                - filters: Number of convolutional filters
                - kernel_size: Tuple for kernel height and width
                - dropout: Optional dropout rate after this layer
                - pool_size: Optional max pooling size
            Default example: 
                [ConvLayerConfig(filters=32, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2)),
                 ConvLayerConfig(filters=64, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2))]
        dense_units (List[Tuple[int, Optional[float]]]): List of fully connected layers after flattening.
            Each tuple is (units, dropout_rate), where dropout_rate can be None.
            Default example: [(128, 0.5)]
        learning_rate (float): Learning rate for the Adam optimizer. Default 0.001.
        epochs (int): Number of training epochs. Default 10.
        batch_size (int): Training batch size. Default 32.
        random_state (Optional[int]): Random seed for reproducibility. Default 42.
    """
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (224, 224, 3)

    conv_layers: List[ConvLayerConfig] = [
        ConvLayerConfig(filters=32, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2)),
        ConvLayerConfig(filters=64, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2)),
        ConvLayerConfig(filters=128, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2)),
    ]

    dense_units: list[tuple[int, Optional[float]]] = [
        (256, 0.5),
        (128, 0.5)
    ]

    learning_rate: float = 0.0001
    epochs: int = 15
    batch_size: int = 32
    random_state: Optional[int] = 42

class TransferLearningParams(BaseModel):
    """
    Hyperparameters for transfer learning classifiers (ResNet, EfficientNet, etc.).

    Attributes:
        base_model (Literal): Name of the pretrained backbone.
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of input images (height, width, channels).
        dense_units (List[Tuple[int, Optional[float]]]): Fully connected layers after the base model.
            Each tuple is (units, dropout_rate), where dropout_rate can be None.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        fine_tune_at (Optional[int]): OPTIONAL layer index from which to start fine-tuning.
        random_state (Optional[int]): Random seed for reproducibility.
    """

    base_model: Literal[
        "resnet50",
        "resnet101",
        "efficientnetb0",
        "mobilenetv2",
        "vgg16"
    ] = Field(
        default="efficientnetb0",
        description="Pretrained backbone name from tf.keras.applications."
    )

    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (224, 224, 3)

    dense_units: List[Tuple[int, Optional[float]]] = Field(
        default=[(256, 0.5), (128, 0.25)],
        description="Dense layers after global pooling (units, dropout)."
    )

    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 32

    fine_tune_at: Optional[int] = None  

    random_state: Optional[int] = 42
    