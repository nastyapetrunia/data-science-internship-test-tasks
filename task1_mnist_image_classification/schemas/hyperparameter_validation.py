from typing import Optional, Annotated, Tuple, List

from pydantic import BaseModel, Field, field_validator


class RFParams(BaseModel):
    """
    Hyperparameters for RandomForestMnistClassifier.

    Attributes:
        n_estimators (int): Number of trees in the forest. Must be > 0. Default is 100.
        max_depth (Optional[int]): Maximum depth of each tree. If None, nodes are expanded until all leaves are pure. Default is None.
        random_state (Optional[int]): Random seed for reproducibility. Default is 42.
        verbose (Optional[int]): Controls verbosity when fitting the trees. Default is 0 (silent).
    """
    n_estimators: Annotated[int, Field(strict=True, gt=0)] = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = 42
    verbose: Optional[int] = 0

class NNParams(BaseModel):
    """
    Hyperparameters for the Feed-Forward Neural Network classifier.

    Attributes:
        input_shape (Tuple[int, int]): 
            Shape of each input sample (height, width). Default is (28, 28).
        
        hidden_units (list[tuple[int, Optional[float]]]): 
            Specification of hidden layers as a list of (units, dropout_rate) tuples. 
            - `units` (int): Number of neurons in the layer. 
            - `dropout_rate` (float or None): Dropout rate to apply after this layer. 
              If None or 0, no dropout is applied. 
            Default is [(128, 0.2), (64, 0.2)].
        
        learning_rate (float): 
            Learning rate for the Adam optimizer. Default 0.001.
        
        epochs (int): 
            Number of training epochs. Default 10.
        
        batch_size (int): 
            Training batch size. Default 32.
        
        random_state (Optional[int]): 
            Random seed for reproducibility. Default 42.
    """
    input_shape: Tuple[int, int] = (28, 28)
    hidden_units: list[tuple[int, Optional[float]]] = [(128, 0.2), (64, 0.2)]
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    random_state: Optional[int] = 42

    @field_validator("hidden_units")
    @classmethod
    def validate_dropout(cls, v):
        normalized = []
        for units, dropout in v:
            if units <= 0:
                raise ValueError(f"Number of units must be > 0, got {units}")
            if dropout == 0:  
                dropout = None 
            elif dropout is not None and not (0 < dropout <= 1):
                raise ValueError(f"Dropout rate must be between 0 and 1, got {dropout}")
            normalized.append((units, dropout))
        return normalized
    
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
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    conv_layers: List[ConvLayerConfig] = [
        ConvLayerConfig(filters=32, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2)),
        ConvLayerConfig(filters=64, kernel_size=(3, 3), dropout=0.25, pool_size=(2, 2))
    ]
    dense_units: list[tuple[int, Optional[float]]] = [(128, 0.5)]
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    random_state: Optional[int] = 42
