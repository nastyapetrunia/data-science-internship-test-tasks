from typing import Optional, Annotated, Tuple

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
