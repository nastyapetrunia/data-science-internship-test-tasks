from typing import Optional

import numpy as np
from pydantic import BaseModel

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
        