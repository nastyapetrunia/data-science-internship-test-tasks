from typing import Optional, Annotated
from pydantic import BaseModel, Field


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
