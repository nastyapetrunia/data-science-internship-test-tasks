from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from a folder with subfolders for each class.
    
    Args:
        data_dir (str): Path to the dataset folder containing class subfolders.
    
    Returns:
        X (np.ndarray): Array of shape (num_samples, height, width, 3)
        y (np.ndarray): Array of integer labels corresponding to classes
    """
    data_dir = Path(data_dir)
    X: List[np.ndarray] = []
    y: List[int] = []
    
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.glob("*"):
            try:
                img = Image.open(img_path).convert("RGB")
                X.append(np.array(img) / 255.0)
                y.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"⚠️ Could not load image {img_path}: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y