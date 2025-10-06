import yaml
from task2_ner_image_classification.cv.utils.translate import translate

def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

sorted_class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
                      'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

def map_indices_to_names(pred_indices, to_language="en"):
    """
    Map predicted class indices to class names in either English or Italian.

    Args:
        pred_indices (list or np.ndarray): Predicted integer labels.
        to_language (str): "en" for English, "it" for Italian.

    Returns:
        List of class names in the requested language.
    """
    names = [sorted_class_names[i] for i in pred_indices]
    if to_language == "en":
        names = [translate[n] for n in names]
    return names