import json

def load_training_data(path):
    """
    Load training data from JSON file.
    Format: [{"text": "There is a cat", "entities": [(10, 13, "ANIMAL")]}]
    """
    with open(path, "r") as f:
        data = json.load(f)
    return [(item["text"], {"entities": item["entities"]}) for item in data]
