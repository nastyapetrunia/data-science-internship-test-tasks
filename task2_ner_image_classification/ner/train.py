import argparse
from pathlib import Path

from task2_ner_image_classification.ner.utils.general import load_config
from task2_ner_image_classification.ner.utils.data_loader import load_training_data
from task2_ner_image_classification.ner.implementation.transformer_ner import TransformerNER

def train_ner_model(config_path: str, n_iter: int = 200):
    config = load_config(config_path)
    train_data_path = config["data"]["train_dir"]
    train_data = load_training_data(train_data_path)

    ner_model = TransformerNER()
    ner_model.train(train_data, n_iter=n_iter)
    ner_model.save(config["model"]["save_dir"])
    print("NER model trained and saved.")
    return ner_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NER model.")
    parser.add_argument("--config", type=str, default=None, help="Path to NER config YAML file.")
    parser.add_argument("--n_iter", type=int, default=200, help="Number of training iterations.")
    args = parser.parse_args()

    # Fallback to interactive input if not provided
    config_path = args.config or input("Enter path to config file: ")
    n_iter = args.n_iter

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    train_ner_model(config_path, n_iter)
