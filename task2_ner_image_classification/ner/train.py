import json

from task2_ner_image_classification.ner.utils.general import load_config
from task2_ner_image_classification.ner.utils.data_loader import load_training_data
from task2_ner_image_classification.ner.implementation.transformer_ner import TransformerNER


if __name__ == "__main__":
    config = load_config("task2_ner_image_classification/config/ner_config.yaml")
    train_data_path = config["data"]["train_dir"]
    train_data = load_training_data(train_data_path)

    ner_model = TransformerNER()
    ner_model.train(train_data, n_iter=200)
    ner_model.save(config["model"]["save_dir"])
    print("NER model trained and saved.")
