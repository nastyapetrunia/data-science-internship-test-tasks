import yaml
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import (
    resnet, efficientnet, mobilenet_v2, vgg16
)

from task2_ner_image_classification.cv.utils.general import load_config
from task2_ner_image_classification.cv.implementation.base_cnn import CNNClassifier
from task2_ner_image_classification.schemas.cv_schemas import CNNParams, TransferLearningParams
from task2_ner_image_classification.cv.implementation.pretrained_models import pretrained_mapping

PREPROCESS_FUNCS = {
    "resnet50": resnet.preprocess_input,
    "resnet101": resnet.preprocess_input,
    "efficientnetb0": efficientnet.preprocess_input,
    "mobilenetv2": mobilenet_v2.preprocess_input,
    "vgg16": vgg16.preprocess_input,
}


def init_params(cfg: dict):
    """Initialize either CNNParams or TransferLearningParams from config."""
    use_default_config = cfg.get("use_default_config", True)
    model_type = cfg.get("model_type", "basecnn")

    if model_type.lower() == "basecnn":
        params = CNNParams()
        if not use_default_config:
            for k, v in cfg.get("cnn_params", {}).items():
                if v is not None and hasattr(params, k):
                    setattr(params, k, v)

    elif model_type.lower() == "transfer":
        params = TransferLearningParams()
        if not use_default_config:
            for k, v in cfg.get("transfer_params", {}).items():
                if v is not None and hasattr(params, k):
                    setattr(params, k, v)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return params


def main(config_path: str):
    print("Loading config...")
    cfg = load_config(config_path)
    model_type = cfg.get("model_type", "basecnn")
    print("Config loaded.")

    # -----------------------------
    # Initialize model params
    # -----------------------------
    print("Initializing model params...")
    params = init_params(cfg)
    print("Model params initialized.")

    # -----------------------------
    # Load and preprocess dataset
    # -----------------------------
    print("Loading dataset...")
    if cfg.get("use_dummy_data"):
        data = cfg["dummy_data"]
    else:
        data = cfg["data"]

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data["train_dir"],     
        image_size=(224, 224),
        batch_size=32,
        shuffle=True
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data["val_dir"],
        image_size=(224, 224),
        batch_size=32
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data["test_dir"],
        image_size=(224, 224),
        batch_size=32
    )

    if model_type == "transfer":
        preprocess_input = PREPROCESS_FUNCS.get(params.base_model)
        train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
        val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))
        test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))
    else:
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
        val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
        test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    print("Dataset loaded.")

    # -----------------------------
    # Initialize classifier
    # -----------------------------
    print("Initializing classifier...")
    if model_type == "basecnn":
        model = CNNClassifier(params=params)
    else: 
        model_class = pretrained_mapping.get(params.base_model)

        if model_class is None:
            raise ValueError(f"Unknown base_model: {params.base_model}")
        
        model = model_class(params=params)

    print("Classifier initialized.")

    # -----------------------------
    # Train
    # -----------------------------
    print("Training model...")
    history = model.train(train_dataset, val_dataset)

    print("Model trained.")

    # -----------------------------
    # Evaluate
    # -----------------------------

    print("Evaluating model...")
    metrics, _ = model.evaluate(test_dataset)
    print("Test Metrics:", metrics)

    print("Model evaluated.")

    # -----------------------------
    # Save model
    # -----------------------------
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / f"{model.to_string()}.keras")

    # Optional: plot training history
    model.plot_history(history)


if __name__ == "__main__": 
    main("task2_ner_image_classification/config/cv_config.yaml")
