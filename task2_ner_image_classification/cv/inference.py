import os
from glob import glob
from pathlib import Path

import tensorflow as tf

from task2_ner_image_classification.utils.general import print_section_separator
from task2_ner_image_classification.cv.utils.general import load_config
from task2_ner_image_classification.cv.utils.general import map_indices_to_names
from task2_ner_image_classification.cv.utils.data_preprocess import resize_images
from task2_ner_image_classification.cv.implementation.base_cnn import CNNClassifier
from task2_ner_image_classification.cv.implementation.pretrained_models import TransferLearningBase


def run_cv_inference(config_path: str, data_path: str):

    print_section_separator(100, start=True)
    print("Running CV inference...\n\n\n")

    print("    Loading config...")
    cfg = load_config(config_path)
    output = cfg.get("output")
    checkpoint_dir = output.get("checkpoint_dir")
    run_inference_using = output.get("run_inference_using")
    print("    Config loaded.")

    # -----------------------------
    # Load model
    # -----------------------------
    print("\n    Loading classifier...")
    if run_inference_using == "basecnn":
        model = CNNClassifier(load_from_path=f"{checkpoint_dir}/basecnn_classifier.keras")
    else: 
        model = TransferLearningBase(load_from_path=f"{checkpoint_dir}/{run_inference_using}_classifier.keras")

    print("    Classifier loaded.")

    # -----------------------------
    # Load and preprocess dataset
    # -----------------------------
    print("\n    Loading and resizing images...")

    imgs, img_paths = resize_images(data_path, target_size=(224, 224), keep_aspect_ratio=True, override_imgs=False)

    print("    Images loaded and resized.")

    # -----------------------------
    # Predict
    # -----------------------------

    print("\n    Predicting...")
    predictions = model.predict(imgs)
    predictions_mapped = map_indices_to_names(predictions)
    results = list(zip(img_paths, predictions_mapped))
    print("        Predictions:", results)

    print("\n\n\nCV inference completed.")
    print_section_separator(100, start=False)
    return results


if __name__ == "__main__": 
    run_cv_inference("task2_ner_image_classification/config/cv_config.yaml", "/Users/anastasiiapetrunia/Downloads/2.png")
