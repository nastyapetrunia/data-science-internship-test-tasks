from dotenv import load_dotenv

import argparse
from pathlib import Path

from task2_ner_image_classification.pipeline.pipeline import AnimalVerificationPipeline

load_dotenv()

def main(ner_config_path: str, cv_config_path: str, sentence: str, image_path: str):
    pipeline = AnimalVerificationPipeline(ner_config_path, cv_config_path)

    # Example data for convenience
    sentences = [
        "I saw a kitty playing with a puppy in the garden.",
        "A heifer and a hen were grazing near the barn.",
        "Look at that tarantula crawling over the branch next to the pony.",
        "The lamb and the chipmunk were both nibbling on leaves.",
        "There is no elephant but a butterfly here.",
        "There is an elephant but no butterfly here.",
        "I see no butterfly here.",
        "I see no elephant here.",
        "I see a cat and no elephant here.",
    ]

    images = [
        "task2_ner_image_classification/data/test_pipeline/1.png",
        "task2_ner_image_classification/data/test_pipeline/2.png",
        "task2_ner_image_classification/data/test_pipeline/3.jpeg",
        "task2_ner_image_classification/data/test_pipeline/4.jpeg",
        "task2_ner_image_classification/data/test_pipeline/5.jpeg",
        "task2_ner_image_classification/data/test_pipeline/6.png",
    ]

    # Use provided or default
    if not sentence:
        sentence = sentences[-1]
        print(f"No sentence provided. Falling back to default: '{sentence}'")

    if not image_path:
        image_path = images[3]
        print(f"No image path provided. Falling back to default: '{image_path}'")

    print(f"\nRunning pipeline...")
    print(f"Sentence: {sentence}")
    print(f"Image: {image_path}\n")

    result = pipeline.run(text=sentence, image_path=image_path)
    print(f"Result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Animal Verification Pipeline.")
    
    parser.add_argument(
        "--ner_config",
        type=str,
        default="task2_ner_image_classification/config/ner_config.yaml",
        help="Path to NER config file.",
    )
    parser.add_argument(
        "--cv_config",
        type=str,
        default="task2_ner_image_classification/config/cv_config.yaml",
        help="Path to CV config file.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        help="Sentence to analyze. If not provided, a default example will be used.",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image path for classification. If not provided, a default example will be used.",
    )

    args = parser.parse_args()

    cv_config_path = args.cv_config or input("Enter path to config file: ")
    ner_config_path = args.ner_config or input("Enter path to config file: ")
    sentence = args.sentence or input("Enter sentence: ")
    image_path = args.image or input("Enter image path: ")


    if not Path(cv_config_path).exists():
        raise FileNotFoundError(f"Config file not found: {cv_config_path}")
    if not Path(ner_config_path).exists():
        raise FileNotFoundError(f"Data path not found: {ner_config_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Data path not found: {image_path}")
    
    main(ner_config_path, cv_config_path, sentence, image_path)
