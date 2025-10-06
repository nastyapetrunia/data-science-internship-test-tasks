import argparse
from task2_ner_image_classification.pipeline import AnimalVerificationPipeline

def main():
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

    pipeline = AnimalVerificationPipeline(args.ner_config, args.cv_config)

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
    sentence = args.sentence or sentences[-1]
    image = args.image or images[3]

    print(f"\nRunning pipeline...")
    print(f"Sentence: {sentence}")
    print(f"Image: {image}\n")

    result = pipeline.run(sentence, image)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
