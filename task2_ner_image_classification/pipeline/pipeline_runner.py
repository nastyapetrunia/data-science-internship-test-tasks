from dotenv import load_dotenv
from task2_ner_image_classification.pipeline.pipeline import AnimalVerificationPipeline

load_dotenv()


if __name__ == "__main__":
    pipeline = AnimalVerificationPipeline("task2_ner_image_classification/config/ner_config.yaml", "task2_ner_image_classification/config/cv_config.yaml")
    
    sentences = [
        "I saw a kitty playing with a puppy in the garden.",
        "A heifer and a hen were grazing near the barn.",
        "Look at that tarantula crawling over the branch next to the pony.",
        "The lamb and the chipmunk were both nibbling on leaves.",
        "There is no elephant but a butterfly here.",
        "There is an elephant but no butterfly here.",
        "I see no butterfly here.",
        "I see no elephant here.",
        "I see a cat and no elephant here."
    ]

    images = [
        "task2_ner_image_classification/data/test_pipeline/1.png",
        "task2_ner_image_classification/data/test_pipeline/2.png",
        "task2_ner_image_classification/data/test_pipeline/3.jpeg",
        "task2_ner_image_classification/data/test_pipeline/4.jpeg",
        "task2_ner_image_classification/data/test_pipeline/5.jpeg",
        "task2_ner_image_classification/data/test_pipeline/6.png"
    ]

    current_sentence = sentences[-1]
    current_image = images[3]
    
    result = pipeline.run(current_sentence, current_image)
    print(f"Result: {result}")
