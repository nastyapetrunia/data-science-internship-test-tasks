from task2_ner_image_classification.ner.inference import run_ner_inference
from task2_ner_image_classification.cv.inference import run_cv_inference 


class AnimalVerificationPipeline:
    def __init__(self, ner_config, cv_config):
        self.ner_config = ner_config
        self.cv_config = cv_config

    def extract_animals_from_text(self, text: str):
        return run_ner_inference(text, self.ner_config)

    def predict_image_class(self, image_path: str):
        return run_cv_inference(self.cv_config, image_path)

    def run(self, text: str, image_path: str) -> bool:
        canonical_animals, _ = self.extract_animals_from_text(text)
        _, image_animal = self.predict_image_class(image_path)[0]
        return image_animal in canonical_animals
    
