from task2_ner_image_classification.ner.inference import run_ner_inference
from task2_ner_image_classification.cv.inference import run_cv_inference 
from task2_ner_image_classification.ner.utils.general import is_entity_negated


class AnimalVerificationPipeline:
    def __init__(self, ner_config, cv_config):
        self.ner_config = ner_config
        self.cv_config = cv_config

    def extract_animals_from_text(self, text: str):
        return run_ner_inference(config_path=self.ner_config, text=text)

    def predict_image_class(self, image_path: str):
        return run_cv_inference(config_path=self.cv_config, data_path=image_path)

    def run(self, text: str, image_path: str) -> bool:
        canonical_animals, _ = self.extract_animals_from_text(text)
        _, image_animal = self.predict_image_class(image_path)[0]

        final_entities = []
        for entity in canonical_animals:
            negated = is_entity_negated(entity, text)
            final_entities.append((entity, negated))

        for entity, negated in final_entities:
            if negated and entity == image_animal:
                # text says "no X", but image has X → False
                return False
            if not negated and entity == image_animal:
                # text says X and image has X → True
                return True

        # If there are positive entities but none match the image → False
        if any(not negated for _, negated in final_entities):
            return False

        # No positive entities at all → True
        return True
