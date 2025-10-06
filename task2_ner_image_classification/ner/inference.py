from task2_ner_image_classification.utils.general import print_section_separator
from task2_ner_image_classification.ner.utils.general import load_config
from task2_ner_image_classification.ner.implementation.transformer_ner import TransformerNER
from task2_ner_image_classification.ner.utils.entity_mapper import map_entities_to_canonical

def run_ner_inference(text: str, config_path: str):
    print_section_separator(50, start=True)
    print("Running NER inference...\n\n\n")
    config = load_config(config_path)
    ner_model = TransformerNER()
    ner_model.load(config["model"]["save_dir"])

    entities = [e[0] for e in ner_model.predict(text)]
    canonical_animals = map_entities_to_canonical(entities)

    print(f"    Detected animals:\n        canonical: {canonical_animals}\n        entities: {entities}")

    print("\n\n\nNER inference completed.")
    print_section_separator(50, start=False)
    
    return canonical_animals, entities

if __name__ == "__main__":
    run_ner_inference("Look at that kitty and puppy! Or a cat with a small elephant and a frog.", "task2_ner_image_classification/config/ner_config.yaml")
