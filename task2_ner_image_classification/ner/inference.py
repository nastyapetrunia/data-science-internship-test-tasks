import argparse
from pathlib import Path

from task2_ner_image_classification.utils.general import print_section_separator
from task2_ner_image_classification.ner.utils.general import load_config
from task2_ner_image_classification.ner.implementation.transformer_ner import TransformerNER
from task2_ner_image_classification.ner.utils.entity_mapper import map_entities_to_canonical

def run_ner_inference(config_path: str, text: str):
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
    parser = argparse.ArgumentParser(description="Run NER inference on input text.")
    parser.add_argument("--config", type=str, default=None, help="Path to NER config YAML file.")
    parser.add_argument("--text", type=str, default=None, help="Text to perform NER on.")
    args = parser.parse_args()

    config_path = args.config or input("Enter path to config file: ")
    text = args.text or input("Enter text for NER inference: ")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    canonical_animals, entities = run_ner_inference(config_path, text)
