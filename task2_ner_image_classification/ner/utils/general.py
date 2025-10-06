import re
import yaml

NEGATIONS = {"no", "not", "none", "never", "without", "doesn't", "isn't", "aren't"}


def is_entity_negated(entity: str, text: str) -> bool:
    """
    Returns True if the entity is negated in the text.
    Looks for negation words before the entity in the sentence, separated by punctuation.
    """
    text_lower = text.lower()
    words = re.findall(r"\w+|[^\w\s]", text_lower)
    entity_lower = entity.lower()
    
    for i, word in enumerate(words):
        if entity_lower == word:
            for prev_word in reversed(words[:i]):
                if prev_word in {",", ".", ";", "!"}:  
                    break
                if prev_word in NEGATIONS:
                    return True
    return False


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def find_entity_span(text, entity_text, label):
    start = text.find(entity_text)
    if start == -1:
        return None
    end = start + len(entity_text)
    return [start, end, label]

entities = [   "cat", "kitty",
    "dog","puppy",
    "cow", "heifer",
    "chicken", "hen",
    "spider", "tarantula",
    "butterfly", "butterfly",
    "horse", "pony",
    "sheep", "lamb",
    "squirrel", "chipmunk",
    "elephant", "pachyderm"]


def find_entities(train_data, entities = entities, label = "ANIMAL"):
    results = []
    for text, _ in train_data:
        curr_entities = []
        for entity in entities:
            span = find_entity_span(text, entity, label)
            if span is not None:
                curr_entities.append(span)
        results.append({"text": text, "entities": curr_entities})
    return results
