synonyms = {
    "cat": ["cat", "kitty"],
    "dog": ["dog", "puppy"],
    "cow": ["cow", "heifer"],
    "chicken": ["chicken", "hen"],
    "spider": ["spider", "tarantula"],
    "butterfly": ["butterfly", "monarch butterfly"],
    "horse": ["horse", "pony"],
    "sheep": ["sheep", "lamb"],
    "squirrel": ["squirrel", "chipmunk"],
    "elephant": ["elephant", "pachyderm"]
}

def map_entities_to_canonical(entities):
    """
    entities: list of entity strings from NER
    returns list of canonical animal names
    """
    canonical = set()
    for ent in entities:
        ent_lower = ent.lower()
        for animal, syns in synonyms.items():
            if any(s in ent_lower for s in syns):
                canonical.add(animal)
    return list(canonical)
