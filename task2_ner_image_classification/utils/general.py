def print_section_separator(length: int, start = True):
    if start:
        print(f"{"#" * length}\n\n")
    else:
        print(f"\n\n{"#" * length}")
