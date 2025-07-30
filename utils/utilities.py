def contains(input: str, values: list[str]) -> bool:
    return any(value in input for value in values)
