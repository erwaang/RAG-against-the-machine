import re


def tokenize(text: str) -> list[str]:
    """
    Tokenizes a string into a list of words.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: A list of words extracted from the input string.
    """
    text = re.sub(r'_', ' ', text)   # Replace underscores with spaces
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    # Add space before capital letters in camelCase
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    # \w = [a-zA-Z0-9_]
    # \s = whitespace characters ([' ', '\t', '\n'], etc.)
    return text.lower().split()
