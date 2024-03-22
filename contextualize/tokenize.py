import tiktoken


def call_tiktoken(text: str, encoding_str="cl100k_base", model_str=None):
    """
    Count the number of tokens in the provided string with tiktoken.

    Args:
        text (str): The text to count tokens for
        encoding_str: The encoding to use. "cl100k_base" for GPT-4/3.5-turbo, "p50k_base" for `text-davinci-003` and `code-davinci-002`, "r50k_base" for previous `davinci`/earlier GPT-3 models
        model_str: Model string to use for fetching an encoding for if `encoding_str` is not provided

    Returns:
        dict: A dictionary containing the tokens, count, and encoding used"
    """
    if encoding_str:
        encoding = tiktoken.get_encoding(encoding_str)
    else:
        if not model_str:
            raise ValueError("Model or encoding must be provided")

        encoding = tiktoken.encoding_for_model(model_str)

    tokens = encoding.encode(text)
    return {"tokens": tokens, "count": len(tokens), "encoding": encoding.name}
