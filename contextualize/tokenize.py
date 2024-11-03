import os
from typing import Dict, Optional, Union

import anthropic
import tiktoken


def count_tokens(text: str, target: str = "cl100k_base") -> Dict[str, Union[int, str]]:
    """
    Count tokens using either Anthropic's API or tiktoken based on the target.

    Args:
        text (str):   The text to count tokens for
        target (str): Either an Anthropic model name (containing 'claude') or
                      a tiktoken encoding name. Defaults to "cl100k_base"

    Returns:
        dict: A dictionary containing the token count and method used
    """
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if "claude" in target.lower() and anthropic_api_key:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        try:
            response = client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=target,
                messages=[{"role": "user", "content": text}],
            )
            return {"count": response.input_tokens, "method": f"anthropic-{target}"}
        except Exception as e:
            print(f"Error using Anthropic API: {str(e)}. Falling back to tiktoken.")

    # fall back to tiktoken if Anthropic is not available or fails
    result = call_tiktoken(
        text, encoding_str=target if "claude" not in target.lower() else "cl100k_base"
    )
    return {"count": result["count"], "method": f"{result['encoding']}"}


def call_tiktoken(
    text: str,
    encoding_str: Optional[str] = "cl100k_base",
    model_str: Optional[str] = None,
):
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
    elif model_str:
        encoding = tiktoken.encoding_for_model(model_str)
    else:
        raise ValueError("Model or encoding must be provided")

    tokens = encoding.encode(text)
    return {"tokens": tokens, "count": len(tokens), "encoding": encoding.name}
