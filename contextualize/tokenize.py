import os
from typing import Dict, Optional, Union


def count_tokens(text: str, target: str = "cl100k_base") -> Dict[str, Union[int, str]]:
    """
    Count tokens using either Anthropic's API or tiktoken, based on the 'target'.

    If the target string includes 'claude' and ANTHROPIC_API_KEY is set, attempt
    Anthropic's token counting. Otherwise, fall back to tiktoken.
    """
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if "claude" in target.lower() and anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_api_key)
            response = client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=target,
                messages=[{"role": "user", "content": text}],
            )
            return {"count": response.input_tokens, "method": f"anthropic-{target}"}
        except Exception as e:
            print(f"Error using Anthropic API: {str(e)}. Falling back to tiktoken.")

    # Fall back to tiktoken
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

    If `encoding_str` is None but `model_str` is provided, detect the encoding for that model.
    """
    import tiktoken

    if encoding_str:
        encoding = tiktoken.get_encoding(encoding_str)
    elif model_str:
        encoding = tiktoken.encoding_for_model(model_str)
    else:
        raise ValueError("Must provide an encoding_str or a model_str")

    tokens = encoding.encode(text)
    return {"tokens": tokens, "count": len(tokens), "encoding": encoding.name}
