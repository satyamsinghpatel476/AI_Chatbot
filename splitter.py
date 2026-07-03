import re

def split_query(text):
    """Split only clear multi-request prompts; preserve normal sentence meaning."""
    parts = re.split(
        r"\b(?:also|then|after that)\b|;|\n+",
        text,
        flags=re.IGNORECASE,
    )
    return [part.strip(" ,.!?") for part in parts if part.strip(" ,.!?")]
