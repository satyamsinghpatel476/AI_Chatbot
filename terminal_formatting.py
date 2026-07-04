import re


HEADINGS = [
    "Direct Answer",
    "Short Explanation",
    "Explanation",
    "Key Points",
    "Practical Advice or Example",
    "Practical Advice",
    "Example",
    "Important Difference",
    "Final Conclusion",
    "Conclusion",
    "Robotics Perspective",
    "Daily-Life Perspective",
    "Relationship Type",
]

ABBREVIATIONS = [
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "U.S.",
    "U.K.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Prof.",
    "ROS2",
    "No.",
    "Fig.",
    "Eq.",
]


def _protect_abbreviations(text):
    replacements = []
    protected = text
    for index, abbreviation in enumerate(ABBREVIATIONS):
        token = f"__TERM_ABBR_{index}__"
        replacements.append((token, abbreviation))
        protected = protected.replace(abbreviation, token)
    return protected, replacements


def _restore_abbreviations(text, replacements):
    restored = text
    for token, abbreviation in replacements:
        restored = restored.replace(token, abbreviation)
    return restored


def _format_plain_segment(text):
    if not text:
        return ""

    formatted, replacements = _protect_abbreviations(str(text).replace("\r\n", "\n"))
    heading_pattern = "|".join(re.escape(heading) for heading in HEADINGS)

    formatted = re.sub(
        rf"\b({heading_pattern}):\s*",
        lambda match: (
            ("\n\n" if match.start() and not formatted[:match.start()].endswith("\n") else "")
            + f"{match.group(1)}:\n"
        ),
        formatted,
        flags=re.IGNORECASE,
    )
    formatted = re.sub(r"(?<!\d)([.!?])[ \t]+(?=[\"'“‘(\[]?[A-Z0-9])", r"\1\n", formatted)
    formatted = re.sub(r"(^|[^\n])[ \t]+([1-9]|10)\.[ \t]+", r"\1\n\2. ", formatted)
    formatted = re.sub(r"([:;.!?])[ \t]+(-[ \t]+(?=\S))", r"\1\n\2", formatted)
    formatted = re.sub(r"(^|[^\n])[ \t]+(•[ \t]+)", r"\1\n\2", formatted)

    formatted = _restore_abbreviations(formatted, replacements)
    formatted = re.sub(r"[ \t]+\n", "\n", formatted)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def format_terminal_answer(text: str) -> str:
    """Format an assistant answer for terminal display without changing raw data."""
    parts = re.split(r"(```[\s\S]*?```)", str(text or ""))
    formatted_parts = [
        part if part.startswith("```") else _format_plain_segment(part)
        for part in parts
    ]
    formatted = "\n".join(part for part in formatted_parts if part)
    return re.sub(r"\n{3,}", "\n\n", formatted).strip()
