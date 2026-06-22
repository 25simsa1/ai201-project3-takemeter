LABELS = ("analysis", "hot_take", "reaction")

SYSTEM_PROMPT = """You classify NBA-community comments into exactly one label.

analysis: a structured argument backed by specific, verifiable evidence \
(stats, historical comparison, tactical or film observation). Strip the \
opinion framing and a reasoned claim still stands.
hot_take: a bold, confident opinion asserted without real support. It may \
cite a stat, but the stat is decorative or cherry-picked, not load-bearing. \
It asserts instead of arguing.
reaction: an in-the-moment emotional response to a specific event, play, or \
game. Little to no argument; it expresses a feeling.

Reply with one label only: analysis, hot_take, or reaction. No other words."""


def build_messages(text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_label(response_text):
    norm = (response_text or "").strip().lower().replace(" ", "_")
    for label in LABELS:
        if label in norm:
            return label
    return None
