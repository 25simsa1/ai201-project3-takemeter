# tests/test_prelabel_groq.py
import prelabel_groq as pl


def test_labels_are_exactly_three():
    assert pl.LABELS == ("analysis", "hot_take", "reaction")


def test_build_messages_includes_defs_and_text():
    msgs = pl.build_messages("LeBron is washed")
    joined = " ".join(m["content"] for m in msgs)
    assert "analysis" in joined and "hot_take" in joined and "reaction" in joined
    assert "LeBron is washed" in msgs[-1]["content"]


def test_parse_label_clean():
    assert pl.parse_label("hot_take") == "hot_take"
    assert pl.parse_label("analysis") == "analysis"


def test_parse_label_messy():
    assert pl.parse_label("Label: Hot Take") == "hot_take"
    assert pl.parse_label("The answer is reaction.") == "reaction"
    assert pl.parse_label("ANALYSIS\n") == "analysis"


def test_parse_label_unknown_returns_none():
    assert pl.parse_label("I am not sure") is None
    assert pl.parse_label("") is None
