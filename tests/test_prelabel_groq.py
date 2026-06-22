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


import csv


class _FakeMessage:
    def __init__(self, content): self.message = type("M", (), {"content": content})


class _FakeCompletions:
    def __init__(self, reply): self._reply = reply
    def create(self, **kwargs):
        assert kwargs["temperature"] == 0
        return type("R", (), {"choices": [_FakeMessage(self._reply)]})


class FakeClient:
    def __init__(self, reply):
        self.chat = type("C", (), {"completions": _FakeCompletions(reply)})


def test_classify_returns_parsed_label():
    assert pl.classify("x", FakeClient("hot_take"), sleep_s=0) == "hot_take"


def test_classify_unparseable_returns_empty():
    assert pl.classify("x", FakeClient("no idea"), sleep_s=0) == ""


def test_prelabel_rows_initializes_label_to_suggestion(tmp_path):
    src = tmp_path / "raw.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "score", "source"])
        w.writeheader()
        w.writerow({"text": "Jokic gravity is real per the numbers", "score": 5, "source": "t1"})
    rows = pl.prelabel_rows(str(src), FakeClient("analysis"), "m", sleep_s=0)
    assert rows[0]["suggested_label"] == "analysis"
    assert rows[0]["label"] == "analysis"
    assert rows[0]["notes"] == ""
    assert rows[0]["text"].startswith("Jokic")
