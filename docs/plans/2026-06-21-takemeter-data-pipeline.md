# TakeMeter Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local pipeline that turns public r/nba threads into a clean, balanced, 200+ example labeled CSV ready to upload to the Colab fine-tuning notebook.

**Architecture:** Three small Python scripts plus a manual review step. `scrape_reddit.py` pulls comments from public Reddit thread JSON and writes raw text. `prelabel_groq.py` asks Groq for a suggested label per comment, keeping the suggestion frozen so overrides are measurable. The human review pass corrects the labels by hand. `dataset_stats.py` reports label balance and override rate, and exports the final `text,label,notes` CSV. Pure logic (parsing, cleaning, filtering, label parsing, counting) is unit-tested; thin HTTP and API wrappers are exercised by a live smoke run.

**Tech Stack:** Python 3.11+, `requests` (HTTP), `groq` (baseline-free pre-labeling), stdlib `csv`/`argparse`, `pytest` for tests.

## Global Constraints

- Python 3.11 or newer.
- Groq model is exactly `llama-3.3-70b-versatile`, called with `temperature=0` for repeatability.
- Labels are exactly these three strings, no others: `analysis`, `hot_take`, `reaction`.
- The final dataset CSV has exactly these columns in this order: `text`, `label`, `notes`. It is not pre-split.
- No secrets in the repo. The Groq key is read from the `GROQ_API_KEY` environment variable. `.env` and `.venv/` are gitignored.
- Public content only. Scraping sends a descriptive `User-Agent` and sleeps between requests to stay polite.
- The pre-labeling Groq call is not the project baseline. The baseline is a separate clean run inside the Colab notebook (Section 5).

---

### Task 1: Project setup and scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `conftest.py`
- Create: `data/.gitkeep`
- Create: `scripts/.gitkeep`
- Create: `tests/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable environment and a working `pytest` setup. `conftest.py` puts `scripts/` on `sys.path` so tests can `import scrape_reddit`, `import prelabel_groq`, `import dataset_stats`.

- [ ] **Step 1: Create `requirements.txt`**

```
requests>=2.31
groq>=0.11
pytest>=8.0
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.env
.DS_Store
```

- [ ] **Step 3: Create `conftest.py` at repo root**

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
```

- [ ] **Step 4: Create directory keepers**

Create empty files `data/.gitkeep`, `scripts/.gitkeep`, `tests/.gitkeep` so the directories exist in git.

- [ ] **Step 5: Create and populate the virtualenv**

Run:
```bash
cd ~/ai201-project3-takemeter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: pip installs requests, groq, pytest and their deps with no error.

- [ ] **Step 6: Verify pytest collects cleanly**

Run: `python -m pytest -q`
Expected: `no tests ran` (exit code 5 is fine) with no import or collection errors.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore conftest.py data/.gitkeep scripts/.gitkeep tests/.gitkeep
git commit -m "Set up data pipeline scaffolding"
```

---

### Task 2: Reddit comment extraction, cleaning, and filtering

**Files:**
- Create: `scripts/scrape_reddit.py`
- Test: `tests/test_scrape_reddit.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `extract_comments(listing_json: list) -> list[dict]` — walks the Reddit two-listing structure and the nested `replies`, returning a list of dicts `{"body": str, "score": int, "id": str}` for every `t1` (comment) node. Skips `more` and non-comment nodes.
  - `clean_comment(text: str) -> str` — trims, collapses all runs of whitespace (including newlines) to single spaces.
  - `LOW_VALUE = {"this", "lol", "same", "^", "^^this", "lmao", "facts", "deadass"}` — module-level set of throwaway one-word comments.
  - `is_valid_comment(comment: dict, min_chars: int = 8, max_chars: int = 1500) -> bool` — rejects deleted/removed/empty bodies, the `AutoModerator` author, lengths outside the bounds, and cleaned text whose lowercase form is in `LOW_VALUE`.
  - `dedupe(comments: list[dict]) -> list[dict]` — keeps the first occurrence per case-insensitive cleaned body.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scrape_reddit.py
import scrape_reddit as sr

SAMPLE = [
    {"kind": "Listing", "data": {"children": [
        {"kind": "t3", "data": {"title": "Post-Game Thread"}}
    ]}},
    {"kind": "Listing", "data": {"children": [
        {"kind": "t1", "data": {
            "body": "Top comment", "score": 42, "id": "a1", "author": "user1",
            "replies": {"data": {"children": [
                {"kind": "t1", "data": {
                    "body": "Nested reply", "score": 3, "id": "a2",
                    "author": "user2", "replies": ""}},
                {"kind": "more", "data": {"children": ["x", "y"]}},
            ]}}}},
        {"kind": "t1", "data": {
            "body": "[deleted]", "score": 1, "id": "a3",
            "author": "user3", "replies": ""}},
    ]}},
]


def test_extract_comments_flattens_tree():
    out = sr.extract_comments(SAMPLE)
    bodies = [c["body"] for c in out]
    assert bodies == ["Top comment", "Nested reply", "[deleted]"]
    assert out[0]["score"] == 42 and out[0]["id"] == "a1"


def test_extract_comments_skips_more_and_t3():
    out = sr.extract_comments(SAMPLE)
    assert all(c["id"] in {"a1", "a2", "a3"} for c in out)


def test_clean_comment_collapses_whitespace():
    assert sr.clean_comment("  hello\n\n  world\t!  ") == "hello world !"


def test_is_valid_comment_rejects_deleted_and_bot():
    assert not sr.is_valid_comment({"body": "[deleted]", "author": "u"})
    assert not sr.is_valid_comment({"body": "[removed]", "author": "u"})
    assert not sr.is_valid_comment({"body": "real take here", "author": "AutoModerator"})


def test_is_valid_comment_length_and_lowvalue():
    assert not sr.is_valid_comment({"body": "short", "author": "u"})  # 5 chars
    assert not sr.is_valid_comment({"body": "lol", "author": "u"})
    assert sr.is_valid_comment({"body": "This is a real comment", "author": "u"})
    assert not sr.is_valid_comment({"body": "x" * 2000, "author": "u"})


def test_dedupe_keeps_first_case_insensitive():
    rows = [
        {"body": "Same Take"}, {"body": "same take"}, {"body": "Different"}]
    out = sr.dedupe(rows)
    assert [r["body"] for r in out] == ["Same Take", "Different"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scrape_reddit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scrape_reddit'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/scrape_reddit.py
import re

LOW_VALUE = {"this", "lol", "same", "^", "^^this", "lmao", "facts", "deadass"}
_WS = re.compile(r"\s+")


def extract_comments(listing_json):
    """Flatten the Reddit two-listing comment tree into a list of dicts."""
    out = []
    if not isinstance(listing_json, list) or len(listing_json) < 2:
        return out
    children = listing_json[1].get("data", {}).get("children", [])

    def walk(nodes):
        for node in nodes:
            if node.get("kind") != "t1":
                continue
            data = node.get("data", {})
            out.append({
                "body": data.get("body", ""),
                "score": data.get("score", 0),
                "id": data.get("id", ""),
                "author": data.get("author", ""),
            })
            replies = data.get("replies")
            if isinstance(replies, dict):
                walk(replies.get("data", {}).get("children", []))

    walk(children)
    return out


def clean_comment(text):
    return _WS.sub(" ", (text or "").strip())


def is_valid_comment(comment, min_chars=8, max_chars=1500):
    body = comment.get("body", "") or ""
    if body in ("[deleted]", "[removed]", ""):
        return False
    if comment.get("author") == "AutoModerator":
        return False
    cleaned = clean_comment(body)
    if not (min_chars <= len(cleaned) <= max_chars):
        return False
    if cleaned.lower() in LOW_VALUE:
        return False
    return True


def dedupe(comments):
    seen = set()
    out = []
    for c in comments:
        key = clean_comment(c.get("body", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scrape_reddit.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/scrape_reddit.py tests/test_scrape_reddit.py
git commit -m "Add Reddit comment extraction and filtering"
```

---

### Task 3: Reddit fetch, CSV output, and CLI

**Files:**
- Modify: `scripts/scrape_reddit.py`
- Test: `tests/test_scrape_reddit.py`

**Interfaces:**
- Consumes: `extract_comments`, `clean_comment`, `is_valid_comment`, `dedupe` from Task 2.
- Produces:
  - `thread_url(ref: str) -> str` — turns a thread id or any reddit thread URL into a `.json` endpoint with `?limit=500&raw_json=1`.
  - `fetch_thread_json(ref: str, timeout: int = 15) -> list` — HTTP GET with a descriptive `User-Agent`, returns parsed JSON. Thin wrapper, not unit-tested.
  - `rows_from_threads(refs, min_chars, max_chars, sleep_s, fetch=fetch_thread_json) -> list[dict]` — fetches each ref (sleeping between), extracts, cleans, filters, dedupes, returns rows `{"text": str, "score": int, "source": str}`. The `fetch` parameter is injectable for tests.
  - `write_csv(rows, path, fields)` — writes rows to CSV with the given field order.
  - `main()` — argparse CLI.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_scrape_reddit.py
import csv


def test_thread_url_from_id_and_url():
    assert sr.thread_url("abc123") == (
        "https://www.reddit.com/comments/abc123.json?limit=500&raw_json=1")
    out = sr.thread_url("https://www.reddit.com/r/nba/comments/abc123/title/")
    assert out.endswith("/comments/abc123.json?limit=500&raw_json=1")


def test_rows_from_threads_uses_injected_fetch():
    def fake_fetch(ref, timeout=15):
        return SAMPLE  # from earlier in the file
    rows = sr.rows_from_threads(
        ["t1"], min_chars=8, max_chars=1500, sleep_s=0, fetch=fake_fetch)
    texts = [r["text"] for r in rows]
    assert "Top comment" in texts
    assert "Nested reply" in texts
    assert "[deleted]" not in texts
    assert all(r["source"] == "t1" for r in rows)


def test_write_csv_roundtrip(tmp_path):
    path = tmp_path / "out.csv"
    sr.write_csv([{"text": "hi", "score": 1, "source": "t1"}],
                 str(path), ["text", "score", "source"])
    with open(path, newline="", encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert read[0]["text"] == "hi" and read[0]["source"] == "t1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scrape_reddit.py -q`
Expected: FAIL with `AttributeError: module 'scrape_reddit' has no attribute 'thread_url'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add to scripts/scrape_reddit.py
import argparse
import csv
import time
import requests

USER_AGENT = "takemeter/0.1 (educational classifier project)"


def thread_url(ref):
    if ref.startswith("http"):
        m = re.search(r"/comments/([a-z0-9]+)", ref)
        tid = m.group(1) if m else ref.rstrip("/").split("/")[-1]
    else:
        tid = ref
    return f"https://www.reddit.com/comments/{tid}.json?limit=500&raw_json=1"


def fetch_thread_json(ref, timeout=15):
    resp = requests.get(
        thread_url(ref), headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def rows_from_threads(refs, min_chars, max_chars, sleep_s, fetch=fetch_thread_json):
    rows = []
    for ref in refs:
        data = fetch(ref)
        for c in extract_comments(data):
            if is_valid_comment(c, min_chars, max_chars):
                rows.append({
                    "text": clean_comment(c["body"]),
                    "score": c["score"],
                    "source": ref,
                })
        if sleep_s:
            time.sleep(sleep_s)
    # dedupe on the cleaned text we just produced
    seen, deduped = set(), []
    for r in rows:
        key = r["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def write_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    p = argparse.ArgumentParser(description="Scrape r/nba thread comments")
    p.add_argument("threads", nargs="+", help="thread ids or full URLs")
    p.add_argument("--out", default="data/raw_comments.csv")
    p.add_argument("--min-chars", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()
    rows = rows_from_threads(
        args.threads, args.min_chars, args.max_chars, args.sleep)
    write_csv(rows, args.out, ["text", "score", "source"])
    print(f"Wrote {len(rows)} comments to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scrape_reddit.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Live smoke run**

Pick a current r/nba post-game or discussion thread in a browser, copy its URL, then run:
```bash
source .venv/bin/activate
python scripts/scrape_reddit.py "<paste-thread-url>" --out data/raw_comments.csv
```
Expected: prints `Wrote N comments to data/raw_comments.csv` with N in the dozens-to-hundreds. Open the CSV and confirm the `text` column holds readable comments, no `[deleted]`.

- [ ] **Step 6: Commit (code only, not the scraped data yet)**

```bash
git add scripts/scrape_reddit.py tests/test_scrape_reddit.py
git commit -m "Add Reddit fetch, CSV writer, and scraper CLI"
```

---

### Task 4: Groq prompt building and label parsing

**Files:**
- Create: `scripts/prelabel_groq.py`
- Test: `tests/test_prelabel_groq.py`

**Interfaces:**
- Consumes: the label definitions from `planning.md`.
- Produces:
  - `LABELS = ("analysis", "hot_take", "reaction")` — module-level tuple, the only valid outputs.
  - `build_messages(text: str) -> list[dict]` — returns the chat messages, system prompt embedding the three label definitions and an instruction to reply with one label name only, user message containing the comment.
  - `parse_label(response_text: str) -> str | None` — normalizes (lowercase, spaces to underscores) and returns the first of `LABELS` found in the response, else `None`.
- `groq` is imported only inside `classify`/`main`, never at module top, so these pure functions are testable without the package or a key.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prelabel_groq.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'prelabel_groq'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/prelabel_groq.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prelabel_groq.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/prelabel_groq.py tests/test_prelabel_groq.py
git commit -m "Add Groq prompt builder and label parser"
```

---

### Task 5: Groq pre-label CLI with override-tracking CSV

**Files:**
- Modify: `scripts/prelabel_groq.py`
- Test: `tests/test_prelabel_groq.py`

**Interfaces:**
- Consumes: `LABELS`, `build_messages`, `parse_label` from Task 4; `write_csv` is reimplemented locally to keep the script standalone.
- Produces:
  - `classify(text: str, client, model: str = "llama-3.3-70b-versatile", retries: int = 3, sleep_s: float = 2.0) -> str` — calls `client.chat.completions.create` with `temperature=0`, parses the label, retries on exception, returns the label string or `""` on persistent failure or unparseable output. `client` is injected so tests use a fake.
  - `prelabel_rows(in_path: str, client, model, limit=None, sleep_s=2.0) -> list[dict]` — reads the `text` column from the input CSV and returns rows `{"text", "suggested_label", "label", "notes"}` where `label` is initialized equal to `suggested_label` and `notes` is empty.
  - `main()` — argparse CLI that builds a real Groq client from `GROQ_API_KEY` and writes `data/prelabeled.csv`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_prelabel_groq.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prelabel_groq.py -q`
Expected: FAIL with `AttributeError: module 'prelabel_groq' has no attribute 'classify'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add to scripts/prelabel_groq.py
import argparse
import csv
import os
import time


def classify(text, client, model="llama-3.3-70b-versatile", retries=3, sleep_s=2.0):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=build_messages(text), temperature=0)
            label = parse_label(resp.choices[0].message.content)
            return label or ""
        except Exception:
            if attempt == retries - 1:
                return ""
            if sleep_s:
                time.sleep(sleep_s)
    return ""


def prelabel_rows(in_path, client, model, limit=None, sleep_s=2.0):
    rows = []
    with open(in_path, newline="", encoding="utf-8") as f:
        for i, src in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                break
            suggestion = classify(src["text"], client, model, sleep_s=sleep_s)
            rows.append({
                "text": src["text"],
                "suggested_label": suggestion,
                "label": suggestion,
                "notes": "",
            })
            if sleep_s:
                time.sleep(sleep_s)
    return rows


def main():
    p = argparse.ArgumentParser(description="Pre-label comments with Groq")
    p.add_argument("--in", dest="in_path", default="data/raw_comments.csv")
    p.add_argument("--out", default="data/prelabeled.csv")
    p.add_argument("--model", default="llama-3.3-70b-versatile")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()

    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY in your environment first.")
    client = Groq(api_key=key)

    rows = prelabel_rows(args.in_path, client, args.model, args.limit, args.sleep)
    fields = ["text", "suggested_label", "label", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} pre-labeled rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prelabel_groq.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Live smoke run on a small slice**

```bash
source .venv/bin/activate
export GROQ_API_KEY=<your-key>   # do not commit this
python scripts/prelabel_groq.py --in data/raw_comments.csv --out data/prelabeled.csv --limit 10
```
Expected: prints `Wrote 10 pre-labeled rows to data/prelabeled.csv`. Open the file and confirm `suggested_label` holds only `analysis`, `hot_take`, `reaction`, or empty, and that `label` mirrors `suggested_label`.

- [ ] **Step 6: Commit (code only)**

```bash
git add scripts/prelabel_groq.py tests/test_prelabel_groq.py
git commit -m "Add Groq pre-labeling CLI with override tracking"
```

---

### Task 6: Dataset stats and final export

**Files:**
- Create: `scripts/dataset_stats.py`
- Test: `tests/test_dataset_stats.py`

**Interfaces:**
- Consumes: the `data/prelabeled.csv` schema (`text`, `suggested_label`, `label`, `notes`).
- Produces:
  - `label_counts(rows: list[dict]) -> dict[str, int]` — counts per `label`, ignoring empty labels.
  - `override_rate(rows: list[dict]) -> float` — fraction of rows where `label` differs from `suggested_label`, counting only rows that have a non-empty `suggested_label`. Returns `0.0` if there are none.
  - `max_share(counts: dict) -> float` — the largest class as a fraction of the labeled total, `0.0` if empty.
  - `export_dataset(rows, out_path)` — writes the final `text,label,notes` CSV, skipping rows whose `label` is empty.
  - `main()` — argparse CLI that prints counts, max share with a 70% warning, and override rate, and optionally exports.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset_stats.py
import csv
import dataset_stats as ds

ROWS = [
    {"text": "a", "suggested_label": "analysis", "label": "analysis", "notes": ""},
    {"text": "b", "suggested_label": "hot_take", "label": "analysis", "notes": "fixed"},
    {"text": "c", "suggested_label": "reaction", "label": "reaction", "notes": ""},
    {"text": "d", "suggested_label": "", "label": "hot_take", "notes": ""},
]


def test_label_counts():
    assert ds.label_counts(ROWS) == {"analysis": 2, "reaction": 1, "hot_take": 1}


def test_override_rate_ignores_empty_suggestion():
    # 3 rows have a suggestion; 1 of them (b) was overridden
    assert abs(ds.override_rate(ROWS) - (1 / 3)) < 1e-9


def test_max_share():
    assert abs(ds.max_share(ds.label_counts(ROWS)) - 0.5) < 1e-9


def test_export_skips_empty_labels(tmp_path):
    rows = ROWS + [{"text": "e", "suggested_label": "", "label": "", "notes": ""}]
    out = tmp_path / "final.csv"
    ds.export_dataset(rows, str(out))
    with open(out, newline="", encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert len(read) == 4
    assert list(read[0].keys()) == ["text", "label", "notes"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataset_stats'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/dataset_stats.py
import argparse
import csv
from collections import Counter


def label_counts(rows):
    return dict(Counter(r["label"] for r in rows if r.get("label")))


def override_rate(rows):
    judged = [r for r in rows if r.get("suggested_label")]
    if not judged:
        return 0.0
    changed = sum(1 for r in judged if r["label"] != r["suggested_label"])
    return changed / len(judged)


def max_share(counts):
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def export_dataset(rows, out_path):
    fields = ["text", "label", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r.get("label"):
                w.writerow({k: r.get(k, "") for k in fields})


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Report dataset balance and export")
    p.add_argument("--in", dest="in_path", default="data/prelabeled.csv")
    p.add_argument("--export", default=None,
                   help="path to write the final text,label,notes CSV")
    args = p.parse_args()

    rows = _read(args.in_path)
    counts = label_counts(rows)
    total = sum(counts.values())
    print(f"Labeled rows: {total}")
    for label, n in sorted(counts.items()):
        share = n / total if total else 0
        print(f"  {label}: {n} ({share:.0%})")
    share = max_share(counts)
    print(f"Largest class share: {share:.0%}"
          + ("  WARNING: over 70%, rebalance" if share > 0.70 else ""))
    print(f"Override rate vs Groq suggestion: {override_rate(rows):.0%}")

    if args.export:
        export_dataset(rows, args.export)
        print(f"Exported final dataset to {args.export}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset_stats.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS (all tests from Tasks 2 through 6).

- [ ] **Step 6: Commit**

```bash
git add scripts/dataset_stats.py tests/test_dataset_stats.py
git commit -m "Add dataset stats and final export"
```

---

### Task 7: Collect, review, and finalize the labeled dataset

This task produces the actual data. It is manual work guided by the scripts, not new code. Run it carefully; the model can only learn what the labels capture.

**Files:**
- Create (data, committed at the end): `data/takemeter_dataset.csv`
- Update: `planning.md` (the "Hard annotation decisions" section)

- [ ] **Step 1: Collect across thread types for natural class variety**

Gather thread URLs for each kind, then scrape them into one raw file by passing several at once:
```bash
source .venv/bin/activate
python scripts/scrape_reddit.py \
  "<game-or-live-thread-url>" \
  "<post-game-thread-url>" \
  "<highlight-thread-url>" \
  "<daily-discussion-or-analysis-thread-url>" \
  --out data/raw_comments.csv
```
Aim for enough raw comments that 200+ survive review. Game and live threads feed `reaction`; daily-discussion and analysis threads feed `analysis` and `hot_take`. If the raw count is low, add more thread URLs and re-run.

- [ ] **Step 2: Pre-label with Groq**

```bash
export GROQ_API_KEY=<your-key>
python scripts/prelabel_groq.py --in data/raw_comments.csv --out data/prelabeled.csv
```

- [ ] **Step 3: Review every label by hand**

Open `data/prelabeled.csv` in a spreadsheet. Read each comment and apply the definitions and edge-case rules from `planning.md`. Correct the `label` column where Groq is wrong. Leave `suggested_label` untouched so the override rate stays meaningful. When a comment makes you pause, write the reasoning in `notes`. Delete rows that are unlabelable junk by clearing the `label` cell (the export drops empty-label rows).

- [ ] **Step 4: Log at least three hard decisions in planning.md**

In the "Hard annotation decisions" section of `planning.md`, write at least three real judgment calls: the comment, which two labels it sat between, and what you decided and why. Remove the placeholder line.

- [ ] **Step 5: Check balance and export the final dataset**

```bash
python scripts/dataset_stats.py --in data/prelabeled.csv --export data/takemeter_dataset.csv
```
Read the output. Confirm the labeled total is 200 or more and the largest class share is at most 70% (target each class above 20%). If a class is short, go back to Step 1, scrape more threads of the type that feeds it (analysis-heavy threads for `analysis`), and re-run Steps 2 through 5. Note the override rate; you will cite it in the README AI-usage section.

- [ ] **Step 6: Commit the dataset and the annotation notes**

```bash
git add data/takemeter_dataset.csv planning.md
git commit -m "Add labeled dataset and hard annotation decisions"
```

Optionally commit `data/raw_comments.csv` and `data/prelabeled.csv` too if you want the full pipeline reproducible in the repo. They are not gitignored, so add them only if you want them tracked.

---

## After this plan

With `data/takemeter_dataset.csv` committed, the rest happens in the Colab starter notebook you already set up: upload the CSV (Section 1), confirm the split (Section 2), run the Groq zero-shot baseline (Section 5, Milestone 4), fine-tune DistilBERT (Sections 3 to 4, Milestone 5), and export `evaluation_results.json` and `confusion_matrix.png` (Section 6). Download those two files into `outputs/` and commit them. The README and evaluation report (Milestone 6) come last.
