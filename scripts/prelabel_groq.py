import argparse
import csv
import os
import time

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
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(rows)} pre-labeled rows to {args.out}")


if __name__ == "__main__":
    main()
