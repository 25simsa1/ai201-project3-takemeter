import re
import argparse
import csv
import glob
import json
import os
import time
import requests

LOW_VALUE = {"this", "lol", "same", "^", "^^this", "lmao", "facts", "deadass"}
_WS = re.compile(r"\s+")

USER_AGENT = "takemeter/0.1 (educational classifier project)"


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


def rows_from_files(paths, min_chars, max_chars):
    """Offline mode: read saved thread .json files instead of fetching."""
    def fetch(path, timeout=15):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return rows_from_threads(paths, min_chars, max_chars, sleep_s=0, fetch=fetch)


def write_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    p = argparse.ArgumentParser(description="Scrape r/nba thread comments")
    p.add_argument("threads", nargs="*", help="thread ids or full URLs")
    p.add_argument("--json-dir",
                   help="offline mode: a folder of saved thread .json files")
    p.add_argument("--out", default="data/raw_comments.csv")
    p.add_argument("--min-chars", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()
    if args.json_dir:
        paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
        if not paths:
            raise SystemExit(f"No .json files found in {args.json_dir}")
        rows = rows_from_files(paths, args.min_chars, args.max_chars)
    elif args.threads:
        rows = rows_from_threads(
            args.threads, args.min_chars, args.max_chars, args.sleep)
    else:
        raise SystemExit("Provide thread URLs/ids, or --json-dir for offline mode")
    write_csv(rows, args.out, ["text", "score", "source"])
    print(f"Wrote {len(rows)} comments to {args.out}")


if __name__ == "__main__":
    main()
