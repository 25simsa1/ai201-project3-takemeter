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
