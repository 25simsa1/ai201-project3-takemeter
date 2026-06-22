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
