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
