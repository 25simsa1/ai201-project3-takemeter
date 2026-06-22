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
