"""Score a semantic_fg_bg_blip.py CSV the way the SGS paper reports it (Table 1).

The paper's evaluation set is one-class: every sample is FG-BG inconsistent, so the positive
class is "Mismatch" and every label is 1. Then Acc = fraction labelled Mismatch, and
F1 = 2 * Acc / (1 + Acc) (precision is 1, recall is Acc).

Usage:
    python scripts/score_sgs.py results/semantic_blip_all.csv [--threshold 0.55] [--out scores.txt]

Without --threshold the CSV's own labels are used; with it, labels are recomputed as
Mismatch iff sts01 < threshold.
"""
import argparse
import csv
import statistics
import sys


def main():
    ap = argparse.ArgumentParser(description="Score an SGS semantic checker CSV (one-class, all inconsistent)")
    ap.add_argument("csv", help="CSV written by semantic_fg_bg_blip.py")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Recompute labels at this threshold instead of using the CSV's labels")
    ap.add_argument("--out", default=None, help="Also write the table to this file")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[ERR] no rows in {args.csv}")

    # Rows that failed in the checker have an empty sts01 and an "ERROR: ..." label
    errors = [r for r in rows if not r["sts01"]]
    scored = [r for r in rows if r["sts01"]]
    s = [float(r["sts01"]) for r in scored]

    if args.threshold is None:
        mismatch = sum(r["label"] == "Mismatch" for r in scored)
        rule = "labels from CSV"
    else:
        mismatch = sum(v < args.threshold for v in s)
        rule = f"Mismatch iff sts01 < {args.threshold}"

    # Error rows count as not detected, so N is every row, as in the paper's N
    n = len(rows)
    acc = mismatch / n
    f1 = 2 * acc / (1 + acc) if acc > 0 else 0.0
    sd = statistics.stdev(s) if len(s) > 1 else 0.0

    lines = [
        f"CSV: {args.csv}",
        f"Rule: {rule}; one-class setting (all samples inconsistent, positive = Mismatch)",
        f"Errors (no score): {len(errors)}",
        "",
        "| N | s~FB mean±sd | median | % Mismatch | Acc / F1 |",
        "|---|---|---|---|---|",
        f"| {n} | {statistics.mean(s):.3f}±{sd:.3f} | {statistics.median(s):.3f} "
        f"| {100 * acc:.1f} | {acc:.3f} / {f1:.3f} |",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
