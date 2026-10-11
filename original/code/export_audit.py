#!/usr/bin/env python3
"""Write the annotation sheets of the native-speaker audits (Section 5.3 and Table 10).

    python export_audit.py --data data --runs runs --out audit_todo

audit_todo/gen_gs.tsv, gen_gu.tsv : 200 generator outputs; ann1/ann2 = 1 if a Tunisian writer
                                    could plausibly produce the output, else 0; final = adjudicated.
audit_todo/errors_tz.tsv, errors_my.tsv : 200 B12 errors (majority vote over seeds); ann1/ann2/final
                                    in {E1,...,E6} (definitions in Table 10 of the paper).
Fill the sheets, copy them to audit/, and re-run diafuse_eval.py with --audit audit.
"""
import argparse
import csv
import os

import numpy as np


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="audit_todo"); ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(7)
    src = {r["id"]: r["text"] for r in read(os.path.join(args.data, "ar.tsv"))}
    for g in ("gs", "gu"):
        rows = read(os.path.join(args.data, f"{g}.tsv"))
        pick = rng.choice(len(rows), min(args.n, len(rows)), replace=False)
        with open(os.path.join(args.out, f"gen_{g}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["id", "source_arabic", "output", "ann1", "ann2", "final"])
            for i in pick:
                w.writerow([rows[i]["id"], src.get(rows[i]["id"], ""), rows[i]["text"], "", "", ""])
    for s in ("tz", "my"):
        rows = read(os.path.join(args.data, f"{s}.tsv")); y = np.array([int(r["label"]) for r in rows])
        sd = os.path.join(args.runs, "preds", "B12")
        seeds = [d for d in os.listdir(sd) if os.path.exists(os.path.join(sd, d, f"{s}.tsv"))] if os.path.isdir(sd) else []
        if not seeds:
            print(f"no B12 predictions for {s}; run diafuse_run.py first"); continue
        votes = np.mean([[int(r["pred"]) for r in read(os.path.join(sd, d, f"{s}.tsv"))] for d in seeds], 0)
        pred = (votes >= 0.5).astype(int); err = np.flatnonzero(pred != y)
        pick = rng.choice(err, min(args.n, len(err)), replace=False)
        with open(os.path.join(args.out, f"errors_{s}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["id", "text", "gold", "pred", "ann1", "ann2", "final"])
            for i in pick:
                w.writerow([rows[i]["id"], rows[i]["text"], y[i], pred[i], "", "", ""])
    print("annotation sheets written to", args.out)


if __name__ == "__main__":
    main()
