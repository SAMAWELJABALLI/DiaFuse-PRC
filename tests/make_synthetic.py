#!/usr/bin/env python3
"""TEST FIXTURE ONLY. Synthetic data set (same file names, id schemes and pairings as prep_data.py) and,
optionally, a synthetic 'archived run' consistent with it. Used by the dry-run test of the orchestrator.
Every text starts with 'SYNTHETIC'. Nothing produced here is, or may be reported as, an experimental result."""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

SIZES = {"train": 300, "val": 120, "ar": 120, "tl": 200, "tz": 150, "na": 80, "my": 160}
AR = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
LA = "abcdefghijklmnopqrstuvwxyz"


def txt(rng, script, i, tag):
    al = AR if script == "ar" else LA
    return "SYNTHETIC " + tag + str(i) + " " + " ".join("".join(rng.choice(list(al)) for _ in range(rng.randint(2, 7))) for _ in range(rng.randint(3, 9)))


def write(D, name, rows):
    with open(os.path.join(D, f"{name}.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL); w.writerow(["id", "text", "label"]); w.writerows(rows)


def make_data(D, seed=0):
    rng = random.Random(seed); os.makedirs(D, exist_ok=True); rep = {}
    def block(prefix, n, script, tag):
        return [(f"{prefix}-{i}", txt(rng, script, i, tag), rng.randint(0, 1)) for i in range(n)]
    tr = block("tr", SIZES["train"], "ar", "tr"); write(D, "train_ar", tr)
    for v in ("gs", "gu", "inf", "noise", "cs"):
        write(D, f"train_{v}", [(i, txt(rng, "la", k, "tr" + v), y) for k, (i, _, y) in enumerate(tr)])
    va = block("va", SIZES["val"], "ar", "va"); write(D, "val_ar", va)
    write(D, "val_gs", [(i, txt(rng, "la", k, "vags"), y) for k, (i, _, y) in enumerate(va)])
    s0 = block("s0", SIZES["ar"], "ar", "s0"); write(D, "ar", s0)
    for v in ("gs", "gu"):
        write(D, v, [(i, txt(rng, "la", k, "s0" + v), y) for k, (i, _, y) in enumerate(s0)])
    for r in ("trs", "tru", "spell", "abbr", "cs", "noise", "tru-s0", "tru-s1", "tru-s2", "tru-s3", "tru-s4"):
        write(D, f"rob-{r}", [(i, txt(rng, "la", k, "rob" + r), y) for k, (i, _, y) in enumerate(s0)])
    write(D, "tl", block("tl", SIZES["tl"], "la", "tl")); write(D, "tz", block("tz", SIZES["tz"], "la", "tz"))
    na = block("na", SIZES["na"], "la", "na"); write(D, "na", na)
    write(D, "naar", [(i, txt(rng, "ar", k, "naar"), y) for k, (i, _, y) in enumerate(na)])
    write(D, "my", block("my", SIZES["my"], "la", "my"))
    myar = block("myar", SIZES["my"], "ar", "myar"); write(D, "myar", myar)
    write(D, "gsmy", [(i, txt(rng, "la", k, "gsmy"), y) for k, (i, _, y) in enumerate(myar)])
    for s in sorted(f[:-4] for f in os.listdir(D) if f.endswith(".tsv")):
        rep[f"final/{s}"] = sum(1 for _ in open(os.path.join(D, f"{s}.tsv"))) - 1
    rep["_synthetic"] = True
    json.dump(rep, open(os.path.join(D, "prep_report.json"), "w"), indent=1)
    return rep


def fake_preds(D, out, system, seed, acc, rng):
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".tsv") or fn.startswith(("train", "val", "tt-")):
            continue
        rows = list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8"), delimiter="\t"))
        d = os.path.join(out, system, str(seed)); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, fn), "w") as f:
            f.write("id\tpred\tp_raw\tp_glob\tp_var\n")
            for r in rows:
                y = int(r["label"]); p = y if rng.random() < acc else 1 - y
                q = min(0.99, max(0.01, (0.75 if p == 1 else 0.25) + rng.uniform(-0.2, 0.2)))
                f.write(f"{r['id']}\t{p}\t{q:.6f}\t{q:.6f}\t{q:.6f}\n")


def make_archive(D, A, code_dir):
    sys.path.insert(0, code_dir); import diafuse_eval as DE
    rng = random.Random(1); os.makedirs(os.path.join(A, "preds"), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(D, "prep_report.json"), os.path.join(A, "prep_report.json"))
    for s, acc in (("B0", .65), ("B6", .74), ("B10", .73), ("B11", .82), ("B12", .76)):
        for seed in range(3):
            fake_preds(D, os.path.join(A, "preds"), s, seed, acc, rng)
    data = {fn[:-4]: DE.load_set(D, fn[:-4]) for fn in os.listdir(D) if fn.endswith(".tsv")}
    preds = DE.load_preds(os.path.join(A, "preds"), data); S = {}
    for s in preds:
        for st in ("ar", "gs", "gu", "tl", "tz", "na", "my", "naar", "myar", "gsmy"):
            for met in ("mf1", "acc"):
                v = DE.per_seed_metric(preds, data, s, st, met)
                S[f"{s}/{st}/{met}"] = [float(np.mean(v)), float(np.std(v, ddof=1))]
    json.dump(S, open(os.path.join(A, "summary.json"), "w"))
    json.dump({f"B12/{k}": 0.8830 + 0.0004 * k for k in range(3)}, open(os.path.join(A, "dev_scores.json"), "w"))
    json.dump({"source": "SYNTHETIC", "best_dev_macro_f1": {"B11": {"0": 0.94, "1": 0.9351, "2": 0.9358}}}, open(os.path.join(A, "finetune_dev_log.json"), "w"))
    json.dump({"torch": "synthetic", "transformers": "synthetic"}, open(os.path.join(A, "environment.json"), "w"))
    os.makedirs(os.path.join(A, "audit_anchors"), exist_ok=True)
    for name, s in (("errors_my", "my"), ("errors_tz", "tz")):
        rows = list(csv.DictReader(open(os.path.join(D, f"{s}.tsv"), encoding="utf-8"), delimiter="\t"))[:20]
        with open(os.path.join(A, "audit_anchors", f"{name}.tsv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["id", "text", "gold", "pred", "ann1", "ann2", "final"])
            for r in rows:
                w.writerow([r["id"], r["text"], r["label"], 0, "", "", ""])
    ar = {r["id"]: r["text"] for r in csv.DictReader(open(os.path.join(D, "ar.tsv"), encoding="utf-8"), delimiter="\t")}
    for name, s in (("gen_gs", "gs"), ("gen_gu", "gu")):
        rows = list(csv.DictReader(open(os.path.join(D, f"{s}.tsv"), encoding="utf-8"), delimiter="\t"))[:20]
        with open(os.path.join(A, "audit_anchors", f"{name}.tsv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["id", "source_arabic", "output", "ann1", "ann2", "final"])
            for r in rows:
                w.writerow([r["id"], ar[r["id"]], r["text"], "", "", ""])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--archive", default=None)
    ap.add_argument("--code", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original", "code"))
    a = ap.parse_args(); make_data(a.data)
    if a.archive:
        make_archive(a.data, a.archive, a.code)
    print("SYNTHETIC test fixture written")
