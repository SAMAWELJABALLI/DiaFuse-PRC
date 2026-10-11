"""Per-seed evaluation. Every metric is computed with the functions of the ORIGINAL diafuse_eval.py
(imported unchanged), so that definitions are exactly those of the paper:
macro-F1, accuracy, minority-class F1, ECE (15 equal-mass bins; raw / global T / variety-wise T),
Brier, NLL, AURC, exact match on paired sets, worst-group accuracy and minority-class F1 on S0 u S3.
"""
import csv
import json
import os
import sys

import numpy as np

ORIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original", "code")
sys.path.insert(0, os.environ.get("DIAFUSE_CODE", ORIG))
import diafuse_eval as DE  # noqa: E402  (original, unmodified)

STAGES = [("S0", "ar"), ("S1", "gs"), ("S2", "gu"), ("S3", "tl"), ("S4", "tz"), ("S5", "na"), ("S6", "my"),
          ("S5AR", "naar"), ("S6AR", "myar"), ("S6SYN", "gsmy")]
CHAIN7 = [s for _, s in STAGES[:7]]
METRICS = ("mf1", "acc", "ece_raw", "ece_glob", "ece_var", "brier", "nll", "aurc")


def load_data(data_dir):
    data = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".tsv"):
            d = DE.load_set(data_dir, fn[:-4])
            if d is not None:
                data[fn[:-4]] = d
    return data


def rescore_archived(data_dir, arch_preds, arch_summary, systems=("B0", "B6", "B10", "B11", "B12")):
    """Re-score archived predictions with the regenerated gold labels; must reproduce the archived summary exactly."""
    data = load_data(data_dir); S = json.load(open(arch_summary))
    preds = DE.load_preds(arch_preds, data)
    mism, n = [], 0
    for sysn in systems:
        for st in CHAIN7 + ["naar", "myar", "gsmy"]:
            for met in ("mf1", "acc"):
                key = f"{sysn}/{st}/{met}"
                if key not in S:
                    continue
                v = DE.per_seed_metric(preds, data, sysn, st, met)
                n += 1
                if not v or abs(float(np.mean(v)) - S[key][0]) > 1e-9:
                    mism.append(key)
    return {"n_checked": n, "mismatches": mism}


def _single(preds, sysn, seed):
    return {sysn: {seed: preds[sysn][seed]}}


def evaluate_all(data_dir, preds_dir, results_dir):
    data = load_data(data_dir); preds = DE.load_preds(preds_dir, data)
    minority = int(np.mean(data["train_ar"]["y"]) > 0.5) ^ 1 if "train_ar" in data else 0  # as in diafuse_eval.main
    rows = []; mean_chain = {}
    for sysn in sorted(preds):
        for seed in sorted(preds[sysn], key=lambda x: int(x)):
            one = _single(preds, sysn, seed); chain = []
            robs = [(s, s) for s in sorted(preds[sysn][seed]) if s.startswith("rob-")]  # perturbation sets (Section 6.4)
            for stage, st in STAGES + robs:
                if st not in data or st not in preds[sysn][seed]:
                    continue
                for met in METRICS:
                    v = DE.per_seed_metric(one, data, sysn, st, met)
                    if v:
                        rows.append((sysn, seed, stage, st, met, v[0]))
                        if met == "mf1" and st in CHAIN7:
                            chain.append(v[0])
                v = DE.per_seed_metric(one, data, sysn, st, "minf1", minority)
                if v:
                    rows.append((sysn, seed, stage, st, "minf1", v[0]))
                pair = DE.PAIR.get(st) or ("ar" if st in ("rob-trs", "rob-tru") or st.startswith("rob-tru-s") else ("gs" if st.startswith("rob-") else None))  # as diafuse_eval
                if pair and pair in preds[sysn][seed]:
                    a = dict(zip(data[st]["id"], preds[sysn][seed][st]["pred"])); b = dict(zip(data[pair]["id"], preds[sysn][seed][pair]["pred"]))
                    common = [i for i in a if i in b]
                    rows.append((sysn, seed, stage, st, "em", float(np.mean([a[i] == b[i] for i in common]))))
            if all(s in preds[sysn][seed] for s in ("ar", "tl")):  # WGA / minority F1 on S0 u S3 (formula of diafuse_eval)
                cells = []
                for name in ("ar", "tl"):
                    y = data[name]["y"]; p = preds[sysn][seed][name]["pred"]
                    cells += [np.mean(p[y == c] == c) for c in (0, 1) if np.any(y == c)]
                y = np.concatenate([data["ar"]["y"], data["tl"]["y"]])
                p = np.concatenate([preds[sysn][seed]["ar"]["pred"], preds[sysn][seed]["tl"]["pred"]])
                rows.append((sysn, seed, "S0+S3", "artl", "wga", float(min(cells))))
                rows.append((sysn, seed, "S0+S3", "artl", "minf1", DE.f1_class(y, p, minority)))
            if len(chain) == 7:
                mean_chain.setdefault(sysn, {})[str(seed)] = float(np.mean(chain))
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    with open(os.path.join(results_dir, "metrics", "seed_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "seed", "stage", "set", "metric", "value"]); w.writerows(rows)
    agg = {}
    for sysn, seed, stage, st, met, v in rows:
        agg.setdefault((sysn, stage, st, met), []).append(v)
    with open(os.path.join(results_dir, "metrics", "summary_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "stage", "set", "metric", "mean", "sd", "n_seeds", "min", "max"])
        for (sysn, stage, st, met), v in sorted(agg.items()):
            w.writerow([sysn, stage, st, met, float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v), min(v), max(v)])
    json.dump({"mean_S0_S6": mean_chain, "minority_class": minority}, open(os.path.join(results_dir, "metrics", "chain_means.json"), "w"), indent=1)
    return {"mean_S0_S6": mean_chain}
