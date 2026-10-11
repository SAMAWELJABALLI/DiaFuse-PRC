"""Statistical tests of the protocol (Section 'Statistical protocol' of the paper), nothing else:
  * paired bootstrap (Koehn 2004) of the seed-averaged macro-F1 difference on the same class-stratified
    resamples: p = diafuse_eval.paired_boot_p (original function), 1,000 resamples as stated in the protocol;
    the 95% CI of the difference uses the same resampling scheme;
  * Holm-Bonferroni over the 7 stages S0-S6 of each comparison (diafuse_eval.holm);
  * decision rule: adjusted p < 0.05 AND mean difference > larger seed standard deviation;
  * Wilcoxon signed-rank test across stages (per comparison);
  * almost stochastic order on seed-level scores (deepsig), reported only if the package is available.
"""
import csv
import json
import os
import sys

import numpy as np

import evaluate as EV

DE = EV.DE
N_BOOT = 1000
COMPARISONS = [("PPA-FINAL", "B11"), ("PPB-FINAL", "B11"), ("PPA-FINAL", "B12"), ("PPB-FINAL", "B12"), ("B12", "B11"),
               ("B11", "PPA-FINAL"), ("B11", "PPB-FINAL")]


def seed_scores(preds, data, sysn, st):
    return DE.per_seed_metric(preds, data, sysn, st, "mf1")


def diff_ci(preds, data, a, b, st, n=N_BOOT):
    y = data[st]["y"]; sa = DE.seeds_with(preds, a, [st]); sb = DE.seeds_with(preds, b, [st])
    pa = [preds[a][s][st]["pred"] for s in sa]; pb = [preds[b][s][st]["pred"] for s in sb]
    d = []
    for _ in range(n):
        idx = DE.strat_boot_idx(y)
        d.append(np.mean([DE.f1_macro(y[idx], p[idx]) for p in pa]) - np.mean([DE.f1_macro(y[idx], p[idx]) for p in pb]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_all(data_dir, preds_dir, P, tables_dir, orig_dir=None):
    data = EV.load_data(data_dir); preds = DE.load_preds(preds_dir, data)
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    try:
        from deepsig import aso
    except Exception:
        aso = None
    rows = []; verdicts = {}
    stages = [(lab, st) for lab, st in EV.STAGES[:7]]
    for a, b in COMPARISONS:
        if a not in preds or b not in preds:
            continue
        pv, eff, sds, cis, aso_v = {}, {}, {}, {}, {}
        for lab, st in stages:
            A, B = seed_scores(preds, data, a, st), seed_scores(preds, data, b, st)
            if not A or not B:
                continue
            eff[lab] = 100 * (np.mean(A) - np.mean(B))
            sds[lab] = (100 * float(np.std(A, ddof=1)) if len(A) > 1 else 0.0, 100 * float(np.std(B, ddof=1)) if len(B) > 1 else 0.0, len(A), len(B))
            pv[lab] = DE.paired_boot_p(preds, data, a, b, st, n=N_BOOT)
            lo, hi = diff_ci(preds, data, a, b, st); cis[lab] = (100 * lo, 100 * hi)
            if aso is not None and len(A) > 1 and len(B) > 1:
                try:
                    aso_v[lab] = float(aso(A, B, seed=0, show_progress=False))
                except Exception:
                    aso_v[lab] = None
        adj = DE.holm(pv)
        wins = []
        for lab in eff:
            sig = adj[lab] < 0.05 and eff[lab] > max(sds[lab][0], sds[lab][1])
            wins.append(lab) if sig else None
            rows.append({"comparison": f"{a} vs {b}", "stage": lab, "effect": round(eff[lab], 3), "ci_low": round(cis[lab][0], 3),
                         "ci_high": round(cis[lab][1], 3), "p_value": pv[lab], "adjusted_p": adj[lab], "significant": bool(sig),
                         "sd_a": round(sds[lab][0], 3), "sd_b": round(sds[lab][1], 3), "n_seeds_a": sds[lab][2], "n_seeds_b": sds[lab][3],
                         "test": "paired bootstrap (1000, one-sided A>B), Holm over stages, decision rule",
                         "aso_eps_min": aso_v.get(lab)})
        if wilcoxon is not None and len(eff) >= 5:
            try:
                w = wilcoxon(list(eff.values()), alternative="greater")
                rows.append({"comparison": f"{a} vs {b}", "stage": "ALL (S0-S6)", "effect": round(float(np.mean(list(eff.values()))), 3),
                             "ci_low": "", "ci_high": "", "p_value": float(w.pvalue), "adjusted_p": "", "significant": bool(w.pvalue < 0.05),
                             "sd_a": "", "sd_b": "", "n_seeds_a": "", "n_seeds_b": "", "test": "Wilcoxon signed-rank across stages (one-sided)",
                             "aso_eps_min": ""})
            except Exception:
                pass
        verdicts[f"{a} vs {b}"] = {"stage_wins": wins, "n_stages": len(eff)}
    final = {}
    for pp in ("PPA-FINAL", "PPB-FINAL"):
        k, rk = f"{pp} vs B11", f"B11 vs {pp}"
        if k not in verdicts:
            continue
        w, lw = verdicts[k]["stage_wins"], verdicts.get(rk, {}).get("stage_wins", [])
        if len(w) >= 4 and not lw:
            v = "OUTPERFORMS"
        elif w:
            v = "PARTIAL"
        else:
            v = "DOES_NOT_OUTPERFORM"
        final[pp] = {"verdict": v, "stages_won": w, "stages_lost": lw, "rule": P["verdict_rule"]}
    os.makedirs(tables_dir, exist_ok=True)
    cols = ["comparison", "stage", "effect", "ci_low", "ci_high", "p_value", "adjusted_p", "significant", "sd_a", "sd_b",
            "n_seeds_a", "n_seeds_b", "test", "aso_eps_min"]
    with open(os.path.join(tables_dir, "statistical_tests.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
    json.dump({"verdicts": final, "stage_wins": verdicts, "aso_available": aso is not None}, open(os.path.join(tables_dir, "verdict.json"), "w"), indent=1)
    return final
