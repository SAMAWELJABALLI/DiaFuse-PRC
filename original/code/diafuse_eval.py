#!/usr/bin/env python3
"""DiaFuse-PRC evaluation pipeline.

Reads real prediction files and writes results/results.tex (one
\\setres{system/set/metric}{value} per computed quantity) plus the data files of
the figures. Nothing is invented: a quantity whose inputs are missing is simply
not written, and the manuscript then prints TBO for it.

Usage
-----
    python diafuse_eval.py --data DATA --preds PREDS [--diag DIAG] [--emb EMB]
                           [--audit AUDIT] [--manual manual.json] [--out results]

Layout (all files are UTF-8 TSV with a header row)
--------------------------------------------------
DATA/<set>.tsv            id, text, label (0 = negative, 1 = positive)
    sets: train_ar, train_gs, val_ar, ar, gs, gu, tl, tz, na, my, naar, myar,
          gsmy, rob-<family> (severity 2), rob-tru-s0 ... rob-tru-s4
    gs/gu/rob-trs/rob-tru*/ share ids with ar; other rob-* share ids with gs;
    naar shares ids with na.
PREDS/<system>/<seed>/<set>.tsv   id, pred [, p_raw, p_glob, p_var]
    p_* = probability of the positive class under each calibration variant.
    systems: B0, B0TT, B1 ... B12, B12L, CEIL, B12-nopc, abl-*, abl2-*,
             lt-<irC>-<irV>-<sys>, fact-<FPHC bits>, B3-e1/B3-e2/B3-e3
DIAG/<system>/<seed>/<set>.tsv    id, alpha1..alphaM, plat, maxproto
    (also <set> = val for the off-support threshold)
EMB/<system>/<set>.npy            fused embeddings z, rows aligned with DATA/<set>.tsv
                                  (needs train_gs.npy for PAD)
AUDIT/gen_gs.tsv, gen_gu.tsv      id, ann1, ann2 [, final]   (1 = acceptable)
AUDIT/errors_tz.tsv, errors_my.tsv id, ann1, ann2, final     (E1 ... E6)
manual.json                       {"key": "value"} copied verbatim (counts, timings)

Outputs (in --out): results.tex, ci.csv (95% CIs of macro-F1), summary.json,
factorial.json, chain.dat, rel-B6-glob.dat, rel-B12-var.dat, robcurve-tru.dat,
umap-B6.dat, umap-B12.dat (each only when its inputs exist).
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

RNG = np.random.default_rng(12345)
N_BOOT = 1000

CHAIN = ["ar", "gs", "gu", "tl", "tz", "na", "my"]
HUMAN = ["tl", "tz", "na", "my"]
ARABIC_SETS = {"train_ar", "val_ar", "ar", "naar", "myar"}
PAIR = {"gs": "ar", "gu": "ar", "na": "naar"}  # EM(S, PAIR[S])
MAIN = ["B0", "B0TT", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
        "B10", "B11", "B12", "B12L"]
OURS = {"B12", "B12L"}
LOWER_BETTER = {"ece_raw", "ece_glob", "ece_var", "brier", "nll", "aurc"}

# Dialect-marker lexicons (Appendix E). Native-speaker verification required.
MARKERS = {
    "tn": {"barcha", "barsha", "behi", "bahi", "tawa", "taw", "chnowa", "chnoua",
           "chniya", "mouch", "moch", "yesser", "yasser", "mte3", "mta3", "famma",
           "برشا", "باهي", "توا", "شنوة", "شنية", "موش", "ياسر", "متاع", "فما"},
    "ma": {"bzzaf", "bzaf", "mzyan", "mezyan", "daba", "achno", "chno", "dyal",
           "dial", "bghit", "wakha", "safi", "zwin",
           "بزاف", "مزيان", "دابا", "اشنو", "ديال", "بغيت", "واخا", "صافي", "زوين"},
    "dz": {"mlih", "drok", "dorka", "ta3", "nta3", "chbab",
           "مليح", "درك", "تاع", "نتاع"},
}
DIGIT_LETTERS = set("235789")


# ----------------------------------------------------------------- utilities
def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_set(data_dir, name):
    p = os.path.join(data_dir, f"{name}.tsv")
    if not os.path.exists(p):
        return None
    rows = read_tsv(p)
    return {"id": [r["id"] for r in rows], "text": [r["text"] for r in rows],
            "y": np.array([int(r["label"]) for r in rows])}


def f1_macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def f1_class(y, p, c):
    tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
    fn = np.sum((p != c) & (y == c))
    return 0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn))


def conf_correct(y, pred, ppos):
    conf = np.where(pred == 1, ppos, 1 - ppos)
    return conf, (pred == y).astype(float)


def ece_equal_mass(conf, corr, bins=15):
    order = np.argsort(conf)
    chunks = np.array_split(order, bins)
    n = len(conf)
    return float(sum(len(c) / n * abs(conf[c].mean() - corr[c].mean()) for c in chunks if len(c)))


def reliability(conf, corr, bins=15):
    order = np.argsort(conf)
    return [(float(conf[c].mean()), float(corr[c].mean())) for c in np.array_split(order, bins) if len(c)]


def aurc(conf, corr):
    order = np.argsort(-conf)
    err = 1 - corr[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def strat_boot_idx(y):
    idx = []
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        idx.append(RNG.choice(pos, size=len(pos), replace=True))
    return np.concatenate(idx)


def cohen_kappa(a, b):
    a = list(a); b = list(b)
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# ------------------------------------------------------------- formatting
class Results:
    def __init__(self):
        self.vals = {}

    def set(self, key, value):
        self.vals[key] = value

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("% Generated by diafuse_eval.py from prediction files. Do not edit by hand.\n")
            for k in sorted(self.vals):
                f.write(f"\\setres{{{k}}}{{{self.vals[k]}}}\n")


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    out = f"{x:.{d}f}"
    return out[1:] if re.fullmatch(r"-0\.0*", out) else out


def fmt_ci(m, lo, hi, d=1):
    return f"{fmt(m, d)}{{\\tiny\\,[{fmt(lo, d)}, {fmt(hi, d)}]}}"


def fmt_int(n):
    return f"{int(n):,}"


SCALE = {"mf1": (100, 1), "acc": (100, 1), "minf1": (100, 1), "wga": (100, 1),
         "em": (100, 1), "ece_raw": (100, 1), "ece_glob": (100, 1), "ece_var": (100, 1),
         "brier": (1, 3), "nll": (1, 3), "aurc": (100, 1), "mf1min": (100, 1)}


# ------------------------------------------------------------- predictions
def load_preds(pred_dir, data):
    """preds[system][seed][set] = dict(pred=..., p_raw/p_glob/p_var=...), aligned to data ids."""
    preds = {}
    if not pred_dir or not os.path.isdir(pred_dir):
        return preds
    for system in sorted(os.listdir(pred_dir)):
        sdir = os.path.join(pred_dir, system)
        if not os.path.isdir(sdir):
            continue
        for seed in sorted(os.listdir(sdir)):
            ddir = os.path.join(sdir, seed)
            if not os.path.isdir(ddir):
                continue
            for fn in os.listdir(ddir):
                if not fn.endswith(".tsv"):
                    continue
                name = fn[:-4]
                if name not in data:
                    print(f"[warn] {system}/{seed}/{fn}: no DATA/{name}.tsv, skipped", file=sys.stderr)
                    continue
                rows = {r["id"]: r for r in read_tsv(os.path.join(ddir, fn))}
                ids = data[name]["id"]
                missing = [i for i in ids if i not in rows]
                if missing:
                    print(f"[warn] {system}/{seed}/{fn}: {len(missing)} ids missing, skipped", file=sys.stderr)
                    continue
                rec = {"pred": np.array([int(rows[i]["pred"]) for i in ids])}
                for col in ("p_raw", "p_glob", "p_var"):
                    if col in rows[ids[0]] and rows[ids[0]][col] != "":
                        rec[col] = np.clip(np.array([float(rows[i][col]) for i in ids]), 1e-7, 1 - 1e-7)
                preds.setdefault(system, {}).setdefault(seed, {})[name] = rec
    return preds


def seeds_with(preds, system, sets):
    return [s for s, d in preds.get(system, {}).items() if all(x in d for x in sets)]


def per_seed_metric(preds, data, system, name, metric, minority=None):
    vals = []
    for seed in seeds_with(preds, system, [name]):
        r = preds[system][seed][name]; y = data[name]["y"]
        if metric == "mf1":
            vals.append(f1_macro(y, r["pred"]))
        elif metric == "acc":
            vals.append(float(np.mean(r["pred"] == y)))
        elif metric == "minf1" and minority is not None:
            vals.append(f1_class(y, r["pred"], minority))
        elif metric in ("ece_raw", "ece_glob", "ece_var"):
            col = "p_" + metric.split("_")[1]
            if col in r:
                vals.append(ece_equal_mass(*conf_correct(y, r["pred"], r[col])))
        elif metric in ("brier", "nll", "aurc"):
            col = next((c for c in ("p_var", "p_glob", "p_raw") if c in r), None)
            if col is None:
                continue
            p = r[col]
            if metric == "brier":
                vals.append(float(np.mean((p - y) ** 2)))
            elif metric == "nll":
                vals.append(float(-np.mean(np.log(np.where(y == 1, p, 1 - p)))))
            else:
                vals.append(aurc(*conf_correct(y, r["pred"], p)))
    return vals


def boot_mf1(preds, data, system, name, n=N_BOOT):
    """Class-stratified bootstrap within each seed, pooled across seeds."""
    y = data[name]["y"]; out = []
    seeds = seeds_with(preds, system, [name])
    per = max(1, n // max(1, len(seeds)))
    for seed in seeds:
        p = preds[system][seed][name]["pred"]
        for _ in range(per):
            idx = strat_boot_idx(y)
            out.append(f1_macro(y[idx], p[idx]))
    return np.array(out)


def paired_boot_p(preds, data, sys_a, sys_b, name, n=N_BOOT):
    """One-sided p for mean macro-F1(A) > mean macro-F1(B), seed-averaged, same resamples."""
    y = data[name]["y"]
    sa = seeds_with(preds, sys_a, [name]); sb = seeds_with(preds, sys_b, [name])
    if not sa or not sb:
        return None
    pa = [preds[sys_a][s][name]["pred"] for s in sa]
    pb = [preds[sys_b][s][name]["pred"] for s in sb]
    worse = 0
    for _ in range(n):
        idx = strat_boot_idx(y)
        d = np.mean([f1_macro(y[idx], p[idx]) for p in pa]) - np.mean([f1_macro(y[idx], p[idx]) for p in pb])
        worse += d <= 0
    return (worse + 1) / (n + 1)


def holm(pvals):
    items = sorted([(p, k) for k, p in pvals.items() if p is not None])
    m = len(items); adj = {}; running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ------------------------------------------------------------- main metrics
def classification_metrics(R, preds, data, minority):
    summary = {}
    for system in preds:
        for name in list(data):
            if name.startswith(("train", "val", "tt-")):
                continue
            for metric in ("mf1", "acc", "ece_raw", "ece_glob", "ece_var", "brier", "nll", "aurc"):
                v = per_seed_metric(preds, data, system, name, metric)
                if not v:
                    continue
                k, d = SCALE[metric]
                m = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                summary[(system, name, metric)] = (m, sd, v)
                R.set(f"{system}/{name}/{metric}:m", fmt(k * m, d))
                R.set(f"{system}/{name}/{metric}", f"{fmt(k * m, d)}\\pmsd{{{fmt(k * sd, d)}}}")
                if metric == "mf1":
                    R.set(f"{system}/{name}/mf1min:m", fmt(100 * min(v), 1))
            # cross-script exact match
            pair = PAIR.get(name) or ("ar" if name in ("rob-trs", "rob-tru") or name.startswith("rob-tru-s")
                                      else ("gs" if name.startswith("rob-") else None))
            if pair and pair in data:
                ems = []
                for seed in seeds_with(preds, system, [name, pair]):
                    a = dict(zip(data[name]["id"], preds[system][seed][name]["pred"]))
                    b = dict(zip(data[pair]["id"], preds[system][seed][pair]["pred"]))
                    common = [i for i in a if i in b]
                    if common:
                        ems.append(np.mean([a[i] == b[i] for i in common]))
                if ems:
                    summary[(system, name, "em")] = (float(np.mean(ems)), 0.0, ems)
                    R.set(f"{system}/{name}/em:m", fmt(100 * np.mean(ems)))
        # aggregates: human chain, drop, robustness consistency
        for metric in ("mf1", "acc", "ece_raw", "ece_glob", "ece_var", "brier", "nll", "aurc"):
            if all((system, s, metric) in summary for s in HUMAN):
                seeds = seeds_with(preds, system, HUMAN)
                if not seeds:
                    continue
                per_seed = []
                for sd in seeds:
                    vals = [per_seed_metric({system: {sd: preds[system][sd]}}, data, system, s, metric) for s in HUMAN]
                    if all(vals):
                        per_seed.append(float(np.mean([v[0] for v in vals])))
                if not per_seed:
                    continue
                k, d = SCALE[metric]
                m = float(np.mean(per_seed)); sdv = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
                summary[(system, "hum", metric)] = (m, sdv, per_seed)
                R.set(f"{system}/hum/{metric}:m", fmt(k * m, d))
                R.set(f"{system}/hum/{metric}", f"{fmt(k * m, d)}\\pmsd{{{fmt(k * sdv, d)}}}")
                if metric == "mf1":
                    R.set(f"{system}/hum/mf1min:m", fmt(100 * min(per_seed)))
        seeds = seeds_with(preds, system, ["ar", "my"])
        if seeds:
            drops = [f1_macro(data["ar"]["y"], preds[system][s]["ar"]["pred"]) -
                     f1_macro(data["my"]["y"], preds[system][s]["my"]["pred"]) for s in seeds]
            sdv = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
            R.set(f"{system}/drop/mf1", f"{fmt(100 * np.mean(drops))}\\pmsd{{{fmt(100 * sdv)}}}")
        rob_em = [summary[(system, n, "em")][0] for n in data
                  if n.startswith("rob-") and "-s" not in n[4:] and (system, n, "em") in summary]
        if rob_em:
            R.set(f"{system}/rob/em:m", fmt(100 * np.mean(rob_em)))
        # worst-group accuracy and minority-class F1 on S0 u S3
        seeds = seeds_with(preds, system, ["ar", "tl"])
        if seeds:
            wgas, minf = [], []
            for s in seeds:
                cells = []
                for name in ("ar", "tl"):
                    y = data[name]["y"]; p = preds[system][s][name]["pred"]
                    cells += [np.mean(p[y == c] == c) for c in (0, 1) if np.any(y == c)]
                wgas.append(min(cells))
                y = np.concatenate([data["ar"]["y"], data["tl"]["y"]])
                p = np.concatenate([preds[system][s]["ar"]["pred"], preds[system][s]["tl"]["pred"]])
                minf.append(f1_class(y, p, minority))
            R.set(f"{system}/artl/wga:m", fmt(100 * np.mean(wgas)))
            R.set(f"{system}/artl/minf1:m", fmt(100 * np.mean(minf)))
    return summary


def bold_and_significance(R, preds, data, summary):
    """Bold best non-reference system per chain column; dagger for B12/B12L under the decision rule."""
    cols = CHAIN + ["hum"]
    for name in cols:
        cands = {s: summary[(s, name, "mf1")] for s in MAIN if (s, name, "mf1") in summary}
        if not cands:
            continue
        best = max(cands, key=lambda s: cands[s][0])
        k = f"{best}/{name}/mf1"
        R.set(k, f"\\textbf{{{R.vals[k].split(chr(92) + 'pmsd')[0]}}}\\pmsd" + R.vals[k].split("\\pmsd")[1])
    for ours in OURS:
        pv, diffs = {}, {}
        for name in CHAIN:
            if (ours, name, "mf1") not in summary:
                continue
            base = {s: summary[(s, name, "mf1")] for s in MAIN if s not in OURS and (s, name, "mf1") in summary}
            if not base:
                continue
            b = max(base, key=lambda s: base[s][0])
            pv[name] = paired_boot_p(preds, data, ours, b, name, n=300)
            ma, sa, _ = summary[(ours, name, "mf1")]; mb, sb, _ = base[b]
            diffs[name] = (ma - mb, max(sa, sb))
        for name, padj in holm(pv).items():
            diff, sdmax = diffs[name]
            if padj < 0.05 and diff > sdmax:
                k = f"{ours}/{name}/mf1"
                head, tail = R.vals[k].split("\\pmsd")
                R.set(k, f"{head}\\sig\\pmsd{tail}")


def decomposition(R, preds, data, n=N_BOOT):
    need = ["ar", "tz", "myar", "my"]
    if not all(s in data for s in need):
        return
    ceil = {}
    for s in need:
        v = per_seed_metric(preds, data, "CEIL", s, "mf1")
        if v:
            ceil[s] = float(np.mean(v))

    def mean_f1(system, name, idx=None):
        y = data[name]["y"]
        vals = []
        for seed in seeds_with(preds, system, [name]):
            p = preds[system][seed][name]["pred"]
            vals.append(f1_macro(y[idx], p[idx]) if idx is not None else f1_macro(y, p))
        return float(np.mean(vals))

    def effects(F):
        a, b, c, d = F["ar"], F["tz"], F["myar"], F["my"]
        return {"script": 0.5 * ((a - b) + (c - d)), "dial": 0.5 * ((a - c) + (b - d)), "int": (c - d) - (a - b)}

    for system in preds:
        if not seeds_with(preds, system, need):
            continue
        F = {s: mean_f1(system, s) for s in need}
        point = effects(F)
        boots = defaultdict(list); boots_rho = defaultdict(list)
        for _ in range(n // 4):
            Fb = {s: mean_f1(system, s, strat_boot_idx(data[s]["y"])) for s in need}
            for k, v in effects(Fb).items():
                boots[k].append(v)
            if len(ceil) == 4:
                for k, v in effects({s: Fb[s] / ceil[s] for s in need}).items():
                    boots_rho[k].append(v)
        for k in ("script", "dial", "int"):
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            R.set(f"{system}/dec/{k}", fmt_ci(100 * point[k], 100 * lo, 100 * hi))
        R.set(f"{system}/dec/total", fmt(100 * (F["ar"] - F["my"])))
        if len(ceil) == 4:
            pr = effects({s: F[s] / ceil[s] for s in need})
            for k in ("script", "dial", "int"):
                lo, hi = np.percentile(boots_rho[k], [2.5, 97.5])
                R.set(f"{system}/decrho/{k}", fmt_ci(100 * pr[k], 100 * lo, 100 * hi))
        for key, (s1, s2) in {"plat/effect": ("tl", "tz"), "dz/script": ("naar", "na")}.items():
            if s1 in data and s2 in data and seeds_with(preds, system, [s1, s2]):
                pt = mean_f1(system, s1) - mean_f1(system, s2)
                bs = [mean_f1(system, s1, strat_boot_idx(data[s1]["y"])) -
                      mean_f1(system, s2, strat_boot_idx(data[s2]["y"])) for _ in range(n // 4)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                R.set(f"{system}/{key}", fmt_ci(100 * pt, 100 * lo, 100 * hi))


# ------------------------------------------------------------- model-free diagnostics
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+|@\w+")


def tokens(text):
    return TOKEN_RE.findall(URL_RE.sub(" ", text.lower()))


def char_ngrams(texts, n=3):
    c = Counter()
    for t in texts:
        s = " " + " ".join(tokens(t)) + " "
        c.update(s[i:i + n] for i in range(len(s) - n + 1))
    return c


def jsd(c1, c2):
    keys = set(c1) | set(c2); n1 = sum(c1.values()); n2 = sum(c2.values())
    p = np.array([c1[k] / n1 for k in keys]); q = np.array([c2[k] / n2 for k in keys])
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ngrams_of(toks):
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


def polarity_cues(texts, y, k=100, min_count=5, a0=1000.0):
    """Weighted log-odds with informative Dirichlet prior (Monroe et al., 2008)."""
    cnt = [Counter(), Counter()]
    for t, lab in zip(texts, y):
        cnt[lab].update(ngrams_of(tokens(t)))
    tot = cnt[0] + cnt[1]; N = sum(tot.values())
    n = [sum(cnt[0].values()), sum(cnt[1].values())]
    z = {}
    for w, c in tot.items():
        if c < min_count:
            continue
        aw = a0 * c / N
        l1 = math.log((cnt[1][w] + aw) / (n[1] + a0 - cnt[1][w] - aw))
        l0 = math.log((cnt[0][w] + aw) / (n[0] + a0 - cnt[0][w] - aw))
        z[w] = (l1 - l0) / math.sqrt(1 / (cnt[1][w] + aw) + 1 / (cnt[0][w] + aw))
    ranked = sorted(z, key=z.get)
    return {w: 0 for w in ranked[:k]} | {w: 1 for w in ranked[-k:]}


def sentence_features(text, vocab, cues):
    toks = tokens(text); grams = ngrams_of(toks)
    hits = [cues[g] for g in grams if g in cues]
    letters = [t for t in toks if re.search(r"[^\W\d_]", t)]
    digit = sum(1 for t in letters if any(ch in DIGIT_LETTERS for ch in t))
    return {"len": len(toks), "oov": (sum(t not in vocab for t in toks) / len(toks)) if toks else 0.0,
            "cue": int(bool(hits)), "vote": (1 if sum(hits) * 2 > len(hits) else 0 if sum(hits) * 2 < len(hits) else None) if hits else None,
            "digit": digit / len(letters) if letters else 0.0,
            "mktn": sum(t in MARKERS["tn"] for t in toks), "mkma": sum(t in MARKERS["ma"] for t in toks),
            "mkdz": sum(t in MARKERS["dz"] for t in toks)}


def model_free(R, data, minority):
    refs = {}
    for script, name in (("ar", "train_ar"), ("lat", "train_gs")):
        if name in data:
            d = data[name]
            refs[script] = {"vocab": {t for x in d["text"] for t in tokens(x)},
                            "cng": char_ngrams(d["text"]), "cues": polarity_cues(d["text"], d["y"])}
    feats = {}
    core = {"train_ar", "val_ar", "ar", "gs", "gu", "tl", "tz", "na", "my", "naar", "myar", "gsmy"}
    for name, d in data.items():
        if name not in core:
            continue
        n = len(d["y"])
        if name == "train_gs":
            continue
        if name == "train_ar":
            key = "train"
        elif name == "val_ar":
            key = "val"
        else:
            key = name
        R.set(f"data/{key}/n", fmt_int(n))
        R.set(f"data/{key}/pos", fmt(100 * d["y"].mean()))
        R.set(f"data/{key}/len", fmt(np.mean([len(tokens(t)) for t in d["text"]])))
        script = "ar" if name in ARABIC_SETS else "lat"
        if script not in refs or name.startswith("train") or name == "val_ar":
            continue
        ref = refs[script]
        F = [sentence_features(t, ref["vocab"], ref["cues"]) for t in d["text"]]
        feats[name] = F
        ntok = sum(f["len"] for f in F) or 1
        R.set(f"data/{name}/oov", fmt(100 * sum(f["oov"] * f["len"] for f in F) / ntok))
        R.set(f"data/{name}/jsd", fmt(100 * jsd(char_ngrams(d["text"]), ref["cng"])))
        R.set(f"data/{name}/cuecov", fmt(100 * np.mean([f["cue"] for f in F])))
        votes = [(f["vote"] if f["vote"] is not None else 1 - minority, yy) for f, yy in zip(F, d["y"]) if f["cue"]]
        if votes:
            R.set(f"data/{name}/cueacc", fmt(100 * np.mean([v == yy for v, yy in votes])))
        for m in ("tn", "ma", "dz"):
            R.set(f"data/{name}/mk{m}", fmt(100 * sum(f["mk" + m] for f in F) / ntok, 2))
        if script == "lat":
            R.set(f"data/{name}/digit", fmt(100 * np.mean([f["digit"] for f in F])))
    return feats


# ------------------------------------------------------------- model-based diagnostics
def load_diag(diag_dir, data):
    out = {}
    if not diag_dir or not os.path.isdir(diag_dir):
        return out
    for system in os.listdir(diag_dir):
        for seed in os.listdir(os.path.join(diag_dir, system)):
            for fn in os.listdir(os.path.join(diag_dir, system, seed)):
                name = fn[:-4]
                rows = read_tsv(os.path.join(diag_dir, system, seed, fn))
                out.setdefault(system, {}).setdefault(seed, {})[name] = rows
    return out


def model_diagnostics(R, diag, preds, data, emb_dir):
    for system, seeds in diag.items():
        acc = defaultdict(list)
        for seed, sets in seeds.items():
            xi = None
            if "val" in sets:
                xi = np.percentile([float(r["maxproto"]) for r in sets["val"]], 5)
            for name, rows in sets.items():
                if name == "val":
                    continue
                alphas = np.array([[float(r[k]) for k in sorted(r) if k.startswith("alpha")] for r in rows])
                for m in range(alphas.shape[1]):
                    acc[(name, f"alpha{m + 1}")].append(alphas[:, m].mean())
                acc[(name, "hgate")].append(float(np.mean(-np.sum(alphas * np.log(np.clip(alphas, 1e-12, 1)), 1))))
                if "plat" in rows[0]:
                    acc[(name, "plat")].append(np.mean([float(r["plat"]) for r in rows]))
                if xi is not None and "maxproto" in rows[0]:
                    acc[(name, "os")].append(100 * np.mean([float(r["maxproto"]) < xi for r in rows]))
                # gate-reliability correlation with single-encoder correctness
                if name in data:
                    y = dict(zip(data[name]["id"], data[name]["y"]))
                    rs = []
                    for m in range(alphas.shape[1]):
                        enc = f"B3-e{m + 1}"
                        s3 = seeds_with(preds, enc, [name])
                        if not s3:
                            continue
                        pr = dict(zip(data[name]["id"], preds[enc][s3[0]][name]["pred"]))
                        corr = np.array([float(pr[r["id"]] == y[r["id"]]) for r in rows])
                        if corr.std() > 0 and alphas[:, m].std() > 0:
                            rs.append(np.corrcoef(alphas[:, m], corr)[0, 1])
                    if rs:
                        acc[(name, "ralpha")].append(float(np.mean(rs)))
        for (name, metric), v in acc.items():
            R.set(f"{system}/{name}/{metric}:m", fmt(np.mean(v), 1 if metric == "os" else 2))
    # proxy A-distance from fused embeddings
    if emb_dir and os.path.isdir(emb_dir):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        for system in os.listdir(emb_dir):
            ref_p = os.path.join(emb_dir, system, "train_gs.npy")
            if not os.path.exists(ref_p):
                continue
            ref = np.load(ref_p)
            for fn in os.listdir(os.path.join(emb_dir, system)):
                name = fn[:-4]
                if name.startswith("train") or name not in data:
                    continue
                X = np.load(os.path.join(emb_dir, system, fn))
                k = min(len(X), len(ref))
                A = ref[RNG.choice(len(ref), k, replace=False)]; B = X[RNG.choice(len(X), k, replace=False)]
                Z = np.vstack([A, B]); t = np.r_[np.zeros(k), np.ones(k)]
                err = 1 - cross_val_score(LogisticRegression(max_iter=2000), Z, t, cv=5).mean()
                R.set(f"{system}/{name}/pad:m", fmt(2 * (1 - 2 * err), 2))


def error_attribution(R, preds, data, feats, diag, system="B12", n_boot=200):
    from sklearn.linear_model import LogisticRegression
    for name in ("tz", "my"):
        if name not in feats:
            continue
        seeds = seeds_with(preds, system, [name])
        if not seeds:
            continue
        votes = np.mean([preds[system][s][name]["pred"] for s in seeds], 0)
        err = ((votes >= 0.5).astype(int) != data[name]["y"]).astype(int)
        F = feats[name]
        cols = ["cue", "oov", "len", "digit", "mktn", "mkma"]
        X = np.array([[f[c] for c in cols] for f in F], float)
        if system in diag:
            seed0 = sorted(diag[system])[0]
            sets = diag[system][seed0]
            if name in sets and "val" in sets and "maxproto" in sets[name][0]:
                xi = np.percentile([float(r["maxproto"]) for r in sets["val"]], 5)
                os_map = {r["id"]: float(float(r["maxproto"]) < xi) for r in sets[name]}
                X = np.c_[X, [os_map.get(i, 0.0) for i in data[name]["id"]]]
                cols = cols + ["os"]
        keep = X.std(0) > 0
        Xs = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
        names = [c for c, k in zip(cols, keep) if k]
        if err.min() == err.max():
            continue
        fit = lambda A, b: LogisticRegression(C=1e6, max_iter=5000).fit(A, b).coef_[0]
        coef = fit(Xs, err)
        boots = []
        for _ in range(n_boot):
            idx = RNG.integers(0, len(err), len(err))
            if err[idx].min() != err[idx].max():
                boots.append(fit(Xs[idx], err[idx]))
        boots = np.array(boots)
        for j, c in enumerate(names):
            lo, hi = np.percentile(np.exp(boots[:, j]), [2.5, 97.5])
            R.set(f"{system}/{name}/or-{c}", fmt_ci(math.exp(coef[j]), lo, hi, 2))


# ------------------------------------------------------------- audits, factorial, figures
def audits(R, audit_dir):
    if not audit_dir or not os.path.isdir(audit_dir):
        return
    for gen in ("gs", "gu"):
        p = os.path.join(audit_dir, f"gen_{gen}.tsv")
        if os.path.exists(p):
            rows = read_tsv(p)
            a = [int(r["ann1"]) for r in rows]; b = [int(r["ann2"]) for r in rows]
            has_final = rows and "final" in rows[0] and all(str(r["final"]).strip() != "" for r in rows)
            final = [int(r["final"]) for r in rows] if has_final else [x * y for x, y in zip(a, b)]
            R.set(f"audit/{gen}/accept", fmt(100 * np.mean(final)))
            R.set(f"audit/{gen}/kappa", fmt(cohen_kappa(a, b), 2))
    for name in ("tz", "my"):
        p = os.path.join(audit_dir, f"errors_{name}.tsv")
        if os.path.exists(p):
            rows = read_tsv(p)
            fin = Counter(r["final"] for r in rows)
            for e in ("E1", "E2", "E3", "E4", "E5", "E6"):
                R.set(f"audit/{name}/{e}", fmt(100 * fin[e] / len(rows)))
            R.set(f"audit/{name}/kappa", fmt(cohen_kappa([r["ann1"] for r in rows], [r["ann2"] for r in rows]), 2))


def factorial(R, preds, data, out):
    systems = [s for s in preds if re.fullmatch(r"fact-[01]{4}", s)]
    if len(systems) < 16:
        return
    rows, ys = [], []
    stages = [s for s in CHAIN if s in data]
    for s in systems:
        bits = [int(b) for b in s[5:]]
        inter = [bits[i] * bits[j] for i in range(4) for j in range(i + 1, 4)]
        for seed in preds[s]:
            for k, st in enumerate(stages):
                if st in preds[s][seed]:
                    dummies = [1.0 if k == j else 0.0 for j in range(1, len(stages))]
                    rows.append([1.0] + bits + inter + dummies)
                    ys.append(100 * f1_macro(data[st]["y"], preds[s][seed][st]["pred"]))
    X = np.array(rows); y = np.array(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    names = ["Fusion", "Proto", "HM", "Cons"]
    terms = names + [f"{names[i]}x{names[j]}" for i in range(4) for j in range(i + 1, 4)]
    table = {}
    for j, t in enumerate(terms, start=1):
        R.set(f"factorial/{t}", fmt_ci(beta[j], beta[j] - 1.96 * se[j], beta[j] + 1.96 * se[j], 2))
        table[t] = [float(beta[j]), float(se[j])]
    json.dump(table, open(os.path.join(out, "factorial.json"), "w"), indent=1)


def write_figures(preds, data, emb_dir, out):
    def stat(system, name):
        v = per_seed_metric(preds, data, system, name, "mf1") if name in data else []
        if not v:
            return float("nan"), float("nan")
        b = boot_mf1(preds, data, system, name, n=200)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return 100 * float(np.mean(v)), 100 * float(hi - lo) / 2
    systems = ["B12", "B10", "B6", "B0", "CEIL"]
    if any(s in preds for s in systems) and "ar" in data:
        with open(os.path.join(out, "chain.dat"), "w") as f:
            f.write("stage " + " ".join(f"{s} {s}err" for s in systems) + "\n")
            for i, name in enumerate(CHAIN):
                vals = []
                for s in systems:
                    m, e = stat(s, name) if (s != "CEIL" or name not in ("gs", "gu")) else (float("nan"),) * 2
                    vals += [m, e]
                f.write(f"{i} " + " ".join("nan" if math.isnan(v) else f"{v:.2f}" for v in vals) + "\n")
    for system, col, tag in (("B6", "p_glob", "glob"), ("B12", "p_var", "var")):
        curves = {}
        for name in HUMAN:
            seeds = seeds_with(preds, system, [name])
            if name in data and seeds and col in preds[system][seeds[0]][name]:
                y = np.concatenate([data[name]["y"]] * len(seeds))
                p = np.concatenate([preds[system][s][name]["pred"] for s in seeds])
                q = np.concatenate([preds[system][s][name][col] for s in seeds])
                curves[name] = reliability(*conf_correct(y, p, q))
        if curves:
            with open(os.path.join(out, f"rel-{system}-{tag}.dat"), "w") as f:
                f.write(" ".join(f"{n}_conf {n}_acc" for n in HUMAN) + "\n")
                for b in range(15):
                    row = []
                    for n in HUMAN:
                        row += list(curves[n][b]) if n in curves and b < len(curves[n]) else [float("nan")] * 2
                    f.write(" ".join("nan" if math.isnan(v) else f"{v:.4f}" for v in row) + "\n")
    sev = [f"rob-tru-s{k}" for k in range(5)]
    if all(s in data for s in sev) and any(s in preds for s in ("B12", "B11", "B6")):
        with open(os.path.join(out, "robcurve-tru.dat"), "w") as f:
            f.write("sev B12 B12err B11 B11err B6 B6err\n")
            for k, name in enumerate(sev):
                row = []
                for s in ("B12", "B11", "B6"):
                    v = per_seed_metric(preds, data, s, name, "acc")
                    row += [100 * np.mean(v), 100 * 1.96 * np.std(v, ddof=1) / math.sqrt(len(v))] if len(v) > 1 else [float("nan")] * 2
                f.write(f"{k} " + " ".join("nan" if math.isnan(v) else f"{v:.2f}" for v in row) + "\n")
    if emb_dir and os.path.isdir(emb_dir):
        try:
            import umap  # umap-learn
        except ImportError:
            print("[info] umap-learn not installed: UMAP figure skipped", file=sys.stderr)
            return
        for system in ("B6", "B12"):
            parts, cls = [], []
            for name in ("ar", "gs", "tl", "my", "myar"):
                p = os.path.join(emb_dir, system, f"{name}.npy")
                if os.path.exists(p) and name in data:
                    X = np.load(p); idx = RNG.choice(len(X), min(400, len(X)), replace=False)
                    parts.append(X[idx]); cls += [f"{name}{data[name]['y'][i]}" for i in idx]
            if parts:
                Y = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(np.vstack(parts))
                with open(os.path.join(out, f"umap-{system}.dat"), "w") as f:
                    f.write("x y cls\n")
                    for (a, b), c in zip(Y, cls):
                        f.write(f"{a:.4f} {b:.4f} {c}\n")


def write_cis(preds, data, out):
    """95% bootstrap CIs of macro-F1 for every system on every chain/control set (ci.csv)."""
    with open(os.path.join(out, "ci.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "set", "mf1_mean", "ci_low", "ci_high", "n_seeds"])
        for system in sorted(preds):
            if system not in MAIN + ["CEIL", "B12-nopc"]:
                continue
            for name in CHAIN + ["naar", "myar", "gsmy"]:
                v = per_seed_metric(preds, data, system, name, "mf1") if name in data else []
                if not v:
                    continue
                lo, hi = np.percentile(boot_mf1(preds, data, system, name), [2.5, 97.5])
                w.writerow([system, name, f"{100 * np.mean(v):.2f}", f"{100 * lo:.2f}", f"{100 * hi:.2f}", len(v)])


# ------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--preds")
    ap.add_argument("--diag"); ap.add_argument("--emb"); ap.add_argument("--audit")
    ap.add_argument("--manual"); ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {}
    for fn in sorted(os.listdir(args.data)):
        if fn.endswith(".tsv"):
            d = load_set(args.data, fn[:-4])
            if d is not None:
                data[fn[:-4]] = d
    minority = int(np.mean(data["train_ar"]["y"]) > 0.5) ^ 1 if "train_ar" in data else 0
    R = Results()
    preds = load_preds(args.preds, data)
    summary = classification_metrics(R, preds, data, minority)
    bold_and_significance(R, preds, data, summary)
    decomposition(R, preds, data)
    feats = model_free(R, data, minority)
    diag = load_diag(args.diag, data)
    model_diagnostics(R, diag, preds, data, args.emb)
    error_attribution(R, preds, data, feats, diag)
    audits(R, args.audit)
    factorial(R, preds, data, args.out)
    write_figures(preds, data, args.emb, args.out)
    write_cis(preds, data, args.out)
    if args.manual:
        for k, v in json.load(open(args.manual, encoding="utf-8")).items():
            R.set(k, v)
    R.write(os.path.join(args.out, "results.tex"))
    json.dump({f"{a}/{b}/{c}": [m, s] for (a, b, c), (m, s, _) in summary.items()},
              open(os.path.join(args.out, "summary.json"), "w"), indent=0)
    print(f"wrote {len(R.vals)} values to {os.path.join(args.out, 'results.tex')}")


if __name__ == "__main__":
    main()
