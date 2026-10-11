"""Reconstruct the Tunisian/Arabizi protocol from the public releases.

Writes data/<set>.tsv (id, text, label) in the format of diafuse_eval.py and
data/prep_report.json with every count of Table 1 and Appendix F.
"""
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))
from translit import (g_seen, g_unseen, script_of, partial, inf_spelling, inf_abbrev,
                      char_noise, code_switch, latin_to_arabic)

RAW = os.environ.get("RAW_DIR", "raw"); OUT = os.environ.get("DATA_DIR", "data")
THETA = float(os.environ.get("NEAR_THETA", "0.9")); SEED = 13
os.makedirs(OUT, exist_ok=True)
report = {}

URL = re.compile(r"https?://\S+|www\.\S+|@\w+")


def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = URL.sub(" ", t)
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] in "LNPZ" or ch in "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def key(t):
    t = t.lower()
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] in "LNZ")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def dedup(rows, tag):
    """rows: list of (text, label). Exact de-duplication on the normalised key;
    keys with conflicting labels are dropped entirely."""
    by = defaultdict(list)
    for t, y in rows:
        k = key(t)
        if k:
            by[k].append((t, y))
    out, conflicts = [], 0
    for k, lst in by.items():
        labs = {y for _, y in lst}
        if len(labs) > 1:
            conflicts += 1
            continue
        out.append(lst[0])
    report[f"{tag}/raw"] = len(rows); report[f"{tag}/unique"] = len(out)
    report[f"{tag}/exact_dup_removed"] = len(rows) - len(out) - conflicts
    report[f"{tag}/label_conflict_keys"] = conflicts
    return out


VEC = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)


def near_mask(cands, refs):
    """True where a candidate has a near-copy (cosine of char 3-gram counts >= THETA) in refs."""
    if not cands or not refs:
        return np.zeros(len(cands), bool)
    VEC.fit([" " + key(t) + " " for t in cands + refs])
    A = normalize(VEC.transform([" " + key(t) + " " for t in cands]))
    B = normalize(VEC.transform([" " + key(t) + " " for t in refs]))
    hit = np.zeros(len(cands), bool)
    for s in range(0, A.shape[0], 500):
        S = A[s:s + 500] @ B.T
        hit[s:s + 500] = np.asarray(S.max(axis=1).todense()).ravel() >= THETA
    return hit


def within_near(rows):
    """Remove near-copies inside a set (keep the first occurrence)."""
    texts = [t for t, _ in rows]
    if not texts:
        return rows, 0
    VEC.fit([" " + key(t) + " " for t in texts])
    A = normalize(VEC.transform([" " + key(t) + " " for t in texts]))
    keep = np.ones(len(rows), bool)
    for s in range(0, A.shape[0], 500):
        S = (A[s:s + 500] @ A.T).tocsr()
        for r in range(S.shape[0]):
            i = s + r
            if not keep[i]:
                continue
            row = S.getrow(r); js = row.indices[row.data >= THETA]
            for j in js:
                if j > i:
                    keep[j] = False
    return [r for r, k in zip(rows, keep) if k], int((~keep).sum())


def split_script(rows, tag):
    by = defaultdict(list)
    for t, y in rows:
        by[script_of(t)].append((t, y))
    report[f"{tag}/mixed"] = len(by["mixed"]); report[f"{tag}/noletters"] = len(by["none"])
    return by["ar"], by["lat"]


def write(name, rows, ids=None):
    with open(os.path.join(OUT, f"{name}.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "text", "label"])
        for i, (t, y) in enumerate(rows):
            w.writerow([ids[i] if ids else f"{name}-{i}", t.replace("\t", " "), int(y)])
    report[f"final/{name}"] = len(rows)


def read_lines(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f if l.strip()]


OWN_SPLITS = os.environ.get("OWN_SPLITS")
if not OWN_SPLITS:
    # ------------------------------------------------------------------ TSAC
    tsac_train = [(clean(t), 1) for t in read_lines(f"{RAW}/TSAC/train_pos.txt")] + \
                 [(clean(t), 0) for t in read_lines(f"{RAW}/TSAC/train_neg.txt")]
    tsac_test = [(clean(t), 1) for t in read_lines(f"{RAW}/TSAC/test_pos.txt")] + \
                [(clean(t), 0) for t in read_lines(f"{RAW}/TSAC/test_neg.txt")]
    test_u = dedup(tsac_test, "tsac_test")
    val_ar, test_lat = split_script(test_u, "tsac_test")
    report["tsac_test/arabic"] = len(val_ar); report["tsac_test/latin"] = len(test_lat)

    train_u = dedup(tsac_train, "tsac_train")
    test_keys = {key(t) for t, _ in test_u}
    train_u = [r for r in train_u if key(r[0]) not in test_keys]
    report["tsac_train/removed_in_test_split"] = report["tsac_train/unique"] - len(train_u)
    train_ar, train_lat = split_script(train_u, "tsac_train")
    report["tsac_train/arabic"] = len(train_ar); report["tsac_train/latin"] = len(train_lat)
    # near-copies of validation inside training (Arabic script)
    nc = near_mask([t for t, _ in train_ar], [t for t, _ in val_ar])
    train_ar = [r for r, h in zip(train_ar, nc) if not h]
    report["tsac_train/near_copy_of_val_removed"] = int(nc.sum())

    rng = random.Random(SEED)
    idx = {0: [], 1: []}
    for i, (_, y) in enumerate(train_ar):
        idx[y].append(i)
    s0 = set()
    for y, lst in idx.items():
        rng.shuffle(lst); s0 |= set(lst[: round(0.1 * len(lst))])
    S0 = [train_ar[i] for i in sorted(s0)]
    TR = [train_ar[i] for i in range(len(train_ar)) if i not in s0]

    # TSAC-Latin: human Latin-script TSAC sentences (training-split Latin + test-split Latin),
    # minus within-set near-copies and near-copies of any training/validation sentence
    tl_pool = dedup(train_lat + test_lat, "tsac_latin_pool")
    tl_pool, n_within = within_near(tl_pool)
    report["tsac_latin/near_within_removed"] = n_within
    ref_lat = [g_seen(t) for t, _ in TR + val_ar + S0] + [g_unseen(t) for t, _ in TR + val_ar + S0]
    hit = near_mask([t for t, _ in tl_pool], ref_lat)
    report["tsac_latin/cross_script_removed"] = int(hit.sum())
    TL = [r for r, h in zip(tl_pool, hit) if not h]

    # reference pool for external overlap: all TSAC in both scripts (+ transliterations)
    tsac_all = [t for t, _ in train_u + test_u]
    tsac_ref_lat = [t for t in tsac_all if script_of(t) == "lat"] + \
                   [g_seen(t) for t in tsac_all if script_of(t) == "ar"] + \
                   [g_unseen(t) for t in tsac_all if script_of(t) == "ar"]
    tsac_ref_ar = [t for t in tsac_all if script_of(t) == "ar"]


    def external(rows, tag, script="lat"):
        rows = dedup(rows, tag)
        ar, lat = split_script(rows, tag)
        rows = lat if script == "lat" else ar
        report[f"{tag}/in_script"] = len(rows)
        rows, n_within = within_near(rows)
        report[f"{tag}/near_within_removed"] = n_within
        ref = tsac_ref_lat if script == "lat" else tsac_ref_ar
        hit = near_mask([t for t, _ in rows], ref)
        report[f"{tag}/tsac_overlap_removed"] = int(hit.sum())
        return [r for r, h in zip(rows, hit) if not h]


    # ------------------------------------------------------------------ TUNIZI
    tz_rows = []
    for l in read_lines(f"{RAW}/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset/TUNIZI-Dataset.txt"):
        if ";" in l:
            lab, txt = l.split(";", 1)
            if lab.strip() in ("1", "-1"):
                tz_rows.append((clean(txt), 1 if lab.strip() == "1" else 0))
    TZ = external(tz_rows, "tunizi")

    # ------------------------------------------------------------------ NArabizi (parallel)
    na, naar = [], []
    for sp in ("train", "dev", "test"):
        a = list(csv.reader(open(f"{RAW}/Narabizi/data/sentiment/ar_na/{sp}.csv", encoding="utf-8")))
        b = list(csv.reader(open(f"{RAW}/Narabizi/data/sentiment/ar_dz/{sp}.csv", encoding="utf-8")))
        for x, y in zip(a, b):
            na.append((clean(x[1]), int(x[0]))); naar.append((clean(y[1]), int(y[0])))
    report["narabizi/raw_pairs"] = len(na)
    seen_k, pairs = set(), []
    for (t, y), (ta, _) in zip(na, naar):
        k = key(t)
        if k and k not in seen_k and script_of(t) == "lat":
            seen_k.add(k); pairs.append(((t, y), (ta, y)))
    report["narabizi/unique_latin"] = len(pairs)
    hit = near_mask([p[0][0] for p in pairs], tsac_ref_lat)
    report["narabizi/tsac_overlap_removed"] = int(hit.sum())
    pairs = [p for p, h in zip(pairs, hit) if not h]
    NA = [p[0] for p in pairs]; NAAR = [p[1] for p in pairs]

    # ------------------------------------------------------------------ MYC
    myc = pd.read_csv(f"{RAW}/MYC/DATA_CLEANED.csv", encoding="utf-16")
    report["myc/raw"] = len(myc); report["myc/polarity_values"] = sorted(map(int, myc.polarity.unique()))
    my_rows = [(clean(t), 1 if int(p) == 1 else 0) for t, p in zip(myc.sentence, myc.polarity)]
    MY = external(my_rows, "myc_lat", "lat")
    MYAR_all = external(my_rows, "myc_ar", "ar")
    report["myc_ar/after_filtering"] = len(MYAR_all)
    # control sampled to match S6 in size and class balance
    r2 = random.Random(SEED)
    n_pos = int(round(np.mean([y for _, y in MY]) * len(MY))); n_neg = len(MY) - n_pos
    pos = [r for r in MYAR_all if r[1] == 1]; neg = [r for r in MYAR_all if r[1] == 0]
    r2.shuffle(pos); r2.shuffle(neg)
    MYAR = pos[:min(n_pos, len(pos))] + neg[:min(n_neg, len(neg))]
    r2.shuffle(MYAR)

else:
    # your own base splits (id, text, label): derived views are generated exactly as for the reconstruction
    def load_base(name):
        with open(os.path.join(OWN_SPLITS, f"{name}.tsv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        return [(clean(r["text"]), int(r["label"])) for r in rows], [r["id"] for r in rows]
    TR, _ = load_base("train_ar"); val_ar, _ = load_base("val_ar"); S0, _ = load_base("ar")
    TL, _ = load_base("tl"); TZ, _ = load_base("tz"); MY, _ = load_base("my"); MYAR, _ = load_base("myar")
    NA, _ida = load_base("na"); _nb, _idb = load_base("naar"); _m = dict(zip(_idb, _nb))
    NAAR = [_m[i] for i in _ida]  # naar.tsv must use the ids of na.tsv
    report["own_splits"] = OWN_SPLITS
# ------------------------------------------------------------------ write sets
# shuffle once (fixed seed) so that files are not ordered by class
_r = random.Random(SEED)
for _lst in (TR, S0, val_ar, TL, TZ, MY):
    _r.shuffle(_lst)
_perm = list(range(len(NA))); _r.shuffle(_perm)
NA = [NA[i] for i in _perm]; NAAR = [NAAR[i] for i in _perm]
ids_s0 = [f"s0-{i}" for i in range(len(S0))]
write("train_ar", TR, [f"tr-{i}" for i in range(len(TR))])
write("train_gs", [(g_seen(t), y) for t, y in TR], [f"tr-{i}" for i in range(len(TR))])
write("val_ar", val_ar, [f"va-{i}" for i in range(len(val_ar))])
write("val_gs", [(g_seen(t), y) for t, y in val_ar], [f"va-{i}" for i in range(len(val_ar))])
write("ar", S0, ids_s0)
write("gs", [(g_seen(t), y) for t, y in S0], ids_s0)
write("gu", [(g_unseen(t), y) for t, y in S0], ids_s0)
write("tl", TL); write("tz", TZ)
ids_na = [f"na-{i}" for i in range(len(NA))]
write("na", NA, ids_na); write("naar", NAAR, ids_na)
ids_my = [f"myar-{i}" for i in range(len(MYAR))]
write("my", MY); write("myar", MYAR, ids_my)
write("gsmy", [(g_seen(t), y) for t, y in MYAR], ids_my)
write("train_gu", [(g_unseen(t), y) for t, y in TR], [f"tr-{i}" for i in range(len(TR))])
# transliterate-test inputs for B0-TT (Latin-script sets mapped back to Arabic script)
for name, rows, ids in (("gs", [(g_seen(t), y) for t, y in S0], ids_s0), ("gu", [(g_unseen(t), y) for t, y in S0], ids_s0),
                        ("tl", TL, None), ("tz", TZ, None), ("na", NA, ids_na), ("my", MY, None),
                        ("gsmy", [(g_seen(t), y) for t, y in MYAR], ids_my)):
    write("tt-" + name, [(latin_to_arabic(t), y) for t, y in rows], ids if ids else [f"{name}-{i}" for i in range(len(rows))])
# training-time variants (consistency) for the Latin view
write("train_inf", [(inf_spelling(g_seen(t), 0.2, i), y) for i, (t, y) in enumerate(TR)], [f"tr-{i}" for i in range(len(TR))])
write("train_noise", [(char_noise(g_seen(t), 0.2, i), y) for i, (t, y) in enumerate(TR)], [f"tr-{i}" for i in range(len(TR))])
write("train_cs", [(g_seen(code_switch(t, 0.2, i)), y) for i, (t, y) in enumerate(TR)], [f"tr-{i}" for i in range(len(TR))])
# robustness sets at severity 2 (20% of tokens) and the severity curve for G_unseen
gs_s0 = [(g_seen(t), y) for t, y in S0]
write("rob-trs", [(partial(t, g_seen, 0.2, i), y) for i, (t, y) in enumerate(S0)], ids_s0)
write("rob-tru", [(partial(t, g_unseen, 0.2, i), y) for i, (t, y) in enumerate(S0)], ids_s0)
write("rob-spell", [(inf_spelling(t, 0.2, i), y) for i, (t, y) in enumerate(gs_s0)], ids_s0)
write("rob-abbr", [(inf_abbrev(t, 0.2, i), y) for i, (t, y) in enumerate(gs_s0)], ids_s0)
write("rob-cs", [(g_seen(code_switch(t, 0.2, i)), y) for i, (t, y) in enumerate(S0)], ids_s0)
write("rob-noise", [(char_noise(t, 0.2, i), y) for i, (t, y) in enumerate(gs_s0)], ids_s0)
for k, frac in enumerate([0.0, 0.1, 0.2, 0.35, 0.5]):
    write(f"rob-tru-s{k}", [(partial(t, g_unseen, frac, i), y) for i, (t, y) in enumerate(S0)], ids_s0)

# optional MADAR Tunis/Sfax-MSA pairs for the +Cross-lingual ablation (licence: request from the MADAR authors)
MADAR_DIR = os.environ.get("MADAR_DIR")
if MADAR_DIR and os.path.isdir(MADAR_DIR):
    by_sid = defaultdict(dict)
    for root, _, files in os.walk(MADAR_DIR):
        for fn in files:
            if not fn.endswith(".tsv"):
                continue
            df = pd.read_csv(os.path.join(root, fn), sep="\t", quoting=3, dtype=str)
            cols = {c.lower(): c for c in df.columns}
            lc = next((cols[c] for c in cols if c.startswith("lang")), None)
            sc_ = next((cols[c] for c in cols if c.startswith("sent") and "id" not in c), None)
            ic = next((cols[c] for c in cols if "id" in c), None)
            spc = next((cols[c] for c in cols if c.startswith("split")), None)
            if not (lc and sc_ and ic):
                continue
            for _, r in df.iterrows():
                if spc and str(r[spc]).lower().startswith(("dev", "test")):
                    continue
                by_sid[str(r[ic])][str(r[lc]).upper()] = clean(r[sc_])
    pairs = [(d[c], d["MSA"]) for d in by_sid.values() for c in ("TUN", "SFX") if c in d and "MSA" in d]
    report["madar/pairs"] = len(pairs)
    ids_m = [f"madar-{i}" for i in range(len(pairs))]
    write("train_madar_tn", [(a, 0) for a, _ in pairs], ids_m)
    write("train_madar_msa", [(b, 0) for _, b in pairs], ids_m)

for name, rows in (("train_ar", TR), ("ar", S0), ("tl", TL), ("tz", TZ), ("na", NA), ("my", MY), ("myar", MYAR)):
    report[f"pos_share/{name}"] = round(float(np.mean([y for _, y in rows])), 4)
json.dump(report, open(os.path.join(OUT, "prep_report.json"), "w"), indent=1, ensure_ascii=False)
# counts for Table 1 / Appendix F of the paper (read by diafuse_eval.py --manual)
if not OWN_SPLITS:
    fi = lambda n: f"{int(n):,}"
    R = report
    manual = {
        "data/train/n0": fi(R["tsac_train/arabic"]),
        "data/train/rm": fi(R["tsac_train/near_copy_of_val_removed"] + len(S0)),
        "data/val/n0": fi(R["tsac_test/unique"]), "data/val/rm": fi(R["tsac_test/unique"] - len(val_ar)),
        "data/val/latin": fi(R["tsac_test/latin"]), "data/val/n": fi(len(val_ar)),
        "data/tl/n0": fi(R["tsac_latin_pool/unique"]),
        "data/tl/rm": fi(R["tsac_latin/near_within_removed"] + R["tsac_latin/cross_script_removed"]),
        "data/tz/n0": fi(R["tunizi/unique"]), "data/tz/rm": fi(R["tunizi/near_within_removed"] + R["tunizi/tsac_overlap_removed"]),
        "data/na/n0": fi(R["narabizi/unique_latin"]), "data/na/rm": fi(R["narabizi/tsac_overlap_removed"]),
        "data/my/n0": fi(R["myc_lat/in_script"]), "data/my/rm": fi(R["myc_lat/near_within_removed"] + R["myc_lat/tsac_overlap_removed"]),
        "data/myar/n0": fi(R["myc_ar/in_script"]), "data/myar/rm": fi(R["myc_ar/near_within_removed"] + R["myc_ar/tsac_overlap_removed"]),
        "data/human/total": fi(len(TL) + len(TZ) + len(NA) + len(MY)),
        "data/train/mixed": fi(R["tsac_train/mixed"]), "data/train/rm-exact": fi(R["tsac_train/exact_dup_removed"]),
        "data/train/rm-within": fi(R["tsac_train/near_copy_of_val_removed"]),
        "data/tl/mixed": fi(R["tsac_train/mixed"] + R["tsac_test/mixed"]), "data/tl/rm-exact": fi(R["tsac_latin_pool/exact_dup_removed"]),
        "data/tl/rm-within": fi(R["tsac_latin/near_within_removed"]), "data/tl/rm-tsac": "0",
        "data/tl/rm-xscript": fi(R["tsac_latin/cross_script_removed"]),
    }
    for tag, key_ in (("tunizi", "tz"), ("myc_lat", "my"), ("myc_ar", "myar")):
        manual[f"data/{key_}/mixed"] = fi(R[f"{tag}/mixed"]); manual[f"data/{key_}/rm-exact"] = fi(R[f"{tag}/exact_dup_removed"])
        manual[f"data/{key_}/rm-within"] = fi(R[f"{tag}/near_within_removed"]); manual[f"data/{key_}/rm-tsac"] = fi(R[f"{tag}/tsac_overlap_removed"])
        manual[f"data/{key_}/rm-xscript"] = "--"
    manual.update({"data/na/mixed": "0", "data/na/rm-exact": fi(R["narabizi/raw_pairs"] - R["narabizi/unique_latin"]),
                   "data/na/rm-within": "--", "data/na/rm-tsac": fi(R["narabizi/tsac_overlap_removed"]), "data/na/rm-xscript": "--"})
    json.dump(manual, open(os.path.join(OUT, "manual_counts.json"), "w"), indent=1, ensure_ascii=False)
else:
    json.dump({"data/human/total": f"{len(TL) + len(TZ) + len(NA) + len(MY):,}"},
              open(os.path.join(OUT, "manual_counts.json"), "w"), indent=1)
for k, v in report.items():
    print(f"{k}: {v}")
