"""Figures (PNG 300 dpi + PDF). Each figure is independent: a failure is reported, not fatal."""
import csv
import glob
import json
import os
import traceback

import numpy as np

CHAIN = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
SETS = {"S0": "ar", "S1": "gs", "S2": "gu", "S3": "tl", "S4": "tz", "S5": "na", "S6": "my"}
STYLE = {"B11": ("#B5523B", "o", "B11 Robust + calibrated"), "B12": ("#7A7A7A", "s", "B12 DiaFuse-PRC (Setting A)"),
         "PPA-FINAL": ("#2F6DB5", "^", "B12++ DiaFuse-PRC++ (Setting A)"), "PPB-FINAL": ("#2E8B57", "D", "B12'++ DiaFuse-PRC++ (Setting B, LoRA)")}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=300); fig.savefig(os.path.join(out, name + ".pdf"))


def _seed_table(results_dir):
    T = {}
    for r in csv.DictReader(open(os.path.join(results_dir, "metrics", "seed_results.csv"))):
        T.setdefault((r["system"], r["stage"], r["metric"]), []).append(float(r["value"]))
    return T


def fig_chain(T, out):
    plt = _plt(); fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for s, (c, m, lab) in STYLE.items():
        ys = [T.get((s, st, "mf1")) for st in CHAIN]
        if not all(ys):
            continue
        mu = [100 * np.mean(v) for v in ys]; sd = [100 * np.std(v, ddof=1) if len(v) > 1 else 0 for v in ys]
        ax.errorbar(range(7), mu, yerr=sd, color=c, marker=m, capsize=2, lw=1.4, ms=4, label=f"{lab} (n={len(ys[0])})")
    ax.set_xticks(range(7)); ax.set_xticklabels(CHAIN); ax.set_ylabel("macro-F1 (%)"); ax.axvline(2.5, color="#BBBBBB", ls="--", lw=0.8)
    ax.set_title("Macro-F1 along the test chain (mean ± sd over seeds)"); ax.legend(frameon=False, fontsize=7); ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "fig_chain_macro_f1")


def fig_delta(T, out):
    plt = _plt(); fig, ax = plt.subplots(figsize=(6.4, 3.2)); w = 0.25; k = 0
    for s in ("B12", "PPA-FINAL", "PPB-FINAL"):
        if not all(T.get((s, st, "mf1")) and T.get(("B11", st, "mf1")) for st in CHAIN):
            continue
        d = [100 * (np.mean(T[(s, st, "mf1")]) - np.mean(T[("B11", st, "mf1")])) for st in CHAIN]
        ax.bar(np.arange(7) + (k - 1) * w, d, w, color=STYLE[s][0], label=STYLE[s][2]); k += 1
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(range(7)); ax.set_xticklabels(CHAIN)
    ax.set_ylabel("Δ macro-F1 vs B11 (points)"); ax.set_title("Difference to B11 (negative = B11 better)"); ax.legend(frameon=False, fontsize=7)
    _save(fig, out, "fig_delta_vs_B11")


def gate_stats(diag_dir, system):
    A = {}
    for st, s in SETS.items():
        vals = []
        for p in glob.glob(os.path.join(diag_dir, system, "*", f"{s}.tsv")):
            rows = list(csv.DictReader(open(p), delimiter="\t"))
            cols = [c for c in (rows[0].keys() if rows else []) if c.startswith("alpha")]
            if cols:
                vals.append(np.array([[float(r[c]) for c in cols] for r in rows]))
        if vals:
            A[st] = np.vstack(vals)
    return A


def fig_gate(diag_dir, out):
    plt = _plt(); systems = [s for s in ("B12", "PPA-FINAL", "PPB-FINAL") if gate_stats(diag_dir, s)]
    if not systems:
        raise RuntimeError("no gate diagnostics")
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 3.0), squeeze=False)
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.0))
    names = ["XLM-R", "mE5", "LaBSE"]; cols = ["#B5523B", "#2F6DB5", "#2E8B57"]
    for ax, s in zip(axes[0], systems):
        A = gate_stats(diag_dir, s); sts = [st for st in CHAIN if st in A]
        means = np.array([A[st].mean(0) for st in sts]); bottom = np.zeros(len(sts))
        for m in range(means.shape[1]):
            ax.bar(sts, means[:, m], bottom=bottom, color=cols[m % 3], label=names[m] if m < 3 else f"enc{m}"); bottom += means[:, m]
        ax.set_title(STYLE.get(s, (None, None, s))[2], fontsize=8); ax.set_ylim(0, 1); ax.set_ylabel("mean gate weight")
        ent = [float((-(A[st] * np.log(A[st] + 1e-12)).sum(1)).mean()) for st in sts]
        ax2.plot(sts, ent, marker="o", label=STYLE.get(s, (None, None, s))[2], color=STYLE.get(s, ("black",))[0])
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax2.axhline(np.log(3), color="#BBBBBB", ls="--", lw=0.8); ax2.set_ylabel("mean gate entropy (nats)"); ax2.set_ylim(bottom=0)
    ax2.set_title("Gate entropy per stage (max = ln 3)"); ax2.legend(frameon=False, fontsize=7)
    _save(fig, out, "fig_gate_distribution"); _save(fig2, out, "fig_gate_entropy")


def fig_calibration(T, out):
    plt = _plt(); fig, ax = plt.subplots(figsize=(6.0, 3.0)); w = 0.2; k = 0
    for s in STYLE:
        if not all(T.get((s, st, "ece_var")) for st in CHAIN[3:]):
            continue
        ax.bar(np.arange(4) + (k - 1.5) * w, [100 * np.mean(T[(s, st, "ece_var")]) for st in CHAIN[3:]], w, color=STYLE[s][0], label=STYLE[s][2]); k += 1
    ax.set_xticks(range(4)); ax.set_xticklabels(CHAIN[3:]); ax.set_ylabel("ECE, variety-wise T (%)"); ax.set_title("Calibration on the human stages (lower is better)")
    ax.legend(frameon=False, fontsize=7); _save(fig, out, "fig_calibration_ece")


def fig_robust(results_dir, out):
    plt = _plt(); R = {}
    for r in csv.DictReader(open(os.path.join(results_dir, "metrics", "seed_results.csv"))):
        if r["set"].startswith("rob-") and "-s" not in r["set"][4:] and r["metric"] == "acc":
            R.setdefault((r["system"], r["set"]), []).append(float(r["value"]))
    if not R:
        raise RuntimeError("no robustness predictions")
    sets = sorted({k[1] for k in R}); fig, ax = plt.subplots(figsize=(6.4, 3.0)); w = 0.2; k = 0
    for s in STYLE:
        if not all((s, x) in R for x in sets):
            continue
        ax.bar(np.arange(len(sets)) + (k - 1.5) * w, [100 * np.mean(R[(s, x)]) for x in sets], w, color=STYLE[s][0], label=STYLE[s][2]); k += 1
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([x[4:] for x in sets]); ax.set_ylabel("accuracy (%)")
    ax.set_title("Robustness to synthetic perturbations (severity 2)"); ax.legend(frameon=False, fontsize=7); _save(fig, out, "fig_robustness")


def fig_ablations(tables_dir, out):
    plt = _plt(); rows = list(csv.DictReader(open(os.path.join(tables_dir, "ablations.csv"))))
    if not rows:
        raise RuntimeError("no ablation rows")
    labels = [f"{r['Setting']}: {r['Ablation']}" for r in rows]
    M = np.array([[float(r[f"Δ {st}"]) for st in CHAIN] for r in rows]); lim = max(1.0, np.abs(M).max())
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(rows) + 1.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(CHAIN); ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(7):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ = selected − ablated (points)"); ax.set_title("Ablations (positive = component helps)")
    _save(fig, out, "fig_ablations")


def fig_screening(pp_out, out):
    plt = _plt(); S = {}
    for p in glob.glob(os.path.join(pp_out, "dev", "*", "*.json")):
        r = json.load(open(p)); S.setdefault(r["config"], []).append(100 * r["dev"])
    if not S:
        raise RuntimeError("no screening results")
    names = sorted(S, key=lambda n: np.mean(S[n])); fig, ax = plt.subplots(figsize=(6.4, 0.25 * len(names) + 1.2))
    ax.errorbar([np.mean(S[n]) for n in names], range(len(names)), xerr=[np.std(S[n], ddof=1) if len(S[n]) > 1 else 0 for n in names],
                fmt="o", ms=3, color="#2F6DB5", capsize=2)
    for ref in ("V0", "FT0"):
        if ref in S:
            ax.axvline(np.mean(S[ref]), color="#B5523B", ls="--", lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel("dev macro-F1 (%), mean ± sd over seeds")
    ax.set_title("Screening on the dev views (dashed: references V0 / FT0)"); _save(fig, out, "fig_screening_dev")


def write_all(results_dir, tables_dir, fig_dir, diag_dir, pp_out, P):
    os.makedirs(fig_dir, exist_ok=True); fails = {}
    T = _seed_table(results_dir)
    for name, fn in (("chain", lambda: fig_chain(T, fig_dir)), ("delta", lambda: fig_delta(T, fig_dir)),
                     ("gate", lambda: fig_gate(diag_dir, fig_dir)), ("calibration", lambda: fig_calibration(T, fig_dir)),
                     ("robustness", lambda: fig_robust(results_dir, fig_dir)), ("ablations", lambda: fig_ablations(tables_dir, fig_dir)),
                     ("screening", lambda: fig_screening(pp_out, fig_dir))):
        try:
            fn()
        except Exception as ex:
            fails[name] = f"{type(ex).__name__}: {ex}"
            open(os.path.join(fig_dir, f"FAILED_{name}.txt"), "w").write(traceback.format_exc())
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    return fails
