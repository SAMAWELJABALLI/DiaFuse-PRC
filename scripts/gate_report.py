#!/usr/bin/env python3
"""Gate behaviour on the dev views (no test data): per config and seed, at the selected (best) epoch.
Reports mean alpha per encoder (XLM-R, mE5, LaBSE), mean gate entropy (max = ln 3 = 1.099), share of
inputs whose argmax is each encoder, the L1 difference of mean routing between the Arabic and the Latin
dev view (does routing depend on the script?), and the epoch at which entropy first falls below 0.05.

    python scripts/gate_report.py --dev runs_pp/dev --out runs_pp/gate_report.md
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", default="runs_pp/dev"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = ["| config | seed | dev | best ep. | α (val_ar) | α (val_gs) | H val_ar | H val_gs | argmax share val_gs | routing Δ AR/LAT | collapse epoch |",
         "|" + "---|" * 11]
    for p in sorted(glob.glob(os.path.join(a.dev, "*", "*.json"))):
        r = json.load(open(p))
        if "epochs" in r and r["epochs"] and "diag" in r["epochs"][0]:
            ep = r.get("best_epoch", len(r["epochs"]) - 1); d = r["epochs"][ep]["diag"]
            col = next((e["epoch"] for e in r["epochs"] if "val_gs" in e["diag"] and e["diag"]["val_gs"]["entropy_mean"] < 0.05), None)
        elif "epochs" in r and r["epochs"] and "gate" in r["epochs"][0]:  # Setting B log
            ep = max(range(len(r["epochs"])), key=lambda i: r["epochs"][i]["dev"]); d = r["epochs"][ep]["gate"]
            col = next((e["epoch"] for e in r["epochs"] if "val_gs" in e["gate"] and e["gate"]["val_gs"]["entropy_mean"] < 0.05), None)
        elif "post_fit_diag" in r:
            ep, d, col = "-", r["post_fit_diag"], "n/a"
        else:
            continue
        f = lambda v: "/".join(f"{x:.2f}" for x in v)
        ar, gs = d.get("val_ar", {}), d.get("val_gs", {})
        L.append(f"| {r['config']} | {r['seed']} | {100 * r['dev']:.2f} | {ep} | {f(ar.get('alpha_mean', []))} | {f(gs.get('alpha_mean', []))} | "
                 f"{ar.get('entropy_mean', float('nan')):.3f} | {gs.get('entropy_mean', float('nan')):.3f} | {f(gs.get('argmax_share', []))} | "
                 f"{d.get('alpha_l1_between_dev_views', float('nan')):.3f} | {col} |")
    txt = "\n".join(L); print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
