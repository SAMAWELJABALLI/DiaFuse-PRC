#!/usr/bin/env python3
"""Pre-registered model-selection rule (dev only; never reads a test file).

Score(config, seed) = mean macro-F1 over the dev views (val_ar, val_gs) at the early-stopping epoch
                      -- the selection criterion of the paper's protocol, also used for B11 and all baselines.
Score(config)       = mean over seeds (>= 3 required); noise(config, ref) = max(seed sd of config, of reference).

  screen : shortlist = configs with Score - Score(reference) > noise; family winner = best shortlisted config of
           each family; writes <out_prefix>_combine.json with ALL (union of family winners) and ALL-minus-<family>.
  final  : candidates = family winners + combine configs (>= 3 seeds, all shortlisted vs reference);
           FINAL = highest Score; ties (< 0.2 macro-F1 points) -> fewer trainable parameters.
           If no candidate beats the reference, FINAL = reference and this is reported as a negative result.
           Writes <out_prefix>_selected.json with FINAL, REF and the pre-declared test-time ablations.

    python scripts/select_config.py screen --configs configs/B12_setting_A/screening_configs.json --dev runs_pp/dev --out_prefix configs/settingA
    python scripts/select_config.py final  --configs configs/B12_setting_A/screening_configs.json --dev runs_pp/dev --out_prefix configs/settingA
"""
import argparse
import glob
import json
import os
import statistics as st

TIE = 0.002  # 0.2 macro-F1 points (scores are on the 0-1 scale)


def raw_overrides(spec, name):
    c = spec["configs"][name]; out = {}
    for parent in c.get("inherit", []):
        out.update(raw_overrides(spec, parent))
    out.update({k: v for k, v in c.items() if k not in ("inherit", "family", "why")})
    return out


def scores(dev_dir, name):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(dev_dir, name, "*.json")))]
    return recs


def summarise(dev_dir, names):
    out = {}
    for n in names:
        r = scores(dev_dir, n)
        if not r:
            continue
        v = [x["dev"] for x in r]
        out[n] = {"n": len(v), "mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
                  "params": r[0].get("trainable_params"), "seeds": [x["seed"] for x in r], "family": r[0].get("family", "")}
    return out


def table(S, ref):
    lines = ["| config | family | seeds | dev mean | sd | Δ vs ref (pts) | noise (pts) | shortlisted |", "|---|---|---|---|---|---|---|---|"]
    for n, s in sorted(S.items(), key=lambda kv: -kv[1]["mean"]):
        d = s["mean"] - S[ref]["mean"]; noise = max(s["sd"], S[ref]["sd"])
        ok = n != ref and s["n"] >= 3 and d > noise
        lines.append(f"| {n} | {s['family']} | {s['n']} | {100 * s['mean']:.2f} | {100 * s['sd']:.2f} | {100 * d:+.2f} | {100 * noise:.2f} | {'yes' if ok else 'no'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["screen", "final"])
    ap.add_argument("--configs", required=True); ap.add_argument("--dev", default="runs_pp/dev")
    ap.add_argument("--out_prefix", required=True); ap.add_argument("--setting", choices=["A", "B"], default="A")
    a = ap.parse_args()
    spec = json.load(open(a.configs))
    ref = [n for n, c in spec["configs"].items() if c.get("family") == "reference"][0]
    order = spec.get("family_order", [])
    names = list(spec["configs"])
    S = summarise(a.dev, names)
    if ref not in S or S[ref]["n"] < 3:
        raise SystemExit(f"reference {ref} needs >= 3 seeds of dev results")
    short = {n for n, s in S.items() if n != ref and s["n"] >= 3 and s["mean"] - S[ref]["mean"] > max(s["sd"], S[ref]["sd"])}
    winners = {}
    for fam in order:
        cand = [n for n in short if spec["configs"][n].get("family") == fam]
        if cand:
            winners[fam] = max(cand, key=lambda n: S[n]["mean"])
    report = [f"# Selection ({a.phase}) — reference = {ref}", "", table(S, ref), "",
              "Family winners: " + (", ".join(f"{f}={w}" for f, w in winners.items()) or "none")]
    if a.phase == "screen":
        combos = {}
        if len(winners) >= 2:
            fams = [f for f in order if f in winners]
            def merged(fs):
                o = {}
                for f in fs:
                    o.update(raw_overrides(spec, winners[f]))
                return o
            combos["ALL"] = {"family": "combine", **merged(fams), "why": "union of family winners"}
            for f in fams:
                combos[f"ALL-minus-{f}"] = {"family": "combine", **merged([g for g in fams if g != f]), "why": f"ALL without the {f} winner"}
        out = {"family_order": order, "configs": {ref: spec["configs"][ref], **combos}}
        json.dump(out, open(a.out_prefix + "_combine.json", "w"), indent=1)
        report.append(f"\nWrote {a.out_prefix}_combine.json with {len(combos)} combination configs.")
    else:
        comb_path = a.out_prefix + "_combine.json"
        comb = json.load(open(comb_path)) if os.path.exists(comb_path) else {"configs": {}}
        Sc = summarise(a.dev, [n for n in comb["configs"] if n != ref])
        allS = {**{w: S[w] for w in winners.values()}, **Sc}
        cands = {n: s for n, s in allS.items() if s["n"] >= 3 and s["mean"] - S[ref]["mean"] > max(s["sd"], S[ref]["sd"])}
        if cands:
            top = max(s["mean"] for s in cands.values())
            tied = [n for n, s in cands.items() if top - s["mean"] < TIE]
            final = min(tied, key=lambda n: (cands[n]["params"] or 0, -cands[n]["mean"]))
            final_over = raw_overrides(comb, final) if final in comb["configs"] else raw_overrides(spec, final)
            verdict = f"FINAL = {final} (dev {100 * cands[final]['mean']:.2f}, ref {100 * S[ref]['mean']:.2f})"
        else:
            final, final_over = ref, raw_overrides(spec, ref)
            verdict = "No configuration beats the reference on dev by more than the seed noise: FINAL = reference (negative result)."
        sel = {"family_order": order, "selected_from": final, "configs": {"FINAL": {"family": "final", **final_over},
                                                                           "REF": {"family": "reference", **raw_overrides(spec, ref)}}}
        if final != ref and a.setting == "A":  # pre-declared component ablations of the selected model
            fo = sel["configs"]["FINAL"]; abl = sel["configs"]
            ab = lambda **kw: {**fo, **kw, "family": "ablation"}
            abl["FINAL-minus-cons"] = ab(cons=False)
            abl["FINAL-minus-cal"] = ab(cal=False)
            abl["FINAL-minus-hm"] = ab(hm=False)
            if fo.get("proto", "hier") != "none":
                abl["FINAL-minus-proto"] = ab(proto="none")
            fusion = fo.get("fusion", "gate")
            if fusion != "static":
                abl["FINAL-static"] = ab(fusion="static")
            abl["FINAL-single"] = ab(encoders=["me5"])  # best single encoder of the archived dev scores (pre-declared)
            gate_keys = {"gate_tau": 1.0, "gate_tau_learn": False, "gate_zero_init": False, "gate_floor": 0.0,
                         "gate_noise": 0.0, "enc_dropout": 0.0, "lam_lb": 0.01, "lam_ent": 0.0}
            if fusion in ("gate", "gated_residual", "attn"):
                if any(k in fo for k in gate_keys):
                    abl["FINAL-origgate"] = ab(**gate_keys)
                if fusion != "attn":
                    abl["FINAL-nosurface"] = ab(gate_input="e")
        if final != ref and a.setting == "B":
            fo = sel["configs"]["FINAL"]; abl = sel["configs"]
            ab = lambda **kw: {**fo, **kw, "family": "ablation"}
            if fo.get("fusion", "gate") in ("gate", "gated_residual"):
                abl["FINAL-static"] = ab(fusion="static")
            if fo.get("lam_js", 0) > 0:
                abl["FINAL-nojs"] = ab(lam_js=0.0)
            if fo.get("lam_p", 0) > 0:
                abl["FINAL-minus-proto"] = ab(lam_p=0.0)
            abl["FINAL-single"] = ab(encoders=["me5"])
            abl["FINAL-frozen"] = ab(adapt="frozen")
        if final in comb["configs"] and final == "ALL":
            for n in comb["configs"]:
                if n.startswith("ALL-minus-"):
                    sel["configs"]["FINAL-" + n[4:]] = {**raw_overrides(comb, n), "family": "ablation"}
        elif final in winners.values() and final != ref:
            pass  # single-family winner: its ablation is REF (the reference), pre-declared
        report += ["", "## Combination candidates", table({**{ref: S[ref]}, **Sc}, ref) if Sc else "(none)", "", verdict]
        json.dump(sel, open(a.out_prefix + "_selected.json", "w"), indent=1)
        report.append(f"Wrote {a.out_prefix}_selected.json (FINAL, REF and pre-declared ablations).")
    txt = "\n".join(report); print(txt)
    open(a.out_prefix + f"_{a.phase}_report.md", "w").write(txt)


if __name__ == "__main__":
    main()
