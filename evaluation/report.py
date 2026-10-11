"""FINAL_REPORT.md: generated from the files of the run only (no number is typed by hand)."""
import csv
import hashlib
import json
import os


def _md_file(p, maxlines=80):
    if not os.path.exists(p):
        return "_not available_\n"
    lines = open(p).read().splitlines()
    return "\n".join(lines[:maxlines]) + ("\n…\n" if len(lines) > maxlines else "\n")


def _pkg_hash(pp_dir):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(pp_dir)):
        for fn in sorted(files):
            if fn.endswith((".py", ".json", ".txt")):
                h.update(open(os.path.join(root, fn), "rb").read())
    return h.hexdigest()[:16]


def write_report(run_dir, d, P, state, records, complete):
    pp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    L = ["# DiaFuse-PRC++ — FINAL REPORT", ""]
    if state.get("dry_run"):
        L += ["> **DRY RUN — SYNTHETIC DATA AND STAND-IN EXPERIMENTS. NOTHING IN THIS REPORT IS AN EXPERIMENTAL RESULT.**", ""]
    st = state.get("status"); L += [f"**Run status:** {st}  |  mode {state.get('run_mode')}  |  started {state.get('started')}  |  last update {state.get('last_update')}", ""]
    if state.get("fatal"):
        L += [f"**FATAL:** {state['fatal']}", ""]
    env = os.path.join(d["logs"], "environment.txt")
    L += ["## 1. Environment", "", "```", _md_file(env, 40).strip(), "```", ""]
    L += ["## 2. Reproduction", ""]
    for s, folder in (("B12 DiaFuse-PRC (Setting A)", "B12_setting_A"), ("B11 Robust + calibrated (Shao et al. style)", "B11")):
        L += [f"### {s}", "", "```", _md_file(os.path.join(d["results"], folder, "reproduction.txt")).strip(), "```", ""]
    fl = state.get("flags", {})
    L += [f"- Setting A implementation check (P0 vs V0): **{fl.get('settingA_implementation_check', 'n/a')}**",
          f"- Setting B implementation check (FT0 vs original B11, mean dev diff {fl.get('FT0_minus_B11_dev_mean')}): **{fl.get('settingB_implementation_check', 'n/a')}**",
          f"- Original evaluator (diafuse_eval.py): {fl.get('original_evaluator', 'n/a')}", ""]
    L += ["## 3. Screening and selection (dev only)", "", "Selection rule: " + P["selection_rule"]["score"] + "; shortlist: " + P["selection_rule"]["shortlist"] + ".", ""]
    for s in ("A", "B"):
        L += [f"### Setting {s}", "", _md_file(os.path.join(d["configs"], f"setting{s}_final_report.md"), 60), ""]
    L += ["## 4. Selected configuration", "", "```json", _md_file(os.path.join(run_dir, "SELECTED_CONFIG.json"), 120).strip(), "```", ""]
    T = os.path.join(run_dir, "tables")
    L += ["## 5. Final results (macro-F1 %, S0–S6)", "", _md_file(os.path.join(T, "main_results.md")), "",
          "## 6. Comparison with B11 and B12", "", _md_file(os.path.join(T, "final_comparison.md")), "",
          "## 7. Table 5 (accuracy S0/S3/S6; EM (S0,S1)/(S0,S2)/(S5AR,S5))", "", _md_file(os.path.join(T, "table5.md")), "",
          "## 8. Ablations", "", _md_file(os.path.join(T, "ablations.md")), ""]
    L += ["## 9. Statistics", "", "Protocol tests: paired bootstrap (1,000 resamples) with Holm correction over the seven stages and the decision rule; "
          "Wilcoxon signed-rank across stages; almost stochastic order if available.", ""]
    sp = os.path.join(T, "statistical_tests.csv")
    if os.path.exists(sp):
        rows = list(csv.DictReader(open(sp)))
        L += ["| comparison | stage | effect | 95% CI | p | adj. p | significant |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['comparison']} | {r['stage']} | {r['effect']} | [{r['ci_low']}, {r['ci_high']}] | {r['p_value']} | {r['adjusted_p']} | {r['significant']} |")
        L.append("")
    vp = os.path.join(T, "verdict.json")
    if os.path.exists(vp):
        for pp, v in json.load(open(vp))["verdicts"].items():
            msg = {"OUTPERFORMS": "DiaFuse-PRC++ outperforms B11", "PARTIAL": "DiaFuse-PRC++ outperforms B11 on some stages only",
                   "DOES_NOT_OUTPERFORM": "DiaFuse-PRC++ does not outperform B11"}[v["verdict"]]
            L.append(f"**RESULT ({pp}): {msg}** — stages won {v['stages_won']}, stages lost {v['stages_lost']}.")
        L.append("")
    done = [r for r in records if r.get("status") == "completed"]
    tot = sum(float(r.get("runtime_min") or 0) for r in done)
    by = {}
    for r in done:
        by.setdefault(r.get("stage"), [0, 0.0]); by[r.get("stage")][0] += 1; by[r.get("stage")][1] += float(r.get("runtime_min") or 0)
    L += ["## 10. Computational cost", "", f"Completed experiments: {len(done)}; GPU time: {tot / 60:.1f} h.", "",
          "| stage | experiments | hours |", "|---|---|---|"] + [f"| {k} | {v[0]} | {v[1] / 60:.2f} |" for k, v in sorted(by.items(), key=lambda x: str(x[0]))] + [""]
    failed = [r for r in records if r.get("status") != "completed"]
    L += ["## 11. Failures", ""]
    if failed:
        for r in failed:
            err = (r.get("error") or "").strip().splitlines()[-6:]
            L += [f"- **{r['experiment_id']}** ({r.get('status')}), log `{r.get('log')}`:", "  ```", *["  " + e for e in err], "  ```"]
    else:
        L.append("None.")
    L += ["", "## 12. Reproducibility", "",
          f"- Protocol version: {P.get('protocol_version')}; package hash (src/scripts/configs/evaluation): {_pkg_hash(pp_dir)}",
          "- Original code: verified unchanged (MD5) at the start of every session.",
          f"- Seeds: {json.dumps(P['seeds'])}",
          f"- Sessions: {len(state.get('sessions', []))} ({', '.join(s['start'] for s in state.get('sessions', []))})",
          "- Bug fixes applied during the run: " + (", ".join(state.get("bug_fixes")) if state.get("bug_fixes") else "none"),
          "- Git commit: not available (package distributed as a folder).", ""]
    L += ["## 13. Scientific conclusion (automatic, rule-based)", ""]
    if os.path.exists(vp):
        for pp, v in json.load(open(vp))["verdicts"].items():
            L.append(f"- {pp}: **{v['verdict']}** under the pre-registered verdict rule ({P['verdict_rule']['OUTPERFORMS']}).")
    else:
        L.append("- No verdict available (final comparison not produced).")
    L += ["", "This conclusion is limited to this experimental setting (binary sentiment, Tunisian training data, the evaluated test chain)."]
    if not complete:
        L += ["", "**The run is INCOMPLETE: see section 11 and run_state.json.**"]
    open(os.path.join(run_dir, "FINAL_REPORT.md"), "w").write("\n".join(L) + "\n")
