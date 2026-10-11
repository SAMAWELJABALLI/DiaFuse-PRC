#!/usr/bin/env bash
# Dry-run test of the orchestrator (no GPU, no torch): synthetic data, stand-in experiments.
# Checks: paths, imports, registry, simulated disconnection + resume, failure handling, selection,
# CSV/LaTeX/figures generation and final archives. NOTHING produced here is an experimental result.
set -u
PKG=$(cd "$(dirname "$0")/.." && pwd); PP="$PKG"
T=${1:-$(mktemp -d)}; RUN=$T/RUN; WORK=$T/work; ARCH=$T/synthetic_archive
python3 "$PP/tests/make_synthetic.py" --data "$RUN/data" --archive "$ARCH" >/dev/null
O="python3 $PP/scripts/run_one_click.py --run_dir $RUN --work_dir $WORK --dry_run --archive_dir $ARCH --mode ${MODE:-FULL}"
echo "== session 1: simulated disconnection after 12 experiments"; PP_STOP_AFTER_EXPERIMENTS=12 $O > $T/s1.log 2>&1; echo "exit=$? (expected 4)"
echo "== session 2: two experiments fail persistently"; DRYRUN_FAIL="A-search:V3b::1,A-final:FINAL:PPA-FINAL:7" $O > $T/s2.log 2>&1; echo "exit=$? (expected 2 = incomplete)"
echo "== session 3: resume, failures recovered"; $O > $T/s3.log 2>&1; echo "exit=$? (expected 0 = completed)"
echo "== assertions on the final state"
python3 - "$RUN" <<'PY'
import csv, glob, json, os, sys
R = sys.argv[1]; err = []
reg = list(csv.DictReader(open(os.path.join(R, "experiment_registry.csv"))))
st = {}
for r in reg:
    st.setdefault(r["stage"], []).append(r["status"])
if any(x != "completed" for r in reg for x in [r["status"]]): err.append("non-completed experiments in registry")
sel = json.load(open(os.path.join(R, "SELECTED_CONFIG.json")))["settings"]
for k, v in sel.items():
    if v["status"] == "FAILED": err.append(f"selection {k} FAILED")
fin = [r["experiment_id"] for r in reg if r["stage"] == "final"]
for sysn in ("B12", "PPA-FINAL", "PPB-FINAL"):
    n = sum(f"-{sysn}-s" in e for e in fin)
    want = 10 if sysn == "B12" or sel.get("AB"[sysn == "PPB-FINAL"] if sysn != "B12" else "A", {}).get("status") == "selected" else 0
    if sysn != "B12" and sel["AB"[sysn == "PPB-FINAL"]]["status"] == "selected" and n != 10: err.append(f"{sysn}: {n} final seeds")
    if sysn == "B12" and n != 10: err.append(f"B12: {n} final seeds")
if sum(e.startswith("B11-s") for e in fin) != 7: err.append("B11 final seeds 3-9 missing")
if any(v["status"] == "selected" for v in sel.values()) and not st.get("ablations"): err.append("no ablations")
fails = glob.glob(os.path.join(R, "figures", "FAILED_*"))
if fails: err.append("figures failed: " + ", ".join(os.path.basename(f) for f in fails))
if not glob.glob(os.path.join(R, "final_archive", "DiaFuse_PRCPP_FINAL_RESULTS_*.zip")): err.append("no results zip")
if glob.glob(os.path.join(R, "**", "*.tmp"), recursive=True): err.append("leftover .tmp files")
print({k: len(v) for k, v in st.items()}, {k: v["status"] for k, v in sel.items()})
print("ASSERTIONS:", "PASS" if not err else "FAIL " + "; ".join(err)); sys.exit(1 if err else 0)
PY
echo "assertions exit=$? (expected 0)"
echo "T=$T"
