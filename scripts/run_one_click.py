#!/usr/bin/env python3
"""DiaFuse-PRC++ one-click orchestrator (resumable).

Runs every stage of the pre-registered protocol (configs/protocol.json) by calling the scientific
scripts as separate processes:
  original/code/*.py           unmodified original pipeline (B11 = finetune.py, data, features, evaluation)
  src/diafuse_pp_setting_A.py        DiaFuse-PRC++ Setting A (frozen encoders)
  src/diafuse_pp_setting_B_lora.py   DiaFuse-PRC++ Setting B (LoRA / fine-tuned encoders)

Stop levels
  FATAL              data / split / model / reproduction problem  -> the whole run stops (exit code 3)
  EXPERIMENT FAILED  one (configuration, seed) crashes            -> recorded with traceback, run continues
  SUCCESS            -> continue

Resume: every experiment has a record in <run_dir>/state/experiments/<id>.json and a completion marker
written by the script itself; completed experiments are never re-run, interrupted ones are cleaned and re-run.

    python run_one_click.py --run_dir /content/drive/MyDrive/DiaFuse_PRCPP_RUN --work_dir /content/pp_work --mode FULL
Exit codes: 0 = RUN COMPLETED, 2 = RUN INCOMPLETE (some experiments failed), 3 = FATAL.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)                        # repository root
PKG = PP
CFG = {"protocol": os.path.join(PP, "configs", "protocol.json"),
       "defaults": {"A": os.path.join(PP, "configs", "B12_setting_A", "defaults.json"),
                    "B": os.path.join(PP, "configs", "B12_setting_B_lora", "defaults.json")},
       "screening": {"A": os.path.join(PP, "configs", "B12_setting_A", "screening_configs.json"),
                     "B": os.path.join(PP, "configs", "B12_setting_B_lora", "screening_configs.json")}}
SRC = {"A": os.path.join(PP, "src", "diafuse_pp_setting_A.py"), "B": os.path.join(PP, "src", "diafuse_pp_setting_B_lora.py")}
ORIG = os.path.join(PKG, "original", "code")
sys.path.insert(0, os.path.join(PP, "evaluation"))

# One results folder per model, never shared (labels of the manuscript tables where they exist).
RESULT_FOLDERS = {"B11": "B11", "B12": "B12_setting_A", "PPA-FINAL": "B12pp_setting_A", "PPB-FINAL": "B12_setting_B_lora"}
MODEL_CARDS = {
    "B11": {"label": "B11 Robust + calibrated (Shao et al. style)", "model_card": "models/B11_robust_calibrated/README.md",
            "code": ["original/code/finetune.py"], "config": "hard-coded in original/code/finetune.py (invoked as in original/code/run_all.sh)"},
    "B12": {"label": "B12 DiaFuse-PRC (Setting A) - original trainer, re-run", "model_card": "models/B12_diafuse_setting_A/README.md",
            "code": ["original/code/diafuse_run.py", "original/code/extract.py", "original/code/translit.py", "original/code/prep_data.py"],
            "config": "configs/B12_setting_A/screening_configs.json#V0 (impl = original diafuse_run.DiaFuseTrainer)"},
    "PPA-FINAL": {"label": "B12++ DiaFuse-PRC++ (Setting A)", "model_card": "models/B12pp_diafuse_pp_setting_A/README.md",
                  "code": ["src/diafuse_pp_setting_A.py", "original/code/diafuse_run.py", "original/code/extract.py"],
                  "config": "configs/settingA_selected.json#FINAL (run directory)"},
    "PPB-FINAL": {"label": "B12'++ DiaFuse-PRC++ (Setting B, LoRA) - NOT the manuscript's B12' specification",
                  "model_card": "models/B12_prime_diafuse_setting_B_lora/README.md",
                  "code": ["src/diafuse_pp_setting_B_lora.py", "original/code/translit.py"],
                  "config": "configs/settingB_selected.json#FINAL (run directory)"},
}
ABL_FOLDER = {"FINAL-origgate": "original_gate", "FINAL-static": "no_gate_static_fusion",
              "FINAL-single": "no_fusion_single_encoder", "FINAL-nosurface": "no_surface_descriptor",
              "FINAL-minus-proto": "no_prototype", "FINAL-minus-hm": "no_hard_mixup",
              "FINAL-minus-cons": "no_consistency", "FINAL-minus-cal": "no_calibration",
              "FINAL-nojs": "no_js_consistency", "FINAL-frozen": "no_finetuning_frozen_encoders"}


class Fatal(Exception):
    pass


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def atomic_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"  # unique per thread: parallel workers never share a temp file
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_trace_card(C, system, out_dir):
    """MODEL -> CODE VERSION -> CONFIG -> DATA/SPLIT -> SEEDS -> CHECKPOINTS -> RESULTS, for one exported system."""
    if not os.path.isdir(out_dir):
        return
    card = MODEL_CARDS.get(system, {"label": system, "code": [], "config": "see experiment_registry.csv"})
    recs = sorted([r for r in C.records() if r.get("system") == system and r.get("status") == "completed"], key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in recs]
    ck = sorted(glob.glob(os.path.join(C.d["checkpoints"], system, "*.pt")))
    split = os.path.join(C.d["logs"], "split_manifest.json")
    trace = {"system": system, "label": card["label"], "model_card": card.get("model_card"),
             "code_version": C.state.get("code_version") or code_version(),
             "code_files": {f: sha256_file(os.path.join(PP, f)) for f in card["code"] if os.path.exists(os.path.join(PP, f))},
             "config": card["config"], "resolved_configs": {r["experiment_id"]: r.get("config") for r in recs},
             "data_split_manifest": {"file": os.path.relpath(split, C.run), "sha256": sha256_file(split) if os.path.exists(split) else None},
             "seeds": seeds, "checkpoints": [os.path.relpath(c, C.run) for c in ck],
             "predictions": os.path.relpath(os.path.join(out_dir, "predictions"), C.run),
             "metrics": os.path.relpath(os.path.join(out_dir, "seed_metrics.csv"), C.run),
             "experiment_logs": [r.get("log") for r in recs], "dry_run": C.dry}
    atomic_json(trace, os.path.join(out_dir, "trace.json"))
    L = [f"# {card['label']}", "", "Generated automatically by scripts/run_one_click.py from the files of this run.", "",
         "| link | value |", "|---|---|",
         f"| code version | `{trace['code_version'].get('git_commit')}` |",
         f"| code files | {', '.join('`' + f + '`' for f in trace['code_files'])} |",
         f"| configuration | {card['config']} (resolved per seed in `trace.json`) |",
         f"| data / split | `{trace['data_split_manifest']['file']}` (sha256 {str(trace['data_split_manifest']['sha256'])[:16]}) |",
         f"| seeds | {seeds} |", f"| checkpoints | {len(ck)} file(s) under `checkpoints/{system}/` |",
         f"| predictions | `predictions/<seed>/<set>.tsv` |", f"| metrics | `seed_metrics.csv` (one row per seed, stage, metric) |", ""]
    if C.dry:
        L.insert(2, "**DRY RUN - synthetic data, nothing here is an experimental result.**\n")
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(L) + "\n")


def code_version():
    """Git commit of the repository checkout that is running (recorded in run_state.json and in every result card)."""
    try:
        h = subprocess.run(["git", "-C", PP, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if h.returncode != 0:
            return {"git_commit": "NOT AVAILABLE (not a git checkout)"}
        dirty = subprocess.run(["git", "-C", PP, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=30).stdout.strip()
        return {"git_commit": h.stdout.strip(), "uncommitted_changes": bool(dirty)}
    except Exception as ex:
        return {"git_commit": f"NOT AVAILABLE ({type(ex).__name__})"}


# ===================================================================== context
class Ctx:
    def __init__(self, a):
        self.a = a; self.mode = a.mode; self.dry = a.dry_run
        self.run = a.run_dir; self.work = a.work_dir
        self.P = json.load(open(CFG["protocol"]))
        self.arch = a.archive_dir or os.path.join(PKG, "original", "archived")
        d = {k: os.path.join(self.run, k) for k in ("logs", "checkpoints", "results", "tables", "figures", "configs",
                                                     "final_archive", "state", "data")}
        d["exp_logs"] = os.path.join(self.run, "logs", "experiments")
        d["exp_state"] = os.path.join(self.run, "state", "experiments")
        d["runs"] = os.path.join(self.run, "results", "runs")          # preds/ diag/ emb/ dev_scores.json (eval format)
        d["pp_out"] = os.path.join(self.run, "results", "pp")          # dev/<cfg>/<seed>.json, final/<sys>/<seed>.json
        d["models"] = a.models_dir or os.path.join(self.run, "cache", "models")
        d["raw"] = os.path.join(self.run, "cache", "raw")
        d["feats"] = os.path.join(self.work, "feats")
        for v in d.values():
            os.makedirs(v, exist_ok=True)
        self.d = d
        self.lock = threading.RLock()
        self.state_path = os.path.join(self.run, "run_state.json")
        self.state = json.load(open(self.state_path)) if (a.resume and os.path.exists(self.state_path)) else {}
        self.state.setdefault("started", now()); self.state.setdefault("sessions", [])
        self.state["sessions"].append({"start": now(), "mode": self.mode, "dry_run": self.dry})
        self.state.update({"run_mode": self.mode, "status": "running", "dry_run": self.dry, "protocol_version": self.P["protocol_version"]})
        self.state.setdefault("stages_done", []); self.state.setdefault("flags", {}); self.state.setdefault("bug_fixes", [])
        self.save_state()
        self.env = dict(os.environ, DIAFUSE_CODE=ORIG, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                        HF_HOME=os.path.join(self.run, "cache", "hf_home"))
        self.stop_after = int(os.environ.get("PP_STOP_AFTER_EXPERIMENTS", "0"))  # test hook (simulated disconnection)
        self.n_run_this_session = 0

    # ------------------------------------------------------------ logging / state
    def log(self, msg, level="INFO"):
        line = f"[{now()}] {level:7s} {msg}"
        print(line, flush=True)
        with self.lock, open(os.path.join(self.d["logs"], "orchestrator.log"), "a") as f:
            f.write(line + "\n")

    def save_state(self, **kw):
        with self.lock if hasattr(self, "lock") else _nolock():
            self.state.update(kw); self.state["last_update"] = now()
            recs = self.records()
            self.state["completed_experiments"] = sum(r["status"] == "completed" for r in recs)
            self.state["failed_experiments"] = sum(r["status"] == "failed" for r in recs)
            self.state["running_experiments"] = [r["experiment_id"] for r in recs if r["status"] == "running"]
            self.state["failed_list"] = [r["experiment_id"] for r in recs if r["status"] == "failed"]
            atomic_json(self.state, self.state_path)

    def records(self):
        out = []
        for p in sorted(glob.glob(os.path.join(getattr(self, "d", {}).get("exp_state", "/nonexistent"), "*.json"))):
            try:
                out.append(json.load(open(p)))
            except Exception:
                pass
        return out

    def stage_done(self, s):
        return s in self.state["stages_done"]

    def mark_stage(self, s):
        if s not in self.state["stages_done"]:
            self.state["stages_done"].append(s)
        self.save_state(current_stage=s)


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ===================================================================== registry
REG_COLS = ["experiment_id", "stage", "setting", "variant", "architecture", "seed", "encoder", "trainable_params", "lr",
            "epochs", "batch_size", "loss_weights", "gate_temperature", "gate_entropy_weight", "prototype_weight",
            "consistency_weight", "calibration_weight", "dev_macro_f1", "test_macro_f1_mean_S0_S6", "status", "checkpoint",
            "start_time", "end_time", "runtime_min", "gpu", "attempts", "log", "error"]


def write_registry(C):
    rows = []
    for r in C.records():
        c = r.get("config") or {}
        lw = {k: c[k] for k in ("lam_cl", "lam_p", "lam_cons", "lam_js", "lam_hm", "lam_cal", "lam_lb", "lam_ent") if k in c}
        rows.append({"experiment_id": r["experiment_id"], "stage": r.get("stage"), "setting": r.get("setting"),
                     "variant": r.get("variant"), "architecture": r.get("architecture") or c.get("fusion", ""),
                     "seed": r.get("seed"), "encoder": ",".join(c.get("encoders") or []) if c.get("encoders") else C.P["encoders"],
                     "trainable_params": r.get("trainable_params"), "lr": c.get("lr", c.get("lr_lora", c.get("lr_enc"))),
                     "epochs": c.get("epochs"), "batch_size": c.get("batch"), "loss_weights": json.dumps(lw),
                     "gate_temperature": c.get("gate_tau"), "gate_entropy_weight": c.get("lam_ent"),
                     "prototype_weight": c.get("lam_p"), "consistency_weight": c.get("lam_cons"),
                     "calibration_weight": c.get("lam_cal"), "dev_macro_f1": r.get("dev_macro_f1"),
                     "test_macro_f1_mean_S0_S6": r.get("test_macro_f1_mean_S0_S6"), "status": r.get("status"),
                     "checkpoint": r.get("checkpoint"), "start_time": r.get("start_time"), "end_time": r.get("end_time"),
                     "runtime_min": r.get("runtime_min"), "gpu": r.get("gpu"), "attempts": r.get("attempts"),
                     "log": r.get("log"), "error": (r.get("error") or "")[-2000:]})
    with C.lock:  # registry is rewritten by several worker threads
        for p in (os.path.join(C.run, "experiment_registry.csv"), os.path.join(C.d["results"], "experiment_registry.csv")):
            tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=REG_COLS); w.writeheader(); w.writerows(rows)
            os.replace(tmp, p)


# ===================================================================== experiments
class Exp:
    def __init__(self, eid, stage, setting, variant, seed, cmd, done_file, cleanup=(), result_json=None, config=None,
                 kind="", system=None, parse_stdout=None):
        self.id, self.stage, self.setting, self.variant, self.seed = eid, stage, setting, variant, seed
        self.cmd, self.done_file, self.cleanup, self.result_json = cmd, done_file, list(cleanup), result_json
        self.config, self.kind, self.system, self.parse_stdout = config or {}, kind, system, parse_stdout


def gpu_name(C):
    if C.dry:
        return "DRY-RUN (no GPU)"
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def run_exp(C, e):
    rp = os.path.join(C.d["exp_state"], f"{e.id}.json")
    rec = json.load(open(rp)) if os.path.exists(rp) else {}
    if os.path.exists(e.done_file) and rec.get("status") == "completed":
        return rec
    if C.stop_after and C.n_run_this_session >= C.stop_after:
        raise KeyboardInterrupt("PP_STOP_AFTER_EXPERIMENTS reached (simulated disconnection)")
    attempts = int(rec.get("attempts", 0))
    if C.a.retry_failed and rec.get("status") == "failed":
        attempts = 0  # owner-requested retry of failed experiments (attempt counters reset)
    if os.path.exists(e.done_file) and rec.get("status") != "completed":
        pass  # finished by the script but the session died before the record was updated: accept below
    else:
        for p in e.cleanup:  # partial outputs of an interrupted run
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
    prev_error = rec.get("error", "")
    rec = {"experiment_id": e.id, "stage": e.stage, "setting": e.setting, "variant": e.variant, "seed": e.seed,
           "kind": e.kind, "system": e.system, "config": e.config, "architecture": e.config.get("fusion"),
           "status": "running", "start_time": now(), "gpu": gpu_name(C), "attempts": attempts,
           "log": os.path.relpath(os.path.join(C.d["exp_logs"], f"{e.id}.log"), C.run), "cmd": " ".join(e.cmd)}
    atomic_json(rec, rp); C.save_state()
    logf = os.path.join(C.d["exp_logs"], f"{e.id}.log")
    t0 = time.time(); ok = os.path.exists(e.done_file); out_tail = prev_error
    env = dict(C.env)
    tries = 0
    while not ok and tries < 2 and attempts < 4:  # 2 attempts per session, at most 4 in total
        attempts += 1; tries += 1
        C.n_run_this_session += 1
        C.log(f"START {e.id} (attempt {attempts})")
        with open(logf, "a") as lf:
            lf.write(f"\n===== {now()} attempt {attempts}: {' '.join(e.cmd)}\n"); lf.flush()
            try:
                p = subprocess.Popen(e.cmd, cwd=C.work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    lf.write(line)
                    if re.search(r"epoch \d+: .*dev|\] .* seed \d+: dev|dev macro-F1", line):
                        print(f"    {e.id}: {line.strip()[:160]}", flush=True)
                rc = p.wait()
            except Exception as ex:  # launching failed
                lf.write(traceback.format_exc()); rc = -1
        out_tail = "".join(open(logf, errors="replace").readlines()[-60:])
        ok = rc == 0 and os.path.exists(e.done_file)
        if not ok and tries < 2 and attempts < 4:  # one immediate retry (transient Drive / CUDA errors); OOM -> expandable segments
            if re.search(r"out of memory|CUDA error: out of memory|OutOfMemoryError", out_tail):
                C.log(f"OOM in {e.id}: retrying once with expandable CUDA segments", "WARN")
                env = dict(env, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            else:
                C.log(f"{e.id} failed (attempt {attempts}): retrying once", "WARN")
            time.sleep(0 if C.dry else 20)
            for p_ in e.cleanup:
                if os.path.isdir(p_):
                    shutil.rmtree(p_, ignore_errors=True)
            continue
        break
    rec.update({"end_time": now(), "runtime_min": round((time.time() - t0) / 60, 2), "attempts": attempts})
    if ok:
        rec["status"] = "completed"
        if e.result_json and os.path.exists(e.result_json):
            r = json.load(open(e.result_json))
            rec["dev_macro_f1"] = r.get("dev"); rec["trainable_params"] = r.get("trainable_params"); rec["checkpoint"] = r.get("checkpoint")
        if e.parse_stdout:
            rec.update(e.parse_stdout(open(logf, errors="replace").read()))
        C.log(f"DONE  {e.id}  dev={rec.get('dev_macro_f1')}  ({rec['runtime_min']} min)")
    else:
        rec["status"] = "failed"; rec["error"] = out_tail or "no further attempt (attempt cap reached)"
        C.log(f"EXPERIMENT FAILED {e.id} — see {rec['log']}", "ERROR")
    atomic_json(rec, rp); C.save_state(); write_registry(C)
    return rec


def run_many(C, exps, workers=1):
    if workers <= 1 or len(exps) <= 1:
        return [run_exp(C, e) for e in exps]
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda e: run_exp(C, e), exps):
            out.append(r)
    return out


def py(C, script, *args):
    return [sys.executable, script, *[str(a) for a in args]]


def fake(C, kind, **kw):
    """Dry-run backend: a stand-in process that writes outputs with the real formats (SYNTHETIC values)."""
    args = [sys.executable, os.path.join(PP, "tests", "fake_experiment.py"), "--kind", kind]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    return args


# ----------------------------------------------------------- experiment factories
def exp_A(C, name, seed, cmd_kind, cfg_file, system=None, stage="screening"):
    """Setting A experiment: 'search' (dev only) or 'final' (writes test predictions)."""
    from_cfg = load_cfg(C, cfg_file, name, "A")
    if cmd_kind == "search":
        done = os.path.join(C.d["pp_out"], "dev", name, f"{seed}.json"); sysn = None; cleanup = []
        eid = f"A-search-{name}-s{seed}"
    else:
        sysn = system or name; done = os.path.join(C.d["pp_out"], "final", sysn, f"{seed}.json")
        cleanup = [os.path.join(C.d["runs"], "preds", sysn, str(seed)), os.path.join(C.d["runs"], "diag", sysn, str(seed))]
        eid = f"A-final-{sysn}-s{seed}"
    if C.dry:
        cmd = fake(C, f"A-{cmd_kind}", name=name, seed=seed, system=sysn or "", data=C.d["data"], runs=C.d["runs"],
                   pp_out=C.d["pp_out"], cfg=cfg_file, ckpt=C.d["checkpoints"])
    else:
        cmd = py(C, SRC["A"], cmd_kind, "--configs", cfg_file, "--only", name, "--seeds", seed,
                 "--data", C.d["data"], "--feats", C.d["feats"], "--out", C.d["runs"], "--pp_out", C.d["pp_out"],
                 "--encoders", C.P["encoders"], *(["--system", sysn, "--ckpt_dir", C.d["checkpoints"]] if cmd_kind == "final" else []))
    return Exp(eid, stage, "A", name, seed, cmd, done, cleanup, done, from_cfg, f"A-{cmd_kind}", sysn)


def exp_B(C, name, seed, cmd_kind, cfg_file, system=None, stage="screening"):
    from_cfg = load_cfg(C, cfg_file, name, "B")
    if cmd_kind == "search":
        done = os.path.join(C.d["pp_out"], "dev", name, f"{seed}.json"); sysn = None; cleanup = []
        eid = f"B-search-{name}-s{seed}"
    else:
        sysn = system or name; done = os.path.join(C.d["pp_out"], "final", sysn, f"{seed}.json")
        cleanup = [os.path.join(C.d["runs"], "preds", sysn, str(seed)), os.path.join(C.d["runs"], "diag", sysn, str(seed))]
        eid = f"B-final-{sysn}-s{seed}"
    if C.dry:
        cmd = fake(C, f"B-{cmd_kind}", name=name, seed=seed, system=sysn or "", data=C.d["data"], runs=C.d["runs"],
                   pp_out=C.d["pp_out"], cfg=cfg_file, ckpt=C.d["checkpoints"])
    else:
        cmd = py(C, SRC["B"], cmd_kind, "--configs", cfg_file, "--only", name, "--seeds", seed,
                 "--data", C.d["data"], "--models_dir", C.d["models"], "--out", C.d["runs"], "--pp_out", C.d["pp_out"],
                 "--ckpt_dir", C.d["checkpoints"], *(["--system", sysn] if cmd_kind == "final" else []))
    return Exp(eid, stage, "B", name, seed, cmd, done, cleanup, done, from_cfg, f"B-{cmd_kind}", sysn)


def parse_finetune_stdout(txt):
    v = [float(x) for x in re.findall(r"\[B11\] seed \d+ epoch \d+: dev macro-F1 ([0-9.]+)", txt)]
    return {"dev_macro_f1": max(v) if v else None, "dev_per_epoch": v}


def exp_B11(C, seed, stage):
    """Original B11 (finetune.py, unmodified). Completion marker written by the orchestrator after success."""
    marker = os.path.join(C.d["pp_out"], "final", "B11", f"{seed}.json")
    pred_dir = os.path.join(C.d["runs"], "preds", "B11", str(seed))
    if C.dry:
        cmd = fake(C, "B11", seed=seed, data=C.d["data"], runs=C.d["runs"], marker=marker)
    else:
        inner = (f"import subprocess,sys,json,os,re; r=subprocess.run([sys.executable,{os.path.join(ORIG, 'finetune.py')!r},'--system','B11',"
                 f"'--model',{os.path.join(C.d['models'], 'xlmr')!r},'--data',{C.d['data']!r},'--out',{C.d['runs']!r},'--seeds','{seed}'],"
                 f"capture_output=True,text=True); sys.stdout.write(r.stdout); sys.stdout.write(r.stderr);"
                 f"v=[float(x) for x in re.findall(r'dev macro-F1 ([0-9.]+)', r.stdout)];"
                 f"ok=r.returncode==0 and os.path.exists(os.path.join({pred_dir!r},'my.tsv'));"
                 f"os.makedirs(os.path.dirname({marker!r}),exist_ok=True);"
                 f"ok and json.dump({{'dev':max(v) if v else None,'dev_per_epoch':v,'seed':{seed}}},open({marker!r},'w'));"
                 f"sys.exit(0 if ok else 1)")
        cmd = [sys.executable, "-c", inner]
    cfg = {"encoders": ["xlmr"], "adapt": "full", "lr": 1e-5, "epochs": 3, "batch": 8, "lam_cons": 0.5, "impl": "original finetune.py",
           "batch_note": "finetune.py defaults: --batch 16, and B11 uses batch//2 = 8 anchors per step (each with a perturbed partner)",
           "max_len": 64}
    return Exp(f"B11-s{seed}", stage, "B11", "B11 (original finetune.py)", seed, cmd, marker, [pred_dir], marker, cfg, "B11", "B11")


def load_cfg(C, cfg_file, name, setting):
    """Resolved configuration for the registry (defaults + inherited overrides), without importing torch."""
    d = json.load(open(CFG["defaults"][setting]))
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    spec = json.load(open(cfg_file))

    def ov(n):
        c = spec["configs"][n]; o = {}
        for par in c.get("inherit", []):
            o.update(ov(par))
        o.update({k: v for k, v in c.items() if k not in ("inherit", "family", "why")})
        return o
    d.update(ov(name)); return d


def dev_results(C, name):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.d["pp_out"], "dev", name, "*.json")))]


# ===================================================================== stages
def s00_environment(C):
    C.log("STAGE 00 — environment")
    info = {"time": now(), "python": sys.version.split()[0], "platform": platform.platform(), "dry_run": C.dry}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["gpu"] = q.stdout.strip() or "none"
    except Exception:
        info["gpu"] = "none"
    probe = ("import json,importlib.metadata as m\nout={}\n"
             "for p in ['torch','transformers','sentence-transformers','datasets','scipy','numpy','pandas','scikit-learn','matplotlib','umap-learn','huggingface_hub','deepsig']:\n"
             "    try: out[p]=m.version(p)\n    except Exception: out[p]=None\n"
             "try:\n import torch; out['cuda_available']=torch.cuda.is_available(); out['cuda']=torch.version.cuda\n"
             " out['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/2**30,1) if torch.cuda.is_available() else 0\n"
             "except Exception as e: out['cuda_available']=False; out['torch_error']=str(e)\nprint(json.dumps(out))")
    try:
        info.update(json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300).stdout.strip().splitlines()[-1]))
    except Exception as ex:
        info["probe_error"] = str(ex)
    try:
        mem = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo")}
        info["ram_total"] = mem.get("MemTotal"); info["ram_available"] = mem.get("MemAvailable")
    except Exception:
        pass
    for k, p in (("disk_work_free_gb", C.work), ("disk_drive_free_gb", C.run)):
        try:
            info[k] = round(shutil.disk_usage(p).free / 2**30, 1)
        except Exception:
            pass
    arch_env = json.load(open(os.path.join(C.arch, "environment.json"))) if os.path.exists(os.path.join(C.arch, "environment.json")) else {}
    info["archived_run_environment"] = arch_env
    with open(os.path.join(C.d["logs"], "environment.txt"), "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    atomic_json(info, os.path.join(C.d["logs"], "environment.json"))
    for k in ("gpu", "vram_gb", "cuda", "python", "torch", "transformers", "sentence-transformers", "datasets", "scipy", "numpy", "pandas"):
        C.log(f"  {k:22s} {info.get(k)}")
    if not C.dry and (C.a.gpu_required and not info.get("cuda_available")):
        raise Fatal("FATAL: GPU required. Select Runtime > Change runtime type > A100 GPU, then Run all again.")
    for k in ("torch", "transformers"):
        if arch_env.get(k) and info.get(k) and arch_env[k] != info[k]:
            C.log(f"  version of {k} differs from the archived run ({arch_env[k]} vs {info[k]}): reproduction checks will tell", "WARN")
    C.state["environment"] = {k: info.get(k) for k in ("gpu", "vram_gb", "cuda", "python", "torch", "transformers")}
    C.save_state(); C.env_info = info


def s01_project(C):
    C.log("STAGE 01 — project integrity (original code read-only, package files)")
    bad = []
    for line in open(os.path.join(ORIG, "MD5SUMS_unmodified.txt")):
        h, fn = line.split()
        if hashlib.md5(open(os.path.join(ORIG, fn), "rb").read()).hexdigest() != h:
            bad.append(fn)
    if bad:
        raise Fatal(f"FATAL: original code modified: {bad}")
    need = [SRC["A"], SRC["B"], os.path.join(PP, "scripts", "select_config.py"), CFG["protocol"]] + list(CFG["defaults"].values()) \
        + list(CFG["screening"].values()) + [os.path.join(PP, "evaluation", f) for f in ("evaluate.py", "stats.py", "tables.py", "figures.py", "report.py")]
    miss = [os.path.relpath(n, PP) for n in need if not os.path.exists(n)]
    if miss:
        raise Fatal(f"FATAL: package files missing: {miss}")
    for n in [CFG["protocol"]] + list(CFG["defaults"].values()) + list(CFG["screening"].values()):  # same layout as the repository
        dst = os.path.join(C.d["configs"], os.path.relpath(n, os.path.join(PP, "configs"))); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(n, dst)
    C.state["code_version"] = code_version(); C.save_state()
    C.log(f"  code version: {C.state['code_version']}")
    C.log("  original code: MD5 verified (unchanged)"); C.mark_stage("01_project")


def s02_data(C):
    C.log("STAGE 02 — data (public releases -> splits of the protocol)")
    if os.path.exists(os.path.join(C.d["data"], "prep_report.json")):
        C.log("  data already prepared on Drive: reused"); return
    if C.dry:
        subprocess.run([sys.executable, os.path.join(PP, "tests", "make_synthetic.py"), "--data", C.d["data"]], check=True)
        return
    r = subprocess.run(py(C, os.path.join(ORIG, "download_models.py"), "--skip_models", "--raw_dir", C.d["raw"]),
                       cwd=C.work, env=C.env, capture_output=True, text=True)
    open(os.path.join(C.d["logs"], "download_corpora.log"), "w").write(r.stdout + r.stderr)
    for need in ("TSAC", "TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset", "Narabizi", "MYC"):
        if not os.path.isdir(os.path.join(C.d["raw"], need)):
            raise Fatal(f"FATAL: corpus {need} could not be downloaded (see logs/download_corpora.log)")
    env = dict(C.env, RAW_DIR=C.d["raw"], DATA_DIR=C.d["data"])
    r = subprocess.run(py(C, os.path.join(ORIG, "prep_data.py")), cwd=C.work, env=env, capture_output=True, text=True)
    open(os.path.join(C.d["logs"], "prep_data.log"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0 or not os.path.exists(os.path.join(C.d["data"], "prep_report.json")):
        raise Fatal("FATAL: prep_data.py failed (see logs/prep_data.log)")


def s03_split_check(C):
    C.log("STAGE 03 — split / label / leakage verification against the archived run")
    import evaluate as EV  # numpy only
    rep = {"time": now(), "checks": {}}
    D = C.d["data"]
    new = json.load(open(os.path.join(D, "prep_report.json"))); old = json.load(open(os.path.join(C.arch, "prep_report.json")))
    diff = {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    rep["checks"]["counts_vs_archived_prep_report"] = "PASS" if not diff else diff
    if diff:
        atomic_json(rep, os.path.join(C.d["results"], "split_check.json"))
        raise Fatal(f"FATAL: split mismatch (counts differ from the archived run): {list(diff.items())[:8]}")
    sets = ["train_ar", "train_gs", "train_gu", "train_inf", "train_noise", "train_cs", "val_ar", "val_gs",
            "ar", "gs", "gu", "tl", "tz", "na", "naar", "my", "myar", "gsmy"]
    rows = {}
    for s in sets:
        p = os.path.join(D, f"{s}.tsv")
        if not os.path.exists(p):
            raise Fatal(f"FATAL: data file missing: {s}.tsv")
        rows[s] = read_tsv(p)
        ids = [r["id"] for r in rows[s]]
        if len(set(ids)) != len(ids):
            raise Fatal(f"FATAL: duplicate ids in {s}")
        if any(r["label"] not in ("0", "1") for r in rows[s]):
            raise Fatal(f"FATAL: labels outside {{0,1}} in {s}")
    for a, b in (("gs", "ar"), ("gu", "ar"), ("naar", "na"), ("gsmy", "myar"), ("val_gs", "val_ar"),
                 ("train_gs", "train_ar"), ("train_gu", "train_ar"), ("train_inf", "train_ar"), ("train_noise", "train_ar"), ("train_cs", "train_ar")):
        if [r["id"] for r in rows[a]] != [r["id"] for r in rows[b]] or [r["label"] for r in rows[a]] != [r["label"] for r in rows[b]]:
            raise Fatal(f"FATAL: paired sets {a}/{b} not aligned (ids or labels)")
    rep["checks"]["ids_labels_pairing"] = "PASS"
    # text anchors from the archived run (audit sheets exported by the archived pipeline)
    n_anchor = 0; bad = []
    idx = {s: {r["id"]: r for r in rows[s]} for s in ("my", "tz", "ar", "gs", "gu")}
    # The anchor sheets contain dataset texts; they are NOT distributed in the git repository (redistribution rights
    # not established). They are used when present in the archive or in <run_dir>/private_inputs/audit_anchors/.
    A = next((x for x in (os.path.join(C.arch, "audit_anchors"), os.path.join(C.run, "private_inputs", "audit_anchors"))
              if all(os.path.exists(os.path.join(x, f)) for f in ("errors_my.tsv", "errors_tz.tsv", "gen_gs.tsv", "gen_gu.tsv"))), None)
    if A is None:
        C.log("  WARNING: archived text anchors not available (not distributed with the repository); identity is checked by "
              "counts, id/label pairing and exact re-scoring of the archived predictions only", "WARN")
    for fn, s in (("errors_my.tsv", "my"), ("errors_tz.tsv", "tz")):
        for r in (read_tsv(os.path.join(A, fn)) if A else []):
            n_anchor += 1; d = idx[s].get(r["id"])
            if d is None or d["text"] != r["text"] or d["label"] != r["gold"]:
                bad.append((fn, r["id"]))
    for fn, s in (("gen_gs.tsv", "gs"), ("gen_gu.tsv", "gu")):
        for r in (read_tsv(os.path.join(A, fn)) if A else []):
            n_anchor += 1
            if idx["ar"].get(r["id"], {}).get("text") != r["source_arabic"] or idx[s].get(r["id"], {}).get("text") != r["output"]:
                bad.append((fn, r["id"]))
    rep["checks"]["text_anchors"] = {"n": n_anchor, "mismatches": len(bad), "examples": bad[:10], "source": A or "NOT AVAILABLE"}
    if bad:
        atomic_json(rep, os.path.join(C.d["results"], "split_check.json"))
        raise Fatal(f"FATAL: split mismatch: {len(bad)}/{n_anchor} archived text/label anchors differ")
    # exact re-scoring of the archived predictions with the regenerated gold labels
    rs = EV.rescore_archived(D, os.path.join(C.arch, "preds"), os.path.join(C.arch, "summary.json"))
    rep["checks"]["archived_predictions_rescored"] = rs
    if rs["mismatches"]:
        atomic_json(rep, os.path.join(C.d["results"], "split_check.json"))
        raise Fatal(f"FATAL: split mismatch: archived predictions do not re-score to the archived metrics ({rs['mismatches'][:5]})")
    # leakage (exact text overlap after lower-casing/stripping)
    norm = lambda t: " ".join(t.lower().split())
    tr_ar = {norm(r["text"]) for s in ("train_ar", "val_ar") for r in rows[s]}
    tr_lat = {norm(r["text"]) for s in ("train_gs", "val_gs") for r in rows[s]}
    leak = {s: sum(norm(r["text"]) in (tr_ar if s in ("ar", "naar", "myar") else tr_lat) for r in rows[s])
            for s in ("ar", "gs", "gu", "tl", "tz", "na", "naar", "my", "myar", "gsmy")}
    rep["checks"]["exact_overlap_with_train_or_dev"] = leak
    if leak["ar"] or leak["gs"]:
        raise Fatal(f"FATAL: leakage: S0/S1 sentences found in training/dev ({leak})")
    if any(leak.values()):
        C.log(f"  exact train/dev overlaps (reported, not fatal): {leak}", "WARN")
    # manifest (hashes) — compared across sessions of this run
    man = {}
    for s in sorted(f[:-4] for f in os.listdir(D) if f.endswith(".tsv")):
        rr = read_tsv(os.path.join(D, f"{s}.tsv"))
        man[s] = {"n": len(rr), "file_sha256": sha256_file(os.path.join(D, f"{s}.tsv")),
                  "ids_sha256": sha256_text("\n".join(r["id"] for r in rr)),
                  "texts_sha256": sha256_text("\n".join(r["text"] for r in rr))}
    mp = os.path.join(C.d["logs"], "split_manifest.json")
    if os.path.exists(mp):
        prev = json.load(open(mp))
        changed = [s for s in man if s in prev and prev[s]["file_sha256"] != man[s]["file_sha256"]]
        if changed:
            raise Fatal(f"FATAL: split mismatch: data files changed since the first session of this run: {changed}")
    else:
        atomic_json(man, mp)
    rep["manifest"] = mp; atomic_json(rep, os.path.join(C.d["results"], "split_check.json"))
    names = [("TRAIN", "train_ar"), ("DEV", "val_ar"), ("DEV (syn)", "val_gs"), ("TEST S0", "ar"), ("TEST S1", "gs"), ("TEST S2", "gu"),
             ("TEST S3", "tl"), ("TEST S4", "tz"), ("TEST S5", "na"), ("TEST S5AR", "naar"), ("TEST S6", "my")]
    for lab, s in names:
        C.log(f"  {lab:10s} n={man[s]['n']:6d}  ids {man[s]['ids_sha256'][:12]}  texts {man[s]['texts_sha256'][:12]}  file {man[s]['file_sha256'][:12]}")
    C.log(f"  identity with the archived run: counts PASS, {(str(n_anchor) + ' text anchors PASS') if A else 'text anchors NOT AVAILABLE'}, archived predictions re-scored exactly "
          f"({rs['n_checked']} system/set/metric values)")


def s04_models(C):
    C.log("STAGE 04 — encoders (persistent cache on Drive)")
    M = C.d["models"]; man_p = os.path.join(M, "MODEL_MANIFEST.json")
    names = ["xlmr", "me5", "labse"]
    if C.dry:
        for n in names:
            os.makedirs(os.path.join(M, n), exist_ok=True)
            p = os.path.join(M, n, "model.safetensors")
            if not os.path.exists(p):
                open(p, "wb").write(os.urandom(1024))
            open(os.path.join(M, n, "config.json"), "w").write("{}")
    else:
        missing = [n for n in names if not glob.glob(os.path.join(M, n, "*.safetensors")) + glob.glob(os.path.join(M, n, "*.bin"))]
        if missing:
            r = subprocess.run(py(C, os.path.join(ORIG, "download_models.py"), "--skip_data", "--models_dir", M), cwd=C.work, env=C.env,
                               capture_output=True, text=True)
            open(os.path.join(C.d["logs"], "download_models.log"), "w").write(r.stdout + r.stderr)
    man = {}
    for n in names:
        ws = sorted(glob.glob(os.path.join(M, n, "*.safetensors")) + glob.glob(os.path.join(M, n, "*.bin")))
        if not ws or not os.path.exists(os.path.join(M, n, "config.json")):
            raise Fatal(f"FATAL: model {n} missing in {M} (see logs/download_models.log)")
        man[n] = {os.path.basename(w): sha256_file(w) for w in ws}
    if os.path.exists(man_p):
        prev = json.load(open(man_p))
        if prev.get("weights") != man:
            raise Fatal("FATAL: model weights changed since they were first downloaded for this run (checksum mismatch)")
    else:
        repo = {}
        if not C.dry:
            q = ("import json;from huggingface_hub import HfApi;a=HfApi();"
                 "print(json.dumps({k:a.model_info(v).sha for k,v in {'xlmr':'FacebookAI/xlm-roberta-large','me5':'intfloat/multilingual-e5-large','labse':'sentence-transformers/LaBSE'}.items()}))")
            try:
                repo = json.loads(subprocess.run([sys.executable, "-c", q], capture_output=True, text=True, timeout=120).stdout.strip())
            except Exception:
                repo = {"note": "Hugging Face revision could not be queried"}
        atomic_json({"weights": man, "hf_revision": repo, "time": now()}, man_p)
    C.log(f"  models verified: " + ", ".join(f"{n} ({len(man[n])} weight file(s))" for n in names))


def s05_features(C):
    C.log("STAGE 05 — cached features (local disk; recomputed after a disconnection)")
    if C.dry:
        return
    r = subprocess.run(py(C, os.path.join(ORIG, "extract.py"), "--models_dir", C.d["models"], "--data", C.d["data"], "--out", C.d["feats"],
                          "--encoders", C.P["encoders"], "--layer_stride", 2, "--batch", 128), cwd=C.work, env=C.env, capture_output=True, text=True)
    open(os.path.join(C.d["logs"], "extract_features.log"), "a").write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise Fatal("FATAL: feature extraction failed (see logs/extract_features.log)")


def repro_table(C, title, rows, tol_pass, tol_warn):
    lines = ["=" * 60, title, "=" * 60, f"{'metric/seed':28s} {'archived':>9s} {'reproduced':>11s} {'diff':>8s} {'tolerance':>10s} status"]
    worst = "PASS"
    for lab, a, b in rows:
        if b is None:
            st = "FAIL"; d = float("nan")
        else:
            d = b - a; st = "PASS" if abs(d) <= tol_pass else ("WARNING" if abs(d) <= tol_warn else "FAIL")
        worst = {"PASS": 0, "WARNING": 1, "FAIL": 2}[st] > {"PASS": 0, "WARNING": 1, "FAIL": 2}[worst] and st or worst
        lines.append(f"{lab:28s} {a:9.4f} {('%.4f' % b) if b is not None else 'missing':>11s} {d:+8.4f} {tol_pass:10.4f} {st}")
    lines.append(f"STATUS: {worst}")
    for l in lines:
        C.log(l)
    return worst, "\n".join(lines)


def s06_reproduce_B12(C):
    C.log("STAGE 06 — reproduction of B12 (V0 = original trainer) and implementation check P0")
    cfgA = CFG["screening"]["A"]; seeds = C.P["seeds"]["reproduction"]
    run_many(C, [exp_A(C, n, s, "search", cfgA, stage="reproduce_B12") for n in ("V0", "P0") for s in seeds])
    arch = json.load(open(os.path.join(C.arch, "dev_scores.json"))); R = C.P["reproduction"]["B12"]
    v0 = {r["seed"]: r["dev"] for r in dev_results(C, "V0")}; p0 = {r["seed"]: r["dev"] for r in dev_results(C, "P0")}
    rows = [(f"B12 dev macro-F1 seed {s}", arch[f"B12/{s}"], v0.get(s)) for s in seeds]
    st, txt = repro_table(C, "B12 REPRODUCTION (V0 = original DiaFuse-PRC trainer, dev only)", rows, R["pass_abs_diff_per_seed"], R["warn_abs_diff_per_seed"])
    os.makedirs(os.path.join(C.d["results"], RESULT_FOLDERS["B12"]), exist_ok=True)
    open(os.path.join(C.d["results"], RESULT_FOLDERS["B12"], "reproduction.txt"), "w").write(txt)
    tol = C.P["reproduction"]["P0_vs_V0"]["pass_abs_diff_per_seed"]
    p0ok = all(s in v0 and s in p0 and abs(v0[s] - p0[s]) <= tol for s in seeds)
    C.state["flags"].update({"B12_reproduction": st, "settingA_implementation_check": "PASS" if p0ok else "FAIL",
                             "P0_vs_V0": {s: (v0.get(s), p0.get(s)) for s in seeds}})
    C.save_state()
    if st == "FAIL":
        raise Fatal("FATAL: B12 reproduction failed. Experimental pipeline stopped.")
    if not p0ok:
        C.log("Setting A implementation check FAILED (P0 differs from V0): Setting A variants will NOT be run", "ERROR")
    C.mark_stage("06_reproduce_B12")


def s07_reproduce_B11(C):
    C.log("STAGE 07 — reproduction of B11 (original finetune.py, unmodified)")
    seeds = C.P["seeds"]["reproduction"]
    recs = run_many(C, [exp_B11(C, s, "reproduce_B11") for s in seeds])
    arch = json.load(open(os.path.join(C.arch, "finetune_dev_log.json")))["best_dev_macro_f1"]["B11"]; R = C.P["reproduction"]["B11"]
    rows = [(f"B11 dev macro-F1 seed {s}", arch[str(s)], r.get("dev_macro_f1") if r.get("status") == "completed" else None) for s, r in zip(seeds, recs)]
    st, txt = repro_table(C, "B11 REPRODUCTION (original finetune.py, dev only)", rows, R["pass_abs_diff_per_seed"], R["warn_abs_diff_per_seed"])
    os.makedirs(os.path.join(C.d["results"], "B11"), exist_ok=True)
    open(os.path.join(C.d["results"], "B11", "reproduction.txt"), "w").write(txt)
    C.state["flags"]["B11_reproduction"] = st; C.state["flags"]["B11_dev_this_run"] = {str(s): b for (_, _, b), s in zip(rows, seeds)}
    C.save_state()
    if st == "FAIL":
        raise Fatal("FATAL: B11 reproduction failed. Experimental pipeline stopped.")
    C.mark_stage("07_reproduce_B11")


def mode_list(C, setting, all_names):
    sel = C.P["modes"][C.mode][f"setting{setting}_configs"]
    return list(all_names) if sel == "all" else [n for n in all_names if n in sel]


def s08_screening(C):
    C.log(f"STAGE 08 — screening (dev only; mode {C.mode})")
    seeds = C.P["seeds"]["search"]
    cfgA = CFG["screening"]["A"]; cfgB = CFG["screening"]["B"]
    if C.state["flags"].get("settingA_implementation_check") == "PASS":
        namesA = [n for n in mode_list(C, "A", json.load(open(cfgA))["configs"]) if n not in ("V0", "P0")]
        workers = parallel_workers(C)
        C.log(f"  Setting A: {len(namesA)} configurations x {len(seeds)} seeds ({workers} parallel worker(s))")
        run_many(C, [exp_A(C, n, s, "search", cfgA) for n in namesA for s in seeds], workers)
    namesB = mode_list(C, "B", json.load(open(cfgB))["configs"])
    order = ["FT0"] + [n for n in namesB if n != "FT0"]
    C.log(f"  Setting B: {len(order)} configurations x {len(seeds)} seeds (sequential)")
    run_many(C, [exp_B(C, "FT0", s, "search", cfgB) for s in seeds])
    ft0 = {r["seed"]: r["dev"] for r in dev_results(C, "FT0")}
    b11 = {s: r.get("dev_macro_f1") for s, r in ((s, json.load(open(os.path.join(C.d["exp_state"], f"B11-s{s}.json")))) for s in C.P["seeds"]["reproduction"])}
    common = [s for s in ft0 if b11.get(s) is not None]
    diff = (sum(ft0[s] for s in common) / len(common) - sum(b11[s] for s in common) / len(common)) if common else None
    okB = diff is not None and abs(diff) <= C.P["reproduction"]["FT0_vs_B11"]["pass_abs_diff_mean"]
    C.state["flags"]["settingB_implementation_check"] = "PASS" if okB else "FAIL"
    C.state["flags"]["FT0_minus_B11_dev_mean"] = diff; C.save_state()
    C.log(f"  FT0 (re-implemented B11) vs original B11, mean dev difference = {diff}: {'PASS' if okB else 'FAIL'}")
    if okB:
        run_many(C, [exp_B(C, n, s, "search", cfgB) for n in order if n != "FT0" for s in seeds])
    else:
        C.log("Setting B implementation check FAILED: Setting B variants will NOT be run", "ERROR")
    C.mark_stage("08_screening")


def parallel_workers(C):
    if C.dry:
        return 2
    info = getattr(C, "env_info", {}) or json.load(open(os.path.join(C.d["logs"], "environment.json")))
    try:
        ram_gb = int(str(info.get("ram_total", "0 kB")).split()[0]) / 2**20
    except Exception:
        ram_gb = 0
    if (info.get("vram_gb") or 0) >= 39 and ram_gb >= C.P["parallel"]["min_ram_gb_for_parallel"]:
        return C.P["parallel"]["settingA_search_workers_if_A100_40GB"]
    return 1


def run_select(C, phase, setting, cfg):
    args = [os.path.join(PP, "scripts", "select_config.py"), phase, "--setting", setting, "--configs", cfg,
            "--dev", os.path.join(C.d["pp_out"], "dev"), "--out_prefix", os.path.join(C.d["configs"], f"setting{setting}")]
    r = subprocess.run([sys.executable, *args], cwd=C.work, capture_output=True, text=True)
    open(os.path.join(C.d["logs"], f"selection_{setting}_{phase}.log"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def s09_selection(C):
    C.log("STAGE 09 — model selection (pre-registered rule; dev only)")
    if C.stage_done("09_selection") and os.path.exists(os.path.join(C.run, "SELECTED_CONFIG.json")):
        C.log("  selection already made in a previous session: frozen (never recomputed once final training may have started)")
        return
    seeds = C.P["seeds"]["search"]; sel = {"rule": C.P["selection_rule"], "seed_policy": C.P["seeds"], "settings": {}}
    prev_p = os.path.join(C.run, "SELECTED_CONFIG.json")
    prev = json.load(open(prev_p))["settings"] if os.path.exists(prev_p) else {}
    for setting, flag in (("A", "settingA_implementation_check"), ("B", "settingB_implementation_check")):
        cfg = CFG["screening"][setting]
        if setting in prev and prev[setting].get("status") != "FAILED":  # a successful selection is frozen per setting
            sel["settings"][setting] = prev[setting]; C.log(f"  Setting {setting}: selection frozen from a previous session"); continue
        if C.state["flags"].get(flag) != "PASS":
            sel["settings"][setting] = {"status": "NOT RUN (implementation check failed)"}; continue
        try:
            run_select(C, "screen", setting, cfg)
            comb = os.path.join(C.d["configs"], f"setting{setting}_combine.json")
            names = [n for n in json.load(open(comb))["configs"] if n.startswith("ALL")]
            if names:
                fac = exp_A if setting == "A" else exp_B
                run_many(C, [fac(C, n, s, "search", comb, stage="selection_combinations") for n in names for s in seeds],
                         parallel_workers(C) if setting == "A" else 1)
            out = run_select(C, "final", setting, cfg)
            selected = json.load(open(os.path.join(C.d["configs"], f"setting{setting}_selected.json")))
            fin = load_cfg(C, os.path.join(C.d["configs"], f"setting{setting}_selected.json"), "FINAL", setting)
            ref = C.P["selection_rule"]["references"][setting]
            sel["settings"][setting] = {"status": "selected" if selected["selected_from"] != ref else "reference (no improvement on dev)",
                                        "selected_from": selected["selected_from"], "final_config": fin,
                                        "ablations": [n for n in selected["configs"] if n.startswith("FINAL-")],
                                        "report": os.path.relpath(os.path.join(C.d["configs"], f"setting{setting}_final_report.md"), C.run)}
            C.log(out.strip().splitlines()[-2] if out.strip() else "")
        except Exception:
            sel["settings"][setting] = {"status": "FAILED", "error": traceback.format_exc()[-2000:]}
            C.log(f"selection for Setting {setting} failed (recorded)", "ERROR")
    atomic_json(sel, os.path.join(C.run, "SELECTED_CONFIG.json")); atomic_json(sel, os.path.join(C.d["configs"], "SELECTED_CONFIG.json"))
    C.state["best_config"] = {k: v.get("selected_from", v["status"]) for k, v in sel["settings"].items()}
    if any(v["status"] == "FAILED" for v in sel["settings"].values()):  # never frozen: retried at the next session
        C.log("selection not frozen because it failed for at least one setting; it will be retried when the notebook is run again", "ERROR")
        C.save_state(); return
    C.mark_stage("09_selection")


def selected(C, setting):
    p = os.path.join(C.run, "SELECTED_CONFIG.json")
    s = json.load(open(p))["settings"].get(setting, {}) if os.path.exists(p) else {}
    return s if s.get("status") == "selected" else None


def s10_final(C):
    C.log("STAGE 10 — final training (10 seeds; test predictions written once)")
    seeds = C.P["seeds"]["final"]; cfgA = CFG["screening"]["A"]
    exps = [exp_B11(C, s, "final") for s in seeds]                                   # seeds 0-2 already done in stage 07
    exps += [exp_A(C, "V0", s, "final", cfgA, system="B12", stage="final") for s in seeds]
    for setting, sysn, fac in (("A", "PPA-FINAL", exp_A), ("B", "PPB-FINAL", exp_B)):
        if selected(C, setting):
            selp = os.path.join(C.d["configs"], f"setting{setting}_selected.json")
            exps += [fac(C, "FINAL", s, "final", selp, system=sysn, stage="final") for s in seeds]
    run_many(C, exps); C.mark_stage("10_final")


def s11_ablations(C):
    C.log("STAGE 11 — ablations of the selected models (5 seeds)")
    seeds = C.P["seeds"]["ablation"]; allowed = C.P["modes"][C.mode]["ablations"]; exps = []
    for setting, fac in (("A", exp_A), ("B", exp_B)):
        s = selected(C, setting)
        if not s:
            continue
        selp = os.path.join(C.d["configs"], f"setting{setting}_selected.json")
        for n in s["ablations"]:
            if allowed != "all" and n not in allowed:
                continue
            exps += [fac(C, n, sd, "final", selp, system=f"PP{setting}-ABL-{n[6:]}", stage="ablations") for sd in seeds]
    run_many(C, exps); C.mark_stage("11_ablations")


def s12_16_analysis(C):
    import evaluate as EV, stats as ST, tables as TB, figures as FG, report as RP
    C.log("STAGE 12 — evaluation (original diafuse_eval.py + per-seed metrics)")
    for s in ("B0", "B6", "B10", "B11", "B12"):  # archived 3-seed predictions, re-scored with this run's labels
        dst = os.path.join(C.d["runs"], "preds", f"ARCH-{s}")
        if not os.path.exists(dst) and os.path.isdir(os.path.join(C.arch, "preds", s)):
            shutil.copytree(os.path.join(C.arch, "preds", s), dst)
    ev_out = os.path.join(C.d["results"], "eval_original")
    r = subprocess.run(py(C, os.path.join(ORIG, "diafuse_eval.py"), "--data", C.d["data"], "--preds", os.path.join(C.d["runs"], "preds"),
                          "--diag", os.path.join(C.d["runs"], "diag"), "--out", ev_out), cwd=C.work, env=C.env, capture_output=True, text=True)
    open(os.path.join(C.d["logs"], "diafuse_eval.log"), "w").write(r.stdout + r.stderr)
    C.state["flags"]["original_evaluator"] = "OK" if r.returncode == 0 else "FAILED (see logs/diafuse_eval.log)"
    met = EV.evaluate_all(C.d["data"], os.path.join(C.d["runs"], "preds"), C.d["results"])
    # test macro-F1 per final experiment into the registry
    for rec in C.records():
        if rec.get("system") and rec["status"] == "completed":
            v = met.get("mean_S0_S6", {}).get(rec["system"], {}).get(str(rec["seed"]))
            if v is not None:
                rec["test_macro_f1_mean_S0_S6"] = v; atomic_json(rec, os.path.join(C.d["exp_state"], f"{rec['experiment_id']}.json"))
    write_registry(C); C.mark_stage("12_evaluation")
    C.log("STAGE 13 — statistics (protocol tests)")
    st = ST.run_all(C.d["data"], os.path.join(C.d["runs"], "preds"), C.P, C.d["tables"], ORIG); C.mark_stage("13_statistics")
    C.log("STAGE 14 — tables")
    TB.write_all(C.d["results"], C.d["tables"], C.d["pp_out"], C.P, C.run); C.mark_stage("14_tables")
    C.log("STAGE 15 — figures")
    fails = FG.write_all(C.d["results"], C.d["tables"], C.d["figures"], os.path.join(C.d["runs"], "diag"), C.d["pp_out"], C.P)
    if fails:
        C.log(f"  figures not produced: {fails}", "WARN")
    C.mark_stage("15_figures")
    for s, folder in RESULT_FOLDERS.items():  # one folder per model: B11 / B12_setting_A / B12pp_setting_A / B12_setting_B_lora
        TB.export_system(C.d["results"], os.path.join(C.d["runs"], "preds"), s, os.path.join(C.d["results"], folder))
        write_trace_card(C, s, os.path.join(C.d["results"], folder))
    for d in glob.glob(os.path.join(C.d["runs"], "preds", "PP?-ABL-*")):
        n = os.path.basename(d); key = "FINAL-" + n.split("-ABL-")[1]
        folder = ABL_FOLDER.get(key) or ("without_" + key[len("FINAL-minus-"):] + "_component" if key.startswith("FINAL-minus-") else key)
        TB.export_system(C.d["results"], os.path.join(C.d["runs"], "preds"), n,
                         os.path.join(C.d["results"], "ablations", f"setting{n[2]}_{folder}"))
        write_trace_card(C, n, os.path.join(C.d["results"], "ablations", f"setting{n[2]}_{folder}"))
    return st


def s16_report_archive(C, complete):
    import report as RP
    C.log("STAGE 16 — report and archives")
    RP.write_report(C.run, C.d, C.P, C.state, C.records(), complete)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    old = os.path.join(C.d["final_archive"], "older_sessions"); os.makedirs(old, exist_ok=True)
    for z in glob.glob(os.path.join(C.d["final_archive"], "DiaFuse_PRCPP_*.zip")):  # archives of earlier sessions
        shutil.move(z, os.path.join(old, os.path.basename(z)))
    zres = os.path.join(C.d["final_archive"], f"DiaFuse_PRCPP_FINAL_RESULTS_{ts}.zip")
    with zipfile.ZipFile(zres, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("FINAL_REPORT.md", "SELECTED_CONFIG.json", "run_state.json", "experiment_registry.csv", "final_comparison.csv",
                  "final_comparison.tex", "final_comparison.md", "main_results.csv", "main_results.tex"):
            if os.path.exists(os.path.join(C.run, f)):
                z.write(os.path.join(C.run, f), f)
        if os.path.exists(os.path.join(C.d["tables"], "statistical_tests.csv")):
            z.write(os.path.join(C.d["tables"], "statistical_tests.csv"), "statistical_tests.csv")
        for top, sub in [(f, f) for f in RESULT_FOLDERS.values()]:  # top-level copies, one folder per model
            for root, _, files in os.walk(os.path.join(C.d["results"], sub)):
                for fn in files:
                    z.write(os.path.join(root, fn), os.path.join(top, os.path.relpath(os.path.join(root, fn), os.path.join(C.d["results"], sub))))
        for sub in ("results", "tables", "figures", "logs", "configs", "state"):
            for root, _, files in os.walk(os.path.join(C.run, sub)):
                for fn in files:
                    p = os.path.join(root, fn)
                    if fn.endswith(".tmp"):
                        continue
                    if os.path.getsize(p) < 200 * 2**20:
                        z.write(p, os.path.relpath(p, C.run))
    zck = os.path.join(C.d["final_archive"], f"DiaFuse_PRCPP_CHECKPOINTS_{ts}.zip")
    with zipfile.ZipFile(zck, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(C.d["checkpoints"]):
            for fn in files:
                if "/resume/" not in os.path.join(root, fn):
                    z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), C.run))
    C.state["final_archive"] = os.path.relpath(zres, C.run); C.state["checkpoint_archive"] = os.path.relpath(zck, C.run)
    C.mark_stage("16_report_archive")
    return zres, zck


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True); ap.add_argument("--work_dir", required=True)
    ap.add_argument("--mode", default="FULL", choices=["FULL", "PRIORITY"])
    ap.add_argument("--resume", type=int, default=1); ap.add_argument("--gpu_required", type=int, default=1)
    ap.add_argument("--run_reproduction", type=int, default=1); ap.add_argument("--run_screening", type=int, default=1)
    ap.add_argument("--run_final", type=int, default=1); ap.add_argument("--run_ablations", type=int, default=1)
    ap.add_argument("--generate_report", type=int, default=1); ap.add_argument("--zip_results", type=int, default=1)
    ap.add_argument("--models_dir", default=None); ap.add_argument("--archive_dir", default=None)
    ap.add_argument("--retry_failed", type=int, default=0, help="reset the attempt counters of failed experiments")
    ap.add_argument("--dry_run", action="store_true", help="test mode: synthetic data and stand-in experiments (NO real results)")
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True); os.makedirs(a.work_dir, exist_ok=True)
    C = Ctx(a)
    banner = "DRY RUN — SYNTHETIC TEST, NOT EXPERIMENTAL RESULTS" if a.dry_run else f"DiaFuse-PRC++ one-click run — mode {a.mode}"
    C.log("#" * 70); C.log(banner); C.log(f"PROJECT ROOT: {PKG}"); C.log(f"ARCHIVE ROOT: {C.arch}"); C.log(f"DATA ROOT:    {C.d['data']}")
    C.log(f"MODEL CACHE:  {C.d['models']}"); C.log(f"OUTPUT ROOT:  {C.run}"); C.log(f"WORK DIR:     {C.work} (local, temporary)"); C.log("#" * 70)
    code = 0
    try:
        s00_environment(C); s01_project(C); s02_data(C); s03_split_check(C); s04_models(C); s05_features(C)
        if a.run_reproduction or not C.stage_done("07_reproduce_B11"):
            s06_reproduce_B12(C); s07_reproduce_B11(C)
        if a.run_screening:
            s08_screening(C); s09_selection(C)
        if a.run_final:
            s10_final(C)
        if a.run_ablations:
            s11_ablations(C)
        s12_16_analysis(C)
        failed = [r["experiment_id"] for r in C.records() if r["status"] != "completed"]
        sp = os.path.join(C.run, "SELECTED_CONFIG.json")
        if a.run_screening and (not os.path.exists(sp) or not C.stage_done("09_selection")):
            failed.append("model selection (stage 09)")
        complete = not failed
        C.save_state(status="completed" if complete else "incomplete")
        if a.generate_report or a.zip_results:
            zres, zck = s16_report_archive(C, complete)
        code = 0 if complete else 2
        bar = "=" * 52
        if complete:
            print(f"\n{bar}\nRUN COMPLETED\n{bar}\n\nResults archive:\n{zres}\n\nCheckpoints archive:\n{zck}\n\n"
                  f"Please send the results archive to the project owner.\n\nDo NOT modify the results manually.\n"
                  f"Do NOT rerun individual experiments unless requested.\n{bar}")
        else:
            print(f"\n{bar}\nRUN INCOMPLETE\n{bar}\n\nSome experiments failed or remain unfinished ({len(failed)}).\n\nSee:\n"
                  f"{os.path.join(C.run, 'FINAL_REPORT.md')}\n{C.state_path}\n{C.d['logs']}\n\n"
                  f"Please send the results archive and the complete run directory\n{C.run}\nto the project owner.\n{bar}")
    except Fatal as ex:
        C.save_state(status="fatal", fatal=str(ex)); C.log(str(ex), "FATAL")
        try:
            import report as RP
            RP.write_report(C.run, C.d, C.P, C.state, C.records(), False)
        except Exception:
            pass
        print(f"\n{'=' * 52}\n{ex}\n{'=' * 52}\nNothing else was run. Send {C.run}/logs and run_state.json to the project owner.")
        code = 3
    except KeyboardInterrupt as ex:
        C.save_state(status="interrupted"); C.log(f"interrupted: {ex}. Re-run the notebook to resume.", "WARN"); code = 4
    sys.exit(code)


if __name__ == "__main__":
    main()
