#!/usr/bin/env python3
"""TEST FIXTURE ONLY (dry run). Stand-in for src/diafuse_pp_setting_A.py / src/diafuse_pp_setting_B_lora.py / original finetune.py that writes files in
exactly the formats of the real scripts, with SYNTHETIC values. Failure injection: environment variable
DRYRUN_FAIL = comma-separated experiment keys "<kind>:<name>:<system>:<seed>" (fails every time while the key is listed, leaving partial outputs)."""
import argparse
import csv
import json
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic import fake_preds  # noqa: E402

ACC = {"B11": .82, "B12": .76, "PPA-FINAL": .78, "PPB-FINAL": .83}
BONUS = {"V1b": .006, "V3a": .005, "V6a": .004, "FT2": .004, "FT3": .001, "ALL": .008}
BONUS.update(json.loads(os.environ.get("DRYRUN_BONUS", "{}")))  # test knob, e.g. '{"FT2": 0.02}' to exercise the Setting B final path


def maybe_fail(a, key, partial=None):
    fails = [x for x in os.environ.get("DRYRUN_FAIL", "").split(",") if x]
    if key in fails:  # fails every time while the key is listed
        if partial:
            partial()
        raise RuntimeError(f"SYNTHETIC injected failure for {key}")


def dev_value(name, seed, arch):
    rng = random.Random(zlib.crc32(f'{name}{seed}'.encode()))
    if name in ("V0", "P0"):
        return arch["dev_B12"].get(seed, 0.8834) + rng.uniform(-0.0004, 0.0004) if name == "V0" else None
    if name == "FT0":
        return arch["dev_B11"].get(seed, 0.937) + rng.uniform(-0.002, 0.002)
    base = 0.935 if name.startswith("FT") or name.startswith("ALL") and "FT" in name else 0.8832
    return base + BONUS.get(name.split("-")[0], 0.0) + rng.uniform(-0.0008, 0.0008)


def main():
    ap = argparse.ArgumentParser()
    for k in ("kind", "name", "seed", "system", "data", "runs", "pp_out", "cfg", "ckpt", "marker"):
        ap.add_argument(f"--{k}", default="")
    a = ap.parse_args(); seed = int(a.seed)
    arch = {"dev_B12": {0: 0.8830, 1: 0.8834, 2: 0.8838}, "dev_B11": {0: 0.94, 1: 0.9351, 2: 0.9358}}
    time.sleep(0.05)
    print(f"SYNTHETIC stand-in experiment {a.kind} {a.name} {a.system} seed {seed}", flush=True)
    if a.kind == "B11":
        maybe_fail(a, f"B11::B11:{seed}")
        rng = random.Random(seed)
        for ep in range(3):
            print(f"[B11] seed {seed} epoch {ep}: dev macro-F1 {arch['dev_B11'].get(seed, 0.937) - 0.01 * (2 - ep) + rng.uniform(-0.001, 0.001):.4f}", flush=True)
        fake_preds(a.data, os.path.join(a.runs, "preds"), "B11", seed, ACC["B11"], random.Random(100 + seed))
        os.makedirs(os.path.dirname(a.marker), exist_ok=True)
        json.dump({"dev": arch["dev_B11"].get(seed, 0.937), "seed": seed, "SYNTHETIC": True}, open(a.marker, "w"))
        return
    spec = json.load(open(a.cfg))["configs"]; fam = spec.get(a.name, {}).get("family", "")
    if a.kind.endswith("search"):
        maybe_fail(a, f"{a.kind}:{a.name}::{seed}")
        dev = dev_value(a.name, seed, arch)
        if a.name == "P0":
            dev = dev_value("V0", seed, arch)
        if dev is None:
            dev = 0.88
        alpha = [0.0, 1.0, 0.0] if a.name in ("V0", "P0") else [0.3, 0.45, 0.25]
        diag = {s: {"alpha_mean": alpha, "entropy_mean": 0.0 if a.name in ("V0", "P0") else 1.05, "argmax_share": alpha} for s in ("val_ar", "val_gs")}
        rec = {"config": a.name, "family": fam, "seed": seed, "dev": dev, "minutes": 0.01, "trainable_params": 3_690_000,
               "SYNTHETIC": True, "best_epoch": 2, "epochs": [{"epoch": e, "dev": dev - 0.01 * (2 - e), "diag": diag} for e in range(3)]}
        p = os.path.join(a.pp_out, "dev", a.name, f"{seed}.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(rec, open(p, "w"), indent=1)
        print(f"[search] {a.name} seed {seed}: dev {dev:.4f}", flush=True)
        return
    sysn = a.system
    partial = lambda: fake_preds(a.data, os.path.join(a.runs, "preds"), sysn, seed, 0.5, random.Random(0))
    maybe_fail(a, f"{a.kind}:{a.name}:{sysn}:{seed}", partial)
    acc = ACC.get(sysn, 0.77)
    fake_preds(a.data, os.path.join(a.runs, "preds"), sysn, seed, acc, random.Random(zlib.crc32(f'{sysn}{seed}'.encode())))
    for fn in os.listdir(os.path.join(a.runs, "preds", sysn, str(seed))):  # gate diagnostics in the format of diafuse_run.write_diag
        rows = list(csv.DictReader(open(os.path.join(a.data, fn), encoding="utf-8"), delimiter="\t"))
        d = os.path.join(a.runs, "diag", sysn, str(seed)); os.makedirs(d, exist_ok=True)
        al = (0.0, 1.0, 0.0) if sysn == "B12" else (0.3, 0.45, 0.25)
        with open(os.path.join(d, fn), "w") as f:
            f.write("id\talpha1\talpha2\talpha3\tplat\tmaxproto\n")
            for r in rows:
                f.write(f"{r['id']}\t{al[0]:.4f}\t{al[1]:.4f}\t{al[2]:.4f}\t0.5\t0.9\n")
    ck = os.path.join(a.ckpt, sysn, f"seed{seed}.pt"); os.makedirs(os.path.dirname(ck), exist_ok=True)
    open(ck, "wb").write(b"SYNTHETIC CHECKPOINT")
    rec = {"config": a.name, "seed": seed, "dev": dev_value(a.name if a.name != "FINAL" else "ALL", seed, arch) or 0.883,
           "trainable_params": 3_690_000, "checkpoint": ck, "SYNTHETIC": True, "minutes": 0.01}
    p = os.path.join(a.pp_out, "final", sysn, f"{seed}.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p, "w"), indent=1)
    print(f"[final] {sysn} seed {seed}: dev {rec['dev']:.4f}", flush=True)


if __name__ == "__main__":
    main()
