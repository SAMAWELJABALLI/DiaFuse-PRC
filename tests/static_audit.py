#!/usr/bin/env python3
"""Static audit of the one-click package (no GPU, no network, nothing executed except compilation/AST parsing).
Checks: every .py compiles; notebook JSON valid and code cells compile; no absolute user paths in shipped code;
no input(); every import resolvable (stdlib / requirements_colab.txt / torch preinstalled on Colab / local module);
every script path referenced by the orchestrator exists; configs are valid JSON; original code MD5 unchanged.
Writes tests/STATIC_AUDIT_REPORT.md and exits 1 if any check fails."""
import ast
import hashlib
import json
import os
import re
import sys

PP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
ROOT = PP
ORIG = os.path.join(ROOT, "original", "code")
STDLIB = set(sys.stdlib_module_names)
# pip distribution -> import names
REQ_IMPORTS = {"transformers": ["transformers"], "sentencepiece": ["sentencepiece"], "huggingface-hub": ["huggingface_hub"],
               "scikit-learn": ["sklearn"], "scipy": ["scipy"], "pandas": ["pandas"], "matplotlib": ["matplotlib"],
               "umap-learn": ["umap"], "deepsig": ["deepsig"]}
COLAB_PREINSTALLED = {"torch", "numpy", "google", "IPython", "tqdm", "requests", "safetensors", "tokenizers", "packaging", "yaml"}
OPTIONAL_GUARDED = {"deepsig", "umap", "google"}  # imported inside try/except or only on Colab

res = []


def check(name, ok, detail=""):
    res.append((name, bool(ok), detail))


def py_files(base, skip_tests=False):
    for r, ds, fs in os.walk(base):
        ds[:] = [d for d in ds if d != "__pycache__"]
        if skip_tests and os.path.basename(r) == "tests":
            continue
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(r, f)


def local_modules():
    mods = set()
    for base in (PP, ORIG):
        for p in py_files(base):
            if p.startswith(os.path.join(PP, "original")) and base == PP:
                continue
            mods.add(os.path.splitext(os.path.basename(p))[0])
    return mods


def requirements():
    allowed = set()
    for line in open(os.path.join(PP, "configs", "requirements_colab.txt")):
        line = line.split("#")[0].strip()
        if not line:
            continue
        dist = re.split(r"[=<>!~\[ ]", line)[0].strip()
        allowed.update(REQ_IMPORTS.get(dist, [dist.replace("-", "_")]))
    return allowed


def imports_of(src):
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
    return out


def main():
    loc = local_modules(); req = requirements()
    allowed = STDLIB | req | COLAB_PREINSTALLED | loc
    # 1. compilation + imports + forbidden patterns
    bad_compile, bad_import, bad_path, bad_input = [], [], [], []
    for base, shipped in ((PP, True), (ORIG, False)):
        for p in py_files(base):
            if base == PP and p.startswith(os.path.join(PP, "original") + os.sep):
                continue
            rel = os.path.relpath(p, ROOT); src = open(p, encoding="utf-8").read()
            try:
                compile(src, p, "exec")
            except SyntaxError as e:
                bad_compile.append(f"{rel}: {e}"); continue
            for m in imports_of(src):
                if m not in allowed:
                    bad_import.append(f"{rel}: {m}")
            if shipped and "/tests/" not in p:
                for pat in ("/home/", "/Users/", "C:\\\\", "/root/"):
                    if pat in src:
                        bad_path.append(f"{rel}: '{pat}'")
                for node in ast.walk(ast.parse(src)):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "input":
                        bad_input.append(f"{rel}:{node.lineno}")
    check("all .py files compile", not bad_compile, "; ".join(bad_compile))
    check("all imports resolvable (stdlib / requirements / Colab preinstalled / local)", not bad_import, "; ".join(bad_import))
    check("no absolute user paths in shipped code", not bad_path, "; ".join(bad_path))
    check("no input() in shipped code", not bad_input, "; ".join(bad_input))
    # 2. notebook
    nbp = os.path.join(ROOT, "notebooks", "DiaFuse_PRCPP_ONE_CLICK.ipynb")
    try:
        nb = json.load(open(nbp)); cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        errs = []
        for i, c in enumerate(cells):
            src = "".join(c["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
            try:
                compile(src, f"cell{i}", "exec")
            except SyntaxError as e:
                errs.append(f"cell {i}: {e}")
            if "input(" in src:
                errs.append(f"cell {i}: input()")
        check("notebook JSON valid, code cells compile, no input()", not errs, "; ".join(errs) or f"{len(cells)} code cells")
        allsrc = "".join("".join(c["source"]) for c in cells)
        check("notebook calls run_one_click.py and mounts Drive", "run_one_click.py" in allsrc and "drive.mount" in allsrc)
        paths = ["scripts/run_one_click.py", "original/code/MD5SUMS_unmodified.txt", "configs/protocol.json", "configs/requirements_colab.txt"]
        miss = [x for x in paths if x.split("/")[-1] not in allsrc and x not in allsrc or not os.path.exists(os.path.join(ROOT, x))]
        check("paths used by the notebook exist in the repository", not miss, ", ".join(miss) or ", ".join(paths))
    except Exception as e:
        check("notebook JSON valid", False, repr(e))
    # 3. referenced scripts exist
    orch = open(os.path.join(PP, "scripts", "run_one_click.py")).read()
    names = set(re.findall(r'["\']([A-Za-z0-9_]+\.py)["\']', orch))
    missing = [n for n in sorted(names) if not any(os.path.exists(os.path.join(d, n)) for d in
               (ORIG, os.path.join(PP, "src"), os.path.join(PP, "scripts"), os.path.join(PP, "evaluation"), os.path.join(PP, "tests")))]
    check("every script named by the orchestrator exists", not missing, ", ".join(missing) or ", ".join(sorted(names)))
    # 4. configs
    bad = []
    for r_, _, fs in os.walk(os.path.join(PP, "configs")):
      for f in sorted(fs):
        if f.endswith(".json"):
            try:
                json.load(open(os.path.join(r_, f)))
            except Exception as e:
                bad.append(f"{f}: {e}")
    check("configs are valid JSON", not bad, "; ".join(bad))
    P = json.load(open(os.path.join(PP, "configs", "protocol.json")))
    t5 = json.dumps(P.get("table5", {}))
    check("Table 5 mapping explicit (EM pairs S0-S1, S0-S2, S5AR-S5)", all(x in t5 for x in ("S1", "S2", "S5AR")), t5[:200])
    check("seeds: final 10, ablation 5", len(P["seeds"]["final"]) == 10 and len(P["seeds"]["ablation"]) == 5, json.dumps(P["seeds"]))
    # 4b. one identifiable implementation per model
    need = {"B11 code": "original/code/finetune.py", "B12 Setting A code (original)": "original/code/diafuse_run.py",
            "synthetic Arabizi (original)": "original/code/translit.py", "data preparation (original)": "original/code/prep_data.py",
            "evaluation (original)": "original/code/diafuse_eval.py", "DiaFuse-PRC++ Setting A code": "src/diafuse_pp_setting_A.py",
            "DiaFuse-PRC++ Setting B / LoRA code": "src/diafuse_pp_setting_B_lora.py",
            "Setting A configs": "configs/B12_setting_A/defaults.json", "Setting A screening configs": "configs/B12_setting_A/screening_configs.json",
            "Setting B configs": "configs/B12_setting_B_lora/defaults.json", "Setting B screening configs": "configs/B12_setting_B_lora/screening_configs.json",
            "B11 config note": "configs/B11/README.md", "B11 model card": "models/B11_robust_calibrated/README.md",
            "B12 model card": "models/B12_diafuse_setting_A/README.md", "B12++ model card": "models/B12pp_diafuse_pp_setting_A/README.md",
            "B12' model card": "models/B12_prime_diafuse_setting_B_lora/README.md", "training/orchestration entry point": "scripts/run_one_click.py",
            "evaluation entry point": "evaluation/evaluate.py", "notebook": "notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb"}
    miss = [f"{k} ({v})" for k, v in need.items() if not os.path.exists(os.path.join(ROOT, v))]
    check("code, configs and model cards present for B11, B12 (A), B12++ (A), B12'++ (B, LoRA)", not miss, "; ".join(miss))
    # 5. original code unchanged
    md5f = os.path.join(ROOT, "original", "code", "MD5SUMS_unmodified.txt"); diff = []
    for line in open(md5f):
        if line.strip():
            h, fn = line.split()[:2]; fn = fn.lstrip("*")
            if hashlib.md5(open(os.path.join(ORIG, fn), "rb").read()).hexdigest() != h:
                diff.append(fn)
    check("original code identical to MD5SUMS_unmodified.txt", not diff, ", ".join(diff))
    # report
    ok = all(r[1] for r in res)
    L = ["# Static audit of the one-click package", "", f"Overall: **{'PASS' if ok else 'FAIL'}**", "", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if o else 'FAIL'} | {d.replace('|', '/')[:300]} |" for n, o, d in res]
    L += ["", "Not covered by a static audit: actual PyTorch training/extraction, model downloads, GPU memory behaviour, Google Drive I/O."]
    open(os.path.join(PP, "tests", "STATIC_AUDIT_REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
