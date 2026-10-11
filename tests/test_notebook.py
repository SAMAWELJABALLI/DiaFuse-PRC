#!/usr/bin/env python3
"""Executes the notebook's code cells outside Colab (dry run): google.colab is stubbed, My Drive is a temporary
folder, pip is offline, the orchestrator runs with --dry_run on synthetic data.
Usage: test_notebook.py [tmpdir] [repo_url]
  without repo_url: a copy of the repository is placed in My Drive (Drive mode);
  with repo_url   : the notebook clones the repository from that URL (GitHub mode; a local file:// URL works)."""
import json, os, re, shutil, subprocess, sys, tempfile, types
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else tempfile.mkdtemp()
URL = sys.argv[2] if len(sys.argv) > 2 else ""
mydrive = "/content/drive/MyDrive"
if os.path.islink("/content/drive/MyDrive") or os.path.exists("/content/drive/MyDrive"):
    shutil.rmtree("/content/drive", ignore_errors=True)
shutil.rmtree("/content/DiaFuse-PRC-PP", ignore_errors=True); shutil.rmtree("/content/pp_pkg", ignore_errors=True)
os.makedirs(os.path.join(T, "MyDrive"), exist_ok=True); os.makedirs("/content/drive", exist_ok=True)
os.symlink(os.path.join(T, "MyDrive"), mydrive)
if not URL:
    shutil.copytree(REPO, os.path.join(mydrive, "DiaFuse-PRC-PP"), dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
arch = os.path.join(T, "synthetic_archive")
subprocess.run([sys.executable, os.path.join(REPO, "tests", "make_synthetic.py"), "--data",
                os.path.join(mydrive, "DiaFuse_PRCPP_RUN", "data"), "--archive", arch], check=True)
os.environ["PP_ORCHESTRATOR_EXTRA_ARGS"] = f"--dry_run --archive_dir {arch}"; os.environ["PIP_NO_INDEX"] = "1"
g = types.ModuleType("google"); c = types.ModuleType("google.colab"); d = types.ModuleType("google.colab.drive")
d.mount = lambda p: None; c.drive = d; g.colab = c; sys.modules.update({"google": g, "google.colab": c, "google.colab.drive": d})
nb = json.load(open(os.path.join(REPO, "notebooks", "DiaFuse_PRCPP_ONE_CLICK.ipynb")))
ns = {}
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        src = src.replace('GPU_REQUIRED     = True', 'GPU_REQUIRED     = False')
        src = re.sub(r'(?m)^REPO_URL\s*=.*$', f'REPO_URL = {URL!r}', src)  # test chooses Drive mode ('') or a clone URL
        exec(compile(src, "notebook_cell", "exec"), ns)
print("NOTEBOOK TEST exit_code =", ns.get("exit_code"))
