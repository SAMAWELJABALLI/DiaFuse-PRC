#!/usr/bin/env python3
"""Download everything DiaFuse-PRC needs, for offline runs on lab GPU servers.

    python download_models.py --models_dir models --raw_dir raw [--extra bgem3,marbert]

Models are written to <models_dir>/<short name>/ (loadable with from_pretrained(path));
corpora are written to <raw_dir>/<repo>/. Afterwards, on the offline machine:

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python extract.py --models_dir models ...
"""
import argparse
import io
import os
import subprocess
import urllib.request
import zipfile

MODELS = {
    "xlmr": "FacebookAI/xlm-roberta-large",
    "me5": "intfloat/multilingual-e5-large",
    "labse": "sentence-transformers/LaBSE",
}
EXTRA = {"bgem3": "BAAI/bge-m3", "marbert": "UBC-NLP/MARBERTv2", "qwen": "Qwen/Qwen2.5-7B",
         "qwen_instruct": "Qwen/Qwen2.5-7B-Instruct"}
CORPORA = {  # repository -> branch
    "fbougares/TSAC": "master",
    "chaymafourati/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset": "master",
    "SamiaTouileb/Narabizi": "main",
    "MouadJb/MYC": "main",
}
ALLOW = ["*.json", "*.txt", "*.model", "*.safetensors", "sentencepiece*", "tokenizer*", "vocab*",
         "*.py", "1_Pooling/*", "2_Dense/*", "modules.json"]


def download_models(models_dir, names):
    from huggingface_hub import snapshot_download
    for name in names:
        repo = {**MODELS, **EXTRA}[name]
        target = os.path.join(models_dir, name)
        print(f"[models] {repo} -> {target}", flush=True)
        try:
            snapshot_download(repo_id=repo, local_dir=target, allow_patterns=ALLOW)
        except Exception as e:  # some repos ship only pytorch_model.bin
            print(f"  safetensors-only download failed ({e}); retrying with *.bin", flush=True)
            snapshot_download(repo_id=repo, local_dir=target, allow_patterns=ALLOW + ["*.bin"])
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in os.listdir(target))
        if not has_weights:
            snapshot_download(repo_id=repo, local_dir=target, allow_patterns=ALLOW + ["*.bin"])


def download_corpora(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for repo, branch in CORPORA.items():
        name = repo.split("/")[1]
        target = os.path.join(raw_dir, name)
        if os.path.isdir(target):
            print(f"[data] {name} already present"); continue
        print(f"[data] {repo} -> {target}", flush=True)
        try:
            subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{repo}.git", target], check=True)
        except Exception:
            for br in (branch, "main", "master"):
                try:
                    data = urllib.request.urlopen(f"https://github.com/{repo}/archive/refs/heads/{br}.zip").read()
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        z.extractall(raw_dir)
                    os.rename(os.path.join(raw_dir, f"{name}-{br}"), target)
                    break
                except Exception:
                    continue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models_dir", default="models"); ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--extra", default="", help="comma list among: bgem3, marbert, qwen, qwen_instruct")
    ap.add_argument("--skip_models", action="store_true"); ap.add_argument("--skip_data", action="store_true")
    args = ap.parse_args()
    if not args.skip_models:
        names = list(MODELS) + [e for e in args.extra.split(",") if e]
        download_models(args.models_dir, names)
    if not args.skip_data:
        download_corpora(args.raw_dir)
    print("done")


if __name__ == "__main__":
    main()
