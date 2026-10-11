"""Cache per-layer pooled sentence vectors (Eq. 1) for every encoder and every set.

For each encoder m and set S writes feats/<m>/<S>.npy with shape [N, L', 2 d_m] in float16
(mean over non-padding tokens || first-token state, for every `--layer_stride`-th layer,
always including the embedding layer 0 and the last layer), and feats/<m>/<S>.frag.npy with
two fragmentation statistics per sentence used by the surface descriptor s(x).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ENCODERS = {
    "xlmr": "FacebookAI/xlm-roberta-large",
    "me5": "intfloat/multilingual-e5-large",
    "labse": "sentence-transformers/LaBSE",
    # optional members (Section 5.4 / Appendix C)
    "bgem3": "BAAI/bge-m3",
    "marbert": "UBC-NLP/MARBERTv2",
    # cost analysis only (Table 17): timed, never stored or trained on
    "qwen": "Qwen/Qwen2.5-7B",
}
PREFIX = {"me5": "query: "}


def read_texts(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


@torch.no_grad()
def encode(model, tok, texts, prefix, max_len, batch, device, layers):
    out, frag = [], []
    model.eval()
    wcache = {}

    def n_pieces(w):
        if w not in wcache:
            wcache[w] = len(tok.tokenize(w))
        return wcache[w]
    order = np.argsort([len(t) for t in texts])  # length bucketing for speed
    res = [None] * len(texts); fr = [None] * len(texts)
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        chunk = [prefix + (texts[i] if texts[i].strip() else ".") for i in idx]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = []
        for l in layers:
            h = hs[l].float()
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(torch.cat([mean, h[:, 0]], -1))
        arr = torch.stack(vecs, 1).half().cpu().numpy()
        for j, i in enumerate(idx):
            res[i] = arr[j]
            words = texts[i].split() or ["."]
            pieces = [n_pieces(w) for w in words[:64]]
            fr[i] = (float(np.mean([p > 1 for p in pieces])), float(np.mean(pieces)))
    return np.stack(res), np.array(fr, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="feats")
    ap.add_argument("--encoders", default="xlmr,me5,labse")
    ap.add_argument("--models_dir", default=None, help="local snapshots written by download_models.py")
    ap.add_argument("--layer_stride", type=int, default=2); ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--sets", default=None)
    ap.add_argument("--cost_only", action="store_true", help="time 1k sentences and batch-1 latency, store nothing")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = args.sets.split(",") if args.sets else sorted(f[:-4] for f in os.listdir(args.data) if f.endswith(".tsv"))
    timing = {}
    for name in args.encoders.split(","):
        src = os.path.join(args.models_dir, name) if args.models_dir else ENCODERS[name]
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {"torch_dtype": torch.float16} if (args.cost_only and device.type == "cuda") else {}
        model = AutoModel.from_pretrained(src, **kw).to(device)
        if args.cost_only:
            texts = read_texts(os.path.join(args.data, "train_ar.tsv"))[:1000]
            L = model.config.num_hidden_layers
            layers = sorted(set(list(range(0, L + 1, args.layer_stride)) + [L]))
            t0 = time.time(); encode(model, tok, texts, "", args.max_len, args.batch, device, layers)
            sec = 1000 * (time.time() - t0) / len(texts)
            t0 = time.time()
            with torch.no_grad():
                for t in texts[:100]:
                    model(**tok([t], return_tensors="pt", truncation=True, max_length=args.max_len).to(device))
            lat = 1000 * (time.time() - t0) / 100
            odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
            json.dump({"layers": layers, "dim": int(model.config.hidden_size), "sec_per_1k": sec, "latency_ms": lat,
                       "params": int(sum(p.numel() for p in model.parameters())), "cost_only": True},
                      open(os.path.join(odir, "meta.json"), "w"), indent=1)
            print(f"[{name}] cost only: {sec:.1f} s/1k, {lat:.1f} ms/ex"); del model; continue
        L = model.config.num_hidden_layers
        layers = sorted(set(list(range(0, L + 1, args.layer_stride)) + [L]))
        odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
        n_done, t_done = 0, 0.0
        for s in sets:
            p = os.path.join(odir, f"{s}.npy")
            if os.path.exists(p):
                continue
            texts = read_texts(os.path.join(args.data, f"{s}.tsv"))
            t0 = time.time()
            U, F = encode(model, tok, texts, PREFIX.get(name, ""), args.max_len, args.batch, device, layers)
            t_done += time.time() - t0; n_done += len(texts)
            np.save(p, U); np.save(os.path.join(odir, f"{s}.frag.npy"), F)
            print(f"[{name}] {s}: {U.shape} in {time.time() - t0:.1f}s", flush=True)
        meta = {"layers": layers, "dim": int(model.config.hidden_size), "source": src,
                "params": int(sum(p.numel() for p in model.parameters()))}
        # single-example inference latency (batch size 1), on in-distribution test sentences
        lat_texts = read_texts(os.path.join(args.data, "ar.tsv"))[:100] if os.path.exists(os.path.join(args.data, "ar.tsv")) else []
        if lat_texts:
            with torch.no_grad():
                t0 = time.time()
                for t in lat_texts:
                    enc = tok([PREFIX.get(name, "") + t], truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                    model(**enc, output_hidden_states=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                meta["latency_ms"] = 1000 * (time.time() - t0) / len(lat_texts)
        if n_done:
            meta["sec_per_1k"] = 1000 * t_done / n_done
        old = os.path.join(odir, "meta.json")
        if os.path.exists(old) and "sec_per_1k" in json.load(open(old)) and not n_done:
            meta["sec_per_1k"] = json.load(open(old))["sec_per_1k"]
        json.dump(meta, open(old, "w"), indent=1)
        timing[name] = meta.get("sec_per_1k")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("extraction s/1k:", timing)


if __name__ == "__main__":
    main()
