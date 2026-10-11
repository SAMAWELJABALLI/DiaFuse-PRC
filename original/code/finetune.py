#!/usr/bin/env python3
"""Fine-tuned baselines (GPU recommended).

B1 : XLM-R-large fully fine-tuned with cross-entropy.
B11: robustness/calibration baseline in the style of Shao et al. -- attention pooling,
     l2 consistency between clean and transformed views, post-hoc temperature scaling.
Both are trained on TSAC-AR train + its G_seen view and selected on val_ar + val_gs, and
write preds/<system>/<seed>/<set>.tsv in the format of diafuse_eval.py.

    python finetune.py --system B1 --seeds 0,1,2 --model models/xlmr
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translit import script_of  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r["id"] for r in rows], [r["text"] for r in rows], np.array([int(r["label"]) for r in rows])


class Clf(nn.Module):
    def __init__(self, src, attention_pool):
        super().__init__()
        self.enc = AutoModel.from_pretrained(src)
        h = self.enc.config.hidden_size
        self.attn = attention_pool
        self.q = nn.Parameter(torch.randn(h) / math.sqrt(h)) if attention_pool else None
        self.drop = nn.Dropout(0.1); self.out = nn.Linear(h, 2)

    def forward(self, **enc):
        H = self.enc(**enc).last_hidden_state
        if self.attn:
            s = (H @ self.q).masked_fill(enc["attention_mask"] == 0, -1e4)
            a = torch.softmax(s, -1)
            v = torch.einsum("bt,bth->bh", a, H)
        else:
            v = H[:, 0]
        return self.out(self.drop(v)), v


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def logits_of(model, tok, texts, max_len, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            out.append(model(**enc)[0].float())
    return torch.cat(out)


def fit_T(logits, y):
    y = torch.tensor(y, device=logits.device); best, bT = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        v = F.cross_entropy(logits / float(T), y).item()
        if v < best:
            best, bT = v, float(T)
    return bT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["B1", "B11"])
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="runs")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=64); ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    D = args.data
    tr_ids, ar_txt, ytr = read(f"{D}/train_ar.tsv")
    views = {v: read(f"{D}/train_{v}.tsv")[1] for v in ("gs", "inf", "noise", "cs") if os.path.exists(f"{D}/train_{v}.tsv")}
    train = [(t, int(y), i, "ar") for i, (t, y) in enumerate(zip(ar_txt, ytr))] + \
            [(t, int(y), i, "gs") for i, (t, y) in enumerate(zip(views["gs"], ytr))]
    dev = [read(f"{D}/{s}.tsv") for s in ("val_ar", "val_gs")]
    eval_sets = sorted(f[:-4] for f in os.listdir(D) if f.endswith(".tsv") and not f.startswith(("train", "val", "tt-")))
    tok = AutoTokenizer.from_pretrained(args.model)
    costs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        if os.path.exists(os.path.join(args.out, "preds", args.system, str(seed), "my.tsv")):
            continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = Clf(args.model, args.system == "B11").to(DEV)
        params = [{"params": model.enc.parameters(), "lr": args.lr},
                  {"params": [p for n, p in model.named_parameters() if not n.startswith("enc.")], "lr": 1e-3}]
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        bs = args.batch if args.system == "B1" else max(1, args.batch // 2)
        steps = args.epochs * math.ceil(len(train) / bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
        best, best_state = -1, None; t0 = time.time()
        for ep in range(args.epochs):
            model.train(); order = np.random.permutation(len(train))
            for b in range(0, len(order), bs):
                batch = [train[i] for i in order[b:b + bs]]
                texts = [x[0] for x in batch]; y = torch.tensor([x[1] for x in batch], device=DEV)
                with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                    enc = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(DEV)
                    logits, h = model(**enc)
                    loss = F.cross_entropy(logits.float(), y)
                    if args.system == "B11":  # l2 consistency with a transformed view of the same sentence
                        partners = []
                        for t, _, i, v in batch:
                            opts = ["ar"] if v != "ar" else []
                            opts += [k for k in views if k != v]
                            k = random.choice(opts)
                            partners.append(ar_txt[i] if k == "ar" else views[k][i])
                        enc2 = tok(partners, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(DEV)
                        logits2, h2 = model(**enc2)
                        loss = loss + F.cross_entropy(logits2.float(), y) + args.lam * ((h.float() - h2.float()) ** 2).sum(-1).mean() / h.shape[-1]
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            sc = np.mean([macro_f1(y, logits_of(model, tok, t, args.max_len).argmax(-1).cpu().numpy()) for _, t, y in dev])
            print(f"[{args.system}] seed {seed} epoch {ep}: dev macro-F1 {sc:.4f}", flush=True)
            if sc > best:
                best = sc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); costs.append((time.time() - t0) / 60)
        Ld = [logits_of(model, tok, t, args.max_len) for _, t, _ in dev]
        T = fit_T(torch.cat(Ld), np.concatenate([y for _, _, y in dev]))
        Tv = {}
        for k, (L, (_, _, y)) in enumerate(zip(Ld, dev)):
            Tk = fit_T(L, y); n = len(y)
            Tv[k] = math.exp((n * math.log(Tk) + 100 * math.log(T)) / (n + 100))
        for s in eval_sets:
            ids, texts, _ = read(f"{D}/{s}.tsv")
            L = logits_of(model, tok, texts, args.max_len)
            vt = torch.tensor([Tv[1] if script_of(t) == "lat" else Tv[0] for t in texts], device=DEV)[:, None]
            p_raw = torch.softmax(L, -1)[:, 1]; p_glob = torch.softmax(L / T, -1)[:, 1]; p_var = torch.softmax(L / vt, -1)[:, 1]
            pred = L.argmax(-1)
            p = os.path.join(args.out, "preds", args.system, str(seed), f"{s}.tsv"); os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write("id\tpred\tp_raw\tp_glob\tp_var\n")
                for i, a, b, c, d in zip(ids, pred.tolist(), p_raw.tolist(), p_glob.tolist(), p_var.tolist()):
                    f.write(f"{i}\t{a}\t{b:.6f}\t{c:.6f}\t{d:.6f}\n")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if costs:
        mp = os.path.join(args.out, "manual.json"); man = json.load(open(mp)) if os.path.exists(mp) else {}
        man[f"cost/{args.system}/train"] = f"{np.mean(costs):.1f}"
        json.dump(man, open(mp, "w"), indent=1)


if __name__ == "__main__":
    main()
