#!/usr/bin/env python3
"""B12'++ (experimental implementation), NOT B12' from the paper.

DiaFuse-PRC++ Setting B: encoders adapted (LoRA or full fine-tuning) under a DiaFuse-style objective.
The manuscript's B12' (Setting B, LoRA) was never executed and has no original implementation. This
script is PARTIAL with respect to its specification: LayerMix, the surface descriptor s(x), hierarchical
class x variety prototypes with kappa shrinkage, cell-rebalanced SupCon with a FIFO queue, hard mixup,
MMCE and logit adjustment are NOT implemented (models/B12_prime_diafuse_setting_B_lora/README.md).
Never executed on a GPU as of the initial repository release.

Why: B11 fine-tunes all ~560M parameters of XLM-R-large, whereas B12 (Setting A) trains ~3.7M head
parameters on frozen features. The manuscript pre-specifies a Setting B comparison (B12'); this script
is an experimental substitute for it, not that system.
This script follows finetune.py (B11) as closely as possible -- same data, views, dev sets, epochs,
max length, warmup schedule, consistency partners, temperature fitting and output format -- and
changes only what defines each variant (encoders, adaptation, fusion, head, auxiliary losses).

Modes (test sets are never read in `search`):
  search : python src/diafuse_pp_setting_B_lora.py search --configs configs/B12_setting_B_lora/screening_configs.json --only FT1 --seeds 0,1,2
  final  : python src/diafuse_pp_setting_B_lora.py final  --configs <run_dir>/configs/settingB_selected.json --only FINAL --system PPB --seeds 0-9
Models are read from --models_dir (download_models.py layout: models/xlmr, models/me5, models/labse).
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CODE = os.environ.get("DIAFUSE_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original", "code"))
sys.path.insert(0, CODE)
from translit import script_of  # noqa: E402


def _wjson(obj, p):
    """Atomic JSON write (file closed and flushed before the rename)."""
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(p + ".tmp", p)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREFIX = {"me5": "query: "}
_DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "B12_setting_B_lora", "defaults.json")
DEFAULTS = {k: v for k, v in json.load(open(_DEFAULTS_FILE)).items() if not k.startswith("_")}  # single source of truth


def load_configs(path):
    spec = json.load(open(path)); out = {}
    for name, c in spec["configs"].items():
        cfg = copy.deepcopy(DEFAULTS)
        for parent in c.get("inherit", []):
            cfg.update({k: v for k, v in spec["configs"][parent].items() if k not in ("inherit", "family", "why")})
        cfg.update({k: v for k, v in c.items() if k not in ("inherit", "family", "why")})
        cfg["_family"] = c.get("family", ""); out[name] = cfg
    return out


def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r["id"] for r in rows], [r["text"] for r in rows], np.array([int(r["label"]) for r in rows])


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def fit_T(logits, y):
    y = torch.tensor(y, device=logits.device); best, bT = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        v = F.cross_entropy(logits / float(T), y).item()
        if v < best:
            best, bT = v, float(T)
    return bT


# ------------------------------------------------------------------ LoRA (no external dependency)
class LoRALinear(nn.Module):
    def __init__(self, base_lin, r, alpha, dropout):
        super().__init__()
        self.base = base_lin
        self.A = nn.Parameter(torch.empty(r, base_lin.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base_lin.out_features, r))
        self.scale = alpha / r; self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model, r, alpha, dropout, targets):
    n = 0
    for _, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout)); n += 1
    return n


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(self, src, name, cfg):
        super().__init__()
        self.name = name; self.enc = AutoModel.from_pretrained(src); h = self.enc.config.hidden_size
        if cfg["adapt"] in ("lora", "frozen"):
            for p in self.enc.parameters():
                p.requires_grad = False
        if cfg["adapt"] == "lora":
            self.n_lora = add_lora(self.enc, cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"], set(cfg["lora_targets"]))
        self.pool = cfg["pool"]
        self.q = nn.Parameter(torch.randn(h) / math.sqrt(h)) if self.pool == "attn" else None
        self.h = h

    def forward(self, enc):
        H = self.enc(**enc).last_hidden_state
        m = enc["attention_mask"]
        if self.pool == "attn":
            a = torch.softmax((H @ self.q).masked_fill(m == 0, -1e4), -1)
            return torch.einsum("bt,bth->bh", a, H)
        if self.pool == "mean":
            mf = m.unsqueeze(-1).to(H.dtype)
            return (H * mf).sum(1) / mf.sum(1).clamp(min=1)
        return H[:, 0]


class FTModel(nn.Module):
    def __init__(self, cfg, models_dir):
        super().__init__()
        self.cfg = cfg; names = cfg["encoders"]; self.M = len(names)
        self.encs = nn.ModuleList([Encoder(os.path.join(models_dir, n), n, cfg) for n in names])
        self.toks = [AutoTokenizer.from_pretrained(os.path.join(models_dir, n)) for n in names]
        d = cfg["d"]
        if cfg["fusion"] == "single" and self.M == 1:
            self.proj = nn.ModuleList([nn.Identity()]); d = self.encs[0].h
        else:
            self.proj = nn.ModuleList([nn.Sequential(nn.Linear(e.h, d), nn.LayerNorm(d)) for e in self.encs])
        self.d = d
        if self.M > 1 and cfg["fusion"] in ("gate", "gated_residual"):
            self.gate = nn.Sequential(nn.Linear(self.M * d, 256), nn.ReLU(), nn.Linear(256, self.M))
            if cfg["gate_zero_init"]:
                nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        if self.M > 1 and cfg["fusion"] in ("static", "gated_residual"):
            self.static = nn.Linear(self.M * d, d)
        if cfg["head"] == "linear":
            self.body = nn.Dropout(cfg["head_dropout"])
        else:
            self.body = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg["head_dropout"]))
        self.out = nn.Linear(d, 2)
        self.proj_f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_h = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))

    def encode(self, texts, max_len):
        es = []
        for e, tok, pj in zip(self.encs, self.toks, self.proj):
            batch = [PREFIX.get(e.name, "") + (t if t.strip() else ".") for t in texts]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
            es.append(pj(e(enc).float()))
        return es

    def fuse(self, es):
        B = es[0].shape[0]; c = self.cfg
        if self.M == 1:
            return es[0], torch.ones(B, 1, device=DEV)
        E = torch.stack(es, 1); cat = torch.cat(es, -1)
        if c["fusion"] == "static":
            return self.static(cat), torch.full((B, self.M), 1.0 / self.M, device=DEV)
        logits = self.gate(torch.cat([F.layer_norm(e, e.shape[-1:]) for e in es], -1))
        if self.training and c["enc_dropout"] > 0:
            mask = torch.rand_like(logits) < c["enc_dropout"]
            full = mask.all(1)
            if full.any():
                mask[full.nonzero().flatten(), torch.randint(self.M, (int(full.sum()),), device=DEV)] = False
            logits = logits.masked_fill(mask, -1e9)
        a = torch.softmax(logits / c["gate_tau"], -1)
        if c["gate_floor"] > 0:
            a = (1 - c["gate_floor"]) * a + c["gate_floor"] / self.M
        z = torch.einsum("bm,bmd->bd", a, E)
        if c["fusion"] == "gated_residual":
            z = z + self.static(cat)
        return z, a

    def forward(self, texts, max_len):
        z, a = self.fuse(self.encode(texts, max_len))
        return self.out(self.body(z)), z, a

    def prototypes(self):
        return F.normalize(self.proj_h(self.out.weight), dim=-1)


@torch.no_grad()
def predict_logits(model, texts, max_len, bs=64):
    model.eval(); L, A = [], []
    for i in range(0, len(texts), bs):
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg, _, a = model(texts[i:i + bs], max_len)
        L.append(lg.float()); A.append(a.float())
    return torch.cat(L), torch.cat(A)


def train_one(cfg, seed, D, models_dir, resume_path=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr_ids, ar_txt, ytr = read(f"{D}/train_ar.tsv")
    views = {v: read(f"{D}/train_{v}.tsv")[1] for v in ("gs", "inf", "noise", "cs") if os.path.exists(f"{D}/train_{v}.tsv")}
    train = [(t, int(y), i, "ar") for i, (t, y) in enumerate(zip(ar_txt, ytr))] + \
            [(t, int(y), i, "gs") for i, (t, y) in enumerate(zip(views["gs"], ytr))]
    dev = [read(f"{D}/{s}.tsv") for s in cfg["dev_sets"]]
    model = FTModel(cfg, models_dir).to(DEV)
    enc_full = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encs.") and ".enc." in n
                and not re.search(r"\.(A|B)$", n)]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and re.search(r"\.(A|B)$", n)]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("encs.") and ".enc." in n)]
    groups = [g for g in ({"params": enc_full, "lr": cfg["lr_enc"]}, {"params": lora, "lr": cfg["lr_lora"]},
                          {"params": rest, "lr": cfg["lr_head"]}) if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"])
    bs = cfg["batch"]; steps = cfg["epochs"] * math.ceil(len(train) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    gated = model.M > 1 and cfg["fusion"] in ("gate", "gated_residual")
    best, best_state, log = -1, None, []
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    t0 = time.time(); start_ep = 0
    # epoch-level resume (LoRA / frozen encoders only: full fine-tuning states are too large for Drive)
    can_resume = resume_path is not None and cfg["adapt"] in ("lora", "frozen")
    if can_resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ck["params"]:
                    p.copy_(ck["params"][n].to(p.device))
        opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"]); scaler.load_state_dict(ck["scaler"])
        best, best_state, log, start_ep = ck["best"], ck["best_state"], ck["log"], ck["epoch"] + 1
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"  resumed from epoch {ck['epoch']} ({resume_path})", flush=True)
    for ep in range(start_ep, cfg["epochs"]):
        model.train(); order = np.random.permutation(len(train)); agg = []
        for b in range(0, len(order), bs):
            batch = [train[i] for i in order[b:b + bs]]
            texts = [x[0] for x in batch]; y = torch.tensor([x[1] for x in batch], device=DEV)
            partners = []
            for t, _, i, v in batch:  # same partner sampling as B11 (finetune.py)
                opts = (["ar"] if v != "ar" else []) + [k for k in views if k != v]
                k = random.choice(opts); partners.append(ar_txt[i] if k == "ar" else views[k][i])
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                lg, z, a = model(texts, cfg["max_len"])
                lg2, z2, a2 = model(partners, cfg["max_len"])
                lg, lg2, z, z2 = lg.float(), lg2.float(), z.float(), z2.float()
                loss = F.cross_entropy(lg, y)
                if cfg["partner_ce"]:
                    loss = loss + F.cross_entropy(lg2, y)
                loss = loss + cfg["lam_cons"] * ((z - z2) ** 2).sum(-1).mean() / z.shape[-1]
                if cfg["lam_js"] > 0:
                    p, q = torch.softmax(lg, -1), torch.softmax(lg2, -1); mm = 0.5 * (p + q)
                    js = 0.5 * (p * (torch.log(p + 1e-8) - torch.log(mm + 1e-8))).sum(-1) + 0.5 * (q * (torch.log(q + 1e-8) - torch.log(mm + 1e-8))).sum(-1)
                    loss = loss + cfg["lam_js"] * js.mean()
                if cfg["lam_p"] > 0:
                    P = model.prototypes(); zh = F.normalize(model.proj_f(torch.cat([z, z2])), dim=-1)
                    loss = loss + cfg["lam_p"] * F.cross_entropy(zh @ P.T / cfg["tau_p"], torch.cat([y, y]))
                if gated:
                    aa = torch.cat([a, a2]).float(); abar = aa.mean(0)
                    loss = loss + cfg["lam_lb"] * (abar * torch.log(abar * model.M + 1e-8)).sum()
                    if cfg["lam_ent"] > 0:
                        loss = loss + cfg["lam_ent"] * (aa * torch.log(aa + 1e-8)).sum(-1).mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            agg.append(float(loss))
        dev_scores, gate_stats = [], {}
        for (ids, t, yv), name in zip(dev, cfg["dev_sets"]):
            L, A = predict_logits(model, t, cfg["max_len"])
            dev_scores.append(macro_f1(yv, L.argmax(-1).cpu().numpy()))
            ent = -(A * torch.log(A + 1e-12)).sum(-1)
            gate_stats[name] = {"alpha_mean": A.mean(0).tolist(), "entropy_mean": float(ent.mean())}
        sc = float(np.mean(dev_scores))
        log.append({"epoch": ep, "loss": float(np.mean(agg)), "dev": sc, "dev_per_set": dev_scores, "gate": gate_stats})
        print(f"  epoch {ep}: loss {np.mean(agg):.4f} dev {sc:.4f}", flush=True)
        if sc > best:  # keep trainable parameters only (frozen weights do not change)
            best = sc; best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        if can_resume:
            tmp = resume_path + ".tmp"; os.makedirs(os.path.dirname(resume_path), exist_ok=True)
            torch.save({"params": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                        "opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                        "best": best, "best_state": best_state, "log": log, "py_rng": random.getstate(),
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
            os.replace(tmp, resume_path)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(p.device))
    info = {"dev": best, "epochs": log, "minutes": (time.time() - t0) / 60,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()), "n_trainable_tensors": len(trainable)}
    return model, dev, info


def cmd(args):
    cfgs = load_configs(args.configs); names = args.only.split(",") if args.only else list(cfgs)
    D = args.data
    for name in names:
        cfg = cfgs[name]; system = args.system if (args.system and len(names) == 1) else name
        for seed in parse_seeds(args.seeds):
            if args.cmd == "search":
                p = os.path.join(args.pp_out, "dev", name, f"{seed}.json")
            else:
                p = os.path.join(args.out, "preds", system, str(seed), "my.tsv")
            if os.path.exists(p):
                continue
            print(f"[{args.cmd}] {name} seed {seed}", flush=True)
            rp = os.path.join(args.ckpt_dir, "resume", args.cmd, system, f"seed{seed}.pt") if args.ckpt_dir else None
            model, dev, info = train_one(cfg, seed, D, args.models_dir, rp)
            rec = {"config": name, "family": cfg["_family"], "seed": seed, **info,
                   "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}}
            if args.cmd == "search":
                os.makedirs(os.path.dirname(p), exist_ok=True); _wjson(rec, p)
            else:
                Ld = [predict_logits(model, t, cfg["max_len"])[0] for _, t, _ in dev]
                T = fit_T(torch.cat(Ld), np.concatenate([yv for _, _, yv in dev]))
                Tv = {}
                for k, (L, (_, _, yv)) in enumerate(zip(Ld, dev)):
                    Tk = fit_T(L, yv); nk = len(yv)
                    Tv[k] = math.exp((nk * math.log(Tk) + 100 * math.log(T)) / (nk + 100))
                eval_sets = sorted(f[:-4] for f in os.listdir(D) if f.endswith(".tsv") and not f.startswith(("train", "val", "tt-")))
                for s in eval_sets:
                    ids, texts, _ = read(f"{D}/{s}.tsv")
                    L, A = predict_logits(model, texts, cfg["max_len"])
                    vt = torch.tensor([Tv[min(1, len(Tv) - 1)] if script_of(t) == "lat" else Tv[0] for t in texts], device=DEV)[:, None]
                    p_raw = torch.softmax(L, -1)[:, 1]; p_glob = torch.softmax(L / T, -1)[:, 1]; p_var = torch.softmax(L / vt, -1)[:, 1]
                    pred = L.argmax(-1)
                    q = os.path.join(args.out, "preds", system, str(seed), f"{s}.tsv"); os.makedirs(os.path.dirname(q), exist_ok=True)
                    with open(q, "w", encoding="utf-8") as f:
                        f.write("id\tpred\tp_raw\tp_glob\tp_var\n")
                        for i, a_, b_, c_, d_ in zip(ids, pred.tolist(), p_raw.tolist(), p_glob.tolist(), p_var.tolist()):
                            f.write(f"{i}\t{a_}\t{b_:.6f}\t{c_:.6f}\t{d_:.6f}\n")
                    g = os.path.join(args.out, "diag", system, str(seed), f"{s}.tsv"); os.makedirs(os.path.dirname(g), exist_ok=True)
                    with open(g, "w", encoding="utf-8") as f:
                        f.write("id\t" + "\t".join(f"alpha{m + 1}" for m in range(A.shape[1])) + "\n")
                        for i, row in zip(ids, A.tolist()):
                            f.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")
                if args.ckpt_dir:  # trainable parameters of the selected epoch (LoRA, gate, heads; full encoder if adapt=full)
                    cp = os.path.join(args.ckpt_dir, system, f"seed{seed}.pt"); os.makedirs(os.path.dirname(cp), exist_ok=True)
                    torch.save({"trainable": {n: q.detach().cpu() for n, q in model.named_parameters() if q.requires_grad},
                                "config": rec["cfg"], "seed": seed, "T": T, "Tv": Tv}, cp)
                    rec["checkpoint"] = cp
                fp = os.path.join(args.pp_out, "final", system, f"{seed}.json"); os.makedirs(os.path.dirname(fp), exist_ok=True)
                _wjson({**rec, "T": T, "Tv": Tv}, fp)
            if rp and os.path.exists(rp):
                os.remove(rp)
            print(f"[{args.cmd}] {name} seed {seed}: dev {info['dev']:.4f} ({info['minutes']:.1f} min, "
                  f"{info['trainable_params'] / 1e6:.1f}M trainable)", flush=True)
            del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "final"])
    ap.add_argument("--configs", required=True); ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--system", default=None)
    ap.add_argument("--data", default="data"); ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", default="runs"); ap.add_argument("--pp_out", default="runs_pp")
    ap.add_argument("--ckpt_dir", default=None)
    cmd(ap.parse_args())


if __name__ == "__main__":
    main()
