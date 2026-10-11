#!/usr/bin/env python3
"""Train and evaluate DiaFuse-PRC and its baselines on the Tunisian/Arabizi protocol.

Inputs : data/*.tsv (prep_data.py) and feats/<encoder>/<set>.npy (extract.py).
Outputs: preds/<system>/<seed>/<set>.tsv, diag/B12/<seed>/<set>.tsv, emb/<system>/<set>.npy,
         manual.json (counts and costs) -- exactly the inputs of diafuse_eval.py.

    python diafuse_run.py --mode quick            # 3 seeds main, 2 seeds ablations
    python diafuse_run.py --mode full             # 10 / 5 seeds, tuning budget of 20 configs
    python diafuse_run.py --only B6,B12           # subset of systems
Runs whose prediction files already exist are skipped (resumable).
"""
import argparse
import copy
import csv
import itertools
import json
import math
import os
import random
import re
import sys
import time
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translit import script_of  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIEWS = {"AR": "train_ar", "LAT": "train_gs", "INF": "train_inf", "NOISE": "train_noise",
         "CS": "train_cs", "GU": "train_gu"}
FAMILY_OF = {"LAT": "TR", "AR": "TR", "GU": "TR", "INF": "INF", "NOISE": "NOISE", "CS": "CS"}
DEFAULT_POOL = {"xlmr": "mean", "me5": "mean", "labse": "tok", "bgem3": "tok", "marbert": "tok"}
HP = dict(d=512, ds=16, tau_beta0=1.0, tau_beta1=0.5, layer_drop=0.1, gate_hidden=256, tau_g=1.0,
          lam_g=0.01, tau=0.1, lam_cl=0.5, gamma=1.0, rho=0.5, n_pos=10, n_neg=500, pool=256,
          mu=0.9, kappa=50.0, tau_p=0.1, lam_p=0.5, k_hard=20, beta_a=0.5, lam_hm=0.5, n_mix=8,
          eta=1.0, lam_cons=0.5, tau_la=1.0, lam_cal=0.1, kappa_T=100.0, lr=1e-3, wd=5e-4,
          batch=64, epochs=30, p1=3, p2=13)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# =========================================================================== data
def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


class Store:
    def __init__(self, data_dir, feats_dir, encoders, layer_stride_check=True):
        self.data_dir, self.feats_dir, self.encoders = data_dir, feats_dir, encoders
        self._rows, self._feat, self._surf = {}, {}, {}
        self.meta = {e: json.load(open(os.path.join(feats_dir, e, "meta.json"))) for e in encoders}

    def rows(self, s):
        if s not in self._rows:
            self._rows[s] = read_tsv(os.path.join(self.data_dir, f"{s}.tsv"))
        return self._rows[s]

    def ids(self, s):
        return [r["id"] for r in self.rows(s)]

    def y(self, s):
        return np.array([int(r["label"]) for r in self.rows(s)])

    def texts(self, s):
        return [r["text"] for r in self.rows(s)]

    def feat(self, s, enc):
        k = (s, enc)
        if k not in self._feat:
            arr = np.load(os.path.join(self.feats_dir, enc, f"{s}.npy"), mmap_mode="r")
            if s.startswith(("train", "val")):  # training/validation views are read at every step: keep in RAM
                arr = np.ascontiguousarray(arr)
            self._feat[k] = arr
        return self._feat[k]

    def has(self, s):
        return os.path.exists(os.path.join(self.data_dir, f"{s}.tsv")) and all(
            os.path.exists(os.path.join(self.feats_dir, e, f"{s}.npy")) for e in self.encoders)

    def pooled(self, s, enc, layer=-1, pool=None):
        U = torch.from_numpy(np.array(self.feat(s, enc)[:, layer], dtype=np.float32))
        d = U.shape[-1] // 2
        pool = pool or DEFAULT_POOL.get(enc, "mean")
        return U[:, :d] if pool == "mean" else U[:, d:]

    def surface(self, s):
        """s(x): 16 cheap surface features (Section 4.3)."""
        if s in self._surf:
            return self._surf[s]
        frags = [np.load(os.path.join(self.feats_dir, e, f"{s}.frag.npy")) for e in self.encoders[:3]]
        while len(frags) < 3:
            frags.append(np.zeros_like(frags[0]))
        out = []
        for i, t in enumerate(self.texts(s)):
            chars = [c for c in t if not c.isspace()] or ["."]
            n = len(chars)
            ar = sum(1 for c in chars if "؀" <= c <= "ۿ") / n
            la = sum(1 for c in chars if c.isascii() and c.isalpha()) / n
            dg = sum(c.isdigit() for c in chars) / n
            toks = t.split() or ["."]
            dl = np.mean([bool(re.search(r"[a-z]", w.lower())) and bool(re.search(r"[235789]", w)) for w in toks])
            acc = np.mean([any(ord(c) > 127 and c.isalpha() and not ("؀" <= c <= "ۿ") for c in w) for w in toks])
            cnt = np.unique(chars, return_counts=True)[1] / n
            ent = float(-(cnt * np.log(cnt)).sum())
            punct = sum(unicodedata.category(c).startswith("P") for c in chars) / n
            rep = np.mean([bool(re.search(r"(.)\1\1", w)) for w in toks])
            f = [ar, la, dg, dl, acc, min(np.mean([len(w) for w in toks]) / 10, 2), min(len(toks) / 50, 1),
                 ent / 5, punct, rep]
            f += [frags[m][i, 0] for m in range(3)] + [min(frags[m][i, 1] / 5, 2) for m in range(3)]
            out.append(f)
        self._surf[s] = torch.tensor(np.array(out, dtype=np.float32))
        return self._surf[s]


class TrainData:
    """Labelled training examples (AR and LAT_syn views) and their consistency partners."""

    def __init__(self, store, syn=True, r_syn=1.0, ir_c=None, twogen=False, families=("TR", "INF", "NOISE", "CS"),
                 seed=0, arabic_only=False):
        rng = np.random.default_rng(1000 + seed)
        y = store.y("train_ar"); n = len(y)
        keep = np.ones(n, bool)
        if ir_c:  # subsample the rarer class so that majority/minority = ir_c
            counts = np.bincount(y, minlength=2); minority = int(np.argmin(counts))
            target = int(counts.max() / ir_c)
            idx = np.flatnonzero(y == minority)
            drop = rng.choice(idx, size=max(0, len(idx) - target), replace=False)
            keep[drop] = False
        src = np.flatnonzero(keep)
        syn_mask = np.zeros(n, bool)
        if syn and not arabic_only:
            k = int(round(r_syn * len(src)))
            syn_mask[rng.choice(src, size=k, replace=False)] = True
        self.items = [("AR", int(i)) for i in src]
        self.items += [("LAT", int(i)) for i in src if syn_mask[i]]
        if twogen:
            self.items += [("GU", int(i)) for i in src if syn_mask[i]]
        self.y_src = y
        self.variety = np.array([0 if v == "AR" else 1 for v, _ in self.items])
        self.label = np.array([y[i] for _, i in self.items])
        self.views_avail = {"AR": np.ones(n, bool), "LAT": syn_mask, "GU": syn_mask if twogen else np.zeros(n, bool),
                            "INF": syn_mask, "NOISE": syn_mask, "CS": syn_mask}
        self.families = set(families)
        self.n_var = 2 if (syn and not arabic_only) else 1
        cells = np.zeros((2, 2))
        for c, v in zip(self.label, self.variety):
            cells[c, v] += 1
        self.cell_counts = cells
        self.prior = np.bincount(self.label, minlength=2) / len(self.label)

    def partner(self, view, i, rng):
        """A transformed view of the same source sentence from an active family (q = 1)."""
        opts = [v for v in VIEWS if v != view and self.views_avail[v][i] and FAMILY_OF[v] in self.families]
        return (opts[rng.integers(len(opts))], i) if opts else None


def stack_layers(store, s, encs, idx=None, layer_mode="both"):
    out = []
    for e in encs:
        U = store.feat(s, e)
        if idx is not None:
            ii = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
            U = U[ii]
        U = torch.from_numpy(np.array(U))
        d = U.shape[-1] // 2
        if layer_mode == "last":
            U = U[:, -1:]
        elif layer_mode == "mean":
            U = U[..., :d]
        elif layer_mode == "tok":
            U = U[..., d:]
        out.append(U)
    return out


# =========================================================================== DiaFuse-PRC
class LayerMix(nn.Module):
    def __init__(self, L, D, d, indepA=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(L, D)); self.b = nn.Parameter(torch.zeros(L, d))
        self.omega = nn.Parameter(torch.zeros(L)); self.indepA = indepA
        if indepA:
            self.A = nn.Parameter(torch.randn(L, D, d) / math.sqrt(D))
        else:
            self.A = nn.Linear(D, d, bias=False)

    def forward(self, U, tau_beta, drop=0.0):
        U = F.layer_norm(U, U.shape[-1:]) * self.gamma
        P = (torch.einsum("bld,ldk->blk", U, self.A) if self.indepA else self.A(U)) + self.b
        logits = self.omega / tau_beta
        if self.training and drop > 0 and U.shape[1] > 1:
            mask = torch.rand(U.shape[1], device=U.device) < drop
            if mask.all():
                mask[torch.randint(U.shape[1], (1,))] = False
            logits = logits.masked_fill(mask, -1e9)
        beta = torch.softmax(logits, 0)
        return torch.einsum("l,bld->bd", beta, P), beta


class DiaFuse(nn.Module):
    def __init__(self, shapes, flags, C=2):
        super().__init__()
        d, ds = HP["d"], HP["ds"]
        self.flags = flags; self.M = len(shapes)
        self.mix = nn.ModuleList([LayerMix(L, D, d, flags.get("indepA", False)) for L, D in shapes])
        gin = {"both": self.M * d + ds, "e": self.M * d, "s": ds}[flags.get("gate_input", "both")]
        if flags["fusion"] == "gate" and self.M > 1:
            self.gate = nn.Sequential(nn.Linear(gin, HP["gate_hidden"]), nn.ReLU(), nn.Linear(HP["gate_hidden"], self.M))
        elif flags["fusion"] == "static" and self.M > 1:
            self.static = nn.Linear(self.M * d, d)
        self.W = nn.Linear(d, C)
        self.proj_f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_h = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, Us, s, tau_beta=0.5):
        es = [m(U.to(DEV).float(), tau_beta, HP["layer_drop"])[0] for m, U in zip(self.mix, Us)]
        B = es[0].shape[0]
        if self.M == 1:
            return es[0], torch.ones(B, 1, device=DEV), es
        if self.flags["fusion"] == "static":
            return self.static(torch.cat(es, -1)), torch.full((B, self.M), 1.0 / self.M, device=DEV), es
        gi = self.flags.get("gate_input", "both")
        parts = []
        if gi in ("both", "e"):
            parts += [F.layer_norm(e, e.shape[-1:]) for e in es]
        if gi in ("both", "s"):
            parts.append(s.to(DEV))
        alpha = torch.softmax(self.gate(torch.cat(parts, -1)) / HP["tau_g"], -1)
        z = torch.einsum("bm,bmd->bd", alpha, torch.stack(es, 1))
        return z, alpha, es

    def prototypes(self):
        return F.normalize(self.proj_h(self.W.weight), dim=-1)


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mmce(conf, correct, width=0.4):
    k = torch.exp(-torch.abs(conf[:, None] - conf[None, :]) / width)
    r = (correct - conf)
    return (r[:, None] * r[None, :] * k).mean()


class DiaFuseTrainer:
    """One training run of DiaFuse-PRC (or of a SharpReCL-style variant, through flags)."""

    def __init__(self, store, td, flags, seed, encs, input_mode="layers"):
        self.store, self.td, self.flags, self.seed, self.encs = store, td, flags, seed, encs
        self.input_mode = input_mode  # "layers" or "pooled" (B10: concatenated default vectors)
        set_seed(seed)
        shapes = [(U.shape[1], U.shape[2]) for U in self._inputs("train_ar", [0])]
        self.model = DiaFuse(shapes, flags).to(DEV)
        self.log_prior = torch.log(torch.tensor(td.prior, dtype=torch.float32, device=DEV).clamp(min=1e-6))
        V = td.n_var
        self.delta = torch.zeros(2, 2, 128, device=DEV)
        cnt = torch.tensor(td.cell_counts, dtype=torch.float32, device=DEV)
        kappa = flags.get("kappa", HP["kappa"])
        self.r_cv = cnt / (cnt + kappa) if math.isfinite(kappa) else torch.zeros_like(cnt)
        self.cell_prior = cnt / cnt.sum()

    # ---------------------------------------------------------------- inputs
    def _inputs(self, s, idx=None):
        if self.input_mode == "pooled":
            X = torch.cat([self.store.pooled(s, e) for e in self.encs], -1)
            X = X if idx is None else X[idx]
            return [X.unsqueeze(1)]
        return stack_layers(self.store, s, self.encs, idx, self.flags.get("layer_mode", "both"))

    def _surf(self, s, idx):
        S = self.store.surface(s)
        return S[idx] if idx is not None else S

    def forward_items(self, items, tau_beta, train=True):
        """items: list of (view, row). Groups by view to gather features."""
        by = {}
        for k, (v, i) in enumerate(items):
            by.setdefault(v, []).append((k, i))
        zs, als = [None] * len(items), [None] * len(items)
        self.model.train(train)
        for v, lst in by.items():
            pos = [k for k, _ in lst]; rows = torch.tensor([i for _, i in lst])
            z, a, _ = self.model(self._inputs(VIEWS[v], rows), self._surf(VIEWS[v], rows), tau_beta)
            for j, k in enumerate(pos):
                zs[k] = z[j]; als[k] = a[j]
        return torch.stack(zs), torch.stack(als)

    def encode_set(self, s, tau_beta, bs=512):
        self.model.eval(); n = len(self.store.ids(s)); Z, A = [], []
        with torch.no_grad():
            for i in range(0, n, bs):
                idx = torch.arange(i, min(n, i + bs))
                z, a, _ = self.model(self._inputs(s, idx), self._surf(s, idx), tau_beta)
                Z.append(z); A.append(a)
        return torch.cat(Z), torch.cat(A)

    # ---------------------------------------------------------------- losses
    def ce_la(self, logits, y):
        return F.cross_entropy(logits + HP["tau_la"] * self.log_prior, y)

    def proto_cv(self):
        p = self.model.prototypes()  # [C,128]
        mode = self.flags.get("proto", "hier")
        if mode == "class" or self.td.n_var == 1:
            return p[:, None, :].expand(2, 2, 128)
        pcv = p[:, None, :] + self.r_cv[..., None] * self.delta
        return F.normalize(pcv, dim=-1)

    def cl_loss(self, zh, y, v, bank_z, bank_y, bank_v, syn, r_t):
        """Cell-rebalanced contrastive loss (Eq. 11) with targets from the epoch bank."""
        tau, gamma, rho = self.flags.get("tau", HP["tau"]), HP["gamma"], HP["rho"]
        classes_only = self.flags.get("cells", "cv") == "class"
        V = 1 if classes_only else self.td.n_var
        bv = bank_v if not classes_only else torch.zeros_like(bank_v)
        pools, pool_cell = [], []
        for c in range(2):
            for vv in range(V):
                ids = torch.nonzero((bank_y == c) & (bv == vv)).flatten()
                if len(ids) == 0:
                    continue
                pools.append(ids[torch.randint(len(ids), (HP["pool"],), device=ids.device)]); pool_cell.append((c, vv))
        T_idx = torch.cat(pools)
        T_c = torch.tensor([c for c, _ in pool_cell], device=DEV).repeat_interleave(HP["pool"])
        T_v = torch.tensor([vv for _, vv in pool_cell], device=DEV).repeat_interleave(HP["pool"])
        T_z = F.normalize(self.model.proj_f(bank_z[T_idx]), dim=-1)
        # anchors: batch + prototypes of every cell
        anchors, ay, av, aw = [zh], [y], [v if not classes_only else torch.zeros_like(v)], []
        cp = self.cell_prior if not classes_only else self.cell_prior.sum(1, keepdim=True)
        w = cp[y, (v if not classes_only else torch.zeros_like(v))].clamp(min=1e-6) ** (-rho)
        aw.append(w / w.mean())
        if self.flags.get("proto", "hier") != "none":
            P = self.proto_cv()
            for c in range(2):
                for vv in range(V):
                    anchors.append(P[c, vv][None]); ay.append(torch.tensor([c], device=DEV))
                    av.append(torch.tensor([vv], device=DEV)); aw.append(torch.ones(1, device=DEV))
        A = torch.cat(anchors); Ay = torch.cat(ay); Av = torch.cat(av); Aw = torch.cat(aw)
        S = A @ T_z.T / tau  # [nA, nT]
        nA = A.shape[0]
        # positives: n_pos per anchor from same-class pools, variety uniform (zeta = 0)
        same = (T_c[None, :] == Ay[:, None]).float()
        other = 1 - same
        pos_idx = torch.multinomial(same + 1e-12 * other, HP["n_pos"], replacement=True)
        neg_idx = torch.multinomial(other + 1e-12 * same, min(HP["n_neg"], int(other.sum(1).min().item())) or 1,
                                    replacement=True)
        s_pos = S.gather(1, pos_idx); s_neg = S.gather(1, neg_idx)
        w_pos = 1 + gamma * (T_v[pos_idx] != Av[:, None]).float()
        extra_pos, extra_neg, extra_wp = [], [], []
        if syn is not None and r_t > 0:
            n_sp = max(1, round(r_t * 5)); n_sn = max(1, round(r_t * 50))
            sp = torch.zeros(nA, n_sp, device=DEV); sn = torch.zeros(nA, n_sn, device=DEV); wp = torch.ones(nA, n_sp, device=DEV)
            for c in range(2):
                m = Ay == c
                if not m.any():
                    continue
                Pz, Pcross, Nz = syn[c]
                if Pz is not None:
                    j = torch.randint(len(Pz), (n_sp,), device=DEV)
                    sp[m] = A[m] @ Pz[j].T / tau; wp[m] = (1 + gamma * Pcross[j].float())[None, :].expand(int(m.sum()), -1)
                if Nz is not None:
                    j = torch.randint(len(Nz), (n_sn,), device=DEV)
                    sn[m] = A[m] @ Nz[j].T / tau
            s_pos = torch.cat([s_pos, sp], 1); s_neg = torch.cat([s_neg, sn], 1); w_pos = torch.cat([w_pos, wp], 1)
        denom = torch.logsumexp(torch.cat([s_pos, s_neg], 1), 1, keepdim=True)
        per = (w_pos * (denom - s_pos)).sum(1) / w_pos.sum(1)
        return (Aw * per).mean()

    def hard_sets(self, bank_z, bank_y, bank_v, noisy):
        """Algorithm 3: hard positives/negatives per class from the epoch bank."""
        with torch.no_grad():
            zh = F.normalize(self.model.proj_f(bank_z), dim=-1)
            logits = self.model.W(bank_z); p = torch.softmax(logits, -1)
            P = self.proto_cv(); pc = self.model.prototypes()
            vv = bank_v.clamp(max=P.shape[1] - 1)
            dist = 1 - (zh * P[bank_y, vv]).sum(-1)
            ent = -(p * torch.log(p + 1e-8)).sum(-1)
            fy = logits.gather(1, bank_y[:, None]).squeeze(1)
            margin = fy - logits.masked_fill(F.one_hot(bank_y, 2).bool(), -1e9).max(1).values
            zs = lambda x: (x - x.mean()) / (x.std() + 1e-8)
            mode = self.flags.get("hardness", "all")
            h = {"all": zs(dist) + zs(ent) + zs(-margin), "dist": zs(dist), "ent": zs(ent), "mar": zs(-margin),
                 "proto": zs(1 - (zh * pc[bank_y]).sum(-1))}[mode]
            h = h.masked_fill(noisy, -1e9)
            k = HP["k_hard"]; out = {}
            for c in range(2):
                posc = torch.nonzero(bank_y == c).flatten()
                negc = torch.nonzero(bank_y != c).flatten()
                hp = posc[torch.topk(h[posc], min(k, len(posc))).indices]
                conf = (zh[negc] @ pc[c]) + p[negc, c]
                hn = negc[torch.topk(conf.masked_fill(noisy[negc], -1e9), min(k, len(negc))).indices]
                out[c] = (hp, hn)
        return out

    def make_mixes(self, hs, bank_z, bank_y, bank_v):
        beta = torch.distributions.Beta(HP["beta_a"], HP["beta_a"])
        syn, boundary = {}, []
        for c in range(2):
            hp, hn = hs[c]; n = HP["n_mix"]
            Pz = Pcross = Nz = None
            if len(hp) >= 2:
                i = hp[torch.randint(len(hp), (n,), device=DEV)]
                # prefer partners of a different variety
                j = torch.empty_like(i)
                for t in range(n):
                    cand = hp[bank_v[hp] != bank_v[i[t]]]
                    cand = cand if len(cand) else hp
                    j[t] = cand[torch.randint(len(cand), (1,), device=DEV)]
                lam = beta.sample((n, 1)).to(DEV)
                Pz = F.normalize(self.model.proj_f(lam * bank_z[i] + (1 - lam) * bank_z[j]), dim=-1)
                Pcross = bank_v[i] != bank_v[j]
            if len(hn) >= 2:
                i = hn[torch.randint(len(hn), (n,), device=DEV)]; j = hn[torch.randint(len(hn), (n,), device=DEV)]
                lam = beta.sample((n, 1)).to(DEV)
                Nz = F.normalize(self.model.proj_f(lam * bank_z[i] + (1 - lam) * bank_z[j]), dim=-1)
            syn[c] = (Pz, Pcross, Nz)
            if len(hp) and len(hn):
                i = hp[torch.randint(len(hp), (2 * n,), device=DEV)]; j = hn[torch.randint(len(hn), (2 * n,), device=DEV)]
                lam = beta.sample((2 * n, 1)).to(DEV)
                zt = lam * bank_z[i] + (1 - lam) * bank_z[j]
                yt = lam * F.one_hot(torch.full((2 * n,), c, device=DEV), 2).float() + (1 - lam) * F.one_hot(bank_y[j], 2).float()
                boundary.append((zt, yt))
        return syn, boundary

    # ---------------------------------------------------------------- training
    def dev_score(self, tau_beta):
        scores = []
        for s in self.dev_sets:
            z, _ = self.encode_set(s, tau_beta)
            pred = self.model.W(z).argmax(-1).cpu().numpy()
            scores.append(macro_f1(self.store.y(s), pred))
        return float(np.mean(scores))

    def fit(self, dev_sets):
        self.dev_sets = dev_sets
        td, fl, E = self.td, self.flags, HP["epochs"]
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
        n = len(td.items); y_all = torch.tensor(td.label, device=DEV); v_all = torch.tensor(td.variety, device=DEV)
        flagged = np.zeros(n); best, best_state = -1, None
        lam_cl = fl.get("lam_cl", HP["lam_cl"]); lam_cons = fl.get("lam_cons", HP["lam_cons"])
        for ep in range(E):
            tau_beta = HP["tau_beta0"] - (HP["tau_beta0"] - HP["tau_beta1"]) * ep / max(1, E - 1)
            phase = 1 if ep < HP["p1"] else (2 if ep < HP["p2"] else 3)
            cal_on = fl.get("cal", True) and phase == 3
            perm = rng.permutation(n)
            cal_slice = perm[: n // 10] if cal_on else np.array([], int)
            main = perm[n // 10:] if cal_on else perm
            bank_z = None; syn = None; boundary = []
            if phase >= 2:
                with torch.no_grad():
                    Z = []
                    for i in range(0, n, 512):
                        z, _ = self.forward_items(td.items[i:i + 512], tau_beta, train=False); Z.append(z)
                    bank_z = torch.cat(Z)
                    loss_i = F.cross_entropy(self.model.W(bank_z) + HP["tau_la"] * self.log_prior, y_all, reduction="none")
                    top = loss_i >= torch.quantile(loss_i, 0.98)
                    flagged += top.cpu().numpy()
                noisy = torch.tensor(flagged > (ep + 1) / 2, device=DEV)
                if phase == 3 and fl.get("hm", True):
                    hs = self.hard_sets(bank_z, y_all, v_all, noisy)
            r_t = 0.5 + 0.5 * (ep - HP["p2"]) / max(1, E - HP["p2"]) if phase == 3 else 0.0
            ramp = min(1.0, (ep - HP["p1"] + 1) / (HP["p2"] - HP["p1"])) if phase >= 2 else 0.0
            for b in range(0, len(main), HP["batch"]):
                bi = main[b:b + HP["batch"]]
                items = [td.items[i] for i in bi]
                y = y_all[bi]; v = v_all[bi]
                z, alpha = self.forward_items(items, tau_beta)
                logits = self.model.W(z)
                loss = self.ce_la(logits, y)
                if self.model.M > 1 and fl["fusion"] == "gate":
                    abar = alpha.mean(0)
                    loss = loss + HP["lam_g"] * (abar * torch.log(abar * self.model.M + 1e-8)).sum()
                if phase >= 2:
                    zh = F.normalize(self.model.proj_f(z), dim=-1)
                    with torch.no_grad():  # EMA variety offsets (Eq. 9)
                        pc = self.model.prototypes()
                        for c in range(2):
                            for vv in range(td.n_var):
                                m = (y == c) & (v == vv)
                                if m.any():
                                    self.delta[c, vv] = HP["mu"] * self.delta[c, vv] + (1 - HP["mu"]) * (zh[m] - pc[c]).mean(0)
                    if fl.get("proto", "hier") != "none":
                        P = self.proto_cv()
                        vv = v.clamp(max=P.shape[1] - 1)
                        logits_p = torch.einsum("bk,bck->bc", zh, P[:, vv].permute(1, 0, 2)) / HP["tau_p"]
                        loss = loss + HP["lam_p"] * F.cross_entropy(logits_p, y)
                    if phase == 3 and fl.get("hm", True):
                        syn, boundary = self.make_mixes(hs, bank_z, y_all, v_all)
                    if lam_cl > 0:
                        loss = loss + lam_cl * self.cl_loss(zh, y, v, bank_z, y_all, v_all, syn if phase == 3 else None, r_t)
                    if fl.get("cons", True):
                        pairs = [(k, td.partner(vw, i, rng)) for k, (vw, i) in enumerate(items)]
                        pairs = [(k, p) for k, p in pairs if p is not None]
                        if pairs:
                            ks = torch.tensor([k for k, _ in pairs], device=DEV)
                            z2, a2 = self.forward_items([p for _, p in pairs], tau_beta)
                            terms = fl.get("cons_terms", "both")
                            lc = 0
                            if terms in ("both", "emb"):
                                lc = lc + (1 - F.cosine_similarity(z[ks], z2, dim=-1))
                            if terms in ("both", "pred"):
                                lc = lc + HP["eta"] * js_div(torch.softmax(logits[ks], -1), torch.softmax(self.model.W(z2), -1))
                            loss = loss + lam_cons * ramp * lc.mean()
                            if fl.get("gatecons", False) and self.model.M > 1:
                                loss = loss + 0.1 * ((alpha[ks] - a2) ** 2).sum(-1).mean()
                if phase >= 2 and fl.get("xling", False) and self.store.has("train_madar_tn"):
                    npairs = len(self.store.ids("train_madar_tn"))
                    pi = torch.tensor(rng.choice(npairs, min(HP["batch"], npairs), replace=False))
                    za, _, _ = self.model(self._inputs("train_madar_tn", pi), self._surf("train_madar_tn", pi), tau_beta)
                    zb, _, _ = self.model(self._inputs("train_madar_msa", pi), self._surf("train_madar_msa", pi), tau_beta)
                    lg = F.normalize(self.model.proj_f(za), dim=-1) @ F.normalize(self.model.proj_f(zb), dim=-1).T / 0.05
                    tgt = torch.arange(len(pi), device=DEV)
                    loss = loss + fl.get("lam_x", 0.1) * 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
                if phase == 3 and fl.get("hm", True) and boundary:
                    zt = torch.cat([b_[0] for b_ in boundary]); yt = torch.cat([b_[1] for b_ in boundary])
                    loss = loss + HP["lam_hm"] * (-(yt * F.log_softmax(self.model.W(zt), -1)).sum(-1).mean())
                if cal_on and len(cal_slice):
                    ci = cal_slice[rng.choice(len(cal_slice), min(HP["batch"], len(cal_slice)), replace=False)]
                    zc, _ = self.forward_items([td.items[i] for i in ci], tau_beta)
                    lg = self.model.W(zc); pr = torch.softmax(lg + HP["tau_la"] * self.log_prior, -1)
                    pred = lg.argmax(-1); conf = pr.gather(1, pred[:, None]).squeeze(1)
                    loss = loss + HP["lam_cal"] * mmce(conf, (pred == y_all[ci]).float())
                opt.zero_grad(); loss.backward(); opt.step()
            if ep >= HP["p1"] - 1:
                sc = self.dev_score(tau_beta)
                if sc > best:
                    best, best_state = sc, (copy.deepcopy(self.model.state_dict()), self.delta.clone(), tau_beta)
        self.model.load_state_dict(best_state[0]); self.delta = best_state[1]; self.tau_beta = best_state[2]
        self.dev = best
        return self

    # ---------------------------------------------------------------- calibration + prediction
    def calibrate(self, prior_corr=True):
        logits, vlab, feats = [], [], []
        for k, s in enumerate(self.dev_sets):
            z, a = self.encode_set(s, self.tau_beta)
            f = self.model.W(z)
            logits.append(f.detach()); vlab += [k] * len(f)
            feats.append(torch.cat([a, self.store.surface(s).to(DEV)], -1).cpu().numpy())
        ys = np.concatenate([self.store.y(s) for s in self.dev_sets])
        L = torch.cat(logits); vlab = np.array(vlab)
        self.cal = {}
        for pc in (True, False):
            G = L + HP["tau_la"] * self.log_prior if pc else L
            T = fit_temperature(G, ys)
            Tv = {}
            for k in range(len(self.dev_sets)):
                m = vlab == k
                Tk = fit_temperature(G[m], ys[m]); nk = int(m.sum())
                Tv[k] = math.exp((nk * math.log(Tk) + HP["kappa_T"] * math.log(T)) / (nk + HP["kappa_T"]))
            self.cal[pc] = (T, Tv)
        X = np.concatenate(feats)
        self.probe = LogisticRegression(max_iter=2000).fit(X, vlab) if len(self.dev_sets) > 1 else None
        # off-support threshold on validation (both views)
        P = self.proto_cv().reshape(-1, 128)
        sims = []
        for s in self.dev_sets:
            z, _ = self.encode_set(s, self.tau_beta)
            sims.append((F.normalize(self.model.proj_f(z), dim=-1) @ P.T).max(1).values.detach().cpu().numpy())
        self.val_maxproto = np.concatenate(sims)
        return self

    def predict(self, s, variety_wise=True):
        with torch.no_grad():
            z, a = self.encode_set(s, self.tau_beta)
            f = self.model.W(z)
            feats = torch.cat([a, self.store.surface(s).to(DEV)], -1).cpu().numpy()
            if self.probe is not None:
                pv = self.probe.predict_proba(feats); vhat = pv.argmax(1); plat = pv[:, -1]
            else:
                vhat = np.zeros(len(f), int); plat = np.zeros(len(f))
            out = {"pred": f.argmax(-1).cpu().numpy(), "plat": plat, "alpha": a.cpu().numpy(), "z": z.cpu().numpy()}
            P = self.proto_cv().reshape(-1, 128)
            out["maxproto"] = (F.normalize(self.model.proj_f(z), dim=-1) @ P.T).max(1).values.cpu().numpy()
            for pc, tag in ((True, ""), (False, "_nopc")):
                G = f + HP["tau_la"] * self.log_prior if pc else f
                T, Tv = self.cal[pc]
                out["p_raw" + tag] = torch.softmax(G, -1)[:, 1].cpu().numpy()
                out["p_glob" + tag] = torch.softmax(G / T, -1)[:, 1].cpu().numpy()
                tv = torch.tensor([Tv.get(int(k), T) for k in vhat], device=DEV, dtype=G.dtype)[:, None]
                out["p_var" + tag] = (torch.softmax(G / tv, -1)[:, 1].cpu().numpy() if variety_wise
                                      else out["p_glob" + tag])
        return out


def fit_temperature(logits, y):
    logits = logits.detach().float(); y = torch.tensor(y, device=logits.device)
    best, bestT = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        nll = F.cross_entropy(logits / float(T), y).item()
        if nll < best:
            best, bestT = nll, float(T)
    return bestT


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


# =========================================================================== baselines on pooled vectors
class MLPHead(nn.Module):
    def __init__(self, din, hidden=256, dropout=0.1, C=2, proj=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.W = nn.Linear(hidden, C)
        self.proj = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 128)) if proj else None

    def forward(self, x):
        h = self.body(x)
        return self.W(h), h


class Fuse7(nn.Module):
    def __init__(self, dims, mode, hidden=256, dropout=0.1):
        super().__init__()
        self.mode = mode; self.dims = dims
        self.P = nn.ModuleList([nn.Linear(d, 512) for d in dims]); self.head = MLPHead(512, hidden, dropout)

    def forward(self, x):
        parts = torch.split(x, self.dims, -1); es = [p(t) for p, t in zip(self.P, parts)]
        s = sum(es); h = es[0]
        for e in es[1:]:
            h = h * e
        z = {"sum": s, "hadamard": h, "residual": s + h}[self.mode]
        return self.head(z)


def supcon(zh, y, tau=0.1):
    S = zh @ zh.T / tau
    eye = torch.eye(len(y), device=DEV, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = S - torch.logsumexp(S, 1, keepdim=True)
    cnt = pos.sum(1).clamp(min=1)
    return -((logp * pos).sum(1) / cnt).mean()


def bcl_loss(zh, y, protos, tau=0.1):
    """Balanced contrastive learning: class-averaged denominators, class prototypes as anchors' targets."""
    allz = torch.cat([zh, protos]); ally = torch.cat([y, torch.arange(2, device=DEV)])
    S = zh @ allz.T / tau
    n = len(y); eye = torch.zeros_like(S, dtype=torch.bool); eye[:, :n] = torch.eye(n, device=DEV, dtype=torch.bool)
    expS = torch.exp(S - S.max(1, keepdim=True).values).masked_fill(eye, 0)
    denom = 0
    for c in range(2):
        mc = (ally == c)[None, :].expand_as(S) & ~eye
        denom = denom + (expS * mc).sum(1) / mc.sum(1).clamp(min=1)
    pos = (ally[None, :] == y[:, None]) & ~eye
    logp = torch.log(expS + 1e-12) - torch.log(denom[:, None] + 1e-12)
    return -((logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def train_pooled(Xtr, ytr, dev_list, seed, kind="mlp", hp=None, extra=None, prior_la=False):
    """Generic trainer for B0/B2-B9/CEIL on fixed vectors. dev_list: [(X, y)]."""
    hp = {"lr": 1e-3, "dropout": 0.1, "hidden": 256, "wd": 5e-4, **(hp or {})}
    set_seed(seed)
    din = Xtr.shape[1]
    if kind.startswith("fuse7:"):
        model = Fuse7(extra["dims"], kind.split(":")[1], hp["hidden"], hp["dropout"]).to(DEV)
    else:
        model = MLPHead(din, hp["hidden"], hp["dropout"], proj=kind in ("supcon", "bcl")).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    Xtr = Xtr.to(DEV); ytr_t = torch.tensor(ytr, device=DEV)
    prior = torch.log(torch.tensor(np.bincount(ytr, minlength=2) / len(ytr), device=DEV, dtype=torch.float32))
    best, state = -1, None; rng = np.random.default_rng(seed)
    for ep in range(HP["epochs"]):
        model.train()
        perm = rng.permutation(len(ytr))
        for b in range(0, len(perm), HP["batch"]):
            bi = torch.tensor(perm[b:b + HP["batch"]], device=DEV)
            logits, h = model(Xtr[bi]); yb = ytr_t[bi]
            loss = F.cross_entropy(logits + (prior if prior_la else 0), yb)
            if kind == "supcon":
                loss = loss + 0.5 * supcon(F.normalize(model.proj(h), dim=-1), yb)
            if kind == "bcl":
                protos = F.normalize(model.proj(model.W.weight), dim=-1)
                loss = loss + 0.5 * bcl_loss(F.normalize(model.proj(h), dim=-1), yb, protos)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = np.mean([macro_f1(y, model(X.to(DEV))[0].argmax(-1).cpu().numpy()) for X, y in dev_list])
        if sc > best:
            best, state = sc, copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval()
    model.dev = best; model.prior = prior if prior_la else torch.zeros(2, device=DEV)
    return model


def pooled_predict(model, X, T=1.0, Tv=None, varieties=None):
    with torch.no_grad():
        f = model(X.to(DEV))[0]
        G = f + model.prior
        out = {"pred": f.argmax(-1).cpu().numpy(), "p_raw": torch.softmax(G, -1)[:, 1].cpu().numpy(),
               "p_glob": torch.softmax(G / T, -1)[:, 1].cpu().numpy()}
        if Tv is not None and varieties is not None:
            tv = torch.tensor([Tv[v] for v in varieties], device=DEV, dtype=G.dtype)[:, None]
            out["p_var"] = torch.softmax(G / tv, -1)[:, 1].cpu().numpy()
        else:
            out["p_var"] = out["p_glob"]
        out["logits"] = f.cpu().numpy()
    return out


# =========================================================================== orchestration
class Runner:
    def __init__(self, args):
        self.a = args
        self.encs = args.encoders.split(",")
        self.store = Store(args.data, args.feats, self.encs)
        self.eval_sets = [s for s in sorted(f[:-4] for f in os.listdir(args.data) if f.endswith(".tsv"))
                          if not s.startswith(("train", "val", "tt-")) and self.store.has(s)]
        self.costs = {}
        self.devfile = os.path.join(args.out, "dev_scores.json")
        self.devlog = json.load(open(self.devfile)) if os.path.exists(self.devfile) else {}

    def log_dev(self, system, seed, score):
        self.devlog[f"{system}/{seed}"] = score
        os.makedirs(self.a.out, exist_ok=True); json.dump(self.devlog, open(self.devfile, "w"), indent=0)

    def dev_of(self, system):
        v = [x for k, x in self.devlog.items() if k.rsplit("/", 1)[0] == system]
        return float(np.mean(v)) if v else None

    def out(self, system, seed, s):
        return os.path.join(self.a.out, "preds", system, str(seed), f"{s}.tsv")

    def done(self, system, seed):
        return os.path.exists(self.out(system, seed, "my"))

    def write_preds(self, system, seed, s, res, ids=None):
        p = self.out(system, seed, s); os.makedirs(os.path.dirname(p), exist_ok=True)
        ids = ids or self.store.ids(s)
        with open(p, "w", encoding="utf-8") as f:
            f.write("id\tpred\tp_raw\tp_glob\tp_var\n")
            for k, i in enumerate(ids):
                f.write(f"{i}\t{int(res['pred'][k])}\t{res['p_raw'][k]:.6f}\t{res['p_glob'][k]:.6f}\t{res['p_var'][k]:.6f}\n")

    def varieties(self, s):
        return [1 if script_of(t) == "lat" else 0 for t in self.store.texts(s)]

    # ------------------------------------------------------------ pooled-vector systems
    def vecs(self, s, spec):
        """spec: list of (enc, layer, pool)."""
        return torch.cat([self.store.pooled(s, e, l, p) for e, l, p in spec], -1)

    def run_pooled(self, system, spec, seeds, kind="mlp", hp=None, train_views=("train_ar", "train_gs"),
                   dev_sets=("val_ar", "val_gs"), prior_la=False, extra=None, eval_map=None, tune=False):
        if tune and self.a.mode == "full":
            hp = self.tune_pooled(spec, kind, train_views, dev_sets, prior_la, extra)
        Xtr = torch.cat([self.vecs(v, spec) for v in train_views]); ytr = np.concatenate([self.store.y(v) for v in train_views])
        devs = [(self.vecs(s, spec), self.store.y(s)) for s in dev_sets]
        devs_score = []
        for seed in seeds:
            if self.done(system, seed):
                continue
            t0 = time.time()
            m = train_pooled(Xtr, ytr, devs, seed, kind, hp, extra, prior_la)
            self.costs.setdefault(system, []).append((time.time() - t0) / 60)
            devs_score.append(m.dev); self.log_dev(system, seed, m.dev)
            G = torch.cat([m(X.to(DEV))[0].detach() + m.prior for X, _ in devs])
            yd = np.concatenate([y for _, y in devs]); T = fit_temperature(G, yd)
            Tv = {}
            for k, (X, y) in enumerate(devs):
                Gk = m(X.to(DEV))[0].detach() + m.prior; Tk = fit_temperature(Gk, y)
                Tv[k] = math.exp((len(y) * math.log(Tk) + HP["kappa_T"] * math.log(T)) / (len(y) + HP["kappa_T"]))
            vmap = {0: Tv.get(0, T), 1: Tv.get(1, Tv.get(0, T))}
            for s in self.eval_sets:
                src = (eval_map or {}).get(s, s)
                if not self.store.has(src):
                    continue
                res = pooled_predict(m, self.vecs(src, spec), T, vmap, self.varieties(src))
                self.write_preds(system, seed, s, res, self.store.ids(s))
            if system == "B6" and seed == seeds[0]:
                self.save_emb("B6", spec)
        return self.dev_of(system)

    def tune_pooled(self, spec, kind, train_views, dev_sets, prior_la, extra):
        grid = list(itertools.product([1e-3, 3e-4], [0.1, 0.3], [256, 512], [5e-4, 1e-2]))[:20]
        Xtr = torch.cat([self.vecs(v, spec) for v in train_views]); ytr = np.concatenate([self.store.y(v) for v in train_views])
        devs = [(self.vecs(s, spec), self.store.y(s)) for s in dev_sets]
        best, best_hp = -1, None
        for lr, do, hid, wd in grid:
            hp = {"lr": lr, "dropout": do, "hidden": hid, "wd": wd}
            sc = train_pooled(Xtr, ytr, devs, 0, kind, hp, extra, prior_la).dev
            if sc > best:
                best, best_hp = sc, hp
        return best_hp

    def save_emb(self, system, spec=None, trainer=None):
        d = os.path.join(self.a.out, "emb", system); os.makedirs(d, exist_ok=True)
        for s in ["train_gs", "ar", "gs", "gu", "tl", "tz", "na", "my", "myar"]:
            if not self.store.has(s):
                continue
            if trainer is not None:
                z, _ = trainer.encode_set(s, trainer.tau_beta); np.save(os.path.join(d, f"{s}.npy"), z.cpu().numpy())
            else:
                np.save(os.path.join(d, f"{s}.npy"), self.vecs(s, spec).numpy())

    def default_spec(self, encs=None):
        return [(e, -1, DEFAULT_POOL.get(e, "mean")) for e in (encs or self.encs[:3])]

    # ------------------------------------------------------------ DiaFuse family
    def run_diafuse(self, system, seeds, flags, data_kw=None, encs=None, input_mode="layers", diag=False,
                    variety_wise=True, also_nopc=False, dev_sets=None):
        flags = {"fusion": "gate", **flags}
        encs = encs or self.encs[:3]
        devs = []
        for seed in seeds:
            if self.done(system, seed):
                continue
            td = TrainData(self.store, seed=seed, **(data_kw or {}))
            ds = dev_sets or (["val_ar", "val_gs"] if td.n_var > 1 else ["val_ar"])
            t0 = time.time()
            tr = DiaFuseTrainer(self.store, td, flags, seed, encs, input_mode).fit(ds).calibrate()
            self.costs.setdefault(system, []).append((time.time() - t0) / 60)
            devs.append(tr.dev); self.log_dev(system, seed, tr.dev)
            self.costs[f"{system}/params"] = sum(p.numel() for p in tr.model.parameters())
            for s in self.eval_sets:
                res = tr.predict(s, variety_wise=variety_wise and td.n_var > 1)
                self.write_preds(system, seed, s, res)
                if also_nopc:
                    self.write_preds(system + "-nopc", seed, s, {"pred": res["pred"], "p_raw": res["p_raw_nopc"],
                                                                 "p_glob": res["p_glob_nopc"], "p_var": res["p_var_nopc"]})
                if diag:
                    self.write_diag(system, seed, s, res)
            if diag:
                vres = {k: np.concatenate([tr.predict(s)[k] for s in ds]) for k in ("alpha", "plat", "maxproto")}
                ids = sum([self.store.ids(s) for s in ds], [])
                self.write_diag(system, seed, "val", vres, ids)
                if seed == seeds[0]:
                    self.save_emb(system, trainer=tr)
                    self.head_latency(tr)
        return self.dev_of(system)

    def write_diag(self, system, seed, s, res, ids=None):
        p = os.path.join(self.a.out, "diag", system, str(seed), f"{s}.tsv"); os.makedirs(os.path.dirname(p), exist_ok=True)
        ids = ids or self.store.ids(s); A = res["alpha"]
        with open(p, "w", encoding="utf-8") as f:
            f.write("id\t" + "\t".join(f"alpha{m + 1}" for m in range(A.shape[1])) + "\tplat\tmaxproto\n")
            for k, i in enumerate(ids):
                f.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in A[k]) + f"\t{res['plat'][k]:.4f}\t{res['maxproto'][k]:.4f}\n")

    def head_latency(self, tr):
        s = "ar"; idx = torch.arange(min(100, len(self.store.ids(s))))
        Us = tr._inputs(s, idx); S = tr._surf(s, idx); tr.model.eval(); t0 = time.time()
        with torch.no_grad():
            for i in range(len(idx)):
                z, _, _ = tr.model([U[i:i + 1] for U in Us], S[i:i + 1], tr.tau_beta); tr.model.W(z)
        self.costs["head_ms"] = 1000 * (time.time() - t0) / len(idx)

    def tune_diafuse(self, seeds_dev=(0,)):
        grid = list(itertools.product([0.1, 0.3, 0.5], [0.1, 0.5, 1.0], [0.1, 0.5, 1.0]))
        random.Random(0).shuffle(grid)
        best, best_f = -1, {}
        for tau, lcl, lcons in grid[:20]:
            f = {"tau": tau, "lam_cl": lcl, "lam_cons": lcons}
            td = TrainData(self.store, seed=0)
            sc = DiaFuseTrainer(self.store, td, {"fusion": "gate", **f}, 0, self.encs[:3]).fit(["val_ar", "val_gs"]).dev
            print(f"  tune {f}: dev {sc:.4f}", flush=True)
            if sc > best:
                best, best_f = sc, f
        return best_f

    # ------------------------------------------------------------ CEIL
    def run_ceil(self, seeds):
        spec = self.default_spec()
        for seed in seeds:
            if self.done("CEIL", seed):
                continue
            for s in ["ar", "tl", "tz", "na", "my", "myar", "naar", "gsmy"]:
                if not self.store.has(s):
                    continue
                X = self.vecs(s, spec); y = self.store.y(s)
                pred = np.zeros(len(y), int); pr = np.zeros(len(y))
                for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    rs = np.random.default_rng(seed); inner = rs.permutation(tr_i); k = len(inner) // 10
                    m = train_pooled(X[inner[k:]], y[inner[k:]], [(X[inner[:k]], y[inner[:k]])], seed)
                    r = pooled_predict(m, X[te_i]); pred[te_i] = r["pred"]; pr[te_i] = r["p_raw"]
                self.write_preds("CEIL", seed, s, {"pred": pred, "p_raw": pr, "p_glob": pr, "p_var": pr})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data"); ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="runs"); ap.add_argument("--encoders", default="xlmr,me5,labse")
    ap.add_argument("--mode", default="quick", choices=["quick", "full"])
    ap.add_argument("--only", default=None, help="comma list of system groups to run")
    ap.add_argument("--skip", default="", help="comma list of groups to skip")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    if args.epochs:
        HP["epochs"] = args.epochs; HP["p1"] = max(1, round(args.epochs * 0.1)); HP["p2"] = max(HP["p1"] + 1, round(args.epochs * 0.43))
    R = Runner(args)
    seeds_main = list(range(10 if args.mode == "full" else 3))
    seeds_abl = list(range(5 if args.mode == "full" else 2))
    groups = ["B0", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B12", "CEIL", "abl", "abl2", "lt", "fact"]
    run = [g for g in (args.only.split(",") if args.only else groups) if g not in args.skip.split(",")]
    E3 = R.encs[:3]; spec6 = R.default_spec(E3)
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

    if "B6" in run:
        log("B6 static concatenation"); R.run_pooled("B6", spec6, seeds_main, tune=True)
    if "B0" in run:
        log("B0 Arabic-only / B0-TT")
        R.run_pooled("B0", spec6, seeds_main, train_views=("train_ar",), dev_sets=("val_ar",))
        tt = {s: f"tt-{s}" for s in ["gs", "gu", "tl", "tz", "na", "my", "gsmy"]}
        R.run_pooled("B0TT", spec6, seeds_main, train_views=("train_ar",), dev_sets=("val_ar",), eval_map=tt)
    if "B3" in run or "B2" in run:
        log("B3 single encoders"); dev = {}
        for k, e in enumerate(E3):
            dev[e] = R.run_pooled(f"B3-e{k + 1}", [(e, -1, DEFAULT_POOL[e])], seeds_main)
        best = max(E3, key=lambda e: dev[e] or -1)
        R.run_pooled("B3", [(best, -1, DEFAULT_POOL[best])], seeds_main)
        cands = [e for e in ["me5", "labse", "bgem3"] if e in R.encs]
        b2 = max(cands, key=lambda e: dev.get(e) or -1)
        R.run_pooled("B2", [(b2, -1, DEFAULT_POOL[b2])], seeds_main)
    if "B4" in run:
        log("B4 last-layer mean")
        sc = {e: R.run_pooled(f"B4-{e}", [(e, -1, "mean")], seeds_main[:1]) for e in E3}
        best = max(E3, key=lambda e: sc[e] or -1); R.run_pooled("B4", [(best, -1, "mean")], seeds_main)
    if "B5" in run:
        log("B5 dev-selected layer and pooling (logistic-regression probe)")
        best, cfg = -1, None
        for e in E3:
            for li in range(R.store.feat("val_ar", e).shape[1]):
                for pool in ("mean", "tok"):
                    sp = [(e, li, pool)]
                    Xtr = torch.cat([R.vecs(v, sp) for v in ("train_ar", "train_gs")]).numpy()
                    ytr = np.concatenate([R.store.y(v) for v in ("train_ar", "train_gs")])
                    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
                    sc = np.mean([macro_f1(R.store.y(s), clf.predict(R.vecs(s, sp).numpy())) for s in ("val_ar", "val_gs")])
                    if sc > best:
                        best, cfg = sc, sp
        log("  B5 selected", cfg); R.run_pooled("B5", cfg, seeds_main)
    if "B7" in run:
        log("B7 static fusion")
        dims = [R.vecs("val_ar", [s]).shape[1] for s in spec6]
        sc = {m: R.run_pooled(f"B7-{m}", spec6, seeds_main[:1], kind=f"fuse7:{m}", extra={"dims": dims}) for m in ("sum", "hadamard", "residual")}
        best = max(sc, key=lambda m: sc[m] or -1); R.run_pooled("B7", spec6, seeds_main, kind=f"fuse7:{best}", extra={"dims": dims})
    if "B8" in run:
        log("B8 CE+SupCon"); R.run_pooled("B8", spec6, seeds_main, kind="supcon")
    if "B9" in run:
        log("B9 BCL"); R.run_pooled("B9", spec6, seeds_main, kind="bcl", prior_la=True)
    b10_flags = {"fusion": "single", "proto": "class", "cells": "class", "hardness": "proto", "cons": False, "cal": False}
    if "B10" in run:
        log("B10 SharpReCL-style")
        R.run_diafuse("B10", seeds_main, b10_flags, encs=E3, input_mode="pooled", variety_wise=False)
    best12 = {}
    if "B12" in run:
        if args.mode == "full":
            log("B12 tuning (20 configurations)"); best12 = R.tune_diafuse()
            json.dump(best12, open(os.path.join(args.out, "b12_hparams.json"), "w"))
        log("B12 DiaFuse-PRC", best12)
        R.run_diafuse("B12", seeds_main, best12, diag=True, also_nopc=True)
    elif os.path.exists(os.path.join(args.out, "b12_hparams.json")):
        best12 = json.load(open(os.path.join(args.out, "b12_hparams.json")))
    if "CEIL" in run:
        log("CEIL target-supervised ceiling"); R.run_ceil(seeds_main)
    if "abl" in run:
        log("ablations")
        A = {"abl-layer": {"layer_mode": "last"}, "abl-static": {"fusion": "static"}, "abl-nos": {"gate_input": "e"},
             "abl-protocls": {"proto": "class"}, "abl-protonone": {"proto": "none"}, "abl-hm": {"hm": False},
             "abl-hmproto": {"hardness": "proto"}, "abl-cons": {"cons": False}, "abl-cal": {"cal": False}}
        for name, f in A.items():
            R.run_diafuse(name, seeds_abl, {**best12, **f}, variety_wise=name != "abl-cal")
        R.run_diafuse("abl-nosyn", seeds_abl, {**best12, "cons": False}, data_kw={"arabic_only": True})
        if R.store.has("train_madar_tn"):
            R.run_diafuse("abl-xling", seeds_abl, {**best12, "xling": True})
        else:
            log("  abl-xling skipped: no MADAR pairs (set MADAR_DIR before prep_data.py)")
        sc = {e: R.run_diafuse(f"abl-single-{e}", seeds_abl[:1], {**best12, "fusion": "single"}, encs=[e]) for e in E3}
        best = max(E3, key=lambda e: sc[e] or -1)
        R.run_diafuse("abl-single", seeds_abl, {**best12, "fusion": "single"}, encs=[best])
    if "abl2" in run:
        log("additional ablations")
        A2 = {"abl2-gatecons": {"gatecons": True}, "abl2-sonly": {"gate_input": "s"}, "abl2-mean": {"layer_mode": "mean"},
              "abl2-tok": {"layer_mode": "tok"}, "abl2-indepA": {"indepA": True}, "abl2-k0": {"kappa": 0.0},
              "abl2-k10": {"kappa": 10.0}, "abl2-kinf": {"kappa": float("inf")}, "abl2-hdist": {"hardness": "dist"},
              "abl2-hent": {"hardness": "ent"}, "abl2-hmar": {"hardness": "mar"}, "abl2-consemb": {"cons_terms": "emb"},
              "abl2-conspred": {"cons_terms": "pred"}}
        for name, f in A2.items():
            R.run_diafuse(name, seeds_abl, {**best12, **f})
        R.run_diafuse("abl2-tronly", seeds_abl, best12, data_kw={"families": ("TR",)})
        R.run_diafuse("abl2-twogen", seeds_abl, best12, data_kw={"twogen": True})
        if len(E3) == 3:
            R.run_diafuse("abl2-m2", seeds_abl, best12, encs=E3[:2])
        if "marbert" in R.encs:
            R.run_diafuse("abl2-m4", seeds_abl, best12, encs=E3 + ["marbert"])
    if "lt" in run:
        log("controlled imbalance")
        for irc, irv in ((10, 1), (1, 4), (1, 10), (10, 10)):
            dk = {"ir_c": irc if irc > 1 else None, "r_syn": 1.0 / irv}
            tag = f"lt-{irc}-{irv}"
            R.run_diafuse(f"{tag}-B12", seeds_main, best12, data_kw=dk)
            R.run_diafuse(f"{tag}-B10", seeds_main, b10_flags, data_kw=dk, input_mode="pooled", variety_wise=False)
            for sysname, kind, la in (("B8", "supcon", False), ("B9", "bcl", True)):
                for seed in seeds_main:
                    if R.done(f"{tag}-{sysname}", seed):
                        continue
                    td = TrainData(R.store, seed=seed, **dk)
                    XA, XL = R.vecs("train_ar", spec6), R.vecs("train_gs", spec6)
                    rows = torch.tensor([i for _, i in td.items]); isl = torch.tensor([v != "AR" for v, _ in td.items])
                    Xtr = torch.where(isl[:, None], XL[rows], XA[rows])
                    m = train_pooled(Xtr, td.label, [(R.vecs(s, spec6), R.store.y(s)) for s in ("val_ar", "val_gs")], seed, kind, None, None, la)
                    for s in R.eval_sets:
                        R.write_preds(f"{tag}-{sysname}", seed, s, pooled_predict(m, R.vecs(s, spec6)))
    if "fact" in run:
        log("factorial 2^4")
        for bits in itertools.product([0, 1], repeat=4):
            fu, pr, hm, co = bits
            f = {**best12, "fusion": "gate" if fu else "static", "proto": "hier" if pr else "none", "hm": bool(hm), "cons": bool(co)}
            R.run_diafuse("fact-" + "".join(map(str, bits)), seeds_abl, f)

    # costs for Table 17
    man_p = os.path.join(args.out, "manual.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    meta = {e: R.store.meta[e] for e in R.encs}
    ext = {e: meta[e].get("sec_per_1k") for e in E3}
    lat = {e: meta[e].get("latency_ms") for e in E3}
    if all(ext.values()):
        man["cost/B3/extract"] = f"{min(ext.values()):.1f}"
        for s in ("B6", "B10", "B12"):
            man[f"cost/{s}/extract"] = f"{sum(ext.values()):.1f}"
    for s in ("B3", "B6", "B10", "B12"):
        if s in R.costs:
            man[f"cost/{s}/train"] = f"{np.mean(R.costs[s]):.1f}"
    if all(lat.values()):
        head = R.costs.get("head_ms", 0.0)
        man["cost/B3/infer"] = f"{min(lat.values()) + head:.1f}"
        for s in ("B6", "B10", "B12"):
            man[f"cost/{s}/infer"] = f"{sum(lat.values()) + head:.1f}"
    for s_ in ("B10", "B12"):
        if f"{s_}/params" in R.costs:
            man[f"cost/{s_}/params"] = f"{R.costs[f'{s_}/params'] / 1e6:.2f}M"
    qmeta = os.path.join(args.feats, "qwen", "meta.json")
    if os.path.exists(qmeta) and all(ext.values()) and all(lat.values()):
        q = json.load(open(qmeta))
        man["cost/qwen/extract"] = f"{sum(ext.values()) + q['sec_per_1k']:.1f}"
        man["cost/qwen/infer"] = f"{sum(lat.values()) + q['latency_ms'] + R.costs.get('head_ms', 0.0):.1f}"
        shapes = [(R.store.feat("val_ar", e).shape[1], R.store.feat("val_ar", e).shape[2]) for e in E3]
        shapes.append((len(q["layers"]), 2 * q["dim"]))
        man["cost/qwen/params"] = f"{sum(p.numel() for p in DiaFuse(shapes, {'fusion': 'gate'}).parameters()) / 1e6:.2f}M"
        man["cost/qwen/train"] = "n/m"
    json.dump(man, open(man_p, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
