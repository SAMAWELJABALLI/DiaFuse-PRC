#!/usr/bin/env python3
"""DiaFuse-PRC++ (Setting A: frozen encoders, cached features).

Separate implementation that *imports* the original pipeline (diafuse_run.py) and never modifies it.
The reference configuration V0 ("impl": "orig") calls the original DiaFuseTrainer unchanged, so that
the reproduced B12 is produced by exactly the code that produced the paper's B12.

Two modes, chosen so that the test sets stay invisible during the search:

  search : train + early-stop on the dev views only; writes runs_pp/dev/<config>/<seed>.json
           (dev macro-F1, per-epoch log, gate/representation diagnostics). NO test-set prediction.
  final  : train, calibrate and predict every evaluation set; writes runs/preds/<SYSTEM>/<seed>/*.tsv
           (format of diafuse_eval.py) and runs/diag/<SYSTEM>/<seed>/*.tsv.

    python src/diafuse_pp_setting_A.py search --configs configs/B12_setting_A/screening_configs.json --only V1a,V1b --seeds 0,1,2
    python src/diafuse_pp_setting_A.py final  --configs <run_dir>/configs/settingA_selected.json --only FINAL --system PPA --seeds 0-9

Environment variable DIAFUSE_CODE points to the directory holding the original scripts
(diafuse_run.py, translit.py); default: <package>/original/code (unmodified original scripts).
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.environ.get("DIAFUSE_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original", "code"))
sys.path.insert(0, CODE)
import diafuse_run as base  # noqa: E402  (original pipeline, unmodified)


def _wjson(obj, p):
    """Atomic JSON write (file closed and flushed before the rename)."""
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(p + ".tmp", p)


HP, DEV = base.HP, base.DEV

# Every key has the value used by the original B12 run, so that {} reproduces the original design.
_DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "B12_setting_A", "defaults.json")
DEFAULTS = {k: v for k, v in json.load(open(_DEFAULTS_FILE)).items() if not k.startswith("_")}  # single source of truth
GATED = ("gate", "gated_residual", "attn")


def load_configs(path):
    spec = json.load(open(path))
    out = {}
    for name, c in spec["configs"].items():
        cfg = copy.deepcopy(DEFAULTS)
        for parent in c.get("inherit", []):
            cfg.update({k: v for k, v in spec["configs"][parent].items() if k not in ("inherit", "family", "why")})
        cfg.update({k: v for k, v in c.items() if k not in ("inherit", "family", "why")})
        cfg["_family"] = c.get("family", "")
        out[name] = cfg
    return out


def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ================================================================== model
class Head(nn.Module):
    def __init__(self, d, C, kind, dropout):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.body = nn.Identity()
        elif kind == "mlp":
            self.body = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        elif kind == "ln_mlp":
            self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        elif kind == "resmlp":
            self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        else:
            raise ValueError(kind)
        self.out = nn.Linear(d, C)

    def features(self, z):
        return z + self.body(z) if self.kind == "resmlp" else self.body(z)

    def forward(self, z):
        return self.out(self.features(z))

    @property
    def weight(self):  # used by the original code to derive class prototypes
        return self.out.weight


class Decision(nn.Module):
    """model.W in the original code. 'head': classifier logits. 'proto_avg': average of the
    log-probabilities of the head and of the cosine prototype classifier on proj_f(z)."""

    def __init__(self, head, proj_f, proj_h, mode, tau_p):
        super().__init__()
        self.head, self.proj_f, self.proj_h, self.mode, self.tau_p = head, proj_f, proj_h, mode, tau_p

    @property
    def weight(self):
        return self.head.weight

    def forward(self, z):
        lg = self.head(z)
        if self.mode == "head":
            return lg
        P = F.normalize(self.proj_h(self.head.weight), dim=-1)
        lp = F.normalize(self.proj_f(z), dim=-1) @ P.T / self.tau_p
        return 0.5 * (F.log_softmax(lg, -1) + F.log_softmax(lp, -1))


class PPModel(nn.Module):
    def __init__(self, shapes, cfg, C=2):
        super().__init__()
        d, ds = HP["d"], HP["ds"]
        self.cfg, self.M = cfg, len(shapes)
        self.mix = nn.ModuleList([base.LayerMix(L, D, d, cfg.get("indepA", False)) for L, D in shapes])
        self.enc_ln = nn.ModuleList([nn.LayerNorm(d) for _ in shapes]) if cfg["enc_norm"] else None
        fu = cfg["fusion"]
        if self.M > 1:
            gin = {"both": self.M * d + ds, "e": self.M * d, "s": ds}[cfg["gate_input"]]
            if fu in ("gate", "gated_residual"):
                self.gate = nn.Sequential(nn.Linear(gin, cfg["gate_hidden"]), nn.ReLU(), nn.Linear(cfg["gate_hidden"], self.M))
                if cfg["gate_zero_init"]:
                    nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
            if fu in ("static", "gated_residual"):
                self.static = nn.Linear(self.M * d, d)
            if fu == "concat_mlp":
                self.cmlp = nn.Sequential(nn.Linear(self.M * d, d), nn.GELU(), nn.Dropout(cfg["head_dropout"]), nn.Linear(d, d))
            if fu == "wsum":
                self.w_logits = nn.Parameter(torch.zeros(self.M))
            if fu == "attn":
                self.q = nn.Linear(d + ds, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
                if cfg["gate_zero_init"]:
                    nn.init.zeros_(self.q.weight); nn.init.zeros_(self.q.bias)
            if cfg["gate_tau_learn"]:
                self.log_tau = nn.Parameter(torch.tensor(math.log(cfg["gate_tau"])))
        head = Head(d, C, cfg["head"], cfg["head_dropout"])
        self.proj_f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_h = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))
        self.W = Decision(head, self.proj_f, self.proj_h, cfg["decision"], cfg["tau_p"])

    def tau_g(self):
        if self.cfg["gate_tau_learn"]:
            return self.log_tau.exp().clamp(min=0.1, max=20.0)
        return self.cfg["gate_tau"]

    def _alpha(self, logits):
        c = self.cfg
        if self.training and c["gate_noise"] > 0:
            logits = logits + c["gate_noise"] * torch.randn_like(logits)
        if self.training and c["enc_dropout"] > 0:
            mask = torch.rand_like(logits) < c["enc_dropout"]
            full = mask.all(1)
            if full.any():  # keep at least one encoder per example
                keep = torch.randint(self.M, (int(full.sum()),), device=logits.device)
                mask[full.nonzero().flatten(), keep] = False
            logits = logits.masked_fill(mask, -1e9)
        a = torch.softmax(logits / self.tau_g(), -1)
        if c["gate_floor"] > 0:
            a = (1 - c["gate_floor"]) * a + c["gate_floor"] / self.M
        return a

    def forward(self, Us, s, tau_beta=0.5):
        es = [m(U.to(DEV).float(), tau_beta, HP["layer_drop"])[0] for m, U in zip(self.mix, Us)]
        if self.enc_ln is not None:
            es = [ln(e) for ln, e in zip(self.enc_ln, es)]
        B = es[0].shape[0]
        if self.M == 1:
            return es[0], torch.ones(B, 1, device=DEV), es
        fu, gi = self.cfg["fusion"], self.cfg["gate_input"]
        E = torch.stack(es, 1)
        unif = torch.full((B, self.M), 1.0 / self.M, device=DEV)
        if fu == "static":
            return self.static(torch.cat(es, -1)), unif, es
        if fu == "concat_mlp":
            return self.cmlp(torch.cat(es, -1)), unif, es
        if fu == "wsum":
            a = torch.softmax(self.w_logits / self.tau_g(), -1)[None].expand(B, -1)
            return torch.einsum("bm,bmd->bd", a, E), a, es
        if fu == "attn":
            q = self.q(torch.cat([E.mean(1), s.to(DEV)], -1))
            logits = torch.einsum("bd,bmd->bm", q, self.k(E)) / math.sqrt(E.shape[-1])
            a = self._alpha(logits)
            return torch.einsum("bm,bmd->bd", a, self.v(E)), a, es
        parts = []
        if gi in ("both", "e"):
            parts += [F.layer_norm(e, e.shape[-1:]) for e in es]
        if gi in ("both", "s"):
            parts.append(s.to(DEV))
        a = self._alpha(self.gate(torch.cat(parts, -1)))
        z = torch.einsum("bm,bmd->bd", a, E)
        if fu == "gated_residual":
            z = z + self.static(torch.cat(es, -1))
        return z, a, es

    def prototypes(self):
        return F.normalize(self.proj_h(self.W.weight), dim=-1)


# ================================================================== trainer
class PPTrainer(base.DiaFuseTrainer):
    def __init__(self, store, td, cfg, seed, encs):
        self.store, self.td, self.flags, self.cfg, self.seed, self.encs = store, td, cfg, cfg, seed, encs
        self.input_mode = "layers"; self._cache = {}
        base.set_seed(seed)
        shapes = [(U.shape[1], U.shape[2]) for U in self._inputs("train_ar", [0])]
        self.model = PPModel(shapes, cfg).to(DEV)
        self.log_prior = torch.log(torch.tensor(td.prior, dtype=torch.float32, device=DEV).clamp(min=1e-6))
        self.delta = torch.zeros(2, 2, 128, device=DEV)
        cnt = torch.tensor(td.cell_counts, dtype=torch.float32, device=DEV)
        kappa = cfg["kappa"]
        self.r_cv = cnt / (cnt + kappa) if math.isfinite(kappa) else torch.zeros_like(cnt)
        self.cell_prior = cnt / cnt.sum()
        self.epoch_log = []

    def _inputs(self, s, idx=None):
        if self.cfg["inputs"] == "last_default":
            out = []
            for e in self.encs:
                if (s, e) not in self._cache:
                    self._cache[(s, e)] = self.store.pooled(s, e)  # [N, d_m], default pooling, last layer
                X = self._cache[(s, e)]
                X = X if idx is None else X[torch.as_tensor(idx)]
                out.append(X.unsqueeze(1))
            return out
        return base.stack_layers(self.store, s, self.encs, idx, self.cfg["layer_mode"])

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def diagnostics(self, tau_beta):
        out = {}
        self.model.eval()
        for s in self.dev_sets:
            n = len(self.store.ids(s)); A, norms = [], []
            for i in range(0, n, 512):
                idx = torch.arange(i, min(n, i + 512))
                _, a, es = self.model(self._inputs(s, idx), self._surf(s, idx), tau_beta)
                A.append(a); norms.append(torch.stack([e.norm(dim=-1) for e in es], 1))
            A = torch.cat(A); Nm = torch.cat(norms)
            ent = -(A * torch.log(A + 1e-12)).sum(-1)
            arg = torch.bincount(A.argmax(-1), minlength=A.shape[1]).float() / len(A)
            out[s] = {"alpha_mean": A.mean(0).tolist(), "entropy_mean": float(ent.mean()), "entropy_max": math.log(A.shape[1]),
                      "argmax_share": arg.tolist(), "e_norm_mean": Nm.mean(0).tolist()}
        p = self.model.prototypes()
        out["proto_cos_c0_c1"] = float((p[0] * p[1]).sum())
        out["layer_beta"] = [torch.softmax(m.omega / tau_beta, 0).tolist() for m in self.model.mix]
        if self.model.M > 1 and len(self.dev_sets) > 1:  # does routing depend on the script (variety)?
            a0 = torch.tensor(out[self.dev_sets[0]]["alpha_mean"]); a1 = torch.tensor(out[self.dev_sets[1]]["alpha_mean"])
            out["alpha_l1_between_dev_views"] = float((a0 - a1).abs().sum())
        return out

    # ------------------------------------------------------------ training (original fit + gate terms, schedule, weights)
    def fit(self, dev_sets):
        self.dev_sets = dev_sets
        td, fl, c = self.td, self.flags, self.cfg
        E = c["epochs"]; p1, p2 = c["p1"], c["p2"]
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=c["lr"], weight_decay=c["wd"])
        n = len(td.items); y_all = torch.tensor(td.label, device=DEV); v_all = torch.tensor(td.variety, device=DEV)
        steps_per_ep = math.ceil(n / c["batch"]); total = E * steps_per_ep; wsteps = int(c["warmup"] * total)
        if c["sched"] == "cosine":
            lam = lambda t: (t + 1) / max(1, wsteps) if t < wsteps else 0.5 * (1 + math.cos(math.pi * (t - wsteps) / max(1, total - wsteps)))
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        else:
            sch = None
        flagged = np.zeros(n); best, best_state = -1, None
        gated = self.model.M > 1 and c["fusion"] in GATED
        for ep in range(E):
            tau_beta = HP["tau_beta0"] - (HP["tau_beta0"] - HP["tau_beta1"]) * ep / max(1, E - 1)
            phase = 1 if ep < p1 else (2 if ep < p2 else 3)
            cal_on = c["cal"] and phase == 3
            perm = rng.permutation(n)
            cal_slice = perm[: n // 10] if cal_on else np.array([], int)
            main = perm[n // 10:] if cal_on else perm
            bank_z = None; syn = None; boundary = []; hs = None
            if phase >= 2:
                with torch.no_grad():
                    Z = []
                    for i in range(0, n, 512):
                        z, _ = self.forward_items(td.items[i:i + 512], tau_beta, train=False); Z.append(z)
                    bank_z = torch.cat(Z)
                    loss_i = F.cross_entropy(self.model.W(bank_z) + HP["tau_la"] * self.log_prior, y_all, reduction="none")
                    flagged += (loss_i >= torch.quantile(loss_i, 0.98)).cpu().numpy()
                noisy = torch.tensor(flagged > (ep + 1) / 2, device=DEV)
                if phase == 3 and c["hm"]:
                    hs = self.hard_sets(bank_z, y_all, v_all, noisy)
            r_t = 0.5 + 0.5 * (ep - p2) / max(1, E - p2) if phase == 3 else 0.0
            ramp = min(1.0, (ep - p1 + 1) / max(1, p2 - p1)) if phase >= 2 else 0.0
            agg = {"ce": 0.0, "gate": 0.0, "n": 0}
            for b in range(0, len(main), c["batch"]):
                bi = main[b:b + c["batch"]]
                items = [td.items[i] for i in bi]
                y = y_all[bi]; v = v_all[bi]
                z, alpha = self.forward_items(items, tau_beta)
                logits = self.model.W(z)
                loss = self.ce_la(logits, y); agg["ce"] += float(loss); agg["n"] += 1
                if gated:
                    abar = alpha.mean(0)
                    g = c["lam_lb"] * (abar * torch.log(abar * self.model.M + 1e-8)).sum()
                    if c["lam_ent"] > 0:
                        g = g + c["lam_ent"] * (alpha * torch.log(alpha + 1e-8)).sum(-1).mean()
                    loss = loss + g; agg["gate"] += float(g)
                if phase >= 2:
                    zh = F.normalize(self.model.proj_f(z), dim=-1)
                    with torch.no_grad():
                        pc = self.model.prototypes()
                        for cc in range(2):
                            for vv in range(td.n_var):
                                m = (y == cc) & (v == vv)
                                if m.any():
                                    self.delta[cc, vv] = HP["mu"] * self.delta[cc, vv] + (1 - HP["mu"]) * (zh[m] - pc[cc]).mean(0)
                    if c["proto"] != "none" and c["lam_p"] > 0:
                        P = self.proto_cv(); vv = v.clamp(max=P.shape[1] - 1)
                        logits_p = torch.einsum("bk,bck->bc", zh, P[:, vv].permute(1, 0, 2)) / c["tau_p"]
                        loss = loss + c["lam_p"] * F.cross_entropy(logits_p, y)
                    if phase == 3 and c["hm"]:
                        syn, boundary = self.make_mixes(hs, bank_z, y_all, v_all)
                    if c["lam_cl"] > 0:
                        loss = loss + c["lam_cl"] * self.cl_loss(zh, y, v, bank_z, y_all, v_all, syn if phase == 3 else None, r_t)
                    if c["cons"]:
                        pairs = [(k, td.partner(vw, i, rng)) for k, (vw, i) in enumerate(items)]
                        pairs = [(k, p) for k, p in pairs if p is not None]
                        if pairs:
                            ks = torch.tensor([k for k, _ in pairs], device=DEV)
                            z2, _ = self.forward_items([p for _, p in pairs], tau_beta)
                            lc = 0
                            if c["cons_terms"] in ("both", "emb"):
                                lc = lc + (1 - F.cosine_similarity(z[ks], z2, dim=-1))
                            if c["cons_terms"] in ("both", "pred"):
                                lc = lc + HP["eta"] * base.js_div(torch.softmax(logits[ks], -1), torch.softmax(self.model.W(z2), -1))
                            loss = loss + c["lam_cons"] * ramp * lc.mean()
                if phase == 3 and c["hm"] and boundary:
                    zt = torch.cat([b_[0] for b_ in boundary]); yt = torch.cat([b_[1] for b_ in boundary])
                    loss = loss + c["lam_hm"] * (-(yt * F.log_softmax(self.model.W(zt), -1)).sum(-1).mean())
                if cal_on and len(cal_slice):
                    ci = cal_slice[rng.choice(len(cal_slice), min(c["batch"], len(cal_slice)), replace=False)]
                    zc, _ = self.forward_items([td.items[i] for i in ci], tau_beta)
                    lg = self.model.W(zc); pr = torch.softmax(lg + HP["tau_la"] * self.log_prior, -1)
                    pred = lg.argmax(-1); conf = pr.gather(1, pred[:, None]).squeeze(1)
                    loss = loss + c["lam_cal"] * base.mmce(conf, (pred == y_all[ci]).float())
                opt.zero_grad(); loss.backward(); opt.step()
                if sch is not None:
                    sch.step()
            rec = {"epoch": ep, "phase": phase, "ce": agg["ce"] / max(1, agg["n"]), "gate_reg": agg["gate"] / max(1, agg["n"])}
            if ep >= p1 - 1:
                sc = self.dev_score(tau_beta); rec["dev"] = sc
                if sc > best:
                    best, best_state = sc, (copy.deepcopy(self.model.state_dict()), self.delta.clone(), tau_beta, ep)
            rec["diag"] = self.diagnostics(tau_beta)
            self.epoch_log.append(rec)
        self.model.load_state_dict(best_state[0]); self.delta = best_state[1]; self.tau_beta = best_state[2]
        self.best_epoch = best_state[3]; self.dev = best
        return self


# ================================================================== runner
def make_runner(args):
    ns = argparse.Namespace(data=args.data, feats=args.feats, out=args.out, encoders=args.encoders, mode="quick")
    return base.Runner(ns)


def build(R, cfg, seed):
    td = base.TrainData(R.store, seed=seed, families=tuple(cfg["families"]))
    encs = list(cfg.get("encoders") or R.encs[:3])
    if cfg["impl"] == "orig":
        return base.DiaFuseTrainer(R.store, td, {"fusion": "gate"}, seed, encs), td
    return PPTrainer(R.store, td, cfg, seed, encs), td


def orig_diagnostics(tr):
    """Post-fit gate statistics for the original trainer (no per-epoch log exists in the original code)."""
    out = {}
    for s in tr.dev_sets:
        _, A = tr.encode_set(s, tr.tau_beta)
        ent = -(A * torch.log(A + 1e-12)).sum(-1)
        out[s] = {"alpha_mean": A.mean(0).tolist(), "entropy_mean": float(ent.mean()),
                  "argmax_share": (torch.bincount(A.argmax(-1), minlength=A.shape[1]).float() / len(A)).tolist()}
    return out


def cmd_search(args):
    R = make_runner(args); cfgs = load_configs(args.configs)
    names = args.only.split(",") if args.only else list(cfgs)
    for name in names:
        cfg = cfgs[name]
        for seed in parse_seeds(args.seeds):
            p = os.path.join(args.pp_out, "dev", name, f"{seed}.json")
            if os.path.exists(p):
                continue
            t0 = time.time()
            tr, _ = build(R, cfg, seed)
            tr.fit(list(cfg["dev_sets"]))
            rec = {"config": name, "family": cfg["_family"], "seed": seed, "dev": tr.dev,
                   "minutes": (time.time() - t0) / 60, "trainable_params": sum(q.numel() for q in tr.model.parameters() if q.requires_grad),
                   "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}}
            if cfg["impl"] == "orig":
                rec["post_fit_diag"] = orig_diagnostics(tr)
            else:
                rec["best_epoch"] = tr.best_epoch; rec["epochs"] = tr.epoch_log
            os.makedirs(os.path.dirname(p), exist_ok=True); _wjson(rec, p)
            print(f"[search] {name} seed {seed}: dev {tr.dev:.4f} ({rec['minutes']:.1f} min)", flush=True)
            del tr; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def cmd_final(args):
    """Train/calibrate/predict ONCE per (config, seed). Test predictions are written here only."""
    R = make_runner(args); cfgs = load_configs(args.configs)
    names = args.only.split(",") if args.only else list(cfgs)
    for name in names:
        cfg = cfgs[name]; system = args.system if (args.system and len(names) == 1) else name
        for seed in parse_seeds(args.seeds):
            if R.done(system, seed):
                continue
            t0 = time.time()
            tr, td = build(R, cfg, seed)
            tr.fit(list(cfg["dev_sets"])).calibrate()
            R.log_dev(system, seed, tr.dev)
            for s in R.eval_sets:
                res = tr.predict(s, variety_wise=td.n_var > 1)
                R.write_preds(system, seed, s, res); R.write_diag(system, seed, s, res)
            meta = {"config": name, "seed": seed, "dev": tr.dev, "minutes": (time.time() - t0) / 60,
                    "trainable_params": sum(q.numel() for q in tr.model.parameters() if q.requires_grad),
                    "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}}
            if cfg["impl"] != "orig":
                meta["epochs"] = tr.epoch_log; meta["best_epoch"] = tr.best_epoch
            if args.ckpt_dir:  # final model checkpoint (heads only: encoders are frozen in Setting A)
                cp = os.path.join(args.ckpt_dir, system, f"seed{seed}.pt"); os.makedirs(os.path.dirname(cp), exist_ok=True)
                torch.save({"state_dict": tr.model.state_dict(), "delta": tr.delta.cpu(), "tau_beta": tr.tau_beta,
                            "config": meta["cfg"], "seed": seed}, cp)
                meta["checkpoint"] = cp
            p = os.path.join(args.pp_out, "final", system, f"{seed}.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
            _wjson(meta, p)
            print(f"[final] {system} seed {seed}: dev {tr.dev:.4f} ({meta['minutes']:.1f} min)", flush=True)
            del tr; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "final"])
    ap.add_argument("--configs", required=True); ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--system", default=None)
    ap.add_argument("--data", default="data"); ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="runs"); ap.add_argument("--pp_out", default="runs_pp")
    ap.add_argument("--encoders", default="xlmr,me5,labse"); ap.add_argument("--ckpt_dir", default=None)
    args = ap.parse_args()
    (cmd_search if args.cmd == "search" else cmd_final)(args)


if __name__ == "__main__":
    main()
