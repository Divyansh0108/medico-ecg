"""Train on PTB-XL folds 1-8, early-stop on fold 9 macro-AUROC; fold 10 only with --eval-test.

Usage: python train.py --model M1 --seed 0 --eval-test                  (original full-length runs)
       python train.py --model xresnet1d101 --crop 250 --tag xresnet1d101_crop --out results/crop
--crop L: random L-sample crop per record per epoch; val/test use sliding windows (length L, --stride)
with per-record averaging of probabilities (--agg).
--aug: Track 2 corruption augmentation (corruptions.py) on the band-passed record before standardization,
p=0.5, seen families, SNR U[0,20] dB; noise RNG seeded by (seed, epoch). --aux (variant F): clean and
corrupted views of every record + consistency (>= --cons-min-snr only) and severity-ranking losses on r.
--avg swa|ema: no early stopping; the final model is the weight average over the last --avg-epochs epochs
(swa: mean of epoch-end weights, ema: per-step EMA), BatchNorm statistics recomputed on train.
Writes {out}/{tag}.json, {out}/probs/{tag}.npz and checkpoints/{tag}.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

import nstdb
from corruptions import FAMILIES, SEEN, TRAIN_P, random_seen
from data import SUPERCLASSES, load_filtered, load_variant, set_seed, standardize
from metrics import macro_auroc, per_class_auroc
from models import MODELS
from strodthoff import STRODTHOFF
from track2_models import T2_MODELS

ALL_MODELS = {**MODELS, **STRODTHOFF, **T2_MODELS}

HERE = os.path.dirname(os.path.abspath(__file__))


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=HERE, text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def code_sha() -> str:
    """sha256 over this directory's .py files (the medico repo has no commits yet)."""
    h = hashlib.sha256()
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py"):
            h.update(f.encode() + open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:12]


@torch.no_grad()
def predict(model, X: torch.Tensor, device, bs: int = 256, crop: int = 0, stride: int = 125,
            agg: str = "mean") -> np.ndarray:
    """Full-length prediction, or sliding windows (crop, stride) aggregated per record."""
    model.eval()
    if not crop:
        out = [torch.sigmoid(model(X[i:i + bs].to(device))).cpu() for i in range(0, len(X), bs)]
        return torch.cat(out).numpy()
    starts = list(range(0, X.shape[-1] - crop + 1, stride))
    out = []
    for i in range(0, len(X), max(1, bs // len(starts))):
        xb = X[i:i + max(1, bs // len(starts))].to(device)
        w = torch.stack([xb[..., s:s + crop] for s in starts], 1)            # (B, W, C, L)
        p = torch.sigmoid(model(w.flatten(0, 1))).view(len(xb), len(starts), -1)
        out.append((p.mean(1) if agg == "mean" else p.amax(1)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def update_bn(model, X: torch.Tensor, bs: int, L: int, g: torch.Generator, device) -> None:
    """Recompute BatchNorm running stats for averaged weights: one pass over train (random crops, no mixup)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                                                    # cumulative average
    model.train()
    perm = torch.randperm(len(X), generator=g)
    for i in range(0, len(X), bs):
        x = X[perm[i:i + bs].to(device)]
        if L < x.shape[-1]:
            st = torch.randint(0, x.shape[-1] - L + 1, (len(x), 1, 1), generator=g).to(device)
            x = x.gather(2, (st + torch.arange(L, device=device)).expand(-1, x.shape[1], -1))
        model(x)
    for m, v in zip(bns, mom):
        m.momentum = v


def build_model(name: str, leads=None) -> nn.Module:
    return ALL_MODELS[name]() if leads is None else ALL_MODELS[name](in_ch=len(leads))


def get_data(a):
    d = load_variant(use_bandpass=not a.no_bandpass, norm=a.norm, leads=getattr(a, "leads", None))
    return d["train"], d["val"], d["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ALL_MODELS), required=True)
    ap.add_argument("--tag", default="", help="run name (default {model}_s{seed})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--sched", choices=["cosine", "onecycle"], default="cosine",
                    help="onecycle: torch OneCycleLR, peak --lr, pct_start 0.3 (fastai fit_one_cycle defaults)")
    ap.add_argument("--crop", type=int, default=0, help="train crop length in samples (0 = full 1000)")
    ap.add_argument("--stride", type=int, default=125, help="eval sliding-window stride")
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--no-bandpass", action="store_true")
    ap.add_argument("--norm", choices=["record", "dataset"], default="record")
    ap.add_argument("--label-smooth", type=float, default=0.0, help="targets y*(1-e) + 0.5*e (after mixup)")
    ap.add_argument("--avg", choices=["none", "swa", "ema"], default="none",
                    help="weight averaging over the last --avg-epochs epochs (disables early stopping)")
    ap.add_argument("--avg-epochs", type=int, default=10)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--aug", action="store_true", help="Track 2 corruption augmentation (p=0.5, seen families)")
    ap.add_argument("--aux", action="store_true", help="variant F losses (implies --aug; model must have a gate)")
    ap.add_argument("--aug-real", action="store_true",
                    help="corrupted copy drawn 50/50 from synthetic seen families and NSTDB TRAIN noise (RULES2.md 1)")
    ap.add_argument("--aug-families", choices=["seen", "all"], default="seen",
                    help="all: synthetic augmentation draws from all five families (RULES3.md ablation)")
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-sev", type=float, default=0.1)
    ap.add_argument("--sev-margin", type=float, default=0.05)
    ap.add_argument("--cons-min-snr", type=float, default=15.0)
    ap.add_argument("--leads", type=int, nargs="+", default=None,
                    help="input lead indices (default all 12; [0] = lead I); dataset norm is fitted on these leads")
    ap.add_argument("--eval-test", action="store_true", help="also predict fold 10 (off: fold 10 untouched)")
    ap.add_argument("--limit", type=int, default=0, help="debug: subsample train set")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    tag = a.tag or f"{a.model}_s{a.seed}"
    (Xtr, Ytr, _), (Xva, Yva, mva), (Xte, Yte, mte) = get_data(a)
    if a.limit:
        Xtr, Ytr = Xtr[: a.limit], Ytr[: a.limit]
    Xtr_t, Ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)
    Xva_t, Xte_t = torch.from_numpy(Xva), torch.from_numpy(Xte)
    a.aug = a.aug or a.aux or a.aug_real

    def draw(x, rng):
        if a.aug_real and rng.random() < 0.5:
            return nstdb.random_train(x, rng)
        return random_seen(x, rng, FAMILIES if a.aug_families == "all" else SEEN)
    if a.aug:   # band-passed, unstandardized train records: noise is added here, then standardized
        assert a.norm == "dataset" and not a.no_bandpass, "--aug needs --norm dataset with band-pass"
        dfl, mu_t, sd_t = load_filtered(leads=a.leads)
        Xtr_raw = dfl["train"][0][: len(Xtr)].astype(np.float32)
        del dfl
        assert np.abs(standardize(Xtr_raw[:16], mu_t, sd_t) - Xtr[:16]).max() < 1e-4

    torch.manual_seed(a.seed)
    model = build_model(a.model, a.leads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pos = Ytr.sum(0)
    pos_weight = torch.tensor((len(Ytr) - pos) / pos, dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(Xtr) / a.bs)
    total, warm = a.epochs * steps_per_epoch, max(1, int(a.warmup * a.epochs * steps_per_epoch))
    if a.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.3)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gh, cs = git_hash(), code_sha()
    ev = dict(crop=a.crop, stride=a.stride, agg=a.agg)
    L = a.crop or Xtr.shape[-1]
    print(f"[{tag}] device={device} params={n_params} train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"pos_weight={pos_weight.cpu().numpy().round(3).tolist()} git={gh} code={cs}", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    beta = torch.distributions.Beta(torch.tensor(a.mixup), torch.tensor(a.mixup)) if a.mixup > 0 else None
    torch.manual_seed(a.seed)
    best, best_ep, bad, hist = -1.0, -1, 0, []
    avg, n_avg, avg_start = None, 0, a.epochs - a.avg_epochs + 1
    ckpt = os.path.join(HERE, "checkpoints", f"{tag}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        rng = np.random.default_rng([a.seed, ep, 7])          # corruption noise; separate from g
        tl, aux_sum = 0.0, np.zeros(2)
        for i in range(0, len(Xtr), a.bs):
            idx_np = perm[i:i + a.bs].numpy()
            idx = perm[i:i + a.bs].to(device)
            x, y = Xtr_t[idx], Ytr_t[idx]
            xn = None
            if a.aux:            # corrupted view of every record; BCE sees it with p=0.5
                cor = [draw(Xtr_raw[k], rng) for k in idx_np]
                xn = torch.from_numpy(standardize(np.stack([c[0] for c in cor]), mu_t, sd_t)).to(device)
                snr_t = torch.tensor([c[1] for c in cor], device=device)
                use_n = torch.from_numpy(rng.random(len(idx_np)) < TRAIN_P).to(device)
            elif a.aug:
                sel = np.flatnonzero(rng.random(len(idx_np)) < TRAIN_P)
                if len(sel):
                    xc = standardize(np.stack([draw(Xtr_raw[idx_np[k]], rng)[0] for k in sel]), mu_t, sd_t)
                    x = x.clone()
                    x[torch.from_numpy(sel).to(device)] = torch.from_numpy(xc).to(device)
            if a.crop:
                st = torch.randint(0, x.shape[-1] - L + 1, (len(idx), 1, 1), generator=g).to(device)
                gi = (st + torch.arange(L, device=device)).expand(-1, x.shape[1], -1)
                x = x.gather(2, gi)
                xn = None if xn is None else xn.gather(2, gi)
            if a.mixup > 0:
                lam = beta.sample().item()
                j = torch.randperm(len(idx), generator=g).to(device)
                x, y = lam * x + (1 - lam) * x[j], lam * y + (1 - lam) * y[j]
                xn = None if xn is None else lam * xn + (1 - lam) * xn[j]
            if a.label_smooth > 0:
                y = y * (1 - a.label_smooth) + 0.5 * a.label_smooth
            if a.aux:
                out = model(torch.cat([x, xn]))
                lc, ln = out[:len(idx)], out[len(idx):]
                rc, rn = model.last_r[:len(idx)], model.last_r[len(idx):]
                mild = snr_t >= a.cons_min_snr         # consistency only against mild (>= 15 dB) copies
                l_cons = nn.functional.mse_loss(lc[mild], ln[mild]) if mild.any() else lc.sum() * 0
                l_sev = torch.relu(rn.mean() - rc.mean() + a.sev_margin)
                loss = crit(torch.where(use_n[:, None], ln, lc), y) + a.w_cons * l_cons + a.w_sev * l_sev
                aux_sum += [l_cons.item() * len(idx), l_sev.item() * len(idx)]
            else:
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            if a.avg == "ema" and ep >= avg_start:
                with torch.no_grad():
                    if avg is None:
                        avg = [p.detach().clone() for p in model.parameters()]
                    else:
                        for e, p in zip(avg, model.parameters()):
                            e.lerp_(p, 1 - a.ema_decay)
        if a.avg == "swa" and ep >= avg_start:
            with torch.no_grad():
                n_avg += 1
                if avg is None:
                    avg = [p.detach().clone() for p in model.parameters()]
                else:
                    for e, p in zip(avg, model.parameters()):
                        e.lerp_(p, 1 / n_avg)
        pva = predict(model, Xva_t, device, **ev)
        auc = macro_auroc(Yva, pva)
        hist.append({"epoch": ep, "train_loss": tl / len(Xtr), "val_macro_auroc": auc,
                     **({"l_cons": aux_sum[0] / len(Xtr), "l_sev": aux_sum[1] / len(Xtr)} if a.aux else {})})
        print(f"[{tag}] ep {ep:2d} loss {tl / len(Xtr):.4f} val_auroc {auc:.4f} "
              f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            torch.save(model.state_dict(), ckpt)
        elif a.avg == "none":
            bad += 1
            if bad >= a.patience:
                print(f"[{tag}] early stop at ep {ep}", flush=True)
                break

    train_time = round(time.time() - t0, 1)
    if a.avg != "none":   # final model = averaged weights (the best-epoch score stays in val_best_epoch_auroc)
        with torch.no_grad():
            for e, p in zip(avg, model.parameters()):
                p.copy_(e)
        update_bn(model, Xtr_t, a.bs, L, g, device)
        torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    pva = predict(model, Xva_t, device, **ev)
    probs = dict(val=pva, y_val=Yva, val_ecg_id=mva["ecg_id"].to_numpy(), val_patient_id=mva["patient_id"].to_numpy())
    res = {"model": a.model, "tag": tag, "seed": a.seed, "n_params": n_params, "git_hash": gh, "code_sha": cs,
           "device": str(device), "best_epoch": best_ep, "epochs_run": len(hist), "train_time_s": train_time,
           "val_macro_auroc": macro_auroc(Yva, pva), "val_best_epoch_auroc": best, "val_per_class_auroc": per_class_auroc(Yva, pva),
           "classes": SUPERCLASSES, "config": vars(a), "history": hist}
    if a.crop:   # alternative aggregation, fold 9 only (reported, not used for selection unless stated)
        res["val_macro_auroc_agg_" + ("max" if a.agg == "mean" else "mean")] = macro_auroc(
            Yva, predict(model, Xva_t, device, **{**ev, "agg": "max" if a.agg == "mean" else "mean"}))
    if a.eval_test:
        pte = predict(model, Xte_t, device, **ev)   # single test evaluation
        probs.update(test=pte, y_test=Yte, test_ecg_id=mte["ecg_id"].to_numpy(),
                     test_patient_id=mte["patient_id"].to_numpy())
        res.update(test_macro_auroc=macro_auroc(Yte, pte), test_per_class_auroc=per_class_auroc(Yte, pte))
    os.makedirs(os.path.join(a.out, "probs"), exist_ok=True)
    np.savez(os.path.join(a.out, "probs", f"{tag}.npz"), **probs)
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] DONE best_ep {best_ep} val {res['val_macro_auroc']:.4f} "
          f"test {res.get('test_macro_auroc', float('nan')):.4f} params {n_params} time {train_time}s", flush=True)


if __name__ == "__main__":
    main()
