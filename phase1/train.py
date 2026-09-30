"""Phase 1 training (docs/master.md B4/B5, docs/CONTRACT.md).

    python train.py --config configs/phase1.yaml --variant {A..G} --regime {clean,aug} --seed S [--smoke]

Writes runs/{variant}_{regime}_s{seed}/: best.pt, config.yaml, meta.json, log.csv,
thresholds.json. A run whose best.pt and meta.json both exist is skipped
(meta.json is written last, so an interrupted run is redone from scratch).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve

REPO = Path(__file__).resolve().parent
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # CONTRACT label order


# --------------------------------------------------------------------------- variants
@dataclass(frozen=True)
class VariantSpec:
    model: str          # models.build_model name
    paired: bool        # clean + corrupted copy of every ECG in the batch
    rank: bool          # margin ranking loss on r (lambda2)
    consistency: bool   # mild-pair logit consistency (lambda1)


VARIANTS: dict[str, VariantSpec] = {
    "A": VariantSpec("cnn", False, False, False),
    "B": VariantSpec("tcn", False, False, False),
    "C": VariantSpec("concat", False, False, False),
    "D": VariantSpec("gate", False, False, False),
    "E": VariantSpec("racer", False, False, False),
    "F": VariantSpec("racer", True, True, True),
    "G": VariantSpec("cnn", True, False, True),
}
REGIMES = ("clean", "aug")


def get_variant(variant: str, regime: str) -> VariantSpec:
    """Validated variant lookup. Paired variants (F, G) exist only in regime aug."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")
    spec = VARIANTS[variant]
    if spec.paired and regime != "aug":
        raise ValueError(f"variant {variant} uses paired corrupted batches and is only valid with regime 'aug'")
    return spec


def sweep_jobs(cfg: dict) -> list[tuple[str, str]]:
    """(variant, regime) pairs of the configured sweep; paired variants only in regime aug."""
    return [(v, r) for v in cfg["variants"] for r in REGIMES if not (VARIANTS[v].paired and r != "aug")]


def needs_nstdb(cfg: dict) -> bool:
    c = cfg["corruptions"]
    return any(f.startswith("nstdb_") for f in c["train_families"] + c["unseen_families"])


def run_name(variant: str, regime: str, seed: int) -> str:
    return f"{variant}_{regime}_s{seed}"


# --------------------------------------------------------------------------- config
def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, smoke: bool = False) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg.pop("smoke", {}) or {}
    if smoke:
        _deep_update(cfg, copy.deepcopy(overrides))
    cfg["smoke"] = bool(smoke)
    return cfg


# --------------------------------------------------------------------------- losses
def rank_loss(r_clean: torch.Tensor, r_corr: torch.Tensor, margin: float) -> torch.Tensor:
    """mean(relu(mean_t r_corr - mean_t r_clean + margin)); r tensors [B, T']."""
    m_clean = r_clean.float().mean(dim=1)
    m_corr = r_corr.float().mean(dim=1)
    return F.relu(m_corr - m_clean + margin).mean()


def consistency_loss(logits_clean: torch.Tensor, logits_corr: torch.Tensor,
                     severity: torch.Tensor) -> torch.Tensor:
    """MSE between sigmoid(logits) of clean and corrupted views, ONLY on mild (severity 0)
    pairs; gradient flows through both views. Exactly zero if the batch has no mild pair."""
    mild = severity.to(logits_clean.device) == 0
    if not bool(mild.any()):
        return logits_clean.float().sum() * 0.0  # zero, keeps the graph valid
    pc = torch.sigmoid(logits_clean.float()[mild])
    px = torch.sigmoid(logits_corr.float()[mild])
    return F.mse_loss(px, pc)


def compute_loss(model, batch: dict, spec: VariantSpec, tcfg: dict, device: torch.device,
                 amp_dtype: torch.dtype | None) -> tuple[torch.Tensor, dict[str, float]]:
    autocast = torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                              enabled=amp_dtype is not None)
    y = batch["y"].to(device, non_blocking=True).float()
    if not spec.paired:
        with autocast:
            out = model(batch["x"].to(device, non_blocking=True))
        bce = F.binary_cross_entropy_with_logits(out["logits"].float(), y)
        return bce, {"bce": bce.item()}

    xc = batch["x_clean"].to(device, non_blocking=True)
    xr = batch["x_corr"].to(device, non_blocking=True)
    b = xc.shape[0]
    with autocast:  # one forward over both views: BN statistics see the same 50/50 mix as regime aug
        out = model(torch.cat([xc, xr], dim=0))
    logits = out["logits"].float()
    bce = F.binary_cross_entropy_with_logits(logits, torch.cat([y, y], dim=0))  # mean over both views
    loss, parts = bce, {"bce": bce.item()}
    if spec.rank:
        r = out["r"]
        if r is None:
            raise RuntimeError(f"rank loss needs a reliability map; model {spec.model} returned r=None")
        lr = rank_loss(r[:b], r[b:], tcfg["margin"])
        loss = loss + tcfg["lambda2"] * lr
        parts["rank"] = lr.item()
    if spec.consistency:
        lc = consistency_loss(logits[:b], logits[b:], batch["severity"])
        loss = loss + tcfg["lambda1"] * lc
        parts["consistency"] = lc.item()
    return loss, parts


# --------------------------------------------------------------------------- metrics
def auroc_per_class(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rank-based (Mann-Whitney) AUROC per column, ties averaged; identical to
    sklearn.roc_auc_score. NaN for a class without both labels present."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=np.float64)
    n = y.shape[0]
    n1 = y.sum(axis=0).astype(np.float64)
    n0 = n - n1
    ranks = rankdata(p, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks * y).sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc[(n1 == 0) | (n0 == 0)] = np.nan
    return auc


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    auc = auroc_per_class(y, p)
    return float(np.nanmean(auc)) if np.isfinite(auc).any() else float("nan")


def tune_thresholds(y: np.ndarray, p: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Per-class threshold maximising F1 (predict positive iff p >= t). Candidates are the
    distinct predicted scores; ties broken by the first (lowest) maximising threshold.
    `default` for a class with no positives."""
    th = np.full(y.shape[1], default, dtype=np.float64)
    for k in range(y.shape[1]):
        if y[:, k].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y[:, k], p[:, k])
        prec, rec = prec[:-1], rec[:-1]  # last point (P=1, R=0) has no threshold
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        th[k] = float(thr[int(np.argmax(f1))])
    return th


def macro_f1(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> float:
    """Unweighted mean over classes of F1 at the given per-class thresholds (F1=0 if undefined)."""
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(p) >= np.asarray(thresholds)[None, :]
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, x_pre: np.ndarray | torch.Tensor, device: torch.device,
            batch_size: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
    """fp32 inference on preprocessed signals [N, T] or [N, 1, T].
    Returns probs [N, K] and per-record mean reliability [N] (None if the model has no r)."""
    model.eval()
    x = torch.as_tensor(x_pre, dtype=torch.float32)
    if x.dim() == 2:
        x = x[:, None, :]
    probs, rs = [], []
    for i in range(0, x.shape[0], batch_size):
        out = model(x[i:i + batch_size].to(device))
        probs.append(torch.sigmoid(out["logits"].float()).cpu())
        if out.get("r") is not None:
            rs.append(out["r"].float().mean(dim=1).cpu())
    return torch.cat(probs).numpy(), (torch.cat(rs).numpy() if rs else None)


# --------------------------------------------------------------------------- infrastructure
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and MPS generators


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def select_amp_dtype(model_name: str, device: torch.device, enabled: bool,
                     fs: int = 100) -> tuple[torch.dtype | None, str]:
    """One-time check that fp16 autocast + GradScaler forward/backward/step works on MPS for
    this architecture (on a throwaway model and a private RNG, so the run's seeded state is
    untouched). Returns (dtype or None=fp32, note)."""
    if not enabled:
        return None, "fp32: amp disabled in config"
    if device.type != "mps":
        return None, f"fp32: autocast fp16 only used on MPS (device={device.type})"
    import models
    try:
        m = models.build_model(model_name, fs=fs).to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("mps")
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 1, 10 * fs, generator=g).to(device)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            out = m(x)
        loss = out["logits"].float().pow(2).mean()
        if out.get("r") is not None:
            loss = loss + out["r"].float().mean()
        if not torch.isfinite(loss):
            return None, "fp32: fp16 check produced a non-finite loss"
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.mps.synchronize()
        return torch.float16, "fp16 autocast on MPS (check passed)"
    except Exception as e:  # noqa: BLE001 - any failure means fall back
        return None, f"fp32: fp16 autocast check failed ({type(e).__name__}: {e})"


def git_info() -> dict:
    def _run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    try:
        h = _run("rev-parse", "HEAD")
        s = _run("status", "--porcelain")
    except OSError:
        return {"git_hash": "nogit", "git_dirty": None}
    return {"git_hash": h.stdout.strip() if h.returncode == 0 else "nogit",
            "git_dirty": bool(s.stdout.strip()) if s.returncode == 0 else None}


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "best.pt").exists() and (run_dir / "meta.json").exists()


# --------------------------------------------------------------------------- training
def train(cfg: dict, variant: str, regime: str, seed: int) -> Path:
    spec = get_variant(variant, regime)
    run_dir = REPO / cfg["paths"]["runs_dir"] / run_name(variant, regime, seed)
    if is_complete(run_dir):
        print(f"[skip] {run_dir} already complete")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    import data
    import models
    from corruptions import Corruptor

    t0 = time.time()
    P, D, T, C = cfg["paths"], cfg["data"], cfg["train"], cfg["corruptions"]
    fs = int(D["fs"])
    device = get_device()
    amp_dtype, amp_note = select_amp_dtype(spec.model, device, T["amp"], fs)
    print(f"[{run_name(variant, regime, seed)}] device={device} amp={amp_note}")

    seed_everything(seed)
    root = str(REPO / P["ptbxl_root"])
    cache = str(REPO / P["cache_dir"])
    Xtr, Ytr, _ = data.load_ptbxl(root, D["train_folds"], lead=D["lead"], cache_dir=cache, fs=fs)
    Xva, Yva, _ = data.load_ptbxl(root, D["val_folds"], lead=D["lead"], cache_dir=cache, fs=fs)
    Xva_pre = data.preprocess(Xva, fs)  # clean validation, deterministic: preprocess once

    corruptor = None
    if regime == "aug":
        nstdb = None
        if needs_nstdb(cfg):  # NSTDB TRAIN split only (first 60% in time)
            nstdb = data.load_nstdb(str(REPO / P["nstdb_root"]), target_fs=fs, train_frac=D["nstdb_train_frac"])
        corruptor = Corruptor(nstdb, "train", fs=fs, train_families=C["train_families"])
    ds = data.ECGDataset(Xtr, Ytr, corruptor=corruptor,
                         p_corrupt=T["p_corrupt"] if regime == "aug" else 0.0,
                         paired=spec.paired, seed=seed, fs=fs)
    nw = int(T["num_workers"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=T["batch_size"], shuffle=True, drop_last=True, num_workers=nw,
        worker_init_fn=data.worker_init_fn if nw > 0 else None,
        generator=torch.Generator().manual_seed(seed), persistent_workers=nw > 0)

    model = models.build_model(spec.model, n_classes=len(SUPERCLASSES), fs=fs).to(device)
    n_params = models.count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    steps_per_epoch = len(loader)
    if T.get("max_batches_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, int(T["max_batches_per_epoch"]))
    total_steps = max(1, steps_per_epoch * T["max_epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)

    resolved = copy.deepcopy(cfg)
    resolved["run"] = {"variant": variant, "regime": regime, "seed": seed, **spec.__dict__}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    log_f = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "val_macro_auroc", "lr", "epoch_sec"])
    best, best_epoch, bad, epoch = -math.inf, 0, 0, 0
    for epoch in range(1, T["max_epochs"] + 1):
        te = time.time()
        model.train()
        tot, nb = 0.0, 0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            loss, _ = compute_loss(model, batch, spec, T, device, amp_dtype)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        probs, _ = predict(model, Xva_pre, device, cfg["eval"]["batch_size"])
        val_auc = macro_auroc(Yva, probs)
        log.writerow([epoch, tot / max(nb, 1), val_auc, opt.param_groups[0]["lr"], round(time.time() - te, 2)])
        log_f.flush()
        score = val_auc if math.isfinite(val_auc) else -math.inf
        if epoch == 1 or score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "model": spec.model, "variant": variant,
                        "regime": regime, "seed": seed, "epoch": epoch, "val_macro_auroc": val_auc},
                       run_dir / "best.pt")
        else:
            bad += 1
        print(f"  epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val_auroc {val_auc:.4f} "
              f"(best {best:.4f} @ {best_epoch})")
        if bad >= T["patience"]:
            break
    log_f.close()

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    probs, _ = predict(model, Xva_pre, device, cfg["eval"]["batch_size"])
    th = tune_thresholds(Yva, probs)
    with open(run_dir / "thresholds.json", "w") as f:
        json.dump(dict(zip(SUPERCLASSES, map(float, th))), f, indent=2)

    meta = {"variant": variant, "regime": regime, "seed": seed, "model": spec.model,
            "paired": spec.paired, "rank_loss": spec.rank, "consistency_loss": spec.consistency,
            **git_info(), "fs": fs, "n_params": n_params, "device": str(device),
            "amp_dtype": "float16" if amp_dtype is not None else "float32", "amp_note": amp_note,
            "epochs_run": epoch, "best_epoch": best_epoch, "best_val_macro_auroc": best,
            "n_train": int(len(ds)), "n_val": int(len(Yva)), "wall_time_sec": round(time.time() - t0, 1),
            "torch": torch.__version__, "smoke": cfg["smoke"]}
    with open(run_dir / "meta.json", "w") as f:  # written last: marks the run complete
        json.dump(meta, f, indent=2)
    print(f"[done] {run_dir} best val macro-AUROC {best:.4f} @ epoch {best_epoch}")
    return run_dir


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="apply the config's smoke overrides")
    a = ap.parse_args(argv)
    try:
        get_variant(a.variant, a.regime)
    except ValueError as e:
        ap.error(str(e))
    train(load_config(a.config, a.smoke), a.variant, a.regime, a.seed)


if __name__ == "__main__":
    main()
