"""Data sanity check for PTB-XL Track 1 (no training)."""
import numpy as np

from data import ROOT, SPLITS, SUPERCLASSES, load_meta, load_split, set_seed

set_seed(0)
db, Y = load_meta()
has = Y.sum(1) > 0
print(f"PTB-XL root: {ROOT}\nTotal records: {len(db)}")

print("\nRecords per strat_fold (all / with >=1 superclass / dropped):")
for f in range(1, 11):
    m = (db["strat_fold"] == f).to_numpy()
    print(f"  fold {f:2d}: {m.sum():5d} / {(m & has).sum():5d} / {(m & ~has).sum():4d}")
print(f"Records with no superclass label: {(~has).sum()} -> DROPPED (as in Strodthoff et al. 2021)")

print("\nLabel prevalence per split (after dropping):")
print("  split     N     " + "  ".join(f"{c:>6s}" for c in SUPERCLASSES))
for s, folds in SPLITS.items():
    m = db["strat_fold"].isin(folds).to_numpy() & has
    print(f"  {s:5s} {m.sum():6d}   " + "  ".join(f"{p:6.3f}" for p in Y[m].mean(0)))

pf = db.groupby("patient_id")["strat_fold"].nunique()
print(f"\nPatients: {len(pf)}; patients in >1 fold: {(pf > 1).sum()}")
assert (pf > 1).sum() == 0, "patient leakage across folds!"
print("OK: no patient_id appears in more than one fold.")

for s in SPLITS:
    X, Ys, meta = load_split(s)
    print(f"{s}: X {X.shape} {X.dtype}, mean {X.mean():.3f} std {X.std():.3f}, finite {np.isfinite(X).all()}")
