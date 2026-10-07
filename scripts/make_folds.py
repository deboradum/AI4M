#!/usr/bin/env python3
# MIT License
#
# 5-fold cross-validation splits that extend the fixed 32/8 split: fold 0 is
# exactly configs/splits/segthor_full_32_8.json's validation set, and the 32
# remaining patients are dealt into folds 1-4 of 8, stratified on the same
# criteria as make_split.py (contrast, spacing outlier, volume outlier), so
# every patient is validated exactly once. The strata are read from the
# per-patient measurements make_split.py already stored in that JSON.
#
#   python scripts/make_folds.py --base configs/splits/segthor_full_32_8.json \
#       --dest_pattern configs/splits/segthor_full_cv5_f{fold}.json
import json
import random
import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    base = json.loads(args.base.read_text())
    rows = {r["id"]: r for r in base["patients"]}
    fold0 = sorted(base["validation"])
    rest = sorted(set(rows) - set(fold0))
    n_folds = 1 + len(rest) // len(fold0)
    assert len(rest) == (n_folds - 1) * len(fold0), (len(rest), len(fold0))

    # Rarest stratum first, so the few outliers are spread one per fold, then
    # round-robin; the shuffle inside each stratum is seeded.
    def key(pid: str) -> tuple:
        r = rows[pid]
        return (not r["spacing_outlier"], not r["volume_outlier"], not r["contrast"])
    rng = random.Random(args.seed)
    rest.sort(key=lambda pid: (key(pid), rng.random()))
    folds: list[list[str]] = [fold0] + [[] for _ in range(n_folds - 1)]
    for i, pid in enumerate(rest):
        folds[1 + i % (n_folds - 1)].append(pid)

    for k, val in enumerate(folds):
        val = sorted(val)
        training = sorted(set(rows) - set(val))
        counts = {s: {"train": sum(rows[p][s] for p in training), "validation": sum(rows[p][s] for p in val)}
                  for s in ("contrast", "spacing_outlier", "volume_outlier")}
        out = {"source_dir": base["source_dir"], "n_train": len(training), "n_validation": len(val),
               "fold": k, "n_folds": n_folds, "base": str(args.base), "seed": args.seed,
               "criteria": base["criteria"], "strata_counts": counts,
               "training": training, "validation": val}
        dest = Path(args.dest_pattern.format(fold=k))
        dest.write_text(json.dumps(out, indent=2) + "\n")
        print(f"fold {k}: {val}  " + "  ".join(f"{s} {c['validation']}" for s, c in counts.items()))

    all_val = [p for f in folds for p in f]
    assert sorted(all_val) == sorted(rows), "every patient must be validated exactly once"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-fold CV splits extending the fixed 32/8 split")
    parser.add_argument("--base", type=Path, default=Path("configs/splits/segthor_full_32_8.json"))
    parser.add_argument("--dest_pattern", type=str, default="configs/splits/segthor_full_cv5_f{fold}.json")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
