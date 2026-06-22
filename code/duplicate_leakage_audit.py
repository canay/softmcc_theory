"""Audit exact duplicate rows and split-crossing duplicates in active caches.

This script does not train models and does not change manuscript results. It
replays the split protocol used by the harden run and reports whether exact
feature-label duplicate rows appear across train/validation/test partitions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "02_data"
OUT = ROOT / "03_experiments" / "results" / "duplicate_leakage_audit_20260621.csv"
SEED = 42
REPEATS = 12


def row_keys(x: np.ndarray, y: np.ndarray) -> list[bytes]:
    arr = np.concatenate([x.reshape((x.shape[0], -1)), y.reshape(-1, 1)], axis=1)
    arr = np.ascontiguousarray(arr)
    row_type = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    return [v.tobytes() for v in arr.view(row_type).ravel()]


def audit_cache(path: Path) -> dict[str, object]:
    data = np.load(path)
    x = data["X"]
    y = data["y"]
    keys = row_keys(x, y)
    n = len(y)
    duplicate_rows = n - len(set(keys))

    test_rates: list[float] = []
    val_rates: list[float] = []
    idx = np.arange(n)
    for r in range(REPEATS):
        seed = SEED + r
        train_idx, test_idx = train_test_split(
            idx, test_size=0.25, stratify=y, random_state=seed
        )
        train2_idx, val_idx = train_test_split(
            train_idx, test_size=0.30, stratify=y[train_idx], random_state=seed + 1
        )
        trainval_keys = {keys[i] for i in train_idx}
        train_keys = {keys[i] for i in train2_idx}
        test_rates.append(sum(keys[i] in trainval_keys for i in test_idx) / len(test_idx))
        val_rates.append(sum(keys[i] in train_keys for i in val_idx) / len(val_idx))

    return {
        "cache": path.name,
        "rows": n,
        "features": x.shape[1] if x.ndim > 1 else 1,
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "exact_duplicate_rows_with_label": duplicate_rows,
        "duplicate_rate": duplicate_rows / n,
        "test_duplicate_in_trainval_mean": float(np.mean(test_rates)),
        "test_duplicate_in_trainval_max": float(np.max(test_rates)),
        "validation_duplicate_in_train_mean": float(np.mean(val_rates)),
        "validation_duplicate_in_train_max": float(np.max(val_rates)),
    }


def main() -> None:
    rows = [audit_cache(path) for path in sorted(DATA.glob("*.npz"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
