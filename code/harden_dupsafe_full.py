"""Duplicate-safe hardening pipeline for SoftMCC Theory.

This script preserves the existing harden_* files and writes a separate
harden_dupsafe_* evidence family. The canonical remediation choice is an exact
feature-label grouped split: rows with the same feature vector and label are
kept in the same train, validation, or test partition.

Outputs include raw utility/ranking/calibration CSVs, calibration diagnostics,
split and dataset manifests, summary CSVs, figures, a table source, and an
artifact hash manifest. No manuscript text is edited by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, spearmanr, wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "03_experiments" / "results"
FIGS = ROOT / "04_manuscript" / "figs"
TABLES = ROOT / "04_manuscript" / "tables"
DATA = ROOT / "02_data"

PREFIX = "harden_dupsafe"
SEED = 42
N_REPEATS = 12
TEMPS = [0.5, 1.5, 2.0, 3.0]
METRICS = ["SoftMCC", "MCC_best", "F1_best", "AUPRC", "AUROC", "Brier_neg", "MCC_05"]
PRETTY = {
    "SoftMCC": "SoftMCC",
    "MCC_best": "MCC@best",
    "F1_best": "F1@best",
    "AUPRC": "AUPRC",
    "AUROC": "AUROC",
    "Brier_neg": "Brier",
    "MCC_05": "MCC@0.5",
}
PRETTY_ORDER = ["SoftMCC", "Brier", "MCC@0.5", "AUROC", "AUPRC", "F1@best", "MCC@best"]
BENCH = ["breast_cancer(37%)", "synth(5%)", "synth(1%)"]
REAL = ["creditcard(1%)", "creditcard(0.5%)", "iotid20(6.4%)"]
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    source: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]
    cache_path: Path | None = None


class Transcript:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def log(self, message: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def output_paths() -> dict[str, Path]:
    return {
        "utility": RESULTS / f"{PREFIX}_utility.csv",
        "rankings": RESULTS / f"{PREFIX}_rankings.csv",
        "calibration": RESULTS / f"{PREFIX}_calibration.csv",
        "calibration_diagnostics": RESULTS / f"{PREFIX}_calibration_diagnostics.csv",
        "split_manifest": RESULTS / f"{PREFIX}_split_manifest.csv",
        "dataset_manifest": RESULTS / f"{PREFIX}_dataset_manifest.csv",
        "environment": RESULTS / f"{PREFIX}_environment.json",
        "stability_summary": RESULTS / f"{PREFIX}_summary_stability.csv",
        "selection_summary": RESULTS / f"{PREFIX}_summary_selection.csv",
        "calibration_summary": RESULTS / f"{PREFIX}_summary_calibration.csv",
        "friedman": RESULTS / f"{PREFIX}_friedman.txt",
        "reconciliation": RESULTS / f"{PREFIX}_evidence_reconciliation.csv",
        "artifact_manifest": RESULTS / f"{PREFIX}_artifact_manifest.csv",
        "transcript": RESULTS / f"{PREFIX}_transcript.txt",
        "fig_stability": FIGS / "fig1_resampling_stability_dupsafe.png",
        "fig_calibration": FIGS / "fig2_calibration_sensitivity_dupsafe.png",
        "fig_cd": FIGS / "fig3_critical_difference_dupsafe.png",
        "table": TABLES / "table_main_results_dupsafe.tex",
    }


def ensure_clean_outputs(paths: dict[str, Path], force: bool, transcript: Transcript | None = None) -> None:
    check = [p for k, p in paths.items() if k != "transcript" and p.exists()]
    if check and not force:
        names = "\n".join(str(p) for p in check)
        raise SystemExit("Refusing to overwrite existing duplicate-safe outputs. Use --force.\n" + names)
    if force:
        for key, path in paths.items():
            if key == "transcript":
                continue
            if path.exists():
                if transcript:
                    transcript.log(f"Removing previous {PREFIX} output before rerun: {path}")
                path.unlink()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["X"], d["y"]


def make_imbalanced(pos_rate: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n,
        n_features=25,
        n_informative=8,
        n_redundant=4,
        weights=[1 - pos_rate, pos_rate],
        flip_y=0.01,
        class_sep=0.9,
        random_state=seed,
    )


def dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec("bc", "breast_cancer(37%)", "sklearn.load_breast_cancer", lambda: _bc()),
        DatasetSpec("synth5", "synth(5%)", "sklearn.make_classification seed=1", lambda: make_imbalanced(0.05, 4000, 1)),
        DatasetSpec("synth1", "synth(1%)", "sklearn.make_classification seed=2", lambda: make_imbalanced(0.01, 4000, 2)),
        DatasetSpec(
            "cc10",
            "creditcard(1%)",
            "02_data/creditcard_pi10.npz",
            lambda: load_npz(DATA / "creditcard_pi10.npz"),
            DATA / "creditcard_pi10.npz",
        ),
        DatasetSpec(
            "cc5",
            "creditcard(0.5%)",
            "02_data/creditcard_pi5.npz",
            lambda: load_npz(DATA / "creditcard_pi5.npz"),
            DATA / "creditcard_pi5.npz",
        ),
        DatasetSpec(
            "iot",
            "iotid20(6.4%)",
            "02_data/iotid20_compact.npz",
            lambda: load_npz(DATA / "iotid20_compact.npz"),
            DATA / "iotid20_compact.npz",
        ),
    ]


def _bc() -> tuple[np.ndarray, np.ndarray]:
    bc = load_breast_cancer()
    # sklearn encodes benign as 1 and malignant as 0. The manuscript treats the
    # malignant class as the positive class, giving the stated 37% prevalence.
    return bc.data, 1 - bc.target


def soft_mcc(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    tp = np.sum(p * y)
    fp = np.sum(p * (1 - y))
    fn = np.sum((1 - p) * y)
    tn = np.sum((1 - p) * (1 - y))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    return float(num / den)


def _thr_grid(p: np.ndarray, n: int = 99) -> np.ndarray:
    qs = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n)))
    return qs if len(qs) else np.array([0.5])


def best_threshold(y: np.ndarray, p: np.ndarray, metric: str = "mcc") -> tuple[float, float]:
    from sklearn.metrics import f1_score

    best_t, best_v = 0.5, -np.inf
    for t in _thr_grid(p):
        yp = (p >= t).astype(int)
        v = matthews_corrcoef(y, yp) if metric == "mcc" else f1_score(y, yp, zero_division=0)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t, best_v


def selection_scores(y_val: np.ndarray, p_val: np.ndarray) -> tuple[dict[str, float], float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    mcc_best_t, mcc_best_v = best_threshold(y_val, p_val, "mcc")
    _, f1_best_v = best_threshold(y_val, p_val, "f1")
    return (
        {
            "SoftMCC": soft_mcc(y_val, p_val),
            "MCC_best": mcc_best_v,
            "F1_best": f1_best_v,
            "AUPRC": float(average_precision_score(y_val, p_val)),
            "AUROC": float(roc_auc_score(y_val, p_val)),
            "Brier_neg": float(-brier_score_loss(y_val, p_val)),
            "MCC_05": float(matthews_corrcoef(y_val, (p_val >= 0.5).astype(int))),
        },
        mcc_best_t,
    )


def temperature_scale(p: np.ndarray, T: float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def pool():
    base = [
        (
            f"logreg_C{C}",
            LogisticRegression(C=C, class_weight="balanced", max_iter=1000, solver="liblinear"),
        )
        for C in (0.1, 1.0, 10.0)
    ]
    base += [
        (
            f"hgb_d{d}",
            HistGradientBoostingClassifier(max_depth=d, learning_rate=0.1, max_iter=150, random_state=0),
        )
        for d in (3, 6)
    ]
    return [(name, CalibratedClassifierCV(est, method="isotonic", cv=2)) for name, est in base]


def row_group_ids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.concatenate([X.reshape((X.shape[0], -1)), y.reshape(-1, 1)], axis=1)
    arr = np.ascontiguousarray(arr)
    row_type = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    keys = arr.view(row_type).ravel()
    unique, inv = np.unique(keys, return_inverse=True)
    return inv.astype(np.int64), int(len(y) - len(unique))


def split_groups(
    group_ids: np.ndarray,
    group_labels: dict[int, int],
    candidate_groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([group_labels[int(g)] for g in candidate_groups], dtype=int)
    left, right = train_test_split(
        candidate_groups,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def indices_for_groups(group_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mask = np.isin(group_ids, selected)
    return np.where(mask)[0]


def class_counts(y: np.ndarray) -> tuple[int, int, float]:
    pos = int(np.sum(y))
    total = int(len(y))
    return pos, total - pos, float(pos / total) if total else float("nan")


def split_manifest_row(
    spec: DatasetSpec,
    repeat: int,
    seed: int,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    duplicate_rows: int,
) -> dict[str, object]:
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    tr_pos, tr_neg, tr_prev = class_counts(y[train_idx])
    va_pos, va_neg, va_prev = class_counts(y[val_idx])
    te_pos, te_neg, te_prev = class_counts(y[test_idx])
    return {
        "dataset_key": spec.key,
        "dataset": spec.label,
        "repeat": repeat,
        "seed": seed,
        "split_method": "exact_feature_label_grouped_stratified",
        "rows_total": int(len(y)),
        "groups_total": int(len(set(groups.tolist()))),
        "exact_duplicate_rows_with_label": duplicate_rows,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_groups": int(len(train_groups)),
        "validation_groups": int(len(val_groups)),
        "test_groups": int(len(test_groups)),
        "train_pos": tr_pos,
        "train_neg": tr_neg,
        "train_prevalence": tr_prev,
        "validation_pos": va_pos,
        "validation_neg": va_neg,
        "validation_prevalence": va_prev,
        "test_pos": te_pos,
        "test_neg": te_neg,
        "test_prevalence": te_prev,
        "train_validation_group_overlap": int(len(train_groups & val_groups)),
        "train_test_group_overlap": int(len(train_groups & test_groups)),
        "validation_test_group_overlap": int(len(val_groups & test_groups)),
        "train_index_sha256": hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest(),
        "validation_index_sha256": hashlib.sha256(np.asarray(val_idx, dtype=np.int64).tobytes()).hexdigest(),
        "test_index_sha256": hashlib.sha256(np.asarray(test_idx, dtype=np.int64).tobytes()).hexdigest(),
    }


def ranking_vector(scores: dict[str, dict[str, float]], metric: str, order: list[str]) -> np.ndarray:
    vals = np.array([scores[c][metric] for c in order])
    ranked = np.argsort(-vals)
    out = np.empty(len(order))
    out[ranked] = np.arange(len(order))
    return out


def kendalls_w(rank_lists: list[list[str]], items: list[str]) -> float:
    n = len(items)
    m = len(rank_lists)
    if m < 2 or n < 2:
        return float("nan")
    idx = {it: i for i, it in enumerate(items)}
    R = np.zeros(n)
    for rl in rank_lists:
        for pos, it in enumerate(rl):
            R[idx[it]] += pos + 1
    S = np.sum((R - R.mean()) ** 2)
    return float(12 * S / (m**2 * (n**3 - n)))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def run_raw(paths: dict[str, Path], transcript: Transcript) -> None:
    util_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    cal_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for spec in dataset_specs():
        X, y = spec.loader()
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        groups, duplicate_rows = row_group_ids(X, y)
        unique_groups, first_idx = np.unique(groups, return_index=True)
        group_labels = {int(g): int(y[i]) for g, i in zip(unique_groups, first_idx)}
        pos, neg, prevalence = class_counts(y)
        dataset_rows.append(
            {
                "dataset_key": spec.key,
                "dataset": spec.label,
                "source": spec.source,
                "cache_path": str(spec.cache_path) if spec.cache_path else "",
                "cache_sha256": sha256_file(spec.cache_path) if spec.cache_path else "",
                "rows": int(len(y)),
                "features": int(X.reshape((X.shape[0], -1)).shape[1]),
                "positive_rows": pos,
                "negative_rows": neg,
                "prevalence": prevalence,
                "unique_exact_feature_label_groups": int(len(unique_groups)),
                "exact_duplicate_rows_with_label": duplicate_rows,
                "X_sha256": sha256_array(X),
                "y_sha256": sha256_array(y),
            }
        )
        transcript.log(
            f"Dataset {spec.label}: rows={len(y)} groups={len(unique_groups)} "
            f"duplicate_rows={duplicate_rows} prevalence={prevalence:.6f}"
        )

        for repeat in range(N_REPEATS):
            seed = SEED + repeat
            trainval_groups, test_groups = split_groups(groups, group_labels, unique_groups, 0.25, seed)
            train_groups, val_groups = split_groups(groups, group_labels, trainval_groups, 0.30, seed + 1)
            train_idx = indices_for_groups(groups, train_groups)
            val_idx = indices_for_groups(groups, val_groups)
            test_idx = indices_for_groups(groups, test_groups)
            split_rows.append(
                split_manifest_row(
                    spec, repeat, seed, y, groups, train_idx, val_idx, test_idx, duplicate_rows
                )
            )

            Xtr2, ytr2 = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]
            Xte, yte = X[test_idx], y[test_idx]
            val_scores: dict[str, dict[str, float]] = {}
            val_prob: dict[str, np.ndarray] = {}
            test_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
            for name, est in pool():
                est.fit(Xtr2, ytr2)
                pva = est.predict_proba(Xva)[:, 1]
                pte = est.predict_proba(Xte)[:, 1]
                scores, mcc_t = selection_scores(yva, pva)
                val_scores[name] = scores
                val_prob[name] = pva
                test_cache[name] = (yte, pte, mcc_t)
                diag_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset": spec.label,
                        "repeat": repeat,
                        "seed": seed,
                        "candidate": name,
                        "validation_brier": float(np.mean((pva - yva) ** 2)),
                        "validation_log_loss": float(log_loss(yva, np.clip(pva, 1e-12, 1 - 1e-12))),
                        "validation_ece_10": ece_score(yva, pva, bins=10),
                        "validation_prob_mean": float(np.mean(pva)),
                        "validation_pos_rate": float(np.mean(yva)),
                    }
                )

            candidate_order = list(val_scores.keys())
            for metric in METRICS:
                ordered = sorted(candidate_order, key=lambda c: val_scores[c][metric], reverse=True)
                rank_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset": spec.label,
                        "repeat": repeat,
                        "seed": seed,
                        "metric": metric,
                        "ranking": ">".join(ordered),
                        "selected_candidate": ordered[0],
                    }
                )
                yte_, pte_, mcc_t = test_cache[ordered[0]]
                util_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset": spec.label,
                        "repeat": repeat,
                        "seed": seed,
                        "metric": metric,
                        "selected_candidate": ordered[0],
                        "validation_metric_value": float(val_scores[ordered[0]][metric]),
                        "test_mcc": float(matthews_corrcoef(yte_, (pte_ >= mcc_t).astype(int))),
                    }
                )

            base_rank = {m: ranking_vector(val_scores, m, candidate_order) for m in METRICS}
            for temp in TEMPS:
                temp_scores = {
                    c: selection_scores(yva, temperature_scale(val_prob[c], temp))[0]
                    for c in candidate_order
                }
                for metric in METRICS:
                    rho = spearmanr(base_rank[metric], ranking_vector(temp_scores, metric, candidate_order)).correlation
                    cal_rows.append(
                        {
                            "dataset_key": spec.key,
                            "dataset": spec.label,
                            "repeat": repeat,
                            "seed": seed,
                            "T": temp,
                            "metric": metric,
                            "spearman_vs_T1": 1.0 if np.isnan(rho) else float(rho),
                        }
                    )
            transcript.log(f"{spec.label} repeat {repeat + 1}/{N_REPEATS} complete")

        pd.DataFrame(util_rows).to_csv(paths["utility"], index=False)
        pd.DataFrame(rank_rows).to_csv(paths["rankings"], index=False)
        pd.DataFrame(cal_rows).to_csv(paths["calibration"], index=False)
        pd.DataFrame(diag_rows).to_csv(paths["calibration_diagnostics"], index=False)
        pd.DataFrame(split_rows).to_csv(paths["split_manifest"], index=False)
        pd.DataFrame(dataset_rows).to_csv(paths["dataset_manifest"], index=False)
        transcript.log(f"Checkpoint saved after dataset {spec.label}")


def bca_ci(data: list[object], stat_fn: Callable[[list[object]], float], rng: np.random.Generator, B: int = 2000) -> tuple[float, float]:
    data = list(data)
    n = len(data)
    theta_hat = stat_fn(data)
    boots = np.array([stat_fn([data[i] for i in rng.integers(0, n, n)]) for _ in range(B)])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return float("nan"), float("nan")
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / (len(boots) + 1)), 1 - 1.0 / (len(boots) + 1))
    z0 = norm.ppf(prop)
    jack = np.array([stat_fn([data[j] for j in range(n) if j != i]) for i in range(n)])
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    acc = 0.0 if den == 0 else num / den

    def adj(q: float) -> float:
        z = norm.ppf(q)
        return norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))

    return (
        float(np.percentile(boots, 100 * adj(0.025))),
        float(np.percentile(boots, 100 * adj(0.975))),
    )


def analyze(paths: dict[str, Path], transcript: Transcript) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    util = pd.read_csv(paths["utility"])
    ranks = pd.read_csv(paths["rankings"])
    cal = pd.read_csv(paths["calibration"])
    datasets = list(util.dataset.unique())
    lines: list[str] = []

    stab_rows: list[dict[str, object]] = []
    for ds in datasets:
        for metric in METRICS:
            rank_lists = [
                s.split(">")
                for s in ranks[(ranks.dataset == ds) & (ranks.metric == metric)]
                .sort_values("repeat")
                .ranking
            ]
            items = sorted(rank_lists[0])
            w = kendalls_w(rank_lists, items)
            lo, hi = bca_ci(rank_lists, lambda d: kendalls_w(d, items), rng)
            stab_rows.append(
                {"dataset": ds, "metric": PRETTY[metric], "kendalls_w": w, "bca_lo": lo, "bca_hi": hi}
            )
    sdf = pd.DataFrame(stab_rows)
    sdf.to_csv(paths["stability_summary"], index=False)

    sel_rows: list[dict[str, object]] = []
    base = util[util.metric == "SoftMCC"].set_index(["dataset", "repeat"])["test_mcc"]
    for metric in [x for x in METRICS if x != "SoftMCC"]:
        comp = util[util.metric == metric].set_index(["dataset", "repeat"])["test_mcc"]
        a, b = base.align(comp, join="inner")
        diffs = (a - b).values
        if np.allclose(diffs, 0):
            row = {
                "baseline": PRETTY[metric],
                "mean_diff": 0.0,
                "bca_lo": 0.0,
                "bca_hi": 0.0,
                "wilcoxon_p": 1.0,
                "cliffs_delta": 0.0,
            }
        else:
            lo, hi = bca_ci(list(diffs), lambda d: float(np.mean(d)), rng)
            try:
                _, p = wilcoxon(diffs)
            except Exception:
                p = float("nan")
            row = {
                "baseline": PRETTY[metric],
                "mean_diff": float(np.mean(diffs)),
                "bca_lo": lo,
                "bca_hi": hi,
                "wilcoxon_p": float(p),
                "cliffs_delta": cliffs_delta(a.values, b.values),
            }
        sel_rows.append(row)
    pd.DataFrame(sel_rows).to_csv(paths["selection_summary"], index=False)

    cal_rows: list[dict[str, object]] = []
    cm = cal.groupby(["dataset", "repeat", "metric"])["spearman_vs_T1"].mean().reset_index()
    for metric in METRICS:
        vals = cm[cm.metric == metric]["spearman_vs_T1"].values
        lo, hi = bca_ci(list(vals), lambda d: float(np.mean(d)), rng)
        cal_rows.append(
            {
                "metric": PRETTY[metric],
                "mean_spearman_vs_T1": float(np.mean(vals)),
                "bca_lo": lo,
                "bca_hi": hi,
            }
        )
    pd.DataFrame(cal_rows).to_csv(paths["calibration_summary"], index=False)

    k = len(METRICS)
    n_datasets = len(datasets)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    def friedman_block(values_by_metric: dict[str, np.ndarray], label: str, higher_better: bool = True) -> np.ndarray:
        mat = np.array([values_by_metric[m] for m in METRICS])
        stat, p = friedmanchisquare(*[mat[i] for i in range(k)])
        sign = -1 if higher_better else 1
        rank_mat = np.array([pd.Series(sign * mat[:, j]).rank().values for j in range(n_datasets)]).T
        mean_ranks = rank_mat.mean(axis=1)
        lines.append(
            f"[{label}] Friedman chi2={stat:.3f} p={p:.4g}  "
            f"(k={k}, N={n_datasets}); Nemenyi CD(0.05)={cd:.3f}"
        )
        for m, rank in sorted(zip(METRICS, mean_ranks), key=lambda t: t[1]):
            lines.append(f"    {PRETTY[m]:9s} mean rank {rank:.3f}")
        return mean_ranks

    wpiv = sdf.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    wvals = {m: wpiv.loc[PRETTY[m], datasets].values for m in METRICS}
    ranks_stab = friedman_block(wvals, "stability (Kendall's W)")

    upiv = util.pivot_table(index="metric", columns="dataset", values="test_mcc", aggfunc="mean")
    uvals = {m: upiv.loc[m, datasets].values for m in METRICS}
    ranks_sel = friedman_block(uvals, "selection (mean test MCC)")

    paths["friedman"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        transcript.log(line)
    return ranks_stab, ranks_sel, float(cd)


def cd_panel(ax, mean_ranks: np.ndarray, cd: float, heading: str) -> None:
    k = len(METRICS)
    order = np.argsort(mean_ranks)
    lo, hi = 1, k
    ax.set_xlim(hi + 0.4, lo - 0.4)
    ax.set_ylim(-1.85, 2.25)
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(range(1, k + 1))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yticks([])
    ax.plot([lo, hi], [1, 1], color="k", lw=1.2, clip_on=False)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [1, 1.12], color="k", lw=1.0, clip_on=False)
    half = int(np.ceil(k / 2))
    for i, idx in enumerate(order):
        rank = mean_ranks[idx]
        if i < half:
            xend, y = lo - 0.35, 0.45 - 0.62 * i
            ha = "right"
        else:
            xend, y = hi + 0.35, 0.45 - 0.62 * (k - 1 - i)
            ha = "left"
        ax.plot([rank, rank], [1, y], color="k", lw=0.8)
        ax.plot([rank, xend], [y, y], color="k", lw=0.8)
        ax.text(
            xend,
            y + 0.02,
            f" {PRETTY[METRICS[idx]]} ({rank:.2f}) ",
            ha=ha,
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )
    sr = np.sort(mean_ranks)
    cliques: list[list[int]] = []
    used: set[tuple[int, ...]] = set()
    for i in range(k):
        grp = [j for j in range(k) if sr[j] - sr[i] <= cd and j >= i]
        if len(grp) > 1 and tuple(grp) not in used:
            dominated = any(set(grp) < set(g) for g in cliques)
            if not dominated:
                cliques.append(grp)
                used.add(tuple(grp))
    cliques = [g for g in cliques if not any(set(g) < set(h) for h in cliques if h != g)]
    for ci, grp in enumerate(cliques):
        y = 0.72 - 0.22 * ci
        ax.plot([sr[grp[0]] - 0.06, sr[grp[-1]] + 0.06], [y, y], color="k", lw=2.6, solid_capstyle="round")
    ax.plot([lo, lo + cd], [1.85, 1.85], color="k", lw=1.4)
    ax.plot([lo, lo], [1.78, 1.92], color="k", lw=1.2)
    ax.plot([lo + cd, lo + cd], [1.78, 1.92], color="k", lw=1.2)
    ax.text(lo + cd / 2, 1.95, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(heading, fontsize=9, pad=30)


def make_figures_and_table(paths: dict[str, Path], ranks_stab: np.ndarray, ranks_sel: np.ndarray, cd: float, transcript: Transcript) -> None:
    st = pd.read_csv(paths["stability_summary"])
    mw = st.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    bench = mw[BENCH].mean(axis=1)
    real = mw[REAL].mean(axis=1)
    order = sorted(PRETTY_ORDER, key=lambda m: -float((bench[m] + real[m]) / 2))

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(order))
    width = 0.38
    b1 = ax.bar(x - width / 2, [bench[m] for m in order], width, label="Benchmark suite (37%/5%/1%)", color="#4878a8")
    b2 = ax.bar(x + width / 2, [real[m] for m in order], width, label="Real suite (creditcard, IoTID20)", color="#b04a4a")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("Kendall's W (ranking stability)", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Duplicate-safe resampling stability by metric (higher = more stable)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["fig_stability"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    cal = pd.read_csv(paths["calibration"])
    cal = cal[cal.dataset.isin(REAL)]
    styles = {
        "SoftMCC": ("#1f5fa8", "-", "o"),
        "Brier_neg": ("#b04a4a", "--", "s"),
        "MCC_best": ("#777777", ":", "^"),
        "F1_best": ("#999999", ":", "v"),
        "AUROC": ("#555555", ":", "D"),
        "AUPRC": ("#bbbbbb", ":", "P"),
        "MCC_05": ("#333333", ":", "X"),
    }
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for metric, (color, ls, marker) in styles.items():
        g = cal[cal.metric == metric].groupby("T")["spearman_vs_T1"].mean()
        ax.plot(g.index, g.values, ls, color=color, marker=marker, ms=3.5, lw=1.3, label=PRETTY[metric])
    ax.set_xlabel("Temperature T (miscalibration; 1.0 = calibrated)", fontsize=8)
    ax.set_ylabel("Spearman ($\\rho$, rank vs T=1)", fontsize=8)
    ax.tick_params(labelsize=8)
    ymin = max(0.0, min(0.6, float(cal["spearman_vs_T1"].min()) - 0.05))
    ax.set_ylim(ymin, 1.02)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    ax.set_title("Duplicate-safe ranking agreement under calibration shift (real suite)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["fig_calibration"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.7))
    cd_panel(axes[0], ranks_stab, cd, "Ranking stability: mean ranks over 6 datasets")
    cd_panel(axes[1], ranks_sel, cd, "Selection utility: mean ranks over 6 datasets")
    axes[0].text(0.5, -0.16, "(a) Ranking stability (Kendall's $W$)", transform=axes[0].transAxes, ha="center", va="top", fontsize=9, fontweight="bold", clip_on=False)
    axes[1].text(0.5, -0.16, "(b) Selection utility (mean test MCC)", transform=axes[1].transAxes, ha="center", va="top", fontsize=9, fontweight="bold", clip_on=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.13, hspace=0.82)
    fig.savefig(paths["fig_cd"], dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    util = pd.read_csv(paths["utility"])
    selm = util.groupby("metric").test_mcc.mean()
    calm = pd.read_csv(paths["calibration_summary"]).set_index("metric")
    table_lines = [
        "\\begin{center}",
        "\\refstepcounter{table}\\label{tab:main}",
        "\\parbox{\\linewidth}{\\small\\textbf{Table~\\thetable:} The duplicate-safe six-dataset threshold-free selection protocol over twelve repeats reports ranking stability, selected-model test MCC, and calibration-shift agreement.}",
        "\\smallskip",
        "\\small",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Stability (Kendall's $W$)} & Selection & Calibration\\\\",
        "\\cmidrule(lr){2-3}",
        "Metric & Benchmark & Real & Test MCC & Spearman\\\\",
        "\\midrule",
    ]
    for metric_key, label in [
        ("SoftMCC", "SoftMCC"),
        ("Brier_neg", "Brier"),
        ("MCC_05", "MCC@0.5"),
        ("AUROC", "AUROC"),
        ("AUPRC", "AUPRC"),
        ("F1_best", "$F_1$@best"),
        ("MCC_best", "MCC@best"),
    ]:
        p = PRETTY[metric_key]
        bval = float(bench[p])
        rval = float(real[p])
        if metric_key == "SoftMCC":
            table_lines.append(f"SoftMCC      & \\textbf{{{bval:.3f}}} & \\textbf{{{rval:.3f}}} & {selm[metric_key]:.3f} & {calm.loc[p, 'mean_spearman_vs_T1']:.3f}\\\\")
        else:
            table_lines.append(f"{label:<13s} & {bval:.3f} & {rval:.3f} & {selm[metric_key]:.3f} & {calm.loc[p, 'mean_spearman_vs_T1']:.3f}\\\\")
    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{center}", ""])
    paths["table"].write_text("\n".join(table_lines), encoding="utf-8", newline="\n")
    transcript.log(f"Saved duplicate-safe figures and table source: {paths['fig_stability']}, {paths['fig_calibration']}, {paths['fig_cd']}, {paths['table']}")


def write_reconciliation(paths: dict[str, Path], transcript: Transcript) -> None:
    rows: list[dict[str, object]] = []
    old_files = {
        "stability": RESULTS / "harden_summary_stability.csv",
        "selection": RESULTS / "harden_summary_selection.csv",
        "calibration": RESULTS / "harden_summary_calibration.csv",
    }
    new_files = {
        "stability": paths["stability_summary"],
        "selection": paths["selection_summary"],
        "calibration": paths["calibration_summary"],
    }
    for family in ["stability", "selection", "calibration"]:
        if not old_files[family].exists():
            rows.append({"family": family, "status": "old_missing", "item": "", "old": "", "new": "", "delta": ""})
            continue
        old = pd.read_csv(old_files[family])
        new = pd.read_csv(new_files[family])
        if family == "stability":
            merged = old.merge(new, on=["dataset", "metric"], suffixes=("_old", "_new"))
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "family": family,
                        "status": "compared",
                        "item": f"{r['dataset']}|{r['metric']}|kendalls_w",
                        "old": r["kendalls_w_old"],
                        "new": r["kendalls_w_new"],
                        "delta": r["kendalls_w_new"] - r["kendalls_w_old"],
                    }
                )
        elif family == "selection":
            merged = old.merge(new, on="baseline", suffixes=("_old", "_new"))
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "family": family,
                        "status": "compared",
                        "item": f"{r['baseline']}|mean_diff",
                        "old": r["mean_diff_old"],
                        "new": r["mean_diff_new"],
                        "delta": r["mean_diff_new"] - r["mean_diff_old"],
                    }
                )
                rows.append(
                    {
                        "family": family,
                        "status": "compared",
                        "item": f"{r['baseline']}|wilcoxon_p",
                        "old": r["wilcoxon_p_old"],
                        "new": r["wilcoxon_p_new"],
                        "delta": r["wilcoxon_p_new"] - r["wilcoxon_p_old"],
                    }
                )
        else:
            merged = old.merge(new, on="metric", suffixes=("_old", "_new"))
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "family": family,
                        "status": "compared",
                        "item": f"{r['metric']}|mean_spearman_vs_T1",
                        "old": r["mean_spearman_vs_T1_old"],
                        "new": r["mean_spearman_vs_T1_new"],
                        "delta": r["mean_spearman_vs_T1_new"] - r["mean_spearman_vs_T1_old"],
                    }
                )
    pd.DataFrame(rows).to_csv(paths["reconciliation"], index=False)
    transcript.log(f"Saved old-vs-duplicate-safe reconciliation: {paths['reconciliation']}")


def write_environment(paths: dict[str, Path]) -> None:
    env = {
        "run_family": PREFIX,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "command": " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": sklearn_version,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "protocol": {
            "seed_base": SEED,
            "repeats": N_REPEATS,
            "temperatures": TEMPS,
            "candidate_pool": ["logreg_C0.1", "logreg_C1.0", "logreg_C10.0", "hgb_d3", "hgb_d6"],
            "calibration": "CalibratedClassifierCV(method='isotonic', cv=2), fit on training fold only",
            "split": "exact feature-label grouped stratified 75/25 then 70/30 train/validation within trainval",
        },
    }
    paths["environment"].write_text(json.dumps(env, indent=2), encoding="utf-8")


def write_artifact_manifest(paths: dict[str, Path], transcript: Transcript) -> None:
    rows: list[dict[str, object]] = []
    for key, path in paths.items():
        if key in {"artifact_manifest", "transcript"}:
            continue
        if path.exists():
            rows.append(
                {
                    "artifact_key": key,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "last_write_time": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    with paths["artifact_manifest"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact_key", "path", "bytes", "sha256", "last_write_time"])
        writer.writeheader()
        writer.writerows(rows)
    transcript.log(f"Saved artifact manifest excluding live transcript self-hash: {paths['artifact_manifest']}")


def verify_outputs(paths: dict[str, Path], transcript: Transcript) -> None:
    util = pd.read_csv(paths["utility"])
    ranks = pd.read_csv(paths["rankings"])
    cal = pd.read_csv(paths["calibration"])
    splits = pd.read_csv(paths["split_manifest"])
    expected_blocks = len(dataset_specs()) * N_REPEATS
    expected_metric_rows = expected_blocks * len(METRICS)
    expected_cal_rows = expected_blocks * len(METRICS) * len(TEMPS)
    checks = {
        "utility_rows": len(util) == expected_metric_rows,
        "ranking_rows": len(ranks) == expected_metric_rows,
        "calibration_rows": len(cal) == expected_cal_rows,
        "split_rows": len(splits) == expected_blocks,
        "train_validation_group_overlap_zero": int(splits["train_validation_group_overlap"].sum()) == 0,
        "train_test_group_overlap_zero": int(splits["train_test_group_overlap"].sum()) == 0,
        "validation_test_group_overlap_zero": int(splits["validation_test_group_overlap"].sum()) == 0,
    }
    for name, ok in checks.items():
        transcript.log(f"VERIFY {name}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError(f"Output verification failed: {checks}")


def run_pipeline(force: bool) -> int:
    paths = output_paths()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(paths["transcript"])
    start = time.time()
    exit_code = 1
    try:
        transcript.log("Duplicate-safe harden pipeline started")
        transcript.log(f"Command: {' '.join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])}")
        transcript.log(f"CWD: {os.getcwd()}")
        transcript.log(f"Python: {sys.version.split()[0]} on {platform.platform()}")
        ensure_clean_outputs(paths, force=force, transcript=transcript)
        write_environment(paths)
        run_raw(paths, transcript)
        ranks_stab, ranks_sel, cd = analyze(paths, transcript)
        make_figures_and_table(paths, ranks_stab, ranks_sel, cd, transcript)
        write_reconciliation(paths, transcript)
        verify_outputs(paths, transcript)
        write_artifact_manifest(paths, transcript)
        exit_code = 0
        return 0
    except Exception:
        transcript.log("ERROR:\n" + traceback.format_exc())
        return 1
    finally:
        elapsed = time.time() - start
        transcript.log(f"Duplicate-safe harden pipeline finished exit_code={exit_code} elapsed_seconds={elapsed:.2f}")
        transcript.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing harden_dupsafe_* outputs")
    args = parser.parse_args()
    return run_pipeline(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
