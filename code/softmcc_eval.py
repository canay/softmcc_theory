"""softmcc_eval.py - Valid evaluation/model-selection harness for Paper 1 (SoftMCC).
Implements 03_experiments/DESIGN.md. No fabricated numbers; everything from real fits.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss)

warnings.filterwarnings("ignore")


def soft_mcc(y_true, y_prob, eps=1e-12):
    """Threshold-free probabilistic MCC (soft confusion-matrix counts)."""
    y = np.asarray(y_true, dtype=float); p = np.asarray(y_prob, dtype=float)
    tp = np.sum(p * y); fp = np.sum(p * (1 - y))
    fn = np.sum((1 - p) * y); tn = np.sum((1 - p) * (1 - y))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    return num / den


def _thr_grid(p, n=99):
    qs = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n)))
    return qs if len(qs) else np.array([0.5])


def best_threshold(y, p, metric="mcc"):
    best_t, best_v = 0.5, -np.inf
    for t in _thr_grid(p):
        yp = (p >= t).astype(int)
        v = matthews_corrcoef(y, yp) if metric == "mcc" else f1_score(y, yp, zero_division=0)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def selection_scores(y_val, p_val):
    mcc_best_t, mcc_best_v = best_threshold(y_val, p_val, "mcc")
    _, f1_best_v = best_threshold(y_val, p_val, "f1")
    out = {"SoftMCC": soft_mcc(y_val, p_val), "MCC_best": mcc_best_v,
           "F1_best": f1_best_v, "AUPRC": average_precision_score(y_val, p_val),
           "AUROC": roc_auc_score(y_val, p_val),
           "Brier_neg": -brier_score_loss(y_val, p_val),
           "MCC_05": matthews_corrcoef(y_val, (p_val >= 0.5).astype(int))}
    return out, mcc_best_t


@dataclass
class Candidate:
    name: str
    estimator: object


def default_pool(calibrate=True, fast=True):
    base = []
    for C in ([0.1, 1.0] if fast else [0.1, 1.0, 10.0]):
        base.append((f"logreg_C{C}", LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000, solver="liblinear")))
    if fast:
        base.append(("hgb", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.1, max_iter=150, random_state=0)))
        base.append(("rf", RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=0, n_jobs=-1)))
    else:
        for d in (3, 6):
            base.append((f"hgb_d{d}", HistGradientBoostingClassifier(
                max_depth=d, learning_rate=0.1, max_iter=150, random_state=0)))
        for n in (100, 200):
            base.append((f"rf_n{n}", RandomForestClassifier(
                n_estimators=n, class_weight="balanced", random_state=0, n_jobs=-1)))
    pool = []
    for name, est in base:
        if calibrate:
            est = CalibratedClassifierCV(est, method="isotonic", cv=2)
        pool.append(Candidate(name, est))
    return pool


def fit_pool_probs(X, y, seed, calibrate=True, fast=True):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.30, stratify=ytr, random_state=seed + 1)
    pool = default_pool(calibrate=calibrate, fast=fast)
    out = {}
    for cand in pool:
        cand.estimator.fit(Xtr2, ytr2)
        out[cand.name] = {"yva": yva, "pva": cand.estimator.predict_proba(Xva)[:, 1],
                          "yte": yte, "pte": cand.estimator.predict_proba(Xte)[:, 1]}
    return out, [c.name for c in pool]


def temperature_scale(p, T, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def run_repeat(X, y, seed, calibrate=True, fast=True):
    probs, cand_order = fit_pool_probs(X, y, seed, calibrate=calibrate, fast=fast)
    val_scores, test_cache = {}, {}
    for c in cand_order:
        scores, mcc_t = selection_scores(probs[c]["yva"], probs[c]["pva"])
        val_scores[c] = scores
        test_cache[c] = (probs[c]["yte"], probs[c]["pte"], mcc_t)
    metrics = list(next(iter(val_scores.values())).keys())
    rankings, selected = {}, {}
    for m in metrics:
        ordered = sorted(val_scores, key=lambda c: val_scores[c][m], reverse=True)
        rankings[m] = ordered; selected[m] = ordered[0]
    test_utility = {}
    for m in metrics:
        yte_, pte_, mcc_t = test_cache[selected[m]]
        test_utility[m] = matthews_corrcoef(yte_, (pte_ >= mcc_t).astype(int))
    return {"rankings": rankings, "selected": selected,
            "test_utility": test_utility, "candidates": cand_order}


def kendalls_w(rank_lists, items):
    n = len(items); m = len(rank_lists)
    if m < 2 or n < 2:
        return float("nan")
    idx = {it: i for i, it in enumerate(items)}
    R = np.zeros(n)
    for rl in rank_lists:
        for pos, it in enumerate(rl):
            R[idx[it]] += pos + 1
    S = np.sum((R - R.mean()) ** 2)
    return 12 * S / (m ** 2 * (n ** 3 - n))


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))
