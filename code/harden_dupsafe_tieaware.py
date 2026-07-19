"""Tie-aware duplicate-safe hardening and control pipeline for SoftMCC Theory.

This runner reuses the data, model pool, grouped splits, and metric definitions
from ``harden_dupsafe_full.py`` but writes a new immutable dated run.  It does
not edit manuscript-facing files.  Its two ranking policies are deliberately
separate:

* model selection uses the pre-specified candidate declaration order only to
  break exact score ties and choose one deployable model;
* Kendall's W and calibration-rank agreement use average ranks, with the
  standard tie correction in W.

The run archives every candidate's validation scores and a compressed bundle
of validation labels/probabilities so that tie and label-permutation controls
can be recomputed without refitting the models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata, spearmanr, wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.metrics import log_loss, matthews_corrcoef


def find_project_root(start: Path) -> Path:
    configured = os.environ.get("SOFTMCC_PROJECT_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "02_data").is_dir():
            return candidate
        raise RuntimeError(
            "SOFTMCC_PROJECT_ROOT must contain an existing 02_data directory"
        )
    for candidate in [start, *start.parents]:
        is_project = (candidate / "02_data").is_dir() and (
            (candidate / "04_manuscript").is_dir()
            or ((candidate / "code").is_dir() and (candidate / "results").is_dir())
        )
        if is_project:
            return candidate
    raise RuntimeError(
        "Could not locate a project/package root with 02_data; set "
        "SOFTMCC_PROJECT_ROOT explicitly"
    )


HERE = Path(__file__).resolve().parent
ROOT = find_project_root(HERE)
SOURCE_SCRIPTS = ROOT / "03_experiments" / "scripts"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
import harden_dupsafe_full as H  # noqa: E402


DEFAULT_RUN_ID = "2026-07-18_codex_local_unknown_tieaware_harden_control"
PERMUTATION_SEED_BASE = 20260718
N_PERMUTATIONS = 200
BOOTSTRAP_REPLICATES = 2000


class Transcript:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("w", encoding="utf-8", newline="\n")

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(run_root: Path) -> dict[str, Path]:
    evidence = run_root / "evidence"
    figures = run_root / "figures"
    tables = run_root / "tables"
    return {
        "utility": evidence / "harden_dupsafe_tieaware_utility.csv",
        "rankings": evidence / "harden_dupsafe_tieaware_rankings.csv",
        "candidate_scores": evidence / "harden_dupsafe_tieaware_candidate_scores.csv",
        "calibration": evidence / "harden_dupsafe_tieaware_calibration.csv",
        "split_manifest": evidence / "harden_dupsafe_tieaware_split_manifest.csv",
        "dataset_manifest": evidence / "harden_dupsafe_tieaware_dataset_manifest.csv",
        "prediction_bundle": evidence / "harden_dupsafe_tieaware_validation_predictions.npz",
        "environment": evidence / "harden_dupsafe_tieaware_environment.json",
        "stability_summary": evidence / "harden_dupsafe_tieaware_summary_stability.csv",
        "selection_summary": evidence / "harden_dupsafe_tieaware_summary_selection.csv",
        "effect_sizes": evidence / "harden_dupsafe_tieaware_effect_sizes.csv",
        "calibration_summary": evidence / "harden_dupsafe_tieaware_summary_calibration.csv",
        "friedman": evidence / "harden_dupsafe_tieaware_friedman.txt",
        "tie_audit": evidence / "harden_dupsafe_tieaware_tie_audit.csv",
        "reconciliation": evidence / "harden_dupsafe_tieaware_vs_ordinal_reconciliation.csv",
        "kappa_per_dataset": evidence / "kappa_tieaware_per_dataset.csv",
        "kappa_spearman": evidence / "kappa_tieaware_spearman.csv",
        "kappa_perm_scores": evidence / "kappa_tieaware_permutation_candidate_scores.csv.gz",
        "kappa_perm_w": evidence / "kappa_tieaware_permutation_w_distribution.csv",
        "kappa_summary": evidence / "kappa_tieaware_summary.json",
        "fig_stability": figures / "fig1_resampling_stability_tieaware.png",
        "fig_calibration": figures / "fig2_calibration_sensitivity_tieaware.png",
        "fig_cd": figures / "fig3_critical_difference_tieaware.png",
        "table": tables / "table_main_results_tieaware.tex",
        "artifact_manifest": run_root / "ARTIFACT_MANIFEST.csv",
        "transcript": run_root / "run.log",
    }


def midrank_vector(values: Iterable[float]) -> np.ndarray:
    """Return ranks 1..M with rank 1 assigned to the largest score."""
    vals = np.asarray(list(values), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"Non-finite candidate score encountered: {vals}")
    return np.asarray(rankdata(-vals, method="average"), dtype=float)


def tie_correction(rank_vector: np.ndarray) -> tuple[float, int, int]:
    """Return sum(t^3-t), number of tied groups, and largest tied group."""
    _, counts = np.unique(np.asarray(rank_vector, dtype=float), return_counts=True)
    tied = counts[counts > 1]
    return (
        float(np.sum(tied**3 - tied)) if len(tied) else 0.0,
        int(len(tied)),
        int(np.max(tied)) if len(tied) else 1,
    )


def kendalls_w_tie_corrected(rank_vectors: Iterable[Iterable[float]]) -> float:
    """Kendall's W for average ranks with the standard per-judge tie correction."""
    matrix = np.asarray(list(rank_vectors), dtype=float)
    if matrix.ndim != 2:
        raise ValueError("rank_vectors must be a two-dimensional repeat-by-candidate matrix")
    m, n = matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = matrix.sum(axis=0)
    expected = m * (n + 1) / 2.0
    s_value = float(np.sum((rank_sums - expected) ** 2))
    total_tie = sum(tie_correction(row)[0] for row in matrix)
    denominator = float(m**2 * (n**3 - n) - m * total_tie)
    if denominator <= 0:
        return float("nan")
    value = 12.0 * s_value / denominator
    if value < -1e-12 or value > 1 + 1e-12:
        raise RuntimeError(f"Tie-corrected Kendall W outside [0,1]: {value}")
    return float(np.clip(value, 0.0, 1.0))


def deterministic_order(score_map: dict[str, float], candidate_order: list[str]) -> list[str]:
    """Choose a model reproducibly; exact ties retain declaration order."""
    return sorted(candidate_order, key=lambda name: score_map[name], reverse=True)


def bca_ci(
    data: Iterable[object],
    statistic: Callable[[list[object]], float],
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    observations = list(data)
    n = len(observations)
    theta = float(statistic(observations))
    if n < 2 or not np.isfinite(theta):
        return float("nan"), float("nan")
    boots = np.asarray(
        [statistic([observations[i] for i in rng.integers(0, n, n)]) for _ in range(replicates)],
        dtype=float,
    )
    boots = boots[np.isfinite(boots)]
    if len(boots) < max(20, replicates // 10):
        return float("nan"), float("nan")
    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1.0 / (len(boots) + 1)), 1.0 - 1.0 / (len(boots) + 1))
    z0 = float(norm.ppf(prop))
    jack = np.asarray(
        [statistic([observations[j] for j in range(n) if j != i]) for i in range(n)],
        dtype=float,
    )
    if np.all(np.isfinite(jack)):
        jack_mean = float(np.mean(jack))
        numerator = float(np.sum((jack_mean - jack) ** 3))
        denominator = float(6.0 * np.sum((jack_mean - jack) ** 2) ** 1.5)
        acceleration = 0.0 if denominator == 0 else numerator / denominator
    else:
        acceleration = 0.0

    def adjusted(q: float) -> float:
        zq = float(norm.ppf(q))
        denom = 1.0 - acceleration * (z0 + zq)
        if denom == 0:
            return q
        return float(norm.cdf(z0 + (z0 + zq) / denom))

    q_lo = float(np.clip(adjusted(0.025), 0.0, 1.0))
    q_hi = float(np.clip(adjusted(0.975), 0.0, 1.0))
    return float(np.quantile(boots, q_lo)), float(np.quantile(boots, q_hi))


def matched_rank_biserial(differences: Iterable[float]) -> float:
    diffs = np.asarray(list(differences), dtype=float)
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if len(diffs) == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs), method="average")
    total = float(np.sum(ranks))
    return float((np.sum(ranks[diffs > 0]) - np.sum(ranks[diffs < 0])) / total)


def paired_sign_effect(differences: Iterable[float]) -> float:
    diffs = np.asarray(list(differences), dtype=float)
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if len(diffs) == 0:
        return 0.0
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / len(diffs))


def self_test() -> None:
    no_ties = np.asarray([[1, 2, 3], [1, 2, 3]], dtype=float)
    reversed_pair = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=float)
    tied_identical = np.asarray([[1.5, 1.5, 3], [1.5, 1.5, 3]], dtype=float)
    all_tied = np.asarray([[2, 2, 2], [2, 2, 2]], dtype=float)
    assert np.isclose(kendalls_w_tie_corrected(no_ties), 1.0)
    assert np.isclose(kendalls_w_tie_corrected(reversed_pair), 0.0)
    assert np.isclose(kendalls_w_tie_corrected(tied_identical), 1.0)
    assert np.isnan(kendalls_w_tie_corrected(all_tied))
    assert np.allclose(midrank_vector([3.0, 3.0, 1.0]), [1.5, 1.5, 3.0])


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_raw(paths: dict[str, Path], transcript: Transcript) -> None:
    utility_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    block_meta: list[tuple[str, str, int, int]] = []
    validation_labels: list[np.ndarray] = []
    validation_probabilities: list[np.ndarray] = []
    candidate_test_mcc: list[np.ndarray] = []

    for dataset_index, spec in enumerate(H.dataset_specs()):
        X, y = spec.loader()
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        groups, duplicate_rows = H.row_group_ids(X, y)
        unique_groups, first_indices = np.unique(groups, return_index=True)
        group_labels = {int(group): int(y[index]) for group, index in zip(unique_groups, first_indices)}
        positives, negatives, prevalence = H.class_counts(y)
        dataset_rows.append(
            {
                "dataset_key": spec.key,
                "dataset": spec.label,
                "source": spec.source,
                "cache_path": str(spec.cache_path) if spec.cache_path else "",
                "cache_sha256": H.sha256_file(spec.cache_path) if spec.cache_path else "",
                "rows": int(len(y)),
                "features": int(X.reshape((X.shape[0], -1)).shape[1]),
                "positive_rows": positives,
                "negative_rows": negatives,
                "prevalence": prevalence,
                "unique_exact_feature_label_groups": int(len(unique_groups)),
                "exact_duplicate_rows_with_label": duplicate_rows,
                "X_sha256": H.sha256_array(X),
                "y_sha256": H.sha256_array(y),
            }
        )
        transcript.log(
            f"Dataset {spec.label}: rows={len(y)} groups={len(unique_groups)} "
            f"duplicate_rows={duplicate_rows} prevalence={prevalence:.6f}"
        )

        for repeat in range(H.N_REPEATS):
            seed = H.SEED + repeat
            trainval_groups, test_groups = H.split_groups(groups, group_labels, unique_groups, 0.25, seed)
            train_groups, validation_groups = H.split_groups(
                groups, group_labels, trainval_groups, 0.30, seed + 1
            )
            train_index = H.indices_for_groups(groups, train_groups)
            validation_index = H.indices_for_groups(groups, validation_groups)
            test_index = H.indices_for_groups(groups, test_groups)
            split_rows.append(
                H.split_manifest_row(
                    spec,
                    repeat,
                    seed,
                    y,
                    groups,
                    train_index,
                    validation_index,
                    test_index,
                    duplicate_rows,
                )
            )

            X_train, y_train = X[train_index], y[train_index]
            X_validation, y_validation = X[validation_index], y[validation_index]
            X_test, y_test = X[test_index], y[test_index]
            validation_scores: dict[str, dict[str, float]] = {}
            validation_probs: dict[str, np.ndarray] = {}
            test_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

            for name, estimator in H.pool():
                estimator.fit(X_train, y_train)
                p_validation = estimator.predict_proba(X_validation)[:, 1]
                p_test = estimator.predict_proba(X_test)[:, 1]
                scores, mcc_threshold = H.selection_scores(y_validation, p_validation)
                validation_scores[name] = scores
                validation_probs[name] = p_validation
                test_cache[name] = (y_test, p_test, mcc_threshold)

            candidate_order = list(validation_scores)
            test_values: list[float] = []
            for name in candidate_order:
                y_test_candidate, p_test_candidate, threshold = test_cache[name]
                test_mcc = float(
                    matthews_corrcoef(y_test_candidate, (p_test_candidate >= threshold).astype(int))
                )
                test_values.append(test_mcc)
                probabilities = validation_probs[name]
                score_row: dict[str, object] = {
                    "dataset_key": spec.key,
                    "dataset": spec.label,
                    "repeat": repeat,
                    "seed": seed,
                    "candidate": name,
                    "candidate_index": candidate_order.index(name),
                    "validation_mccbest_threshold": threshold,
                    "test_mcc_at_validation_mccbest_threshold": test_mcc,
                    "validation_log_loss": float(
                        log_loss(y_validation, np.clip(probabilities, 1e-12, 1 - 1e-12))
                    ),
                    "validation_ece_10": H.ece_score(y_validation, probabilities, bins=10),
                    "validation_prob_mean": float(np.mean(probabilities)),
                    "validation_prob_variance": float(np.var(probabilities)),
                    "validation_pos_rate": float(np.mean(y_validation)),
                }
                for metric in H.METRICS:
                    score_row[f"score_{metric}"] = float(validation_scores[name][metric])
                candidate_rows.append(score_row)

            for metric in H.METRICS:
                score_map = {name: float(validation_scores[name][metric]) for name in candidate_order}
                values = [score_map[name] for name in candidate_order]
                ranks = midrank_vector(values)
                correction, tied_groups, largest_tie = tie_correction(ranks)
                ordered = deterministic_order(score_map, candidate_order)
                ranking_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset": spec.label,
                        "repeat": repeat,
                        "seed": seed,
                        "metric": metric,
                        "candidate_order_json": json.dumps(candidate_order),
                        "validation_scores_json": json.dumps(values),
                        "midranks_json": json.dumps(ranks.tolist()),
                        "tie_correction_T": correction,
                        "tied_group_count": tied_groups,
                        "largest_tie_size": largest_tie,
                        "deterministic_selection_order": ">".join(ordered),
                        "selected_candidate": ordered[0],
                        "selection_tie_break": "candidate_declaration_order_only_for_exact_score_ties",
                    }
                )
                selected_index = candidate_order.index(ordered[0])
                utility_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset": spec.label,
                        "repeat": repeat,
                        "seed": seed,
                        "metric": metric,
                        "selected_candidate": ordered[0],
                        "validation_metric_value": score_map[ordered[0]],
                        "test_mcc": test_values[selected_index],
                    }
                )

            baseline_ranks = {
                metric: midrank_vector(
                    [validation_scores[name][metric] for name in candidate_order]
                )
                for metric in H.METRICS
            }
            for temperature in H.TEMPS:
                shifted_scores = {
                    name: H.selection_scores(
                        y_validation, H.temperature_scale(validation_probs[name], temperature)
                    )[0]
                    for name in candidate_order
                }
                for metric in H.METRICS:
                    shifted_ranks = midrank_vector(
                        [shifted_scores[name][metric] for name in candidate_order]
                    )
                    agreement = spearmanr(baseline_ranks[metric], shifted_ranks).statistic
                    calibration_rows.append(
                        {
                            "dataset_key": spec.key,
                            "dataset": spec.label,
                            "repeat": repeat,
                            "seed": seed,
                            "T": temperature,
                            "metric": metric,
                            "spearman_vs_T1": 1.0 if np.isnan(agreement) else float(agreement),
                        }
                    )

            block_meta.append((spec.key, spec.label, repeat, seed))
            validation_labels.append(np.asarray(y_validation, dtype=np.int8))
            validation_probabilities.append(
                np.column_stack([validation_probs[name] for name in candidate_order]).astype(np.float64)
            )
            candidate_test_mcc.append(np.asarray(test_values, dtype=np.float64))
            transcript.log(f"{spec.label} repeat {repeat + 1}/{H.N_REPEATS} complete")

        save_csv(utility_rows, paths["utility"])
        save_csv(ranking_rows, paths["rankings"])
        save_csv(candidate_rows, paths["candidate_scores"])
        save_csv(calibration_rows, paths["calibration"])
        save_csv(split_rows, paths["split_manifest"])
        save_csv(dataset_rows, paths["dataset_manifest"])
        transcript.log(f"Checkpoint saved after dataset {spec.label}")

    offsets = [0]
    for labels in validation_labels:
        offsets.append(offsets[-1] + len(labels))
    np.savez_compressed(
        paths["prediction_bundle"],
        dataset_key=np.asarray([row[0] for row in block_meta]),
        dataset=np.asarray([row[1] for row in block_meta]),
        repeat=np.asarray([row[2] for row in block_meta], dtype=np.int16),
        seed=np.asarray([row[3] for row in block_meta], dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_order=np.asarray(candidate_order),
        y_validation=np.concatenate(validation_labels).astype(np.int8),
        p_validation=np.vstack(validation_probabilities).astype(np.float64),
        candidate_test_mcc=np.vstack(candidate_test_mcc).astype(np.float64),
    )
    transcript.log(f"Saved validation prediction bundle: {paths['prediction_bundle']}")


def effect_size_outputs(utility: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    base = utility[utility.metric == "SoftMCC"].set_index(["dataset", "repeat"])["test_mcc"]
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for metric in [item for item in H.METRICS if item != "SoftMCC"]:
        comparator = utility[utility.metric == metric].set_index(["dataset", "repeat"])["test_mcc"]
        aligned_base, aligned_comparator = base.align(comparator, join="inner")
        dataset_cliffs: list[float] = []
        dataset_rb: list[float] = []
        for dataset in aligned_base.index.get_level_values("dataset").unique():
            a = aligned_base.xs(dataset, level="dataset").sort_index()
            b = aligned_comparator.xs(dataset, level="dataset").sort_index()
            diffs = (a - b).to_numpy()
            cliff = H.cliffs_delta(a.to_numpy(), b.to_numpy())
            rb = matched_rank_biserial(diffs)
            dataset_cliffs.append(cliff)
            dataset_rb.append(rb)
            detail_rows.append(
                {
                    "baseline": H.PRETTY[metric],
                    "dataset": dataset,
                    "n_pairs": len(diffs),
                    "mean_difference": float(np.mean(diffs)),
                    "within_dataset_unpaired_cliffs_delta": cliff,
                    "matched_pairs_rank_biserial": rb,
                    "paired_sign_effect": paired_sign_effect(diffs),
                    "positive_pairs": int(np.sum(diffs > 0)),
                    "negative_pairs": int(np.sum(diffs < 0)),
                    "zero_pairs": int(np.sum(diffs == 0)),
                }
            )
        all_differences = (aligned_base - aligned_comparator).to_numpy()
        summary_rows.append(
            {
                "baseline": H.PRETTY[metric],
                "mean_diff": float(np.mean(all_differences)),
                "within_dataset_cliffs_delta_mean": float(np.mean(dataset_cliffs)),
                "within_dataset_cliffs_delta_min": float(np.min(dataset_cliffs)),
                "within_dataset_cliffs_delta_max": float(np.max(dataset_cliffs)),
                "matched_pairs_rank_biserial_all_blocks": matched_rank_biserial(all_differences),
                "matched_pairs_rank_biserial_dataset_mean": float(np.mean(dataset_rb)),
                "matched_pairs_rank_biserial_dataset_min": float(np.min(dataset_rb)),
                "matched_pairs_rank_biserial_dataset_max": float(np.max(dataset_rb)),
                "paired_sign_effect_all_blocks": paired_sign_effect(all_differences),
            }
        )
    save_csv(detail_rows, paths["effect_sizes"])
    return pd.DataFrame(summary_rows)


def analyze(paths: dict[str, Path], transcript: Transcript) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(H.SEED)
    utility = pd.read_csv(paths["utility"])
    rankings = pd.read_csv(paths["rankings"])
    calibration = pd.read_csv(paths["calibration"])
    datasets = list(utility.dataset.unique())
    lines: list[str] = []

    stability_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for metric in H.METRICS:
            subset = rankings[(rankings.dataset == dataset) & (rankings.metric == metric)].sort_values("repeat")
            rank_vectors = [np.asarray(json.loads(value), dtype=float) for value in subset.midranks_json]
            value = kendalls_w_tie_corrected(rank_vectors)
            lower, upper = bca_ci(rank_vectors, kendalls_w_tie_corrected, rng)
            stability_rows.append(
                {
                    "dataset": dataset,
                    "metric": H.PRETTY[metric],
                    "kendalls_w": value,
                    "bca_lo": lower,
                    "bca_hi": upper,
                    "ranking_policy": "midranks_with_standard_tie_corrected_denominator",
                    "repeat_count": len(rank_vectors),
                }
            )
            tie_rows.append(
                {
                    "dataset": dataset,
                    "metric": H.PRETTY[metric],
                    "blocks_with_any_tie": int(np.sum(subset.tied_group_count > 0)),
                    "total_tied_groups": int(subset.tied_group_count.sum()),
                    "largest_tie_size": int(subset.largest_tie_size.max()),
                    "total_tie_correction_T": float(subset.tie_correction_T.sum()),
                    "repeat_count": len(subset),
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(paths["stability_summary"], index=False)
    save_csv(tie_rows, paths["tie_audit"])

    effects = effect_size_outputs(utility, paths)
    selection_rows: list[dict[str, object]] = []
    base = utility[utility.metric == "SoftMCC"].set_index(["dataset", "repeat"])["test_mcc"]
    for metric in [item for item in H.METRICS if item != "SoftMCC"]:
        comparator = utility[utility.metric == metric].set_index(["dataset", "repeat"])["test_mcc"]
        aligned_base, aligned_comparator = base.align(comparator, join="inner")
        differences = (aligned_base - aligned_comparator).to_numpy()
        if np.allclose(differences, 0):
            lower, upper, p_value = 0.0, 0.0, 1.0
        else:
            lower, upper = bca_ci(differences.tolist(), lambda values: float(np.mean(values)), rng)
            try:
                p_value = float(wilcoxon(differences).pvalue)
            except Exception:
                p_value = float("nan")
        effect = effects[effects.baseline == H.PRETTY[metric]].iloc[0].to_dict()
        selection_rows.append(
            {
                "baseline": H.PRETTY[metric],
                "mean_diff": float(np.mean(differences)),
                "bca_lo": lower,
                "bca_hi": upper,
                "wilcoxon_p": p_value,
                **{key: value for key, value in effect.items() if key not in {"baseline", "mean_diff"}},
            }
        )
    pd.DataFrame(selection_rows).to_csv(paths["selection_summary"], index=False)

    calibration_rows: list[dict[str, object]] = []
    means = calibration.groupby(["dataset", "repeat", "metric"])["spearman_vs_T1"].mean().reset_index()
    for metric in H.METRICS:
        values = means[means.metric == metric].spearman_vs_T1.to_numpy()
        lower, upper = bca_ci(values.tolist(), lambda items: float(np.mean(items)), rng)
        calibration_rows.append(
            {
                "metric": H.PRETTY[metric],
                "mean_spearman_vs_T1": float(np.mean(values)),
                "bca_lo": lower,
                "bca_hi": upper,
            }
        )
    pd.DataFrame(calibration_rows).to_csv(paths["calibration_summary"], index=False)

    k = len(H.METRICS)
    n_datasets = len(datasets)
    cd = H.Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    def friedman_block(values_by_metric: dict[str, np.ndarray], label: str) -> np.ndarray:
        matrix = np.asarray([values_by_metric[metric] for metric in H.METRICS], dtype=float)
        statistic, p_value = friedmanchisquare(*[matrix[index] for index in range(k)])
        rank_matrix = np.asarray(
            [pd.Series(-matrix[:, column]).rank(method="average").to_numpy() for column in range(n_datasets)]
        ).T
        mean_ranks = rank_matrix.mean(axis=1)
        lines.append(
            f"[{label}] Friedman chi2={statistic:.3f} p={p_value:.4g} "
            f"(k={k}, N={n_datasets}); Nemenyi CD(0.05)={cd:.3f}"
        )
        for metric, mean_rank in sorted(zip(H.METRICS, mean_ranks), key=lambda pair: pair[1]):
            lines.append(f"    {H.PRETTY[metric]:9s} mean rank {mean_rank:.3f}")
        return mean_ranks

    stability_pivot = stability.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    stability_values = {
        metric: stability_pivot.loc[H.PRETTY[metric], datasets].to_numpy() for metric in H.METRICS
    }
    stability_ranks = friedman_block(stability_values, "stability (tie-corrected Kendall's W)")
    utility_pivot = utility.pivot_table(index="metric", columns="dataset", values="test_mcc", aggfunc="mean")
    utility_values = {metric: utility_pivot.loc[metric, datasets].to_numpy() for metric in H.METRICS}
    selection_ranks = friedman_block(utility_values, "selection (mean test MCC)")
    paths["friedman"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        transcript.log(line)
    return stability_ranks, selection_ranks, float(cd)


def run_kappa_controls(paths: dict[str, Path], transcript: Transcript) -> None:
    bundle = np.load(paths["prediction_bundle"], allow_pickle=False)
    dataset_keys = bundle["dataset_key"]
    datasets = bundle["dataset"]
    repeats = bundle["repeat"]
    seeds = bundle["seed"]
    offsets = bundle["offsets"]
    candidates = bundle["candidate_order"].tolist()
    labels = bundle["y_validation"]
    probabilities = bundle["p_validation"]
    test_mcc = bundle["candidate_test_mcc"]

    base_ranks: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    base_scores: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    selected_test: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    spearman_rows: list[dict[str, object]] = []
    perm_rank_store: dict[int, dict[str, list[np.ndarray]]] = {
        index: defaultdict(list) for index in range(N_PERMUTATIONS)
    }
    permutation_score_rows: list[dict[str, object]] = []

    for block in range(len(datasets)):
        start, stop = int(offsets[block]), int(offsets[block + 1])
        y_validation = labels[start:stop].astype(float)
        p_validation = probabilities[start:stop, :]
        dataset = str(datasets[block])
        repeat = int(repeats[block])
        dataset_index = list(dict.fromkeys(datasets.tolist())).index(dataset)
        soft_scores = np.asarray(
            [H.soft_mcc(y_validation, p_validation[:, index]) for index in range(len(candidates))]
        )
        kappa_scores = np.asarray(
            [
                math.sqrt(
                    np.var(p_validation[:, index])
                    / max(np.mean(p_validation[:, index]) * (1 - np.mean(p_validation[:, index])), 1e-15)
                )
                for index in range(len(candidates))
            ]
        )
        rho_scores = np.asarray(
            [
                0.0
                if np.std(p_validation[:, index]) == 0 or np.std(y_validation) == 0
                else float(np.corrcoef(p_validation[:, index], y_validation)[0, 1])
                for index in range(len(candidates))
            ]
        )
        score_sets = {"SoftMCC": soft_scores, "kappa_only": kappa_scores, "rho_only": rho_scores}
        base_scores[(dataset, repeat)] = score_sets
        for label, score_values in score_sets.items():
            ranks = midrank_vector(score_values)
            base_ranks[dataset][label].append(ranks)
            order = deterministic_order(dict(zip(candidates, score_values)), candidates)
            selected_test[dataset][label].append(float(test_mcc[block, candidates.index(order[0])]))
        spearman_rows.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "seed": int(seeds[block]),
                "spearman_soft_vs_kappa": float(spearmanr(midrank_vector(soft_scores), midrank_vector(kappa_scores)).statistic),
                "spearman_soft_vs_rho": float(spearmanr(midrank_vector(soft_scores), midrank_vector(rho_scores)).statistic),
            }
        )

        for permutation_index in range(N_PERMUTATIONS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [PERMUTATION_SEED_BASE, permutation_index, dataset_index, repeat]
                )
            )
            permuted_labels = rng.permutation(y_validation)
            scores = np.asarray(
                [H.soft_mcc(permuted_labels, p_validation[:, index]) for index in range(len(candidates))]
            )
            ranks = midrank_vector(scores)
            perm_rank_store[permutation_index][dataset].append(ranks)
            for candidate_index, candidate in enumerate(candidates):
                permutation_score_rows.append(
                    {
                        "permutation_index": permutation_index,
                        "seed_base": PERMUTATION_SEED_BASE,
                        "dataset": dataset,
                        "repeat": repeat,
                        "candidate": candidate,
                        "permuted_softmcc": float(scores[candidate_index]),
                        "midrank": float(ranks[candidate_index]),
                    }
                )
    pd.DataFrame(permutation_score_rows).to_csv(paths["kappa_perm_scores"], index=False, compression="gzip")
    save_csv(spearman_rows, paths["kappa_spearman"])

    per_dataset_rows: list[dict[str, object]] = []
    observed_by_dataset: dict[str, float] = {}
    for dataset in list(dict.fromkeys(datasets.tolist())):
        row: dict[str, object] = {"dataset": dataset}
        for label in ("SoftMCC", "kappa_only", "rho_only"):
            value = kendalls_w_tie_corrected(base_ranks[dataset][label])
            row[f"W_{label}"] = value
            row[f"meanTestMCC_{label}"] = float(np.mean(selected_test[dataset][label]))
            if label == "SoftMCC":
                observed_by_dataset[dataset] = value
        per_dataset_rows.append(row)
    save_csv(per_dataset_rows, paths["kappa_per_dataset"])

    benchmark = set(H.BENCH)
    real = set(H.REAL)
    perm_w_rows: list[dict[str, object]] = []
    for permutation_index in range(N_PERMUTATIONS):
        dataset_w: dict[str, float] = {}
        for dataset, rank_vectors in perm_rank_store[permutation_index].items():
            value = kendalls_w_tie_corrected(rank_vectors)
            dataset_w[dataset] = value
            perm_w_rows.append(
                {
                    "permutation_index": permutation_index,
                    "seed_base": PERMUTATION_SEED_BASE,
                    "scope": "dataset",
                    "dataset": dataset,
                    "kendalls_w": value,
                }
            )
        for scope, members in (("all6", set(dataset_w)), ("benchmark", benchmark), ("real", real)):
            values = [dataset_w[item] for item in members]
            perm_w_rows.append(
                {
                    "permutation_index": permutation_index,
                    "seed_base": PERMUTATION_SEED_BASE,
                    "scope": scope,
                    "dataset": "",
                    "kendalls_w": float(np.mean(values)),
                }
            )
    perm_w = pd.DataFrame(perm_w_rows)
    perm_w.to_csv(paths["kappa_perm_w"], index=False)

    observed_rows = pd.DataFrame(per_dataset_rows)
    observed_scope = {
        "all6": float(observed_rows.W_SoftMCC.mean()),
        "benchmark": float(observed_rows[observed_rows.dataset.isin(benchmark)].W_SoftMCC.mean()),
        "real": float(observed_rows[observed_rows.dataset.isin(real)].W_SoftMCC.mean()),
    }
    permutation_summary: dict[str, object] = {}
    for scope in ("all6", "benchmark", "real"):
        values = perm_w[(perm_w.scope == scope) & (perm_w.dataset.fillna("") == "")].kendalls_w.to_numpy()
        permutation_summary[scope] = {
            "observed_softmcc_w": observed_scope[scope],
            "permutation_count": int(len(values)),
            "permuted_w_mean": float(np.mean(values)),
            "permuted_w_sd": float(np.std(values, ddof=1)),
            "permuted_w_q025": float(np.quantile(values, 0.025)),
            "permuted_w_median": float(np.median(values)),
            "permuted_w_q975": float(np.quantile(values, 0.975)),
            "exceedance_fraction": float((1 + np.sum(values >= observed_scope[scope])) / (len(values) + 1)),
        }
    summary = {
        "ranking_policy": "midranks_with_standard_tie_corrected_kendalls_w",
        "permutation_design": {
            "seed_base": PERMUTATION_SEED_BASE,
            "permutation_count": N_PERMUTATIONS,
            "block_seed_sequence": "SeedSequence([seed_base, permutation_index, dataset_index, repeat])",
        },
        "observed": {
            "SoftMCC_W_all6": observed_scope["all6"],
            "kappa_only_W_all6": float(observed_rows.W_kappa_only.mean()),
            "rho_only_W_all6": float(observed_rows.W_rho_only.mean()),
            "mean_spearman_soft_vs_kappa": float(pd.DataFrame(spearman_rows).spearman_soft_vs_kappa.mean()),
            "mean_spearman_soft_vs_rho": float(pd.DataFrame(spearman_rows).spearman_soft_vs_rho.mean()),
            "max_abs_dataset_test_mcc_gap_soft_vs_kappa": float(
                np.max(np.abs(observed_rows.meanTestMCC_SoftMCC - observed_rows.meanTestMCC_kappa_only))
            ),
        },
        "permutation_distribution": permutation_summary,
    }
    paths["kappa_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    transcript.log(
        "Tie-aware kappa control complete: "
        f"SoftMCC W={summary['observed']['SoftMCC_W_all6']:.6f}, "
        f"permuted mean W={permutation_summary['all6']['permuted_w_mean']:.6f}"
    )


def reconcile_with_ordinal(paths: dict[str, Path], transcript: Transcript) -> None:
    old_root = ROOT / "experiments" / "2026-06-22_codex_local_unknown_duplicate_safe_harden_rerun" / "evidence"
    old_path = old_root / "harden_dupsafe_summary_stability.csv"
    if not old_path.exists():
        save_csv([{"status": "old_evidence_missing", "path": str(old_path)}], paths["reconciliation"])
        return
    old = pd.read_csv(old_path)
    new = pd.read_csv(paths["stability_summary"])
    merged = old.merge(new, on=["dataset", "metric"], suffixes=("_ordinal", "_tieaware"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "metric": row.metric,
                "kendalls_w_ordinal": row.kendalls_w_ordinal,
                "kendalls_w_tieaware": row.kendalls_w_tieaware,
                "delta_tieaware_minus_ordinal": row.kendalls_w_tieaware - row.kendalls_w_ordinal,
                "bca_lo_ordinal": row.bca_lo_ordinal,
                "bca_hi_ordinal": row.bca_hi_ordinal,
                "bca_lo_tieaware": row.bca_lo_tieaware,
                "bca_hi_tieaware": row.bca_hi_tieaware,
            }
        )
    save_csv(rows, paths["reconciliation"])
    transcript.log(f"Reconciled {len(rows)} dataset-metric W values against ordinal evidence")


def make_figures_and_table(
    paths: dict[str, Path], stability_ranks: np.ndarray, selection_ranks: np.ndarray, cd: float
) -> None:
    stability = pd.read_csv(paths["stability_summary"])
    pivot = stability.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    benchmark = pivot[H.BENCH].mean(axis=1)
    real = pivot[H.REAL].mean(axis=1)
    order = sorted(H.PRETTY_ORDER, key=lambda metric: -float((benchmark[metric] + real[metric]) / 2))

    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(order))
    width = 0.38
    left = axis.bar(x - width / 2, [benchmark[m] for m in order], width, label="Benchmark suite", color="#4878a8")
    right = axis.bar(x + width / 2, [real[m] for m in order], width, label="Real suite", color="#b04a4a")
    for bars in (left, right):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}", ha="center", fontsize=6.5)
    axis.set_xticks(x)
    axis.set_xticklabels(order, fontsize=8)
    axis.set_ylabel("Tie-corrected Kendall's W", fontsize=8)
    axis.set_ylim(0, 1.0)
    axis.tick_params(axis="y", labelsize=8)
    axis.legend(fontsize=7.5, frameon=False)
    axis.set_title("Duplicate-safe resampling stability (midranks; higher = more stable)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["fig_stability"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    calibration = pd.read_csv(paths["calibration"])
    calibration = calibration[calibration.dataset.isin(H.REAL)]
    styles = {
        "SoftMCC": ("#1f5fa8", "-", "o"), "Brier_neg": ("#b04a4a", "--", "s"),
        "MCC_best": ("#777777", ":", "^"), "F1_best": ("#999999", ":", "v"),
        "AUROC": ("#555555", ":", "D"), "AUPRC": ("#bbbbbb", ":", "P"),
        "MCC_05": ("#333333", ":", "X"),
    }
    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    for metric, (color, line_style, marker) in styles.items():
        values = calibration[calibration.metric == metric].groupby("T").spearman_vs_T1.mean()
        axis.plot(values.index, values.values, line_style, color=color, marker=marker, ms=3.5, lw=1.3, label=H.PRETTY[metric])
    axis.set_xlabel("Temperature T (1.0 = unshifted)", fontsize=8)
    axis.set_ylabel("Spearman rank agreement vs T=1", fontsize=8)
    axis.tick_params(labelsize=8)
    axis.set_ylim(max(0.0, min(0.6, float(calibration.spearman_vs_T1.min()) - 0.05)), 1.02)
    axis.legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    axis.set_title("Duplicate-safe ranking agreement under calibration shift (real suite)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["fig_calibration"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.7))
    H.cd_panel(axes[0], stability_ranks, cd, "Ranking stability: mean ranks over 6 datasets")
    H.cd_panel(axes[1], selection_ranks, cd, "Selection utility: mean ranks over 6 datasets")
    axes[0].text(0.5, -0.16, "(a) Tie-corrected Kendall's W", transform=axes[0].transAxes, ha="center", va="top", fontsize=9, fontweight="bold", clip_on=False)
    axes[1].text(0.5, -0.16, "(b) Selection utility (mean test MCC)", transform=axes[1].transAxes, ha="center", va="top", fontsize=9, fontweight="bold", clip_on=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.13, hspace=0.82)
    fig.savefig(paths["fig_cd"], dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    utility = pd.read_csv(paths["utility"])
    selected_means = utility.groupby("metric").test_mcc.mean()
    calibration_means = pd.read_csv(paths["calibration_summary"]).set_index("metric")
    table_lines = [
        "\\begin{center}", "\\refstepcounter{table}\\label{tab:main}",
        "\\parbox{\\linewidth}{\\small\\textbf{Table~\\thetable:} The duplicate-safe six-dataset protocol over twelve repeats reports tie-corrected ranking stability from midranks, selected-model test MCC, and calibration-shift agreement.}",
        "\\smallskip", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & \\multicolumn{2}{c}{Stability (Kendall's $W$)} & Selection & Calibration\\\\",
        "\\cmidrule(lr){2-3}", "Metric & Benchmark & Real & Test MCC & Spearman\\\\", "\\midrule",
    ]
    for metric_key, label in [
        ("SoftMCC", "SoftMCC"), ("Brier_neg", "Brier"), ("MCC_05", "MCC@0.5"),
        ("AUROC", "AUROC"), ("AUPRC", "AUPRC"), ("F1_best", "$F_1$@best"),
        ("MCC_best", "MCC@best"),
    ]:
        pretty = H.PRETTY[metric_key]
        b_value, r_value = float(benchmark[pretty]), float(real[pretty])
        table_lines.append(
            f"{label:<13s} & {b_value:.3f} & {r_value:.3f} & {selected_means[metric_key]:.3f} & {calibration_means.loc[pretty, 'mean_spearman_vs_T1']:.3f}\\\\"
        )
    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{center}", ""])
    paths["table"].write_text("\n".join(table_lines), encoding="utf-8", newline="\n")


def write_environment(paths: dict[str, Path], run_id: str) -> None:
    environment = {
        "run_id": run_id,
        "run_family": "harden_dupsafe_tieaware",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__, "scikit_learn": sklearn_version,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "protocol": {
            "seed_base": H.SEED, "repeats": H.N_REPEATS, "temperatures": H.TEMPS,
            "candidate_pool": ["logreg_C0.1", "logreg_C1.0", "logreg_C10.0", "hgb_d3", "hgb_d6"],
            "calibration": "CalibratedClassifierCV(method='isotonic', cv=2), training fold only",
            "split": "exact feature-label grouped stratified 75/25 then 70/30 train/validation within trainval",
            "selection_tie_break": "fixed candidate declaration order for exact ties",
            "stability_ranking": "average ranks (midranks) from raw candidate scores",
            "kendalls_w": "standard tie-corrected denominator; undefined if every repeat is fully tied",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_seed_base": PERMUTATION_SEED_BASE,
            "permutation_replicates": N_PERMUTATIONS,
        },
    }
    paths["environment"].write_text(json.dumps(environment, indent=2), encoding="utf-8")


def verify_outputs(paths: dict[str, Path], transcript: Transcript) -> None:
    utility = pd.read_csv(paths["utility"])
    rankings = pd.read_csv(paths["rankings"])
    candidates = pd.read_csv(paths["candidate_scores"])
    calibration = pd.read_csv(paths["calibration"])
    splits = pd.read_csv(paths["split_manifest"])
    expected_blocks = len(H.dataset_specs()) * H.N_REPEATS
    checks = {
        "utility_rows": len(utility) == expected_blocks * len(H.METRICS),
        "ranking_rows": len(rankings) == expected_blocks * len(H.METRICS),
        "candidate_score_rows": len(candidates) == expected_blocks * 5,
        "calibration_rows": len(calibration) == expected_blocks * len(H.METRICS) * len(H.TEMPS),
        "split_rows": len(splits) == expected_blocks,
        "train_validation_group_overlap_zero": int(splits.train_validation_group_overlap.sum()) == 0,
        "train_test_group_overlap_zero": int(splits.train_test_group_overlap.sum()) == 0,
        "validation_test_group_overlap_zero": int(splits.validation_test_group_overlap.sum()) == 0,
        "prediction_bundle_exists": paths["prediction_bundle"].exists(),
        "kappa_distribution_rows": len(pd.read_csv(paths["kappa_perm_w"])) == N_PERMUTATIONS * 9,
    }
    for name, passed in checks.items():
        transcript.log(f"VERIFY {name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError(f"Output verification failed: {checks}")


def write_artifact_manifest(run_root: Path, manifest_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "last_write_time_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "last_write_time_utc"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_run_root(run_root: Path, paths: dict[str, Path], force: bool) -> None:
    existing_outputs = [path for key, path in paths.items() if key != "transcript" and path.exists()]
    if existing_outputs and not force:
        raise SystemExit("Refusing to overwrite existing run outputs; use --force only for this dated run")
    if force:
        for path in existing_outputs:
            path.unlink()
    for folder in (run_root / "evidence", run_root / "figures", run_root / "tables", run_root / "code"):
        folder.mkdir(parents=True, exist_ok=True)


def run_pipeline(run_id: str, force: bool) -> int:
    run_root = ROOT / "experiments" / run_id
    paths = output_paths(run_root)
    prepare_run_root(run_root, paths, force)
    transcript = Transcript(paths["transcript"])
    started = time.time()
    exit_code = 1
    try:
        self_test()
        transcript.log("Tie-aware duplicate-safe harden/control pipeline started")
        transcript.log(f"Run root: {run_root}")
        transcript.log(f"Python: {sys.version.split()[0]} on {platform.platform()}")
        write_environment(paths, run_id)
        run_raw(paths, transcript)
        stability_ranks, selection_ranks, cd = analyze(paths, transcript)
        run_kappa_controls(paths, transcript)
        reconcile_with_ordinal(paths, transcript)
        make_figures_and_table(paths, stability_ranks, selection_ranks, cd)
        verify_outputs(paths, transcript)
        write_artifact_manifest(run_root, paths["artifact_manifest"])
        exit_code = 0
        return 0
    except Exception:
        transcript.log("ERROR:\n" + traceback.format_exc())
        return 1
    finally:
        elapsed = time.time() - started
        transcript.log(f"Pipeline finished exit_code={exit_code} elapsed_seconds={elapsed:.2f}")
        transcript.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_only:
        self_test()
        print("SELF-TEST PASS")
        return 0
    return run_pipeline(args.run_id, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
