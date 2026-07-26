"""Aggregation, statistics, reproduction gate, and locked decision artifact
for the dataset-expansion run (MCL-20260724-dataset-expansion-01).

Reads ONLY the immutable unit shards produced by ``expansion_runner.py`` and
rebuilds the canonical tie-aware evidence family with the exact schemas of
``harden_dupsafe_tieaware.py``, then reuses that module's dataset-agnostic
``analyze`` and ``run_kappa_controls`` unchanged. Figures for the manuscript
are deliberately NOT produced here; figure design belongs to the (separately
approved) integration step, and the tieaware figure code hardcodes the
original six-dataset benchmark/real grouping.

Decision rules were locked before any expansion result existed (see
METHODOLOGY_CHANGE_LEDGER). This script persists every criterion value,
threshold, and discriminator statistic (durability contract section 6.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

import harden_dupsafe_full as H
import harden_dupsafe_tieaware as T

SCORE_ATOL_PASS = 1e-7
SCORE_ATOL_WARN = 1e-4


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Load and merge unit shards
# ---------------------------------------------------------------------------

def load_units(run_root: Path) -> tuple[list[dict], list[str], int]:
    manifest = json.loads((run_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    datasets: list[str] = manifest["planned_datasets"]
    repeats: int = int(manifest["repeats"])
    payloads: list[dict] = []
    missing: list[str] = []
    for key in datasets:
        for repeat in range(repeats):
            unit_id = f"{key}__r{repeat:02d}"
            json_path = run_root / "units" / f"{unit_id}.json"
            if not json_path.exists():
                missing.append(unit_id)
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            npz_path = run_root / "units" / f"{unit_id}.npz"
            if sha256_file(npz_path) != payload["npz_sha256"]:
                raise SystemExit(f"npz hash mismatch for {unit_id}; aborting")
            payloads.append(payload)
    if missing:
        raise SystemExit(
            f"Aggregation refused: {len(missing)} planned units missing "
            f"(partial promotion prohibited): {missing[:8]}..."
        )
    return payloads, datasets, repeats


def merge_units(run_root: Path, payloads: list[dict], paths: dict[str, Path]) -> None:
    candidate_rows: list[dict] = []
    ranking_rows: list[dict] = []
    utility_rows: list[dict] = []
    calibration_rows: list[dict] = []
    split_rows: list[dict] = []
    dataset_rows: dict[str, dict] = {}

    block_meta: list[tuple[str, str, int, int]] = []
    labels_parts: list[np.ndarray] = []
    probs_parts: list[np.ndarray] = []
    test_mcc_parts: list[np.ndarray] = []
    candidate_order: list[str] | None = None

    for payload in payloads:
        key = payload["dataset_key"]
        if key in dataset_rows:
            if dataset_rows[key] != payload["dataset_row"]:
                raise SystemExit(f"dataset_row mismatch across repeats for {key}")
        else:
            dataset_rows[key] = payload["dataset_row"]
        candidate_rows.extend(payload["candidate_rows"])
        ranking_rows.extend(payload["ranking_rows"])
        utility_rows.extend(payload["utility_rows"])
        calibration_rows.extend(payload["calibration_rows"])
        split_rows.append(payload["split_row"])

        bundle = np.load(run_root / "units" / f"{payload['unit_id']}.npz",
                         allow_pickle=False)
        order = [str(item) for item in bundle["candidate_order"]]
        if candidate_order is None:
            candidate_order = order
        elif candidate_order != order:
            raise SystemExit(f"candidate_order mismatch in {payload['unit_id']}")
        block_meta.append(
            (payload["dataset_key"], payload["dataset"],
             int(payload["repeat"]), int(payload["seed"]))
        )
        labels_parts.append(bundle["y_validation"].astype(np.int8))
        probs_parts.append(bundle["p_validation"].astype(np.float64))
        test_mcc_parts.append(bundle["candidate_test_mcc"].astype(np.float64))

    T.save_csv(utility_rows, paths["utility"])
    T.save_csv(ranking_rows, paths["rankings"])
    T.save_csv(candidate_rows, paths["candidate_scores"])
    T.save_csv(calibration_rows, paths["calibration"])
    T.save_csv(split_rows, paths["split_manifest"])
    T.save_csv(list(dataset_rows.values()), paths["dataset_manifest"])

    offsets = [0]
    for labels in labels_parts:
        offsets.append(offsets[-1] + len(labels))
    paths["prediction_bundle"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["prediction_bundle"],
        dataset_key=np.asarray([row[0] for row in block_meta]),
        dataset=np.asarray([row[1] for row in block_meta]),
        repeat=np.asarray([row[2] for row in block_meta], dtype=np.int16),
        seed=np.asarray([row[3] for row in block_meta], dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_order=np.asarray(candidate_order),
        y_validation=np.concatenate(labels_parts).astype(np.int8),
        p_validation=np.vstack(probs_parts).astype(np.float64),
        candidate_test_mcc=np.vstack(test_mcc_parts).astype(np.float64),
    )


def verify_merged(paths: dict[str, Path], n_datasets: int, repeats: int,
                  transcript: T.Transcript) -> None:
    expected_blocks = n_datasets * repeats
    utility = pd.read_csv(paths["utility"])
    rankings = pd.read_csv(paths["rankings"])
    candidates = pd.read_csv(paths["candidate_scores"])
    calibration = pd.read_csv(paths["calibration"])
    splits = pd.read_csv(paths["split_manifest"])
    checks = {
        "utility_rows": len(utility) == expected_blocks * len(H.METRICS),
        "ranking_rows": len(rankings) == expected_blocks * len(H.METRICS),
        "candidate_score_rows": len(candidates) == expected_blocks * 5,
        "calibration_rows": len(calibration)
        == expected_blocks * len(H.METRICS) * len(H.TEMPS),
        "split_rows": len(splits) == expected_blocks,
        "train_validation_group_overlap_zero": int(splits.train_validation_group_overlap.sum()) == 0,
        "train_test_group_overlap_zero": int(splits.train_test_group_overlap.sum()) == 0,
        "validation_test_group_overlap_zero": int(splits.validation_test_group_overlap.sum()) == 0,
        "prediction_bundle_exists": paths["prediction_bundle"].exists(),
        "finite_scores": bool(
            np.all(np.isfinite(candidates[[c for c in candidates.columns if c.startswith("score_")]].to_numpy()))
        ),
    }
    for name, passed in checks.items():
        transcript.log(f"VERIFY {name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError(f"Merged-output verification failed: {checks}")


# ---------------------------------------------------------------------------
# Reproduction gate against the canonical 2026-07-18 evidence
# ---------------------------------------------------------------------------

def reproduction_check(paths: dict[str, Path], reference_dir: Path,
                       out_path: Path, transcript: T.Transcript,
                       blocking: bool = True) -> dict:
    reference = {
        "rankings": reference_dir / "harden_dupsafe_tieaware_rankings.csv",
        "candidate_scores": reference_dir / "harden_dupsafe_tieaware_candidate_scores.csv",
        "utility": reference_dir / "harden_dupsafe_tieaware_utility.csv",
        "split_manifest": reference_dir / "harden_dupsafe_tieaware_split_manifest.csv",
    }
    for name, path in reference.items():
        if not path.exists():
            raise SystemExit(f"Reference evidence missing: {path}")

    report: dict = {"reference_dir": str(reference_dir), "checked_at": now_iso()}
    canonical_keys = [spec.key for spec in H.dataset_specs()]

    new_splits = pd.read_csv(paths["split_manifest"])
    ref_splits = pd.read_csv(reference["split_manifest"])
    join = ["dataset_key", "repeat"]
    merged = ref_splits.merge(new_splits, on=join, suffixes=("_ref", "_new"))
    merged = merged[merged.dataset_key.isin(canonical_keys)]
    hash_cols = ["train_index_sha256", "validation_index_sha256", "test_index_sha256"]
    split_exact = all(
        bool((merged[f"{col}_ref"] == merged[f"{col}_new"]).all()) for col in hash_cols
    )
    report["split_blocks_compared"] = int(len(merged))
    report["split_index_hashes_exact"] = split_exact

    new_rank = pd.read_csv(paths["rankings"])
    ref_rank = pd.read_csv(reference["rankings"])
    join = ["dataset_key", "repeat", "metric"]
    merged = ref_rank.merge(new_rank, on=join, suffixes=("_ref", "_new"))
    merged = merged[merged.dataset_key.isin(canonical_keys)]
    order_equal = bool(
        (merged.deterministic_selection_order_ref == merged.deterministic_selection_order_new).all()
    )
    selected_equal = bool((merged.selected_candidate_ref == merged.selected_candidate_new).all())
    midrank_max_diff = 0.0
    for _, row in merged.iterrows():
        a = np.asarray(json.loads(row.midranks_json_ref), dtype=float)
        b = np.asarray(json.loads(row.midranks_json_new), dtype=float)
        midrank_max_diff = max(midrank_max_diff, float(np.max(np.abs(a - b))))
    report["ranking_rows_compared"] = int(len(merged))
    report["selection_orders_exact"] = order_equal
    report["selected_candidates_exact"] = selected_equal
    report["midrank_max_abs_diff"] = midrank_max_diff

    new_scores = pd.read_csv(paths["candidate_scores"])
    ref_scores = pd.read_csv(reference["candidate_scores"])
    join = ["dataset_key", "repeat", "candidate"]
    merged = ref_scores.merge(new_scores, on=join, suffixes=("_ref", "_new"))
    merged = merged[merged.dataset_key.isin(canonical_keys)]
    score_cols = [f"score_{metric}" for metric in H.METRICS] + [
        "test_mcc_at_validation_mccbest_threshold"
    ]
    score_max_diff = 0.0
    for col in score_cols:
        diff = float(np.max(np.abs(merged[f"{col}_ref"] - merged[f"{col}_new"])))
        report[f"max_abs_diff__{col}"] = diff
        score_max_diff = max(score_max_diff, diff)
    report["candidate_rows_compared"] = int(len(merged))
    report["score_max_abs_diff"] = score_max_diff

    new_utility = pd.read_csv(paths["utility"])
    ref_utility = pd.read_csv(reference["utility"])
    join = ["dataset_key", "repeat", "metric"]
    merged = ref_utility.merge(new_utility, on=join, suffixes=("_ref", "_new"))
    merged = merged[merged.dataset_key.isin(canonical_keys)]
    utility_max_diff = float(np.max(np.abs(merged.test_mcc_ref - merged.test_mcc_new)))
    utility_selected_equal = bool(
        (merged.selected_candidate_ref == merged.selected_candidate_new).all()
    )
    report["utility_rows_compared"] = int(len(merged))
    report["utility_test_mcc_max_abs_diff"] = utility_max_diff
    report["utility_selected_candidates_exact"] = utility_selected_equal

    structure_ok = split_exact and order_equal and selected_equal and utility_selected_equal
    numeric = max(score_max_diff, utility_max_diff, midrank_max_diff)
    if structure_ok and numeric <= SCORE_ATOL_PASS:
        status = "PASS_exact_within_1e-7"
    elif structure_ok and numeric <= SCORE_ATOL_WARN:
        status = "PASS_with_numeric_warning_within_1e-4"
    else:
        status = "FAIL"
    report["numeric_max_abs_diff"] = numeric
    report["status"] = status
    report["mode"] = "blocking_gate" if blocking else "informational_report"
    atomic_write_json(out_path, report)
    transcript.log(
        f"REPRODUCTION {'GATE' if blocking else 'REPORT'}: {status} "
        f"(numeric max abs diff {numeric:.3e})"
    )
    if status == "FAIL" and blocking:
        raise SystemExit(
            "Reproduction gate FAILED: expansion environment does not reproduce "
            "the canonical six-dataset evidence; integration is blocked "
            "(see reproduction report)."
        )
    return report


# ---------------------------------------------------------------------------
# Locked decision artifact (outcome A/B/C)
# ---------------------------------------------------------------------------

def decision_artifact(paths: dict[str, Path], run_root: Path,
                      transcript: T.Transcript) -> dict:
    stability = pd.read_csv(paths["stability_summary"])
    datasets = list(pd.read_csv(paths["utility"]).dataset.unique())
    n_datasets = len(datasets)
    k = len(H.METRICS)
    cd = H.Q05[k] * float(np.sqrt(k * (k + 1) / (6.0 * n_datasets)))

    pivot = stability.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    pretty = [H.PRETTY[m] for m in H.METRICS]
    matrix = np.asarray([pivot.loc[p, datasets].to_numpy() for p in pretty], dtype=float)
    statistic, p_value = friedmanchisquare(*[matrix[i] for i in range(k)])
    rank_matrix = np.asarray(
        [pd.Series(-matrix[:, col]).rank(method="average").to_numpy()
         for col in range(n_datasets)]
    ).T
    mean_ranks = rank_matrix.mean(axis=1)
    mean_w = matrix.mean(axis=1)
    soft_idx = pretty.index("SoftMCC")

    criteria: list[dict] = []

    def criterion(cid: str, value, threshold, passed: bool, contributes: str,
                  note: str = "") -> None:
        criteria.append(
            {"criterion_id": cid, "value": value, "threshold": threshold,
             "pass": bool(passed), "contributes_to": contributes, "note": note}
        )

    criterion("C1_stability_friedman_p", float(p_value), "< 0.05",
              bool(p_value < 0.05), "A", f"chi2={statistic:.3f}, N={n_datasets}, k={k}")
    highest_mean_w = bool(np.argmax(mean_w) == soft_idx)
    criterion("C2_softmcc_highest_mean_W",
              {p: round(float(w), 6) for p, w in zip(pretty, mean_w)},
              "SoftMCC mean W strictly highest", highest_mean_w, "B_vs_C")
    best_mean_rank = bool(np.argmin(mean_ranks) == soft_idx)
    criterion("C3_softmcc_best_stability_mean_rank",
              {p: round(float(r), 3) for p, r in zip(pretty, mean_ranks)},
              "SoftMCC mean rank lowest", best_mean_rank, "A")
    significant_pairs = []
    for i, name in enumerate(pretty):
        if i == soft_idx:
            continue
        gap = float(mean_ranks[i] - mean_ranks[soft_idx])
        if gap >= cd:
            significant_pairs.append({"baseline": name, "rank_gap": round(gap, 3)})
        criterion(f"C4_nemenyi_gap__{name}", round(gap, 3), f">= CD {cd:.3f}",
                  gap >= cd, "A")
    any_pair = bool(significant_pairs)

    outcome = "C_NEGATIVE"
    if bool(p_value < 0.05) and best_mean_rank and any_pair:
        outcome = "A_SUPPORTED"
    elif highest_mean_w:
        outcome = "B_PROTOCOL_ONLY"

    selection = pd.read_csv(paths["selection_summary"])
    utility_guard = []
    for _, row in selection.iterrows():
        worse = bool(row.mean_diff < 0 and row.wilcoxon_p < 0.05)
        utility_guard.append(
            {"baseline": row.baseline, "mean_diff": float(row.mean_diff),
             "wilcoxon_p": float(row.wilcoxon_p),
             "significantly_worse_than_baseline": worse}
        )
    any_worse = any(item["significantly_worse_than_baseline"] for item in utility_guard)
    criterion("U1_selected_utility_not_significantly_worse", utility_guard,
              "no baseline with mean_diff<0 and p<0.05", not any_worse,
              "limitation_reporting")

    kappa_summary_path = paths["kappa_summary"]
    permutation = None
    if kappa_summary_path.exists():
        kappa = json.loads(kappa_summary_path.read_text(encoding="utf-8"))
        permutation = kappa.get("permutation_distribution", {}).get("all6")
        criterion("P1_label_permutation_exceedance",
                  permutation, "observed W above permuted distribution",
                  bool(permutation and permutation.get("exceedance_fraction", 1) < 0.05),
                  "mechanism_reporting",
                  "scope key 'all6' means all datasets in this run")

    summary = {
        "methodology_change_id": "MCL-20260724-dataset-expansion-01",
        "decided_at": now_iso(),
        "n_datasets": n_datasets,
        "repeats": int(pd.read_csv(paths["split_manifest"]).repeat.max()) + 1,
        "nemenyi_cd": cd,
        "friedman_chi2": float(statistic),
        "friedman_p": float(p_value),
        "mean_w_by_metric": {p: float(w) for p, w in zip(pretty, mean_w)},
        "stability_mean_ranks": {p: float(r) for p, r in zip(pretty, mean_ranks)},
        "significant_nemenyi_pairs_vs_softmcc": significant_pairs,
        "outcome": outcome,
        "outcome_rule": (
            "A if Friedman p<0.05 AND SoftMCC best mean rank AND >=1 Nemenyi pair; "
            "B if SoftMCC highest mean W without A; else C"
        ),
        "utility_guard": utility_guard,
        "permutation_all_datasets": permutation,
        "input_hashes": {
            "stability_summary": sha256_file(paths["stability_summary"]),
            "selection_summary": sha256_file(paths["selection_summary"]),
            "utility": sha256_file(paths["utility"]),
        },
    }
    analysis_dir = run_root / "analysis"
    T.save_csv(criteria, analysis_dir / "decision_criteria.csv")
    atomic_write_json(analysis_dir / "decision_summary.json", summary)
    transcript.log(
        f"DECISION: {outcome} (Friedman p={p_value:.4g}, CD={cd:.3f}, "
        f"significant pairs={len(significant_pairs)})"
    )
    return summary


def write_environment(paths: dict[str, Path], run_root: Path) -> None:
    try:
        from imblearn import __version__ as imblearn_version
    except Exception:  # noqa: BLE001
        imblearn_version = "not_installed"
    payload = {
        "run_family": "expansion_dupsafe_tieaware",
        "methodology_change_id": "MCL-20260724-dataset-expansion-01",
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "imbalanced_learn": imblearn_version,
        },
        "protocol_inherited_from": "harden_dupsafe_tieaware 2026-07-18 canonical run",
        "expansion_rule_manifest": (
            str(H.DATA / "expansion" / "expansion_rule_manifest.json")
        ),
    }
    manifest_path = H.DATA / "expansion" / "expansion_rule_manifest.json"
    if manifest_path.exists():
        payload["expansion_rule_manifest_sha256"] = sha256_file(manifest_path)
    paths["environment"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--reference-evidence",
        default=str(
            H.ROOT / "experiments"
            / "2026-07-18_codex_local_unknown_tieaware_harden_control" / "evidence"
        ),
    )
    parser.add_argument("--skip-kappa", action="store_true")
    parser.add_argument("--skip-reproduction", action="store_true")
    parser.add_argument(
        "--reproduction-mode", choices=["gate", "report", "skip"], default="gate",
        help=(
            "gate: FAIL blocks (original locked rule); report: run the comparison "
            "and persist differences without blocking (author-superseded rule of "
            "2026-07-24, ARM canonical environment); skip: do not compare"
        ),
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    paths = T.output_paths(run_root)
    transcript = T.Transcript(run_root / "aggregate.log")
    started = time.time()
    try:
        T.self_test()
        payloads, datasets, repeats = load_units(run_root)
        transcript.log(
            f"Aggregating {len(payloads)} units over {len(datasets)} datasets x {repeats} repeats"
        )
        merge_units(run_root, payloads, paths)
        verify_merged(paths, len(datasets), repeats, transcript)
        write_environment(paths, run_root)
        mode = "skip" if args.skip_reproduction else args.reproduction_mode
        if mode != "skip":
            reproduction_check(
                paths, Path(args.reference_evidence),
                run_root / "analysis" / "reproduction_report.json", transcript,
                blocking=(mode == "gate"),
            )
        stability_ranks, selection_ranks, cd = T.analyze(paths, transcript)
        if not args.skip_kappa:
            T.run_kappa_controls(paths, transcript)
        decision = decision_artifact(paths, run_root, transcript)
        T.write_artifact_manifest(run_root, run_root / "ARTIFACT_MANIFEST.csv")
        transcript.log(
            f"Aggregate complete in {time.time() - started:.1f}s; outcome {decision['outcome']}"
        )
        return 0
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        transcript.log("ERROR:\n" + traceback.format_exc())
        return 1
    finally:
        transcript.close()


if __name__ == "__main__":
    raise SystemExit(main())
