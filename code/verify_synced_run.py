"""Independent local closure checks for the synced dataset-expansion run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


METRICS = ["SoftMCC", "MCC@best", "F1@best", "AUPRC", "AUROC", "Brier", "MCC@0.5"]
NEMENYI_Q05_K7 = 2.949


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    manifest = json.loads((run_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    final_status = json.loads(
        (run_root / "state" / "final_status.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (run_root / "analysis" / "decision_summary.json").read_text(encoding="utf-8")
    )

    datasets = list(manifest["planned_datasets"])
    repeats = int(manifest["repeats"])
    expected_units = {
        f"{dataset}__r{repeat:02d}" for dataset in datasets for repeat in range(repeats)
    }
    json_units = {path.stem for path in (run_root / "units").glob("*.json")}
    npz_units = {path.stem for path in (run_root / "units").glob("*.npz")}
    phase_units = {
        path.name.removesuffix(".phase.jsonl")
        for path in (run_root / "units").glob("*.phase.jsonl")
    }
    require(json_units == expected_units, "unit JSON set differs from manifest")
    require(npz_units == expected_units, "unit NPZ set differs from manifest")
    require(phase_units == expected_units, "unit phase-log set differs from manifest")

    for unit_id in sorted(expected_units):
        payload = json.loads(
            (run_root / "units" / f"{unit_id}.json").read_text(encoding="utf-8")
        )
        require(payload["unit_id"] == unit_id, f"unit_id mismatch: {unit_id}")
        require(
            sha256_file(run_root / "units" / f"{unit_id}.npz") == payload["npz_sha256"],
            f"NPZ hash mismatch: {unit_id}",
        )

    require(final_status["status"] == "completed", "final status is not completed")
    require(final_status["exit_code"] == 0, "compute exit code is nonzero")
    require(not final_status["failed_units"], "failed units are present")
    require(
        set(final_status["completed_units"]) == expected_units,
        "final completed-unit set differs from manifest",
    )

    stability_path = run_root / "evidence" / "harden_dupsafe_tieaware_summary_stability.csv"
    selection_path = run_root / "evidence" / "harden_dupsafe_tieaware_summary_selection.csv"
    utility_path = run_root / "evidence" / "harden_dupsafe_tieaware_utility.csv"
    input_paths = {
        "stability_summary": stability_path,
        "selection_summary": selection_path,
        "utility": utility_path,
    }
    require(
        {name: sha256_file(path) for name, path in input_paths.items()}
        == decision["input_hashes"],
        "decision input hashes do not match synced evidence",
    )

    stability = pd.read_csv(stability_path)
    utility = pd.read_csv(utility_path)
    dataset_labels = list(utility["dataset"].unique())
    require(len(dataset_labels) == len(datasets) == 18, "dataset count is not 18")
    pivot = stability.pivot_table(index="metric", columns="dataset", values="kendalls_w")
    matrix = np.asarray(
        [pivot.loc[metric, dataset_labels].to_numpy() for metric in METRICS],
        dtype=float,
    )
    require(np.isfinite(matrix).all(), "non-finite stability value")
    friedman_chi2, friedman_p = friedmanchisquare(*matrix)
    rank_matrix = np.asarray(
        [
            pd.Series(-matrix[:, column]).rank(method="average").to_numpy()
            for column in range(len(dataset_labels))
        ]
    ).T
    mean_ranks = rank_matrix.mean(axis=1)
    mean_w = matrix.mean(axis=1)
    cd = NEMENYI_Q05_K7 * np.sqrt(
        len(METRICS) * (len(METRICS) + 1) / (6.0 * len(dataset_labels))
    )
    soft_index = METRICS.index("SoftMCC")
    significant_pairs = [
        METRICS[index]
        for index in range(len(METRICS))
        if index != soft_index and mean_ranks[index] - mean_ranks[soft_index] >= cd
    ]
    recomputed_outcome = (
        "A_SUPPORTED"
        if friedman_p < 0.05
        and int(np.argmin(mean_ranks)) == soft_index
        and significant_pairs
        else "B_PROTOCOL_ONLY"
        if int(np.argmax(mean_w)) == soft_index
        else "C_NEGATIVE"
    )
    require(recomputed_outcome == decision["outcome"], "A/B/C outcome mismatch")
    require(np.isclose(friedman_chi2, decision["friedman_chi2"]), "Friedman chi2 mismatch")
    require(np.isclose(friedman_p, decision["friedman_p"]), "Friedman p mismatch")
    require(np.isclose(cd, decision["nemenyi_cd"]), "Nemenyi CD mismatch")

    selection = pd.read_csv(selection_path)
    utility_limitations = list(
        selection.loc[
            (selection["mean_diff"] < 0) & (selection["wilcoxon_p"] < 0.05),
            "baseline",
        ]
    )
    summary_limitations = [
        item["baseline"]
        for item in decision["utility_guard"]
        if item["significantly_worse_than_baseline"]
    ]
    require(
        utility_limitations == summary_limitations,
        "utility limitation flags differ from selection summary",
    )

    kappa = json.loads(
        (run_root / "evidence" / "kappa_tieaware_summary.json").read_text(encoding="utf-8")
    )
    permutation = kappa["permutation_distribution"]["all6"]
    require(
        np.isclose(
            permutation["exceedance_fraction"],
            decision["permutation_all_datasets"]["exceedance_fraction"],
        ),
        "permutation summary mismatch",
    )
    require(permutation["exceedance_fraction"] < 0.05, "permutation check failed")

    remote_manifest = pd.read_csv(run_root / "ARTIFACT_MANIFEST_REMOTE.csv")
    selected_prefixes = (
        "evidence/",
        "analysis/",
        "state/",
        "units/",
        "logs/",
        "reference_evidence/",
        "smoke/",
    )
    selected = remote_manifest[
        remote_manifest["relative_path"].str.startswith(selected_prefixes)
    ]
    checked = 0
    for row in selected.itertuples(index=False):
        path = run_root / row.relative_path
        require(path.exists(), f"synced artifact missing: {row.relative_path}")
        require(path.stat().st_size == row.bytes, f"size mismatch: {row.relative_path}")
        require(sha256_file(path) == row.sha256, f"hash mismatch: {row.relative_path}")
        checked += 1

    report = {
        "verification_status": "PASS",
        "verified_at": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(),
        "run_id": manifest["run_id"],
        "methodology_change_id": manifest["methodology_change_id"],
        "planned_datasets": len(datasets),
        "repeats": repeats,
        "verified_units": len(expected_units),
        "failed_units": 0,
        "synced_remote_manifest_rows_verified": checked,
        "recomputed_outcome": recomputed_outcome,
        "friedman_chi2": float(friedman_chi2),
        "friedman_p": float(friedman_p),
        "nemenyi_cd": float(cd),
        "softmcc_mean_w": float(mean_w[soft_index]),
        "softmcc_stability_mean_rank": float(mean_ranks[soft_index]),
        "significant_nemenyi_pairs_vs_softmcc": significant_pairs,
        "utility_limitation_baselines": utility_limitations,
        "permutation_exceedance_fraction": float(permutation["exceedance_fraction"]),
        "decision_summary_sha256": sha256_file(
            run_root / "analysis" / "decision_summary.json"
        ),
        "remote_artifact_manifest_sha256": sha256_file(
            run_root / "ARTIFACT_MANIFEST_REMOTE.csv"
        ),
        "sync_bundle_sha256": sha256_file(run_root / "sync_bundle_20260725.tar.gz"),
        "note": (
            "The remote manifest predates the aggregate transcript's terminal line; "
            "directory-scoped synced artifacts were checked against it, while the "
            "post-aggregate logs and transfer bundle are covered by the local manifest."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
