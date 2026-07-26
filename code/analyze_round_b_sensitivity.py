"""Derive Round B sensitivity results from the packaged ARM18 evidence.

The script reads three hash-checked files from the public package. It writes
only to ``results/round_b_sensitivity``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


OPERATION_ID = "REV-20260726-CODEX-ROUND-B-REMEDIATION"
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "round_b_sensitivity"
FROZEN_RUN = PROJECT_ROOT
RESULTS = ANALYSIS_ROOT

STABILITY_REL = Path(
    "evidence/harden_dupsafe_tieaware_summary_stability.csv"
)
SELECTION_REL = Path(
    "evidence/harden_dupsafe_tieaware_summary_selection.csv"
)
EFFECT_REL = Path("evidence/harden_dupsafe_tieaware_effect_sizes.csv")

METRICS = [
    "SoftMCC",
    "MCC@best",
    "F1@best",
    "AUPRC",
    "AUROC",
    "Brier",
    "MCC@0.5",
]
Q_ALPHA_005 = 2.949
ALPHA = 0.05

# Families are fixed from documented source overlap before this sensitivity is
# evaluated. Singleton settings retain their original identifier.
SOURCE_FAMILIES = {
    "breast_cancer(37%)": "breast_cancer",
    "synth(5%)": "synthetic_generator",
    "synth(1%)": "synthetic_generator",
    "creditcard(1%)": "creditcard_source",
    "creditcard(0.5%)": "creditcard_source",
    "iotid20(6.4%)": "iotid20",
    "abalone(9.4%)": "abalone",
    "car_eval_34(7.8%)": "car_evaluation_source",
    "car_eval_4(3.8%)": "car_evaluation_source",
    "coil_2000(6.0%)": "coil_2000",
    "mammography(2.3%)": "mammography",
    "ozone_level(2.9%)": "ozone_level",
    "protein_homo(0.9%)": "protein_homo",
    "sick_euthyroid(9.3%)": "sick_euthyroid",
    "thyroid_sick(6.1%)": "thyroid_sick",
    "us_crime(7.5%)": "us_crime",
    "wine_quality(3.7%)": "wine_quality",
    "yeast_ml8(7.4%)": "yeast_ml8",
}
CONSERVATIVE_SOURCE_FAMILIES = {
    **SOURCE_FAMILIES,
    "sick_euthyroid(9.3%)": "thyroid_disease_collection",
    "thyroid_sick(6.1%)": "thyroid_disease_collection",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_input(relative_path: Path) -> tuple[Path, str]:
    manifest_path = FROZEN_RUN / "PACKAGE_MANIFEST.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {
            Path(row["path"]): row["sha256"].lower()
            for row in csv.DictReader(handle)
        }
    packaged_relative = Path("results") / relative_path.name
    if packaged_relative not in manifest:
        raise RuntimeError(
            f"Package manifest lacks {packaged_relative.as_posix()}"
        )
    path = FROZEN_RUN / packaged_relative
    actual = sha256_file(path)
    expected = manifest[packaged_relative]
    if actual != expected:
        raise RuntimeError(
            f"Packaged input hash mismatch for {packaged_relative.as_posix()}: "
            f"expected={expected} actual={actual}"
        )
    return path, actual


def holm_adjust(raw_p_values: pd.Series) -> pd.Series:
    ordered = raw_p_values.sort_values(kind="mergesort")
    adjusted: dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for order_index, (row_index, raw_p) in enumerate(ordered.items()):
        candidate = min(1.0, (count - order_index) * float(raw_p))
        running = max(running, candidate)
        adjusted[int(row_index)] = running
    return pd.Series(adjusted).reindex(raw_p_values.index)


def analyze_stability_scheme(
    stability: pd.DataFrame,
    family_mapping: dict[str, str],
    scheme: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    expanded = stability.copy()
    expanded["source_family"] = expanded["dataset"].map(family_mapping)
    family_values = (
        expanded.groupby(["source_family", "metric"], as_index=False)
        .agg(
            family_mean_kendalls_w=("kendalls_w", "mean"),
            setting_count=("dataset", "nunique"),
        )
        .sort_values(["source_family", "metric"], kind="mergesort")
    )
    family_values.insert(0, "scheme", scheme)

    family_pivot = family_values.pivot(
        index="source_family",
        columns="metric",
        values="family_mean_kendalls_w",
    ).loc[:, METRICS]
    expected_family_count = len(set(family_mapping.values()))
    if family_pivot.shape != (expected_family_count, 7):
        raise RuntimeError(
            f"Expected a {expected_family_count} by 7 family matrix for "
            f"{scheme}, got {family_pivot.shape}"
        )

    family_rank_matrix = pd.DataFrame(
        np.vstack(
            [
                rankdata(-row.to_numpy(dtype=float), method="average")
                for _, row in family_pivot.iterrows()
            ]
        ),
        index=family_pivot.index,
        columns=METRICS,
    )
    mean_ranks = family_rank_matrix.mean(axis=0)
    n_families = len(family_pivot)
    k_metrics = len(METRICS)
    cd = float(
        Q_ALPHA_005
        * math.sqrt(k_metrics * (k_metrics + 1) / (6.0 * n_families))
    )
    friedman = friedmanchisquare(
        *[family_pivot[metric].to_numpy(dtype=float) for metric in METRICS]
    )

    soft_rank = float(mean_ranks["SoftMCC"])
    rank_rows = []
    for metric in METRICS:
        gap = float(mean_ranks[metric] - soft_rank)
        rank_rows.append(
            {
                "scheme": scheme,
                "metric": metric,
                "mean_rank": float(mean_ranks[metric]),
                "rank_gap_vs_softmcc": gap,
                "nemenyi_separated_from_softmcc": bool(
                    metric != "SoftMCC" and abs(gap) > cd
                ),
            }
        )
    rank_frame = pd.DataFrame(rank_rows).sort_values(
        "mean_rank", kind="mergesort"
    )
    separated = rank_frame.loc[
        rank_frame["nemenyi_separated_from_softmcc"], "metric"
    ].tolist()
    result = {
        "aggregation_rule": (
            "Mean Kendall W within each source family for each metric, "
            "followed by within-family midranks"
        ),
        "n_settings": int(stability["dataset"].nunique()),
        "n_source_families": n_families,
        "n_metrics": k_metrics,
        "friedman_chi2": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "nemenyi_cd_0_05": cd,
        "softmcc_mean_rank": soft_rank,
        "separated_from_softmcc": separated,
        "mean_ranks": {
            metric: float(mean_ranks[metric]) for metric in METRICS
        },
    }
    return result, family_values, rank_frame


def stability_analysis(stability: pd.DataFrame) -> dict[str, object]:
    found_datasets = set(stability["dataset"])
    if found_datasets != set(SOURCE_FAMILIES):
        missing = sorted(set(SOURCE_FAMILIES) - found_datasets)
        extra = sorted(found_datasets - set(SOURCE_FAMILIES))
        raise RuntimeError(f"Dataset mapping mismatch: missing={missing} extra={extra}")
    if set(stability["metric"]) != set(METRICS):
        raise RuntimeError("Metric set differs from the locked seven metrics")

    definition_rows = []
    for scheme, mapping in {
        "primary_18_settings": {setting: setting for setting in SOURCE_FAMILIES},
        "documented_overlap_15_families": SOURCE_FAMILIES,
        "conservative_source_collection_14_families": (
            CONSERVATIVE_SOURCE_FAMILIES
        ),
    }.items():
        definition_rows.extend(
            {
                "scheme": scheme,
                "setting": setting,
                "source_family": family,
            }
            for setting, family in mapping.items()
        )
    family_definition = pd.DataFrame(definition_rows).sort_values(
        ["scheme", "source_family", "setting"], kind="mergesort"
    )
    family_definition.to_csv(
        RESULTS / "source_family_definition.csv", index=False
    )

    scheme_results = {}
    value_frames = []
    rank_frames = []
    for scheme, mapping in {
        "primary_18_settings": {setting: setting for setting in SOURCE_FAMILIES},
        "documented_overlap_15_families": SOURCE_FAMILIES,
        "conservative_source_collection_14_families": (
            CONSERVATIVE_SOURCE_FAMILIES
        ),
    }.items():
        result, values, ranks = analyze_stability_scheme(
            stability, mapping, scheme
        )
        scheme_results[scheme] = result
        value_frames.append(values)
        rank_frames.append(ranks)
    pd.concat(value_frames, ignore_index=True).to_csv(
        RESULTS / "source_family_stability_values.csv", index=False
    )
    pd.concat(rank_frames, ignore_index=True).to_csv(
        RESULTS / "source_family_stability_mean_ranks.csv", index=False
    )
    return scheme_results


def selection_analysis(
    selection: pd.DataFrame, effects: pd.DataFrame
) -> dict[str, object]:
    if len(selection) != 6 or set(selection["baseline"]) != set(METRICS[1:]):
        raise RuntimeError("Expected six SoftMCC versus baseline comparisons")

    effect_totals = (
        effects.groupby("baseline", as_index=False)
        .agg(
            total_pairs=("n_pairs", "sum"),
            positive_pairs=("positive_pairs", "sum"),
            negative_pairs=("negative_pairs", "sum"),
            zero_pairs=("zero_pairs", "sum"),
        )
    )
    effect_totals["effective_nonzero_pairs"] = (
        effect_totals["total_pairs"] - effect_totals["zero_pairs"]
    )
    if not (effect_totals["total_pairs"] == 216).all():
        raise RuntimeError("Each paired comparison must contain 216 total blocks")

    output = selection[
        [
            "baseline",
            "mean_diff",
            "bca_lo",
            "bca_hi",
            "wilcoxon_p",
            "matched_pairs_rank_biserial_all_blocks",
        ]
    ].copy()
    output = output.merge(effect_totals, on="baseline", validate="one_to_one")
    output["holm_adjusted_p"] = holm_adjust(output["wilcoxon_p"])
    output["holm_significant_0_05"] = output["holm_adjusted_p"] < ALPHA
    output["raw_p_significant_0_05"] = output["wilcoxon_p"] < ALPHA
    output = output[
        [
            "baseline",
            "mean_diff",
            "bca_lo",
            "bca_hi",
            "wilcoxon_p",
            "holm_adjusted_p",
            "raw_p_significant_0_05",
            "holm_significant_0_05",
            "matched_pairs_rank_biserial_all_blocks",
            "total_pairs",
            "effective_nonzero_pairs",
            "zero_pairs",
            "positive_pairs",
            "negative_pairs",
        ]
    ]
    output.to_csv(RESULTS / "selection_inference_holm.csv", index=False)

    significant = output.loc[
        output["holm_significant_0_05"], "baseline"
    ].tolist()
    return {
        "family_definition": (
            "Six prespecified SoftMCC versus baseline paired Wilcoxon tests"
        ),
        "correction": "Holm familywise error control at alpha=0.05",
        "holm_significant_baselines": significant,
        "comparisons": output.to_dict(orient="records"),
    }


def write_readme(summary: dict[str, object]) -> None:
    stability = summary["source_family_stability"]
    selection = summary["selection_inference"]
    documented = stability["documented_overlap_15_families"]
    conservative = stability["conservative_source_collection_14_families"]
    documented_separated = (
        ", ".join(documented["separated_from_softmcc"]) or "none"
    )
    conservative_separated = (
        ", ".join(conservative["separated_from_softmcc"]) or "none"
    )
    holm_significant = ", ".join(selection["holm_significant_baselines"]) or "none"
    text = f"""# Round B source-family and multiplicity sensitivity

This deterministic analysis addresses audit issues RB-C-01, RB-C-03, RB-C-04,
and RB-C-05 without changing the frozen ARM18 evidence.

## Inputs and protection

The script verifies the SHA-256 values of the packaged stability, selection,
and effect-size summaries against `PACKAGE_MANIFEST.csv`. It writes only to
this derived-analysis folder.

## Source-family sensitivity

The documented-overlap analysis collapses the 18 settings into 15 families.
The two credit-card prevalence variants, two synthetic settings, and two
car-evaluation targets form three paired families. This analysis gives
Friedman `p={documented['friedman_p']:.6f}` and Nemenyi CD
{documented['nemenyi_cd_0_05']:.6f}; separation from SoftMCC is retained for
{documented_separated}.

A more conservative source-collection analysis also groups the two thyroid
tasks from the UCI Thyroid Disease collection. It therefore uses 14 families.
This analysis gives Friedman `p={conservative['friedman_p']:.6f}` and Nemenyi
CD {conservative['nemenyi_cd_0_05']:.6f}; separation from SoftMCC is retained
for {conservative_separated}. The manuscript uses this more conservative result
to bound the stability claim.

The thyroid grouping follows the imbalanced-learn dataset descriptions and the
UCI Thyroid Disease collection:

- https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html
- https://archive.ics.uci.edu/dataset/102/thyroid+disease

For both analyses, Kendall's W is averaged within a family for each metric.
Metrics are then midranked within each family before the Friedman and Nemenyi
calculations.

## Selection multiplicity and paired effects

Holm correction is applied as one family across the six prespecified paired
Wilcoxon comparisons. At familywise alpha 0.05, the retained comparison is
{holm_significant}. The output also reports total blocks, zero differences,
effective nonzero pairs, and matched-pairs rank-biserial correlations.

## Reproduction

Run:

```text
python code/analyze_round_b_sensitivity.py
```

The generated CSV and JSON files are listed in `ARTIFACT_MANIFEST.csv`.
"""
    (ANALYSIS_ROOT / "README.md").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(ANALYSIS_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ANALYSIS_ROOT).as_posix()
        if relative == "ARTIFACT_MANIFEST.csv" or "__pycache__" in path.parts:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (ANALYSIS_ROOT / "ARTIFACT_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    input_records = {}
    verified_paths = {}
    for label, relative in {
        "stability_summary": STABILITY_REL,
        "selection_summary": SELECTION_REL,
        "effect_sizes": EFFECT_REL,
    }.items():
        path, digest = verified_input(relative)
        verified_paths[label] = path
        input_records[label] = {
            "relative_path": relative.as_posix(),
            "sha256": digest,
        }
    input_records["input_verification"] = {
        "relative_path": "PACKAGE_MANIFEST.csv",
        "scope": "The three packaged input rows are SHA-256 verified",
    }
    (RESULTS / "input_hashes.json").write_text(
        json.dumps(input_records, indent=2) + "\n", encoding="utf-8"
    )

    stability = pd.read_csv(verified_paths["stability_summary"])
    selection = pd.read_csv(verified_paths["selection_summary"])
    effects = pd.read_csv(verified_paths["effect_sizes"])
    summary = {
        "operation_id": OPERATION_ID,
        "status": "verified_derived_analysis",
        "frozen_arm18_inputs_modified": False,
        "source_family_stability": stability_analysis(stability),
        "selection_inference": selection_analysis(selection, effects),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(summary)
    write_manifest()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
