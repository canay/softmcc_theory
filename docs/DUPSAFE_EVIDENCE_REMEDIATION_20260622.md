Date/time: 2026-06-22 03:45 +03:00
Tool: Codex
Operation ID: REV-20260622-0345-CODEX-26 / VER-20260622-0345-CODEX-26

# Duplicate-Safe Evidence Remediation

## Scope

This run implements the approved Q1 audit actions A-001, A-002, A-003, A-004,
A-006, A-009, and A-010 in bounded form. It does not add citations, DOI
records, external datasets, p-values, or unverified claims. It preserves the
old `harden_*` evidence family and creates a new canonical candidate evidence
family with the prefix `harden_dupsafe_*`.

## Evidence

- New runner: `03_experiments/scripts/harden_dupsafe_full.py`.
- Canonical candidate raw outputs:
  - `03_experiments/results/harden_dupsafe_utility.csv`
  - `03_experiments/results/harden_dupsafe_rankings.csv`
  - `03_experiments/results/harden_dupsafe_calibration.csv`
  - `03_experiments/results/harden_dupsafe_calibration_diagnostics.csv`
- Reproducibility records:
  - `03_experiments/results/harden_dupsafe_transcript.txt`
  - `03_experiments/results/harden_dupsafe_split_manifest.csv`
  - `03_experiments/results/harden_dupsafe_dataset_manifest.csv`
  - `03_experiments/results/harden_dupsafe_environment.json`
  - `03_experiments/results/harden_dupsafe_artifact_manifest.csv`
- Summaries and manuscript assets:
  - `03_experiments/results/harden_dupsafe_summary_stability.csv`
  - `03_experiments/results/harden_dupsafe_summary_selection.csv`
  - `03_experiments/results/harden_dupsafe_summary_calibration.csv`
  - `03_experiments/results/harden_dupsafe_friedman.txt`
  - `03_experiments/results/harden_dupsafe_evidence_reconciliation.csv`
  - `04_manuscript/tables/table_main_results_dupsafe.tex`
  - `04_manuscript/figs/fig1_resampling_stability_dupsafe.png`
  - `04_manuscript/figs/fig2_calibration_sensitivity_dupsafe.png`
  - `04_manuscript/figs/fig3_critical_difference_dupsafe.png`
- Split-manifest verification: 72 blocks; sums of train-validation,
  train-test, and validation-test exact duplicate-group overlaps are all zero.
- Final duplicate-safe results after correcting the breast-cancer positive class
  to malignant:
  - SoftMCC stability: benchmark Kendall's W 0.711; real-suite Kendall's W
    0.764; all-six average 0.737.
  - Baseline all-six stability range: 0.495 to 0.628.
  - Stability Friedman: chi-square 7.433, p=0.2827, Nemenyi CD=3.678.
  - Selected-model test MCC range: 0.650 to 0.665.
  - Pairwise SoftMCC-vs-baseline selection differences: all BCa 95% intervals
    cross zero; all Wilcoxon p>=0.151; all |Cliff delta|<0.02.
  - Calibration-shift agreement: SoftMCC mean Spearman 0.838, BCa 95% CI
    [0.777, 0.881]; Brier 0.816 [0.760, 0.854]; rank/threshold metrics 1.000.
  - SoftMCC at severe temperature scaling T=3 on the real suite: mean Spearman
    0.675.
- Build and validation:
  - `pdflatex -> bibtex -> pdflatex x3` built
    `04_manuscript/build/softmcc_theory.pdf`, 13 pages, 707529 bytes.
  - Log scan found no fatal errors, undefined control sequences, undefined
    citations/references, or rerun warnings.
  - `tools/generate_manuscript_validation_mirror.py` refreshed
    `manuscript/softmcc_theory.tex`, `manuscript/references.bib`, and
    `manuscript/softmcc_theory.pdf`.
  - Abstract word count is 242 in canonical and mirror sources.
  - `check_repo_hygiene.py` PASS.
  - `validate_pipeline.py --allow-incomplete` PASS.
  - `validate_manuscript.py --allow-incomplete` PASS.
  - `validate_manuscript.py --quality-profile full_empirical --allow-incomplete`
    PASS.
  - `validate_manuscript.py --quality-profile q1 --allow-incomplete` PASS.
  - Rendered pages 1, 7, 8, 9, 10, and 11 were visually checked.

## Inference

- The previous P1 duplicate-leakage blocker is remediated for the new canonical
  candidate evidence family because exact feature-label duplicate groups no
  longer cross train, validation, or test partitions.
- The previous P1 transcript blocker is remediated for the new canonical
  candidate evidence family because the full runner records command, working
  directory, package versions, dataset hashes, split manifests, run outputs, and
  artifact hashes.
- The old `harden_*` evidence family should remain provenance only and should
  not be used for manuscript-facing numerical claims.
- The breast-cancer class orientation in the first duplicate-safe attempt was
  inconsistent with the manuscript's 37% positive-class statement. The corrected
  run uses malignant as positive (`1 - sklearn.target`) and supersedes the first
  duplicate-safe attempt.

## Interpretation

- The manuscript can no longer claim statistically significant SoftMCC stability
  superiority. The corrected claim is: SoftMCC has the highest observed ranking
  reproducibility in the duplicate-safe six-dataset protocol, while the
  dataset-level Friedman-Nemenyi comparison is not significant.
- The manuscript should not use "statistical equivalence" without a formal
  equivalence margin/TOST. The corrected claim is: selected-model test MCC is
  similar across selectors and no SoftMCC selected-utility advantage is detected.
- The empirical layer is now evidence-backed for a bounded draft, but not
  submission-ready. Remaining submission blockers include ORCID/funding and
  institutional metadata, dataset-license confirmation, target journal/template
  decision, AI-use disclosure decision, and public `softmcc_theory` repository
  release/commit details.
