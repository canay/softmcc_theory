# softmcc_theory

Repository: https://github.com/canay/softmcc_theory

Replication package for:

**SoftMCC: An MCC-Brier Calibration Bridge for Threshold-Free Model Selection
under Class Imbalance**

This package supports the manuscript's duplicate-safe, tie-aware 18-setting ARM
evidence. It does not claim that the general soft-count MCC construction is new.
SoftMCC is used here as a post-training evaluation and validation-ranking
framework around a probability-valued MCC core score.

## Contents

- `code/softmcc_scorer.py`: scikit-learn compatible SoftMCC scorer.
- `code/softmcc_eval.py`: metric definitions and helper functions.
- `code/harden_dupsafe_full.py`: shared duplicate-safe grouped training and
  metric implementation.
- `code/harden_dupsafe_tieaware.py`: canonical runner; archives every
  candidate score, uses midranks and standard tie-corrected Kendall's W, runs
  the 200-permutation kappa/rho controls, and regenerates the evidence family.
- `code/expansion_datasets.py`: applies the prespecified inclusion rule to the
  imbalanced-learn benchmark collection and records every include/exclude
  decision.
- `code/expansion_runner.py`: restartable 18-setting, 12-repeat unit runner.
- `code/expansion_aggregate.py`: rebuilds the evidence family and locked
  decision artifact from immutable unit shards.
- `code/verify_synced_run.py`: verifies a synchronized expansion run before
  aggregation.
- `code/analyze_round_b_sensitivity.py`: recomputes source-family stability
  sensitivity, Holm-adjusted inference, effective pair counts, and
  matched-pairs rank-biserial effects from the packaged ARM summaries.
- `code/verify_softmcc_identities.py`: numerical checks for Propositions 1--3
  and the population calibration identity, including a non-converse example.
- `code/duplicate_leakage_audit.py`: audit for exact duplicate rows and
  cross-split leakage under ordinary stratified splitting.
- `code/real_prep.py`: cache construction notes for the public real datasets.
- `results/harden_dupsafe_tieaware_*`: manuscript-facing ARM outputs, archived
  candidate scores, midranks, tie audit, effect sizes, summaries, split and
  dataset manifests.
- `results/kappa_tieaware_*`: 200 label-permutation results plus kappa-only and
  rho-only controls reconstructed from the archived prediction bundle.
- `results/arm18_*`: run manifest, decision criteria, decision summary,
  synchronization verification, reproduction report, and artifact manifest.
- `results/round_b_sensitivity/*`: derived 18-setting, 15-family, and
  conservative 14-family stability analyses plus Holm-adjusted paired results.
- `results/harden_dupsafe_*` without `tieaware`: prior ordinal evidence retained
  for provenance; it is not the manuscript-facing W estimator.
- `figures/*_dupsafe.png`: manuscript result figures generated from the
  duplicate-safe evidence family.
- Public Figure 1 maps to manuscript Figure 2, public Figure 2 maps to
  manuscript Figure 3, and public Figure 3 maps to manuscript Figure 4.
- `results/harden_dupsafe_tieaware_*`: tabular result sources used to construct
  the manuscript-facing summary table.
- `docs/TIEAWARE_EVIDENCE_REMEDIATION_20260718.md`: current evidence,
  interpretation, and provenance of the earlier six-dataset run.
- `docs/ARM18_EVIDENCE_INTEGRATION_20260725.md`: interpretation and scope of the
  manuscript-facing ARM expansion.
- `docs/ROUND_B_SENSITIVITY_20260726.md`: source-family, multiplicity, figure,
  and legacy-key interpretation for the final derived analysis.
- `docs/DUPSAFE_EVIDENCE_REMEDIATION_20260622.md`: prior remediation provenance.

## Data

Raw data are not redistributed in this package.

- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  supplies the two controlled prevalence variants.
- [IoTID20](https://sites.google.com/view/iot-network-intrusion-dataset/home)
  supplies the network-intrusion setting.
- [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
  is loaded through scikit-learn.
- Twelve additional settings come from the
  [Zenodo imbalanced benchmark collection](https://doi.org/10.5281/zenodo.61452)
  exposed by
  [`imblearn.datasets.fetch_datasets`](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html).
- The two synthetic settings are generated locally with
  [`sklearn.datasets.make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).
- The exact 18-setting inventory, row counts, feature counts, positive counts,
  prevalence values, and source labels are in
  `results/harden_dupsafe_tieaware_dataset_manifest.csv`.

Source terms and sharing boundaries are recorded in `data/README.md`. Raw data
remain excluded; the manifests record cache hashes and split construction.

## Reproducing

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `02_data/` at the repository root with the required real-data caches,
or set `SOFTMCC_PROJECT_ROOT` to a directory that contains them. Then run:

```bash
python code/harden_dupsafe_tieaware.py --force
```

The six-setting provenance run is available through
`code/harden_dupsafe_tieaware.py`. For the 18-setting protocol, first prepare
the expansion cache and then launch the restartable unit runner:

```bash
python code/expansion_datasets.py
python code/expansion_runner.py --run-root experiments/arm18 \
  --run-id arm18 --datasets all --repeats 12 --prepare-expansion
python code/expansion_aggregate.py --run-root experiments/arm18 \
  --skip-reproduction
python code/analyze_round_b_sensitivity.py
```

The runner keeps exact feature-label duplicate groups within a single train,
validation, or test partition and writes immutable per-setting-by-repeat
shards. The manuscript-facing run was executed in the pinned ARM environment
recorded in `results/harden_dupsafe_tieaware_environment.json`.

## Headline Duplicate-Safe Results

- Across 18 settings, SoftMCC has the highest mean tie-corrected Kendall's W
  (`0.659`) and the best stability mean rank (`2.31`).
- The stability Friedman test is significant (`chi-square=17.652`,
  `p=0.0072`); primary Nemenyi separation from SoftMCC is established for
  AUPRC and MCC@0.5. In the conservative 14-source-family sensitivity,
  Friedman remains significant (`p=0.0055`) but only MCC@0.5 remains separated.
- The selected-utility Friedman test is not significant (`p=0.1169`).
  SoftMCC-selected models nevertheless have observed mean test-MCC losses
  against MCC@best (`difference=-0.0045`), F1@best
  (`difference=-0.0057`), and Brier (`difference=-0.0031`). Holm correction
  across all six paired comparisons retains only F1@best
  (`adjusted p=0.0139`); all three mean losses remain explicit limitations.
- Calibration shift remains a limitation: SoftMCC mean Spearman agreement is
  `0.851` with BCa 95% CI `[0.824, 0.872]`, and it is `0.785` at `T=3`.
- Across 200 deterministic validation-label permutations, SoftMCC W has mean
  `0.092` and central 95% interval `[0.058, 0.129]` versus observed `0.659`
  (one-sided empirical exceedance `0.005`). Kappa-only and rho-only W are
  `0.602` and `0.520`, respectively.

These findings support a comparison-specific stability claim, not a general
performance-superiority claim. The earlier Windows evidence remains provenance;
all current manuscript-facing values come from the pinned ARM evidence family.

`results/kappa_tieaware_summary.json` preserves legacy field names ending in
`all6`; in this frozen file those fields contain all 18 ARM settings. Its
`benchmark` and `real` subsections describe the two original three-setting
subsets, not a partition of all 18 settings. The file is not renamed or
rewritten so its frozen evidence hash remains stable.

## License

Code is released under the MIT license. Raw dataset redistribution is governed
by the original dataset licenses.
