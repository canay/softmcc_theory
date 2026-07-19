# softmcc_theory

Repository: https://github.com/canay/softmcc_theory

Replication package for:

**SoftMCC: A Probability-Valued MCC Evaluation Framework for Reproducible
Validation Ranking under Class Imbalance**

This package supports the manuscript's duplicate-safe empirical evidence. It
does not claim that the general soft-count MCC construction is new. SoftMCC is
used here as a post-training evaluation and validation-ranking framework around
a probability-valued MCC core score.

## Contents

- `code/softmcc_scorer.py`: scikit-learn compatible SoftMCC scorer.
- `code/softmcc_eval.py`: metric definitions and helper functions.
- `code/harden_dupsafe_full.py`: shared duplicate-safe grouped training and
  metric implementation.
- `code/harden_dupsafe_tieaware.py`: canonical runner; archives every
  candidate score, uses midranks and standard tie-corrected Kendall's W, runs
  the 200-permutation kappa/rho controls, and regenerates the evidence family.
- `code/verify_softmcc_identities.py`: numerical checks for Propositions 1--3
  and the population calibration identity, including a non-converse example.
- `code/duplicate_leakage_audit.py`: audit for exact duplicate rows and
  cross-split leakage under ordinary stratified splitting.
- `code/real_prep.py`: cache construction notes for the public real datasets.
- `results/harden_dupsafe_tieaware_*`: canonical raw outputs, archived
  candidate scores, midranks, tie audit, effect sizes, summaries, split and
  dataset manifests, and reconciliation against the prior ordinal evidence.
- `results/kappa_tieaware_*`: 200 label-permutation results plus kappa-only and
  rho-only controls reconstructed from the archived prediction bundle.
- `results/harden_dupsafe_*` without `tieaware`: prior ordinal evidence retained
  for provenance; it is not the manuscript-facing W estimator.
- `figures/*_dupsafe.png`: manuscript result figures generated from the
  duplicate-safe evidence family.
- `results/harden_dupsafe_tieaware_*`: tabular result sources used to construct
  the manuscript-facing summary table.
- `docs/TIEAWARE_EVIDENCE_REMEDIATION_20260718.md`: current evidence,
  interpretation, and supersession note.
- `docs/DUPSAFE_EVIDENCE_REMEDIATION_20260622.md`: prior remediation provenance.

## Data

Raw data are not redistributed in this package.

- Credit Card Fraud Detection: Kaggle `mlg-ulb/creditcardfraud`; the official
  Kaggle record labels the license `Database: Open Database, Contents: Database
  Contents`.
- IoTID20: the author-maintained source grants perpetual free use for academic
  research with citation of DOI `10.1007/978-3-030-47358-7_52`; the minority
  `normal` class is treated as positive in the cache used here.
- Breast Cancer Wisconsin: `sklearn.datasets.load_breast_cancer`; malignant is
  treated as positive by using `1 - target`.
- Synthetic datasets are generated with `sklearn.datasets.make_classification`.

Source links and sharing boundaries are recorded in `data/README.md`. Raw data
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

The runner writes a dated immutable run under `experiments/`, validates row
counts and split overlaps, and generates figures and Table 2. It keeps exact
feature-label duplicate groups within a single train, validation, or test
partition. Use `python code/harden_dupsafe_tieaware.py --self-test-only` for
the tie-correction smoke tests.

## Headline Duplicate-Safe Results

- SoftMCC has the highest observed ranking reproducibility in both suites:
  Kendall's W is 0.711 for the benchmark suite and 0.764 for the real suite.
- The dataset-level Friedman test for stability is not significant
  (`chi-square=6.645`, `p=0.3547`); no significant multi-dataset superiority is
  claimed.
- Selected-model test MCC remains in a narrow range across selectors
  (`0.650` to `0.665`); all SoftMCC-vs-baseline BCa intervals cross zero and all
  Wilcoxon tests have `p>=0.151`.
- Calibration shift remains a limitation: SoftMCC mean Spearman agreement is
  0.838 with BCa 95% CI [0.777, 0.881], and it drops to 0.675 at `T=3`.
- SoftMCC has no exact candidate-score ties in this design, so its W remains
  0.737 after the corrected estimator. Threshold-based baselines do change;
  the largest correction is MCC@0.5 on synth(1%), from ordinal 0.408 to
  tie-corrected 0.008.
- Across 200 deterministic validation-label permutations, SoftMCC W has mean
  0.098 and central 95% interval [0.048, 0.167] versus observed 0.737
  (one-sided empirical `p=0.005`). Kappa-only and rho-only W are 0.654 and
  0.686, respectively.

## License

Code is released under the MIT license. Raw dataset redistribution is governed
by the original dataset licenses.
