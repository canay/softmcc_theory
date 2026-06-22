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
- `code/harden_dupsafe_full.py`: duplicate-safe grouped runner, analysis,
  figure generation, table generation, transcript, and manifests.
- `code/duplicate_leakage_audit.py`: audit for exact duplicate rows and
  cross-split leakage under ordinary stratified splitting.
- `code/real_prep.py`: cache construction notes for the public real datasets.
- `results/harden_dupsafe_*`: canonical raw outputs, summaries, split manifests,
  dataset manifest, environment file, artifact manifest, and reconciliation.
- `figures/*_dupsafe.png`: manuscript result figures generated from the
  duplicate-safe evidence family.
- `tables/table_main_results_dupsafe.tex`: manuscript Table 2 source.
- `docs/DUPSAFE_EVIDENCE_REMEDIATION_20260622.md`: evidence, inference, and
  interpretation log for the remediation run.

## Data

Raw data are not redistributed in this package.

- Credit Card Fraud Detection: Kaggle `mlg-ulb/creditcardfraud`.
- IoTID20: public IoT intrusion detection dataset; minority `normal` class is
  treated as positive in the cache used here.
- Breast Cancer Wisconsin: `sklearn.datasets.load_breast_cancer`; malignant is
  treated as positive by using `1 - target`.
- Synthetic datasets are generated with `sklearn.datasets.make_classification`.

Dataset license and access-date details should be verified before public
release. The manifest files record local cache hashes and split construction.

## Reproducing

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the duplicate-safe pipeline from a project layout containing `02_data/`
with the required real-data caches:

```bash
python code/harden_dupsafe_full.py --force
```

The runner writes `harden_dupsafe_*` outputs, validates row counts and split
overlaps, and generates figures and Table 2. It keeps exact feature-label
duplicate groups within a single train, validation, or test partition.

## Headline Duplicate-Safe Results

- SoftMCC has the highest observed ranking reproducibility in both suites:
  Kendall's W is 0.711 for the benchmark suite and 0.764 for the real suite.
- The dataset-level Friedman test for stability is not significant
  (`chi-square=7.433`, `p=0.2827`); no significant multi-dataset superiority is
  claimed.
- Selected-model test MCC remains in a narrow range across selectors
  (`0.650` to `0.665`); all SoftMCC-vs-baseline BCa intervals cross zero and all
  Wilcoxon tests have `p>=0.151`.
- Calibration shift remains a limitation: SoftMCC mean Spearman agreement is
  0.838 with BCa 95% CI [0.777, 0.881], and it drops to 0.675 at `T=3`.

## License

Code is released under the MIT license. Raw dataset redistribution is governed
by the original dataset licenses.
