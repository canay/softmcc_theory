# Round B derived sensitivity analysis

This layer is derived from the frozen 18-setting ARM evidence and does not
modify that evidence family. Run `python code/analyze_round_b_sensitivity.py`
to verify the packaged inputs and regenerate `results/round_b_sensitivity/`.

## Source-family sensitivity

The primary analysis uses 18 settings. The documented-overlap analysis combines
the credit-card prevalence variants, synthetic variants, and car-evaluation
targets, yielding 15 families. The conservative analysis also combines the two
tasks from the UCI Thyroid Disease collection, yielding 14 families.

For each metric, Kendall's W is averaged within each family. Metrics are then
midranked within families before the Friedman and Nemenyi calculations. The
conservative result is Friedman chi-square 18.3077, p = 0.005508, SoftMCC mean
rank 2.0357, and CD 2.4078. Only MCC@0.5 remains separated from SoftMCC; the
AUPRC rank gap is 2.3571 and remains below the CD.

The thyroid grouping follows the public dataset descriptions:

- https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html
- https://archive.ics.uci.edu/dataset/102/thyroid+disease

## Multiplicity and paired effects

The six prespecified SoftMCC-versus-baseline Wilcoxon tests form one family.
Holm correction at familywise alpha 0.05 retains only F1@best, with adjusted
p = 0.0138905. The observed mean losses against MCC@best, F1@best, and Brier
remain reported as limitations. Their adjusted p-values are 0.0689513,
0.0138905, and 0.146222, respectively; effective nonzero pair counts are 111,
114, and 60, and matched-pairs rank-biserial correlations are -0.269, -0.329,
and -0.310.

## Public-package interpretation

Public figures 1, 2, and 3 correspond to manuscript Figures 2, 3, and 4.
Panel (b) of public Figure 3 is descriptive because the selection-utility
Friedman omnibus test is nonsignificant; it contains no CD or clique marks.

The frozen `kappa_tieaware_summary.json` uses legacy keys ending in `all6` for
values that now summarize all 18 ARM settings. Its `benchmark` and `real`
subsections retain the original three-setting subsets. These keys are explained
here rather than renamed so the frozen evidence hash remains unchanged.
