# Round B source-family and multiplicity sensitivity

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
Friedman `p=0.002336` and Nemenyi CD
2.326203; separation from SoftMCC is retained for
AUPRC, MCC@0.5.

A more conservative source-collection analysis also groups the two thyroid
tasks from the UCI Thyroid Disease collection. It therefore uses 14 families.
This analysis gives Friedman `p=0.005508` and Nemenyi
CD 2.407848; separation from SoftMCC is retained
for MCC@0.5. The manuscript uses this more conservative result
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
F1@best. The output also reports total blocks, zero differences,
effective nonzero pairs, and matched-pairs rank-biserial correlations.

## Reproduction

Run:

```text
python code/analyze_round_b_sensitivity.py
```

The generated CSV and JSON files are listed in `ARTIFACT_MANIFEST.csv`.
