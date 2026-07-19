# Tie-Aware Evidence Remediation — 2026-07-18

Canonical run:
`2026-07-18_codex_local_unknown_tieaware_harden_control`

## Why the evidence was refreshed

The earlier grouped-split evidence ranked candidates with an ordinal stable
sort. Exact score ties were therefore broken by candidate declaration order
when Kendall's W was computed, although the manuscript described midranks. The
refreshed analysis separates two policies:

- candidate declaration order resolves an exact selection-score tie only when
  one model must be selected for test utility;
- ranking stability uses archived candidate scores, average ranks for ties,
  and the standard tie-corrected Kendall's W denominator.

The prior evidence is retained for provenance and is not silently overwritten.

## Verification

- 504 utility rows, 504 ranking rows, 360 candidate-score rows, 2016
  calibration rows, and 72 split-manifest rows.
- Zero train--validation, train--test, or validation--test duplicate-group
  overlap.
- Proposition checks: maximum numerical error is below `7e-15` for the
  finite-sample identities and below `2e-16` for the population calibration
  identity; no Pearson-bound violation was observed in 500 trials.
- The non-converse example has calibration error `0.19` while SoftMCC and Brier
  skill score agree to floating-point precision. Their difference is therefore
  not an identifiable calibration-error measure.

## Results

- SoftMCC mean W: 0.737269; benchmark 0.710648; real 0.763889.
- Baseline mean-W range: 0.495929--0.627546.
- Stability Friedman: chi-square 6.645, p 0.3547; Nemenyi CD 3.678.
- Selection Friedman: chi-square 5.927, p 0.4314; Nemenyi CD 3.678.
- Selection means remain 0.650--0.665 and all SoftMCC-versus-baseline Wilcoxon
  p-values remain at least 0.151.
- SoftMCC, Brier, AUROC, and AUPRC contain no exact candidate-score ties in the
  archived blocks. Thresholded baselines do; the largest correction is
  MCC@0.5 on synth(1%), from ordinal W 0.408333 to tie-corrected W 0.008152.
- Across 200 deterministic validation-label permutations, all-six-dataset W
  has mean 0.097977, SD 0.031935, central 95% interval
  [0.047662, 0.167396], and one-sided empirical p 1/201 = 0.004975 against the
  observed W 0.737269.
- Kappa-only W is 0.653704 and rho-only W is 0.685880. Sharpness contributes to
  stability but is not its sole explanation.
- The largest absolute dataset-level test-MCC gap between SoftMCC and the
  kappa-only selector is 0.007252.

## Interpretation boundary

The corrected estimator preserves the manuscript's descriptive SoftMCC mean-W
result but weakens the already non-significant Friedman statistic and materially
changes several threshold-baseline W values. No multi-dataset superiority or
selection-equivalence claim is supported. Proposition 6 is a one-way calibrated
population identity; observed finite-sample departures are not calibration
diagnostics.
