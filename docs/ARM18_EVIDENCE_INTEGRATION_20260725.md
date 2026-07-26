# ARM 18-setting evidence integration

The manuscript-facing empirical evidence comes from
`2026-07-24_claude_vps_marsilya_dataset_expansion`, executed on pinned Linux
ARM with 18 imbalanced binary tabular settings and 12 duplicate-safe grouped
repeats per setting.

The primary setting-level stability result is comparison-specific. SoftMCC has
the best mean stability rank, and the omnibus Friedman test is significant,
with Nemenyi separation from AUPRC and MCC@0.5. A conservative 14-source-family
sensitivity retains the significant omnibus result but only the MCC@0.5
separation. No separation that survives both analyses is established against
AUPRC, MCC@best, F1@best, AUROC, or Brier.

The utility result is a limitation. Under the prespecified paired block-level
rule, SoftMCC-selected models have observed mean test-MCC losses against
MCC@best, F1@best, and Brier. Holm correction across the six paired comparisons
retains only F1@best. The selection-utility omnibus test is not significant,
and the MCC@best BCa interval crosses zero; all qualifications and all three
observed mean losses are retained in the manuscript.

`results/arm18_decision_summary.json` is the compact decision record.
`results/arm18_decision_criteria.csv` preserves the locked decision criteria.
The `harden_dupsafe_tieaware_*` and `kappa_tieaware_*` files are the derived
evidence family used for manuscript tables and figures.

The earlier Windows run is retained only as immutable provenance. Its numerical
outputs are not mixed with the ARM values because the cross-platform
reproduction comparison did not satisfy the locked tolerance, selected
candidates differed, and the absolute test-MCC difference reached 0.38.

The derived Round B outputs and their executable analysis are under
`results/round_b_sensitivity/` and `code/analyze_round_b_sensitivity.py`.
The frozen ARM evidence files are not modified by this derived layer.
