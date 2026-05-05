# Reproduce Post-V13

Post-V13 rows are learned calibration and audit results on top of the V13-valid set. They are not traditional `BBoxLoomingV13` results.

Typical order:

```bash
python Code/scripts/analysis/run_v13_frontier_tabular.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --enable-lightgbm --run-name V13FrontierLGBM_full_v2
python Code/scripts/analysis/run_post_v13_confidence_gate.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name V13ConfidenceGate_iqrStd_target6p9_v2 --target-e-ttc-pct 6.9
python Code/scripts/analysis/run_post_v13_boundary_audit.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name PostV13BoundaryAudit_v1
python Code/scripts/analysis/run_post_v13_traditional_policy_audit.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name PostV13TraditionalPolicyAudit_v1
python Code/scripts/analysis/run_post_v13_event_motion_audit.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name PostV13EventMotionAudit_v1
python Code/scripts/analysis/run_post_v13_consistency_gate.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name V13ConsistencyGate_centerWideSweep_v1 --windows 11 --modes center --targets 6.875
python Code/scripts/analysis/run_post_v13_low_observability_audit.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name PostV13LowObservabilityAudit_v2
```

Most scripts default to reading the parent feature table from `Code/Experiments/PostV13Frontier/V13FrontierLGBM_full_v2/FeatureTable.csv`. Override `--feature-table` when using a different output root.
