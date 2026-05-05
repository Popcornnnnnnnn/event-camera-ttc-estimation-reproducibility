# Reproduce Legacy Baselines

Legacy clean reruns use current formal data loading but external legacy algorithm kernels.

Required environment:

```bash
export EVTTC_CMAX_CORE_PATH=/path/to/cmax_aligned_core.py
export EVTTC_STRTTC_BASELINES_DIR=/path/to/strttc/baselines
```

Full clean direct rerun:

```bash
python Code/scripts/analysis/run_legacy_clean_rerun.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/LegacyCleanReruns" --max-intervals 0 --methods cmax strttc-baseline strttc-quality-aware --max-ttc 25 --cmax-coarse-grid 11 --strttc-max-points 600 --strttc-negative-ttc-policy abs
```

These rows are direct formal comparisons. They are not claims of official end-to-end CMax or official MATLAB STRTTC reproduction.
