# Reproduce Main Table

Traditional main-table rows use `Code/scripts/data/run_sparsity_aware_lts_eval.py`.

Core commands:

```bash
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant HeuristicCueV1
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant BBoxLoomingV4 --disable-lts-fallback
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant BBoxLoomingV10 --disable-lts-fallback
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant BBoxLoomingV13 --disable-lts-fallback
```

The full version chain from `HeuristicCueV1` through `BBoxLoomingV13` is implemented in `Code/evttc/sparsity_aware_lts.py` and selectable through the `--variant` flag.

Exact paper rows and source run directories are listed in `PAPER_RESULTS_MAP.md` and `manifests/paper_results_manifest.json`.
