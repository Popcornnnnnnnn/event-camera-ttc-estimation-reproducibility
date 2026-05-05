# Reproduce ML Baselines

ML rows require a derived ML dataset exported from the full formal dataset.

```bash
export EVTTC_ML_DATASET_ROOT=/path/to/Derived/MlBaselineV1/tau10_20_40_64x64_g3x3_polsplit_all
python Code/scripts/data/export_ml_baseline_dataset.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$(dirname "$EVTTC_ML_DATASET_ROOT")/.." --tag "$(basename "$EVTTC_ML_DATASET_ROOT")" --image-variant multi_tau_polarity_split_v1 --multi-tau-ms 10 20 40 --image-height 64 --image-width 64 --grid-rows 3 --grid-cols 3
```

NumPy image-only baseline:

```bash
python Code/scripts/data/run_ml_baseline_train.py --dataset-root "$EVTTC_ML_DATASET_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/MlBaseline" --model cnn --feature-mode image_only --target log_ttc --run-name TinyCnnImageOnlyV1_AllData_MultiTauPolSplitV1
```

PyTorch image-only baseline:

```bash
python Code/scripts/data/run_ml_baseline_train_torch.py --dataset-root "$EVTTC_ML_DATASET_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/MlBaselineTorch" --device auto --feature-mode image_only --target log_ttc --run-name TorchTinyCnnImageOnlyV1_AllData_MultiTauPolSplitV1_Mps
```

BBoxAux and residual baselines require an auxiliary `IntervalMetrics.csv` from the BBoxLooming run used as the auxiliary cue.
