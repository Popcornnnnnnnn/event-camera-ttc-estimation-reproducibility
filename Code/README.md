# Code

Public code entry points for the event-camera TTC reproducibility repository.

## Package

`Code/evttc/` contains the reusable Python package:

| file | role |
|---|---|
| `formal_loader.py` | formal artifact reader and lazy event-slice loader |
| `native_adapter.py` | adapter from formal samples to algorithm inputs |
| `representations.py` | event representations and local statistics |
| `confidence.py` | early local confidence model |
| `sparsity_aware_lts.py` | `HeuristicCueV1` and `BBoxLoomingV1..V13` |
| `ml_baseline.py` | lightweight ML baseline export and model helpers |

## Scripts

| area | entry |
|---|---|
| traditional main table | `Code/scripts/data/run_sparsity_aware_lts_eval.py` |
| formal data preparation | `Code/scripts/data/prepare_formal_batch.py`, `Code/scripts/data/build_formal_sequence_artifacts.py` |
| ML dataset export | `Code/scripts/data/export_ml_baseline_dataset.py` |
| NumPy ML baseline | `Code/scripts/data/run_ml_baseline_train.py` |
| PyTorch/MPS ML baseline | `Code/scripts/data/run_ml_baseline_train_torch.py` |
| legacy clean reruns | `Code/scripts/analysis/run_legacy_clean_rerun.py` |
| post-V13 learned calibration | `Code/scripts/analysis/run_v13_frontier_tabular.py` |
| post-V13 audits | `Code/scripts/analysis/run_post_v13_*.py` |

## Data Policy

Large prepared data and experiment outputs are excluded:

- `Code/DatasetFormal/`
- `Code/Derived/`
- `Code/Experiments/`

Use environment variables such as `EVTTC_FORMAL_ROOT`, `EVTTC_ML_DATASET_ROOT`, and `EVTTC_OUTPUT_ROOT` instead of hard-coded local paths.
