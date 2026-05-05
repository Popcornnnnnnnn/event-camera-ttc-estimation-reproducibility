# Time-To-Collision Estimation with Event-Based Cameras: Reproducibility Repository

This repository contains the code and lightweight evidence needed to reproduce the experiments behind the thesis `Time-To-Collision (TTC) estimation with event-based cameras` / `基于事件相机的碰撞时间（TTC）估计`.

It is a reproducibility repository. It keeps the full method lineage from `HeuristicCueV1` through `BBoxLoomingV13`, the ML and hybrid baselines, legacy CMax/STRTTC clean rerun adapters, and post-V13 learned calibration/audit scripts. It does not include full datasets, thesis drafts, templates, private process notes, or large experiment outputs.

Recommended public repository name: `event-camera-ttc-estimation-reproducibility`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional families:

```bash
python -m pip install -e '.[ml]'
python -m pip install -e '.[post-v13]'
python -m pip install -e '.[figures]'
```

## Smoke Test

The smoke test uses only the synthetic fixture in `fixtures/synthetic_formal/`.

```bash
python scripts/run_smoke_test.py
```

It checks formal loading, event-slice reading, `BBoxLoomingV13` no-fallback prediction, and metric calculation. These fixture numbers are not paper results.

## Full Reproduction Audit

The full audit verifies the local full-data evidence chain for every paper result row in `manifests/paper_results_manifest.json`.

```bash
python scripts/run_full_reproduction_audit.py --write-report
```

The audit checks that each row has an existing code entry, config file, original run directory, summary artifact, and metric match against the recorded full-run summary. It does not vendor large datasets or re-run long ML training by default.

## Data Preparation

Full paper-number reproduction requires the formal dataset artifacts:

```bash
export EVTTC_FORMAL_ROOT=/path/to/Code/DatasetFormal
export EVTTC_OUTPUT_ROOT=/path/to/repro_outputs
```

If starting from raw local data, use:

```bash
python Code/scripts/data/prepare_formal_batch.py --help
python Code/scripts/data/build_formal_sequence_artifacts.py --help
```

Large raw data and generated formal artifacts are intentionally excluded from git. See `docs/reproducibility/DataPreparation.md`.

## Reproduce Main Table

Traditional method rows:

```bash
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant HeuristicCueV1
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant BBoxLoomingV10 --disable-lts-fallback
python Code/scripts/data/run_sparsity_aware_lts_eval.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/SparsityAwareLTS" --variant BBoxLoomingV13 --disable-lts-fallback
```

All paper result rows are mapped in:

- `PAPER_RESULTS_MAP.md`
- `manifests/paper_results_manifest.json`
- `results_summaries/paper_results_summary.csv`

## Reproduce ML Baselines

Export the grouped ML dataset, then train the selected baseline:

```bash
export EVTTC_ML_DATASET_ROOT=/path/to/Derived/MlBaselineV1/tau10_20_40_64x64_g3x3_polsplit_all
python Code/scripts/data/export_ml_baseline_dataset.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$(dirname "$EVTTC_ML_DATASET_ROOT")/.." --tag "$(basename "$EVTTC_ML_DATASET_ROOT")" --image-variant multi_tau_polarity_split_v1 --multi-tau-ms 10 20 40
python Code/scripts/data/run_ml_baseline_train_torch.py --dataset-root "$EVTTC_ML_DATASET_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/MlBaselineTorch" --device auto --feature-mode image_only
```

BBox auxiliary and residual variants additionally need the BBoxLooming interval metrics CSV. See `docs/reproducibility/ReproduceMLBaselines.md`.

## Reproduce Post-V13 Audits

Post-V13 results depend on V13 full-run outputs and derived feature tables:

```bash
python Code/scripts/analysis/run_v13_frontier_tabular.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --enable-lightgbm --run-name V13FrontierLGBM_full_v2
python Code/scripts/analysis/run_post_v13_confidence_gate.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name V13ConfidenceGate_iqrStd_target6p9_v2 --target-e-ttc-pct 6.9
python Code/scripts/analysis/run_post_v13_low_observability_audit.py --output-root "$EVTTC_OUTPUT_ROOT/PostV13Frontier" --run-name PostV13LowObservabilityAudit_v2
```

See `docs/reproducibility/ReproducePostV13.md`.

## Reproduce Legacy Clean Reruns

Legacy adapters require external CMax/STRTTC source paths:

```bash
export EVTTC_CMAX_CORE_PATH=/path/to/cmax_aligned_core.py
export EVTTC_STRTTC_BASELINES_DIR=/path/to/strttc/baselines
python Code/scripts/analysis/run_legacy_clean_rerun.py --formal-root "$EVTTC_FORMAL_ROOT" --output-root "$EVTTC_OUTPUT_ROOT/LegacyCleanReruns" --max-intervals 0 --methods cmax strttc-baseline strttc-quality-aware --max-ttc 25 --cmax-coarse-grid 11 --strttc-max-points 600 --strttc-negative-ttc-policy abs
```

See `docs/reproducibility/ReproduceLegacyBaselines.md`.

## Traceability

`PAPER_RESULTS_MAP.md` is the top-level row map. Each row points to the code entry, config file, original run directory, lightweight summary, reproduction command, and whether full data or external code is required.
