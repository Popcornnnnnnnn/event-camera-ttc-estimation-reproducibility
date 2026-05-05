# Open Source Reproducibility Audit

> Scope: prepare this workspace as a clean reproducibility repository for the paper experiments, not as a thesis-writing archive and not as a final-algorithm-only demo.

## Release Target

The public repository should let a reader trace every paper result to:

- code entry
- configuration or command
- original run directory
- lightweight summary
- reproduction command
- data or external-code requirements

The release branch is `codex/open-source-reproducibility-release`. It must stay separate from thesis writing branches such as `codex/thesis-writing-eams-compliance`.

## Keep

| area | keep | reason |
|---|---|---|
| core package | `Code/evttc/` | formal loader, native adapter, representations, confidence, BBoxLooming, ML baseline components |
| traditional scripts | `Code/scripts/data/run_sparsity_aware_lts_eval.py`, diagnostic scripts, formal data preparation scripts | reproduces HeuristicCueV1 through BBoxLoomingV13 and main traditional table rows |
| ML scripts | `export_ml_baseline_dataset.py`, `run_ml_baseline_train.py`, `run_ml_baseline_train_torch.py` | reproduces NumPy, PyTorch/MPS, BBoxAux, residual, and stats/meta+bbox baselines |
| legacy adapter | `Code/scripts/analysis/run_legacy_clean_rerun.py` | reproduces direct formal CMax/STRTTC comparisons when external kernels are supplied |
| post-V13 scripts | `run_v13_frontier_tabular.py`, confidence/boundary/traditional/event-motion/sequence/low-observability audits | reproduces learned calibration and post-V13 frontier/audit rows |
| configs | `configs/` | public, path-parameterized run commands |
| manifests | `manifests/paper_results_manifest.json`, `PAPER_RESULTS_MAP.md` | row-level traceability for all paper result rows |
| summaries | `results_summaries/` | lightweight result tables, no large per-interval outputs |
| fixtures | `fixtures/synthetic_formal/` | small smoke data that verifies the formal loader and BBoxLooming metric path |
| public docs | `README.md`, `docs/reproducibility/` | cold-reader instructions for installation, data prep, and reproduction |

## Exclude

| area | exclude | reason |
|---|---|---|
| full datasets | `Dataset-Full/`, `Data/`, `Dataset/`, `Code/DatasetFormal/` | large data and/or local prepared artifacts |
| large experiment outputs | `Code/Experiments/**`, `Code/Derived/**` | too large for source control; keep summaries/manifests only |
| thesis writing | `Template/`, `Tex/`, `Thesis.pdf`, LaTeX build outputs | private thesis submission material, not reproducibility code |
| private process docs | `docs/active/`, `docs/history/`, `LongTermMemory`, `SessionLog`, `Tasks`, learning paths | working notes and personal process documents |
| planning/runtime state | `.planning/`, `.codex/`, `.claude/`, local agent state | local workflow state, not public source |
| literature downloads | `Papers/*.pdf` | downloaded PDFs should not be redistributed here |
| local environments | `.venv/`, `venv/`, caches, `.DS_Store`, logs | machine-local artifacts |
| credentials | keys, tokens, absolute private config | no credential material was found in the kept code path; continue to scan before publishing |

## Needs Desensitization

| item | current issue | action |
|---|---|---|
| legacy clean rerun defaults | previously used machine-local worktree paths | changed defaults to `external/...` and environment variables `EVTTC_CMAX_CORE_PATH`, `EVTTC_STRTTC_BASELINES_DIR` |
| historical run summaries | original `Summary.json` files may contain absolute local paths | do not publish raw full summaries by default; publish `results_summaries/paper_results_summary.*` |
| generated formal metadata | full local `SequenceMeta.json` can include local raw-data paths | do not commit `Code/DatasetFormal/`; regenerate locally from documented data paths |
| license | MIT file added as a draft public license | confirm copyright holder and final license before publishing |

## External Data Or Code Dependencies

| experiment family | dependency |
|---|---|
| traditional BBoxLooming | full formal dataset generated under `EVTTC_FORMAL_ROOT` for paper-number reproduction |
| ML baselines | full formal dataset plus exported ML dataset under `EVTTC_ML_DATASET_ROOT` |
| CMax clean rerun | external CMax kernel path via `EVTTC_CMAX_CORE_PATH` |
| STRTTC clean rerun | external STRTTC Python baseline directory via `EVTTC_STRTTC_BASELINES_DIR` |
| Post-V13 learned calibration/audits | V13 full-run outputs and feature tables generated from full formal data |

## One-Command Status

| item | status |
|---|---|
| synthetic formal loader smoke | runnable without full dataset |
| synthetic BBoxLoomingV13 no-fallback smoke | runnable without full dataset |
| paper main traditional rows | reproducible after full formal dataset preparation |
| ML baseline rows | reproducible after full formal dataset and ML export |
| CMax/STRTTC rows | reproducible after full formal dataset plus external legacy kernels |
| Post-V13 rows | reproducible after V13 full run and derived feature tables |

## Current Gaps Before Public Release

- The repository still needs a final tracked-file cleanup before a public commit: tracked private docs and thesis files should be removed from the public branch index without deleting local working files.
- External CMax/STRTTC code is referenced but not vendored. Public docs must tell users how to provide compatible local paths.
- Full dataset acquisition is documented as user-provided. Large raw data is intentionally not included.
- Exact Post-V13 reproduction depends on generated V13 feature tables; the manifest records source runs, but a fresh public user must run the upstream commands first.
