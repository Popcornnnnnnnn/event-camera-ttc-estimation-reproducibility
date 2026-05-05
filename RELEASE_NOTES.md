# v0.1.0 - Initial Reproducibility Release

This release publishes a clean reproducibility repository for the thesis
`Time-To-Collision (TTC) estimation with event-based cameras`.

## What Is Included

- Core `evttc` Python package for formal loading, native adapter support,
  event representations, confidence features, BBoxLooming variants, and ML baseline helpers.
- Reproduction scripts for the traditional method lineage from `HeuristicCueV1`
  through `BBoxLoomingV13`.
- Reproduction scripts and configs for ML baselines, hybrid BBox-assisted baselines,
  Post-V13 learned calibration, confidence gates, sequence consistency, low-observability
  audit, and legacy CMax/STRTTC clean rerun adapters.
- Row-level result map for all paper-table entries:
  `PAPER_RESULTS_MAP.md`, `manifests/paper_results_manifest.json`, and
  `results_summaries/paper_results_summary.*`.
- Full local reproduction audit summary:
  `FULL_REPRODUCTION_AUDIT.md` and `results_summaries/full_reproduction_audit.json`.
- A small synthetic fixture and smoke test that runs without the full dataset.

## What Is Excluded

- Full datasets and generated formal artifacts.
- Large experiment output directories.
- Thesis drafts, LaTeX templates, school materials, private process docs, and local planning state.
- External CMax/STRTTC source trees. The adapter documents the required environment variables.

## Verification

Verified before publication:

```text
FullReproductionAudit: formal_sequences=32 manifest_rows=40 passed_rows=40 failed_rows=0
SmokePass: est_valid=5, e_ttc_pct=8.586%
json_ok=3 manifest_rows=40 csv_rows=40 audit_passed=40 audit_failed=0
```

The full audit was performed against the local complete dataset and full-run artifacts.
The public repository intentionally contains only lightweight summaries and reproducibility
metadata; users need to prepare the full dataset paths documented in `README.md` to rerun
the complete experiments.
