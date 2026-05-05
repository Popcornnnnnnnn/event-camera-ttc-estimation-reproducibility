# Data Preparation

Full paper-number reproduction expects formal sequence artifacts under `EVTTC_FORMAL_ROOT`.

Each formal sequence directory must contain:

- `SequenceMeta.json`
- `Protocol.json`
- `Calib.json`
- `bbox.csv`
- `gt_ttc.csv`
- `IntervalIndex.csv`

Event data is loaded lazily from the HDF5 path recorded in `SequenceMeta.json`. Relative HDF5 paths are resolved against the sequence directory; absolute paths are supported for local prepared datasets.

Recommended environment:

```bash
export EVTTC_FORMAL_ROOT=/path/to/Code/DatasetFormal
export EVTTC_OUTPUT_ROOT=/path/to/repro_outputs
```

Preparation commands:

```bash
python Code/scripts/data/prepare_formal_batch.py --help
python Code/scripts/data/build_formal_sequence_artifacts.py --help
python Code/scripts/data/run_formal_loader_smoke.py --formal-root "$EVTTC_FORMAL_ROOT"
```

The public repository does not include `Dataset-Full/`, `Code/DatasetFormal/`, raw HDF5/MP4/BAG files, or generated full experiment outputs.
