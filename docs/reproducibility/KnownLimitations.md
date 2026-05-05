# Known Limitations

- Full datasets are not redistributed.
- Full `Code/Experiments/**` outputs are not redistributed; only lightweight summaries and manifests are kept.
- Legacy CMax/STRTTC adapters require compatible external source trees.
- Post-V13 learned calibration depends on upstream V13 feature tables. Reproduce V13 first, then run post-V13 scripts.
- Some original historical summaries contain absolute local paths. Public summaries avoid those paths; raw local summaries should not be published without review.
- Synthetic fixtures verify code paths only. They do not validate any paper metric.
