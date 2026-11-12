Original competition dataset (read-only):
/mnt/disk1/wangxinbei_contest_dataset/C01

Read all three rf_signal_benchmark_train_v1.0 Parquet files.
Schema and sample IO are supplied alongside the original files.
No external training datasets or pretrained weights were used in preliminary tests.

Generated content-hash audit and manifests:
artifacts/c01_data_audit/

Complete preliminary pipeline (small real-data subset):
artifacts/full_pipeline_smoke/

Actual source I/Q bytes are verified against the audit manifest during cache creation.
All training/validation splits lock identical waveform hashes together.
Sample IO waveform groups are excluded from the performance-reporting split.
Normalization and regression value grids are fitted on the training split only.

The current weights/ model is a three-step preliminary test model, not a
fully trained competition submission. See docs/self_test_report.md.
