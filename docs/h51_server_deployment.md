# H5.1 server deployment

Install the current package in the same Python 3.11 / torch 2.7.1 environment used for H5. H5.1
does not run a backbone or train a checkpoint, but it loads the completed H5 heads and reuses the
read-only H3 aligned feature caches.

First run the scientific synthetic smoke and then a minimal real two-video diagnostic using the
already completed H5 output: one `project_train` video is needed only for gradient-probe input
construction and one `development_validation` video is needed for teacher-forced audit and causal
replay. Do not run a fixed detector or final test.

```bash
cp configs/h51_smoke.example.yaml configs/h51_smoke.local.yaml
# Replace external paths, then retain one project_train plus one development_validation video.
bash scripts/h51_smoke_test.sh configs/h51_smoke.local.yaml ~/venvs/beeid-dino/bin/python
```

Inspect `h51_gradient_coverage.csv`, `h51_path_audit.json`, `h51_replay_equivalence.json`, and
`h51_decision.json`. The instrumentation-versus-H5 tracker equivalence is additionally locked by a
pytest/CPU-CI contract; it is not a separate runtime log. After the diagnostic smoke completes, copy
the full `configs/h51.example.yaml` to the ignored `configs/h51.local.yaml`, edit external paths only, and run
all development videos:

```bash
bash scripts/run_h51.sh configs/h51.local.yaml ~/venvs/beeid-dino/bin/python
bash scripts/monitor_h51.sh configs/h51.local.yaml ~/venvs/beeid-dino/bin/python 5
```

Interrupted replay is resumed with `scripts/resume_h51.sh`. Atomic jobs are keyed by checkpoint
SHA/fingerprint, both protocol hashes, cache fingerprint, observation IDs, variant, and all tracker
parameters. Signature mismatch is an error; move an obsolete H5.1 output directory aside instead of
mixing it with a new run.

Scripts default to 16 CPU threads and validate that `BEEID_H51_CPU_THREADS` is a positive integer.
H5.1 does not train or run a backbone: it performs a small-head gradient probe and sequential replay,
so RTX 4090 utilization can be brief and low while CPU, CSV, and cache I/O dominate. This does not
mean the run silently fell back to CPU; use `environment.actual_torch_device` in
`h51_run_metadata.json` as the authoritative device record.
Package only the auditable core after completion:

```bash
bash scripts/package_h51_results.sh /external/h51-output /external/h51-results.tar.gz
```

The packaged metadata records git commit, dirty state hashes, actual torch device,
`checkpoint_modified=false`, and `final_test_read=false`. `runtime_validation` and
`gpu_execution_validation` distinguish a test-only smoke, a real CPU diagnostic with GPU validation
still `SERVER_VALIDATION_PENDING`, and a real diagnostic completed on the configured CUDA runtime;
the device name is recorded separately and no particular GPU model is inferred.
