# SLTR server deployment

Status before a real BEE24 run: `SERVER_VALIDATION_PENDING`.

SLTR does not extract features or train a GPU backbone. It reads the completed H1 manifest,
the completed H3 ResNet50/DINOv3 caches and H3 observation signals. The only fitted parameters
are two small NumPy logistic selectors, one per backbone. A GPU is therefore not expected to be
the bottleneck; do not interpret low GPU utilization as a failure.

## Prepare

```bash
cd ~/projects/bee-appearance-identifiability
git checkout codex/sltr-selective-intervention
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .
cp configs/sltr.local.yaml.example configs/sltr.local.yaml
```

Edit only external paths in `configs/sltr.local.yaml`. In particular, `h1_output_root` and
`h3_output_root` must point to the completed runs; `output_root` must be a new external directory.
Never point it at an H1/H3/H4/H5 output or feature cache.

## Validate and smoke

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli sltr-validate-protocol \
  --protocol configs/sltr_protocol.lock.yaml \
  --checksum configs/sltr_protocol.lock.sha256

~/venvs/beeid-dino/bin/python -m beeid.cli sltr-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/sltr-synthetic

bash scripts/sltr_smoke_test.sh \
  configs/sltr_smoke.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

The real subset smoke may legitimately stop at the train-support gate. A STOP is a scientific
result, not a reason to relax the frozen threshold.

## Full development run

```bash
bash scripts/run_sltr.sh \
  configs/sltr.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

Monitor or resume with:

```bash
bash scripts/monitor_sltr.sh configs/sltr.local.yaml ~/venvs/beeid-dino/bin/python 5
bash scripts/resume_sltr.sh configs/sltr.local.yaml ~/venvs/beeid-dino/bin/python
```

The workflow is `audit -> fit -> gated development tracking -> report`. If either backbone lacks
training support or a safe OOF threshold, it stops before learned development tracking. Final-test
videos are never read. Do not unlock or run final test from this development protocol.

## Package results

```bash
bash scripts/package_sltr_results.sh /path/to/sltr-output /path/to/sltr-results.tar.gz
```

Send the archive only after checking `sltr_run_metadata.json`, `sltr_fit_decision.json`, and
`sltr_method_decision.json`. GT-box results remain an association experiment without detector
false positives or false negatives; they are not end-to-end MOT results.
