# H3 GT-box server deployment

All commands run on Linux from the repository root. H3 uses the modern `beeid-dino` environment for the primary ResNet50 and DINOv3 models. No old TOPIC detector/tracker environment is required.

## 1. Update code and reinstall editable package

```bash
cd ~/projects/bee-appearance-identifiability
git fetch origin
git checkout codex/h3-gt-tracking
git pull --ff-only
~/venvs/beeid-dino/bin/python -m pip install --no-deps -e .
```

## 2. Validate the unchanged protocol

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h3-validate-protocol \
  --protocol configs/h3_protocol.lock.yaml \
  --checksum configs/h3_protocol.lock.sha256
```

The command must return `final_test_access: false`.

## 3. Synthetic smoke

```bash
~/venvs/beeid-dino/bin/python -m beeid.cli h3-synthetic-smoke \
  --output ~/experiments/bee-appearance-identifiability/h3-synthetic
```

This uses `test_only_encoder` and is never an experiment result.

## 4. Prepare a real smoke configuration

```bash
cp configs/h3_smoke.example.yaml configs/h3_smoke.local.yaml
```

Edit every external path. `paths.h1_output_root` must be the completed H1 directory containing the frozen full-train manifest. The example selects:

- BEE24-01 from `project_train`, needed to exercise threshold fitting;
- BEE24-03 from `development_validation`, needed to exercise tracking evaluation;
- at most the first 300 frames of each.

Then run:

```bash
bash scripts/h3_smoke_test.sh \
  configs/h3_smoke.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

Do not use a development-only subset for the smoke: threshold fitting must still receive at least one project-train video.

## 5. Full GT-box development

After smoke output is complete and plausible:

```bash
cp configs/h3.local.yaml.example configs/h3.local.yaml
# Fill external paths. Keep allow_subset=false, stage=gt_detection_boxes and final-test protocol unchanged.
bash scripts/run_h3.sh \
  configs/h3.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

The workflow performs:

```text
input/split audit
  -> outcome-blind train+development signal cache
  -> ResNet50 and DINOv3 train+development feature caches
  -> project_train-only threshold artifact
  -> four development GT-box trackers
  -> metrics, paired video bootstrap, metadata
```

Feature extraction is the expensive step because H3 must cover project-train observations that H2/H2.5 never extracted. Valid shards, signal fingerprints and threshold fingerprints are reused. Resume with:

Signal construction progress is written to `<output_root>/h3_logs/signal_progress.json`; feature extraction progress is visible from completed atomic shards under `<cache_root>/features/h3__<model>/.../`.

```bash
bash scripts/resume_h3.sh \
  configs/h3.local.yaml \
  ~/venvs/beeid-dino/bin/python
```

## 6. Manual stage commands

```bash
PY=~/venvs/beeid-dino/bin/python
CFG=configs/h3.local.yaml

"$PY" -m beeid.cli h3-validate --config "$CFG"
"$PY" -m beeid.cli h3-signals --config "$CFG"
"$PY" -m beeid.cli h3-extract --config "$CFG" --model resnet50
"$PY" -m beeid.cli h3-extract --config "$CFG" --model dinov3
"$PY" -m beeid.cli h3-fit-thresholds --config "$CFG" --models resnet50 dinov3
"$PY" -m beeid.cli h3-track --config "$CFG" --models resnet50 dinov3
"$PY" -m beeid.cli h3-report --config "$CFG" --models resnet50 dinov3
```

## 7. GO/STOP after GT development

Send these files for analysis:

- `h3_summary.csv`;
- `h3_per_video_metrics.csv`;
- `h3_paired_video_metrics.csv`;
- `h3_video_cluster_bootstrap.csv`;
- `h3_run_metadata.json`;
- `h3_tracking_metadata.json`;
- `h3_thresholds.json`.

Do not run final test. The next possible stage is `fixed_detector_boxes`, and its adapter/result format remains `SERVER_VALIDATION_PENDING` until the GT development result is reviewed and one fixed detector output is selected without looking at final-test results.
