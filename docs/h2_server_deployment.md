# H2 server deployment

H2 reuses the two modern H1 environments and the completed H1 manifest. It does not require the old TOPIC detector/tracker environment and does not read BEE24 final test.

## Prepare

From the repository root in the DINO environment:

```bash
git pull
~/venvs/beeid-dino/bin/python -m pip install -e .
~/venvs/beeid-topic/bin/python -m pip install -e .
cp configs/h2.local.yaml.example configs/h2.local.yaml
cp configs/h2_smoke.example.yaml configs/h2_smoke.local.yaml
```

Fill both local YAML files. The important new path is `paths.h1_output_root`, which must point to the completed H1 directory containing:

```text
manifests/observations.csv
resolved_split.yaml
cache_locations.json
run_metadata.json
summary.csv
query_results.csv
```

Set H2 `output_root` to a new external directory. The loader rejects an H2 output directory equal to H1 output. Keep the frozen project split at `configs/splits/project_split.yaml`.

## Validation order

```bash
# 1. CPU-only code and artifact smoke; no real model result is claimed.
~/venvs/beeid-dino/bin/python -m beeid.cli h2-synthetic-smoke \
  --output /path/to/experiments/beeid/h2-synthetic-smoke

# 2. Read-only validation of the completed H1 manifest and frozen split.
~/venvs/beeid-dino/bin/python -m beeid.cli h2-validate \
  --config configs/h2_smoke.local.yaml

# 3. One frozen development video, all interventions, all available models.
source ~/venvs/beeid-dino/bin/activate
bash scripts/h2_smoke_test.sh configs/h2_smoke.local.yaml
```

The smoke config selects `BEE24-03` and at most 75 frames. Change that video only to another ID already listed under frozen development validation. Do not select project-train or final-test videos for H2 outcome analysis.

Review:

- `h2_cache_locations.json` and cache sizes;
- `h2_context_ablation.csv` for every configured variant;
- `h2_memory_summary.csv` for nonzero event counts;
- `h2_run_metadata.json` for `final_test_read: false`;
- GPU utilization, peak memory, wall time, and disk use from the smoke run.

Record a measured estimate for each model (use its actual embedding dimension and smoke throughput), for example:

```bash
python -m beeid.cli h2-estimate --config configs/h2.local.yaml \
  --model dinov3 --embedding-dimension 384 \
  --observations-per-second 100 --peak-memory-gib 8
```

Do not copy the example throughput or memory into a report; replace both with measured values.

The primary `context_e020_raw_i224` cache should report `reused_h1_cache` when the completed H1 signature and shard IDs match. A subset smoke cannot reuse full-H1 shards safely and will create its own intervention cache.

## Full development run

Only after the real H2 smoke is reviewed:

```bash
source ~/venvs/beeid-dino/bin/activate
bash scripts/run_h2.sh configs/h2.local.yaml
```

Resume the same immutable configuration with:

```bash
bash scripts/resume_h2.sh configs/h2.local.yaml
```

Every cache shard is validated before reuse. Changing an intervention, input size, AMP state, checkpoint, selected observation list, or manifest creates a different fingerprint. Do not delete the completed H1 output and do not point either output root inside the repository.

## Acceptance boundary

The full H2 server run must preserve its console log plus all listed H2 artifacts. Until those files are returned and audited, H2 remains `SERVER_VALIDATION_PENDING`. Even after completion, H2 is development evidence: it may select the later reliability method, but it is not a final-test or end-to-end MOT result.
