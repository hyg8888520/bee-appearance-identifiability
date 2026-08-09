## Scope

Implements BEE24 H1 GT-crop appearance identifiability and H2 observation-reliability/template-contamination diagnostics without committing data, weights, caches, outputs or local YAML.

## Locally verified

- [ ] CPU unit test suite (paste exact command/result)
- [ ] Synthetic manifest → cache → evaluate → report smoke (test-only encoder)
- [ ] H2 synthetic signals → intervention caches → diagnostics → contamination → report smoke
- [ ] Staged diff, large-file, secret and local-YAML audit

## SERVER_VALIDATION_PENDING

- [ ] RTX 4090 CUDA tensor and AMP
- [ ] ResNet50 real-crop batch
- [ ] DINOv3 ViT-S/16 `forward_features()` keys/shapes and real-crop batch
- [ ] TOPIC AGW strict checkpoint load and real-crop batch
- [ ] 100–300 frames, single video, throughput/memory/disk estimate
- [x] First full H1 development run (link audited external evidence; do not commit outputs)
- [ ] H2 one-development-video intervention smoke
- [ ] Full H2 development diagnostics (only after user confirmation)
- [ ] Final test (`NOT_RUN_DO_NOT_PEEK` until method freeze)

## Server inputs required

- BEE24 root
- DINOv3 checkout and checkpoint
- TOPICTrack checkout and BEE AGW checkpoint
- Completed H1 output root containing the frozen manifest/cache registry/metadata
- Both environment interpreters, cache root and separate H2 output root in `configs/h2.local.yaml`

## AGW status

Official checkpoint: `REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED`. Compatibility must still be strict; paste `BLOCKED_TOPIC_AGW_COMPATIBILITY` evidence if it fails and never substitute ResNet50.

## Pinned upstreams and first command

See `references.lock.yaml`. First server command after filling the smoke config:

```bash
python -m beeid.cli h2-synthetic-smoke
```
