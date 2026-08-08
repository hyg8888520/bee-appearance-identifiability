## Scope

Implements BEE24 H1 GT-crop appearance identifiability without committing data, weights, caches, outputs or local YAML.

## Locally verified

- [ ] CPU unit test suite (paste exact command/result)
- [ ] Synthetic manifest → cache → evaluate → report smoke (test-only encoder)
- [ ] Staged diff, large-file, secret and local-YAML audit

## SERVER_VALIDATION_PENDING

- [ ] RTX 4090 CUDA tensor and AMP
- [ ] ResNet50 real-crop batch
- [ ] DINOv3 ViT-S/16 `forward_features()` keys/shapes and real-crop batch
- [ ] TOPIC AGW strict checkpoint load and real-crop batch
- [ ] 100–300 frames, single video, throughput/memory/disk estimate
- [ ] Full H1 (only after user confirmation)

## Server inputs required

- BEE24 root
- DINOv3 checkout and checkpoint
- TOPICTrack checkout and BEE AGW checkpoint
- Both environment interpreters, cache root and output root in `configs/h1.local.yaml`

## AGW status

`SERVER_VALIDATION_PENDING` (or paste `BLOCKED_TOPIC_AGW_COMPATIBILITY` evidence; never substitute ResNet50).

## Pinned upstreams and first command

See `references.lock.yaml`. First server command after filling the smoke config:

```bash
bash scripts/check_server_env.sh configs/h1_smoke.local.yaml
```
