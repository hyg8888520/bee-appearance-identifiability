# TOPIC BEE AGW modern-PyTorch compatibility record

Status before target-server execution: `SERVER_VALIDATION_PENDING`.

## Scope

Only TOPICTrack's BEE24 appearance baseline at `fast-reid/configs/bee/AGW_S50.yml` is in scope. The detector, YOLOX, tracker and official PyTorch 1.8/CUDA 11.1 environment are deliberately excluded. Runtime uses the FastReID copy within pinned TOPICTrack commit `e7b260f41a92fbe94419ce448aa7b53fe80a6814` and a separate modern Python 3.11 / PyTorch 2.7.1 CUDA 12.8 environment.

## Minimal adaptations

1. Temporarily add `<topictrack>/fast-reid` to Python import lookup; no source is copied or patched in place.
2. Merge the official BEE `AGW_S50.yml` first, retaining ResNeSt-50, GeM and BN neck settings.
3. Override only `MODEL.DEVICE`, `MODEL.WEIGHTS` (to prevent an implicit second load), `MODEL.BACKBONE.PRETRAIN=False`, inferred `MODEL.HEADS.NUM_CLASSES`, and `INPUT.SIZE_TEST`.
4. Infer classifier count from the unique `heads.weight` tensor in the checkpoint.
5. Compare every model/checkpoint key and shape. Any missing, unexpected or shape-mismatched entry produces a complete report and blocks AGW. Only an empty report proceeds to `load_state_dict(strict=True)`.
6. Pass RGB float32 in 0–255 range because FastReID's Baseline applies configured pixel mean/std internally.
7. Replace the official 384×384 test size with the common H1 224×224 protocol; 256×256 is a supported sensitivity setting. This is an intentional comparability override, not the official AGW test protocol.

No compatibility fallback substitutes torchvision ResNet50. Failure is reported as `BLOCKED_TOPIC_AGW_COMPATIBILITY`; ResNet50/DINOv3 extraction, evaluation and reporting continue.

## Required server evidence

The target report must include checkpoint SHA-256, inferred class count, the empty strict-load report, one real BEE24 crop embedding shape/dtype/finite/norm checks, PyTorch/CUDA/device versions, and AMP result. Until those exist, AGW remains `SERVER_VALIDATION_PENDING`; if strict compatibility fails it becomes `BLOCKED_TOPIC_AGW_COMPATIBILITY`.
