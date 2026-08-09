# TOPIC BEE AGW modern-PyTorch compatibility record

Status: official artifact/config mismatch confirmed on the target server. The
signature-gated adaptation remains `SERVER_VALIDATION_PENDING` until strict loading
and real-crop inference are rerun successfully.

## Scope

Only TOPICTrack's BEE24 appearance baseline at
`fast-reid/configs/bee/AGW_S50.yml` is in scope. The detector, YOLOX, tracker and
official PyTorch 1.8/CUDA 11.1 environment are deliberately excluded. Runtime uses
the FastReID copy within pinned TOPICTrack commit
`e7b260f41a92fbe94419ce448aa7b53fe80a6814` and a separate modern Python 3.11 /
PyTorch 2.7.1 CUDA 12.8 environment.

## Minimal adaptations

1. Temporarily add `<topictrack>/fast-reid` to Python import lookup; no source is
   copied or patched in place.
2. Merge the official BEE `AGW_S50.yml` first. The published BEE checkpoint was
   observed to contain a standard ResNet-50 plus Non-local backbone even though this
   YAML selects ResNeSt-50. Only the unambiguous checkpoint signature
   (`backbone.conv1.weight`, `backbone.layer1.0.conv2.weight`, and
   `backbone.NL_2.0.theta.weight`/`backbone.NL_3.0.theta.weight`, with no ResNeSt
   markers) activates
   `MODEL.BACKBONE.NAME=build_resnet_backbone`. GeM, BN neck, Non-local, and all
   other Base-AGW settings remain unchanged. A true ResNeSt checkpoint continues to
   use the unmodified YAML; an unknown or mixed signature is blocked.
3. Override `MODEL.DEVICE`, `MODEL.WEIGHTS` (to prevent an implicit second load),
   `MODEL.BACKBONE.PRETRAIN=False`, inferred `MODEL.HEADS.NUM_CLASSES`, and
   `INPUT.SIZE_TEST`.
4. Infer classifier count from the unique `heads.weight` tensor in the checkpoint.
5. Compare every model/checkpoint key and shape. Any missing, unexpected or
   shape-mismatched entry produces a complete report and blocks AGW. Only an empty
   report proceeds to `load_state_dict(strict=True)`.
6. Pass RGB float32 in 0–255 range because FastReID's Baseline applies configured
   pixel mean/std internally.
7. Replace the official 384×384 test size with the common H1 224×224 protocol;
   256×256 is a supported sensitivity setting. This is an intentional comparability
   override, not the official AGW test protocol.

This is not the independent torchvision ResNet50 baseline and can never be reported
as that model. It is the architecture encoded by TOPICTrack's published BEE AGW
checkpoint, built through TOPICTrack's bundled FastReID with its AGW head and
Non-local settings. Strict loading remains mandatory. Failure is reported as
`BLOCKED_TOPIC_AGW_COMPATIBILITY`; ResNet50/DINOv3 extraction, evaluation and
reporting continue.

## Target-server evidence that triggered the adaptation

The checkpoint downloaded from TOPICTrack's published Google Drive model link was
observed to have this mutually exclusive signature:

- present: `backbone.conv1.weight`, `backbone.layer1.0.conv2.weight`,
  `backbone.NL_2.0.theta.weight`;
- absent: `backbone.conv1.0.weight`, `backbone.layer1.0.conv2.conv.weight`.

The initial strict comparison against the unmodified ResNeSt model therefore ended
in `BLOCKED_TOPIC_AGW_COMPATIBILITY`, including structural downsample shape
mismatches. The checkpoint SHA-256 must be retained in server-generated run
metadata; no weight file is stored in this repository.

## Required server evidence

The target report must include checkpoint SHA-256, inferred class count, the empty
strict-load report, one real BEE24 crop embedding shape/dtype/finite/norm checks,
PyTorch/CUDA/device versions, and AMP result. Until those exist, the adaptation is
`SERVER_VALIDATION_PENDING`; if strict compatibility still fails it remains
`BLOCKED_TOPIC_AGW_COMPATIBILITY`.
