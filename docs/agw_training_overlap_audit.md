# TOPIC BEE AGW training-overlap audit

## Current ruling

```text
TRAINING_OVERLAP_UNCONFIRMED
REFERENCE_ONLY
NOT_A_GENERALIZATION_BASELINE
```

This is the conservative ruling for the official TOPICTrack `Model (Re-ID)` checkpoint used in H1. The checkpoint is a BEE24-specific Re-ID model, and H1 development validation was drawn from the original BEE24 train videos. Until the Re-ID crop-to-MOT-video mapping is proven, a train/evaluation overlap cannot be excluded.

The H1 checkpoint loaded strictly with no missing, unexpected, or shape-mismatched keys. That establishes implementation compatibility, not split cleanliness.

## Evidence currently available

| Evidence | Finding |
|---|---|
| Pinned TOPICTrack revision | [`e7b260f41a92fbe94419ce448aa7b53fe80a6814`](https://github.com/holmescao/TOPICTrack/tree/e7b260f41a92fbe94419ce448aa7b53fe80a6814) |
| Published configuration | `fast-reid/configs/bee/AGW_S50.yml`, with BEE as train and test dataset |
| Dataset loader | `fast-reid/fastreid/data/datasets/bee.py` reads `bounding_box_train`, `query`, and `bounding_box_test` |
| H1 checkpoint SHA-256 | `815ec60ec611075ed5fe57b4dc09c9b0a20111f1442134b0d0f8964963ae3d86` |
| H1 strict-load report | missing `[]`, unexpected `[]`, shape mismatch `[]` |
| Development videos | BEE24-03, BEE24-09, BEE24-17, BEE24-22, BEE24-25, BEE24-26 |
| Missing evidence | Authoritative mapping from every `bounding_box_train` crop to original BEE24 video, frame, and identity |

Repository documentation establishes that the checkpoint is trained for BEE24 Re-ID, but does not establish that the six development videos are absent from `bounding_box_train`. A statement such as “AGW used all BEE24 videos” must also remain an unverified claim until the mapping or training manifest is obtained.

## Interpretation policy

| Audit outcome | Required label | Permitted interpretation |
|---|---|---|
| Any development image/video/identity is found in AGW training | `TRAIN_OVERLAP_REFERENCE_ONLY` | Same-domain reference or optimistic ceiling only |
| Complete, authoritative mapping proves video-level disjointness | `SPLIT_CLEAN_CONFIRMED` | May be compared as a generalization baseline, subject to protocol differences |
| Mapping remains unavailable or incomplete | `TRAINING_OVERLAP_UNCONFIRMED`, `REFERENCE_ONLY` | Keep score in a separate reference row; never use it to rank clean models |

The current H1 report follows the third row. No result is discarded: the AGW score remains useful as evidence that even a strong task-specific representation degrades with temporal gap, but it cannot support a leakage-free superiority claim.

## Audit procedure

1. Obtain the exact BEE24-ReID archive and any generation script or split manifest used for the Model Zoo checkpoint; record file hashes and provenance without committing data.
2. Inventory `bounding_box_train`, `query`, and `bounding_box_test`, preserving filenames, relative paths, parsed identity/camera fields, dimensions, and SHA-256 values.
3. Recover an authoritative mapping to original MOT `(video_id, frame_id, track_id)`. Prefer author-provided metadata or the original conversion script over filename inference.
4. Compare all mapped training rows against the frozen partitions in [`configs/splits/project_split.yaml`](../configs/splits/project_split.yaml), with video overlap as the primary leakage criterion.
5. If metadata is unavailable, use exact pixel hashes followed by perceptual matching against candidate GT crops only as supporting evidence; manually review matches and record thresholds. A failed image match does not prove disjointness.
6. Save a compact audit report containing counts, mappings, hashes, unmatched rows, and the final ruling. Keep crops and the Re-ID archive outside Git.

## Required clean replacement for the primary comparison

A primary task-specific baseline must be a **split-clean AGW**:

- initialize from ImageNet or another source without BEE24 identity supervision;
- train only on `project_train` videos from the frozen split;
- exclude all development-validation and final-test frames, crops, identities, and caches;
- use development validation for early stopping and hyperparameter selection;
- record the training video list, code commit, resolved config, random seed, initialization hash, and final checkpoint SHA-256;
- run the final test only once after method freeze.

Until this model exists, the clean backbone comparison is ResNet50 versus DINOv3, with official AGW shown separately as a BEE24-trained reference.
