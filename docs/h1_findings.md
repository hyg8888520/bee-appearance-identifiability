# H1 frozen findings

## Decision record

The first full BEE24 H1 run is complete and its core artifacts have been cross-checked. It is now a **frozen development benchmark**, not the final paper test. The six evaluated videos are permanently assigned to `development_validation`; the official BEE24 test videos are locked until the method and configuration are frozen. The authoritative partition is [`configs/splits/project_split.yaml`](../configs/splits/project_split.yaml).

The official TOPIC BEE AGW checkpoint is reported only as `REFERENCE_ONLY`. It is BEE24-trained, while the exact mapping from its Re-ID training crops to the original MOT videos has not been established. Its score must not be used as evidence of leakage-free generalization. See [`agw_training_overlap_audit.md`](agw_training_overlap_audit.md).

## Completed run

| Item | Frozen value |
|---|---|
| Evaluation role | Development validation |
| Videos | BEE24-03, BEE24-09, BEE24-17, BEE24-22, BEE24-25, BEE24-26 |
| Valid query observations | 4,038 |
| Composite identities | 164 |
| Deltas | 1, 5, 10, 25 frames |
| Main metric | Full-gallery Rank-1; cosine ties count as failure |
| Project commit | `9a92b27dce432ad6ce79e0b71dff4a801a7b0afe` |
| Observation manifest SHA-256 | `aec25bde8c9a023d78a96c85dbdaab4740f2ce5ebc08d01b53f1e2f334691bac` |
| Runtime | PyTorch 2.7.1+cu128, NVIDIA RTX 4090, AMP enabled |
| Benchmark interval | 2026-08-09 06:51:03 to 08:17:31 UTC |

The run metadata records a real RTX 4090 execution. The core analysis archive does not contain the standalone `gpu_smoke.json`, so that separate acceptance artifact remains `SERVER_VALIDATION_PENDING`; this does not turn the completed H1 scores back into synthetic results.

## Primary results

Full-gallery Rank-1 on the frozen development validation split:

| Model | delta=1 | delta=5 | delta=10 | delta=25 | Interpretation role |
|---|---:|---:|---:|---:|---|
| ImageNet ResNet50 | 71.75% | 44.24% | 36.22% | 33.37% | Clean generic baseline |
| Frozen DINOv3 ViT-S/16 | 78.34% | 48.99% | 40.79% | 32.90% | Clean generic representation |
| Official TOPIC BEE AGW | 90.16% | 60.18% | 47.96% | 38.98% | `REFERENCE_ONLY` |

The number of queries with a positive at the target frame falls from 3,873 at delta 1 to 1,711 at delta 25. Consequently, long-delta Rank-1 is conditional on the identity still being visible; disappearance is not counted as a model error.

## What H1 establishes

1. Changing the backbone improves short-term appearance matching, but the gain is not stable at longer gaps. DINOv3 exceeds ResNet50 by 6.58, 4.75, and 4.58 percentage points at deltas 1, 5, and 10, then trails by 0.47 points at delta 25.
2. All three representations degrade strongly with time. At delta 25, all three models fail on 45.41% of valid paired queries, while all three succeed on only 19.40%.
3. Mean positive similarity falls with delta while the strongest hard-negative similarity is comparatively stable. The shared failure is therefore consistent with loss of same-identity appearance consistency, not merely a weak choice of backbone.
4. Video-to-video variation is large, and size-quartile performance is not monotonic. Context, pose, motion blur, overlap, occlusion, and trajectory duration must be investigated jointly.

These are descriptive results on GT-box crops. They support the research hypothesis that observation-dependent reliability and memory update deserve explicit modeling. They do **not** yet prove that a proposed tracker reduces ID switches.

## Data audit relevant to interpretation

- Source observations before duplicate filtering: 278,678.
- Conflicting `(video, frame, track_id)` keys: 994; 1,991 rows were conservatively excluded (0.714% of source observations).
- Conflict sequences: BEE24-14, BEE24-15, BEE24-33, and BEE24-35; none is in development validation.
- Invalid or empty crops skipped: 169.
- Final valid observations: 276,518.
- BEE24-33 and BEE24-35 used audited image-based sequence metadata inference; neither is in development validation.

## External evidence hashes

The artifacts stay outside Git. Their received SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `h1-analysis-core.tar.gz` | `45B611B0C7728602A28CD04704BE5FFDF80CEA2F4986DD4CAE28D123D8EA515F` |
| `h1-failure-figures.tar.gz` | `000A82E78F6B380EC833B718059231E641448AF10C134D7E11F4DA9968DB8D39` |
| `h1-query-results.csv.gz` | `A5BB09BA5D88BEC2F03DB0090FE87ED254E5F0D11970ECEEC45A5CD16417606F` |

The query archive contains 48,456 rows (`4,038 queries × 4 deltas × 3 models`), no duplicate model/delta/query key, and 35,052 valid query-model-delta rows. Recomputed aggregate metrics agree with `summary.csv` to floating-point precision.

## Status after freezing

| Work item | Status |
|---|---|
| First full H1 development run | `COMPLETED_AND_AUDITED` |
| Fixed project train/development-validation split | `FROZEN` |
| Official AGW role | `TRAINING_OVERLAP_UNCONFIRMED`, `REFERENCE_ONLY` |
| Split-clean AGW trained only on project train | `SERVER_VALIDATION_PENDING` |
| H2 reliability-factor annotations and diagnostics | `CODE_COMPLETE_SERVER_VALIDATION_PENDING` |
| Final locked BEE24 test evaluation | `NOT_RUN_DO_NOT_PEEK` |
| Standalone GPU smoke report archived with evidence | `SERVER_VALIDATION_PENDING` |

No current H1 output should be overwritten. New experiments must use a new external output directory and retain their resolved config, manifest hash, split ID, checkpoint hash, and feature fingerprint.
