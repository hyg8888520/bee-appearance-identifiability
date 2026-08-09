# H2 observation-reliability protocol

## Research boundary

H2 asks whether observation conditions predict appearance-retrieval failure and whether an unconditional trajectory-template update turns a transient bad observation into persistent errors. It is a diagnosis experiment on the frozen `development_validation` split. It does not implement RAM-Bee, a detector, a tracker, or final-test evaluation.

The input is the completed H1 observation manifest and its frozen split. H2 refuses to rebuild that manifest, refuses videos outside development validation, checks the frozen manifest SHA-256 for a full run, and requires a different output directory. `allow_subset: true` is permitted only for an explicitly selected smoke subset.

## Pre-registered interventions

The default configuration evaluates the same observation IDs under:

| Variant | Expansion | Input | Pixel intervention |
|---|---:|---:|---|
| `context_e000_raw_i224` | 0% | 224 | Raw rectangular GT crop |
| `context_e020_raw_i224` | 20% | 224 | H1 primary view; eligible for strict H1-cache reuse |
| `context_e050_raw_i224` | 50% | 224 | More context |
| `context_e020_foreground_i224` | 20% | 224 | Keep the rectangular GT box; neutralize surrounding context |
| `context_e020_only_i224` | 20% | 224 | Neutralize the rectangular GT box; retain surrounding context |
| `context_e020_blur_i224` | 20% | 224 | Deterministic Gaussian-blur intervention |
| `context_e020_raw_i256` | 20% | 256 | Input-size control |

The neutral value is fixed RGB `(127,127,127)` and is part of the cache signature. “Foreground” means the original rectangular GT box, not a segmentation mask. “Context only” is therefore a control for information outside that rectangle, not proof that a model uses background causally.

Each model/variant cache fingerprints the completed H1 manifest, the exact selected observation-ID list, model/checkpoint/transform, intervention, input size, AMP, and neutral value. Complete H1 `20%/224/raw` caches are reused only when all signature fields and shard observation IDs match. Other variants are written atomically and resumed shard by shard.

## Outcome-blind observation signals

`h2-signals` computes signals before reading retrieval outcomes:

- raw bbox area, width, height, aspect ratio, crop/context fractions;
- resized bbox size and approximate ViT-S/16 patch-area coverage;
- variance of a discrete grayscale Laplacian and gradient energy; cross-observation reliability bins use the raw GT rectangle after a fixed 64×64 bilinear resize, while intervention-view values are retained separately;
- structure-tensor dominant orientation and coherence;
- same-frame maximum GT-box IoU, overlap count, nearest-center distance, and neighbour counts;
- image-boundary contact and intervention clipping;
- declared MOT visibility when present;
- track observation count, duration, age, remaining duration, and age fraction;
- optional manual blur, occlusion, and pose annotations.

Important naming boundaries:

- Laplacian variance is a sharpness proxy, not a calibrated motion-blur label.
- Dominant gradient orientation is an image-structure proxy, not bee pose ground truth.
- GT-box IoU is an overlap/occlusion proxy, not an occlusion annotation.
- Approximate patch coverage is literal only for patch-based models; for CNNs it is an input-geometry descriptor.

An optional external manual CSV may contain:

```text
observation_id,manual_blur,manual_occlusion,manual_pose_deg,manual_notes
```

Blur and occlusion are in `[0,1]`; pose is in `[0,360)`. The file stays outside Git. Labels must be defined without reading model outcomes.

## Query diagnostics

For every base model, intervention, H1 delta, and query, H2 retains the H1 Full/Hard gallery rules, strict cosine tie failure, positive similarity, maximum negative similarity, and margin. It adds:

- query observation signals;
- query/positive orientation, bbox-scale, and sharpness changes;
- manual query/positive pose change when external pose labels are supplied;
- similarity to the normalized mean of up to five prior same-identity embeddings;
- history outlier score `1 - history_similarity`.

Factor bins are computed from unique query observations without reading correctness or margin. Default continuous bins are quartiles; boolean factors retain `true/false`. The same definition is used for every delta and reported with sample count, Rank-1, margin, positive similarity, per-video macro mean, and population standard deviation.

Predictiveness reports tie-aware ROC AUC for failure and Spearman correlation between the predeclared unreliability direction and negative margin. These are descriptive diagnostics. Adjacent query frames are not treated as independent evidence: Rank-1 uncertainty is bootstrapped separately by video (macro) and composite identity (pooled micro), with deterministic seeds.

## Controlled template-contamination experiment

The experiment uses only fixed GT identity tracks:

1. Find the first consecutive, non-low-quality window of `trusted_history_length` observations.
2. Build a normalized mean template.
3. Inject one of three controlled events when available:
   - a naturally observed low-quality same-identity embedding;
   - the configured Gaussian-blur embedding of a same-identity observation;
   - the spatially nearest different identity in the same frame.
4. Apply the ordinary unconditional EMA update with configured `ema_alpha`.
5. Compare against a matched control: skip a naturally low-quality event, substitute the original unblurred feature for synthetic blur, or substitute the correct same-frame identity for a wrong-identity injection.
6. At later true observations, record positive-similarity degradation and induced Rank-1 errors, then update both templates with the true observation.
7. Record the first step that returns within `recovery_tolerance` and restores Rank-1, or mark the event unrecovered within `recovery_horizon`.

The low-quality rule uses only signal quantiles, clipping, and optional manual labels—never model correctness. This controlled GT-track result may establish a memory-contamination mechanism; it does not establish an end-to-end tracker improvement.

## Model interpretation

ResNet50 and frozen DINOv3 are the clean generic comparison. The official TOPIC BEE AGW checkpoint remains:

```text
TRAINING_OVERLAP_UNCONFIRMED
REFERENCE_ONLY
NOT_A_GENERALIZATION_BASELINE
```

Its H2 trend may be shown as a task-specific reference, but it is excluded from leakage-free model ranking. A split-clean AGW is a separate future training experiment.

## Artifacts

A completed H2 report writes:

- `h2_summary.csv`;
- `h2_observation_signals.csv` and `h2_signal_metadata.json`;
- `h2_query_diagnostics.csv`;
- `h2_context_ablation.csv` and `h2_paired_ablation.csv`;
- `h2_factor_summary.csv`, `h2_predictiveness.csv`, and `h2_factor_definitions.json`;
- `h2_cluster_bootstrap.csv` and paired intervention differences in `h2_paired_bootstrap.csv`;
- `h2_diagnostic_cases.csv` plus local diagnostic panels for low-sharpness, overlap, history-outlier, orientation-change, clipping, and intervention outcome reversals;
- `h2_memory_events.csv`, `h2_memory_trajectories.csv`, `h2_memory_summary.csv`, and `h2_memory_metadata.json`;
- `h2_run_metadata.json` and `h2_resolved_config.yaml`;
- `h2_logs/` and `h2_figures/`.

All artifacts, features, annotations, data, crops, and checkpoints remain outside Git. Until executed on the real server, H2 results are `SERVER_VALIDATION_PENDING`.
