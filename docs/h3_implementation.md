# H3 reliability-aware association implementation

## Scope

H3 implements the first frozen stage, `gt_detection_boxes`, on exact BEE24 GT detections. It creates online predicted track IDs without reading GT identity during association. GT identity is attached only after assignment for evaluation.

This stage answers whether causal observation reliability improves association and memory management when detector FP/FN are removed. It is not a detector benchmark and not yet the frozen `fixed_detector_boxes` stage.

## Split and leakage boundary

- `project_train`: only partition allowed to fit reliability CDFs and thresholds;
- `development_validation`: tracking variants, ablations, error analysis and GO/STOP;
- `final_test`: inaccessible until every freeze-gate item is complete;
- identity and isolation units remain `(video_id, track_id)` and `video_id`;
- the H1 manifest SHA-256 must equal the provenance hash in the frozen project split;
- every feature cache, signal table and threshold artifact carries protocol/split/manifest hashes;
- H3 refuses a threshold artifact produced with a different model cache.

The H2.5 development Q75 values are diagnostic evidence only and are never copied into the H3 method artifact.

## Causal reliability

For a proposed track–detection pair, H3 uses only the track state before the current frame and current detection information:

1. `identity_history_outlier = 1 - cosine(track_memory, detection_embedding)`;
2. absolute log bbox-area ratio relative to the track's previous observation;
3. axial structure-orientation change modulo 180 degrees;
4. absolute log Laplacian-ratio change;
5. maximum of current crowding-count and overlap-risk percentiles.

Static bbox area and static Laplacian are retained only to compute changes and are not direct reliability features. Each risk is mapped through a project-train empirical CDF represented by deterministic quantile knots:

```text
risk = max(train_CDF(feature_1), ..., train_CDF(feature_5))
q = clip(1 - risk, min_reliability, 1)
```

## Frozen association variants

| Variant | Association | Memory update |
|---|---|---|
| `baseline_association` | appearance + constant-velocity motion | unconditional EMA |
| `selective_memory_update` | same as baseline | EMA only when `q >= update_gate` |
| `reliability_weighted_association` | appearance weight multiplied by `q` | unconditional EMA |
| `full_ram_bee` | reliability-weighted appearance | hard gate, then `alpha = alpha_max × q` |

All variants share the same detections, embeddings, motion model, assignment solver, gates and initialization. The assignment solver is an exact dependency-free Hungarian implementation with explicit unmatched dummy choices.

## Metrics

GT detections are passed directly to the tracker, so every predicted detection has an exact source-GT correspondence. The implementation computes global IDF1 assignment, AssA, HOTA and CLEAR-style IDSW using the definitions audited against TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`.

In this stage:

- `DetA = 1` by construction;
- `HOTA = sqrt(AssA)`;
- `Frag = 0` because no GT detection is missing; an identity change is counted as IDSW, not fragmentation;
- fixed-detector HOTA/Frag remain pending and must not be inferred from GT-box results.

The report includes per-video and micro-pooled metrics, macro mean/population std, paired video differences, and video-cluster bootstrap intervals. Six videos make the intervals descriptive rather than definitive.

## Outputs

| Artifact | Purpose |
|---|---|
| `h3_observation_signals.csv` | train/dev outcome-blind signal cache |
| `h3_signal_metadata.json` | signal fingerprint and split audit |
| `h3_cache_locations.json` | model cache registry |
| `h3_thresholds.json` | project-train-only reliability artifact |
| `h3_assignments.csv` | per-detection predicted ID and diagnostics |
| `h3_per_video_metrics.csv` | per-video primary metrics |
| `h3_summary.csv` | pooled and macro metrics |
| `h3_paired_video_metrics.csv` | each method minus baseline |
| `h3_video_cluster_bootstrap.csv` | paired video-cluster intervals |
| `h3_mot_results/` | MOTChallenge-format tracker text |
| `h3_tracking_metadata.json` | tracker/cache/threshold provenance |
| `h3_run_metadata.json` | claims, environment and next gate |
| `h3_resolved_config.yaml`, `h3_logs/` | resolved run record |

## Interpretation gate

The GT stage can justify proceeding to fixed detector boxes only when the primary backbones show a stable development benefit that is not driven by one video. It cannot unlock final test. The frozen final-test gate still requires method, configuration, checkpoints, threshold artifact and reporting code to be frozen, plus signed-off GT and fixed-detector development results.
