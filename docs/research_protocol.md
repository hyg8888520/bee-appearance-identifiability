# H1 research protocol

## Data contract

BEE24 is read as MOTChallenge trees at `train|test/<video>/`. When present, `seqinfo.ini` defines sequence length, image dimensions, image directory and extension; `gt/gt.txt` must contain at least frame, track, x, y, width and height. If a declared `imDir` is absent but exactly one canonical `img1/` or `images/` directory exists, that unique directory is used without modifying source metadata. Missing `seqinfo.ini` remains an error by default. The explicit `missing_seqinfo_policy: infer_from_images` mode accepts exactly one canonical image directory, requires one consistent extension, six-digit contiguous frame names starting at 1, readable images with identical dimensions, and derives sequence length and dimensions in memory. Frame rate is recorded as unknown (`0`), and filenames, file sizes, and dimensions form an inventory SHA-256. Every declared or inferred sequence is recorded in `manifests/sequence_metadata_audit.json`. Optional confidence, class, visibility and additional numeric columns are preserved. Duplicate `(split, video, frame, track)` entries are rejected.

The canonical identity is `(video_id, track_id)`; track IDs are never treated as globally unique. Within one video and frame, however, an identity must have exactly one GT box. The default `duplicate_identity_policy: error` enforces the MOT/TrackEval invariant. The explicit `exclude_conflict` policy first scans every selected GT row, then excludes every row belonging to each conflicting `(source_split, video_id, frame, track_id)` key without modifying source GT. Paths, SHA-256 hashes, line numbers, raw rows, counts, and the excluded fraction are written to `manifests/duplicate_identity_audit.json`; the policy is also part of the feature-cache fingerprint. With `bbox_origin: one`, only x and y are shifted by -1. Original floating-point xywh remain in the manifest. Zero/non-positive area and completely outside boxes are errors by default; `invalid_bbox_policy: skip` retains their reason but excludes them from extraction. Partial overlap is clipped and recorded.

The stable CSV explicitly includes `video_id`, `frame_id`, `track_id`, `image_path`, expanded integer crop coordinates `x1,y1,x2,y2`, original bbox width/height and floating xywh, image dimensions, area, expansion ratio and clipping flag, plus split, confidence, class, visibility, center, composite identity, observation ID and skip reason.

For an original box `[x0,y0,x1,y1)`, expansion 0.2 subtracts 0.1 width/height on the low sides and adds 0.1 on the high sides. Bounds are integerized with floor on low sides, ceil on high sides, then clipped to `[0,W]×[0,H]`. Images are opened lazily and crops are never materialized as a separate corpus.

## Split resolution

If no official validation exists, train video IDs are sorted by the pair `(SHA256(f"{seed}:{video_id}"), video_id)`, with seed fixed to 24. The first `max(1, round(0.2*n_train))` videos become project validation. The result and manifest hash are written to `resolved_split.yaml`. The official test split is never used for method selection.

The first completed full H1 run permanently fixes those six videos as development validation. The train/development/final-test partition is versioned in [`configs/splits/project_split.yaml`](../configs/splits/project_split.yaml). Development validation may be used for thresholds, ablations and error analysis, but is no longer an unbiased final test. The official test videos remain locked until the method, training recipe and reporting code are frozen.

## Embeddings

Every extractor returns a finite 2-D float32 matrix with one L2-normalized row per crop.

- ResNet50: torchvision v0.22.1 `ResNet50_Weights.IMAGENET1K_V2`, classification head replaced by Identity, official weights transforms.
- DINOv3: pinned local Hub `dinov3_vits16`; `forward_features()` must contain finite `x_norm_patchtokens` and `x_norm_clstoken` with valid shapes. H1 defaults to patch-token mean. CLS and official intermediate-layer access remain available.
- TOPIC AGW: pinned TOPICTrack BEE `AGW_S50.yml`, ResNeSt-50, GeM and BN neck. Number of classes is inferred from `heads.weight`; checkpoint/model keys and shapes are compared before a strict state load. The common protocol overrides official 384×384 to 224×224 (256 supported).

Strict checkpoint compatibility does not establish split cleanliness. The official BEE AGW checkpoint is `REFERENCE_ONLY` while its Re-ID training-crop mapping is unresolved; the primary supervised Re-ID comparison requires a split-clean AGW trained only on project-train videos. The ruling and evidence requirements are recorded in [`agw_training_overlap_audit.md`](agw_training_overlap_audit.md).

Parameters are frozen and inference-only. AMP state/dtype is part of the cache signature. Random weights and test embeddings are never experimental results.

## Cache contract

The fingerprint covers the observation manifest SHA-256, evaluation split, model/upstream/checkpoint signature, transform details, crop expansion, input size and AMP. Each `.npz` contains Unicode observation IDs and float32 embeddings. Writes use a same-directory temporary file followed by atomic replacement. Resume validates exact IDs, dtype, shape, finite values and norms. Signature disagreement produces another fingerprint; corrupt metadata is refused.

## Retrieval

For every valid query observation at frame `t`, each configured delta targets frame `t+delta` in the same video. A unique same-identity observation is the positive. If absent, the query is counted by reason but excluded from metric denominators.

Full Rank-1 compares the positive against all different-identity observations in the candidate frame. Equality with the best negative is failure. Hard Rank-1 compares the positive with up to five different-identity observations in the candidate frame, selected by ascending squared distance from their original GT centers to the query's original GT center. Without a negative, Hard Rank-1 is not defined and is excluded.

Reported values include Rank-1, Hard Rank-1, positive similarity, maximum hard-negative similarity, margin and query counts per video. Micro metrics pool queries. Macro mean and population standard deviation are computed over video metrics. Q1–Q4 thresholds are the 25/50/75 percentiles of unique valid query raw bbox areas in the evaluation split and are persisted.

## Failure and interpretation

Failure selections cover model correctness combinations, lowest margin, smallest targets and boundary-clipped crops. They are written only below external `output_root`. GT crops measure an appearance upper bound; background/context leakage, expansion bias and detector errors are outside H1. No H1 result alone establishes fewer tracker ID switches.

The frozen first-run evidence and permitted conclusions are recorded in [`h1_findings.md`](h1_findings.md).

## H2 handoff

H2 consumes the completed H1 manifest and caches without changing the frozen retrieval contract. It is restricted to development validation and diagnoses observation-quality/context/history signals plus controlled EMA memory contamination. It does not implement a tracker or read final test. The pre-registered interventions, proxy boundaries, clustered statistics, and artifact contract are defined in [`h2_protocol.md`](h2_protocol.md).

## H2.5 and H3 handoff

H2.5 replaces the unsuccessful static-low-quality contamination rule with outcome-blind dynamic event selection and exactly paired oracle, unconditional, skip, and reliability-weighted updates. It reuses completed H2 primary caches, remains on development validation, and reports both video- and identity-cluster bootstrap intervals. Its frozen definitions and claim limits are in [`h25_protocol.md`](h25_protocol.md).

H3 is locked before tracker implementation in [`h3_frozen_protocol.md`](h3_frozen_protocol.md). Deployable reliability thresholds must be fit on project train; development is reserved for selection and ablation; final test remains unread until method, checkpoints, thresholds, configuration, and reporting code are frozen.
