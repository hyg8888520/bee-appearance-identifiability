# SLTR: selective local trajectory repair

SLTR is a frozen, development-only GT-box association experiment. It reuses the H1 manifest and
the completed H3 feature cache and observation signals strictly read-only. The real backbones are
exactly ResNet50 and DINOv3. TOPIC is topical inspiration only: this is not an official TOPIC,
TOPICTrack, AGW, DINOv3, H3, or H4 reproduction.

## Scientific question and falsifiable hypotheses

The question is not whether another backbone or memory network wins. It is whether a causal tracker
can recognize the small subset of local ambiguity events where changing the frozen baseline is more
likely to help than harm. Three claims are tested in order:

1. **Selectability:** project-train group-OOF features predict positive counterfactual utility with
   at least 0.80 precision and at most 0.05 harm, without selecting more than 30% of events.
2. **Specificity:** the learned selector outperforms `frequency_matched_random`; otherwise any gain
   may be explained by intervention frequency rather than event recognition.
3. **Closed-loop safety:** both ResNet50 and DINOv3 reduce IDSW while keeping IDF1 and HOTA within
   0.002 of the frozen immediate baseline and avoiding harm in at least four videos.

Failure at any stage is a valid STOP result. It means the current observable feature set cannot
safely decide when to repair; it does not justify relaxing the threshold on development data.

`immediate_baseline` is the frozen H4 appearance-motion Hungarian reference used by the preceding
experiments. It is a reproducible causal reference, not a claim to reproduce the strongest published
TOPIC/OC-SORT tracker. `always_repair` tests whether the repair operator itself is beneficial;
`frequency_matched_random` tests whether selection is informative; `oracle_selector` is an offline
upper bound only; `learned_selector` is the sole proposed deployable variant.

## Where the design comes from

- TOPIC's parallel motion/appearance conflict handling motivates not letting one cue silently erase
  another, but no TOPIC code or learned module is used here.
- DeconfuseTrack motivates decomposing global association into local confusion subproblems.
- Robust MOT by Marginal Inference motivates relative candidate competition instead of interpreting
  an absolute similarity as universal confidence.
- Tracking by Associating Clips and path-consistency work motivate short multi-frame evidence.
- The one-frame horizon and selective rather than universal intervention come directly from this
  project's H4.1 result: median first recovery lag was one frame, while broad five-frame deferral
  caused collateral IDF1/HOTA loss.

These are sources of components and constraints, not novelty claims. The experiment-specific claim,
if supported, is the risk-controlled combination and its BEE24 failure-mechanism evidence.

At a H4 local one-to-one assignment ambiguity, branch A takes rank-0 immediately and continues
deterministically for one future frame; branch B takes rank-1 immediately and then the same deterministic
rank-0 horizon. The one-frame primary horizon is fixed from H4.1's median first-recovery lag rather
than selected on this experiment's development results. Offline labels first build a unique
GT-to-predicted-ID mapping from the most recent five pre-event observations per identity,
then compute `Δcorrect observations − 2·ΔIDSW` for B relative to A. Labels that lack a unique map
are excluded from fitting and OOF thresholding.

Only project-train labels fit the deterministic NumPy L2 logistic selector, using leave-one-video-out
OOF predictions. Development is never used for fit or threshold selection. A threshold is eligible
only with precision ≥ .80, harm ≤ .05, at least 10 selected events from at least 3 videos, and an
intervention fraction ≤ .30. It maximizes coverage, then utility, then chooses the higher threshold.
Fit stops fail-closed below 30 events, 4 videos, or 10 positives per backbone. Features must be finite
and observable; GT, identity, label, utility and correct fields are rejected.

The deployable learned selector uses exact A fallback below threshold and the selected branch state is
the only state that continues the trajectory. `oracle_selector` is diagnostic only. Development GO
requires positive IDSW reduction for each backbone, IDF1/HOTA non-inferiority within .002, harm ≤ .05,
and at least four non-harmed videos. Final test is locked: zero reads and zero development runs.

Primary artifacts are `sltr_event_counterfactuals.csv`, `sltr_feature_schema.json`,
`sltr_oof_predictions.csv`, `sltr_selector_models.json`, `sltr_fit_decision.json`,
`sltr_assignments.csv`, `sltr_per_video_metrics.csv`, `sltr_summary.csv`,
`sltr_paired_video_metrics.csv`, `sltr_intervention_metrics.csv`,
`sltr_failure_cases.csv`, `sltr_method_decision.json`, and `sltr_run_metadata.json`.

Run `beeid sltr-synthetic-smoke` first. The synthetic encoder and its temporary relaxed lock are
test-only, never real models/results, and never read final test. Then use `scripts/run_sltr.sh`; atomic,
signed video jobs in `work/` can be resumed with `scripts/resume_sltr.sh`.
