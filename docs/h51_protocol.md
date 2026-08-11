# H5.1 closed-loop diagnostic protocol

H5.1 is an independent, post-hoc diagnostic defined after the H5 development GT-box gate returned
`STOP_OR_REVISE_BEETRACKQUERY`. It is not an H5.2 fix, retraining run, threshold search, method
improvement, or final-test experiment. Its immutable lock is
`configs/h51_protocol.lock.yaml`; every threshold is `diagnostic_not_confirmatory`.

## Inputs and isolation

The only method weights are the two completed H5 `final.pt` files for ResNet50 and DINOv3. H5.1
checks each checkpoint SHA-256, training fingerprint/signature, H5 run/training/tracking metadata,
H5/H3 protocol hashes, frozen split, manifest, source cache fingerprint, and `final_test_read=false`.
The H5 and H3 roots are read-only and external to the new H5.1 output root. No checkpoint is written,
modified, or trained. `project_train` may be read only to construct the runtime gradient-probe
transition inputs. Teacher-forced offline audit, causal replay, and all reported identity metrics use
only `development_validation`. Final test is never read or unlocked.

## Audits

1. Runtime gradient coverage recreates the checkpoint-pinned H5 v2 `_batched_transition_loss` on
   the current branch, calls backward, and
   records every parameter's gradient presence, finiteness, norm, and objective use. The known six
   `memory_attention`/`memory_norm` parameters with no gradient make `method_ready=false` but do not
   abort diagnostics. Readiness additionally requires every trainable gradient to be present, finite,
   and above the locked zero threshold; any finite zero or non-finite gradient is reported and stops
   readiness without automatic causal attribution. Exact zero norms of individual non-memory
   parameters can vary with CPU/GPU kernels and reduction order, so the smoke test reconciles the
   complete finite-zero list in the gradient CSV against the path summary rather than requiring a
   particular named parameter to be zero. This does not weaken the core finding: the six
   `memory_attention`/`memory_norm` parameters are missing gradients and independently force STOP.
2. Runtime hooks distinguish the teacher-forced training path, the GT-conditioned offline one-step
   audit, and deployable causal rollout. The offline table is marked `offline_gt_audit=true` and is
   never presented as a tracker result.
3. The causal replay preserves H5 tracker-v4 score, motion gate, Hungarian assignment, query update,
   memory selection, reliability gate, and track creation semantics. GT identity is absent from every
   inference decision; it is attached only after decisions for metric/audit labels. Replay jobs use
   diagnostic schema v2, and the completed assignments must match the hashed source H5 assignments
   observation-by-observation before any result is reported. Numeric replay equivalence uses an
   absolute tolerance of `5e-6` (with relative tolerance `1e-6`), set from the RTX server's measured
   `2.81e-6` maximum absolute difference plus safety margin. This accommodates floating-point
   execution-order variation only: identity and update decisions remain exactly equal, missing or
   non-finite values remain invalid, and a difference outside tolerance still fails closed. It is not
   tolerance for approximate assignment decisions.
4. Every observation receives one exhaustive rejection reason: `matched`, `no_active_track`,
   `no_motion_valid_candidate`, `score_below_threshold`, or `assignment_conflict`. Candidate samples
   are bounded, while candidate counts and reason totals remain complete.
5. The fuse checks gradient coverage, new-track/prediction collapse, gate degeneracy, and the
   teacher-forced-versus-rollout gap. It emits only `STOP_H51_IMPLEMENTATION_NOT_READY` or
   `READY_FOR_H52_PROTOCOL_DESIGN`; neither status permits final-test access.

The full diagnostic locks `max_teacher_forced_rows=250000` and
`candidate_sample_per_observation=3`. A subset smoke may reduce only the teacher-forced cap to the
locked bounded override of 10000 rows; candidate sampling remains 3. Subset output is labelled
`scope=bounded_subset_diagnostic`, cannot emit a readiness decision, and remains
`diagnostic_not_confirmatory` just like the full post-hoc diagnostic.

Synthetic smoke is test-only. Correctly detecting the six missing gradients counts as diagnostic
pipeline success while the method remains not ready. Real RTX 4090/BEE24 validation remains
`SERVER_VALIDATION_PENDING` until separately executed.

The source H5 checkpoint signature must report `implementation=beeid.h5:v2-batched-transitions`.
The legacy H5 metadata records a git commit but no source diff or training-binary digest. Therefore
the gradient probe is runtime evidence about the pinned H5 v2 loss recreation, not direct forensic
attestation of the exact historical server binary.
