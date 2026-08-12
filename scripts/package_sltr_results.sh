#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SLTR_OUTPUT_ROOT ARCHIVE_PATH" >&2; exit 2
fi
OUTPUT_ROOT="$1"; ARCHIVE_PATH="$2"
REQUIRED=(sltr_event_counterfactuals.csv sltr_feature_schema.json sltr_oof_predictions.csv sltr_selector_models.json sltr_fit_decision.json sltr_assignments.csv sltr_per_video_metrics.csv sltr_summary.csv sltr_paired_video_metrics.csv sltr_interventions.csv sltr_intervention_metrics.csv sltr_failure_cases.csv sltr_method_decision.json sltr_run_metadata.json sltr_resolved_config.yaml sltr_result_guide.md sltr_artifact_signatures.json)
for name in "${REQUIRED[@]}"; do
  [[ -f "${OUTPUT_ROOT}/${name}" ]] || { echo "SLTR result is incomplete; missing ${OUTPUT_ROOT}/${name}" >&2; exit 2; }
done
tar -czf "${ARCHIVE_PATH}" -C "${OUTPUT_ROOT}" "${REQUIRED[@]}" logs figures
