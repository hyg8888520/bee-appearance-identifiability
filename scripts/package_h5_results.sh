#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 H5_OUTPUT_ROOT ARCHIVE_PATH" >&2
  exit 2
fi
OUTPUT_ROOT="$1"
ARCHIVE_PATH="$2"
for name in h5_run_metadata.json h5_summary.csv h5_per_video_metrics.csv h5_paired_video_metrics.csv h5_reliability_calibration.csv h5_video_cluster_bootstrap.csv h5_failure_cases.csv h5_method_decision.json h5_result_guide.md; do
  if [[ ! -f "${OUTPUT_ROOT}/${name}" ]]; then
    echo "H5 result is incomplete; missing ${OUTPUT_ROOT}/${name}" >&2
    exit 2
  fi
done
tar -czf "${ARCHIVE_PATH}" -C "${OUTPUT_ROOT}" \
  h5_run_metadata.json h5_training_metadata.json h5_tracking_metadata.json \
  h5_summary.csv h5_per_video_metrics.csv h5_paired_video_metrics.csv \
  h5_reliability_calibration.csv h5_video_cluster_bootstrap.csv h5_failure_cases.csv \
  h5_method_decision.json h5_result_guide.md h5_figures
