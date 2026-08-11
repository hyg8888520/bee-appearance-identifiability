#!/usr/bin/env bash
set -euo pipefail

CPU_THREAD_DEFAULT="${BEEID_H51_CPU_THREADS:-16}"
if [[ ! "${CPU_THREAD_DEFAULT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEEID_H51_CPU_THREADS must be a positive integer when set." >&2
  exit 2
fi
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS="${CPU_THREAD_DEFAULT}"
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 H51_OUTPUT_ROOT ARCHIVE_PATH" >&2
  exit 2
fi
OUTPUT_ROOT="$1"
ARCHIVE_PATH="$2"
REQUIRED=(
  h51_gradient_coverage.csv h51_path_audit.json h51_teacher_forced_queries.csv
  h51_rollout_observation_diagnostics.csv h51_frame_diagnostics.csv
  h51_variant_summary.csv h51_rejection_summary.csv h51_decision.json
  h51_replay_equivalence.json
  h51_run_metadata.json h51_result_guide.md h51_artifact_signatures.json
)
for name in "${REQUIRED[@]}"; do
  if [[ ! -f "${OUTPUT_ROOT}/${name}" ]]; then
    echo "H5.1 result is incomplete; missing ${OUTPUT_ROOT}/${name}" >&2
    exit 2
  fi
done
tar -czf "${ARCHIVE_PATH}" -C "${OUTPUT_ROOT}" "${REQUIRED[@]}" logs
