#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 H41_RESULT_DIR ARCHIVE_DIR" >&2
  exit 2
fi

RESULT_DIR="$(realpath -e "$1")"
ARCHIVE_DIR="$(realpath -m "$2")"
if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "H4.1 result directory does not exist: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${ARCHIVE_DIR}"

CORE_TMP="${ARCHIVE_DIR}/.h41-analysis-core.tar.gz.tmp"
TABLE_TMP="${ARCHIVE_DIR}/.h41-analysis-tables.tar.gz.tmp"
ASSIGN_TMP="${ARCHIVE_DIR}/.h41-assignments.csv.gz.tmp"

tar -czf "${CORE_TMP}" \
  --exclude='h41_assignments.csv' \
  --exclude='h41_recomputed_baseline_assignments.csv' \
  --exclude='h41_exact_recoverability_events.csv' \
  --exclude='h41_event_timelines.csv' \
  --exclude='h41_mot_results' \
  --exclude='h41_work' \
  -C "${RESULT_DIR}" .

TABLE_FILES=(
  h41_exact_recoverability_events.csv
  h41_event_timelines.csv
  h41_recomputed_baseline_assignments.csv
  h41_exact_horizon_summary.csv
  h41_cumulative_summary.csv
  h41_conflict_events.csv
  h41_per_video_metrics.csv
  h41_summary.csv
  h41_paired_video_metrics.csv
  h41_video_cluster_bootstrap.csv
  h41_failure_cases.csv
)
EXISTING_TABLES=()
for name in "${TABLE_FILES[@]}"; do
  if [[ -f "${RESULT_DIR}/${name}" ]]; then
    EXISTING_TABLES+=("${name}")
  fi
done
if [[ ${#EXISTING_TABLES[@]} -eq 0 ]]; then
  echo "No H4.1 analysis tables found in ${RESULT_DIR}" >&2
  exit 2
fi
tar -czf "${TABLE_TMP}" -C "${RESULT_DIR}" "${EXISTING_TABLES[@]}"

mv -f "${CORE_TMP}" "${ARCHIVE_DIR}/h41-analysis-core.tar.gz"
mv -f "${TABLE_TMP}" "${ARCHIVE_DIR}/h41-analysis-tables.tar.gz"
echo "Created:"
echo "  ${ARCHIVE_DIR}/h41-analysis-core.tar.gz"
echo "  ${ARCHIVE_DIR}/h41-analysis-tables.tar.gz"

if [[ -f "${RESULT_DIR}/h41_assignments.csv" ]]; then
  gzip -c "${RESULT_DIR}/h41_assignments.csv" > "${ASSIGN_TMP}"
  mv -f "${ASSIGN_TMP}" "${ARCHIVE_DIR}/h41-assignments.csv.gz"
  echo "  ${ARCHIVE_DIR}/h41-assignments.csv.gz"
else
  echo "Tracking did not run; no assignments archive was created. Audit archives are complete."
fi
