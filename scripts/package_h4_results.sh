#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 H4_RESULT_DIR ARCHIVE_DIR" >&2
  exit 2
fi

RESULT_DIR="$(realpath -e "$1")"
ARCHIVE_DIR="$(realpath -m "$2")"
if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "H4 result directory does not exist: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${ARCHIVE_DIR}"

CORE_TMP="${ARCHIVE_DIR}/.h4-analysis-core.tar.gz.tmp"
TABLE_TMP="${ARCHIVE_DIR}/.h4-analysis-tables.tar.gz.tmp"
ASSIGN_TMP="${ARCHIVE_DIR}/.h4-assignments.csv.gz.tmp"

tar -czf "${CORE_TMP}" \
  --exclude='h4_assignments.csv' \
  --exclude='h4_recoverability_baseline_assignments.csv' \
  --exclude='h4_recoverability_events.csv' \
  --exclude='h4_mot_results' \
  --exclude='h4_work' \
  -C "${RESULT_DIR}" .

tar -czf "${TABLE_TMP}" \
  -C "${RESULT_DIR}" \
  h4_recoverability_events.csv \
  h4_recoverability_baseline_assignments.csv \
  h4_conflict_events.csv \
  h4_per_video_metrics.csv \
  h4_summary.csv \
  h4_paired_video_metrics.csv \
  h4_failure_cases.csv

gzip -c "${RESULT_DIR}/h4_assignments.csv" > "${ASSIGN_TMP}"
mv -f "${CORE_TMP}" "${ARCHIVE_DIR}/h4-analysis-core.tar.gz"
mv -f "${TABLE_TMP}" "${ARCHIVE_DIR}/h4-analysis-tables.tar.gz"
mv -f "${ASSIGN_TMP}" "${ARCHIVE_DIR}/h4-assignments.csv.gz"

echo "Created:"
echo "  ${ARCHIVE_DIR}/h4-analysis-core.tar.gz"
echo "  ${ARCHIVE_DIR}/h4-analysis-tables.tar.gz"
echo "  ${ARCHIVE_DIR}/h4-assignments.csv.gz"
