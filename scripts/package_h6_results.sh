#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-}"
ARCHIVE="${2:-}"
if [[ -z "$OUTPUT_ROOT" || -z "$ARCHIVE" ]]; then
  echo "Usage: bash scripts/package_h6_results.sh /path/to/h6-output /path/to/h6-results.tar.gz" >&2
  exit 2
fi
OUTPUT_ROOT="$(realpath "$OUTPUT_ROOT")"
ARCHIVE="$(realpath -m "$ARCHIVE")"
for required in h6_run_metadata.json h6_method_decision.json h6_summary.csv h6_per_video_metrics.csv h6_trajectory_decisions.csv h6_artifact_signatures.json h6_report.md; do
  if [[ ! -f "$OUTPUT_ROOT/$required" ]]; then
    echo "H6 result is incomplete; missing $OUTPUT_ROOT/$required" >&2
    exit 2
  fi
done
mkdir -p "$(dirname "$ARCHIVE")"
EXCLUDES=(
  --exclude='h6_assignments.csv'
  --exclude='h6_trajectory_diagnostics.csv'
  --exclude='work'
  --exclude='h6_checkpoints'
)
if [[ "$ARCHIVE" == "$OUTPUT_ROOT/"* ]]; then
  RELATIVE_ARCHIVE="${ARCHIVE#"$OUTPUT_ROOT/"}"
  EXCLUDES+=(--exclude="$RELATIVE_ARCHIVE" --exclude="./$RELATIVE_ARCHIVE")
fi
tar -czf "$ARCHIVE" "${EXCLUDES[@]}" -C "$OUTPUT_ROOT" .
echo "$ARCHIVE"
