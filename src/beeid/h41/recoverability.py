"""Continuous-lag and cumulative-window recoverability for H4.1."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


CUMULATIVE_EVENT_FIELDS = (
    "model", "video_id", "event_id", "event_frame", "gt_identity",
    "previous_predicted_track_id", "current_predicted_track_id",
    "component_identities", "component_size", "deadline",
    "lag_count_in_window", "available_lag_count", "ever_available",
    "appearance_recoverable_by_deadline", "appearance_first_recovery_lag",
    "appearance_last_recovery_lag", "appearance_recoverable_lag_count",
    "motion_recoverable_by_deadline", "motion_first_recovery_lag",
    "motion_last_recovery_lag", "motion_recoverable_lag_count",
    "joint_recoverable_by_deadline", "joint_first_recovery_lag",
    "joint_last_recovery_lag", "joint_recoverable_lag_count",
    "joint_recovery_lags", "denominator_policy", "oracle_uses_gt",
    "method_input", "final_test_read",
)


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def cumulative_event_timelines(
    exact_rows: Sequence[dict[str, Any]], deadlines: Sequence[int]
) -> list[dict[str, Any]]:
    """Convert exact-lag oracle rows into all-event cumulative-window estimands."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        groups[(str(row["model"]), str(row["event_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for _, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["horizon"]))
        first = ordered[0]
        for deadline in deadlines:
            window = [row for row in ordered if int(row["horizon"]) <= int(deadline)]
            if not window:
                raise RuntimeError(f"No exact H4.1 rows at or before deadline {deadline}")
            values: dict[str, Any] = {}
            for signal in ("appearance", "motion", "joint"):
                recovery_lags = [
                    int(row["horizon"])
                    for row in window
                    if _truth(row[f"{signal}_recoverable"])
                ]
                values.update(
                    {
                        f"{signal}_recoverable_by_deadline": bool(recovery_lags),
                        f"{signal}_first_recovery_lag": recovery_lags[0] if recovery_lags else "",
                        f"{signal}_last_recovery_lag": recovery_lags[-1] if recovery_lags else "",
                        f"{signal}_recoverable_lag_count": len(recovery_lags),
                    }
                )
                if signal == "joint":
                    values["joint_recovery_lags"] = "|".join(map(str, recovery_lags))
            output.append(
                {
                    "model": first["model"],
                    "video_id": first["video_id"],
                    "event_id": first["event_id"],
                    "event_frame": first["event_frame"],
                    "gt_identity": first["gt_identity"],
                    "previous_predicted_track_id": first["previous_predicted_track_id"],
                    "current_predicted_track_id": first["current_predicted_track_id"],
                    "component_identities": first["component_identities"],
                    "component_size": first["component_size"],
                    "deadline": deadline,
                    "lag_count_in_window": len(window),
                    "available_lag_count": sum(_truth(row["available"]) for row in window),
                    "ever_available": any(_truth(row["available"]) for row in window),
                    **values,
                    "denominator_policy": "all_baseline_idsw_events_missing_lags_remain_not_recoverable",
                    "oracle_uses_gt": True,
                    "method_input": False,
                    "final_test_read": False,
                }
            )
    return output


def summarize_cumulative_recoverability(
    timeline_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    deadlines: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in model_names:
        for deadline in deadlines:
            rows = [
                row
                for row in timeline_rows
                if row["model"] == model and int(row["deadline"]) == int(deadline)
            ]
            count = len(rows)
            videos = sorted({str(row["video_id"]) for row in rows})
            record: dict[str, Any] = {
                "model": model,
                "deadline": deadline,
                "switch_event_count": count,
                "videos_with_events": len(videos),
                "video_ids": "|".join(videos),
                "ever_available_count": sum(bool(row["ever_available"]) for row in rows),
                "ever_available_fraction": (
                    sum(bool(row["ever_available"]) for row in rows) / count if count else 0.0
                ),
                "denominator_policy": "all_baseline_idsw_events",
                "estimand": "ever_recoverable_at_any_exact_lag_up_to_deadline",
                "oracle_uses_gt": True,
                "method_input": False,
                "final_test_read": False,
            }
            for signal in ("appearance", "motion", "joint"):
                key = f"{signal}_recoverable_by_deadline"
                recovered = sum(bool(row[key]) for row in rows)
                first_lags = [
                    int(row[f"{signal}_first_recovery_lag"])
                    for row in rows
                    if row[f"{signal}_first_recovery_lag"] != ""
                ]
                record.update(
                    {
                        f"{signal}_recoverable_count": recovered,
                        f"{signal}_recoverable_fraction": recovered / count if count else 0.0,
                        f"{signal}_median_first_recovery_lag": (
                            float(np.median(first_lags)) if first_lags else ""
                        ),
                    }
                )
            output.append(record)
    return output


def cumulative_recoverability_decision(
    summary_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    *,
    gate_deadline: int,
    min_fraction: float,
    min_events: int,
    min_videos: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for model in model_names:
        candidates = [
            row
            for row in summary_rows
            if row["model"] == model and int(row["deadline"]) == gate_deadline
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Missing H4.1 cumulative gate row for {model}")
        row = candidates[0]
        event_ok = int(row["switch_event_count"]) >= min_events
        videos_ok = int(row["videos_with_events"]) >= min_videos
        fraction = float(row["joint_recoverable_fraction"])
        fraction_ok = fraction >= min_fraction
        checks.append(
            {
                "model": model,
                "event_count": int(row["switch_event_count"]),
                "event_count_pass": event_ok,
                "videos_with_events": int(row["videos_with_events"]),
                "video_count_pass": videos_ok,
                "cumulative_joint_recoverable_fraction": fraction,
                "recoverable_fraction_pass": fraction_ok,
                "pass": event_ok and videos_ok and fraction_ok,
            }
        )
    passed = bool(checks) and all(row["pass"] for row in checks)
    return {
        "status": (
            "GO_ADAPTIVE_METHOD_EVALUATION"
            if passed
            else "STOP_NO_WINDOW_RECOVERABILITY_SIGNAL"
        ),
        "gate_passed": passed,
        "gate_deadline": gate_deadline,
        "estimand": "ever_joint_recoverable_at_any_exact_lag_up_to_deadline",
        "thresholds": {
            "min_cumulative_joint_recoverable_fraction": min_fraction,
            "min_events_per_model": min_events,
            "min_videos_with_events": min_videos,
        },
        "model_checks": checks,
        "failure_action": "STOP_BEFORE_METHOD_EVALUATION",
        "design_timing": "DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT",
        "oracle_uses_gt": True,
        "method_uses_gt": False,
        "final_test_read": False,
    }


def crosscheck_h4_v1_exact_rows(
    exact_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, str]],
    model_names: Sequence[str],
    comparison_frame_limit: int | None = None,
) -> dict[str, Any]:
    """Prove H4.1 did not reinterpret any frozen H4-v1 exact-horizon result."""
    legacy_horizons = {1, 3, 5, 10}

    def key(row: dict[str, Any]) -> tuple[str, str, int, str, str, str, int]:
        return (
            str(row["model"]),
            str(row["video_id"]),
            int(row["event_frame"]),
            str(row["gt_identity"]),
            str(row["previous_predicted_track_id"]),
            str(row["current_predicted_track_id"]),
            int(row["horizon"]),
        )

    wanted_models = set(model_names)
    current = {
        key(row): row
        for row in exact_rows
        if row["model"] in wanted_models and int(row["horizon"]) in legacy_horizons
        and (
            comparison_frame_limit is None
            or int(row["future_frame"]) <= comparison_frame_limit
        )
    }
    source = {
        identifier: row
        for row in source_rows
        if row["model"] in wanted_models
        and (identifier := key(row)) in current
    }
    if set(current) != set(source):
        missing = sorted(set(source) - set(current))[:3]
        extra = sorted(set(current) - set(source))[:3]
        raise RuntimeError(
            f"H4.1 exact-horizon event keys differ from H4-v1; missing={missing}, extra={extra}"
        )
    fields = (
        "available",
        "appearance_recoverable",
        "motion_recoverable",
        "joint_recoverable",
    )
    margin_fields = (
        "appearance_min_margin",
        "motion_min_margin",
        "joint_min_margin",
    )
    for identifier, source_row in source.items():
        current_row = current[identifier]
        for field in fields:
            if _truth(current_row[field]) != _truth(source_row[field]):
                raise RuntimeError(f"H4.1 changed H4-v1 {field} for {identifier}")
        if str(current_row["unavailable_reason"]) != str(source_row["unavailable_reason"]):
            raise RuntimeError(f"H4.1 changed H4-v1 unavailable_reason for {identifier}")
        for field in margin_fields:
            left, right = current_row[field], source_row[field]
            if left == "" and right == "":
                continue
            if left == "" or right == "" or not math.isclose(
                float(left), float(right), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise RuntimeError(f"H4.1 changed H4-v1 {field} for {identifier}")
    return {
        "status": "passed",
        "compared_row_count": len(source),
        "legacy_horizons": sorted(legacy_horizons),
        "comparison_frame_limit": (
            "" if comparison_frame_limit is None else comparison_frame_limit
        ),
        "source_h4_v1_conclusion_preserved": True,
        "final_test_read": False,
    }
