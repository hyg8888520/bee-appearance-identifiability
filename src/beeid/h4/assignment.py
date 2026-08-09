"""Deterministic local alternatives around a global assignment ambiguity."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..h3.assignment import maximum_weight_matching


@dataclass(frozen=True)
class AssignmentOption:
    matches: tuple[tuple[int, int], ...]
    objective: float


@dataclass(frozen=True)
class ConflictComponent:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    assignment_margin: float | None
    alternative_count: int
    oversized: bool


def _objective(
    scores: np.ndarray,
    matches: Sequence[tuple[int, int]],
    unmatched_penalty: float,
) -> float:
    rows, columns = scores.shape
    unmatched = (rows - len(matches)) + (columns - len(matches))
    return float(sum(scores[row, column] for row, column in matches)) - (
        unmatched_penalty * unmatched
    )


def _candidate_components(
    scores: np.ndarray,
    allowed: np.ndarray,
    best: Sequence[tuple[int, int]],
    ambiguity_margin: float,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    matched_score = {row: float(scores[row, column]) for row, column in best}
    matched_row_by_column = {column: row for row, column in best}
    edges: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for row, reference in matched_score.items():
        candidates = [
            column
            for column in range(scores.shape[1])
            if allowed[row, column]
            and reference - float(scores[row, column]) <= ambiguity_margin + 1e-12
        ]
        if len(candidates) < 2:
            continue
        for column in candidates:
            left = ("r", row)
            right = ("c", column)
            edges.setdefault(left, set()).add(right)
            edges.setdefault(right, set()).add(left)
            matched_row = matched_row_by_column.get(column)
            if matched_row is not None:
                matched_node = ("r", matched_row)
                edges.setdefault(right, set()).add(matched_node)
                edges.setdefault(matched_node, set()).add(right)
            competing_rows = [
                other
                for other, other_reference in matched_score.items()
                if allowed[other, column]
                and other_reference - float(scores[other, column])
                <= ambiguity_margin + 1e-12
            ]
            for other in competing_rows:
                other_node = ("r", other)
                edges.setdefault(right, set()).add(other_node)
                edges.setdefault(other_node, set()).add(right)
    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    unseen = set(edges)
    while unseen:
        root = min(unseen)
        stack = [root]
        nodes: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            unseen.discard(node)
            stack.extend(sorted(edges.get(node, ()), reverse=True))
        rows = tuple(sorted(index for kind, index in nodes if kind == "r"))
        columns = tuple(sorted(index for kind, index in nodes if kind == "c"))
        if rows and columns and (len(rows) > 1 or len(columns) > 1):
            components.append((rows, columns))
    return components


def _local_matchings(
    rows: Sequence[int],
    columns: Sequence[int],
    allowed: np.ndarray,
    target_cardinality: int,
) -> list[tuple[tuple[int, int], ...]]:
    output: set[tuple[tuple[int, int], ...]] = set()
    if target_cardinality <= 0:
        return [()]
    for selected_rows in itertools.combinations(rows, target_cardinality):
        for selected_columns in itertools.combinations(columns, target_cardinality):
            for permutation in itertools.permutations(selected_columns):
                pairs = tuple(sorted(zip(selected_rows, permutation)))
                if all(allowed[row, column] for row, column in pairs):
                    output.add(pairs)
    return sorted(output)


def local_assignment_options(
    scores: np.ndarray,
    *,
    valid: np.ndarray | None,
    minimum_score: float,
    ambiguity_margin: float,
    max_component_size: int,
    beam_width: int,
    unmatched_penalty: float,
) -> tuple[list[AssignmentOption], ConflictComponent | None]:
    """Return the best matching plus close alternatives in one local component."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("H4 assignment scores must be a finite matrix")
    allowed = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if allowed.shape != values.shape:
        raise ValueError("H4 assignment validity mask must match scores")
    allowed &= values >= float(minimum_score)
    best = tuple(maximum_weight_matching(values, valid=allowed))
    best_option = AssignmentOption(best, _objective(values, best, unmatched_penalty))
    components = _candidate_components(values, allowed, best, ambiguity_margin)
    if not components:
        return [best_option], None

    candidates: list[tuple[float, list[AssignmentOption], ConflictComponent]] = []
    oversized_components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for rows, columns in components:
        if len(rows) > max_component_size or len(columns) > max_component_size:
            oversized_components.append((rows, columns))
            continue
        row_set, column_set = set(rows), set(columns)
        outside = tuple(
            (row, column)
            for row, column in best
            if row not in row_set and column not in column_set
        )
        local_best = tuple(
            (row, column)
            for row, column in best
            if row in row_set and column in column_set
        )
        local = _local_matchings(rows, columns, allowed, len(local_best))
        options_by_matches: dict[tuple[tuple[int, int], ...], AssignmentOption] = {}
        for matching in local:
            combined = tuple(sorted(outside + matching))
            option = AssignmentOption(
                combined, _objective(values, combined, unmatched_penalty)
            )
            options_by_matches[combined] = option
        options_by_matches[best] = best_option
        options = sorted(
            options_by_matches.values(), key=lambda item: (-item.objective, item.matches)
        )
        if len(options) < 2:
            continue
        margin = (options[0].objective - options[1].objective) / max(1, len(best))
        component = ConflictComponent(
            tuple(rows), tuple(columns), float(margin), len(options), False
        )
        candidates.append((float(margin), options, component))
    if candidates:
        margin, options, component = min(
            candidates,
            key=lambda item: (item[0], item[2].rows, item[2].columns),
        )
        if margin <= ambiguity_margin + 1e-12:
            return options[:beam_width], component
        return [best_option], None
    if oversized_components:
        rows, columns = min(oversized_components, key=lambda item: (len(item[0]) + len(item[1]), item))
        return [best_option], ConflictComponent(rows, columns, None, 0, True)
    return [best_option], None
