"""BEE24 MOTChallenge parsing and lazy crop access."""

from .manifest import MANIFEST_FIELDS, build_manifest, load_manifest
from .mot import MotDataError, Observation, SequenceInfo, expanded_crop_box, parse_mot_row

__all__ = [
    "MANIFEST_FIELDS",
    "MotDataError",
    "Observation",
    "SequenceInfo",
    "build_manifest",
    "expanded_crop_box",
    "load_manifest",
    "parse_mot_row",
]
