"""Frozen H5 protocol validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..utils import sha256_file
from . import H5_MODELS, H5_PRIMARY_VARIANT, H5_VARIANTS


class H5ProtocolError(RuntimeError):
    """Raised when the frozen BeeTrackQuery protocol is altered or incomplete."""


def validate_h5_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    source = path.resolve(strict=False)
    if not source.is_file():
        raise H5ProtocolError(f"H5 protocol does not exist: {source}")
    digest = sha256_file(source)
    if checksum_path is not None:
        check = checksum_path.resolve(strict=False)
        if not check.is_file():
            raise H5ProtocolError(f"H5 checksum does not exist: {check}")
        match = re.match(r"^([0-9a-fA-F]{64})(?:\s+.+)?$", check.read_text(encoding="utf-8").strip())
        if match is None or match.group(1).lower() != digest:
            raise H5ProtocolError("H5 protocol checksum mismatch")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H5ProtocolError("H5 protocol must be a YAML mapping")
    required = {
        "schema_version", "protocol_id", "status", "identity_definition", "project_split",
        "source_h3", "model_policy", "beetrackquery", "association_variants",
        "primary_variant", "metrics", "selection_gate", "final_test", "claims",
    }
    missing = sorted(required - set(value))
    if missing:
        raise H5ProtocolError("H5 protocol is missing: " + ", ".join(missing))
    split = value["project_split"]
    if split.get("fit_partition") != "project_train" or split.get("evaluation_partition") != "development_validation":
        raise H5ProtocolError("H5 must fit on project_train and evaluate on development_validation")
    final = value["final_test"]
    if final.get("access") is not False or final.get("runs_allowed_in_h5_development") != 0:
        raise H5ProtocolError("H5 development must keep final test locked")
    variants = tuple(value["association_variants"])
    if variants != H5_VARIANTS or value["primary_variant"] != H5_PRIMARY_VARIANT:
        raise H5ProtocolError("H5 variants differ from the frozen implementation")
    if tuple(value["model_policy"]["frozen_backbones"]) != H5_MODELS:
        raise H5ProtocolError("H5 frozen backbone list differs from the implementation")
    return {
        "status": "valid",
        "protocol_id": value["protocol_id"],
        "protocol_sha256": digest,
        "fit_partition": "project_train",
        "evaluation_partition": "development_validation",
        "models": list(H5_MODELS),
        "variants": list(H5_VARIANTS),
        "primary_variant": H5_PRIMARY_VARIANT,
        "final_test_access": False,
        "final_test_runs_allowed": 0,
        "parameters": dict(value["beetrackquery"]),
        "source_h3_protocol_sha256": value["source_h3"]["protocol_sha256"],
    }
