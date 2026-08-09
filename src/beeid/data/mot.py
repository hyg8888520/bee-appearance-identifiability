"""Strict MOTChallenge parsing for the BEE24 layout."""

from __future__ import annotations

import configparser
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class MotDataError(ValueError):
    """Raised when BEE24 metadata or annotations violate the H1 contract."""


@dataclass(frozen=True)
class SequenceInfo:
    video_id: str
    source_split: str
    directory: Path
    image_directory: Path
    declared_image_directory: str
    image_directory_fallback_used: bool
    image_extension: str
    frame_rate: int
    sequence_length: int
    image_width: int
    image_height: int

    def image_path(self, frame: int) -> Path:
        return self.image_directory / f"{frame:06d}{self.image_extension}"


@dataclass(frozen=True)
class MotRow:
    frame: int
    track_id: int
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float
    confidence: float
    object_class: int | None
    visibility: float | None
    extra_columns: tuple[float, ...]


@dataclass(frozen=True)
class Observation:
    observation_id: str
    split: str
    source_split: str
    video_id: str
    track_id: int
    identity: str
    frame: int
    frame_id: int
    image_path: str
    image_width: int
    image_height: int
    original_width: float
    original_height: float
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_expansion: float
    crop_clipped: bool
    center_x: float
    center_y: float
    bbox_area: float
    confidence: float
    object_class: int | None
    visibility: float | None
    skip_reason: str
    extra_columns: str

    @property
    def valid(self) -> bool:
        return not self.skip_reason

    @property
    def crop_x0(self) -> int:
        return self.x1

    @property
    def crop_y0(self) -> int:
        return self.y1

    @property
    def crop_x1(self) -> int:
        return self.x2

    @property
    def crop_y1(self) -> int:
        return self.y2

    @property
    def clipped(self) -> bool:
        return self.crop_clipped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_seqinfo(video_directory: Path, source_split: str) -> SequenceInfo:
    path = video_directory / "seqinfo.ini"
    if not path.is_file():
        raise MotDataError(f"Missing seqinfo.ini: {path}")
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
        section = parser["Sequence"]
        declared_image_directory = section.get("imDir", "img1").strip()
        if not declared_image_directory:
            raise ValueError("imDir must not be empty")
        declared_path = Path(declared_image_directory)
        if declared_path.is_absolute() or ".." in declared_path.parts:
            raise ValueError(f"imDir must stay within the sequence directory: {declared_image_directory!r}")
        image_directory = video_directory / declared_path
        fallback_used = False
        if not image_directory.is_dir():
            canonical_candidates = [
                candidate
                for name in ("img1", "images")
                if (candidate := video_directory / name) != image_directory and candidate.is_dir()
            ]
            if len(canonical_candidates) == 1:
                image_directory = canonical_candidates[0]
                fallback_used = True
            elif len(canonical_candidates) > 1:
                choices = ", ".join(str(candidate) for candidate in canonical_candidates)
                raise ValueError(
                    f"imDir={declared_image_directory!r} does not exist and fallback is ambiguous: {choices}"
                )
            else:
                raise ValueError(
                    f"imDir={declared_image_directory!r} does not exist; also checked canonical "
                    f"directories {video_directory / 'img1'} and {video_directory / 'images'}"
                )
        extension = section.get("imExt", ".jpg")
        if not extension.startswith("."):
            extension = f".{extension}"
        values = SequenceInfo(
            video_id=video_directory.name,
            source_split=source_split,
            directory=video_directory,
            image_directory=image_directory,
            declared_image_directory=declared_image_directory,
            image_directory_fallback_used=fallback_used,
            image_extension=extension,
            frame_rate=section.getint("frameRate", fallback=0),
            sequence_length=section.getint("seqLength"),
            image_width=section.getint("imWidth"),
            image_height=section.getint("imHeight"),
        )
    except (KeyError, ValueError, configparser.Error) as error:
        raise MotDataError(f"Invalid seqinfo.ini {path}: {error}") from error
    if values.sequence_length <= 0 or values.image_width <= 0 or values.image_height <= 0:
        raise MotDataError(f"Non-positive dimensions or sequence length in {path}")
    return values


def _integer(value: str, label: str, path: Path, line_number: int) -> int:
    try:
        number = float(value)
    except ValueError as error:
        raise MotDataError(f"{path}:{line_number}: {label} is not numeric: {value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise MotDataError(f"{path}:{line_number}: {label} must be an integer: {value!r}")
    return int(number)


def _float(value: str, label: str, path: Path, line_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise MotDataError(f"{path}:{line_number}: {label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise MotDataError(f"{path}:{line_number}: {label} is not finite")
    return number


def parse_mot_row(line: str, path: Path, line_number: int) -> MotRow:
    columns = [item.strip() for item in line.split(",")]
    if len(columns) < 6:
        raise MotDataError(f"{path}:{line_number}: expected at least 6 MOT columns, got {len(columns)}")
    frame = _integer(columns[0], "frame", path, line_number)
    track_id = _integer(columns[1], "track_id", path, line_number)
    if frame <= 0 or track_id <= 0:
        raise MotDataError(f"{path}:{line_number}: frame and track_id must be positive")
    raw = tuple(_float(columns[index], ("x", "y", "w", "h")[index - 2], path, line_number) for index in range(2, 6))
    confidence = _float(columns[6], "confidence", path, line_number) if len(columns) >= 7 else 1.0
    object_class = _integer(columns[7], "class", path, line_number) if len(columns) >= 8 else None
    visibility = _float(columns[8], "visibility", path, line_number) if len(columns) >= 9 else None
    extra = tuple(_float(value, f"column_{index + 1}", path, line_number) for index, value in enumerate(columns[9:], start=9))
    return MotRow(frame, track_id, *raw, confidence, object_class, visibility, extra)


def expanded_crop_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_width: int,
    image_height: int,
    expansion: float,
) -> tuple[int, int, int, int, bool]:
    """Expand by ``expansion`` total, then floor/ceil and clip to half-open bounds."""
    horizontal = (x1 - x0) * expansion / 2.0
    vertical = (y1 - y0) * expansion / 2.0
    unbounded = (
        math.floor(x0 - horizontal),
        math.floor(y0 - vertical),
        math.ceil(x1 + horizontal),
        math.ceil(y1 + vertical),
    )
    clipped = (
        max(0, min(image_width, unbounded[0])),
        max(0, min(image_height, unbounded[1])),
        max(0, min(image_width, unbounded[2])),
        max(0, min(image_height, unbounded[3])),
    )
    return (*clipped, clipped != unbounded)


def row_to_observation(
    row: MotRow,
    sequence: SequenceInfo,
    split: str,
    dataset_root: Path,
    bbox_origin: str,
    crop_expansion: float,
    confidence_min: float,
) -> Observation:
    x0 = row.raw_x - (1.0 if bbox_origin == "one" else 0.0)
    y0 = row.raw_y - (1.0 if bbox_origin == "one" else 0.0)
    x1 = x0 + row.raw_w
    y1 = y0 + row.raw_h
    reasons: list[str] = []
    if row.frame > sequence.sequence_length:
        reasons.append("frame_out_of_range")
    if row.raw_w <= 0 or row.raw_h <= 0:
        reasons.append("non_positive_area")
    if x1 <= 0 or y1 <= 0 or x0 >= sequence.image_width or y0 >= sequence.image_height:
        reasons.append("fully_out_of_bounds")
    if row.confidence < confidence_min:
        reasons.append("below_confidence_threshold")
    crop_x0, crop_y0, crop_x1, crop_y1, clipped = expanded_crop_box(
        x0, y0, x1, y1, sequence.image_width, sequence.image_height, crop_expansion
    )
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        reasons.append("empty_crop_after_clipping")
    image = sequence.image_path(row.frame)
    try:
        image_relative = image.resolve(strict=False).relative_to(dataset_root.resolve(strict=False)).as_posix()
    except ValueError as error:
        raise MotDataError(f"Image path escapes dataset root: {image}") from error
    observation_id = f"{split}:{sequence.video_id}:{row.frame:06d}:{row.track_id}"
    return Observation(
        observation_id=observation_id,
        split=split,
        source_split=sequence.source_split,
        video_id=sequence.video_id,
        track_id=row.track_id,
        identity=f"{sequence.video_id}:{row.track_id}",
        frame=row.frame,
        frame_id=row.frame,
        image_path=image_relative,
        image_width=sequence.image_width,
        image_height=sequence.image_height,
        original_width=row.raw_w,
        original_height=row.raw_h,
        raw_x=row.raw_x,
        raw_y=row.raw_y,
        raw_w=row.raw_w,
        raw_h=row.raw_h,
        bbox_x1=x0,
        bbox_y1=y0,
        bbox_x2=x1,
        bbox_y2=y1,
        x1=crop_x0,
        y1=crop_y0,
        x2=crop_x1,
        y2=crop_y1,
        crop_expansion=crop_expansion,
        crop_clipped=clipped,
        center_x=x0 + row.raw_w / 2.0,
        center_y=y0 + row.raw_h / 2.0,
        bbox_area=row.raw_w * row.raw_h,
        confidence=row.confidence,
        object_class=row.object_class,
        visibility=row.visibility,
        skip_reason=";".join(dict.fromkeys(reasons)),
        extra_columns=json.dumps(row.extra_columns, separators=(",", ":")),
    )
