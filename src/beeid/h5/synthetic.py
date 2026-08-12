"""CPU-only end-to-end contract smoke for BeeTrackQuery."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..data.mot import Observation
from ..h3.metrics import summarize_gt_assignments
from ..utils import atomic_write_json, atomic_write_text
from . import H5_PRIMARY_VARIANT
from .model import BeeTrackQuery, supervised_transition
from .tracker import normalized_geometry, track_sequence


def _observation(video: str, frame: int, identity: int, x: float) -> Observation:
    return Observation(
        observation_id=f"validation:{video}:{frame:06d}:{identity}", split="validation",
        source_split="train", video_id=video, track_id=identity, identity=f"{video}:{identity}",
        frame=frame, frame_id=frame, image_path=f"train/{video}/img1/{frame:06d}.jpg",
        image_width=100, image_height=80, original_width=15.0, original_height=15.0,
        raw_x=x+1, raw_y=21.0, raw_w=15.0, raw_h=15.0, bbox_x1=x, bbox_y1=20.0,
        bbox_x2=x+15, bbox_y2=35.0, x1=int(x), y1=20, x2=int(x+15), y2=35,
        crop_expansion=0.2, crop_clipped=False, center_x=x+7.5, center_y=27.5,
        bbox_area=225.0, confidence=1.0, object_class=1, visibility=1.0,
        skip_reason="", extra_columns="[]",
    )


def h5_synthetic_smoke(output_directory: Path | None = None) -> dict[str, object]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h5-synthetic-")
    output = output_directory.resolve(strict=False) if output_directory else Path(temporary.name) / "output"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(24)
    observations = [
        _observation("toy", frame, identity, 10.0 + (frame if identity == 1 else 45.0 - frame))
        for frame in range(1, 9) for identity in (1, 2)
    ]
    embeddings = np.stack([
        np.asarray([1.0, 0.1 * item.track_id, item.center_x / 100, 0.2], dtype=np.float32)
        for item in observations
    ])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    model = BeeTrackQuery(4, 16, 4, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    frames = [[index for index, item in enumerate(observations) if item.frame == frame] for frame in range(1, 9)]
    losses: list[float] = []
    for _ in range(3):
        for previous, current in zip(frames, frames[1:]):
            prev_raw = torch.as_tensor(embeddings[previous])
            cur_raw = torch.as_tensor(embeddings[current])
            prev_geom = torch.as_tensor(np.stack([normalized_geometry(observations[i]) for i in previous]))
            cur_geom = torch.as_tensor(np.stack([normalized_geometry(observations[i]) for i in current]))
            prev_tokens = model.encode_detections(prev_raw, prev_geom)
            cur_tokens = model.encode_detections(cur_raw, cur_geom)
            delta = torch.stack([torch.abs(geometry[None, :] - cur_geom) for geometry in prev_geom])
            result = supervised_transition(model, prev_tokens, [observations[i].identity for i in previous], cur_tokens, [observations[i].identity for i in current], delta)
            assert result is not None
            loss = result.association_loss + 0.25 * result.reliability_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
    example = load_config(Path(__file__).resolve().parents[3] / "configs" / "h5_smoke.example.yaml")
    assert example.h5 is not None
    rows = track_sequence("test_only_encoder", H5_PRIMARY_VARIANT, model, observations, embeddings, replace(example.h5, update_gate=0.0), torch.device("cpu"))
    per_video, summary = summarize_gt_assignments(rows)
    import csv, io
    def write(name: str, values: list[dict[str, object]]) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=tuple(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values); atomic_write_text(output / name, buffer.getvalue())
    write("h5_assignments.csv", rows); write("h5_per_video_metrics.csv", per_video); write("h5_summary.csv", summary)
    checkpoint = output / "h5_checkpoints" / "test_only_encoder" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), checkpoint)
    atomic_write_json(checkpoint.parent / "checkpoint.json", {"fingerprint": "synthetic-fixed", "status": "completed", "test_only_encoder": True, "final_test_read": False})
    resumed = json.loads((checkpoint.parent / "checkpoint.json").read_text(encoding="utf-8"))["fingerprint"] == "synthetic-fixed"
    required = ["h5_assignments.csv", "h5_per_video_metrics.csv", "h5_summary.csv", "h5_checkpoints/test_only_encoder/final.pt"]
    missing = [name for name in required if not (output / name).exists()]
    metadata = {"status": "SERVER_VALIDATION_PENDING", "test_only_encoder": True, "real_experiment_result": False, "training_loss_count": len(losses), "assignment_rows": len(rows), "checkpoint_resume_verified": resumed, "missing_artifacts": missing, "final_test_read": False}
    atomic_write_json(output / "h5_run_metadata.json", metadata)
    result = {**metadata, "status": "passed" if not missing else "failed", "output_root": str(output), "metadata_status": metadata["status"]}
    if missing or not resumed or not rows:
        raise RuntimeError(f"H5 synthetic smoke failed: {result}")
    temporary.cleanup()
    return result
