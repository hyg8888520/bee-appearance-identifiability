from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from beeid.config import H2VariantConfig, load_config
from beeid.data.mot import Observation
from beeid.h2.crops import load_variant_crop, variant_geometry
from beeid.h2.signals import image_quality_signals, spatial_context_signals


def _observation(
    observation_id: str = "validation:v1:000001:1",
    *,
    track_id: int = 1,
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 30.0,
    y2: float = 30.0,
) -> Observation:
    width, height = x2 - x1, y2 - y1
    return Observation(
        observation_id=observation_id,
        split="validation",
        source_split="train",
        video_id="v1",
        track_id=track_id,
        identity=f"v1:{track_id}",
        frame=1,
        frame_id=1,
        image_path="train/v1/img1/000001.png",
        image_width=100,
        image_height=80,
        original_width=width,
        original_height=height,
        raw_x=x1 + 1,
        raw_y=y1 + 1,
        raw_w=width,
        raw_h=height,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        crop_expansion=0.0,
        crop_clipped=False,
        center_x=(x1 + x2) / 2,
        center_y=(y1 + y2) / 2,
        bbox_area=width * height,
        confidence=1.0,
        object_class=1,
        visibility=1.0,
        skip_reason="",
        extra_columns="[]",
    )


def test_h2_example_configs_parse_and_keep_final_test_locked():
    root = Path(__file__).resolve().parents[1]
    for name, allow_subset in (("h2.example.yaml", False), ("h2_smoke.example.yaml", True)):
        config = load_config(root / "configs" / name)
        assert config.h2 is not None
        assert config.h2.allow_subset is allow_subset
        assert config.protocol.evaluation_split == "validation"
        assert config.dataset.source_splits == ("train",)
        assert config.paths.h1_output_root != config.paths.output_root
        assert {item.crop_expansion for item in config.h2.variants} >= {0.0, 0.2, 0.5}


def test_variant_geometry_and_bbox_context_controls(tmp_path):
    root = tmp_path / "BEE24"
    image_path = root / "train" / "v1" / "img1" / "000001.png"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (100, 80), (0, 180, 0))
    for x in range(10, 30):
        for y in range(20, 30):
            image.putpixel((x, y), (220, 0, 0))
    image.save(image_path)
    observation = _observation()
    raw = H2VariantConfig("raw", 0.2, 224, "raw", 0.0)
    foreground = H2VariantConfig("foreground", 0.2, 224, "bbox_foreground_only", 0.0)
    context = H2VariantConfig("context", 0.2, 224, "context_only", 0.0)

    geometry = variant_geometry(observation, raw, patch_size=16)
    assert (geometry["crop_width"], geometry["crop_height"]) == (24, 12)
    assert 0 < geometry["context_fraction"] < 1
    assert geometry["approx_patch_coverage"] > 0

    foreground_crop = load_variant_crop(root, observation, foreground, patch_size=16)
    context_crop = load_variant_crop(root, observation, context, patch_size=16)
    try:
        assert foreground_crop.getpixel((0, 0)) == (127, 127, 127)
        assert foreground_crop.getpixel((12, 6))[0] > 200
        assert context_crop.getpixel((0, 0))[1] > 150
        assert context_crop.getpixel((12, 6)) == (127, 127, 127)
    finally:
        foreground_crop.close()
        context_crop.close()


def test_quality_and_spatial_signals_have_predeclared_directions():
    sharp = Image.new("RGB", (32, 32), "black")
    for x in range(32):
        for y in range(32):
            if (x + y) % 2:
                sharp.putpixel((x, y), (255, 255, 255))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3.0))
    try:
        assert image_quality_signals(sharp)["laplacian_variance"] > image_quality_signals(blurred)[
            "laplacian_variance"
        ]
    finally:
        sharp.close()
        blurred.close()

    query = _observation()
    overlap = _observation(
        "validation:v1:000001:2", track_id=2, x1=20, y1=20, x2=40, y2=30
    )
    distant = _observation(
        "validation:v1:000001:3", track_id=3, x1=80, y1=60, x2=90, y2=70
    )
    signals = spatial_context_signals(query, [query, overlap, distant], [1.0, 2.0])
    assert signals["max_bbox_iou"] > 0
    assert signals["overlap_count"] == 1
    assert signals["neighbor_count_wide"] >= signals["neighbor_count_near"]
