from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beeid.models import REAL_MODEL_NAMES
from beeid.models.base import ModelContractError, validate_embeddings


def test_real_model_choices_exclude_synthetic_encoder():
    assert REAL_MODEL_NAMES == ("resnet50", "dinov3", "topic_agw")
    assert "synthetic" not in " ".join(REAL_MODEL_NAMES)


def test_embedding_contract_normalizes_float32_and_rejects_bad_values():
    output = validate_embeddings([[3, 4], [0, 2]])
    assert output.dtype == np.float32 and output.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0)
    with pytest.raises(ModelContractError):
        validate_embeddings([1, 2, 3])
    with pytest.raises(ModelContractError):
        validate_embeddings([[0, 0]])


def test_torch_wrappers_freeze_normalize_and_validate_dino_keys(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from beeid.models.dinov3 import DINOv3Extractor
    from beeid.models.resnet50 import ResNet50Extractor

    transform = lambda image: torch.ones(3, 8, 8)

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            return torch.stack([batch.mean((1, 2, 3)), batch.sum((1, 2, 3))], dim=1) * self.scale

    resnet = ResNet50Extractor(224, "cpu", False, "bfloat16", model=TinyResNet(), transform=transform)
    result = resnet.encode([Image.new("RGB", (8, 8))])
    assert result.shape == (1, 2)
    assert all(not parameter.requires_grad for parameter in resnet.model.parameters())
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)

    class TinyDino(torch.nn.Module):
        def __init__(self, missing=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.missing = missing

        def forward_features(self, batch):
            values = {"x_norm_patchtokens": torch.ones(len(batch), 4, 3) * self.weight}
            if not self.missing:
                values["x_norm_clstoken"] = torch.ones(len(batch), 3) * self.weight
            return values

    dino = DINOv3Extractor(
        tmp_path, tmp_path / "unused.pth", 224, "patch_mean", "cpu", False, "bfloat16",
        model=TinyDino(), transform=transform, verify_commit=False,
    )
    assert dino.encode([Image.new("RGB", (8, 8))]).shape == (1, 3)
    assert all(not parameter.requires_grad for parameter in dino.model.parameters())
    bad = DINOv3Extractor(
        tmp_path, tmp_path / "unused.pth", 224, "patch_mean", "cpu", False, "bfloat16",
        model=TinyDino(missing=True), transform=transform, verify_commit=False,
    )
    with pytest.raises(ModelContractError, match="missing keys"):
        bad.encode([Image.new("RGB", (8, 8))])


def test_topic_wrapper_strict_state_contract(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from beeid.models.topic_agw import TopicAGWExtractor, TopicCompatibilityError

    class TinyAGW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.Linear(3, 2, bias=False)

        def forward(self, batch):
            return batch.mean((2, 3))

    model = TinyAGW()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    transform = lambda image: torch.ones(3, 8, 8)
    wrapper = TopicAGWExtractor(
        tmp_path, tmp_path / "unused.pth", Path("fast-reid/configs/bee/AGW_S50.yml"),
        224, "cpu", False, "bfloat16", model=model, transform=transform,
        verify_commit=False, state_override=state,
    )
    result = wrapper.encode([Image.new("RGB", (8, 8))])
    assert result.shape == (1, 3)
    assert wrapper.load_report == {"missing_keys": [], "unexpected_keys": [], "shape_mismatch": []}
    assert all(not parameter.requires_grad for parameter in wrapper.model.parameters())

    mismatched = {"heads.weight": torch.ones(3, 3)}
    with pytest.raises(TopicCompatibilityError, match="strict checkpoint comparison"):
        TopicAGWExtractor(
            tmp_path, tmp_path / "unused.pth", Path("x.yml"), 224, "cpu", False, "bfloat16",
            model=TinyAGW(), transform=transform, verify_commit=False, state_override=mismatched,
        )


def test_resnet50_cpu_architecture_single_batch():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet50(weights=None).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 1000)
    assert torch.isfinite(output).all()
    assert all(not parameter.requires_grad for parameter in model.parameters())
