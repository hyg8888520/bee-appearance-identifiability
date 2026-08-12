"""Read-only server environment and GPU smoke checks."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .references import DINOV3_COMMIT, TOPICTRACK_COMMIT
from .utils import atomic_write_json, git_head


def check_server_environment(config: ExperimentConfig) -> dict[str, Any]:
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise RuntimeError("Server environment requires torch==2.7.1 and torchvision==0.22.1") from error
    checks: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "conda": shutil.which("conda"),
        "mamba": shutil.which("mamba"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "paths": {},
        "pins": {},
    }
    for name, path, kind in (
        ("bee24_root", config.paths.bee24_root, "directory"),
        ("output_root", config.paths.output_root, "output"),
        ("cache_root", config.paths.cache_root, "output"),
        ("dinov3_repo", config.paths.dinov3_repo, "directory"),
        ("dinov3_weights", config.paths.dinov3_weights, "file"),
        ("topictrack_repo", config.paths.topictrack_repo, "directory"),
        ("topic_agw_weights", config.paths.topic_agw_weights, "file"),
    ):
        exists = path.is_dir() if kind == "directory" else path.is_file() if kind == "file" else True
        if kind == "output":
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".beeid-write-probe"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            exists = True
        checks["paths"][name] = {"path": str(path), "kind": kind, "ok": exists}
    if config.paths.dinov3_repo.is_dir():
        observed = git_head(config.paths.dinov3_repo)
        checks["pins"]["dinov3"] = {"expected": DINOV3_COMMIT, "observed": observed, "ok": observed == DINOV3_COMMIT}
    if config.paths.topictrack_repo.is_dir():
        observed = git_head(config.paths.topictrack_repo)
        checks["pins"]["topictrack"] = {"expected": TOPICTRACK_COMMIT, "observed": observed, "ok": observed == TOPICTRACK_COMMIT}
    if torch.cuda.is_available():
        checks["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    checks["disk"] = {
        "path": str(config.paths.output_root),
        "total_bytes": shutil.disk_usage(config.paths.output_root).total,
        "free_bytes": shutil.disk_usage(config.paths.output_root).free,
    }
    checks["scheduler"] = {
        "slurm_sbatch": shutil.which("sbatch"),
        "slurm_sinfo": shutil.which("sinfo"),
    }
    try:
        nvidia = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False, capture_output=True, text=True, timeout=10,
        )
        checks["nvidia_smi"] = {
            "returncode": nvidia.returncode,
            "driver_gpu_memory": nvidia.stdout.strip().splitlines(),
            "stderr": nvidia.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        checks["nvidia_smi"] = {"error": str(error)}
    try:
        with urllib.request.urlopen("https://github.com", timeout=5) as response:
            checks["github_network"] = {"ok": 200 <= response.status < 400, "status": response.status}
    except Exception as error:
        checks["github_network"] = {"ok": False, "error": str(error)}
    checks["ok"] = (
        torch.__version__.split("+")[0] == "2.7.1"
        and torchvision.__version__.split("+")[0] == "0.22.1"
        and torch.cuda.is_available()
        and all(item["ok"] for item in checks["paths"].values())
        and all(item["ok"] for item in checks["pins"].values())
        and checks["github_network"]["ok"]
    )
    atomic_write_json(config.paths.output_root / "logs" / "server_env.json", checks)
    return checks


def gpu_smoke(config: ExperimentConfig) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU smoke must run on the target server")
    from .data.crops import load_crop
    from .data.manifest import build_manifest, load_manifest
    from .models.dinov3 import DINOv3Extractor
    from .models.resnet50 import ResNet50Extractor
    from .models.topic_agw import TopicAGWExtractor, TopicCompatibilityError

    if not config.manifest_path.is_file():
        build_manifest(config)
    observations = load_manifest(
        config.manifest_path, split=config.protocol.evaluation_split, valid_only=True
    )
    if not observations:
        raise RuntimeError("GPU smoke needs at least one real, valid BEE24 crop")
    crop = load_crop(config.paths.bee24_root, observations[0])
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "real_crop_observation_id": observations[0].observation_id,
        "amp_requested": config.runtime.amp,
    }
    left = torch.randn(128, 128, device="cuda")
    report["cuda_tensor_smoke"] = {"finite": bool(torch.isfinite(left @ left).all())}
    common = {
        "input_size": config.protocol.input_size,
        "device_name": "cuda:0",
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    resnet = ResNet50Extractor(**common)
    report["resnet50"] = {"embedding_shape": list(resnet.encode([crop]).shape), "amp": config.runtime.amp}
    dino = DINOv3Extractor(
        config.paths.dinov3_repo,
        config.paths.dinov3_weights,
        feature_source=config.models.dinov3_feature_source,
        **common,
    )
    tensor = dino.torch.stack([dino.transform(crop)]).to(dino.device)
    amp_dtype = dino.torch.float16 if config.runtime.amp_dtype == "float16" else dino.torch.bfloat16
    with dino.torch.inference_mode(), dino.torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=config.runtime.amp
    ):
        features = dino.model.forward_features(tensor)
    report["dinov3"] = {
        "embedding_shape": list(dino.encode([crop]).shape),
        "forward_features": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all())}
            for key, value in features.items()
            if hasattr(value, "shape")
        },
        "amp": config.runtime.amp,
    }
    try:
        topic = TopicAGWExtractor(
            config.paths.topictrack_repo,
            config.paths.topic_agw_weights,
            config.models.topic_config_relative,
            **common,
        )
        report["topic_agw"] = {
            "status": "passed",
            "embedding_shape": list(topic.encode([crop]).shape),
            "strict_load_report": topic.load_report,
            "extractor_details": topic.details,
            "amp": config.runtime.amp,
        }
    except TopicCompatibilityError as error:
        report["topic_agw"] = {"status": "BLOCKED_TOPIC_AGW_COMPATIBILITY", "error": str(error)}
    finally:
        crop.close()
    report["ok"] = (
        report["cuda_tensor_smoke"]["finite"]
        and report["topic_agw"].get("status") == "passed"
    )
    destination = config.paths.output_root / "logs" / "gpu_smoke.json"
    atomic_write_json(destination, report)
    return report
