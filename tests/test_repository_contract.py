from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def test_example_config_has_required_placeholders() -> None:
    text = (REPOSITORY / "configs" / "h1_bee24.example.yaml").read_text(encoding="utf-8")
    assert "root: /CHANGE/ME/BEE24" in text
    assert "annotation: /CHANGE/ME/BEE24/train.json" in text
    assert "checkpoint: /CHANGE/ME/dinov3_vits16_pretrain.pth" in text
    assert "root: /CHANGE/ME/experiments/bee24_h1" in text


def test_local_configs_and_checkpoints_are_ignored() -> None:
    text = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("configs/*.local.yaml", "*.pth", "*.pt", "*.ckpt"):
        assert pattern in text


def test_python_source_has_no_server_specific_paths_or_sys_path_changes() -> None:
    forbidden = ("/mnt/", "/home/", "sys.path.append", "sys.path.insert")
    for path in (REPOSITORY / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden), path

