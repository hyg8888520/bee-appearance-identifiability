from pathlib import Path

from bee24_dinotrack.cli import main
from conftest import write_runtime


def test_validate_only_does_not_import_model_stack(tmp_path: Path, capsys: object) -> None:
    config_path, _ = write_runtime(tmp_path)
    assert main(["--config", str(config_path), "--validate-only"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Experiment summary" in captured.out
    assert "model inference was not started" in captured.out
