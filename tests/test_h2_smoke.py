from __future__ import annotations

import json

from beeid.h2.synthetic import h2_synthetic_smoke


def test_h2_synthetic_smoke_is_complete_and_test_only(tmp_path):
    output = tmp_path / "h2-output"
    result = h2_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["test_only_encoder"] is True
    assert result["primary_cache_status"] == "reused_h1_cache"
    assert result["resumed_variant_count"] == 7
    assert result["signal_rows"] > 0
    assert result["query_rows"] > 0
    assert result["memory_events"] > 0
    metadata = json.loads((output / "h2_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "SERVER_VALIDATION_PENDING"
    assert metadata["final_test_read"] is False
    assert metadata["models_requested"] == ["synthetic_test"]
    assert metadata["official_topic_agw_role"].startswith("REFERENCE_ONLY")
    assert (output / "h2_figures" / "index.txt").is_file()
