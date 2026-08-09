from __future__ import annotations

from pathlib import Path

import yaml


def test_frozen_project_split_is_complete_and_disjoint():
    root = Path(__file__).resolve().parents[1]
    split = yaml.safe_load(
        (root / "configs" / "splits" / "project_split.yaml").read_text(encoding="utf-8")
    )

    assert split["schema_version"] == 1
    assert split["status"] == "FROZEN"
    assert split["identity_definition"] == ["video_id", "track_id"]
    assert split["isolation_unit"] == "video_id"

    partitions = split["partitions"]
    project_train = set(partitions["project_train"]["video_ids"])
    development = set(partitions["development_validation"]["video_ids"])
    final_test = set(partitions["final_test"]["video_ids"])

    assert len(project_train) == 25
    assert development == {
        "BEE24-03", "BEE24-09", "BEE24-17", "BEE24-22", "BEE24-25", "BEE24-26",
    }
    assert final_test == {"BEE24-12", "BEE24-13", "BEE24-16", "BEE24-34", "BEE24-36"}
    assert project_train.isdisjoint(development)
    assert project_train.isdisjoint(final_test)
    assert development.isdisjoint(final_test)

    inventory = split["source_inventory"]
    assert project_train | development == set(inventory["train_video_ids"])
    assert final_test == set(inventory["test_video_ids"])
    assert partitions["final_test"]["may_select_hyperparameters"] is False
    assert split["selection"]["final_test"]["used_for_method_selection"] is False


def test_frozen_project_split_retains_h1_provenance():
    root = Path(__file__).resolve().parents[1]
    split = yaml.safe_load(
        (root / "configs" / "splits" / "project_split.yaml").read_text(encoding="utf-8")
    )
    provenance = split["provenance"]

    assert provenance["project_git_commit"] == "9a92b27dce432ad6ce79e0b71dff4a801a7b0afe"
    assert provenance["manifest_sha256"] == (
        "aec25bde8c9a023d78a96c85dbdaab4740f2ce5ebc08d01b53f1e2f334691bac"
    )
