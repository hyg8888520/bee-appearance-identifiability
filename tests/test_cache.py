from __future__ import annotations

import json

import numpy as np
import pytest

from beeid.cache.features import CacheError, FeatureCache, cache_fingerprint, open_cache


def test_cache_atomic_write_resume_and_roundtrip(tmp_path):
    signature = {"manifest": "abc", "model": {"name": "m"}, "input_size": 224}
    cache = FeatureCache(tmp_path, "m", signature)
    ids = ["a", "b"]
    embeddings = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    path = cache.write_shard(0, ids, embeddings)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp.npz"))
    assert cache.validate_shard(0, ids)
    assert not cache.validate_shard(0, ["b", "a"])
    np.testing.assert_array_equal(cache.load_all(ids, 2), embeddings)
    assert open_cache(cache.directory, "m").fingerprint == cache.fingerprint


def test_cache_selected_load_ignores_historical_shard_boundaries(tmp_path):
    cache = FeatureCache(tmp_path, "m", {"x": 1})
    cache.write_shard(0, ["a", "b"], np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    cache.write_shard(1, ["c"], np.asarray([[1, 0]], dtype=np.float32))
    selected = cache.load_selected(["c", "a"])
    np.testing.assert_array_equal(selected, np.asarray([[1, 0], [1, 0]], dtype=np.float32))
    with pytest.raises(CacheError, match="do not exactly match"):
        cache.load_selected(["c", "a"], require_exact_ids=True)
    with pytest.raises(CacheError, match="missing 1 requested"):
        cache.load_selected(["missing"])


def test_cache_signature_is_stable_and_metadata_mismatch_refused(tmp_path):
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert cache_fingerprint(first) == cache_fingerprint(second)
    cache = FeatureCache(tmp_path, "m", first)
    cache.initialize()
    cache.metadata_path.write_text(json.dumps({"fingerprint": "wrong"}), encoding="utf-8")
    with pytest.raises(CacheError, match="does not match"):
        cache.initialize()


def test_cache_rejects_non_normalized_or_nonfinite(tmp_path):
    cache = FeatureCache(tmp_path, "m", {"x": 1})
    with pytest.raises(CacheError, match="L2-normalized"):
        cache.write_shard(0, ["a"], np.asarray([[2.0, 0.0]], dtype=np.float32))
    with pytest.raises(CacheError, match="non-finite"):
        cache.write_shard(0, ["a"], np.asarray([[np.nan, 0.0]], dtype=np.float32))
