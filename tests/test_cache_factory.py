from __future__ import annotations

import importlib
import sys
import types

import pytest


class DummyKVCache:
    def __init__(self) -> None:
        self.offset = 0
        self.keys = None
        self.values = None


class RecorderBackend:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class DummyModel:
    def __init__(self, layer_count: int) -> None:
        self.layers = [object() for _ in range(layer_count)]


def load_cache_module(monkeypatch: pytest.MonkeyPatch):
    mlx_module = types.ModuleType("mlx")
    mlx_core_module = types.ModuleType("mlx.core")
    mlx_core_module.array = object
    mlx_module.core = mlx_core_module

    mlx_lm_module = types.ModuleType("mlx_lm")
    mlx_lm_models_module = types.ModuleType("mlx_lm.models")
    mlx_lm_cache_module = types.ModuleType("mlx_lm.models.cache")
    mlx_lm_cache_module.KVCache = DummyKVCache
    mlx_lm_models_module.cache = mlx_lm_cache_module
    mlx_lm_module.models = mlx_lm_models_module

    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", mlx_lm_cache_module)

    quantized_module = types.ModuleType("metalquant.cache_quantized")
    quantized_module.INT8CacheBackend = RecorderBackend
    monkeypatch.setitem(sys.modules, "metalquant.cache_quantized", quantized_module)

    turboquant_module = types.ModuleType("metalquant.cache_turboquant")
    turboquant_module.TurboQuantMSECache = RecorderBackend
    monkeypatch.setitem(sys.modules, "metalquant.cache_turboquant", turboquant_module)

    outlier_module = types.ModuleType("metalquant.cache_turboquant_v2")
    outlier_module.TurboQuantOutlierCache = RecorderBackend
    monkeypatch.setitem(sys.modules, "metalquant.cache_turboquant_v2", outlier_module)

    fp16_module = types.ModuleType("metalquant.cache_fp16outlier")
    fp16_module.TurboQuantFp16OutlierCache = RecorderBackend
    monkeypatch.setitem(sys.modules, "metalquant.cache_fp16outlier", fp16_module)

    sys.modules.pop("metalquant.cache", None)
    return importlib.import_module("metalquant.cache")


def test_make_cache_builds_baseline_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_module = load_cache_module(monkeypatch)

    caches = cache_module.make_cache(DummyModel(layer_count=3), backend="baseline")

    assert len(caches) == 3
    assert all(isinstance(cache, cache_module.BaselineCacheBackend) for cache in caches)


def test_make_cache_passes_calibration_to_tq4(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_module = load_cache_module(monkeypatch)
    calibration = [
        {"k_ch_stds": [1.0, 2.0], "v_ch_stds": [3.0, 4.0]},
        {"k_ch_stds": [5.0, 6.0], "v_ch_stds": [7.0, 8.0]},
    ]

    caches = cache_module.make_cache(DummyModel(layer_count=2), backend="tq4", calibration=calibration)

    assert [cache.kwargs for cache in caches] == [
        {"bits": 4, "layer_idx": 0, "k_ch_stds": [1.0, 2.0], "v_ch_stds": [3.0, 4.0]},
        {"bits": 4, "layer_idx": 1, "k_ch_stds": [5.0, 6.0], "v_ch_stds": [7.0, 8.0]},
    ]


def test_make_cache_requires_calibration_for_outlier_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_module = load_cache_module(monkeypatch)
    model = DummyModel(layer_count=1)

    with pytest.raises(ValueError, match="tq-outlier requires calibration data"):
        cache_module.make_cache(model, backend="tq-outlier")

    with pytest.raises(ValueError, match="fp16-outlier requires calibration data"):
        cache_module.make_cache(model, backend="fp16-outlier")


def test_make_cache_builds_fp16_outlier_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_module = load_cache_module(monkeypatch)
    calibration = [{"outlier_channels": [1, 5, 9]}]

    caches = cache_module.make_cache(DummyModel(layer_count=1), backend="fp16-outlier", calibration=calibration)

    assert caches[0].kwargs == {
        "outlier_channels": [1, 5, 9],
        "bits": 4,
        "layer_idx": 0,
    }


def test_make_cache_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_module = load_cache_module(monkeypatch)

    with pytest.raises(ValueError, match="Unknown cache backend"):
        cache_module.make_cache(DummyModel(layer_count=1), backend="nope")
