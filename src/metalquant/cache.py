"""Cache backend abstraction for MetalQuant experiments.

The benchmark runner accepts any list of objects that conform to the
CacheBackend protocol.  mlx-lm's KVCache satisfies this protocol
natively, so BaselineCacheBackend is just a thin name alias used to
make the runner's intent explicit.

Custom backends (e.g. quantized, compressed) subclass CacheBackend and
override update() to intercept K/V tensors before storage.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import mlx.core as mx
from mlx_lm.models.cache import KVCache


@runtime_checkable
class CacheBackend(Protocol):
    """Structural protocol matching mlx-lm KVCache public API.

    Any object exposing these attributes can be passed directly to an
    mlx-lm model as its cache list.
    """

    @property
    def state(self) -> list[mx.array]:
        ...

    @property
    def nbytes(self) -> int:
        ...

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        ...


class BaselineCacheBackend(KVCache):
    """Unmodified mlx-lm KVCache — the zero-compression baseline."""

    pass


def make_cache(model, backend: str = "baseline", calibration: list | None = None) -> list:
    """Create a per-layer cache list for *model* using the named backend.

    Args:
        model: An mlx-lm model with a .layers attribute.
        backend: One of "baseline", "int8", "turboquant"/"tq2"/"tq3"/"tq4",
            "tq-outlier".
        calibration: For "tq-outlier" — list of per-layer dicts from
            metalquant.calibrate.load_calibration().

    Returns:
        A list of cache objects, one per transformer layer.
    """
    if backend == "baseline":
        return [BaselineCacheBackend() for _ in range(len(model.layers))]

    if backend == "int8":
        from metalquant.cache_quantized import INT8CacheBackend
        return [INT8CacheBackend() for _ in range(len(model.layers))]

    if backend in ("turboquant", "tq2"):
        from metalquant.cache_turboquant import TurboQuantMSECache
        k_stds = [calibration[i]["k_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        v_stds = [calibration[i]["v_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        return [TurboQuantMSECache(bits=2, layer_idx=i, k_ch_stds=k_stds[i], v_ch_stds=v_stds[i]) for i in range(len(model.layers))]

    if backend == "tq3":
        from metalquant.cache_turboquant import TurboQuantMSECache
        k_stds = [calibration[i]["k_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        v_stds = [calibration[i]["v_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        return [TurboQuantMSECache(bits=3, layer_idx=i, k_ch_stds=k_stds[i], v_ch_stds=v_stds[i]) for i in range(len(model.layers))]

    if backend == "tq4":
        from metalquant.cache_turboquant import TurboQuantMSECache
        k_stds = [calibration[i]["k_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        v_stds = [calibration[i]["v_ch_stds"] for i in range(len(model.layers))] if calibration else [None] * len(model.layers)
        return [TurboQuantMSECache(bits=4, layer_idx=i, k_ch_stds=k_stds[i], v_ch_stds=v_stds[i]) for i in range(len(model.layers))]

    if backend == "tq-outlier":
        from metalquant.cache_turboquant_v2 import TurboQuantOutlierCache
        if calibration is None:
            raise ValueError(
                "tq-outlier requires calibration data. "
                "Run benchmarks/run_calibrate.py first and pass --calibration <path>."
            )
        return [
            TurboQuantOutlierCache(
                outlier_channels=calibration[i]["outlier_channels"],
                bits_outlier=3,
                bits_regular=2,
                layer_idx=i,
                k_ch_stds=calibration[i].get("k_ch_stds"),
                v_ch_stds=calibration[i].get("v_ch_stds"),
            )
            for i in range(len(model.layers))
        ]

    if backend in ("fp16-outlier", "tq-fp16"):
        from metalquant.cache_fp16outlier import TurboQuantFp16OutlierCache
        if calibration is None:
            raise ValueError(
                "fp16-outlier requires calibration data. "
                "Run benchmarks/run_calibrate.py first and pass --calibration <path>."
            )
        return [
            TurboQuantFp16OutlierCache(
                outlier_channels=calibration[i]["outlier_channels"],
                bits=4,
                layer_idx=i,
            )
            for i in range(len(model.layers))
        ]

    if backend == "fp16-outlier-tq2":
        from metalquant.cache_fp16outlier import TurboQuantFp16OutlierCache
        if calibration is None:
            raise ValueError(
                "fp16-outlier-tq2 requires calibration data. "
                "Run benchmarks/run_calibrate.py first and pass --calibration <path>."
            )
        return [
            TurboQuantFp16OutlierCache(
                outlier_channels=calibration[i]["outlier_channels"],
                bits=2,
                layer_idx=i,
            )
            for i in range(len(model.layers))
        ]

    raise ValueError(
        f"Unknown cache backend: {backend!r}.  "
        f"Valid: baseline, int8, turboquant (tq2), tq3, tq4, tq-outlier, "
        f"fp16-outlier, fp16-outlier-tq2"
    )
