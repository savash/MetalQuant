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


def make_cache(model, backend: str = "baseline") -> list:
    """Create a per-layer cache list for *model* using the named backend.

    Args:
        model: An mlx-lm model with a .layers attribute.
        backend: One of "baseline", "int8".  More will be added.

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
        return [TurboQuantMSECache(bits=2, layer_idx=i) for i in range(len(model.layers))]

    if backend == "tq3":
        from metalquant.cache_turboquant import TurboQuantMSECache
        return [TurboQuantMSECache(bits=3, layer_idx=i) for i in range(len(model.layers))]

    if backend == "tq4":
        from metalquant.cache_turboquant import TurboQuantMSECache
        return [TurboQuantMSECache(bits=4, layer_idx=i) for i in range(len(model.layers))]

    raise ValueError(
        f"Unknown cache backend: {backend!r}.  "
        f"Valid: baseline, int8, turboquant (tq2), tq3, tq4"
    )
