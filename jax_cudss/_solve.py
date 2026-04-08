from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

_IMPORT_ERROR: Exception | None = None
_EXTENSION: Any | None = None

try:
    from . import _cudss as _EXTENSION
except Exception as exc:  # pragma: no cover - exercised indirectly in tests
    _IMPORT_ERROR = exc


def cudss_import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_cudss_binding() -> bool:
    return _EXTENSION is not None


def cudss_last_profile() -> dict[str, Any] | None:
    if _EXTENSION is None:
        return None
    return _EXTENSION.last_profile()


def clear_cudss_last_profile() -> None:
    if _EXTENSION is not None:
        _EXTENSION.clear_last_profile()


@functools.lru_cache(maxsize=1)
def _register_ffi_target() -> str:
    if _EXTENSION is None:
        raise RuntimeError("cuDSS extension is unavailable") from _IMPORT_ERROR

    target_name = "jax_cudss_uniform_batch_solve"
    registrations = _EXTENSION.registrations()
    for name, capsule in registrations.items():
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA", api_version=1)
    return target_name


def _validate_csr_inputs(
    indptr: jax.Array, indices: jax.Array, values: jax.Array, rhs: jax.Array
) -> tuple[int, int]:
    if jax.default_backend() != "gpu":
        raise ValueError("uniform_batch_solve_csr requires the GPU backend")
    if indptr.ndim != 1:
        raise ValueError(f"indptr must be rank-1, got shape {indptr.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank-1, got shape {indices.shape}")
    if values.ndim != 2:
        raise ValueError(f"values must be rank-2 [batch, nnz], got shape {values.shape}")
    if rhs.ndim != 2:
        raise ValueError(f"rhs must be rank-2 [batch, n], got shape {rhs.shape}")
    if indptr.dtype != jnp.int32:
        raise ValueError(f"indptr must have dtype int32, got {indptr.dtype}")
    if indices.dtype != jnp.int32:
        raise ValueError(f"indices must have dtype int32, got {indices.dtype}")
    if values.dtype != jnp.float32:
        raise ValueError(f"values must have dtype float32, got {values.dtype}")
    if rhs.dtype != jnp.float32:
        raise ValueError(f"rhs must have dtype float32, got {rhs.dtype}")

    n = indptr.shape[0] - 1
    if n <= 0:
        raise ValueError("indptr must describe a non-empty square matrix")
    nnz = indices.shape[0]
    if values.shape[1] != nnz:
        raise ValueError(
            f"values second dimension must equal nnz={nnz}, got {values.shape[1]}"
        )
    if rhs.shape[0] != values.shape[0]:
        raise ValueError(
            "rhs batch dimension must match values batch dimension, got "
            f"{rhs.shape[0]} and {values.shape[0]}"
        )
    if rhs.shape[1] != n:
        raise ValueError(
            f"rhs second dimension must equal matrix size n={n}, got {rhs.shape[1]}"
        )
    return values.shape[0], n


def uniform_batch_solve_csr(
    indptr: jax.Array | Any,
    indices: jax.Array | Any,
    values: jax.Array | Any,
    b: jax.Array | Any,
) -> jax.Array:
    if _EXTENSION is None:
        raise RuntimeError(
            "cuDSS extension is unavailable. Install the optional CUDA dependency "
            "and build the extension first."
        ) from _IMPORT_ERROR

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    if b.ndim == 1:
        if values.ndim != 2:
            raise ValueError(
                "values must be rank-2 [batch, nnz] when broadcasting a single rhs"
            )
        b = jnp.broadcast_to(b, (values.shape[0], b.shape[0]))
    elif b.ndim != 2:
        raise ValueError(f"b must be rank-1 or rank-2, got shape {b.shape}")

    batch, n = _validate_csr_inputs(indptr, indices, values, b)
    result_spec = jax.ShapeDtypeStruct((batch, n), jnp.float32)
    ffi = jax.ffi.ffi_call(_register_ffi_target(), result_spec)
    return ffi(indptr, indices, values, b)
