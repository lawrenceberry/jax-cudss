from __future__ import annotations

import functools
import hashlib
import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib import xla_client

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


def _release_prepared_solver(token: int) -> None:
    if _EXTENSION is not None:
        _EXTENSION.release_prepared_solver(int(token))


@jax.tree_util.register_pytree_node_class
class PreparedSolverHandle:
    def __init__(
        self,
        *,
        token: int,
        n: int,
        nnz: int,
        batch_size: int,
        device_ordinal: int,
        structure_hash: int,
        finalize: bool = True,
    ) -> None:
        self.token = int(token)
        self.n = int(n)
        self.nnz = int(nnz)
        self.batch_size = int(batch_size)
        self.device_ordinal = int(device_ordinal)
        self.structure_hash = int(structure_hash)
        self._finalizer = (
            weakref.finalize(self, _release_prepared_solver, self.token)
            if finalize
            else None
        )

    def tree_flatten(self) -> tuple[tuple[()], tuple[int, int, int, int, int, int]]:
        return (), (
            self.token,
            self.n,
            self.nnz,
            self.batch_size,
            self.device_ordinal,
            self.structure_hash,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[int, int, int, int, int, int], children: tuple[()]
    ) -> "PreparedSolverHandle":
        del children
        token, n, nnz, batch_size, device_ordinal, structure_hash = aux_data
        return cls(
            token=token,
            n=n,
            nnz=nnz,
            batch_size=batch_size,
            device_ordinal=device_ordinal,
            structure_hash=structure_hash,
            finalize=False,
        )

    def __repr__(self) -> str:
        return (
            "PreparedSolverHandle("
            f"token={self.token}, n={self.n}, nnz={self.nnz}, "
            f"batch_size={self.batch_size}, device_ordinal={self.device_ordinal}, "
            f"structure_hash={self.structure_hash})"
        )


_SETUP_TARGET_NAME = "jax_cudss_setup_solver"
_FACTORIZE_TARGET_NAME = "jax_cudss_factorize_graph"
_SOLVE_TARGET_NAME = "jax_cudss_solve_graph"
_GRAPH_TARGET_TRAITS = xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE
_FACTORIZE_CMD_BUFFER_COMPATIBLE: bool | None = None
_SOLVE_CMD_BUFFER_COMPATIBLE: bool | None = None


@functools.lru_cache(maxsize=1)
def _register_ffi_targets() -> tuple[str, str, str]:
    global _FACTORIZE_CMD_BUFFER_COMPATIBLE, _SOLVE_CMD_BUFFER_COMPATIBLE
    if _EXTENSION is None:
        raise RuntimeError("cuDSS extension is unavailable") from _IMPORT_ERROR

    registrations = _EXTENSION.registrations()
    for name, capsule in registrations.items():
        if name == _SETUP_TARGET_NAME:
            jax.ffi.register_ffi_target(name, capsule, platform="CUDA", api_version=1)
            continue
        try:
            jax.ffi.register_ffi_target(
                name,
                capsule,
                platform="CUDA",
                api_version=1,
                traits=_GRAPH_TARGET_TRAITS,
            )
        except jax.errors.JaxRuntimeError as exc:
            if "does not support custom call traits" in str(exc):
                raise RuntimeError(
                    "The current JAX CUDA plugin does not support "
                    "COMMAND_BUFFER_COMPATIBLE custom call traits, so "
                    "factorize_graph_csr/solve_graph_csr cannot be used."
                ) from exc
            raise
        if name == _FACTORIZE_TARGET_NAME:
            _FACTORIZE_CMD_BUFFER_COMPATIBLE = True
        elif name == _SOLVE_TARGET_NAME:
            _SOLVE_CMD_BUFFER_COMPATIBLE = True
    return _SETUP_TARGET_NAME, _FACTORIZE_TARGET_NAME, _SOLVE_TARGET_NAME


def factorize_graph_is_cmd_buffer_compatible() -> bool:
    _validate_backend()
    _register_ffi_targets()
    return bool(_FACTORIZE_CMD_BUFFER_COMPATIBLE)


def solve_graph_is_cmd_buffer_compatible() -> bool:
    _validate_backend()
    _register_ffi_targets()
    return bool(_SOLVE_CMD_BUFFER_COMPATIBLE)


def _current_device_ordinal() -> int:
    devices = jax.devices("gpu")
    return int(devices[0].id) if devices else 0


def _structure_hash(indptr: jax.Array, indices: jax.Array) -> int:
    indptr_host = np.asarray(jax.device_get(indptr), dtype=np.int32, order="C")
    indices_host = np.asarray(jax.device_get(indices), dtype=np.int32, order="C")
    digest = hashlib.blake2b(digest_size=8)
    digest.update(indptr_host.tobytes())
    digest.update(indices_host.tobytes())
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _validate_backend() -> None:
    if jax.default_backend() != "gpu":
        raise ValueError("cuDSS bindings require the GPU backend")


def _validate_csr_structure(indptr: jax.Array, indices: jax.Array) -> tuple[int, int]:
    if indptr.ndim != 1:
        raise ValueError(f"indptr must be rank-1, got shape {indptr.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank-1, got shape {indices.shape}")
    if indptr.dtype != jnp.int32:
        raise ValueError(f"indptr must have dtype int32, got {indptr.dtype}")
    if indices.dtype != jnp.int32:
        raise ValueError(f"indices must have dtype int32, got {indices.dtype}")
    n = indptr.shape[0] - 1
    if n <= 0:
        raise ValueError("indptr must describe a non-empty square matrix")
    nnz = indices.shape[0]
    return n, nnz


def _prepare_rhs(values: jax.Array, b: jax.Array) -> jax.Array:
    b = jnp.asarray(b, dtype=jnp.float32)
    if b.ndim == 1:
        if values.ndim != 2:
            raise ValueError(
                "values must be rank-2 [batch, nnz] when broadcasting a single rhs"
            )
        b = jnp.broadcast_to(b, (values.shape[0], b.shape[0]))
    elif b.ndim != 2:
        raise ValueError(f"b must be rank-1 or rank-2, got shape {b.shape}")
    return b


def _validate_factorize_inputs(
    handle: PreparedSolverHandle, values: jax.Array
) -> tuple[int, int]:
    if not isinstance(handle, PreparedSolverHandle):
        raise TypeError("handle must be a PreparedSolverHandle")
    if values.ndim != 2:
        raise ValueError(f"values must be rank-2 [batch, nnz], got shape {values.shape}")
    if values.dtype != jnp.float32:
        raise ValueError(f"values must have dtype float32, got {values.dtype}")
    if values.shape[0] != handle.batch_size:
        raise ValueError(
            f"values batch dimension must equal prepared batch_size={handle.batch_size}, "
            f"got {values.shape[0]}"
        )
    if values.shape[1] != handle.nnz:
        raise ValueError(
            f"values second dimension must equal prepared nnz={handle.nnz}, "
            f"got {values.shape[1]}"
        )
    return handle.batch_size, handle.nnz


def _validate_solve_inputs(
    handle: PreparedSolverHandle, rhs: jax.Array
) -> tuple[int, int]:
    if not isinstance(handle, PreparedSolverHandle):
        raise TypeError("handle must be a PreparedSolverHandle")
    if rhs.ndim != 2:
        raise ValueError(f"rhs must be rank-2 [batch, n], got shape {rhs.shape}")
    if rhs.dtype != jnp.float32:
        raise ValueError(f"rhs must have dtype float32, got {rhs.dtype}")
    if rhs.shape[0] != handle.batch_size or rhs.shape[1] != handle.n:
        raise ValueError(
            f"rhs must have shape [{handle.batch_size}, {handle.n}], got {rhs.shape}"
        )
    return handle.batch_size, handle.n


def setup_solver_csr(
    indptr: jax.Array | Any,
    indices: jax.Array | Any,
    *,
    batch_size: int,
) -> PreparedSolverHandle:
    if _EXTENSION is None:
        raise RuntimeError(
            "cuDSS extension is unavailable. Install the optional CUDA dependency "
            "and build the extension first."
        ) from _IMPORT_ERROR

    _validate_backend()
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    n, nnz = _validate_csr_structure(indptr, indices)
    structure_hash = _structure_hash(indptr, indices)

    setup_target, _, _ = _register_ffi_targets()
    result_spec = jax.ShapeDtypeStruct((), jnp.int64)
    ffi = jax.ffi.ffi_call(setup_target, result_spec, has_side_effect=True)
    token_array = ffi(
        indptr,
        indices,
        batch_size=int(batch_size),
        structure_hash=structure_hash,
    )
    token = int(np.asarray(jax.device_get(jax.block_until_ready(token_array))).item())
    return PreparedSolverHandle(
        token=token,
        n=n,
        nnz=nnz,
        batch_size=int(batch_size),
        device_ordinal=_current_device_ordinal(),
        structure_hash=structure_hash,
    )


def factorize_graph_csr(
    handle: PreparedSolverHandle,
    values: jax.Array | Any,
) -> None:
    if _EXTENSION is None:
        raise RuntimeError(
            "cuDSS extension is unavailable. Install the optional CUDA dependency "
            "and build the extension first."
        ) from _IMPORT_ERROR

    _validate_backend()
    values = jnp.asarray(values, dtype=jnp.float32)
    _validate_factorize_inputs(handle, values)

    _, factorize_target, _ = _register_ffi_targets()
    ffi = jax.ffi.ffi_call(factorize_target, [], has_side_effect=True)
    ffi(values, token=handle.token)
    return None


def solve_graph_csr(
    handle: PreparedSolverHandle,
    b: jax.Array | Any,
) -> jax.Array:
    if _EXTENSION is None:
        raise RuntimeError(
            "cuDSS extension is unavailable. Install the optional CUDA dependency "
            "and build the extension first."
        ) from _IMPORT_ERROR

    _validate_backend()
    rhs = _prepare_rhs(
        jnp.empty((handle.batch_size, handle.nnz), dtype=jnp.float32),
        b,
    )
    batch, n = _validate_solve_inputs(handle, rhs)

    _, _, solve_target = _register_ffi_targets()
    result_spec = jax.ShapeDtypeStruct((batch, n), jnp.float32)
    ffi = jax.ffi.ffi_call(solve_target, result_spec, has_side_effect=True)
    return ffi(rhs, token=handle.token)
