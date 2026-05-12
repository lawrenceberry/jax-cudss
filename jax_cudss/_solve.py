from __future__ import annotations

import functools
import hashlib
import ctypes
import importlib.metadata as md
import weakref
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_IMPORT_ERROR: Exception | None = None
_EXTENSION: Any | None = None


def _preload_cuda_libraries() -> None:
    try:
        cudss_dist = md.distribution("nvidia-cudss-cu13")
    except md.PackageNotFoundError:
        return

    lib_dir = Path(cudss_dist.locate_file("nvidia/cu13/lib"))
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for library in ("libcudart.so.13", "libcudss.so.0"):
        path = lib_dir / library
        if path.exists():
            ctypes.CDLL(str(path), mode=mode)


try:
    _preload_cuda_libraries()
    from . import _cudss as _EXTENSION
except Exception as exc:  # pragma: no cover - exercised indirectly in tests
    _IMPORT_ERROR = exc


def cudss_import_error() -> Exception | None:
    """Return the exception raised while importing the native cuDSS extension.

    Returns ``None`` when the optional ``jax_cudss._cudss`` extension imported
    successfully. If the extension is unavailable, returns the original import
    exception so callers can report or inspect why GPU cuDSS support is missing.
    """
    return _IMPORT_ERROR


def has_cudss_binding() -> bool:
    """Return whether the native cuDSS extension is available.

    This is a lightweight availability check for the optional
    ``jax_cudss._cudss`` extension. It does not validate that the active JAX
    backend is GPU-capable.
    """
    return _EXTENSION is not None


def cudss_last_profile() -> dict[str, Any] | None:
    """Return profiling data recorded by the most recent cuDSS setup call.

    Returns ``None`` when the native extension is unavailable. Profiling is
    controlled by the native extension and is primarily useful for inspecting
    setup, analysis, allocation, and cache behavior.
    """
    if _EXTENSION is None:
        return None
    return _EXTENSION.last_profile()


def clear_cudss_last_profile() -> None:
    """Clear the native extension's most recent cuDSS profiling record.

    This is a no-op when the optional native extension is unavailable.
    """
    if _EXTENSION is not None:
        _EXTENSION.clear_last_profile()


def _release_prepared_solver(token: int) -> None:
    if _EXTENSION is not None:
        _EXTENSION.release_prepared_solver(int(token))


@jax.tree_util.register_pytree_node_class
class PreparedSolverHandle:
    """Opaque handle for a prepared cuDSS CSR solver.

    Instances are returned by :func:`setup_solver_csr` and passed to
    :func:`factorize_graph_csr` and :func:`solve_graph_csr`. The handle stores
    metadata about the prepared CSR structure and owns a native prepared solver
    token while the Python object is alive.
    """

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
_FACTORIZE_CMD_BUFFER_COMPATIBLE: bool | None = None
_SOLVE_CMD_BUFFER_COMPATIBLE: bool | None = None


@functools.lru_cache(maxsize=1)
def _register_ffi_targets() -> tuple[str, str, str]:
    global _FACTORIZE_CMD_BUFFER_COMPATIBLE, _SOLVE_CMD_BUFFER_COMPATIBLE
    if _EXTENSION is None:
        raise RuntimeError("cuDSS extension is unavailable") from _IMPORT_ERROR

    registrations = _EXTENSION.registrations()
    for name, capsule in registrations.items():
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA", api_version=1)
        if name == _FACTORIZE_TARGET_NAME:
            _FACTORIZE_CMD_BUFFER_COMPATIBLE = True
        elif name == _SOLVE_TARGET_NAME:
            _SOLVE_CMD_BUFFER_COMPATIBLE = True
    return _SETUP_TARGET_NAME, _FACTORIZE_TARGET_NAME, _SOLVE_TARGET_NAME


def factorize_graph_is_cmd_buffer_compatible() -> bool:
    """Return whether the factorize FFI handler is command-buffer compatible.

    Registers the FFI targets if needed and reports whether
    ``factorize_graph_csr`` is exposed with command-buffer-compatible XLA FFI
    handler metadata. Requires a JAX GPU backend and the native cuDSS extension.
    """
    _validate_backend()
    _register_ffi_targets()
    return bool(_FACTORIZE_CMD_BUFFER_COMPATIBLE)


def solve_graph_is_cmd_buffer_compatible() -> bool:
    """Return whether the solve FFI handler is command-buffer compatible.

    Registers the FFI targets if needed and reports whether
    ``solve_graph_csr`` is exposed with command-buffer-compatible XLA FFI
    handler metadata. Requires a JAX GPU backend and the native cuDSS extension.
    """
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
    """Prepare or reuse a cuDSS solver for a batched CSR sparsity structure.

    ``indptr`` and ``indices`` describe a square CSR matrix structure and are
    converted to rank-1 ``int32`` JAX arrays. ``batch_size`` is the number of
    matrices that will share this sparsity pattern. The native extension hashes
    the structure and creates or reuses a cached cuDSS analysis object for the
    active GPU device.

    This setup path performs host synchronization to materialize the native
    solver token and is expected to run outside repeated CUDA Graph /
    command-buffer execution. Use the returned handle with
    :func:`factorize_graph_csr` and :func:`solve_graph_csr`.
    """
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
    """Factorize batched CSR matrix values for a prepared solver.

    ``handle`` must be a :class:`PreparedSolverHandle` returned by
    :func:`setup_solver_csr`. ``values`` is converted to ``float32`` and must
    have shape ``[handle.batch_size, handle.nnz]``. Each row contains the
    numeric nonzero values for one matrix in the batch, using the CSR structure
    captured by ``handle``.

    The call updates native cuDSS factorization state and returns no JAX value.
    It is intended to be used inside ``jax.jit`` as a side-effecting FFI custom
    call, and its native handler is marked command-buffer compatible.
    """
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
    """Solve batched CSR systems using the latest factorization for a handle.

    ``handle`` must be a :class:`PreparedSolverHandle` that has already been
    factorized with :func:`factorize_graph_csr`. ``b`` is converted to
    ``float32`` and may have shape ``[handle.n]`` or
    ``[handle.batch_size, handle.n]``. A rank-1 right-hand side is broadcast
    across the batch.

    Returns a ``float32`` JAX array with shape
    ``[handle.batch_size, handle.n]``. The native handler is marked
    command-buffer compatible for use in jitted repeated solve paths.
    """
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
