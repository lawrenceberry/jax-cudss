# jax-cudss

`jax-cudss` is a small experimental JAX package that exposes NVIDIA cuDSS
sparse direct solves through JAX FFI.

Its main use case is solving batches of CSR sparse linear systems on GPU where
every matrix in the batch shares the same sparsity pattern, but has different
`float32` matrix values.

The package is currently focused on a prepared-solver workflow:

1. Create or reuse a cuDSS analysis object for a CSR sparsity structure.
2. Factorize batched matrix values for that structure.
3. Solve one or more right-hand sides using the prepared factorization.

## Current state

This is an early `0.1.0` package. The core Python API and native cuDSS binding
are implemented, along with GPU integration tests and benchmarks.

Implemented pieces include:

- Python public API in `jax_cudss/_solve.py`.
- Native C++/CUDA/cuDSS FFI extension in `jax_cudss/_cudss.cc`.
- Optional extension build logic in `setup.py`.
- Package metadata in `pyproject.toml`.
- GPU-only correctness, cache, eviction, timing, and benchmark tests in
  `tests/test_uniform_batch_benchmark.py`.

The native binding is optional at import time. If `jax_cudss._cudss` cannot be
imported, the Python package still imports, and callers can inspect availability
with:

```python
import jax_cudss

jax_cudss.has_cudss_binding()
jax_cudss.cudss_import_error()
```

The extension build is skipped if JAX, cuDSS, or CUDA runtime headers/libraries
cannot be found.

## Requirements

- Python `>=3.13`
- JAX `>=0.9.2`
- A JAX GPU backend
- NVIDIA cuDSS for CUDA 12

The optional CUDA dependency group declares:

```toml
jax[cuda12]>=0.9.2
nvidia-cudss-cu12>=0.7.1.6
```

The build script can discover cuDSS and CUDA runtime files from the NVIDIA
Python wheels. It can also use explicit environment variables:

- `CUDSS_ROOT`
- `CUDSS_INCLUDE_DIR`
- `CUDSS_LIBRARY_DIR`
- `CUDA_RUNTIME_INCLUDE_DIR`
- `CUDA_RUNTIME_LIBRARY_DIR`

## API overview

The public API is re-exported from `jax_cudss/__init__.py`.

```python
import jax
import jax.numpy as jnp
import jax_cudss

indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)

values = jnp.array(
    [
        [2.0, 0.1, 3.0, 4.0],
        [5.0, 0.2, 6.0, 7.0],
    ],
    dtype=jnp.float32,
)
rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

handle = jax_cudss.setup_solver_csr(indptr, indices, batch_size=2)

factorize = jax.jit(
    lambda vals: (
        jax_cudss.factorize_graph_csr(handle, vals),
        jnp.int8(0),
    )[1]
)
solve = jax.jit(lambda b: jax_cudss.solve_graph_csr(handle, b))

jax.block_until_ready(factorize(values))
x = jax.block_until_ready(solve(rhs))
```

### `setup_solver_csr(indptr, indices, *, batch_size)`

Creates a `PreparedSolverHandle` for a CSR sparsity pattern.

- `indptr` must be rank-1 `int32`.
- `indices` must be rank-1 `int32`.
- `batch_size` must be positive.
- The active JAX backend must be GPU.

Internally, the package computes a hash of the CSR structure and asks the native
extension to create or reuse a cuDSS analysis object.

### `factorize_graph_csr(handle, values)`

Factorizes matrix values for a prepared CSR structure.

- `handle` must be a `PreparedSolverHandle`.
- `values` must have shape `[batch_size, nnz]`.
- `values` must be `float32`.

This call has side effects in the native cuDSS handle and returns `None`.

### `solve_graph_csr(handle, b)`

Solves using the most recent factorization attached to `handle`.

- `b` may have shape `[n]`, in which case it is broadcast across the batch.
- `b` may also have shape `[batch_size, n]`.
- `b` must be `float32`.

`factorize_graph_csr` must be called successfully before `solve_graph_csr`.

## Structure

### `jax_cudss/__init__.py`

Re-exports the public API:

- `PreparedSolverHandle`
- `setup_solver_csr`
- `factorize_graph_csr`
- `solve_graph_csr`
- binding availability helpers
- command-buffer compatibility helpers
- profiling helpers

### `jax_cudss/_solve.py`

Python-facing wrapper around the native extension.

Responsibilities include:

- Importing the optional `_cudss` extension.
- Registering JAX FFI targets.
- Validating CSR, value, and RHS shapes/dtypes.
- Hashing CSR structure on the host.
- Managing `PreparedSolverHandle` lifetime.
- Releasing native prepared solver ownership with `weakref.finalize`.
- Broadcasting rank-1 right-hand sides across the batch.

### `jax_cudss/_cudss.cc`

Native C++/CUDA/cuDSS implementation.

Responsibilities include:

- Creating cuDSS handles, configs, data objects, and matrix descriptors.
- Running cuDSS analysis, factorization, and solve phases.
- Registering JAX FFI custom calls:
  - `jax_cudss_setup_solver`
  - `jax_cudss_factorize_graph`
  - `jax_cudss_solve_graph`
- Caching prepared solvers by device, cuDSS options, CSR structure hash, matrix
  size, nonzero count, and batch size.
- Tracking Python-side handle ownership.
- Evicting unused prepared solvers with an LRU policy.
- Supporting CUDA async allocation through cuDSS device memory handlers.
- Exposing lightweight profiling and debug hooks.

### `setup.py`

Builds the optional `jax_cudss._cudss` extension.

It discovers headers and libraries from environment variables or installed
NVIDIA Python packages, adds JAX FFI headers, and links against cuDSS and the
CUDA runtime shared objects.

### `tests/test_uniform_batch_benchmark.py`

Contains GPU-only tests and benchmarks covering:

- cuDSS binding availability.
- setup profiling and cache reuse.
- command-buffer compatibility registration.
- LRU eviction of prepared solver objects.
- input validation.
- solve-before-factorize errors.
- factorization reuse.
- repeated factorization.
- correctness against dense residual checks.
- comparisons with batched dense JAX LU.
- timing grids for analysis, factorization, solve, and scaling.

Tests skip automatically when the JAX GPU backend or cuDSS binding is
unavailable.

## Runtime configuration

The native extension reads several environment variables:

- `JAX_CUDSS_PROFILE`: enable last-setup profiling data.
- `JAX_CUDSS_DEBUG_LOG`: print debug logging to stderr.
- `JAX_CUDSS_REORDERING_ALG`: choose a cuDSS reordering algorithm.
- `JAX_CUDSS_ND_NLEVELS`: set nested-dissection levels.
- `JAX_CUDSS_ENABLE_MT`: enable cuDSS host multithreading.
- `JAX_CUDSS_HOST_NTHREADS`: set cuDSS host thread count.
- `JAX_CUDSS_THREADING_LIB`: choose the cuDSS threading library.
- `CUDSS_THREADING_LIB`: fallback threading library setting.
- `JAX_CUDSS_MAX_PREPARED_SOLVERS`: prepared solver cache capacity.

Profiling data can be inspected from Python:

```python
jax_cudss.clear_cudss_last_profile()
handle = jax_cudss.setup_solver_csr(indptr, indices, batch_size=2)
profile = jax_cudss.cudss_last_profile()
```

## Limitations

- GPU backend only.
- CSR structure only.
- `indptr` and `indices` must be `int32`.
- Matrix values and right-hand sides must be `float32`.
- Matrices are assumed to be square.
- Every matrix in a batch must share the same sparsity pattern.
- `factorize_graph_csr` must be called before `solve_graph_csr`.
- The current API separates setup, factorization, and solve explicitly.
- The README and package surface are still minimal; this is not yet a polished
  end-user library.

## Development notes

Use the `uv` environment where available:

```bash
uv run python -m pytest
```

The tests use the GPU and should be run one after another so that they do not
conflict over GPU memory or cuDSS resources.
