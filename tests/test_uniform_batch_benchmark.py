from __future__ import annotations

import time
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

import jax_cudss


class Scenario(NamedTuple):
    name: str
    batch_size: int
    matrix_size: int
    offsets: tuple[int, ...]


SMALL_DENSE_FAVORABLE = Scenario(
    "small_banded_dense_favorable", 1_000, 100, (-2, -1, 0, 1, 2)
)
SPARSE_FAVORABLE = Scenario(
    "larger_tridiagonal_sparse_favorable", 100, 256, (-1, 0, 1)
)
SCALING_BATCHES = (1, 10, 100, 1_000)
RESIDUAL_RTOL = 5e-4


def _require_gpu() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("These benchmarks require the JAX GPU backend.")


def _require_cudss() -> None:
    if not jax_cudss.has_cudss_binding():
        pytest.skip(f"cuDSS binding unavailable: {jax_cudss.cudss_import_error()!r}")


def _pattern(
    matrix_size: int, offsets: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_indices: list[int] = []
    col_indices: list[int] = []
    indptr = [0]
    diag_positions = np.full(matrix_size, -1, dtype=np.int32)
    for row in range(matrix_size):
        count_before = len(row_indices)
        for offset in offsets:
            col = row + offset
            if 0 <= col < matrix_size:
                row_indices.append(row)
                col_indices.append(col)
                if col == row:
                    diag_positions[row] = len(col_indices) - 1
        indptr.append(len(row_indices))
        assert len(row_indices) > count_before
    return (
        np.asarray(indptr, dtype=np.int32),
        np.asarray(row_indices, dtype=np.int32),
        np.asarray(col_indices, dtype=np.int32),
        diag_positions,
    )


def _problem_from_pattern(
    *,
    batch_size: int,
    matrix_size: int,
    indptr_np: np.ndarray,
    row_np: np.ndarray,
    col_np: np.ndarray,
    diag_np: np.ndarray,
) -> dict[str, jax.Array]:
    rng = np.random.default_rng(0)
    nnz = row_np.shape[0]

    values = rng.uniform(-0.05, 0.05, size=(batch_size, nnz)).astype(np.float32)
    rhs = rng.uniform(-1.0, 1.0, size=(matrix_size,)).astype(np.float32)

    offdiag_mask = row_np != col_np
    values[:, ~offdiag_mask] = 0.0
    for row, diag_pos in enumerate(diag_np.tolist()):
        row_mask = row_np == row
        row_abs = np.abs(values[:, row_mask]).sum(axis=1)
        values[:, diag_pos] = 1.0 + row_abs + rng.uniform(
            0.1, 0.5, size=(batch_size,)
        ).astype(np.float32)

    dense = np.zeros(
        (batch_size, matrix_size, matrix_size),
        dtype=np.float32,
    )
    dense[:, row_np, col_np] = values

    indptr = jnp.asarray(indptr_np)
    indices = jnp.asarray(col_np)
    values_jax = jnp.asarray(values)
    rhs_single = jnp.asarray(rhs)
    rhs_batch = jnp.broadcast_to(rhs_single, (batch_size, matrix_size))
    dense_batch = jnp.asarray(dense)
    return {
        "indptr": indptr,
        "indices": indices,
        "values": values_jax,
        "rhs_single": rhs_single,
        "rhs_batch": rhs_batch,
        "dense_batch": dense_batch,
    }


@lru_cache(maxsize=None)
def benchmark_problem(scenario: Scenario) -> dict[str, jax.Array]:
    indptr_np, row_np, col_np, diag_np = _pattern(
        scenario.matrix_size, scenario.offsets
    )
    return _problem_from_pattern(
        batch_size=scenario.batch_size,
        matrix_size=scenario.matrix_size,
        indptr_np=indptr_np,
        row_np=row_np,
        col_np=col_np,
        diag_np=diag_np,
    )


def _time_call(fn, *args, repeats: int = 3) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    return min(times)


@lru_cache(maxsize=None)
def compiled_solvers(scenario: Scenario) -> dict[str, object]:
    problem = benchmark_problem(scenario)
    solve_cudss = jax.jit(
        lambda values, rhs: jax_cudss.uniform_batch_solve_csr(
            problem["indptr"], problem["indices"], values, rhs
        )
    )
    solve_lu = jax.jit(
        lambda dense, rhs: jax.vmap(
            lambda matrix, vec: jsp_linalg.lu_solve(jsp_linalg.lu_factor(matrix), vec)
        )(dense, rhs)
    )

    try:
        cudss_warm = solve_cudss(problem["values"], problem["rhs_single"])
        jax.block_until_ready(cudss_warm)
    except jax.errors.JaxRuntimeError as exc:
        if "CUDSS_STATUS_ALLOC_FAILED" in str(exc):
            pytest.skip(
                f"Full-batch cuDSS factorization for scenario {scenario.name!r} "
                "exhausted device memory on this GPU."
            )
        raise

    lu_warm = solve_lu(problem["dense_batch"], problem["rhs_batch"])
    jax.block_until_ready(lu_warm)

    return {"cudss": solve_cudss, "lu": solve_lu}


def _assert_small_residual(
    dense_batch: jax.Array, rhs_batch: jax.Array, x: jax.Array
) -> None:
    residual = jnp.einsum("bij,bj->bi", dense_batch, x) - rhs_batch
    max_residual = float(jnp.max(jnp.linalg.norm(residual, axis=1)))
    assert max_residual < RESIDUAL_RTOL, f"max residual too large: {max_residual}"


def test_cudss_profile_reports_phase_timings_and_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_PROFILE", "1")
    jax_cudss.clear_cudss_last_profile()

    indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
    indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
    values = jnp.array(
        [[4.0, 0.2, 5.0, 6.0], [5.0, 0.3, 6.0, 7.0]], dtype=jnp.float32
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    solve = jax.jit(
        lambda vals, vec: jax_cudss.uniform_batch_solve_csr(indptr, indices, vals, vec)
    )

    first = jax.block_until_ready(solve(values, rhs))
    first_profile = jax_cudss.cudss_last_profile()
    assert first_profile is not None
    assert first_profile["cache_hit"] is False
    for field in ("setup_ms", "analysis_ms", "factorization_ms", "solve_ms"):
        assert field in first_profile
        assert first_profile[field] >= 0.0
    assert first_profile["batch"] == 2
    assert first_profile["n"] == 3
    assert first_profile["nnz"] == 4

    second = jax.block_until_ready(solve(values, rhs))
    second_profile = jax_cudss.cudss_last_profile()
    assert second_profile is not None
    assert second_profile["cache_hit"] is True
    assert second_profile["setup_ms"] <= first_profile["setup_ms"]
    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=1e-6, atol=1e-6)


def test_cudss_scaling_smoke() -> None:
    _require_gpu()
    _require_cudss()

    timings_ms: list[tuple[int, float]] = []
    for batch_size in SCALING_BATCHES:
        scenario = Scenario(
            f"scaling_batch_{batch_size}",
            batch_size,
            SMALL_DENSE_FAVORABLE.matrix_size,
            SMALL_DENSE_FAVORABLE.offsets,
        )
        problem = benchmark_problem(scenario)
        solve = jax.jit(
            lambda values, rhs: jax_cudss.uniform_batch_solve_csr(
                problem["indptr"], problem["indices"], values, rhs
            )
        )
        result = jax.block_until_ready(solve(problem["values"], problem["rhs_single"]))
        _assert_small_residual(problem["dense_batch"], problem["rhs_batch"], result)
        timings_ms.append(
            (batch_size, _time_call(solve, problem["values"], problem["rhs_single"]) * 1e3)
        )

    print(
        "\ncuDSS scaling timings (ms): "
        + ", ".join(f"batch={batch}: {timing:.3f}" for batch, timing in timings_ms)
    )


def test_cudss_uniform_batch_smoke() -> None:
    _require_gpu()
    _require_cudss()
    indptr = jnp.array([0, 1, 2], dtype=jnp.int32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    values = jnp.array([[2.0, 3.0], [4.0, 5.0]], dtype=jnp.float32)
    rhs = jnp.array([8.0, 15.0], dtype=jnp.float32)
    solve = jax.jit(
        lambda vals, vec: jax_cudss.uniform_batch_solve_csr(indptr, indices, vals, vec)
    )
    result = solve(values, rhs)
    result = jax.block_until_ready(result)
    np.testing.assert_allclose(
        np.asarray(result),
        np.asarray([[4.0, 5.0], [2.0, 3.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.benchmark(group="uniform-batch-solves")
@pytest.mark.parametrize(
    "scenario",
    [SMALL_DENSE_FAVORABLE, SPARSE_FAVORABLE],
    ids=lambda scenario: scenario.name,
)
def test_cudss_uniform_batch_benchmark(
    benchmark: pytest.BenchmarkFixture,
    scenario: Scenario,
) -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(scenario)
    solvers = compiled_solvers(scenario)

    def run() -> jax.Array:
        return jax.block_until_ready(
            solvers["cudss"](problem["values"], problem["rhs_single"])
        )

    result = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    _assert_small_residual(problem["dense_batch"], problem["rhs_batch"], result)


@pytest.mark.benchmark(group="uniform-batch-solves")
@pytest.mark.parametrize(
    "scenario",
    [SMALL_DENSE_FAVORABLE, SPARSE_FAVORABLE],
    ids=lambda scenario: scenario.name,
)
def test_jax_lu_uniform_batch_benchmark(
    benchmark: pytest.BenchmarkFixture, scenario: Scenario
) -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(scenario)
    solvers = compiled_solvers(scenario)

    def run() -> jax.Array:
        return jax.block_until_ready(
            solvers["lu"](problem["dense_batch"], problem["rhs_batch"])
        )

    result = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    _assert_small_residual(problem["dense_batch"], problem["rhs_batch"], result)

    cudss_result = solvers["cudss"](problem["values"], problem["rhs_single"])
    cudss_result = jax.block_until_ready(cudss_result)
    np.testing.assert_allclose(
        np.asarray(cudss_result),
        np.asarray(result),
        rtol=5e-3,
        atol=5e-3,
    )

    cudss_time = _time_call(solvers["cudss"], problem["values"], problem["rhs_single"])
    lu_time = _time_call(solvers["lu"], problem["dense_batch"], problem["rhs_batch"])
    ratio = lu_time / cudss_time
    print(
        f"\nscenario={scenario.name} manual timing: "
        f"cudss={cudss_time:.6f}s jax_lu={lu_time:.6f}s speedup={ratio:.3f}x"
    )
