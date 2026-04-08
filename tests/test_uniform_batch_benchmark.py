from __future__ import annotations

import gc
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
TIMING_DIMENSIONS = (10, 100, 1_000)
TIMING_BATCHES = (10, 100, 1_000)
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
    include_dense: bool = True,
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

    indptr = jnp.asarray(indptr_np)
    row_indices = jnp.asarray(row_np)
    indices = jnp.asarray(col_np)
    values_jax = jnp.asarray(values)
    rhs_single = jnp.asarray(rhs)
    rhs_batch = jnp.broadcast_to(rhs_single, (batch_size, matrix_size))
    problem = {
        "indptr": indptr,
        "row_indices": row_indices,
        "indices": indices,
        "values": values_jax,
        "rhs_single": rhs_single,
        "rhs_batch": rhs_batch,
    }
    if include_dense:
        dense = np.zeros((batch_size, matrix_size, matrix_size), dtype=np.float32)
        dense[:, row_np, col_np] = values
        problem["dense_batch"] = jnp.asarray(dense)
    return problem


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
        include_dense=True,
    )


def sparse_problem(
    batch_size: int, matrix_size: int, offsets: tuple[int, ...]
) -> dict[str, jax.Array]:
    indptr_np, row_np, col_np, diag_np = _pattern(matrix_size, offsets)
    return _problem_from_pattern(
        batch_size=batch_size,
        matrix_size=matrix_size,
        indptr_np=indptr_np,
        row_np=row_np,
        col_np=col_np,
        diag_np=diag_np,
        include_dense=False,
    )


def _time_call(fn, *args, repeats: int = 3) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    return min(times)


def compiled_solvers(scenario: Scenario) -> dict[str, object]:
    problem = benchmark_problem(scenario)
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=scenario.batch_size
    )
    factorize_cudss = jax.jit(
        lambda vals: (
            jax_cudss.factorize_graph_csr(handle, vals),
            jnp.int8(0),
        )[1]
    )
    solve_cudss = jax.jit(lambda rhs: jax_cudss.solve_graph_csr(handle, rhs))
    solve_end_to_end_cudss = jax.jit(
        lambda vals, rhs: (
            jax_cudss.factorize_graph_csr(handle, vals),
            jax_cudss.solve_graph_csr(handle, rhs),
        )[1]
    )
    solve_lu = jax.jit(
        lambda dense, rhs: jax.vmap(
            lambda matrix, vec: jsp_linalg.lu_solve(jsp_linalg.lu_factor(matrix), vec)
        )(dense, rhs)
    )

    try:
        cudss_warm = solve_end_to_end_cudss(problem["values"], problem["rhs_single"])
        jax.block_until_ready(cudss_warm)
    except jax.errors.JaxRuntimeError as exc:
        if "CUDSS_STATUS_ALLOC_FAILED" in str(exc):
            pytest.skip(
                f"Prepared cuDSS solve for scenario {scenario.name!r} "
                "exhausted device memory on this GPU."
            )
        raise

    lu_warm = solve_lu(problem["dense_batch"], problem["rhs_batch"])
    jax.block_until_ready(lu_warm)

    return {
        "handle": handle,
        "cudss_factorize": factorize_cudss,
        "cudss_solve": solve_cudss,
        "cudss_end_to_end": solve_end_to_end_cudss,
        "lu": solve_lu,
    }


def _assert_small_residual(
    dense_batch: jax.Array, rhs_batch: jax.Array, x: jax.Array
) -> None:
    residual = jnp.einsum("bij,bj->bi", dense_batch, x) - rhs_batch
    max_residual = float(jnp.max(jnp.linalg.norm(residual, axis=1)))
    assert max_residual < RESIDUAL_RTOL, f"max residual too large: {max_residual}"


def _assert_small_sparse_residual(problem: dict[str, jax.Array], x: jax.Array) -> None:
    rows = problem["row_indices"]
    cols = problem["indices"]
    values = problem["values"]
    rhs_batch = problem["rhs_batch"]
    ax = jnp.zeros_like(rhs_batch).at[:, rows].add(values * x[:, cols])
    residual = ax - rhs_batch
    max_residual = float(jnp.max(jnp.linalg.norm(residual, axis=1)))
    assert max_residual < RESIDUAL_RTOL, f"max residual too large: {max_residual}"


def test_cudss_setup_profile_reports_analysis_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_PROFILE", "1")
    jax_cudss.clear_cudss_last_profile()

    indptr = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
    indices = jnp.array([0, 1, 1, 2], dtype=jnp.int32)

    first = jax_cudss.setup_solver_csr(indptr, indices, batch_size=2)
    first_profile = jax_cudss.cudss_last_profile()
    assert first_profile is not None
    assert first_profile["cache_hit"] is False
    assert first_profile["analysis_ms"] >= 0.0
    assert first_profile["async_allocator_enabled"] is True
    assert first_profile["token"] == first.token
    assert first_profile["batch"] == 2
    assert first_profile["n"] == 3
    assert first_profile["nnz"] == 4

    second = jax_cudss.setup_solver_csr(indptr, indices, batch_size=2)
    second_profile = jax_cudss.cudss_last_profile()
    assert second_profile is not None
    assert second_profile["cache_hit"] is True
    assert second_profile["token"] == first.token
    assert second.token == first.token


def test_graph_calls_are_command_buffer_compatible() -> None:
    _require_gpu()
    _require_cudss()
    assert jax_cudss.factorize_graph_is_cmd_buffer_compatible() is True
    assert jax_cudss.solve_graph_is_cmd_buffer_compatible() is True


def test_cudss_lru_eviction_only_affects_unowned_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_MAX_PREPARED_SOLVERS", "1")

    base = benchmark_problem(Scenario("base", 2, 8, (-1, 0, 1)))
    alt = benchmark_problem(Scenario("alt", 2, 8, (-2, -1, 0, 1)))

    handle_a = jax_cudss.setup_solver_csr(base["indptr"], base["indices"], batch_size=2)
    token_a = handle_a.token

    handle_b = jax_cudss.setup_solver_csr(alt["indptr"], alt["indices"], batch_size=2)
    assert handle_b.token != token_a

    handle_a_again = jax_cudss.setup_solver_csr(
        base["indptr"], base["indices"], batch_size=2
    )
    assert handle_a_again.token == token_a

    del handle_a
    del handle_a_again
    gc.collect()

    handle_c = jax_cudss.setup_solver_csr(
        base["indptr"], base["indices"], batch_size=2
    )
    assert handle_c.token != token_a


def test_cudss_factorize_graph_validates_batch_and_nnz() -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(Scenario("validation", 2, 8, (-1, 0, 1)))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=2
    )

    with pytest.raises(ValueError, match="prepared nnz"):
        jax_cudss.factorize_graph_csr(handle, problem["values"][:, :-1])

    with pytest.raises(ValueError, match="prepared batch_size"):
        jax_cudss.factorize_graph_csr(handle, problem["values"][:1])


def test_cudss_solve_graph_validates_rhs_shape() -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(Scenario("solve_validation", 2, 8, (-1, 0, 1)))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=2
    )

    with pytest.raises(ValueError, match="rhs must have shape"):
        jax_cudss.solve_graph_csr(handle, problem["rhs_batch"][:1])


def test_cudss_solve_requires_factorization() -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(Scenario("solve_requires_factorization", 2, 8, (-1, 0, 1)))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=2
    )
    solve = jax.jit(lambda rhs: jax_cudss.solve_graph_csr(handle, rhs))

    with pytest.raises(
        jax.errors.JaxRuntimeError, match="call factorize_graph_csr first"
    ):
        jax.block_until_ready(solve(problem["rhs_single"]))


def test_cudss_rebuilds_when_sparsity_changes() -> None:
    _require_gpu()
    _require_cudss()
    left = benchmark_problem(Scenario("left", 2, 8, (-1, 0, 1)))
    right = benchmark_problem(Scenario("right", 2, 8, (-2, -1, 0, 1)))

    handle_left = jax_cudss.setup_solver_csr(
        left["indptr"], left["indices"], batch_size=2
    )
    handle_right = jax_cudss.setup_solver_csr(
        right["indptr"], right["indices"], batch_size=2
    )

    assert handle_left.structure_hash != handle_right.structure_hash
    assert handle_left.token != handle_right.token


def test_cudss_repeated_solve_reuses_factorization() -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(Scenario("repeated_solve", 2, 8, (-1, 0, 1)))
    rhs_alt = problem["rhs_single"] * 0.5
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=2
    )
    factorize = jax.jit(
        lambda vals: (
            jax_cudss.factorize_graph_csr(handle, vals),
            jnp.int8(0),
        )[1]
    )
    solve = jax.jit(lambda rhs: jax_cudss.solve_graph_csr(handle, rhs))

    jax.block_until_ready(factorize(problem["values"]))
    result_first = jax.block_until_ready(solve(problem["rhs_single"]))
    result_second = jax.block_until_ready(solve(rhs_alt))

    _assert_small_residual(problem["dense_batch"], problem["rhs_batch"], result_first)
    rhs_alt_batch = jnp.broadcast_to(rhs_alt, problem["rhs_batch"].shape)
    _assert_small_residual(problem["dense_batch"], rhs_alt_batch, result_second)


def test_cudss_repeated_factorization_overwrites_prior_factors() -> None:
    _require_gpu()
    _require_cudss()
    problem = benchmark_problem(Scenario("refactorize", 2, 8, (-1, 0, 1)))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=2
    )
    factorize = jax.jit(
        lambda vals: (
            jax_cudss.factorize_graph_csr(handle, vals),
            jnp.int8(0),
        )[1]
    )
    solve = jax.jit(lambda rhs: jax_cudss.solve_graph_csr(handle, rhs))

    first_values = problem["values"]
    second_values = problem["values"] * jnp.float32(1.5)
    dense_second = problem["dense_batch"] * jnp.float32(1.5)

    jax.block_until_ready(factorize(first_values))
    _ = jax.block_until_ready(solve(problem["rhs_single"]))
    jax.block_until_ready(factorize(second_values))
    result = jax.block_until_ready(solve(problem["rhs_single"]))

    _assert_small_residual(dense_second, problem["rhs_batch"], result)


@pytest.mark.parametrize("matrix_size", TIMING_DIMENSIONS)
def test_cudss_analysis_timing_by_dimension(
    monkeypatch: pytest.MonkeyPatch, matrix_size: int
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_PROFILE", "1")
    monkeypatch.setenv("JAX_CUDSS_MAX_PREPARED_SOLVERS", "1")

    problem = sparse_problem(100, matrix_size, (-1, 0, 1))
    jax_cudss.clear_cudss_last_profile()
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=100
    )
    profile = jax_cudss.cudss_last_profile()
    assert profile is not None
    assert profile["analysis_ms"] >= 0.0
    print(
        f"\ncuDSS analysis timing (ms): batch=100, n={matrix_size}, "
        f"analysis={float(profile['analysis_ms']):.3f}"
    )

    del handle
    gc.collect()


@pytest.mark.parametrize("matrix_size", TIMING_DIMENSIONS)
@pytest.mark.parametrize("batch_size", TIMING_BATCHES)
def test_cudss_prepared_factorize_timing_grid(
    monkeypatch: pytest.MonkeyPatch, matrix_size: int, batch_size: int
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_MAX_PREPARED_SOLVERS", "1")
    if matrix_size == 1_000 and batch_size == 1_000:
        pytest.skip(
            "Prepared cuDSS solve timing for n=1000, batch=1000 exceeds the "
            "memory budget on this GPU."
        )

    problem = sparse_problem(batch_size, matrix_size, (-1, 0, 1))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=batch_size
    )
    factorize = jax.jit(
        lambda values: (
            jax_cudss.factorize_graph_csr(handle, values),
            jnp.int8(0),
        )[1]
    )
    _ = jax.block_until_ready(factorize(problem["values"]))
    timing_ms = _time_call(factorize, problem["values"]) * 1e3
    print(
        f"\ncuDSS prepared factorize timing (ms): n={matrix_size}, "
        f"batch={batch_size}, factorize={timing_ms:.3f}"
    )

    del factorize
    del handle
    gc.collect()


@pytest.mark.parametrize("matrix_size", TIMING_DIMENSIONS)
@pytest.mark.parametrize("batch_size", TIMING_BATCHES)
def test_cudss_prepared_solve_timing_grid(
    monkeypatch: pytest.MonkeyPatch, matrix_size: int, batch_size: int
) -> None:
    _require_gpu()
    _require_cudss()
    monkeypatch.setenv("JAX_CUDSS_MAX_PREPARED_SOLVERS", "1")
    if matrix_size == 1_000 and batch_size == 1_000:
        pytest.skip(
            "Prepared cuDSS solve timing for n=1000, batch=1000 exceeds the "
            "memory budget on this GPU."
        )

    problem = sparse_problem(batch_size, matrix_size, (-1, 0, 1))
    handle = jax_cudss.setup_solver_csr(
        problem["indptr"], problem["indices"], batch_size=batch_size
    )
    factorize = jax.jit(
        lambda values: (
            jax_cudss.factorize_graph_csr(handle, values),
            jnp.int8(0),
        )[1]
    )
    solve = jax.jit(lambda rhs: jax_cudss.solve_graph_csr(handle, rhs))
    jax.block_until_ready(factorize(problem["values"]))
    result = jax.block_until_ready(solve(problem["rhs_single"]))
    _assert_small_sparse_residual(problem, result)
    timing_ms = _time_call(solve, problem["rhs_single"]) * 1e3
    print(
        f"\ncuDSS prepared solve timing (ms): n={matrix_size}, "
        f"batch={batch_size}, solve={timing_ms:.3f}"
    )

    del result
    del solve
    del factorize
    del handle
    gc.collect()


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
        handle = jax_cudss.setup_solver_csr(
            problem["indptr"], problem["indices"], batch_size=batch_size
        )
        end_to_end = jax.jit(
            lambda values, rhs: (
                jax_cudss.factorize_graph_csr(handle, values),
                jax_cudss.solve_graph_csr(handle, rhs),
            )[1]
        )
        try:
            result = jax.block_until_ready(
                end_to_end(problem["values"], problem["rhs_single"])
            )
        except jax.errors.JaxRuntimeError as exc:
            if "CUDSS_STATUS_ALLOC_FAILED" in str(exc):
                pytest.skip(
                    f"Prepared cuDSS solve for scaling batch {batch_size} "
                    "exhausted device memory on this GPU."
                )
            raise
        _assert_small_residual(problem["dense_batch"], problem["rhs_batch"], result)
        timings_ms.append(
            (
                batch_size,
                _time_call(end_to_end, problem["values"], problem["rhs_single"]) * 1e3,
            )
        )
        del result
        del end_to_end
        del handle
        gc.collect()

    print(
        "\ncuDSS scaling timings (ms): "
        + ", ".join(f"batch={batch}: {timing:.3f}" for batch, timing in timings_ms)
    )


def test_cudss_prepared_handle_smoke() -> None:
    _require_gpu()
    _require_cudss()
    indptr = jnp.array([0, 1, 2], dtype=jnp.int32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    values = jnp.array([[2.0, 3.0], [4.0, 5.0]], dtype=jnp.float32)
    rhs = jnp.array([8.0, 15.0], dtype=jnp.float32)
    handle = jax_cudss.setup_solver_csr(indptr, indices, batch_size=2)
    factorize = jax.jit(
        lambda solver_handle, vals: (
            jax_cudss.factorize_graph_csr(solver_handle, vals),
            jnp.int8(0),
        )[1]
    )
    solve = jax.jit(lambda solver_handle, vec: jax_cudss.solve_graph_csr(solver_handle, vec))
    jax.block_until_ready(factorize(handle, values))
    result = jax.block_until_ready(solve(handle, rhs))
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
            solvers["cudss_end_to_end"](problem["values"], problem["rhs_single"])
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

    cudss_result = solvers["cudss_end_to_end"](problem["values"], problem["rhs_single"])
    cudss_result = jax.block_until_ready(cudss_result)
    np.testing.assert_allclose(
        np.asarray(cudss_result),
        np.asarray(result),
        rtol=5e-3,
        atol=5e-3,
    )

    cudss_time = _time_call(
        solvers["cudss_end_to_end"], problem["values"], problem["rhs_single"]
    )
    lu_time = _time_call(solvers["lu"], problem["dense_batch"], problem["rhs_batch"])
    ratio = lu_time / cudss_time
    print(
        f"\nscenario={scenario.name} manual timing: "
        f"cudss={cudss_time:.6f}s jax_lu={lu_time:.6f}s speedup={ratio:.3f}x"
    )
