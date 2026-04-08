from __future__ import annotations

import importlib.metadata as md
import os
import warnings
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _dist_path(dist_name: str, relative: str) -> Path | None:
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return None
    path = Path(dist.locate_file(relative))
    return path if path.exists() else None


def _dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _discover_cudss() -> tuple[list[str], list[str]]:
    include_dirs: list[str] = []
    lib_dirs: list[str] = []

    cudss_include = os.environ.get("CUDSS_INCLUDE_DIR")
    cudss_lib = os.environ.get("CUDSS_LIBRARY_DIR")
    cudss_root = os.environ.get("CUDSS_ROOT")
    if cudss_include and cudss_lib:
        include_dirs.append(cudss_include)
        lib_dirs.append(cudss_lib)
    elif cudss_root:
        include_dirs.append(str(Path(cudss_root) / "include"))
        lib_dirs.append(str(Path(cudss_root) / "lib"))
    else:
        include = _dist_path("nvidia-cudss-cu12", "nvidia/cu12/include")
        lib = _dist_path("nvidia-cudss-cu12", "nvidia/cu12/lib")
        if include and lib:
            include_dirs.append(str(include))
            lib_dirs.append(str(lib))

    cuda_runtime_include = os.environ.get("CUDA_RUNTIME_INCLUDE_DIR")
    cuda_runtime_lib = os.environ.get("CUDA_RUNTIME_LIBRARY_DIR")
    if cuda_runtime_include:
        include_dirs.append(cuda_runtime_include)
    else:
        include = _dist_path("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/include")
        if include:
            include_dirs.append(str(include))
    if cuda_runtime_lib:
        lib_dirs.append(cuda_runtime_lib)
    else:
        lib = _dist_path("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/lib")
        if lib:
            lib_dirs.append(str(lib))

    return _dedupe(include_dirs), _dedupe(lib_dirs)


def _discover_cudss_shared_object(lib_dirs: list[str]) -> str | None:
    for lib_dir in lib_dirs:
        for candidate in ("libcudss.so", "libcudss.so.0"):
            path = Path(lib_dir) / candidate
            if path.exists():
                return str(path)
    return None


def _discover_cuda_runtime_shared_object(lib_dirs: list[str]) -> str | None:
    for lib_dir in lib_dirs:
        for candidate in ("libcudart.so", "libcudart.so.12"):
            path = Path(lib_dir) / candidate
            if path.exists():
                return str(path)
    return None


def _discover_jax_include() -> str | None:
    try:
        import jax.ffi
    except Exception:
        return None
    include_dir = Path(jax.ffi.include_dir())
    return str(include_dir) if include_dir.exists() else None


class OptionalBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        include_dirs, lib_dirs = _discover_cudss()
        cudss_shared_object = _discover_cudss_shared_object(lib_dirs)
        cudart_shared_object = _discover_cuda_runtime_shared_object(lib_dirs)
        jax_include = _discover_jax_include()
        if (
            not include_dirs
            or not lib_dirs
            or cudss_shared_object is None
            or cudart_shared_object is None
            or jax_include is None
        ):
            warnings.warn(
                "Skipping cuDSS extension build because JAX or cuDSS headers/libraries "
                "were not found. Install the optional CUDA dependencies or set "
                "CUDSS_ROOT/CUDSS_INCLUDE_DIR/CUDSS_LIBRARY_DIR and "
                "CUDA_RUNTIME_LIBRARY_DIR as needed.",
                stacklevel=2,
            )
            return

        ext.include_dirs = _dedupe([*ext.include_dirs, *include_dirs, jax_include])
        ext.library_dirs = _dedupe([*ext.library_dirs, *lib_dirs])
        ext.libraries = []
        ext.extra_objects = [cudss_shared_object, cudart_shared_object]
        if os.name != "nt":
            ext.runtime_library_dirs = _dedupe(
                [*(ext.runtime_library_dirs or []), *lib_dirs]
            )
        super().build_extension(ext)


setup(
    ext_modules=[
        Extension(
            "jax_cudss._cudss",
            sources=["jax_cudss/_cudss.cc"],
            include_dirs=[],
            library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
