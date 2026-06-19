from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

from .environment import (
    _default_extensions_dir,
    _freeze_jit_config,
    _unfreeze_jit_config,
    configure_jit,
    get_jit_config,
    validate_jit_environment,
)


_EXTENSION_LOCK = threading.Lock()
_EXTENSION: ModuleType | None = None
_BINARY_MODULE = "torch_blend._torch_blend_cuda"
_SOURCE_SUFFIXES = {".cpp", ".cu", ".h"}


def _import_binary_extension() -> ModuleType:
    return importlib.import_module(_BINARY_MODULE)


def _extension_sources() -> tuple[list[str], Path]:
    extension_dir = Path(__file__).resolve().parent / "ext"
    sources = [
        extension_dir / "bindings.cpp",
        extension_dir / "blend.cpp",
        extension_dir / "blend_cpu.cpp",
        extension_dir / "blend_cuda.cu",
    ]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "Torch Blend JIT source files are missing from the installed package:\n"
            f"{missing_list}\n"
            "Reinstall from a wheel or source distribution that includes "
            "torch_blend/ext."
        )
    return [str(path) for path in sources], extension_dir


def _source_digest(extension_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(extension_dir.iterdir()):
        if path.suffix not in _SOURCE_SUFFIXES:
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _jit_module_name(extension_dir: Path) -> str:
    return f"torch_blend_cuda_jit_{_source_digest(extension_dir)}"


def _jit_build_directory() -> Path:
    _, extension_dir = _extension_sources()
    config = get_jit_config()
    extensions_dir = config.extensions_dir or _default_extensions_dir()
    return extensions_dir / _jit_module_name(extension_dir)


def clear_jit_cache(*, force: bool = False) -> Path:
    """Remove the cache for the current Torch Blend JIT source version."""
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            raise RuntimeError(
                "Cannot clear the JIT cache after the native extension is loaded."
            )

        build_directory = _jit_build_directory()
        lock_path = build_directory / "lock"
        if lock_path.exists() and not force:
            raise RuntimeError(
                "The JIT cache contains a build lock. Another process may be "
                f"compiling the extension: {lock_path}\n"
                "Pass force=True only after confirming no build is running."
            )
        if build_directory.exists():
            shutil.rmtree(build_directory)
        return build_directory


def build_jit_extension() -> ModuleType:
    """Force JIT loading now instead of waiting for the first blend call."""
    configure_jit(force_jit=True)
    return get_extension()


def _wait_for_build_lock(
    build_directory: Path,
    timeout_seconds: float,
    verbose: bool,
) -> None:
    lock_path = build_directory / "lock"
    if not lock_path.exists():
        return

    started = time.monotonic()
    lock_age = max(0.0, time.time() - lock_path.stat().st_mtime)
    if verbose:
        print(
            "Torch Blend JIT: another build lock exists:\n"
            f"  Lock: {lock_path}\n"
            f"  Age:  {lock_age:.1f}s\n"
            f"  Waiting up to {timeout_seconds:.1f}s for it to clear...",
            file=sys.stderr,
            flush=True,
        )

    while lock_path.exists():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                "Timed out waiting for an existing PyTorch JIT build lock.\n"
                f"Lock file: {lock_path}\n"
                f"Initial lock age: {lock_age:.1f}s\n"
                "If no other JIT build is running, remove this stale lock and retry:\n"
                f"  rm '{lock_path}'"
            )
        time.sleep(min(0.2, timeout_seconds - elapsed))


def _build_jit_extension() -> ModuleType:
    sources, extension_dir = _extension_sources()
    environment = validate_jit_environment()
    config = get_jit_config()
    module_name = _jit_module_name(extension_dir)
    build_directory = environment.extensions_dir / module_name
    build_directory.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils import cpp_extension

        cpp_extension.CUDA_HOME = str(environment.cuda_home)
        _wait_for_build_lock(
            build_directory,
            config.lock_timeout_seconds,
            config.verbose,
        )
        if config.verbose:
            print(
                "Torch Blend JIT: starting CUDA extension build:\n"
                f"  Module: {module_name}\n"
                f"  Build directory: {build_directory}\n"
                f"  CUDA architectures: {environment.cuda_arch_list}\n"
                f"  Sources: {len(sources)}",
                file=sys.stderr,
                flush=True,
            )

        return cpp_extension.load(
            name=module_name,
            sources=sources,
            build_directory=str(build_directory),
            extra_include_paths=[str(extension_dir)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=config.verbose,
            with_cuda=True,
            keep_intermediates=config.verbose,
        )
    except Exception as error:
        raise RuntimeError(
            "Torch Blend JIT compilation failed.\n"
            f"CUDA_HOME: {environment.cuda_home}\n"
            f"NVCC: {environment.nvcc_path} (CUDA {environment.nvcc_version})\n"
            f"PyTorch CUDA: {environment.torch_cuda_version}\n"
            f"CUDA architectures: {environment.cuda_arch_list}\n"
            f"Build cache: {environment.extensions_dir}\n"
            f"Original error: {error}\n"
            "Run with TORCH_BLEND_JIT_VERBOSE=1 for full compiler output, or build "
            "manually with 'python setup.py build_ext --inplace'."
        ) from error


def get_extension() -> ModuleType:
    """Load the prebuilt native extension or lazily compile a JIT fallback."""
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION

        _freeze_jit_config()
        config = get_jit_config()
        if config.disabled and config.force_jit:
            _unfreeze_jit_config()
            raise RuntimeError("JIT cannot be both disabled and forced.")

        if not config.force_jit:
            try:
                _EXTENSION = _import_binary_extension()
                return _EXTENSION
            except ModuleNotFoundError as error:
                if error.name != _BINARY_MODULE:
                    _unfreeze_jit_config()
                    raise
            except (ImportError, OSError) as error:
                _unfreeze_jit_config()
                raise RuntimeError(
                    "The prebuilt Torch Blend extension exists but could not be loaded. "
                    "This usually indicates a PyTorch/CUDA ABI mismatch. Rebuild it with "
                    "'python setup.py build_ext --inplace --force'. JIT fallback was not "
                    "attempted because silently replacing an incompatible binary can hide "
                    "deployment errors."
                ) from error

        if config.disabled:
            _unfreeze_jit_config()
            raise RuntimeError(
                "Torch Blend native extension is unavailable and JIT compilation is "
                "disabled by TORCH_BLEND_DISABLE_JIT=1. Build manually with:\n"
                "  python setup.py build_ext --inplace"
            )

        try:
            _EXTENSION = _build_jit_extension()
        except Exception:
            _unfreeze_jit_config()
            raise
        return _EXTENSION


def _reset_extension_for_tests() -> None:
    global _EXTENSION
    with _EXTENSION_LOCK:
        _EXTENSION = None
