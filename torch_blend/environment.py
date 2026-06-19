from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch


class JITEnvironmentError(RuntimeError):
    """Raised when the local toolchain cannot build the native extension."""


@dataclass(frozen=True)
class JITConfig:
    disabled: bool = False
    force_jit: bool = False
    verbose: bool = False
    lock_timeout_seconds: float = 30.0
    extensions_dir: Optional[Path] = None
    cuda_arch_list: Optional[str] = None
    cuda_home: Optional[Path] = None


@dataclass(frozen=True)
class JITEnvironment:
    torch_cuda_version: str
    nvcc_version: str
    nvcc_path: Path
    cxx_path: Path
    ninja_path: Path
    cuda_home: Path
    cuda_arch_list: str
    extensions_dir: Path


_CONFIG_LOCK = threading.RLock()
_CONFIG = JITConfig()
_CONFIG_FROZEN = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_or_none(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _config_from_environment() -> JITConfig:
    return JITConfig(
        disabled=_env_flag("TORCH_BLEND_DISABLE_JIT"),
        force_jit=_env_flag("TORCH_BLEND_FORCE_JIT"),
        verbose=_env_flag("TORCH_BLEND_JIT_VERBOSE"),
        extensions_dir=_path_or_none(os.getenv("TORCH_EXTENSIONS_DIR")),
        cuda_arch_list=os.getenv("TORCH_CUDA_ARCH_LIST"),
        cuda_home=_path_or_none(os.getenv("CUDA_HOME")),
    )


def get_jit_config() -> JITConfig:
    """Return the effective JIT configuration."""
    with _CONFIG_LOCK:
        environment_config = _config_from_environment()
        return JITConfig(
            disabled=_CONFIG.disabled or environment_config.disabled,
            force_jit=_CONFIG.force_jit or environment_config.force_jit,
            verbose=_CONFIG.verbose or environment_config.verbose,
            lock_timeout_seconds=_CONFIG.lock_timeout_seconds,
            extensions_dir=_CONFIG.extensions_dir or environment_config.extensions_dir,
            cuda_arch_list=_CONFIG.cuda_arch_list or environment_config.cuda_arch_list,
            cuda_home=_CONFIG.cuda_home or environment_config.cuda_home,
        )


def configure_jit(
    *,
    disabled: Optional[bool] = None,
    force_jit: Optional[bool] = None,
    verbose: Optional[bool] = None,
    lock_timeout_seconds: Optional[float] = None,
    extensions_dir: Optional[str | os.PathLike[str]] = None,
    cuda_arch_list: Optional[str] = None,
    cuda_home: Optional[str | os.PathLike[str]] = None,
) -> JITConfig:
    """Configure JIT compilation before the native extension is first loaded."""
    global _CONFIG

    with _CONFIG_LOCK:
        if _CONFIG_FROZEN:
            raise RuntimeError(
                "Torch Blend JIT configuration is frozen because native extension "
                "loading has already started."
            )

        updates = {}
        if disabled is not None:
            updates["disabled"] = bool(disabled)
            os.environ["TORCH_BLEND_DISABLE_JIT"] = "1" if disabled else "0"
        if force_jit is not None:
            updates["force_jit"] = bool(force_jit)
            os.environ["TORCH_BLEND_FORCE_JIT"] = "1" if force_jit else "0"
        if verbose is not None:
            updates["verbose"] = bool(verbose)
            os.environ["TORCH_BLEND_JIT_VERBOSE"] = "1" if verbose else "0"
        if lock_timeout_seconds is not None:
            if lock_timeout_seconds < 0:
                raise ValueError("JIT lock timeout must be non-negative.")
            updates["lock_timeout_seconds"] = float(lock_timeout_seconds)
        if extensions_dir is not None:
            updates["extensions_dir"] = Path(extensions_dir).expanduser().resolve()
        if cuda_arch_list is not None:
            updates["cuda_arch_list"] = cuda_arch_list
        if cuda_home is not None:
            updates["cuda_home"] = Path(cuda_home).expanduser().resolve()

        candidate = replace(_CONFIG, **updates)
        environment_config = _config_from_environment()
        config = JITConfig(
            disabled=candidate.disabled or environment_config.disabled,
            force_jit=candidate.force_jit or environment_config.force_jit,
            verbose=candidate.verbose or environment_config.verbose,
            lock_timeout_seconds=candidate.lock_timeout_seconds,
            extensions_dir=candidate.extensions_dir or environment_config.extensions_dir,
            cuda_arch_list=candidate.cuda_arch_list or environment_config.cuda_arch_list,
            cuda_home=candidate.cuda_home or environment_config.cuda_home,
        )
        if config.disabled and config.force_jit:
            raise ValueError("JIT cannot be both disabled and forced.")
        _CONFIG = candidate
        _apply_environment(config)
        return config


def _apply_environment(config: JITConfig) -> None:
    if config.extensions_dir is not None:
        os.environ["TORCH_EXTENSIONS_DIR"] = str(config.extensions_dir)
    if config.cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = config.cuda_arch_list
    if config.cuda_home is not None:
        os.environ["CUDA_HOME"] = str(config.cuda_home)


def _freeze_jit_config() -> None:
    global _CONFIG_FROZEN
    with _CONFIG_LOCK:
        _CONFIG_FROZEN = True


def _unfreeze_jit_config() -> None:
    global _CONFIG_FROZEN
    with _CONFIG_LOCK:
        _CONFIG_FROZEN = False


def _torch_cuda_version() -> Optional[str]:
    return torch.version.cuda


def _find_nvcc(config: JITConfig) -> Optional[Path]:
    cuda_homes = [
        config.cuda_home,
        _path_or_none(os.getenv("CUDA_HOME")),
        _path_or_none(os.getenv("CONDA_PREFIX")),
    ]
    executable = "nvcc.exe" if os.name == "nt" else "nvcc"

    for cuda_home in cuda_homes:
        if cuda_home is None:
            continue
        candidate = cuda_home / "bin" / executable
        if candidate.is_file():
            return candidate

    discovered = shutil.which("nvcc")
    return Path(discovered).resolve() if discovered else None


def _read_nvcc_version(nvcc_path: Path) -> str:
    result = subprocess.run(
        [str(nvcc_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if not match:
        raise JITEnvironmentError(
            f"Could not parse the CUDA version from '{nvcc_path} --version'."
        )
    return match.group(1)


def _major_minor(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def _find_compiler() -> Optional[Path]:
    configured = os.getenv("CXX")
    if configured:
        discovered = shutil.which(configured)
        if discovered:
            return Path(discovered).resolve()
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        return None

    for executable in ("c++", "g++", "clang++"):
        discovered = shutil.which(executable)
        if discovered:
            return Path(discovered).resolve()
    return None


def _find_ninja() -> Optional[Path]:
    discovered = shutil.which("ninja")
    return Path(discovered).resolve() if discovered else None


def _visible_cuda_architectures() -> list[str]:
    if not torch.cuda.is_available():
        return []
    supported = [
        (int(architecture[3:-1]) // 10, int(architecture[3:-1]) % 10)
        if architecture.endswith("a")
        else (int(architecture[3:]) // 10, int(architecture[3:]) % 10)
        for architecture in torch.cuda.get_arch_list()
        if architecture.startswith("sm_")
    ]
    max_supported = max(supported) if supported else None
    return sorted(
        {
            f"{target[0]}.{target[1]}"
            for index in range(torch.cuda.device_count())
            for major, minor in [torch.cuda.get_device_capability(index)]
            for target in [
                min((major, minor), max_supported)
                if max_supported is not None
                else (major, minor)
            ]
        }
    )


def detect_cuda_architectures() -> list[str]:
    """Return build architectures for the currently visible CUDA devices."""
    return _visible_cuda_architectures()


def _default_extensions_dir() -> Path:
    return Path(tempfile.gettempdir()) / "torch_extensions"


def _ensure_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, prefix=".torch_blend_", delete=True):
        pass


def validate_jit_environment() -> JITEnvironment:
    """Validate and configure the local CUDA extension build environment."""
    config = get_jit_config()
    if config.disabled and config.force_jit:
        raise JITEnvironmentError("JIT cannot be both disabled and forced.")
    _apply_environment(config)
    issues: list[str] = []

    torch_cuda_version = _torch_cuda_version()
    if torch_cuda_version is None:
        issues.append(
            "PyTorch was installed without CUDA support. Install a CUDA-enabled "
            "PyTorch build before using Torch Blend JIT compilation."
        )

    nvcc_path = _find_nvcc(config)
    nvcc_version = None
    if nvcc_path is None:
        expected = torch_cuda_version or "<matching PyTorch CUDA version>"
        issues.append(
            "CUDA compiler 'nvcc' was not found. PyTorch wheels include the CUDA "
            "runtime but not necessarily the compiler.\n"
            f"Install a matching toolkit, for example:\n"
            f"  conda install nvidia::cuda-toolkit=={expected} -y\n"
            "Then set CUDA_HOME to the toolkit root and add CUDA_HOME/bin to PATH."
        )
    else:
        try:
            nvcc_version = _read_nvcc_version(nvcc_path)
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            JITEnvironmentError,
        ) as error:
            issues.append(str(error))

    if torch_cuda_version and nvcc_version:
        torch_cuda = _major_minor(torch_cuda_version)
        nvcc_cuda = _major_minor(nvcc_version)
        if torch_cuda[0] != nvcc_cuda[0]:
            issues.append(
                "CUDA version mismatch:\n"
                f"  PyTorch CUDA: {torch_cuda_version}\n"
                f"  NVCC CUDA:    {nvcc_version}\n"
                f"  NVCC path:    {nvcc_path}\n"
                "Install a CUDA toolkit matching torch.version.cuda or point "
                "CUDA_HOME to the correct toolkit."
            )
        elif torch_cuda != nvcc_cuda:
            warnings.warn(
                "PyTorch and NVCC use different CUDA minor versions "
                f"({torch_cuda_version} and {nvcc_version}). PyTorch permits this "
                "for matching CUDA major versions, but compilation compatibility "
                "is not guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )

    cxx_path = _find_compiler()
    if cxx_path is None:
        issues.append(
            "No C++ compiler was found. Install g++/clang++ or set CXX to a "
            "compatible compiler executable."
        )

    ninja_path = _find_ninja()
    if ninja_path is None:
        issues.append(
            "Ninja was not found. Install it with 'pip install ninja' or "
            "'conda install ninja'."
        )

    cuda_arch_list = config.cuda_arch_list
    if not cuda_arch_list:
        architectures = detect_cuda_architectures()
        if architectures:
            cuda_arch_list = ";".join(architectures)
            os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        else:
            issues.append(
                "No CUDA GPU is visible and TORCH_CUDA_ARCH_LIST is not set. "
                "Expose a GPU during the build or configure the target architecture "
                "manually, for example TORCH_CUDA_ARCH_LIST=8.6."
            )

    extensions_dir = config.extensions_dir or _default_extensions_dir()
    try:
        _ensure_writable_directory(extensions_dir)
    except OSError as error:
        issues.append(
            f"JIT cache directory is not writable: {extensions_dir}\n"
            f"Set TORCH_EXTENSIONS_DIR to a writable path. Original error: {error}"
        )

    if issues:
        raise JITEnvironmentError(
            "Torch Blend JIT environment validation failed:\n\n- "
            + "\n\n- ".join(issues)
        )

    assert torch_cuda_version is not None
    assert nvcc_version is not None
    assert nvcc_path is not None
    assert cxx_path is not None
    assert ninja_path is not None
    assert cuda_arch_list is not None

    cuda_home = config.cuda_home or nvcc_path.parent.parent
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(extensions_dir)

    return JITEnvironment(
        torch_cuda_version=torch_cuda_version,
        nvcc_version=nvcc_version,
        nvcc_path=nvcc_path,
        cxx_path=cxx_path,
        ninja_path=ninja_path,
        cuda_home=cuda_home,
        cuda_arch_list=cuda_arch_list,
        extensions_dir=extensions_dir,
    )


def _reset_jit_configuration_for_tests() -> None:
    global _CONFIG, _CONFIG_FROZEN
    with _CONFIG_LOCK:
        _CONFIG = JITConfig()
        _CONFIG_FROZEN = False
