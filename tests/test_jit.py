import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from torch_blend import environment
from torch_blend import _extension


_JIT_ENV_VARS = [
    "TORCH_BLEND_DISABLE_JIT",
    "TORCH_BLEND_FORCE_JIT",
    "TORCH_BLEND_JIT_VERBOSE",
    "TORCH_EXTENSIONS_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_HOME",
    "CXX",
]


@pytest.fixture(autouse=True)
def reset_jit_state(monkeypatch):
    """Reset loader state and JIT-related environment variables."""
    for name in _JIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    environment._reset_jit_configuration_for_tests()
    _extension._reset_extension_for_tests()
    yield
    environment._reset_jit_configuration_for_tests()
    _extension._reset_extension_for_tests()


def test_configure_jit_applies_environment(tmp_path):
    """Apply Python configuration before cpp_extension is imported."""
    cuda_home = tmp_path / "cuda"
    cache_dir = tmp_path / "extensions"

    config = environment.configure_jit(
        verbose=True,
        lock_timeout_seconds=12.5,
        extensions_dir=cache_dir,
        cuda_arch_list="8.6",
        cuda_home=cuda_home,
    )

    assert config.verbose is True
    assert config.lock_timeout_seconds == 12.5
    assert config.extensions_dir == cache_dir.resolve()
    assert config.cuda_home == cuda_home.resolve()
    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(cache_dir.resolve())
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert os.environ["CUDA_HOME"] == str(cuda_home.resolve())


def test_python_boolean_config_overrides_environment(monkeypatch):
    """Allow explicit Python booleans to override inherited environment flags."""
    monkeypatch.setenv("TORCH_BLEND_JIT_VERBOSE", "1")

    config = environment.configure_jit(verbose=False)

    assert config.verbose is False
    assert os.environ["TORCH_BLEND_JIT_VERBOSE"] == "0"


def test_validate_jit_environment_detects_visible_gpu(tmp_path, monkeypatch):
    """Automatically configure target architectures from visible GPUs."""
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    compiler = tmp_path / "bin" / "c++"
    ninja = tmp_path / "bin" / "ninja"
    cache_dir = tmp_path / "extensions"
    environment.configure_jit(extensions_dir=cache_dir)

    monkeypatch.setattr(environment, "_torch_cuda_version", lambda: "12.8")
    monkeypatch.setattr(environment, "_find_nvcc", lambda config: nvcc)
    monkeypatch.setattr(environment, "_read_nvcc_version", lambda path: "12.8")
    monkeypatch.setattr(environment, "_find_compiler", lambda: compiler)
    monkeypatch.setattr(environment, "_find_ninja", lambda: ninja)
    monkeypatch.setattr(
        environment,
        "detect_cuda_architectures",
        lambda: ["8.6", "8.9"],
    )
    monkeypatch.setattr(environment, "_ensure_writable_directory", lambda path: None)

    result = environment.validate_jit_environment()

    assert result.cuda_arch_list == "8.6;8.9"
    assert result.cuda_home == nvcc.parent.parent
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.6;8.9"


def test_validate_jit_environment_reports_cuda_mismatch(tmp_path, monkeypatch):
    """Report actionable details before attempting an incompatible build."""
    environment.configure_jit(extensions_dir=tmp_path / "extensions")
    monkeypatch.setattr(environment, "_torch_cuda_version", lambda: "12.8")
    monkeypatch.setattr(
        environment,
        "_find_nvcc",
        lambda config: tmp_path / "cuda" / "bin" / "nvcc",
    )
    monkeypatch.setattr(environment, "_read_nvcc_version", lambda path: "11.5")
    monkeypatch.setattr(
        environment,
        "_find_compiler",
        lambda: tmp_path / "bin" / "c++",
    )
    monkeypatch.setattr(
        environment,
        "_find_ninja",
        lambda: tmp_path / "bin" / "ninja",
    )
    monkeypatch.setattr(
        environment,
        "detect_cuda_architectures",
        lambda: ["8.6"],
    )
    monkeypatch.setattr(environment, "_ensure_writable_directory", lambda path: None)

    with pytest.raises(environment.JITEnvironmentError, match="CUDA version mismatch"):
        environment.validate_jit_environment()


def test_validate_jit_environment_warns_for_cuda_minor_mismatch(
    tmp_path,
    monkeypatch,
):
    """Permit matching CUDA majors while exposing minor-version risk."""
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    environment.configure_jit(
        extensions_dir=tmp_path / "extensions",
        cuda_arch_list="8.6",
    )
    monkeypatch.setattr(environment, "_torch_cuda_version", lambda: "12.8")
    monkeypatch.setattr(environment, "_find_nvcc", lambda config: nvcc)
    monkeypatch.setattr(environment, "_read_nvcc_version", lambda path: "12.6")
    monkeypatch.setattr(
        environment,
        "_find_compiler",
        lambda: tmp_path / "bin" / "c++",
    )
    monkeypatch.setattr(
        environment,
        "_find_ninja",
        lambda: tmp_path / "bin" / "ninja",
    )
    monkeypatch.setattr(environment, "_ensure_writable_directory", lambda path: None)

    with pytest.warns(RuntimeWarning, match="minor versions"):
        result = environment.validate_jit_environment()

    assert result.nvcc_version == "12.6"


def test_validate_jit_environment_reports_missing_nvcc(tmp_path, monkeypatch):
    """Explain that CUDA-enabled PyTorch does not necessarily include nvcc."""
    environment.configure_jit(
        extensions_dir=tmp_path / "extensions",
        cuda_arch_list="8.6",
    )
    monkeypatch.setattr(environment, "_torch_cuda_version", lambda: "12.8")
    monkeypatch.setattr(environment, "_find_nvcc", lambda config: None)
    monkeypatch.setattr(
        environment,
        "_find_compiler",
        lambda: tmp_path / "bin" / "c++",
    )
    monkeypatch.setattr(
        environment,
        "_find_ninja",
        lambda: tmp_path / "bin" / "ninja",
    )
    monkeypatch.setattr(environment, "_ensure_writable_directory", lambda path: None)

    with pytest.raises(environment.JITEnvironmentError, match="nvcc.*not found"):
        environment.validate_jit_environment()


def test_binary_extension_is_preferred(monkeypatch):
    """Use an installed binary without invoking JIT compilation."""
    binary = SimpleNamespace(blend=object())
    monkeypatch.setattr(_extension, "_import_binary_extension", lambda: binary)
    monkeypatch.setattr(
        _extension,
        "_build_jit_extension",
        lambda: pytest.fail("JIT build should not run"),
    )

    assert _extension.get_extension() is binary


def test_force_jit_skips_prebuilt_binary(monkeypatch):
    """Build through JIT even when a prebuilt extension is importable."""
    jit_module = SimpleNamespace(blend=object())
    environment.configure_jit(force_jit=True)
    monkeypatch.setattr(
        _extension,
        "_import_binary_extension",
        lambda: pytest.fail("Prebuilt binary should be skipped"),
    )
    monkeypatch.setattr(_extension, "_build_jit_extension", lambda: jit_module)

    assert _extension.get_extension() is jit_module


def test_force_and_disable_jit_are_mutually_exclusive():
    """Reject contradictory JIT configuration."""
    with pytest.raises(ValueError, match="both disabled and forced"):
        environment.configure_jit(disabled=True, force_jit=True)


def test_missing_binary_uses_jit_fallback(monkeypatch):
    """Compile lazily only when the native binary is unavailable."""
    jit_module = SimpleNamespace(blend=object())

    def missing_binary():
        raise ModuleNotFoundError(
            "missing native extension",
            name="torch_blend._torch_blend_cuda",
        )

    monkeypatch.setattr(_extension, "_import_binary_extension", missing_binary)
    monkeypatch.setattr(_extension, "_build_jit_extension", lambda: jit_module)

    assert _extension.get_extension() is jit_module
    assert _extension.get_extension() is jit_module


def test_disabled_jit_requires_manual_build(monkeypatch):
    """Raise a clear manual-build instruction when JIT is disabled."""
    environment.configure_jit(disabled=True)

    def missing_binary():
        raise ModuleNotFoundError(
            "missing native extension",
            name="torch_blend._torch_blend_cuda",
        )

    monkeypatch.setattr(_extension, "_import_binary_extension", missing_binary)

    with pytest.raises(RuntimeError, match="JIT compilation is disabled"):
        _extension.get_extension()


def test_incompatible_binary_does_not_silently_rebuild(monkeypatch):
    """Expose ABI errors instead of hiding them behind a JIT fallback."""
    monkeypatch.setattr(
        _extension,
        "_import_binary_extension",
        lambda: (_ for _ in ()).throw(ImportError("undefined symbol")),
    )
    monkeypatch.setattr(
        _extension,
        "_build_jit_extension",
        lambda: pytest.fail("JIT build should not run"),
    )

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _extension.get_extension()


def test_nested_missing_module_does_not_freeze_configuration(monkeypatch):
    """Unfreeze configuration when binary import fails for another dependency."""
    monkeypatch.setattr(
        _extension,
        "_import_binary_extension",
        lambda: (_ for _ in ()).throw(
            ModuleNotFoundError("missing dependency", name="missing_dependency")
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        _extension.get_extension()

    assert environment.configure_jit(verbose=True).verbose is True


def test_configuration_freezes_after_loading_starts(monkeypatch):
    """Prevent configuration changes after extension loading begins."""
    binary = SimpleNamespace(blend=object())
    monkeypatch.setattr(_extension, "_import_binary_extension", lambda: binary)

    _extension.get_extension()

    with pytest.raises(RuntimeError, match="configuration is frozen"):
        environment.configure_jit(verbose=True)


def test_failed_jit_load_allows_reconfiguration(monkeypatch):
    """Allow users to fix CUDA paths and retry after a failed JIT attempt."""
    def missing_binary():
        raise ModuleNotFoundError(
            "missing native extension",
            name="torch_blend._torch_blend_cuda",
        )

    monkeypatch.setattr(_extension, "_import_binary_extension", missing_binary)
    monkeypatch.setattr(
        _extension,
        "_build_jit_extension",
        lambda: (_ for _ in ()).throw(RuntimeError("build failed")),
    )

    with pytest.raises(RuntimeError, match="build failed"):
        _extension.get_extension()

    config = environment.configure_jit(cuda_arch_list="8.6")
    assert config.cuda_arch_list == "8.6"


def test_concurrent_first_load_builds_once(monkeypatch):
    """Serialize concurrent first use and cache the resulting extension."""
    jit_module = SimpleNamespace(blend=object())
    build_count = 0
    build_lock = threading.Lock()

    def missing_binary():
        raise ModuleNotFoundError(
            "missing native extension",
            name="torch_blend._torch_blend_cuda",
        )

    def build_once():
        nonlocal build_count
        with build_lock:
            build_count += 1
        return jit_module

    monkeypatch.setattr(_extension, "_import_binary_extension", missing_binary)
    monkeypatch.setattr(_extension, "_build_jit_extension", build_once)

    with ThreadPoolExecutor(max_workers=4) as executor:
        modules = list(executor.map(lambda _: _extension.get_extension(), range(8)))

    assert all(module is jit_module for module in modules)
    assert build_count == 1


def test_existing_build_lock_times_out_with_removal_instruction(tmp_path):
    """Stop waiting on a stale PyTorch lock and report its exact path."""
    build_directory = tmp_path / "extension"
    build_directory.mkdir()
    lock_path = build_directory / "lock"
    lock_path.touch()

    with pytest.raises(RuntimeError, match="remove this stale lock") as error:
        _extension._wait_for_build_lock(build_directory, 0, verbose=False)

    assert str(lock_path) in str(error.value)


def test_negative_build_lock_timeout_is_rejected():
    """Reject invalid lock timeout configuration before loading starts."""
    with pytest.raises(ValueError, match="non-negative"):
        environment.configure_jit(lock_timeout_seconds=-1)


def test_clear_jit_cache_removes_current_build_directory(tmp_path, monkeypatch):
    """Delete only the cache directory for the current source version."""
    build_directory = tmp_path / "torch_blend_cuda_jit_test"
    build_directory.mkdir()
    (build_directory / "artifact.o").touch()
    monkeypatch.setattr(
        _extension,
        "_jit_build_directory",
        lambda: build_directory,
    )

    removed_path = _extension.clear_jit_cache()

    assert removed_path == build_directory
    assert not build_directory.exists()


def test_clear_jit_cache_requires_force_for_existing_lock(tmp_path, monkeypatch):
    """Avoid deleting a cache that may belong to an active build."""
    build_directory = tmp_path / "torch_blend_cuda_jit_test"
    build_directory.mkdir()
    (build_directory / "lock").touch()
    monkeypatch.setattr(
        _extension,
        "_jit_build_directory",
        lambda: build_directory,
    )

    with pytest.raises(RuntimeError, match="Pass force=True"):
        _extension.clear_jit_cache()

    _extension.clear_jit_cache(force=True)
    assert not build_directory.exists()


def test_build_jit_extension_forces_eager_jit_loading(monkeypatch):
    """Expose an explicit API for compiling before the first blend call."""
    jit_module = SimpleNamespace(blend=object())
    monkeypatch.setattr(_extension, "get_extension", lambda: jit_module)

    result = _extension.build_jit_extension()

    assert result is jit_module
    assert environment.get_jit_config().force_jit is True
