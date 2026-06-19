from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_blend import ImageBlender


SIZE_PRESETS = {
    "256p": (256, 256),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "4k": (2160, 3840),
}
DTYPES = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class BenchmarkCase:
    device: str
    height: int
    width: int
    layout: str
    dtype: str
    channels: int
    batch: int
    alpha: float

    @property
    def shape(self) -> tuple[int, ...]:
        if self.layout == "HWC":
            return (self.height, self.width, self.channels)
        if self.layout == "CHW":
            return (self.channels, self.height, self.width)
        return (self.batch, self.channels, self.height, self.width)

    @property
    def pixels(self) -> int:
        return self.height * self.width * self.batch


@dataclass
class BenchmarkResult:
    backend: str
    status: str
    reason: str
    device: str
    height: int
    width: int
    layout: str
    dtype: str
    channels: int
    batch: int
    alpha: float
    warmup: int
    runs: int
    min_ms: float | None = None
    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    std_ms: float | None = None
    fps: float | None = None
    megapixels_per_second: float | None = None
    estimated_gbps: float | None = None
    speedup_vs_pytorch: float | None = None
    max_abs_error: float | None = None


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_system_metadata() -> dict[str, Any]:
    cpu_info = command_output(["lscpu"])
    memory_total_bytes = None
    try:
        memory_total_bytes = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, ValueError, OSError):
        pass

    cpu_model = None
    for line in (cpu_info or "").splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
            break

    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version.replace("\n", " "),
        "cpu_count_logical": os.cpu_count(),
        "cpu_model": cpu_model,
        "cpu_info": cpu_info,
        "memory_total_bytes": memory_total_bytes,
        "memory_info": command_output(["free", "-b"]),
        "gpu_devices": gpu_devices,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": command_output(["nvcc", "--version"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_cuda_devices": (
            cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else None
        ),
        "numpy": np.__version__,
        "dependencies": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "opencv-python-headless",
                "pytest",
                "ninja",
                "setuptools",
                "wheel",
            )
        },
        "git_commit": command_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ),
        "git_dirty": bool(
            command_output(["git", "status", "--short"], cwd=PROJECT_ROOT)
        ),
        "git_status": command_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT
        ),
    }


def environment_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings = []
    if not metadata["cuda_available"]:
        warnings.append("CUDA benchmark skipped: torch.cuda.is_available() is False.")
    if metadata["nvidia_smi"] and "failed" in metadata["nvidia_smi"].lower():
        warnings.append("nvidia-smi cannot communicate with the NVIDIA driver.")

    torch_cuda = metadata["torch_cuda"]
    nvcc = metadata["nvcc"] or ""
    if torch_cuda and "release " in nvcc:
        nvcc_version = nvcc.split("release ", 1)[1].split(",", 1)[0].strip()
        if nvcc_version.split(".", 1)[0] != torch_cuda.split(".", 1)[0]:
            warnings.append(
                f"CUDA major mismatch: PyTorch uses {torch_cuda}, "
                f"but NVCC reports {nvcc_version}."
            )
    return warnings


def parse_sizes(values: list[str]) -> list[tuple[int, int]]:
    sizes = []
    for value in values:
        normalized = value.lower()
        if normalized in SIZE_PRESETS:
            sizes.append(SIZE_PRESETS[normalized])
            continue
        try:
            width, height = normalized.split("x", 1)
            sizes.append((int(height), int(width)))
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid size '{value}'. Use a preset or WIDTHxHEIGHT."
            ) from error
    return sizes


def make_inputs(case: BenchmarkCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(case.device)
    dtype = DTYPES[case.dtype]
    if dtype == torch.uint8:
        img1 = torch.randint(0, 256, case.shape, dtype=dtype, device=device)
        img2 = torch.randint(0, 256, case.shape, dtype=dtype, device=device)
        mask_value = round(case.alpha * 255)
    else:
        img1 = torch.rand(case.shape, dtype=dtype, device=device)
        img2 = torch.rand(case.shape, dtype=dtype, device=device)
        mask_value = case.alpha

    mask_shape = (
        (case.batch, case.height, case.width)
        if case.layout == "BCHW"
        else (case.height, case.width)
    )
    mask = torch.full(mask_shape, mask_value, dtype=dtype, device=device)
    return img1, img2, mask


def effective_alpha(case: BenchmarkCase) -> float:
    if case.dtype == "uint8":
        return round(case.alpha * 255) / 255.0
    return case.alpha


def pytorch_blend(
    img1: torch.Tensor,
    img2: torch.Tensor,
    case: BenchmarkCase,
) -> torch.Tensor:
    alpha = effective_alpha(case)
    if img1.dtype == torch.uint8:
        return (
            img1.float() * alpha + img2.float() * (1.0 - alpha)
        ).to(img1.dtype)
    return img1 * alpha + img2 * (1.0 - alpha)


def opencv_blend(
    img1: torch.Tensor,
    img2: torch.Tensor,
    case: BenchmarkCase,
) -> np.ndarray:
    return cv2.addWeighted(
        img1.numpy(),
        effective_alpha(case),
        img2.numpy(),
        1.0 - effective_alpha(case),
        0.0,
    )


def synchronize(case: BenchmarkCase) -> None:
    if case.device == "cuda":
        torch.cuda.synchronize()


def measure(
    function: Callable[[], Any],
    case: BenchmarkCase,
    warmup: int,
    runs: int,
) -> tuple[list[float], Any]:
    output = None
    for _ in range(warmup):
        output = function()
    synchronize(case)

    samples = []
    for _ in range(runs):
        synchronize(case)
        start = time.perf_counter_ns()
        output = function()
        synchronize(case)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples, output


def to_float_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, np.ndarray):
        return torch.from_numpy(output).float()
    return output.detach().cpu().float()


def estimate_bytes(case: BenchmarkCase, backend: str) -> int:
    element_size = torch.empty((), dtype=DTYPES[case.dtype]).element_size()
    image_bytes = case.pixels * case.channels * element_size
    mask_bytes = case.pixels * element_size
    return image_bytes * 3 + (mask_bytes if backend == "torch_blend" else 0)


def completed_result(
    backend: str,
    case: BenchmarkCase,
    warmup: int,
    samples: list[float],
    output: Any,
    reference: torch.Tensor,
) -> BenchmarkResult:
    median_ms = statistics.median(samples)
    seconds = median_ms / 1000.0
    error = torch.max(torch.abs(to_float_tensor(output) - reference)).item()
    tolerance = {
        "uint8": 1.0,
        "float16": 1e-3,
        "float32": 1e-5,
    }[case.dtype]
    status = "passed" if error <= tolerance else "failed"
    reason = (
        ""
        if status == "passed"
        else f"Maximum absolute error {error} exceeds tolerance {tolerance}."
    )
    return BenchmarkResult(
        backend=backend,
        status=status,
        reason=reason,
        device=case.device,
        height=case.height,
        width=case.width,
        layout=case.layout,
        dtype=case.dtype,
        channels=case.channels,
        batch=case.batch,
        alpha=case.alpha,
        warmup=warmup,
        runs=len(samples),
        min_ms=min(samples),
        mean_ms=statistics.mean(samples),
        median_ms=median_ms,
        p95_ms=float(np.percentile(np.asarray(samples), 95)),
        std_ms=statistics.pstdev(samples),
        fps=1.0 / seconds,
        megapixels_per_second=case.pixels / seconds / 1_000_000.0,
        estimated_gbps=(
            estimate_bytes(case, backend) / seconds / 1_000_000_000.0
        ),
        max_abs_error=error,
    )


def unavailable_result(
    backend: str,
    case: BenchmarkCase,
    warmup: int,
    runs: int,
    status: str,
    reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        status=status,
        reason=reason,
        device=case.device,
        height=case.height,
        width=case.width,
        layout=case.layout,
        dtype=case.dtype,
        channels=case.channels,
        batch=case.batch,
        alpha=case.alpha,
        warmup=warmup,
        runs=runs,
    )


def run_case(case: BenchmarkCase, warmup: int, runs: int) -> list[BenchmarkResult]:
    backend_names = ["torch_blend", "pytorch", "opencv"]
    if case.device == "cuda" and not torch.cuda.is_available():
        return [
            unavailable_result(
                backend, case, warmup, runs, "skipped", "CUDA is not available."
            )
            for backend in backend_names
        ]
    if case.device == "cpu" and case.dtype == "float16":
        return [
            unavailable_result(
                backend,
                case,
                warmup,
                runs,
                "skipped",
                "CPU float16 is excluded because it is not representative.",
            )
            for backend in backend_names
        ]

    img1, img2, mask = make_inputs(case)
    reference = pytorch_blend(img1, img2, case).detach().cpu().float()
    results = []
    backends: list[tuple[str, Callable[[], Any]]] = [
        (
            "torch_blend",
            lambda: ImageBlender.blend(img1, img2, mask, layout=case.layout),
        ),
        ("pytorch", lambda: pytorch_blend(img1, img2, case)),
    ]
    if case.device == "cpu" and case.layout == "HWC" and case.batch == 1:
        backends.append(("opencv", lambda: opencv_blend(img1, img2, case)))
    else:
        results.append(
            unavailable_result(
                "opencv",
                case,
                warmup,
                runs,
                "skipped",
                "cv2.addWeighted supports only CPU HWC single-image cases.",
            )
        )

    for backend, function in backends:
        try:
            samples, output = measure(function, case, warmup, runs)
            results.append(
                completed_result(
                    backend, case, warmup, samples, output, reference
                )
            )
        except Exception as error:
            results.append(
                unavailable_result(
                    backend,
                    case,
                    warmup,
                    runs,
                    "failed",
                    f"{type(error).__name__}: {error}",
                )
            )

    pytorch_result = next(
        (
            result
            for result in results
            if result.backend == "pytorch" and result.status == "passed"
        ),
        None,
    )
    if pytorch_result and pytorch_result.median_ms:
        for result in results:
            if result.status == "passed" and result.median_ms:
                result.speedup_vs_pytorch = (
                    pytorch_result.median_ms / result.median_ms
                )
    return results


def format_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(results: list[BenchmarkResult]) -> str:
    headers = [
        "Backend",
        "Status",
        "Device",
        "Input",
        "Layout",
        "Dtype",
        "Median ms",
        "P95 ms",
        "FPS",
        "MP/s",
        "GB/s",
        "Speedup",
        "Max error",
    ]
    rows = []
    for result in results:
        input_name = (
            f"{result.batch}x{result.height}x{result.width}x{result.channels}"
            if result.layout == "BCHW"
            else f"{result.height}x{result.width}x{result.channels}"
        )
        rows.append(
            [
                result.backend,
                result.status,
                result.device,
                input_name,
                result.layout,
                result.dtype,
                format_number(result.median_ms),
                format_number(result.p95_ms),
                format_number(result.fps, 2),
                format_number(result.megapixels_per_second, 2),
                format_number(result.estimated_gbps, 2),
                (
                    "-"
                    if result.speedup_vs_pytorch is None
                    else f"{result.speedup_vs_pytorch:.2f}x"
                ),
                format_number(result.max_abs_error),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            headers[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    warnings: list[str],
    results: list[BenchmarkResult],
    command: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (output_dir / "system.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dependency_lines = [
        f"{name}=={version}"
        for name, version in metadata["dependencies"].items()
        if version
    ]
    (output_dir / "requirements.txt").write_text(
        "\n".join(dependency_lines) + "\n", encoding="utf-8"
    )

    warning_section = "\n".join(f"- {warning}" for warning in warnings) or "- None"
    failure_details = "\n".join(
        f"- `{result.backend}` `{result.device}` `{result.layout}` "
        f"`{result.dtype}` `{result.width}x{result.height}`: "
        f"{' '.join(result.reason.split())}"
        for result in results
        if result.status != "passed"
    ) or "- None"
    dependency_details = "\n".join(
        f"- `{name}`: `{version}`"
        for name, version in metadata["dependencies"].items()
        if version
    ) or "- Unavailable"
    gpu_names = ", ".join(
        device["name"] for device in metadata["gpu_devices"]
    ) or "Unavailable"
    report = f"""# Torch Blend Benchmark Report

Generated: `{metadata["timestamp_utc"]}`

## Reproduction

```bash
{command}
```

## System

- Host: `{metadata["hostname"]}`
- OS: `{metadata["os"]}`
- CPU: `{metadata["cpu_model"]}`
- CPU logical processors: `{metadata["cpu_count_logical"]}`
- RAM bytes: `{metadata["memory_total_bytes"]}`
- GPU: `{gpu_names}`
- Python: `{metadata["python"]}`
- PyTorch: `{metadata["torch"]}`
- PyTorch CUDA: `{metadata["torch_cuda"]}`
- CUDA available: `{metadata["cuda_available"]}`
- OpenCV: `{metadata["opencv"]}`
- OpenCV threads: `{metadata["opencv_threads"]}`
- NumPy: `{metadata["numpy"]}`
- Git commit: `{metadata["git_commit"]}`
- Git working tree dirty: `{metadata["git_dirty"]}`

## Dependencies

{dependency_details}

## CUDA Toolchain

```text
PyTorch CUDA: {metadata["torch_cuda"]}

nvidia-smi:
{metadata["nvidia_smi"] or "Unavailable"}

nvcc:
{metadata["nvcc"] or "Unavailable"}
```

## Environment Warnings

{warning_section}

## Results

{markdown_table(results)}

Speedup is calculated against pure PyTorch median latency for the same case.
Estimated GB/s uses minimum algorithm I/O: two image reads and one output write,
plus one mask read for Torch Blend.

## Skipped And Failed Cases

{failure_details}

## Methodology

- Inputs are allocated on the target device before timing.
- Warmup iterations are excluded.
- Every measured invocation creates a new output.
- CUDA measurements synchronize before and after every invocation.
- OpenCV is compared only for CPU HWC single-image inputs.
- Torch Blend receives a constant mask and the baselines use the equivalent scalar alpha.
- Correctness is checked against the pure PyTorch implementation.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = []
    for device in args.devices:
        for height, width in parse_sizes(args.sizes):
            for layout in args.layouts:
                for dtype in args.dtypes:
                    for channels in args.channels:
                        batches = args.batches if layout == "BCHW" else [1]
                        for batch in batches:
                            cases.append(
                                BenchmarkCase(
                                    device=device,
                                    height=height,
                                    width=width,
                                    layout=layout,
                                    dtype=dtype,
                                    channels=channels,
                                    batch=batch,
                                    alpha=args.alpha,
                                )
                            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch_blend against OpenCV and pure PyTorch."
    )
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--sizes", nargs="+", default=["256p", "720p", "1080p"]
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["HWC", "CHW", "BCHW"],
        default=["HWC", "CHW", "BCHW"],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=sorted(DTYPES),
        default=["uint8", "float32"],
    )
    parser.add_argument("--channels", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--output-root", type=Path, default=Path("benchmark_results")
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be non-negative and --runs must be positive.")
    if any(channel < 1 for channel in args.channels):
        parser.error("--channels values must be positive.")
    if any(batch < 1 for batch in args.batches):
        parser.error("--batches values must be positive.")
    return args


def main() -> int:
    args = parse_args()
    metadata = collect_system_metadata()
    warnings = environment_warnings(metadata)
    output_dir = args.output_root / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    cases = build_cases(args)
    results = []

    print(f"Running {len(cases)} benchmark cases...")
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case.device} {case.layout} "
            f"{case.dtype} {case.shape}"
        )
        results.extend(run_case(case, args.warmup, args.runs))

    command = " ".join([sys.executable, *sys.argv])
    write_report(output_dir, metadata, warnings, results, command)
    print()
    print(markdown_table(results))
    print(f"\nReports written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
