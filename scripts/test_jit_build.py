#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from torch_blend import (
    ImageBlender,
    JITEnvironmentError,
    clear_jit_cache,
    configure_jit,
    detect_cuda_architectures,
    validate_jit_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force a Torch Blend JIT build and run CUDA smoke tests."
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        help="CUDA toolkit root containing bin/nvcc.",
    )
    parser.add_argument(
        "--extensions-dir",
        type=Path,
        help="Writable PyTorch JIT extension cache directory.",
    )
    parser.add_argument(
        "--cuda-arch-list",
        help="Optional target architectures, for example '8.6' or '8.6;8.9'.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose compiler output.",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for another JIT build lock before failing.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the current source-version JIT cache before building.",
    )
    parser.add_argument(
        "--force-clear-lock",
        action="store_true",
        help="Allow --rebuild to remove an existing build lock.",
    )
    return parser.parse_args()


def reference_blend(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    alpha = mask.unsqueeze(1)
    return img1 * alpha + img2 * (1.0 - alpha)


def main() -> int:
    args = parse_args()
    if args.force_clear_lock and not args.rebuild:
        print("--force-clear-lock requires --rebuild.", file=sys.stderr)
        return 2

    detected_architectures = detect_cuda_architectures()
    cuda_arch_list = args.cuda_arch_list
    if cuda_arch_list is None and detected_architectures:
        cuda_arch_list = ";".join(detected_architectures)

    configure_jit(
        force_jit=True,
        verbose=not args.quiet,
        lock_timeout_seconds=args.lock_timeout,
        extensions_dir=args.extensions_dir,
        cuda_arch_list=cuda_arch_list,
        cuda_home=args.cuda_home,
    )

    try:
        environment = validate_jit_environment()
    except JITEnvironmentError as error:
        print(error, file=sys.stderr)
        return 2

    print("JIT environment validated:", flush=True)
    print(f"  PyTorch:            {torch.__version__}", flush=True)
    print(f"  PyTorch CUDA:       {environment.torch_cuda_version}", flush=True)
    print(f"  NVCC:               {environment.nvcc_path}", flush=True)
    print(f"  NVCC CUDA:          {environment.nvcc_version}", flush=True)
    print(f"  CUDA architectures: {environment.cuda_arch_list}", flush=True)
    print(f"  Extension cache:    {environment.extensions_dir}", flush=True)
    if args.cuda_arch_list is None and detected_architectures:
        print(
            "  Architecture source: visible CUDA device(s)",
            flush=True,
        )
    if args.rebuild:
        try:
            removed_path = clear_jit_cache(force=args.force_clear_lock)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 3
        print(f"  Rebuild cache cleared: {removed_path}", flush=True)
    print("Starting first blend call; this triggers the JIT build...", flush=True)

    device = torch.device("cuda")
    img1 = torch.rand((2, 3, 64, 64), device=device)
    img2 = torch.rand_like(img1)
    mask = torch.rand((2, 64, 64), device=device)

    started = time.perf_counter()
    result = ImageBlender.blend(img1, img2, mask, layout="BCHW")
    torch.cuda.synchronize()
    first_call_seconds = time.perf_counter() - started

    expected = reference_blend(img1, img2, mask)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)

    started = time.perf_counter()
    cached_result = ImageBlender.blend(
        img1,
        img2,
        mask,
        layout="BCHW",
        mode="screen",
    )
    torch.cuda.synchronize()
    cached_call_seconds = time.perf_counter() - started

    screen = 1.0 - (1.0 - img1) * (1.0 - img2)
    expected_screen = screen * mask.unsqueeze(1) + img2 * (
        1.0 - mask.unsqueeze(1)
    )
    torch.testing.assert_close(cached_result, expected_screen, rtol=0, atol=1e-6)

    print("JIT smoke tests passed:")
    print(f"  First call/build:  {first_call_seconds:.3f}s")
    print(f"  Cached call:       {cached_call_seconds:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
