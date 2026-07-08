#!/usr/bin/env python3
"""Grid-search hyperparameters for the CUTLASS GEMM kernel in kernel.cu.

For each valid combination of BLOCK_M, BLOCK_N, SMEM_K, WARP_M, WARP_N, and
STAGES_K, patches kernel.cu, runs `make`, parses throughput/latency from the
build output, and records success or failure.  Dumps all results and reports
the best configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

# Fixed problem size (must match main.cu).
M = 8192
N = 8192
K = 8192

# Search space (SMEM_K is fixed).
BLOCK_M_VALUES = [32, 64, 128, 256, 512]
BLOCK_N_VALUES = [32, 64, 128, 256, 512]
SMEM_K_VALUE = 64
WARP_M_VALUES = [1, 2, 4, 8, 16, 32]
WARP_N_VALUES = [1, 2, 4, 8, 16, 32]
STAGES_K_VALUES = [1, 2, 3]

PARAM_NAMES = ("BLOCK_M", "BLOCK_N", "SMEM_K", "WARP_M", "WARP_N", "STAGES_K")

THROUGHPUT_RE = re.compile(r"GPU Throughput:\s*([\d.]+)\s*TFLOPS")
LATENCY_RE = re.compile(r"GPU Kernel Time:\s*([\d.]+)\s*ms")
CORRECTNESS_RE = re.compile(r"Correctness Status:\s*(PASSED|FAILED)")

PARAM_LINE_RES = {
    name: re.compile(rf"^constexpr int {name}\s*=\s*\d+;", re.MULTILINE)
    for name in PARAM_NAMES
}


@dataclass(frozen=True)
class Config:
    BLOCK_M: int
    BLOCK_N: int
    SMEM_K: int
    WARP_M: int
    WARP_N: int
    STAGES_K: int


@dataclass
class SweepResult:
    config: Config
    status: str  # "ok", "compile_error", "runtime_error", "correctness_failed", "parse_error", "timeout", "skipped"
    throughput_tflops: float | None = None
    latency_ms: float | None = None
    correctness: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    skip_reason: str | None = None


@dataclass
class SweepState:
    started_at: str
    project_dir: str
    total_configs: int
    completed: int = 0
    results: list[SweepResult] = field(default_factory=list)


def is_valid_config(cfg: Config) -> str | None:
    """Return a reason string if cfg should be skipped, else None."""
    if cfg.WARP_M * cfg.WARP_N > 32:
        return "WARP_M * WARP_N > 32"
    if M % cfg.BLOCK_M != 0 or N % cfg.BLOCK_N != 0 or K % cfg.SMEM_K != 0:
        return "problem size not divisible by tile size"
    if cfg.BLOCK_M % (cfg.WARP_M * 16) != 0:
        return f"BLOCK_M % (WARP_M * 16) != 0"
    if cfg.BLOCK_N % (cfg.WARP_N * 8) != 0:
        return f"BLOCK_N % (WARP_N * 8) != 0"
    threads = cfg.WARP_M * cfg.WARP_N * 32
    if threads > 1024:
        return f"threads per block ({threads}) > 1024"
    return None


def iter_configs() -> list[tuple[Config, str | None]]:
    configs: list[tuple[Config, str | None]] = []
    for block_m, block_n, warp_m, warp_n, stages_k in product(
        BLOCK_M_VALUES,
        BLOCK_N_VALUES,
        WARP_M_VALUES,
        WARP_N_VALUES,
        STAGES_K_VALUES,
    ):
        cfg = Config(
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            SMEM_K=SMEM_K_VALUE,
            WARP_M=warp_m,
            WARP_N=warp_n,
            STAGES_K=stages_k,
        )
        configs.append((cfg, is_valid_config(cfg)))
    return configs


def patch_kernel_cu(path: Path, cfg: Config) -> None:
    content = path.read_text()
    values = {
        "BLOCK_M": cfg.BLOCK_M,
        "BLOCK_N": cfg.BLOCK_N,
        "SMEM_K": cfg.SMEM_K,
        "WARP_M": cfg.WARP_M,
        "WARP_N": cfg.WARP_N,
        "STAGES_K": cfg.STAGES_K,
    }
    for name, value in values.items():
        pattern = PARAM_LINE_RES[name]
        replacement = f"constexpr int {name:<7} = {value};"
        new_content, count = pattern.subn(replacement, content, count=1)
        if count != 1:
            raise RuntimeError(f"Failed to patch {name} in {path}")
        content = new_content
    path.write_text(content)


def parse_make_output(output: str) -> tuple[float | None, float | None, str | None]:
    throughput_match = THROUGHPUT_RE.search(output)
    latency_match = LATENCY_RE.search(output)
    correctness_match = CORRECTNESS_RE.search(output)

    throughput = float(throughput_match.group(1)) if throughput_match else None
    latency = float(latency_match.group(1)) if latency_match else None
    correctness = correctness_match.group(1) if correctness_match else None
    return throughput, latency, correctness


def run_make(project_dir: Path, timeout_s: int) -> tuple[int, str, str, float]:
    start = time.monotonic()
    proc = subprocess.run(
        ["make"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    duration = time.monotonic() - start
    combined = proc.stdout + proc.stderr
    return proc.returncode, combined, proc.stderr, duration


def evaluate_config(
    project_dir: Path,
    kernel_path: Path,
    cfg: Config,
    timeout_s: int,
) -> SweepResult:
    start = time.monotonic()
    try:
        patch_kernel_cu(kernel_path, cfg)
    except Exception as exc:
        return SweepResult(
            config=cfg,
            status="patch_error",
            error=str(exc),
            duration_s=time.monotonic() - start,
        )

    try:
        returncode, output, _, make_duration = run_make(project_dir, timeout_s)
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") + (exc.stderr or "")
        return SweepResult(
            config=cfg,
            status="timeout",
            error=partial[-4000:] if partial else f"make exceeded {timeout_s}s",
            duration_s=time.monotonic() - start,
        )

    throughput, latency, correctness = parse_make_output(output)

    if returncode != 0:
        return SweepResult(
            config=cfg,
            status="compile_error" if "error:" in output.lower() else "runtime_error",
            throughput_tflops=throughput,
            latency_ms=latency,
            correctness=correctness,
            error=output[-4000:],
            duration_s=time.monotonic() - start,
        )

    if throughput is None or latency is None:
        return SweepResult(
            config=cfg,
            status="parse_error",
            throughput_tflops=throughput,
            latency_ms=latency,
            correctness=correctness,
            error=output[-4000:],
            duration_s=time.monotonic() - start,
        )

    if correctness != "PASSED":
        return SweepResult(
            config=cfg,
            status="correctness_failed",
            throughput_tflops=throughput,
            latency_ms=latency,
            correctness=correctness,
            error=output[-4000:],
            duration_s=time.monotonic() - start,
        )

    return SweepResult(
        config=cfg,
        status="ok",
        throughput_tflops=throughput,
        latency_ms=latency,
        correctness=correctness,
        duration_s=make_duration,
    )


def result_to_row(result: SweepResult) -> dict[str, Any]:
    row = asdict(result.config)
    row.update(
        {
            "status": result.status,
            "throughput_tflops": result.throughput_tflops,
            "latency_ms": result.latency_ms,
            "correctness": result.correctness,
            "error": result.error,
            "skip_reason": result.skip_reason,
            "duration_s": round(result.duration_s, 3),
        }
    )
    return row


def save_results(
    results: list[SweepResult],
    json_path: Path,
    csv_path: Path,
    meta: dict[str, Any],
) -> None:
    payload = {
        "meta": meta,
        "results": [result_to_row(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    fieldnames = [
        "BLOCK_M",
        "BLOCK_N",
        "SMEM_K",
        "WARP_M",
        "WARP_N",
        "STAGES_K",
        "status",
        "throughput_tflops",
        "latency_ms",
        "correctness",
        "duration_s",
        "skip_reason",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))


def pick_best(results: list[SweepResult]) -> SweepResult | None:
    ok = [r for r in results if r.status == "ok" and r.throughput_tflops is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: r.throughput_tflops)  # type: ignore[arg-type,type-var]


def print_summary(results: list[SweepResult], best: SweepResult | None) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("\n=== Sweep Summary ===")
    print(f"Total entries: {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    if best is None:
        print("\nNo successful configuration found.")
        return

    c = best.config
    print("\n=== Best Configuration ===")
    print(
        f"BLOCK_M={c.BLOCK_M}, BLOCK_N={c.BLOCK_N}, SMEM_K={c.SMEM_K}, "
        f"WARP_M={c.WARP_M}, WARP_N={c.WARP_N}, STAGES_K={c.STAGES_K}"
    )
    print(f"Throughput: {best.throughput_tflops:.4f} TFLOPS")
    print(f"Latency:    {best.latency_ms:.6f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search GEMM kernel hyperparameters.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing kernel.cu and Makefile",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each `make` invocation (default: 300)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most N runnable configs (0 = all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs already present in sweep_results.json",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Leave kernel.cu patched with the best config instead of restoring the backup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned configs without building",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="sweep_results",
        help="Prefix for sweep_results.json / sweep_results.csv",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    kernel_path = project_dir / "kernel.cu"
    backup_path = project_dir / "kernel.cu.sweep_backup"
    json_path = project_dir / f"{args.output_prefix}.json"
    csv_path = project_dir / f"{args.output_prefix}.csv"

    if not kernel_path.exists():
        print(f"kernel.cu not found at {kernel_path}", file=sys.stderr)
        return 1

    all_entries = iter_configs()
    runnable = [(cfg, reason) for cfg, reason in all_entries if reason is None]
    skipped = [
        SweepResult(config=cfg, status="skipped", skip_reason=reason)
        for cfg, reason in all_entries
        if reason is not None
    ]

    if args.dry_run:
        print(f"Runnable configs: {len(runnable)}")
        print(f"Skipped configs:  {len(skipped)}")
        for i, (cfg, _) in enumerate(runnable[:20], 1):
            print(
                f"{i:4d}. BLOCK_M={cfg.BLOCK_M}, BLOCK_N={cfg.BLOCK_N}, "
                f"WARP_M={cfg.WARP_M}, WARP_N={cfg.WARP_N}, STAGES_K={cfg.STAGES_K}"
            )
        if len(runnable) > 20:
            print(f"... and {len(runnable) - 20} more")
        return 0

    done_keys: set[tuple[int, ...]] = set()
    results: list[SweepResult] = []

    if args.resume and json_path.exists():
        data = json.loads(json_path.read_text())
        for row in data.get("results", []):
            cfg = Config(
                BLOCK_M=row["BLOCK_M"],
                BLOCK_N=row["BLOCK_N"],
                SMEM_K=row["SMEM_K"],
                WARP_M=row["WARP_M"],
                WARP_N=row["WARP_N"],
                STAGES_K=row["STAGES_K"],
            )
            results.append(
                SweepResult(
                    config=cfg,
                    status=row["status"],
                    throughput_tflops=row.get("throughput_tflops"),
                    latency_ms=row.get("latency_ms"),
                    correctness=row.get("correctness"),
                    error=row.get("error"),
                    skip_reason=row.get("skip_reason"),
                    duration_s=row.get("duration_s", 0.0),
                )
            )
            if row.get("status") != "skipped":
                done_keys.add(
                    (
                        cfg.BLOCK_M,
                        cfg.BLOCK_N,
                        cfg.SMEM_K,
                        cfg.WARP_M,
                        cfg.WARP_N,
                        cfg.STAGES_K,
                    )
                )
    else:
        results = list(skipped)

    if not backup_path.exists():
        shutil.copy2(kernel_path, backup_path)
        print(f"Backed up kernel.cu -> {backup_path}")

    evaluated = 0
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Sweep started at {started_at}")
    print(f"Runnable configs: {len(runnable)} (skipped upfront: {len(skipped)})")

    try:
        for idx, (cfg, _) in enumerate(runnable, 1):
            key = (
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.SMEM_K,
                cfg.WARP_M,
                cfg.WARP_N,
                cfg.STAGES_K,
            )
            if key in done_keys:
                continue
            if args.limit and evaluated >= args.limit:
                break

            print(
                f"[{idx}/{len(runnable)}] "
                f"BLOCK_M={cfg.BLOCK_M}, BLOCK_N={cfg.BLOCK_N}, "
                f"WARP_M={cfg.WARP_M}, WARP_N={cfg.WARP_N}, STAGES_K={cfg.STAGES_K} ... ",
                end="",
                flush=True,
            )

            result = evaluate_config(project_dir, kernel_path, cfg, args.timeout)
            results.append(result)

            if result.status == "ok":
                print(
                    f"{result.throughput_tflops:.4f} TFLOPS, "
                    f"{result.latency_ms:.4f} ms ({result.duration_s:.1f}s)"
                )
            else:
                print(f"{result.status} ({result.duration_s:.1f}s)")

            evaluated += 1

            meta = {
                "started_at": started_at,
                "project_dir": str(project_dir),
                "problem_shape": [M, N, K],
                "runnable_configs": len(runnable),
                "evaluated_this_run": evaluated,
            }
            save_results(results, json_path, csv_path, meta)
    finally:
        best = pick_best(results)
        if args.apply_best and best is not None:
            patch_kernel_cu(kernel_path, best.config)
            print(f"\nApplied best configuration to {kernel_path}")
        elif backup_path.exists():
            shutil.copy2(backup_path, kernel_path)
            print(f"\nRestored original kernel.cu from {backup_path}")

    best = pick_best(results)
    print_summary(results, best)
    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
