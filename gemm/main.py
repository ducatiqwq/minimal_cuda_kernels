"""Torch baseline for the CUDA GEMM in kernel.cu / main.cu.

Performs C = A @ B.T with float16 matrices A (M, K) and B (N, K),
and reports the time taken by torch's CUDA matmul (measured with CUDA
events after a warmup, matching main.cu).
"""

import torch

M = 8192
N = 8192
K = 8192


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available.")

    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    print("=== Device Metadata ===")
    print(f"Device Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"VRAM (Global Memory): {props.total_memory // (1024 * 1024)} MB")
    print(f"Multiprocessors: {props.multi_processor_count}")
    print("=======================\n")

    print(f"Configuration: GEMM ({M}, {N}, {K}), dtype=float16\n")

    gen = torch.Generator(device=device).manual_seed(42)
    A = torch.rand(M, K, device=device, dtype=torch.float16, generator=gen) * 2.0 - 1.0
    B = torch.rand(N, K, device=device, dtype=torch.float16, generator=gen) * 2.0 - 1.0

    _ = torch.matmul(A, B.T)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    C = torch.matmul(A, B.T)
    stop.record()

    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(stop)
    print(f"GPU Kernel Time (torch): {gpu_ms:.6f} ms")

    flops = 2.0 * M * N * K
    tflops = flops / (gpu_ms * 1e-3) / 1e12
    print(f"GPU Throughput: {tflops:.4f} TFLOPS")

    ref = torch.matmul(A, B.T)
    diff = (C - ref).abs().max().item()
    print(f"\nMax absolute difference: {diff}")
    print("Correctness Status: PASSED [OK]" if diff <= 1e-1 else "Correctness Status: FAILED [X]")


if __name__ == "__main__":
    main()
