"""Torch equivalent of the CUDA vector-add kernel in kernel.cu / main.cu.

Performs C = A + B on the GPU for two float32 vectors of length N = 2^25,
and reports the time taken by torch's CUDA add kernel (measured with CUDA
events after a warmup, matching main.cu).
"""

import torch

# Match main.cu: N = 2^25 (~33.5 million elements), float32 vectors.
N = 33554432

def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available.")

    device = torch.device("cuda")

    # Device metadata (mirrors print_device_metadata in main.cu).
    props = torch.cuda.get_device_properties(device)
    print("=== Device Metadata ===")
    print(f"Device Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"VRAM (Global Memory): {props.total_memory // (1024 * 1024)} MB")
    print(f"Multiprocessors: {props.multi_processor_count}")
    print("=======================\n")

    print(f"Configuration: Vector Length (N) = {N}\n")

    # Same fixed seed / uniform range [-10, 10) as main.cu's release path.
    gen = torch.Generator(device=device).manual_seed(42)
    A = (torch.rand(N, device=device, dtype=torch.float32, generator=gen) * 20.0) - 10.0
    B = (torch.rand(N, device=device, dtype=torch.float32, generator=gen) * 20.0) - 10.0

    # Warmup (kernel launch + any lazy allocation), matching main.cu.
    C = A + B
    torch.cuda.synchronize()

    # Time the add kernel with CUDA events, like cudaEvent* in main.cu.
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    C = A + B
    stop.record()

    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(stop)
    print(f"GPU Kernel Time (torch): {gpu_ms:.6f} ms")

    # Correctness sanity check against a reference add.
    ref = A + B
    max_diff = (C - ref).abs().max().item()
    print(f"\nMax absolute difference: {max_diff}")
    print("Correctness Status: PASSED [OK]" if max_diff <= 1e-5 else "Correctness Status: FAILED [X]")


if __name__ == "__main__":
    main()
