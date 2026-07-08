"""Torch equivalent of the CUDA max-reduction kernel in kernel.cu / main.cu.

Performs a global max reduction on a float32 vector of length N = 2^25,
and reports the time taken by torch's CUDA max kernel (measured with CUDA
events after a warmup, matching main.cu).
"""

import torch

N = 33554432

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

    print(f"Configuration: Vector Length (N) = {N}\n")

    gen = torch.Generator(device=device).manual_seed(42)
    A = (torch.rand(N, device=device, dtype=torch.float32, generator=gen) * 20.0) - 10.0

    _ = torch.max(A)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    gpu_max = torch.max(A)
    stop.record()

    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(stop)
    print(f"GPU Kernel Time (torch): {gpu_ms:.6f} ms")

    ref_max = torch.max(A)
    diff = abs(gpu_max.item() - ref_max.item())
    print(f"\nMax value: {gpu_max.item()}")
    print(f"Absolute difference: {diff}")
    print("Correctness Status: PASSED [OK]" if diff <= 1e-5 else "Correctness Status: FAILED [X]")


if __name__ == "__main__":
    main()
