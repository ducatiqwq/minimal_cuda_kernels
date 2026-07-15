// main.cu — launch the tensor-core torture kernel and report achieved TFLOPS.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

extern void launch_torture_tensor_core(float* d_sink, int blocks, int threads, int iters);

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — "      \
                << cudaGetErrorString(err__) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

void print_device_metadata() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "=== Device Metadata ===\n"
            << "Device Name: " << prop.name << "\n"
            << "Compute Capability: " << prop.major << "." << prop.minor << "\n"
            << "SMs: " << prop.multiProcessorCount << "\n"
            << "=======================\n"
            << std::endl;
}

int main(int argc, char** argv) {
  print_device_metadata();

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Saturate the GPU: many resident warps, long inner loop of MMAs only.
  const int threads = 256;  // 8 warps / block
  const int blocks = prop.multiProcessorCount * 32;
  const int pipes = 8;  // must match TC_PIPES in kernel.cu
  int iters = 500000;
  if (argc > 1) {
    iters = std::atoi(argv[1]);
  }

  const int warps = (blocks * threads) / 32;
  // m16n8k16 FMA count: 16*8*16*2 FLOPs per MMA per warp
  const double flops_per_mma = 16.0 * 8.0 * 16.0 * 2.0;
  const double total_flops =
      double(warps) * double(iters) * double(pipes) * flops_per_mma;

  std::cout << "Config: blocks=" << blocks << " threads=" << threads
            << " warps=" << warps << " iters=" << iters << " pipes=" << pipes
            << "\nMMA atom: m16n8k16.row.col.f16.f16.f16.f16\n"
            << std::endl;

  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

  // Warmup
  launch_torture_tensor_core(d_sink, blocks, threads, iters / 10);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  launch_torture_tensor_core(d_sink, blocks, threads, iters);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double tflops = (total_flops / (ms * 1e-3)) / 1e12;

  float h_sink = 0.f;
  CUDA_CHECK(cudaMemcpy(&h_sink, d_sink, sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "Kernel time: " << ms << " ms\n"
            << "Achieved:    " << tflops << " TFLOPS\n"
            << "Sink:        " << h_sink << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
