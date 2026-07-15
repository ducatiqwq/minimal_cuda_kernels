// kernel.cu — tensor-core torture: register init, then raw m16n8k16 MMA PTX only.
#include <cuda_runtime.h>
#include <cstdint>

// Eight independent accumulator pipelines for MMA ILP.
// f16.f16.f16.f16: D/C are 2× uint32 (packed half2 each), A is 4× uint32, B is 2× uint32.
#define MMA_M16N8K16_F16F16F16F16(d0, d1, a0, a1, a2, a3, b0, b1) \
  asm volatile(                                                   \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "        \
      "{%0, %1}, "                                                \
      "{%2, %3, %4, %5}, "                                        \
      "{%6, %7}, "                                                \
      "{%0, %1};"                                                 \
      : "+r"(d0), "+r"(d1)                                        \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// Hot path after init: only mma.sync PTX in the loop body.
__global__ void torture_tensor_core_kernel(float* __restrict__ sink, int iters) {
  // Arbitrary register initialization (no memory traffic in the timed loop).
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t a0 = 0x3C003C00u ^ tid;  // packed half2 ≈ (1,1)
  uint32_t a1 = 0x3C003C00u + tid;
  uint32_t a2 = 0x3C003C00u | (tid << 1);
  uint32_t a3 = 0x3C003C00u + tid * 3u;
  uint32_t b0 = 0x3C003C00u ^ (tid * 7u);
  uint32_t b1 = 0x3C003C00u + tid * 11u;

  // Accumulators: 2 regs/pipe (each holds two fp16 values).
  uint32_t d00 = tid,       d01 = tid + 1u;
  uint32_t d10 = tid + 2u,  d11 = tid + 3u;
  uint32_t d20 = tid + 4u,  d21 = tid + 5u;
  uint32_t d30 = tid + 6u,  d31 = tid + 7u;
  uint32_t d40 = tid + 8u,  d41 = tid + 9u;
  uint32_t d50 = tid + 10u, d51 = tid + 11u;
  uint32_t d60 = tid + 12u, d61 = tid + 13u;
  uint32_t d70 = tid + 14u, d71 = tid + 15u;

  // Pure tensor-core PTX. Each mma is m16n8k16 → 16*8*16*2 = 4096 FLOPs / warp.
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    MMA_M16N8K16_F16F16F16F16(d00, d01, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d10, d11, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d20, d21, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d30, d31, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d40, d41, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d50, d51, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d60, d61, a0, a1, a2, a3, b0, b1);
    MMA_M16N8K16_F16F16F16F16(d70, d71, a0, a1, a2, a3, b0, b1);
  }

  // Sink so the compiler cannot discard the MMAs.
  uint32_t acc = d00 ^ d01 ^ d10 ^ d11 ^ d20 ^ d21 ^ d30 ^ d31 ^ d40 ^ d41 ^ d50 ^
                 d51 ^ d60 ^ d61 ^ d70 ^ d71;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *sink = __uint_as_float(acc);
  }
}

void launch_torture_tensor_core(float* d_sink, int blocks, int threads, int iters) {
  torture_tensor_core_kernel<<<blocks, threads>>>(d_sink, iters);
}
