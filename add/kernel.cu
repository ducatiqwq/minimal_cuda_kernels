// kernel.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Use __restrict__ to indicate that the pointers are not aliased
__global__ void vector_add_kernel(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 a = __ldcg(&A[i]);   // caching might be detrimental for streaming data, so disable L1 cache
        float4 b = __ldcg(&B[i]);
        __stcg(&C[i], make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w));
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void launch_vector_add(const float* d_A, const float* d_B, float* d_C, int N) {
    assert(N % 4 == 0 && "N must be divisible by 4");

    int threadsPerBlock = 256;
    int blocksPerGrid = ceil_div(N / 4, threadsPerBlock);
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(d_A),
        reinterpret_cast<const float4*>(d_B),
        reinterpret_cast<float4*>(d_C),
        N / 4
    );
}