// kernel.cu
#include <cuda_runtime.h>
#include <cassert>
#include <cfloat>

constexpr int BlockDimX = 512;
constexpr int BlockNumWarps = BlockDimX / 32;
constexpr int NumLoadsPerThread = 1;

template <int BlockDimX_, int NumLoadsPerThread_>
__global__ void reduce_max_kernel(
    const float4* __restrict__ in,
    float* __restrict__ partial,
    int N
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int gid = blockIdx.x * BlockDimX_ + tid;
    const int stride4 = (BlockDimX_ * gridDim.x) / 4;
    
    float M = -FLT_MAX;
    __shared__ float smem[BlockNumWarps];

    // Part 1: thread-local
    #pragma unroll
    for (int k = 0; k < NumLoadsPerThread_; ++k) {
        const float4 v = in[gid + k * stride4];
        M = fmaxf(M, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    
    // Part 2: warp-local
    // It seems the naive warp-level max-reduction for `float` is supported only after SM_100 (blackwell)
    // For H100/A100, we have to use warp shuffling (which requires logarithmic number of instructions).
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        float neighbor = __shfl_xor_sync(0xffffffff, M, ofs);
        M = fmaxf(M, neighbor);
    }
    if (lane_id == 0) {
        smem[warp_id] = M;
    }
    
    // Part 3: block-local
    __syncthreads();
    static_assert(BlockDimX == 512, "BlockDimX must be 512");
    // (theoretically) occupancy is 66.67% for BlockDimX = 1024, since there can only be one 32-warp block in a 48-warp SM
    // However, occupancy can be 100% for BlockDimX = 512, as there can be three 16-warp blocks in a 48-warp SM
    // Above optimization is for RTX4090D; A100/H100 SM is 64-warp

    if (warp_id == 0) {
        float M = (lane_id < BlockNumWarps) ? smem[lane_id] : -FLT_MAX;

        #pragma unroll
        for (int ofs = (BlockNumWarps >> 1); ofs > 0; ofs >>= 1) {
            float neighbor = __shfl_xor_sync(0xffffffff, M, ofs);
            M = fmaxf(M, neighbor);
        }
        if (lane_id == 0) {
            partial[blockIdx.x] = M;
        }
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

size_t reduce_max_partial_size(int N) {
    return static_cast<size_t>(ceil_div(N, BlockDimX * NumLoadsPerThread * 4));
}

void launch_reduce_max(const float* d_input, float* d_partial, int N) {
    assert(N % 4 == 0 && "N must be divisible by 4");

    const int gridDimX = ceil_div(N, BlockDimX * NumLoadsPerThread * 4);
    reduce_max_kernel<BlockDimX, NumLoadsPerThread><<<gridDimX, BlockDimX>>>(
        reinterpret_cast<const float4*>(d_input),
        d_partial,
        N
    );
}
