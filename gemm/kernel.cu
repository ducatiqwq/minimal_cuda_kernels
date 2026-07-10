#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

constexpr int BLOCK_M   = 256;
constexpr int BLOCK_N   = 128;
constexpr int SMEM_K    = 64;
constexpr int WARP_M    = 2;
constexpr int WARP_N    = 4;
constexpr int STAGES_K  = 2;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
    class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma)
{
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using Storage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    int k_tile_count = size<3>(tAgA);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    auto K_BLOCK_MAX = size<2>(tCrA);

    int k_tile_next = 0;
    CUTE_UNROLL
    for (int stage = 0; stage < STAGES_K; ++stage) {
        if (k_tile_next < k_tile_count) {
            copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, stage));
            copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, stage));
            cp_async_fence();
            ++k_tile_next;
        }
    }

    CUTE_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int read_stage = k_tile % STAGES_K;
        int wait_n = min(k_tile_next - k_tile - 1, STAGES_K - 1);

        static_assert(STAGES_K <= 3 && "STAGES_K must be smaller or equal than 3");
        if (wait_n <= 0) {
            cp_async_wait<0>();
        } else if (wait_n <= 1) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<2>();
        }
        __syncthreads();

        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_atom_a, tXsA(_, _, k_block, read_stage), tXrA(_, _, k_block));
            copy(s2r_atom_b, tXsB(_, _, k_block, read_stage), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
        __syncthreads();    // strange that deleting this line leads to incorrect results

        if (k_tile_next < k_tile_count) {
            copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, read_stage));
            copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, read_stage));
            cp_async_fence();
            ++k_tile_next;
        }
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = bf16(tCrC(i));
    }
}

void launch_gemm_impl(
    int m, int n, int k,
    bf16 const* A, int ldA,
    bf16 const* B, int ldB,
    bf16* C, int ldC,
    cudaStream_t stream = 0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // Column-major: A(M,K) lda=M, B(N,K) ldb=N, C(M,N) ldc=M
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<BLOCK_M>{};
    auto bN = Int<BLOCK_N>{};
    auto bK = Int<SMEM_K>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<STAGES_K>{};

    auto swizzle_atom = composition(
        Swizzle<log_2(static_cast<uint32_t>(SMEM_K)) - 3, 3, 3>{},
        Layout<Shape<_8, Int<SMEM_K>>, Stride<Int<SMEM_K>, _1>>{});

    auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<Int<WARP_M * WARP_N * (256 / SMEM_K)>, Int<SMEM_K / 8>>, Stride<Int<SMEM_K / 8>, _1>>{},
        Layout<Shape<_1, _8>>{});
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<Int<WARP_M * WARP_N * (256 / SMEM_K)>, Int<SMEM_K / 8>>, Stride<Int<SMEM_K / 8>, _1>>{},
        Layout<Shape<_1, _8>>{});

    TiledMMA mmaC = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<Int<WARP_M>, Int<WARP_N>>>{});

    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_a;
    Copy_Atom<SM75_U32x2_LDSM_N, bf16> s2r_atom_b;

    int smem_size = int(sizeof(SharedStorage<bf16, bf16, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    auto kernel_fptr = gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        bf16, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_a),
        bf16, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_b),
        bf16, decltype(dC), decltype(sC), decltype(mmaC)>;

    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, copyA, s2r_atom_a,
        B, dB, sB, copyB, s2r_atom_b,
        C, dC, sC, mmaC);
}

void launch_gemm(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, __nv_bfloat16* d_C, int M, int N, int K) {
    launch_gemm_impl(
        M, N, K,
        reinterpret_cast<bf16 const*>(d_A), K,
        reinterpret_cast<bf16 const*>(d_B), K,
        reinterpret_cast<bf16*>(d_C), M);
}

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCreate failed with status %d\n", static_cast<int>(status));
            std::exit(EXIT_FAILURE);
        }
        initialized = true;
    }
    return handle;
}

void launch_gemm_cublas_impl(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    __nv_bfloat16* d_C,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_R_16BF, K,
        d_B, CUDA_R_16BF, K,
        &beta,
        d_C, CUDA_R_16BF, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasGemmEx failed with status %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

void launch_gemm_cublas(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, __nv_bfloat16* d_C, int M, int N, int K) {
    launch_gemm_cublas_impl(d_A, d_B, d_C, M, N, K);
}
