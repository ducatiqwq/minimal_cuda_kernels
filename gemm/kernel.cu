#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace cute;
using fp16 = cutlass::half_t;

constexpr int BLOCK_M = 128;  // {64, 128, 256}
constexpr int BLOCK_N = 128;  // {64, 128, 256}
constexpr int SMEM_K  = 32;   // {32, 64, 128}
constexpr int WARP_M  = 1;    // {1, 2, 4}
constexpr int WARP_N  = 4;    // {1, 2, 4}
constexpr int STAGES_K = 2;    // {1, 2}
constexpr int MMA_ATOM_TYPE = 16816;    // {16816, 1688}

constexpr int VECTOR_BITS = 128;
constexpr int NUM_THREADS = WARP_M * WARP_N * 32;
constexpr int VECTOR_ELEMS = (VECTOR_BITS / 8) / sizeof(fp16);
constexpr int G2S_COPY_K = SMEM_K / VECTOR_ELEMS;
constexpr int VECTOR_LOG2 = log_2(static_cast<uint32_t>(VECTOR_ELEMS));
constexpr int SWIZZLE_K = log_2(static_cast<uint32_t>(SMEM_K)) - VECTOR_LOG2;
constexpr int C_COPY_THREADS_M = BLOCK_M / VECTOR_ELEMS;
constexpr int C_COPY_THREADS_N = NUM_THREADS / C_COPY_THREADS_M;

static_assert(SMEM_K % VECTOR_ELEMS == 0);
static_assert(BLOCK_M % VECTOR_ELEMS == 0);
static_assert(NUM_THREADS % C_COPY_THREADS_M == 0);

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <int AtomType = MMA_ATOM_TYPE>
auto make_mma_s2r_atoms()
{
    static_assert(AtomType == 16816 || AtomType == 1688, "MMA_ATOM_TYPE must be 16816 or 1688");

    auto thr_layout = Layout<Shape<Int<WARP_M>, Int<WARP_N>>>{};
    if constexpr (AtomType == 16816) {
        return make_tuple(
            make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, thr_layout),
            Copy_Atom<SM75_U32x4_LDSM_N, fp16>{},
            Copy_Atom<SM75_U32x2_LDSM_N, fp16>{}
        );
    } else if constexpr (AtomType == 1688) {
        return make_tuple(
            make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{}, thr_layout),
            Copy_Atom<SM75_U32x2_LDSM_N, fp16>{},
            Copy_Atom<SM75_U32x1_LDSM_N, fp16>{}
        );
    }
}

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct MainloopStorage {
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <
    class ElementA, class ElementB, class ElementC,
    class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
union SharedStorage {
    MainloopStorage<ElementA, ElementB, SmemLayoutA, SmemLayoutB> mainloop;
    cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> epilogue;
};

template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
    class TC, class CStride, class CSmemLayout, class TiledCopyC, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c, TiledMma mma)
{
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

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
    using Storage = SharedStorage<TA, TB, TC, ASmemLayout, BSmemLayout, CSmemLayout>;
    Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout);

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
            copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, stage));
            copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, stage));
            cp_async_fence();
            ++k_tile_next;
        }
    }

    CUTE_NO_UNROLL
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
            copy(s2r_atom_b, tXsB(_, _, k_block, read_stage), tXrB(_, _, k_block));
            copy(s2r_atom_a, tXsA(_, _, k_block, read_stage), tXrA(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
        __syncthreads();

        if (k_tile_next < k_tile_count) {
            copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, read_stage));
            copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, read_stage));
            cp_async_fence();
            ++k_tile_next;
        }
    }

    Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.begin()), sC_layout);
    Tensor tCrC_out = make_fragment_like<TC>(tCrC);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCrC_out(i) = TC(tCrC(i));
    }

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TC>{}, mma);
    auto r2s_thr_copy = r2s_copy.get_slice(threadIdx.x);
    Tensor tCrC_r2s = r2s_thr_copy.retile_S(tCrC_out);
    Tensor tCsC_r2s = r2s_thr_copy.partition_D(sC);

    copy(r2s_copy, tCrC_r2s, tCsC_r2s);
    __syncthreads();

    auto s2g_thr_copy = copy_c.get_slice(threadIdx.x);
    Tensor tCsC_s2g = s2g_thr_copy.partition_S(sC);
    Tensor tCgC_s2g = s2g_thr_copy.partition_D(gC);
    Tensor tCrC_s2g = make_tensor<TC>(shape(tCgC_s2g));

    copy(copy_c, tCsC_s2g, tCrC_s2g);
    copy(copy_c, tCrC_s2g, tCgC_s2g);
}

void launch_gemm_impl(
    int m, int n, int k,
    fp16 const* A, int ldA,
    fp16 const* B, int ldB,
    fp16* C, int ldC,
    cudaStream_t stream = 0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<BLOCK_M>{};
    auto bN = Int<BLOCK_N>{};
    auto bK = Int<SMEM_K>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<STAGES_K>{};

    auto swizzle_atom = composition(
        Swizzle<VECTOR_LOG2, VECTOR_LOG2, MAX(VECTOR_LOG2, SWIZZLE_K)>{},  // https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900
        Layout<Shape<Int<VECTOR_ELEMS>, Int<SMEM_K>>, Stride<Int<SMEM_K>, _1>>{}
    );

    auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
    auto swizzle_atom_c = composition(
        Swizzle<VECTOR_LOG2, VECTOR_LOG2, MAX(VECTOR_LOG2, SWIZZLE_K)>{},
        Layout<Shape<Int<SMEM_K>, Int<VECTOR_ELEMS>>, Stride<_1, Int<SMEM_K>>>{}
    );
    auto sC = tile_to_shape(swizzle_atom_c, make_shape(bM, bN));

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, fp16>{},
        Layout<Shape<Int<NUM_THREADS / G2S_COPY_K>, Int<G2S_COPY_K>>, Stride<Int<G2S_COPY_K>, _1>>{},
        Layout<Shape<_1, Int<VECTOR_ELEMS>>>{});
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, fp16>{},
        Layout<Shape<Int<NUM_THREADS / G2S_COPY_K>, Int<G2S_COPY_K>>, Stride<Int<G2S_COPY_K>, _1>>{},
        Layout<Shape<_1, Int<VECTOR_ELEMS>>>{});
    TiledCopy copyC = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, fp16>{},
        Layout<Shape<Int<C_COPY_THREADS_M>, Int<C_COPY_THREADS_N>>, Stride<_1, Int<C_COPY_THREADS_M>>>{},
        Layout<Shape<Int<VECTOR_ELEMS>, _1>>{});

    auto [mmaC, s2r_atom_a, s2r_atom_b] = make_mma_s2r_atoms();

    int smem_size = int(sizeof(SharedStorage<fp16, fp16, fp16, decltype(sA), decltype(sB), decltype(sC)>));
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    auto kernel_fptr = gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        fp16, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_a),
        fp16, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_b),
        fp16, decltype(dC), decltype(sC), decltype(copyC), decltype(mmaC)>;

    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, copyA, s2r_atom_a,
        B, dB, sB, copyB, s2r_atom_b,
        C, dC, sC, copyC, mmaC
    );
}

void launch_gemm(const __half* d_A, const __half* d_B, __half* d_C, int M, int N, int K) {
    launch_gemm_impl(
        M, N, K,
        reinterpret_cast<fp16 const*>(d_A), K,
        reinterpret_cast<fp16 const*>(d_B), K,
        reinterpret_cast<fp16*>(d_C), M);
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
    const __half* d_A,
    const __half* d_B,
    __half* d_C,
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
        d_A, CUDA_R_16F, K,
        d_B, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_16F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasGemmEx failed with status %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

void launch_gemm_cublas(const __half* d_A, const __half* d_B, __half* d_C, int M, int N, int K) {
    launch_gemm_cublas_impl(d_A, d_B, d_C, M, N, K);
}
