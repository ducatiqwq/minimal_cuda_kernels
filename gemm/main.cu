// main.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using fp16 = __half;

extern void launch_gemm(const fp16* d_A, const fp16* d_B, fp16* d_C, int M, int N, int K);
extern void launch_gemm_cublas(const fp16* d_A, const fp16* d_B, fp16* d_C, int M, int N, int K);

void print_device_metadata() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== Device Metadata ===" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "VRAM (Global Memory): " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "=======================\n" << std::endl;
}

float cpu_dot(
    const std::vector<fp16>& A,
    const std::vector<fp16>& B,
    int m, int n, int K, int N
) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
    }
    return sum;
}

void cpu_gemm_naive(
    const std::vector<fp16>& A,
    const std::vector<fp16>& B,
    std::vector<fp16>& C,
    int M, int N, int K
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m + n * M] = __float2half(cpu_dot(A, B, m, n, K, N));
        }
    }
}

int main(int argc, char** argv) {
    print_device_metadata();

    bool debug = false;
    bool use_cublas = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--debug") {
            debug = true;
        } else if (arg == "--cublas") {
            use_cublas = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return EXIT_FAILURE;
        }
    }

    auto launch = use_cublas ? launch_gemm_cublas : launch_gemm;
    const char* implementation = use_cublas ? "cuBLAS" : "custom CUTLASS";

    int M = debug ? 128 : 8192;
    int N = debug ? 512 : 8192;
    int K = debug ? 64 : 8192;

    std::cout << "Configuration: GEMM (" << M << ", " << N << ", " << K << "), dtype=float16" << std::endl;
    std::cout << "Implementation: " << implementation << "\n" << std::endl;

    size_t size_A = M * static_cast<size_t>(K) * sizeof(fp16);
    size_t size_B = N * static_cast<size_t>(K) * sizeof(fp16);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(fp16);

    std::vector<fp16> h_A(M * static_cast<size_t>(K));
    std::vector<fp16> h_B(N * static_cast<size_t>(K));
    std::vector<fp16> h_C_cpu(static_cast<size_t>(M) * N);
    std::vector<fp16> h_C_gpu(static_cast<size_t>(M) * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            h_A[m * K + k] = __float2half(dist(gen));
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            h_B[n * K + k] = __float2half(dist(gen));
        }
    }

    if (debug) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm_naive(h_A, h_B, h_C_cpu, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
        std::cout << "CPU Brute Force Time: " << cpu_duration.count() << " ms" << std::endl;
    }

    fp16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    std::cout << "GPU Kernel Time: " << gpu_ms << " ms" << std::endl;

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double tflops = flops / (gpu_ms * 1e-3) / 1e12;
    std::cout << "GPU Throughput: " << tflops << " TFLOPS" << std::endl;

    bool correct = true;
    float max_diff = 0.0f;
    int max_diff_row = -1;
    int max_diff_col = -1;
    const float EPSILON = 0.5f;

    if (debug) {
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                float diff = std::abs(__half2float(h_C_cpu[m + n * M]) - __half2float(h_C_gpu[m + n * M]));
                if (diff > max_diff) {
                    max_diff = diff;
                    max_diff_row = m;
                    max_diff_col = n;
                    printf("Expected: %f, Actual: %f, Diff: %f\n", __half2float(h_C_cpu[m + n * M]), __half2float(h_C_gpu[m + n * M]), diff);
                }
            }
        }
    } else {
        std::mt19937 verify_gen(123);
        std::uniform_int_distribution<int> row_dist(0, M - 1);
        std::uniform_int_distribution<int> col_dist(0, N - 1);
        constexpr int NUM_SAMPLES = 1000;

        for (int s = 0; s < NUM_SAMPLES; ++s) {
            int m = row_dist(verify_gen);
            int n = col_dist(verify_gen);
            float ref = cpu_dot(h_A, h_B, m, n, K, N);
            float gpu_val = __half2float(h_C_gpu[m + n * M]);
            h_C_cpu[m + n * M] = __float2half(ref);

            float diff = std::abs(ref - gpu_val);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_row = m;
                max_diff_col = n;
            }
        }
        std::cout << "Correctness check: " << NUM_SAMPLES << " random output samples vs CPU dot products" << std::endl;
    }

    if (max_diff > EPSILON) {
        correct = false;
        std::cout << "Mismatch at (" << max_diff_row << ", " << max_diff_col << ")" << std::endl;
        std::cout << "Expected: " << __half2float(h_C_cpu[max_diff_row + max_diff_col * M]) << std::endl;
        std::cout << "Actual: " << __half2float(h_C_gpu[max_diff_row + max_diff_col * M]) << std::endl;
    }

    std::cout << "\nMax absolute difference: " << max_diff << std::endl;
    if (correct) {
        std::cout << "Correctness Status: PASSED [OK]" << std::endl;
    } else {
        std::cout << "Correctness Status: FAILED [X]" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return correct ? 0 : 1;
}
