// main.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <random>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

extern void launch_reduce_max(const float* d_input, float* d_partial, int N);
extern size_t reduce_max_partial_size(int N);

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

float cpu_reduce_max(const std::vector<float>& A, int N) {
    float m = -FLT_MAX;
    for (int i = 0; i < N; ++i) {
        m = std::max(m, A[i]);
    }
    return m;
}

float host_reduce_max(const std::vector<float>& partial) {
    float m = -FLT_MAX;
    for (float v : partial) {
        m = std::max(m, v);
    }
    return m;
}

int main(int argc, char** argv) {
    print_device_metadata();

    int N = 33554432;

    std::cout << "Configuration: Vector Length (N) = " << N << "\n" << std::endl;

    size_t size_bytes = N * sizeof(float);
    size_t partial_size = reduce_max_partial_size(N);

    std::vector<float> h_A(N);

    if (argc > 1 && argv[1] == std::string("--debug")) {
        for (int i = 0; i < N; ++i) {
            h_A[i] = static_cast<float>(i);
        }
    } else {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        for (int i = 0; i < N; ++i) {
            h_A[i] = dist(gen);
        }
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_max = cpu_reduce_max(h_A, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Brute Force Time: " << cpu_duration.count() << " ms" << std::endl;

    float *d_A = nullptr, *d_partial = nullptr;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_partial, partial_size * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch_reduce_max(d_A, d_partial, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch_reduce_max(d_A, d_partial, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    std::vector<float> h_partial(partial_size);
    cudaMemcpy(h_partial.data(), d_partial, partial_size * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_max = host_reduce_max(h_partial);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    std::cout << "Total Execution Time: " << total_ms << " ms" << std::endl;

    float diff = std::abs(cpu_max - gpu_max);
    const float EPSILON = 1e-5f;

    std::cout << "\nCPU max: " << cpu_max << ", GPU max: " << gpu_max << std::endl;
    std::cout << "Partial outputs: " << partial_size << std::endl;
    std::cout << "Absolute difference: " << diff << std::endl;
    if (diff <= EPSILON) {
        std::cout << "Correctness Status: PASSED [OK]" << std::endl;
    } else {
        std::cout << "Correctness Status: FAILED [X]" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
