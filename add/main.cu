// main.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// Forward declaration of the kernel launcher defined in kernel.cu
extern void launch_vector_add(const float* d_A, const float* d_B, float* d_C, int N);

// Print GPU metadata
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

// CPU Implementation of Vector Addition (Brute Force)
void cpu_vector_add(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    print_device_metadata();

    // Hyperparameters
    // N = 2^25 (approx 33.5 million elements). 
    // This requires ~134 MB per vector (A, B, C), totaling ~400 MB.
    int N = 33554432; 

    std::cout << "Configuration: Vector Length (N) = " << N << "\n" << std::endl;

    size_t size_bytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C_cpu(N);
    std::vector<float> h_C_gpu(N);

    // Generate data
    if (argc > 1 && argv[1] == std::string("--debug")) {
        for (int i = 0; i < N; ++i) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(i * 2);
        }
    } else {
        std::mt19937 gen(42); // fixed seed for determinism
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
        for (int i = 0; i < N; ++i) {
            h_A[i] = dist(gen);
            h_B[i] = dist(gen);
        }
    }

    // --- CPU Brute Force Execution ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_vector_add(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Brute Force Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- GPU Execution ---
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    launch_vector_add(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Time the kernel
    cudaEventRecord(start);
    launch_vector_add(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    std::cout << "GPU Kernel Time: " << gpu_ms << " ms" << std::endl;

    // Fetch GPU results
    cudaMemcpy(h_C_gpu.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);

    // --- Correctness Check ---
    bool correct = true;
    float max_diff = 0.0f;
    int max_diff_index = -1;
    const float EPSILON = 1e-5f; // Tolerance for 32-bit floats

    for (int i = 0; i < N; ++i) {
        float val_cpu = h_C_cpu[i];
        float val_gpu = h_C_gpu[i];
        float diff = std::abs(val_cpu - val_gpu);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_index = i;
        }
    }
    
    if (max_diff > EPSILON) {
        correct = false;
        std::cout << "Mismatch at index " << max_diff_index
                  << ": CPU=" << h_C_cpu[max_diff_index]
                  << ", GPU=" << h_C_gpu[max_diff_index] << std::endl;
    }

    std::cout << "\nMax absolute difference: " << max_diff << std::endl;
    if (correct) {
        std::cout << "Correctness Status: PASSED [OK]" << std::endl;
    } else {
        std::cout << "Correctness Status: FAILED [X]" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}