#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // Small launch dimensions make the thread-to-element mapping easy to trace.
    constexpr int n = 10;
    constexpr int threads_per_block = 4;
    constexpr int blocks = (n + threads_per_block - 1) / threads_per_block;
    const size_t bytes = n * sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (int index = 0; index < n; ++index) {
        a[index] = static_cast<float>(index);
        b[index] = static_cast<float>(2 * index);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    check_cuda(cudaMalloc(&device_a, bytes), "allocate a");
    check_cuda(cudaMalloc(&device_b, bytes), "allocate b");
    check_cuda(cudaMalloc(&device_c, bytes), "allocate c");

    check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a to GPU");
    check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b to GPU");

    add_arrays<<<blocks, threads_per_block>>>(device_a, device_b, device_c, n);
    check_cuda(cudaGetLastError(), "launch add_arrays");
    check_cuda(cudaDeviceSynchronize(), "wait for add_arrays");
    check_cuda(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy c to CPU");

    bool correct = true;
    for (int index = 0; index < n; ++index) {
        const float expected = a[index] + b[index];
        if (!std::isfinite(c[index]) || std::fabs(c[index] - expected) > 1e-6f) {
            std::fprintf(stderr, "Mismatch at %d: got %g, expected %g\n", index, c[index], expected);
            correct = false;
            break;
        }
    }

    check_cuda(cudaFree(device_a), "free a");
    check_cuda(cudaFree(device_b), "free b");
    check_cuda(cudaFree(device_c), "free c");
    if (!correct) return EXIT_FAILURE;
    std::printf("PASS: %d elements; C[9] = %.0f\n", n, c[9]);
    return EXIT_SUCCESS;
}
