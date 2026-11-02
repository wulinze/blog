#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <vector>
inline void check_cuda(cudaError_t code, const char* expression, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA line %d: %s: %s\n", line, expression, cudaGetErrorString(code));
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(expr) check_cuda((expr), #expr, __LINE__)
inline int argument(char* value, int maximum) {
    char* end = nullptr; errno = 0;
    long n = std::strtol(value, &end, 10);
    if (errno || *end || end == value || n < 1 || n > maximum) {
        std::fprintf(stderr, "Expected integer in [1, %d]\n", maximum); std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(n);
}
inline cudaDeviceProp device() {
    int ordinal = 0; CHECK(cudaGetDevice(&ordinal));
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, ordinal));
    std::printf("GPU: %s; CC %d.%d; SMs %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return prop;
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
template<class Launch> float measure(Launch launch, int repetitions = 20) {
    for (int i = 0; i < 5; ++i) { launch(); CHECK(cudaGetLastError()); }
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) { launch(); CHECK(cudaGetLastError()); }
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return ms / repetitions;
}
inline void validate(const std::vector<float>& actual, const std::vector<float>& expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1e-4f + 1e-4f * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "FAIL index %zu: got %.9g expected %.9g\n", i, actual[i], expected[i]);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("PASS: %zu elements\n", actual.size());
}
