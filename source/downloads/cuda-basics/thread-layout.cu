#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t error, const char* step) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
struct ThreadInfo { int block, thread, warp, lane; };
__global__ void record_threads(ThreadInfo* output) {
    const int thread = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + thread;
    output[index] = {static_cast<int>(blockIdx.x), thread, thread / warpSize, thread % warpSize};
}
int main() {
    constexpr int blocks = 2, threads = 40, count = blocks * threads;
    ThreadInfo result[count];
    ThreadInfo* device = nullptr;
    check(cudaMalloc(&device, sizeof(result)), "allocate records");
    record_threads<<<blocks, threads>>>(device);
    check(cudaGetLastError(), "launch");
    check(cudaDeviceSynchronize(), "wait");
    check(cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost), "copy records");
    bool correct = true;
    for (int i = 0; i < count; ++i) {
        const auto record = result[i];
        correct = correct && record.block == i / threads && record.thread == i % threads
            && record.warp == (i % threads) / 32 && record.lane == (i % threads) % 32;
        std::printf("i=%d block=%d thread=%d warp_in_block=%d lane=%d\n",
                    i, record.block, record.thread, record.warp, record.lane);
    }
    check(cudaFree(device), "free records");
    std::puts(correct ? "PASS" : "FAIL");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
