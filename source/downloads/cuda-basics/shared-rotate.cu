#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t error, const char* step) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
// This teaching kernel requires exactly 2 blocks, with exactly 4 threads each.
__global__ void rotate_each_block(const int* input, int* output) {
    __shared__ int tile[4];
    const int thread = threadIdx.x;
    const int index = blockIdx.x * 4 + thread;
    tile[thread] = input[index];
    __syncthreads();
    output[index] = tile[(thread + 1) % 4];
}
int main() {
    const int input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int expected[8] = {1, 2, 3, 0, 5, 6, 7, 4};
    int output[8] = {};
    int *device_input = nullptr, *device_output = nullptr;
    check(cudaMalloc(&device_input, sizeof(input)), "allocate input");
    check(cudaMalloc(&device_output, sizeof(output)), "allocate output");
    check(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice), "copy input");
    rotate_each_block<<<2, 4>>>(device_input, device_output);
    check(cudaGetLastError(), "launch");
    check(cudaDeviceSynchronize(), "wait");
    check(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy output");
    bool correct = true;
    for (int i = 0; i < 8; ++i) {
        correct = correct && output[i] == expected[i];
        std::printf("%d%s", output[i], i == 7 ? "\n" : " ");
    }
    check(cudaFree(device_input), "free input");
    check(cudaFree(device_output), "free output");
    std::puts(correct ? "PASS" : "FAIL");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
