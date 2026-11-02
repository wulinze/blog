#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t error, const char* step) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
__global__ void produce(int* value) { *value = 5; }
__global__ void consume(int* value) { *value *= 3; }
int main() {
    cudaStream_t producer_stream, consumer_stream;
    cudaEvent_t produced;
    int *device_value = nullptr, *host_value = nullptr;
    check(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking), "create producer stream");
    check(cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking), "create consumer stream");
    check(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming), "create event");
    check(cudaMalloc(&device_value, sizeof(int)), "allocate device value");
    check(cudaMallocHost(&host_value, sizeof(int)), "allocate pinned host value");

    produce<<<1, 1, 0, producer_stream>>>(device_value);
    check(cudaGetLastError(), "launch producer");
    check(cudaEventRecord(produced, producer_stream), "record producer completion");
    check(cudaStreamWaitEvent(consumer_stream, produced, 0), "set dependency");
    consume<<<1, 1, 0, consumer_stream>>>(device_value);
    check(cudaGetLastError(), "launch consumer");
    check(cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, consumer_stream), "enqueue result copy");
    check(cudaStreamSynchronize(consumer_stream), "wait for result");
    const bool correct = *host_value == 15;
    std::printf("%s: result = %d\n", correct ? "PASS" : "FAIL", *host_value);

    check(cudaFree(device_value), "free device value");
    check(cudaFreeHost(host_value), "free pinned value");
    check(cudaEventDestroy(produced), "destroy event");
    check(cudaStreamDestroy(producer_stream), "destroy producer stream");
    check(cudaStreamDestroy(consumer_stream), "destroy consumer stream");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
