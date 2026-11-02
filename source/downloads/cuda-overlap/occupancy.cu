#include "common.cuh"
__global__ void probe(float* out) {
    extern __shared__ float tile[];
    tile[threadIdx.x] = static_cast<float>(threadIdx.x);
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = tile[(threadIdx.x + 1) % blockDim.x];
}
int main() {
    auto prop = device();
    cudaFuncAttributes attributes{}; CHECK(cudaFuncGetAttributes(&attributes, probe));
    std::printf("Registers/thread: %d; static shared/block: %zu bytes\n", attributes.numRegs, attributes.sharedSizeBytes);
    for (int threads : {128, 256, 512}) for (int bytes : {8192, 16384, 32768}) {
        if (threads > attributes.maxThreadsPerBlock || bytes + attributes.sharedSizeBytes > prop.sharedMemPerBlock) continue;
        int blocks = 0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, probe, threads, bytes));
        int warps = blocks * ((threads + prop.warpSize - 1) / prop.warpSize);
        int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        std::printf("threads=%d dynamic_shared=%d: theoretical blocks/SM=%d warps/SM=%d occupancy=%.2f%%\n", threads, bytes, blocks, warps, 100.0 * warps / max_warps);
    }
    std::puts("This queries capacity; it does not launch the kernel or measure achieved occupancy/performance.");
}
