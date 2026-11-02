#include "common.cuh"
#include <cuda_pipeline.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "Build overlap.cu for compute capability 8.0 or newer."
#endif
constexpr int THREADS = 128;
__host__ __device__ float transform(float x, int rounds) {
    for (int r = 0; r < rounds; ++r) x = fmaf(x, 1.0001f, 0.0001f);
    return x;
}
__host__ __device__ float4 compute(float4 x, int rounds) {
    return make_float4(transform(x.x, rounds), transform(x.y, rounds),
                       transform(x.z, rounds), transform(x.w, rounds));
}
// The baseline uses the same number of shared slots, so resource allocation is comparable.
__global__ void serial_tiles(const float4* in, float4* out, int tiles, int rounds) {
    __shared__ float4 slot[2][THREADS];
    int t = threadIdx.x;
    for (int k = 0; k < tiles; ++k) {
        int base = (blockIdx.x * tiles + k) * THREADS;
        slot[k % 2][t] = in[base + t];
        __syncthreads();
        out[base + t] = compute(slot[k % 2][(t + 1) % THREADS], rounds);
        __syncthreads();
    }
}
__global__ void pipelined_tiles(const float4* in, float4* out, int tiles, int rounds) {
    __shared__ float4 slot[2][THREADS];
    int t = threadIdx.x;
    int first = blockIdx.x * tiles * THREADS;
    __pipeline_memcpy_async(&slot[0][t], &in[first + t], sizeof(float4));
    __pipeline_commit(); // Submit tile 0 before entering the loop.
    for (int k = 0; k < tiles; ++k) {
        int base = (blockIdx.x * tiles + k) * THREADS;
        if (k + 1 < tiles) {
            __pipeline_memcpy_async(&slot[(k + 1) % 2][t], &in[base + THREADS + t], sizeof(float4));
            __pipeline_commit();
            __pipeline_wait_prior(1); // Current tile complete; newest tile may still be in flight.
        } else {
            __pipeline_wait_prior(0); // Drain the final tile, including the tiles=1 case.
        }
        __syncthreads(); // Every thread has waited for its copies; allow cross-thread reads.
        out[base + t] = compute(slot[k % 2][(t + 1) % THREADS], rounds);
        __syncthreads(); // All consumers finished before a future iteration reuses this slot.
    }
}
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) { std::fprintf(stderr, "Usage: overlap [tiles_per_block rounds]\n"); return 1; }
    const int tiles = argc == 3 ? argument(argv[1], 64) : 8;
    const int rounds = argc == 3 ? argument(argv[2], 256) : 16;
    auto prop = device();
    if (prop.major < 8) { std::fprintf(stderr, "Requires CC 8.0+ for this async-copy example.\n"); return 1; }
    const int blocks = prop.multiProcessorCount * 2;
    const size_t count = static_cast<size_t>(blocks) * tiles * THREADS;
    std::vector<float4> input(count), output(count);
    std::vector<float> actual(count * 4), expected(count * 4);
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(i % 97) / 128.0f;
        input[i] = make_float4(x, x + .25f, x + .5f, x + .75f);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t neighbor = (i / THREADS) * THREADS + (i + 1) % THREADS;
        float4 y = compute(input[neighbor], rounds);
        expected[4*i] = y.x; expected[4*i+1] = y.y; expected[4*i+2] = y.z; expected[4*i+3] = y.w;
    }
    DeviceBuffer<float4> din(count), dout(count);
    CHECK(cudaMemcpy(din.data, input.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(dout.data, 0xff, count * sizeof(float4)));
        auto launch = [&] {
            if (variant == 0) serial_tiles<<<blocks, THREADS>>>(din.data, dout.data, tiles, rounds);
            else pipelined_tiles<<<blocks, THREADS>>>(din.data, dout.data, tiles, rounds);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(output.data(), dout.data, count * sizeof(float4), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            actual[4*i] = output[i].x; actual[4*i+1] = output[i].y;
            actual[4*i+2] = output[i].z; actual[4*i+3] = output[i].w;
        }
        validate(actual, expected);
        std::printf("%s: %.6f ms; tiles/block=%d rounds=%d\n", variant ? "async_double_buffer" : "serial", measure(launch), tiles, rounds);
    }
    std::puts("Timing compares whole implementations, not isolated overlap. Check SASS/profiling; speedup is not guaranteed.");
}
