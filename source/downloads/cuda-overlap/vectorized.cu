#include "common.cuh"
__global__ void copy_scalar(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) out[i] = in[i];
}
__global__ void copy_float4(const float* in, float* out, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) out4[i] = in4[i];
    if (tid == 0) for (int i = (n / 4) * 4; i < n; ++i) out[i] = in[i];
}
int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "Usage: vectorized [N]\n"); return 1; }
    const int n = argc == 2 ? argument(argv[1], 1 << 28) : (1 << 24) + 3;
    auto prop = device();
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>((i % 127) - 63);
    DeviceBuffer<float> d_in(n), d_out(n); // cudaMalloc base pointers satisfy float4 alignment.
    CHECK(cudaMemcpy(d_in.data, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = std::min((n + 255) / 256, prop.multiProcessorCount * 8);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(d_out.data, 0xff, n * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) copy_scalar<<<blocks, 256>>>(d_in.data, d_out.data, n);
            else copy_float4<<<blocks, 256>>>(d_in.data, d_out.data, n);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), d_out.data, n * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, in);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GB/s\n", variant ? "float4" : "scalar", ms, 2.0 * n * sizeof(float) / (ms * 1e6));
    }
    std::puts("Both variants are coalesced. float4 changes per-thread width, not total useful bytes. Inspect SASS for actual instruction width.");
}
