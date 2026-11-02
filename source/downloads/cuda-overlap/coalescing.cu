#include "common.cuh"
// Both variants copy exactly the same N elements. Only thread-to-address mapping changes.
template<bool Strided> __global__ void copy_layout(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int j = i;
        if constexpr (Strided) j = (i / 1024) * 1024 + (i % 32) * 32 + (i % 1024) / 32;
        out[j] = in[j];
    }
}
int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "Usage: coalescing [N, multiple of 1024]\n"); return 1; }
    const int n = argc == 2 ? argument(argv[1], 1 << 28) : 1 << 24;
    if (n % 1024) { std::fprintf(stderr, "N must be a multiple of 1024 for the permutation.\n"); return 1; }
    auto prop = device();
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>((i % 251) - 125);
    DeviceBuffer<float> d_in(n), d_out(n);
    CHECK(cudaMemcpy(d_in.data, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = std::min((n + 255) / 256, prop.multiProcessorCount * 8);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(d_out.data, 0xff, n * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) copy_layout<true><<<blocks, 256>>>(d_in.data, d_out.data, n);
            else copy_layout<false><<<blocks, 256>>>(d_in.data, d_out.data, n);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), d_out.data, n * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, in);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GB/s\n", variant ? "coalesced" : "strided", ms, 2.0 * n * sizeof(float) / (ms * 1e6));
    }
    std::puts("Effective bytes = input + output. Repeated working set may be cached; this is not a DRAM bandwidth measurement.");
}
