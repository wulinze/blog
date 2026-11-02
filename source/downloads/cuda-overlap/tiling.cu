#include "common.cuh"
constexpr int TILE = 16;
__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0;
        for (int p = 0; p < k; ++p) sum += a[row * k + p] * b[p * n + col];
        c[row * n + col] = sum;
    }
}
__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float at[TILE][TILE], bt[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < k; base += TILE) {
        int ak = base + threadIdx.x, bk = base + threadIdx.y;
        at[threadIdx.y][threadIdx.x] = row < m && ak < k ? a[row * k + ak] : 0.0f;
        bt[threadIdx.y][threadIdx.x] = bk < k && col < n ? b[bk * n + col] : 0.0f;
        __syncthreads(); // All tile values ready before cross-thread reads.
        for (int p = 0; p < TILE; ++p) sum += at[threadIdx.y][p] * bt[p][threadIdx.x];
        __syncthreads(); // All readers done before next iteration overwrites the tile.
    }
    if (row < m && col < n) c[row * n + col] = sum;
}
int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) { std::fprintf(stderr, "Usage: tiling [M N K]\n"); return 1; }
    int m = argc == 4 ? argument(argv[1], 1024) : 129;
    int n = argc == 4 ? argument(argv[2], 1024) : 193;
    int k = argc == 4 ? argument(argv[3], 1024) : 257;
    device();
    std::vector<float> a(m * k), b(k * n), out(m * n), expected(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = ((i % 13) - 6) / 8.0f;
    for (int i = 0; i < k * n; ++i) b[i] = ((i % 11) - 5) / 16.0f;
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int p = 0; p < k; ++p) sum += static_cast<double>(a[row * k + p]) * b[p * n + col];
        expected[row * n + col] = static_cast<float>(sum);
    }
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(out.size());
    CHECK(cudaMemcpy(da.data, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db.data, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    for (int variant = 0; variant < 2; ++variant) {
        CHECK(cudaMemset(dc.data, 0xff, out.size() * sizeof(float)));
        auto launch = [&] {
            if (variant == 0) matmul_naive<<<grid, block>>>(da.data, db.data, dc.data, m, n, k);
            else matmul_tiled<<<grid, block>>>(da.data, db.data, dc.data, m, n, k);
        };
        launch(); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out.data(), dc.data, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        validate(out, expected);
        const float ms = measure(launch);
        std::printf("%s: %.6f ms, effective %.3f GFLOP/s\n", variant ? "tiled" : "naive", ms, 2.0 * m * n * k / (ms * 1e6));
    }
    std::puts("FP32 teaching kernels, not Tensor Core GEMM. CPU reference and copies excluded from event timing.");
}
