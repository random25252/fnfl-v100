#include "fn/kernels/embed/iq4xs_embed.cuh"
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
using namespace fn::embed;
__global__ void k(const uint8_t * rows, float * out) {  // block: row, thread: sb*4+q
    const int b = blockIdx.x, t = threadIdx.x, sb = t / 4, q = t % 4;
    float4 o[16];
    iq4xs_quarter(rows + (long)b * iq4xs_row_bytes + sb * iq4xs_super_bytes, q, o);
    for (int i = 0; i < 16; ++i) reinterpret_cast<float4 *>(out + (long)b * 1280 + sb * 256 + 64 * q)[i] = o[i];
}
int main() {
    FILE * f = fopen("rows.bin", "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 680; rewind(f);
    std::vector<uint8_t> h(n * 680); fread(h.data(), 1, h.size(), f); fclose(f);
    std::vector<float> ref(n * 1280), got(n * 1280);
    f = fopen("ref.bin", "rb"); fread(ref.data(), 4, ref.size(), f); fclose(f);
    uint8_t * dr; float * dout;
    cudaMalloc(&dr, h.size() + 16); cudaMalloc(&dout, got.size() * 4);
    // misalign rows to 2 bytes as in the real table (rows at 680-byte strides)
    cudaMemcpy(dr + 2, h.data(), h.size(), cudaMemcpyHostToDevice);
    k<<<n, 20>>>(dr + 2, dout);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    long bad = 0; float mx = 0;
    for (long i = 0; i < (long)got.size(); ++i) {
        if (memcmp(&got[i], &ref[i], 4) != 0) { if (bad < 5) printf("mismatch %ld: got %g ref %g\n", i, got[i], ref[i]); ++bad; }
        mx = fmaxf(mx, fabsf(got[i]));
    }
    printf("%ld rows, %ld values, %ld bitwise mismatches, max|x| %g\n", n, (long)got.size(), bad, mx);
    return bad != 0;
}
