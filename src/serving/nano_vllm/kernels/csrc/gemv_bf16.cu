// bf16 GEMV for decode: y[M, N] = x[M, K] @ w[N, K]^T with M <= 8.
//
// Bound by reading w once from HBM, so the design is about keeping enough
// loads in flight: each warp owns ROWS rows of w and every lane issues one
// 128-bit load (8 bf16) per row per step, so a warp reads 512 contiguous
// bytes of each of its rows at a time. x is staged through shared memory in
// K tiles (it is tiny and every warp of the block reuses it), the products
// accumulate in fp32 registers and a warp shuffle reduces each output at the
// end. There is no split-K: the row count per block (4 warps x ROWS) is the
// knob instead. Fewer rows per block means more blocks, which evens out the
// last wave on short weights like o_proj (4096 rows: 256 blocks at ROWS=4,
// 1024 at ROWS=1).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kVec = 8;         // bf16 per 128-bit load
constexpr int kKTile = 2048;    // x tile: M * 2048 * 2 B <= 32 KB of shared memory

__device__ __forceinline__ void bf16x8_to_float(const uint4& v, float out[kVec]) {
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    const float2 f = __bfloat1622float2(pairs[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <int M, int ROWS>
__global__ void __launch_bounds__(kThreads) gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int n,
    int k) {
  __shared__ __align__(16) __nv_bfloat16 xs[M * kKTile];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * kWarps + warp) * ROWS;

  float acc[M][ROWS];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.f;

  for (int k0 = 0; k0 < k; k0 += kKTile) {
    const int kt = min(kKTile, k - k0);
    // Stage x[:, k0:k0+kt] (k and kt are multiples of 8, rows 16-byte aligned).
    for (int i = threadIdx.x * kVec; i < M * kt; i += kThreads * kVec) {
      const int m = i / kt;
      const int kk = i - m * kt;
      *reinterpret_cast<uint4*>(&xs[m * kKTile + kk]) =
          *reinterpret_cast<const uint4*>(&x[static_cast<int64_t>(m) * k + k0 + kk]);
    }
    __syncthreads();

    for (int kk = lane * kVec; kk < kt; kk += 32 * kVec) {
      uint4 wv[ROWS];
      // Issue every row's load before using any of them.
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int row = row0 + r;
        wv[r] = row < n ? __ldg(reinterpret_cast<const uint4*>(
                              &w[static_cast<int64_t>(row) * k + k0 + kk]))
                        : make_uint4(0, 0, 0, 0);
      }
      // Convert each weight once, not once per row of x.
      float wf[ROWS][kVec];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) bf16x8_to_float(wv[r], wf[r]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[kVec];
        bf16x8_to_float(*reinterpret_cast<const uint4*>(&xs[m * kKTile + kk]), xf);
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
          for (int i = 0; i < kVec; ++i) acc[m][r] = fmaf(xf[i], wf[r][i], acc[m][r]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = acc[m][r];
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffff, v, offset);
      const int row = row0 + r;
      if (lane == 0 && row < n) y[static_cast<int64_t>(m) * n + row] = __float2bfloat16(v);
    }
  }
}

template <int M, int ROWS>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
  const int n = static_cast<int>(w.size(0));
  const int k = static_cast<int>(w.size(1));
  const dim3 grid((n + kWarps * ROWS - 1) / (kWarps * ROWS));
  // The current stream, so the kernel can be captured in a CUDA graph.
  gemv_bf16_kernel<M, ROWS><<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
      n,
      k);
}

template <int ROWS>
void launch_rows(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
  switch (x.size(0)) {
    case 1: launch<1, ROWS>(x, w, y); break;
    case 2: launch<2, ROWS>(x, w, y); break;
    case 3: launch<3, ROWS>(x, w, y); break;
    case 4: launch<4, ROWS>(x, w, y); break;
    case 5: launch<5, ROWS>(x, w, y); break;
    case 6: launch<6, ROWS>(x, w, y); break;
    case 7: launch<7, ROWS>(x, w, y); break;
    default: launch<8, ROWS>(x, w, y); break;
  }
}

}  // namespace

torch::Tensor gemv_bf16(torch::Tensor x, torch::Tensor w, int64_t rows_per_warp) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "gemv_bf16: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
              "gemv_bf16: bf16 tensors expected");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "gemv_bf16: shape mismatch");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "gemv_bf16: contiguous tensors expected");
  TORCH_CHECK(w.size(1) % kVec == 0, "gemv_bf16: in_features must be a multiple of 8");
  const int64_t m = x.size(0);
  TORCH_CHECK(m >= 1 && m <= 8, "gemv_bf16: 1 to 8 rows supported");
  auto y = torch::empty({m, w.size(0)}, x.options());
  switch (rows_per_warp) {
    case 1: launch_rows<1>(x, w, y); break;
    case 2: launch_rows<2>(x, w, y); break;
    case 4: launch_rows<4>(x, w, y); break;
    default: TORCH_CHECK(false, "gemv_bf16: rows_per_warp must be 1, 2 or 4");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv_bf16", &gemv_bf16, "bf16 GEMV for 1-8 rows (y = x @ w.T)",
             pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("rows_per_warp") = 2);
}
