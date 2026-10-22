// W4A16 GEMV for decode: y[M, N] = x[M, K] @ dequant(qweight, scales)[N, K]^T,
// plus the full dequantization used by large batches.
//
// qweight is int32 [N, K/8]: each word holds 8 consecutive K values of one
// row, nibble p holding k = {0, 2, 4, 6, 1, 3, 5, 7}[p], so masking
// (word >> 4i) & 0x000F000F puts k = 2i and 2i+1 in the low and high halves.
// OR-ing in the bits of 128.0 (bf16 0x4300) or 1024.0 (fp16 0x6400) turns
// each half into the float 128+q (1024+q) without an int-to-float
// conversion, and one packed subtract of 136 (1032) leaves q-8 exactly.
//
// The GEMV follows gemv_bf16.cu: a warp owns ROWS rows and each lane loads
// one 128-bit word (32 weights) per row per step, so a warp covers 1024 K
// values of each row per step. Products accumulate in fp32. A lane's 32
// weights sit in one quantization group (group_size is a multiple of 32),
// so the group scale is applied once per 32 products. x is tiny and shared
// by every warp, so it is read through the read-only cache instead of being
// staged in shared memory, and the K loop needs no barriers.
//
// One launch handles up to 8 rows of x; the op loops over 8-row chunks.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxRows = 8;

template <typename T>
struct Traits;

template <>
struct Traits<__nv_bfloat16> {
  using T2 = __nv_bfloat162;
  static constexpr uint32_t kMagic = 0x43004300u;  // 128.0 in both halves
  static constexpr uint32_t kBias = 0x43084308u;   // 136.0 in both halves
  __device__ static float2 to_float2(T2 v) { return __bfloat1622float2(v); }
  __device__ static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
};

template <>
struct Traits<__half> {
  using T2 = __half2;
  static constexpr uint32_t kMagic = 0x64006400u;  // 1024.0 in both halves
  static constexpr uint32_t kBias = 0x64086408u;   // 1032.0 in both halves
  __device__ static float2 to_float2(T2 v) { return __half22float2(v); }
  __device__ static float to_float(__half v) { return __half2float(v); }
  __device__ static __half from_float(float v) { return __float2half(v); }
};

template <typename T>
__device__ __forceinline__ typename Traits<T>::T2 as_t2(uint32_t bits) {
  return *reinterpret_cast<typename Traits<T>::T2*>(&bits);
}

// One word -> its 8 weights minus 8, in K order.
template <typename T>
__device__ __forceinline__ void dequant8(uint32_t word, float out[8]) {
  using Tr = Traits<T>;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t bits = ((word >> (4 * i)) & 0x000F000Fu) | Tr::kMagic;
    const float2 f = Tr::to_float2(__hsub2(as_t2<T>(bits), as_t2<T>(Tr::kBias)));
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <typename T>
__device__ __forceinline__ void load8(const T* p, float out[8]) {
  const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
  const uint32_t* words = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = Traits<T>::to_float2(as_t2<T>(words[i]));
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <typename T, int M, int ROWS>
__global__ void __launch_bounds__(kThreads) gemv_w4a16_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    const T* __restrict__ scales,
    T* __restrict__ y,
    int n,
    int k,
    int group_size) {
  using Tr = Traits<T>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * kWarps + warp) * ROWS;
  const int words_per_row = k / 8;
  const int groups_per_row = k / group_size;

  float acc[M][ROWS];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.f;

  for (int k0 = lane * 32; k0 < k; k0 += 32 * 32) {
    uint4 wq[ROWS];
    float scale[ROWS];
    // Issue every row's load before using any of them.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int row = row0 + r;
      if (row < n) {
        wq[r] = __ldg(reinterpret_cast<const uint4*>(
            qweight + static_cast<int64_t>(row) * words_per_row + k0 / 8));
        scale[r] = Tr::to_float(scales[static_cast<int64_t>(row) * groups_per_row + k0 / group_size]);
      } else {
        wq[r] = make_uint4(0, 0, 0, 0);
        scale[r] = 0.f;
      }
    }
    float part[M][ROWS];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < ROWS; ++r) part[m][r] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float wf[ROWS][8];
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
        dequant8<T>(reinterpret_cast<const uint32_t*>(&wq[r])[j], wf[r]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        load8<T>(x + static_cast<int64_t>(m) * k + k0 + 8 * j, xf);
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
          for (int i = 0; i < 8; ++i) part[m][r] = fmaf(xf[i], wf[r][i], part[m][r]);
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < ROWS; ++r) acc[m][r] = fmaf(part[m][r], scale[r], acc[m][r]);
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = acc[m][r];
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffff, v, offset);
      const int row = row0 + r;
      if (lane == 0 && row < n) y[static_cast<int64_t>(m) * n + row] = Tr::from_float(v);
    }
  }
}

template <typename T>
__global__ void dequantize_w4a16_kernel(
    const uint32_t* __restrict__ qweight,
    const T* __restrict__ scales,
    T* __restrict__ out,
    int64_t total_words,
    int words_per_row,
    int groups_per_row,
    int group_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_words) return;
  const int64_t row = i / words_per_row;
  const int word = static_cast<int>(i - row * words_per_row);
  const float scale = Traits<T>::to_float(scales[row * groups_per_row + word * 8 / group_size]);
  float f[8];
  dequant8<T>(qweight[i], f);
  __align__(16) T values[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) values[j] = Traits<T>::from_float(f[j] * scale);
  *reinterpret_cast<uint4*>(out + i * 8) = *reinterpret_cast<const uint4*>(values);
}

struct Problem {
  const void* x;
  const void* qweight;
  const void* scales;
  void* y;
  int n;
  int k;
  int group_size;
};

template <typename T, int M, int ROWS>
void launch(const Problem& p, cudaStream_t stream) {
  const dim3 grid((p.n + kWarps * ROWS - 1) / (kWarps * ROWS));
  gemv_w4a16_kernel<T, M, ROWS><<<grid, kThreads, 0, stream>>>(
      static_cast<const T*>(p.x), static_cast<const uint32_t*>(p.qweight),
      static_cast<const T*>(p.scales), static_cast<T*>(p.y), p.n, p.k, p.group_size);
}

template <typename T, int ROWS>
void launch_rows(const Problem& p, int m, cudaStream_t stream) {
  switch (m) {
    case 1: launch<T, 1, ROWS>(p, stream); break;
    case 2: launch<T, 2, ROWS>(p, stream); break;
    case 3: launch<T, 3, ROWS>(p, stream); break;
    case 4: launch<T, 4, ROWS>(p, stream); break;
    case 5: launch<T, 5, ROWS>(p, stream); break;
    case 6: launch<T, 6, ROWS>(p, stream); break;
    case 7: launch<T, 7, ROWS>(p, stream); break;
    default: launch<T, 8, ROWS>(p, stream); break;
  }
}

template <typename T>
void launch_type(const Problem& p, int m, int64_t rows_per_warp, cudaStream_t stream) {
  switch (rows_per_warp) {
    case 1: launch_rows<T, 1>(p, m, stream); break;
    case 2: launch_rows<T, 2>(p, m, stream); break;
    case 4: launch_rows<T, 4>(p, m, stream); break;
    default: TORCH_CHECK(false, "gemv_w4a16: rows_per_warp must be 1, 2 or 4");
  }
}

void check_weight(const torch::Tensor& qweight, const torch::Tensor& scales, int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "w4a16: CUDA tensors expected");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32 && qweight.dim() == 2 && qweight.is_contiguous(),
              "w4a16: qweight must be a contiguous int32 [N, K/8] tensor");
  TORCH_CHECK(scales.scalar_type() == torch::kBFloat16 || scales.scalar_type() == torch::kHalf,
              "w4a16: bf16 or fp16 scales expected");
  const int64_t k = qweight.size(1) * 8;
  TORCH_CHECK(group_size > 0 && group_size % 32 == 0 && k % group_size == 0,
              "w4a16: group_size must be a multiple of 32 that divides K");
  TORCH_CHECK(scales.dim() == 2 && scales.is_contiguous() && scales.size(0) == qweight.size(0) &&
                  scales.size(1) == k / group_size,
              "w4a16: scales must be a contiguous [N, K / group_size] tensor");
}

}  // namespace

torch::Tensor gemv_w4a16(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                         int64_t group_size, int64_t rows_per_warp) {
  check_weight(qweight, scales, group_size);
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "gemv_w4a16: contiguous CUDA x expected");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "gemv_w4a16: x and scales must share a dtype");
  const int64_t m = x.size(0);
  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1) * 8;
  TORCH_CHECK(x.size(1) == k, "gemv_w4a16: shape mismatch");
  auto y = torch::empty({m, n}, x.options());
  const auto stream = c10::cuda::getCurrentCUDAStream();  // so the op can be graph-captured
  const int64_t elem = x.element_size();
  for (int64_t m0 = 0; m0 < m; m0 += kMaxRows) {
    const int rows = static_cast<int>(std::min<int64_t>(kMaxRows, m - m0));
    const Problem p{static_cast<const char*>(x.data_ptr()) + m0 * k * elem, qweight.data_ptr(),
                    scales.data_ptr(), static_cast<char*>(y.data_ptr()) + m0 * n * elem,
                    static_cast<int>(n), static_cast<int>(k), static_cast<int>(group_size)};
    if (x.scalar_type() == torch::kBFloat16) {
      launch_type<__nv_bfloat16>(p, rows, rows_per_warp, stream);
    } else {
      launch_type<__half>(p, rows, rows_per_warp, stream);
    }
  }
  return y;
}

torch::Tensor dequantize_w4a16(torch::Tensor qweight, torch::Tensor scales, int64_t group_size) {
  check_weight(qweight, scales, group_size);
  const int64_t n = qweight.size(0);
  const int64_t words = qweight.size(1);
  auto out = torch::empty({n, words * 8}, scales.options());
  const int64_t total = n * words;
  if (total == 0) return out;
  const int threads = 256;
  const dim3 grid(static_cast<unsigned>((total + threads - 1) / threads));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int groups = static_cast<int>(words * 8 / group_size);
  if (scales.scalar_type() == torch::kBFloat16) {
    dequantize_w4a16_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
        static_cast<const uint32_t*>(qweight.data_ptr()),
        static_cast<const __nv_bfloat16*>(scales.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), total, static_cast<int>(words), groups,
        static_cast<int>(group_size));
  } else {
    dequantize_w4a16_kernel<__half><<<grid, threads, 0, stream>>>(
        static_cast<const uint32_t*>(qweight.data_ptr()), static_cast<const __half*>(scales.data_ptr()),
        static_cast<__half*>(out.data_ptr()), total, static_cast<int>(words), groups,
        static_cast<int>(group_size));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv_w4a16", &gemv_w4a16, "W4A16 GEMV (y = x @ dequant(qweight, scales).T)",
             pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("scales"),
             pybind11::arg("group_size") = 128, pybind11::arg("rows_per_warp") = 4);
  module.def("dequantize_w4a16", &dequantize_w4a16, "int4 weight -> scales dtype",
             pybind11::arg("qweight"), pybind11::arg("scales"), pybind11::arg("group_size") = 128);
}
