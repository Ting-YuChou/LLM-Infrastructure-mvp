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
//
// gemm_w4a16_mma is the tensor-core version for up to 64 rows (Marlin's
// idea, with a simpler layout). It reads the same qweight. mma.m16n8k16
// gives thread (g = lane / 4, c = lane % 4) the B values of column g at
// k = {2c, 2c+1, 2c+8, 2c+9} and the A values of rows g and g+8 at the same
// k. A dot product does not care how K is ordered, so inside each
// 128-wide group (one scale) the kernel maps those k to thread c's own 32
// contiguous k (32c .. 32c+31): step t (0..3) of the group uses
// 32c + 8t .. +7, which is one int32 word of qweight (pairs (0,1), (2,3),
// (4,5), (6,7) = b0 and b1 of the two k16 tiles) and one 128-bit load of
// each x row. Per group the products of q - 8 accumulate in fp32 and are
// scaled once, as in the GEMV. A block owns 32 output columns (four n8
// tiles); its four warps take interleaved groups and reduce through shared
// memory. When there are too few blocks to fill the GPU, gridDim.y splits K
// further and the splits meet in an fp32 workspace with atomicAdd. From 33
// rows on a block owns 16 columns instead, to stay within 255 registers.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <type_traits>

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

// One pair of codes (mask shift 4i) as the packed T2 bits of q - 8.
template <typename T>
__device__ __forceinline__ uint32_t dequant_pair(uint32_t word, int i) {
  using Tr = Traits<T>;
  const uint32_t bits = ((word >> (4 * i)) & 0x000F000Fu) | Tr::kMagic;
  const typename Tr::T2 v = __hsub2(as_t2<T>(bits), as_t2<T>(Tr::kBias));
  return *reinterpret_cast<const uint32_t*>(&v);
}

template <typename T>
__device__ __forceinline__ void mma16816(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

constexpr int kMmaGroup = 128;  // K per scale group and per kernel step

// MT m16 tiles (up to 16 * MT rows of x) by NT n8 tiles (8 * NT output
// columns per block). Two fp32 accumulators of MT * NT * 4 values per thread
// (the running sum and the current group's) bound MT * NT: 4 x 4 spills.
template <typename T, int MT, int NT>
__global__ void __launch_bounds__(kThreads) gemm_w4a16_mma_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    const T* __restrict__ scales,
    T* __restrict__ y,
    float* __restrict__ workspace,
    int m,
    int n,
    int k,
    int groups_per_split) {
  using Tr = Traits<T>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const int n0 = blockIdx.x * 8 * NT;
  const int groups = k / kMmaGroup;
  const int words_per_row = k / 8;
  const int group_begin = blockIdx.y * groups_per_split;
  const int group_end = min(group_begin + groups_per_split, groups);

  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mt][j][r] = 0.f;

  // B: thread (g, c) reads column g of each n8 tile, words 16s + 4c .. +3.
  auto load_b = [&](int s, uint4 (&bq)[NT]) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row = n0 + 8 * j + g;
      bq[j] = row < n ? __ldg(reinterpret_cast<const uint4*>(
                            qweight + static_cast<int64_t>(row) * words_per_row + s * 16 + 4 * c))
                      : make_uint4(0, 0, 0, 0);
    }
  };

  uint4 bq[NT], bq_next[NT];
  int s = group_begin + warp;
  if (s < group_end) load_b(s, bq);
  for (; s < group_end; s += kWarps) {
    if (s + kWarps < group_end) load_b(s + kWarps, bq_next);  // prefetch the next group
    // The accumulator rows hold output columns 2c and 2c+1 of each tile.
    float scale[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = n0 + 8 * j + 2 * c + e;
        scale[j][e] = col < n ? Tr::to_float(scales[static_cast<int64_t>(col) * groups + s]) : 0.f;
      }
    float part[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) part[mt][j][r] = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      // A: rows g and g+8 of each m16 tile, x[row][128s + 32c + 8t .. +7].
      uint4 xa[MT][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = 16 * mt + g + 8 * h;
          xa[mt][h] = row < m ? __ldg(reinterpret_cast<const uint4*>(
                                    x + static_cast<int64_t>(row) * k + s * kMmaGroup + 32 * c + 8 * t))
                              : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint32_t word = reinterpret_cast<const uint32_t*>(&bq[j])[t];
        uint32_t b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) b[i] = dequant_pair<T>(word, i);
#pragma unroll
        for (int u = 0; u < 2; ++u)  // the two k16 tiles of this step
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const uint32_t* lo = reinterpret_cast<const uint32_t*>(&xa[mt][0]);
            const uint32_t* hi = reinterpret_cast<const uint32_t*>(&xa[mt][1]);
            const uint32_t a[4] = {lo[2 * u], hi[2 * u], lo[2 * u + 1], hi[2 * u + 1]};
            mma16816<T>(part[mt][j], a, b[2 * u], b[2 * u + 1]);
          }
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][j][r] = fmaf(part[mt][j][r], scale[j][r & 1], acc[mt][j][r]);
#pragma unroll
    for (int j = 0; j < NT; ++j) bq[j] = bq_next[j];
  }

  // Sum the four warps' partial tiles; each warp then writes a quarter.
  constexpr int kValues = MT * NT * 4;
  __shared__ float reduce[kWarps][kValues][32];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) reduce[warp][(mt * NT + j) * 4 + r][lane] = acc[mt][j][r];
  __syncthreads();
  for (int v = warp; v < kValues; v += kWarps) {
    const float sum = reduce[0][v][lane] + reduce[1][v][lane] + reduce[2][v][lane] + reduce[3][v][lane];
    const int r = v % 4;
    const int j = (v / 4) % NT;
    const int mt = v / (4 * NT);
    const int row = 16 * mt + g + 8 * (r / 2);
    const int col = n0 + 8 * j + 2 * c + (r % 2);
    if (row < m && col < n) {
      if (workspace == nullptr) {
        y[static_cast<int64_t>(row) * n + col] = Tr::from_float(sum);
      } else {
        atomicAdd(workspace + static_cast<int64_t>(row) * n + col, sum);
      }
    }
  }
}

template <typename T>
__global__ void cast_workspace_kernel(const float* __restrict__ in, T* __restrict__ out, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) out[i] = Traits<T>::from_float(in[i]);
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

namespace {

// Four n8 tiles per block up to two m16 tiles, two beyond (see the kernel).
constexpr int mma_tiles_n(int mt) { return mt <= 2 ? 4 : 2; }

template <typename T, int MT>
void launch_mma(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                torch::Tensor& y, int64_t splits, cudaStream_t stream) {
  constexpr int NT = mma_tiles_n(MT);
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(qweight.size(1) * 8);
  const int groups = k / kMmaGroup;
  const int per_split = (groups + splits - 1) / splits;
  const int grid_y = (groups + per_split - 1) / per_split;
  const dim3 grid((n + 8 * NT - 1) / (8 * NT), grid_y);
  torch::Tensor workspace;
  if (grid_y > 1) workspace = torch::zeros({m, n}, x.options().dtype(torch::kFloat32));
  gemm_w4a16_mma_kernel<T, MT, NT><<<grid, kThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()), static_cast<const uint32_t*>(qweight.data_ptr()),
      static_cast<const T*>(scales.data_ptr()), static_cast<T*>(y.data_ptr()),
      grid_y > 1 ? workspace.data_ptr<float>() : nullptr, m, n, k, per_split);
  if (grid_y > 1) {
    const int64_t count = static_cast<int64_t>(m) * n;
    cast_workspace_kernel<T><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        workspace.data_ptr<float>(), static_cast<T*>(y.data_ptr()), count);
  }
}

template <typename T>
void launch_mma_rows(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                     torch::Tensor& y, int64_t splits, cudaStream_t stream) {
  switch ((x.size(0) + 15) / 16) {
    case 1: launch_mma<T, 1>(x, qweight, scales, y, splits, stream); break;
    case 2: launch_mma<T, 2>(x, qweight, scales, y, splits, stream); break;
    case 3: launch_mma<T, 3>(x, qweight, scales, y, splits, stream); break;
    default: launch_mma<T, 4>(x, qweight, scales, y, splits, stream); break;
  }
}

}  // namespace

torch::Tensor gemm_w4a16_mma(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                             int64_t group_size, int64_t splits) {
  check_weight(qweight, scales, group_size);
  TORCH_CHECK(group_size == kMmaGroup, "gemm_w4a16_mma: group_size must be 128");
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "gemm_w4a16_mma: contiguous CUDA x expected");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "gemm_w4a16_mma: x and scales must share a dtype");
  const int64_t m = x.size(0);
  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1) * 8;
  TORCH_CHECK(x.size(1) == k, "gemm_w4a16_mma: shape mismatch");
  TORCH_CHECK(m >= 1 && m <= 64, "gemm_w4a16_mma: 1 to 64 rows supported");
  if (splits <= 0) {  // enough blocks for about four per SM, at least one group per warp
    const int64_t cols = 8 * mma_tiles_n(static_cast<int>((m + 15) / 16));
    const int64_t blocks = (n + cols - 1) / cols;
    // Not ATen/cuda/CUDAContext.h: it needs the cuSPARSE/cuBLAS headers, and
    // the build only has nvcc, cudart and cccl.
    int device = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    splits = std::max<int64_t>(1, std::min<int64_t>((4 * sms + blocks - 1) / blocks, k / kMmaGroup / kWarps));
  }
  auto y = torch::empty({m, n}, x.options());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kBFloat16) {
    launch_mma_rows<__nv_bfloat16>(x, qweight, scales, y, splits, stream);
  } else {
    launch_mma_rows<__half>(x, qweight, scales, y, splits, stream);
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
  module.def("gemm_w4a16_mma", &gemm_w4a16_mma, "W4A16 tensor-core GEMM for 1-64 rows",
             pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("scales"),
             pybind11::arg("group_size") = 128, pybind11::arg("splits") = 0);
  module.def("dequantize_w4a16", &dequantize_w4a16, "int4 weight -> scales dtype",
             pybind11::arg("qweight"), pybind11::arg("scales"), pybind11::arg("group_size") = 128);
}
