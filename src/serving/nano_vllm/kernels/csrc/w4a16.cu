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
// further and the splits meet in an fp32 workspace (finish_split). From 33
// rows on a block owns 16 columns instead, to stay within 255 registers.
//
// gemm_w4a16_wide (17-64 rows) shares one shared-memory copy of x among
// warps that own different columns; see the kernel.
//
// gemm_w4a16_tiled is the Marlin-style kernel for the tiled layout
// (w4a16.py): weights pre-arranged as mma fragments, a multi-stage cp.async
// pipeline and a persistent striped grid; see the kernel.

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

// (a & b) | c in one instruction; nvcc emits an AND and an OR for the C++
// expression (SASS of gemm_w4a16_tiled, run marlin-t1d).
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));  // 0xEA = (0xF0 & 0xCC) | 0xAA
  return d;
}

// One pair of codes (mask shift 4i) as the packed T2 bits of q - 8.
template <typename T>
__device__ __forceinline__ uint32_t dequant_pair(uint32_t word, int i) {
  using Tr = Traits<T>;
  const uint32_t bits = and_or(word >> (4 * i), 0x000F000Fu, Tr::kMagic);
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

// Split-K finish. Every split adds its partial tile into the fp32 workspace;
// the last split to arrive (per-tile counter) writes T to y and re-zeroes
// the workspace and the counter, so the next call starts from zeros without
// a memset or a separate cast kernel. The workspace persists across calls
// (w4a16.py reserves it before CUDA graphs are captured).
template <typename T>
__device__ __forceinline__ void finish_split(float* workspace, int* counters, T* y, int m, int n, int col0,
                                             int cols) {
  __shared__ int last;
  __threadfence();  // this block's atomics are visible before the counter moves
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&counters[blockIdx.x], 1) == static_cast<int>(gridDim.y) - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = threadIdx.x; i < m * cols; i += blockDim.x) {
    const int row = i / cols;
    const int col = col0 + i % cols;
    if (col < n) {
      float* p = workspace + static_cast<int64_t>(row) * n + col;
      y[static_cast<int64_t>(row) * n + col] = Traits<T>::from_float(__ldcg(p));
      __stcg(p, 0.f);
    }
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;
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
    int* __restrict__ counters,
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
  if (workspace != nullptr) finish_split(workspace, counters, y, m, n, n0, 8 * NT);
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes) {
  // src_bytes = 0 zero-fills the 16 bytes (padding rows of x).
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 16-byte chunk ch (0..15) of x tile row `row` lives at this chunk of the
// shared-memory row. Thread (g, c) reads chunk 4c + t of rows g and g+8, so
// the eight lanes of a quarter warp (g in {2q, 2q+1}, c in 0..3) would hit
// two bank groups without the XOR; with it they hit eight.
__device__ __forceinline__ int swizzle(int row, int ch) { return ch ^ (((ch >> 3) & 1) << 1) ^ (row & 1); }

template <typename T>
__device__ __forceinline__ uint32_t broadcast_scale(T s) {
  using T2 = typename Traits<T>::T2;
  T2 v;
  v.x = s;
  v.y = s;
  return *reinterpret_cast<const uint32_t*>(&v);
}

template <typename T>
__device__ __forceinline__ uint32_t hmul2_bits(uint32_t a, uint32_t b) {
  const auto v = __hmul2(as_t2<T>(a), as_t2<T>(b));
  return *reinterpret_cast<const uint32_t*>(&v);
}

// The tensor-core kernel for 17-64 rows. gemm_w4a16_mma re-reads x from L2
// in every block of 32 columns, which at 64 rows is 8x the weight traffic.
// Here the four warps own different columns (4 n8 tiles each, 128 per
// block) and share one shared-memory copy of x per group, loaded with
// cp.async one group ahead. The group scale multiplies the dequantized B
// fragment (q - 8 is exact, the product is rounded once to T, as in Marlin),
// so a thread keeps a single set of fp32 accumulators. Same K permutation
// and qweight layout as gemm_w4a16_mma.
constexpr int kWideTiles = 4;                     // n8 tiles per warp
constexpr int kWideCols = kWarps * kWideTiles * 8;  // 128 columns per block

template <typename T, int MT>
__global__ void __launch_bounds__(kThreads) gemm_w4a16_wide_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    const T* __restrict__ scales,
    T* __restrict__ y,
    float* __restrict__ workspace,
    int* __restrict__ counters,
    int m,
    int n,
    int k,
    int groups_per_split) {
  using Tr = Traits<T>;
  constexpr int kRows = 16 * MT;
  __shared__ __align__(16) T xs[2][kRows * kMmaGroup];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const int col0 = blockIdx.x * kWideCols + warp * kWideTiles * 8;
  const int groups = k / kMmaGroup;
  const int words_per_row = k / 8;
  const int group_begin = blockIdx.y * groups_per_split;
  const int group_end = min(group_begin + groups_per_split, groups);

  auto stage = [&](int s, int buf) {
    for (int i = threadIdx.x; i < kRows * 16; i += kThreads) {
      const int row = i / 16;
      const int ch = i % 16;
      const bool real = row < m;
      const T* src = real ? x + static_cast<int64_t>(row) * k + s * kMmaGroup + ch * 8 : x;
      cp_async_16(&xs[buf][row * kMmaGroup + swizzle(row, ch) * 8], src, real ? 16 : 0);
    }
    cp_async_commit();
  };
  auto load_b = [&](int s, uint4 (&bq)[kWideTiles]) {
#pragma unroll
    for (int j = 0; j < kWideTiles; ++j) {
      const int row = col0 + 8 * j + g;
      bq[j] = row < n ? __ldg(reinterpret_cast<const uint4*>(
                            qweight + static_cast<int64_t>(row) * words_per_row + s * 16 + 4 * c))
                      : make_uint4(0, 0, 0, 0);
    }
  };

  float acc[MT][kWideTiles][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < kWideTiles; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mt][j][r] = 0.f;

  uint4 bq[kWideTiles], bq_next[kWideTiles];
  if (group_begin < group_end) {
    stage(group_begin, 0);
    load_b(group_begin, bq);
  }
  for (int s = group_begin, buf = 0; s < group_end; ++s, buf ^= 1) {
    if (s + 1 < group_end) {
      stage(s + 1, buf ^ 1);
      load_b(s + 1, bq_next);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    // The B fragment holds column g of each tile: one scale per tile.
    uint32_t scale[kWideTiles];
#pragma unroll
    for (int j = 0; j < kWideTiles; ++j) {
      const int col = col0 + 8 * j + g;
      scale[j] = broadcast_scale<T>(col < n ? scales[static_cast<int64_t>(col) * groups + s] : T{});
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      uint32_t b[kWideTiles][4];
#pragma unroll
      for (int j = 0; j < kWideTiles; ++j) {
        const uint32_t word = reinterpret_cast<const uint32_t*>(&bq[j])[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) b[j][i] = hmul2_bits<T>(dequant_pair<T>(word, i), scale[j]);
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = 16 * mt + g;
        const uint4 lo4 = *reinterpret_cast<const uint4*>(&xs[buf][row * kMmaGroup + swizzle(row, 4 * c + t) * 8]);
        const uint4 hi4 =
            *reinterpret_cast<const uint4*>(&xs[buf][(row + 8) * kMmaGroup + swizzle(row + 8, 4 * c + t) * 8]);
        const uint32_t* lo = reinterpret_cast<const uint32_t*>(&lo4);
        const uint32_t* hi = reinterpret_cast<const uint32_t*>(&hi4);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const uint32_t a[4] = {lo[2 * u], hi[2 * u], lo[2 * u + 1], hi[2 * u + 1]};
#pragma unroll
          for (int j = 0; j < kWideTiles; ++j) mma16816<T>(acc[mt][j], a, b[j][2 * u], b[j][2 * u + 1]);
        }
      }
    }
    __syncthreads();  // everyone is done with xs[buf] before it is refilled
#pragma unroll
    for (int j = 0; j < kWideTiles; ++j) bq[j] = bq_next[j];
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < kWideTiles; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = 16 * mt + g + 8 * (r / 2);
        const int col = col0 + 8 * j + 2 * c + (r % 2);
        if (row < m && col < n) {
          if (workspace == nullptr) {
            y[static_cast<int64_t>(row) * n + col] = Tr::from_float(acc[mt][j][r]);
          } else {
            atomicAdd(workspace + static_cast<int64_t>(row) * n + col, acc[mt][j][r]);
          }
        }
      }
  if (workspace != nullptr) finish_split(workspace, counters, y, m, n, blockIdx.x * kWideCols, kWideCols);
}

// ---------------------------------------------------------------------------
// gemm_w4a16_tiled: y[M, N] = x[M, K] @ dequant(W)^T for up to 64 rows, W in
// the tiled layout: int32 [K/16, N/64, 32, 4], each 512-byte unit the B
// operand of a 16 x 64 (k x n) tile with lane 4g + c's fragments in one
// 16-byte word, and scales [K/128, N/64, 8, 8] ([.., g, j] = column 8j + g).
//
// A block of 8 warps owns TILE_N output columns (a slice) at a time and
// walks K in TILE_K steps. Each pipeline stage holds the x tile (cp.async,
// rows swizzled by chunk ^ row % 8, read with ldmatrix), the weight tile
// (TILE_K/16 k16 rows of TILE_N/64 units, one shared load per lane per k16
// row) and the slice's scales; kTiledStages stages keep a few tiles of
// weights in flight per SM, which the row-layout kernels never managed.
// The warps of one n64 column split the stage's k16 rows; dequantized
// weights are multiplied by their scale in T (rounded once) before the mma.
//
// The grid is persistent. Tiles are numbered slice * (K / TILE_K) + k tile
// and block b takes tiles [b * iters, (b + 1) * iters), so every block gets
// the same work for any shape (Marlin's striped partition). The loader
// addresses tiles by that number, so the pipeline runs on across slice
// boundaries instead of draining. At the end of its part of a slice a block
// sums its warps into an fp32 tile in shared memory (outside the pipeline
// stages). A slice with one block writes y from it; otherwise every block of
// the slice stores its partial in slot (block + slice) of a scratch buffer
// and bumps the slice's counter, and the last to arrive adds the slots in
// block order and writes y. No block waits for another and the sum does not
// depend on timing. w4a16.py's tiled_schedule mirrors this arithmetic.
constexpr int kTiledThreads = 256;
constexpr int kTiledWarps = kTiledThreads / 32;
constexpr int kTiledStages = 4;

// The trace build (kTrace) records, per block, where its time goes: thread 0
// keeps clock64() stamps in shared memory and writes them out at the end, so
// no global store waits on the slot release. A row of kTraceEvents int64:
// globaltimer and clock at entry, the clock once the first tile has landed,
// the tile count, four stamps for each of the first kTraceFinishes slice
// ends (start, warps summed, counter bumped or y written, done), a flag word
// (bit f: last to arrive at finish f, bit 8 + f: sole owner, y written
// directly, bit 16 + f: last and saw so before its slot store), the clock and
// globaltimer at exit, and the exit order. Row
// `blocks` holds the exit ticket counter. w4a16.py's TILED_TRACE_* mirror this.
constexpr int kTraceEvents = 24;
constexpr int kTraceFinishes = 4;
constexpr int kTraceGtEntry = 0, kTraceClkEntry = 1, kTraceClkFirst = 2, kTraceTiles = 3, kTraceFinish0 = 4;
constexpr int kTraceFlags = 20, kTraceClkExit = 21, kTraceGtExit = 22, kTraceTicket = 23;

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

template <int MT, int TILE_K, int TILE_N>
struct TiledShape {
  static constexpr int kCols = TILE_N / 64;           // n64 columns per slice
  static constexpr int kKRows = TILE_K / 16;          // k16 rows per stage
  static constexpr int kWarpsPerCol = kTiledWarps / kCols;
  static constexpr int kRowsPerWarp = kKRows / kWarpsPerCol;
  static constexpr int kRows = 16 * MT;               // rows of x, padded
  static constexpr int kAChunks = TILE_K / 8;         // 16-byte chunks per row of the x tile
  static constexpr int kABytes = kRows * TILE_K * 2;
  static constexpr int kBWords = kKRows * kCols * 32;  // 16-byte words of weights per stage
  static constexpr int kGroups = TILE_K >= 128 ? TILE_K / 128 : 1;  // scale groups per stage
  static constexpr int kSWords = kGroups * kCols * 8;               // 16-byte words of scales per stage
  static constexpr int kStageBytes = kABytes + 16 * (kBWords + kSWords);
  static constexpr int kTileStride = TILE_N + 8;  // fp32 slice tile row; the pad spreads banks
  static constexpr int kTileOffset = kTiledStages * kStageBytes;
  // The fp32 slice tile, and when they fit (kParallel) one per warp of a
  // column but the last: those warps store their sums at once and the last
  // adds them and its own into the first. Then the last-arrival flag and the
  // trace build's stamps.
  static constexpr int kPartFloats = kRows * kTileStride;
  static constexpr int kMaxSmem = 163 * 1024;  // A100's opt-in limit per block
  static constexpr bool kParallel =
      kWarpsPerCol >= 3 && kTileOffset + (kWarpsPerCol - 1) * kPartFloats * 4 + 16 + kTraceEvents * 8 <= kMaxSmem;
  static constexpr int kFlagOffset = kTileOffset + (kParallel ? kWarpsPerCol - 1 : 1) * kPartFloats * 4;
  static constexpr int kTraceOffset = kFlagOffset + 16;
  static constexpr int kSmemBytes = kTraceOffset + kTraceEvents * 8;
  // Otherwise, at the last slice end, one fp32 tile per warp laid over the drained stages.
  static constexpr bool kSpreadFits = !kParallel && kWarpsPerCol * kPartFloats * 4 <= kTileOffset;
  static_assert(kStageBytes % 128 == 0, "stages start on 128-byte lines");
  static_assert(kSmemBytes <= kMaxSmem, "shared memory per block");
  static_assert(kKRows % kWarpsPerCol == 0 && kRowsPerWarp >= 1, "the warps of a column split the k16 rows");
  static_assert(kAChunks >= 8, "the x swizzle spans 8 chunks");
  static_assert(kBWords % kTiledThreads == 0 && kSWords <= kTiledThreads, "whole cp.async rounds");
  static_assert(TILE_K < 128 || 8 % kRowsPerWarp == 0, "a warp's k16 rows share one scale group");
};

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
  __align__(16) T out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = Traits<T>::from_float(v[i]);
  return *reinterpret_cast<const uint4*>(out);
}

__device__ __forceinline__ void add8(float (&sum)[8], float4 lo, float4 hi) {
  sum[0] += lo.x; sum[1] += lo.y; sum[2] += lo.z; sum[3] += lo.w;
  sum[4] += hi.x; sum[5] += hi.y; sum[6] += hi.z; sum[7] += hi.w;
}

template <typename T, int MT, int TILE_K, int TILE_N, bool kTrace = false>
__global__ void __launch_bounds__(kTiledThreads, 1) gemm_w4a16_tiled_kernel(
    const T* __restrict__ x,
    const uint4* __restrict__ qweight,
    const uint4* __restrict__ scales,
    T* __restrict__ y,
    float* __restrict__ slots,
    int* __restrict__ counters,
    int m,
    int n,
    int k,
    int iters,
    long long* __restrict__ trace) {
  using Sh = TiledShape<MT, TILE_K, TILE_N>;
  constexpr int R = Sh::kRowsPerWarp;
  // Everything lives in the dynamic allocation, which then starts at shared
  // address 0: a static __shared__ variable would shift it by 16 bytes, and
  // with the stages off 128-byte lines every cp.async of a warp moved each
  // L2 sector twice (ncu, run down-d1).
  extern __shared__ __align__(128) unsigned char smem[];
  int& last_block = *reinterpret_cast<int*>(smem + Sh::kFlagOffset);
  long long* trace_sh = reinterpret_cast<long long*>(smem + Sh::kTraceOffset);
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const int col = warp % Sh::kCols;  // this warp's n64 column of the slice
  const int row0 = (warp / Sh::kCols) * R;  // its first k16 row of each stage
  const int k_tiles = k / TILE_K;
  const int units = n / 64;  // weight units per k16 row
  const int total = k_tiles * (n / TILE_N);
  const int begin = blockIdx.x * iters;
  const int end = min(begin + iters, total);
  if (begin >= end) return;
  auto mark = [&](int event) {
    if constexpr (kTrace) {
      if (tid == 0) trace_sh[event] = clock64();
    }
  };
  if constexpr (kTrace) {
    if (tid == 0) {
      trace_sh[kTraceGtEntry] = globaltimer();
      for (int e = kTraceClkEntry; e < kTraceEvents; ++e) trace_sh[e] = 0;
      trace_sh[kTraceClkEntry] = clock64();
      trace_sh[kTraceTiles] = end - begin;
    }
  }
  float* tile = reinterpret_cast<float*>(smem + Sh::kTileOffset);
  auto stage = [&](int s) { return smem + s * Sh::kStageBytes; };

  // Rows m .. 16 MT - 1 of x are never loaded: zero them once in every stage.
  {
    const int pad = (Sh::kRows - m) * Sh::kAChunks;
    for (int i = tid; i < kTiledStages * pad; i += kTiledThreads)
      reinterpret_cast<uint4*>(stage(i / pad))[m * Sh::kAChunks + i % pad] = make_uint4(0, 0, 0, 0);
  }

  // This thread's weight words of a stage sit at fixed offsets from the
  // tile's first unit: computed once, then each load adds the tile's base.
  int b_off[Sh::kBWords / kTiledThreads];
#pragma unroll
  for (int j = 0; j < Sh::kBWords / kTiledThreads; ++j) {
    const int i = tid + j * kTiledThreads;
    const int unit = i / 32;  // k16 row unit / kCols, column unit % kCols
    b_off[j] = ((unit / Sh::kCols) * units + unit % Sh::kCols) * 32 + i % 32;
  }
  auto load = [&](int kr, int slice, int s) {
    T* a = reinterpret_cast<T*>(stage(s));
    const T* xs = x + kr * TILE_K;
    for (int i = tid; i < m * Sh::kAChunks; i += kTiledThreads) {
      const int row = i / Sh::kAChunks;
      const int ch = i % Sh::kAChunks;
      cp_async_16(a + row * TILE_K + (ch ^ (row % 8)) * 8, xs + static_cast<int64_t>(row) * k + ch * 8, 16);
    }
    uint4* b = reinterpret_cast<uint4*>(stage(s) + Sh::kABytes);
    const uint4* bs = qweight + (static_cast<int64_t>(kr) * Sh::kKRows * units + slice * Sh::kCols) * 32;
#pragma unroll
    for (int j = 0; j < Sh::kBWords / kTiledThreads; ++j) cp_async_16(b + tid + j * kTiledThreads, bs + b_off[j], 16);
    if (tid < Sh::kSWords) {
      const int group = kr * TILE_K / 128 + tid / (Sh::kCols * 8);
      cp_async_16(b + Sh::kBWords + tid,
                  scales + (static_cast<int64_t>(group) * units + slice * Sh::kCols) * 8 + tid % (Sh::kCols * 8), 16);
    }
  };
  // The load stream runs kTiledStages - 1 tiles ahead of the compute stream.
  int ld_t = begin;
  int ld_slice = begin / k_tiles;
  int ld_kr = begin - ld_slice * k_tiles;
  auto load_next = [&](int s) {
    if (ld_t < end) load(ld_kr, ld_slice, s);
    cp_async_commit();  // an empty group past the end keeps the wait counts uniform
    ++ld_t;
    if (++ld_kr == k_tiles) {
      ld_kr = 0;
      ++ld_slice;
    }
  };

  float acc[MT][8][4];
  auto zero_acc = [&]() {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][j][r] = 0.f;
  };

  // The fragments of one k16 row: x by ldmatrix (lane l addresses row l % 16
  // of the m16 tile and k8 half l / 16, giving a0..a3 = rows g / g + 8 at k 2c
  // and 2c + 8, the mma A layout), the weights by one 16-byte shared load.
  auto fetch = [&](int s, int r, uint32_t (&af)[MT][4], uint4& bw) {
    const T* a = reinterpret_cast<const T*>(stage(s));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldmatrix_x4(af[mt], a + (16 * mt + lane % 16) * TILE_K + ((2 * r + lane / 16) ^ (lane % 8)) * 8);
    bw = reinterpret_cast<const uint4*>(stage(s) + Sh::kABytes)[(r * Sh::kCols + col) * 32 + lane];
  };
  // The scales of this lane's eight columns (n8 tiles 0..7, column g), for
  // the group holding the warp's k16 rows; kept packed (4 registers) and
  // broadcast to both halves where they are used.
  auto fetch_scales = [&](int s) {
    const int group = TILE_K >= 128 ? row0 / 8 : 0;
    return reinterpret_cast<const uint4*>(stage(s) + Sh::kABytes)[Sh::kBWords + (group * Sh::kCols + col) * 8 + g];
  };
  auto mma_row = [&](const uint32_t (&af)[MT][4], const uint4& bw, const uint4& packed_scales) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&bw);
    const T* s8 = reinterpret_cast<const T*>(&packed_scales);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t sc[2] = {broadcast_scale<T>(s8[2 * i]), broadcast_scale<T>(s8[2 * i + 1])};
      uint32_t bf[4];  // b0, b1 of n8 tile 2i, then of tile 2i + 1
#pragma unroll
      for (int p = 0; p < 4; ++p) bf[p] = hmul2_bits<T>(dequant_pair<T>(words[i], p), sc[p / 2]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        mma16816<T>(acc[mt][2 * i], af[mt], bf[0], bf[1]);
        mma16816<T>(acc[mt][2 * i + 1], af[mt], bf[2], bf[3]);
      }
    }
  };

  // Every thread runs this (it has barriers). The block's sum of the slice
  // is ((w0 + w1) + w2) + ... over the column's warps w in order, whichever
  // way it is formed, so every path gives the same bits.
  auto finish_slice = [&](int slice, bool final_end) {
    const int f = slice - begin / k_tiles;  // this block's f-th slice end (trace only)
    auto mark_finish = [&](int step) {
      if (f < kTraceFinishes) mark(kTraceFinish0 + 4 * f + step);
    };
    mark_finish(0);
    const int first = slice * k_tiles;
    const int lo = first / iters;  // the blocks whose stripes meet this slice
    const int hi = (first + k_tiles - 1) / iters;
    // A shared slice's counter, read now so the load overlaps the warps'
    // sums: when every other block has already arrived, this one is the last
    // and needs neither its own slot nor the atomic.
    int seen = -1;
    if (lo != hi && tid == 0) seen = ld_acquire(counters + slice);
    // acc[mt][j] holds rows g, g + 8 of m16 tile mt at columns 2c, 2c + 1 of
    // n8 tile j. With room for a tile per warp but the last (kParallel), all
    // warps of a column but the last store theirs at once, and the last adds
    // them in order and its own into the first. Else, at the block's last slice end,
    // the drained stages hold one copy per warp (spread) and readers add them
    // in warp order. Else the column's warps add into one tile in turn.
    const bool spread = Sh::kSpreadFits && final_end;
    float* parts = reinterpret_cast<float*>(smem);
    const int kw = warp / Sh::kCols;
    auto store_acc = [&](float* dst, bool add) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float2* p = reinterpret_cast<float2*>(dst + (16 * mt + g + 8 * h) * Sh::kTileStride + col * 64 + 8 * j + 2 * c);
            float2 v = make_float2(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
            if (add) {
              const float2 o = *p;
              v = make_float2(o.x + v.x, o.y + v.y);
            }
            *p = v;
          }
    };
    auto decide = [&]() {  // before the barrier that ends the sums
      if (tid == 0) last_block = seen == hi - lo ? 2 : 0;  // 2: last already, skip the slot
    };
    if constexpr (Sh::kParallel) {
      if (kw < Sh::kWarpsPerCol - 1) store_acc(tile + kw * Sh::kPartFloats, false);
      __syncthreads();
      if (kw == Sh::kWarpsPerCol - 1) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int at = (16 * mt + g + 8 * h) * Sh::kTileStride + col * 64 + 8 * j + 2 * c;
              float2 v = *reinterpret_cast<const float2*>(tile + at);
#pragma unroll
              for (int w = 1; w < Sh::kWarpsPerCol - 1; ++w) {
                const float2 o = *reinterpret_cast<const float2*>(tile + w * Sh::kPartFloats + at);
                v = make_float2(v.x + o.x, v.y + o.y);
              }
              v = make_float2(v.x + acc[mt][j][2 * h], v.y + acc[mt][j][2 * h + 1]);
              *reinterpret_cast<float2*>(tile + at) = v;
            }
      }
      decide();
      __syncthreads();
    } else if (spread) {
      __syncthreads();  // every warp is done with the last stage
      store_acc(parts + kw * Sh::kPartFloats, false);
      decide();
      __syncthreads();
    } else {
#pragma unroll
      for (int w = 0; w < Sh::kWarpsPerCol; ++w) {
        if (kw == w) store_acc(tile, w > 0);
        if (w == Sh::kWarpsPerCol - 1) decide();
        __syncthreads();
      }
    }
    mark_finish(1);
    zero_acc();
    // The block's sum at 4 columns from 4 * v4 of a row.
    auto block_sum4 = [&](int row, int v4) {
      if (!spread) return *reinterpret_cast<const float4*>(tile + row * Sh::kTileStride + 4 * v4);
      float4 b = *reinterpret_cast<const float4*>(parts + row * Sh::kTileStride + 4 * v4);
#pragma unroll
      for (int w = 1; w < Sh::kWarpsPerCol; ++w) {
        const float4 o = *reinterpret_cast<const float4*>(parts + w * Sh::kPartFloats + row * Sh::kTileStride + 4 * v4);
        b = make_float4(b.x + o.x, b.y + o.y, b.z + o.z, b.w + o.w);
      }
      return b;
    };
    T* ys = y + slice * TILE_N;
    constexpr int kVecs = TILE_N / 8;  // 8 outputs per thread and step
    if (lo == hi) {
      for (int i = tid; i < m * kVecs; i += kTiledThreads) {
        const int row = i / kVecs;
        const int v = i % kVecs;
        float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        add8(sum, block_sum4(row, 2 * v), block_sum4(row, 2 * v + 1));
        *reinterpret_cast<uint4*>(ys + static_cast<int64_t>(row) * n + 8 * v) = pack8<T>(sum);
      }
      mark_finish(2);
      __syncthreads();  // the tile is rewritten at the next slice
      if constexpr (kTrace) {
        if (tid == 0 && f < kTraceFinishes) trace_sh[kTraceFlags] |= 1ll << (8 + f);
      }
      mark_finish(3);
      return;
    }
    const int64_t slot_floats = Sh::kRows * TILE_N;
    const bool early = last_block == 2;
    if (!early) {
      float* mine = slots + (blockIdx.x + slice) * slot_floats;
      for (int i = tid; i < m * TILE_N / 4; i += kTiledThreads) {
        const int row = i / (TILE_N / 4);
        const int v = i % (TILE_N / 4);
        __stcg(reinterpret_cast<float4*>(mine + row * TILE_N) + v, block_sum4(row, v));
      }
      // Release and acquire in one atomic: the barrier orders every thread's
      // slot stores before thread 0's release (cumulative, as in CUTLASS's
      // semaphore and Marlin's locks), and the barrier after it orders the
      // last block's L2 reads (ld.cg) after its acquire.
      __syncthreads();
      if (tid == 0) last_block = atom_add_acq_rel(counters + slice, 1) == hi - lo;
      __syncthreads();
    }
    mark_finish(2);
    if constexpr (kTrace) {
      if (tid == 0 && last_block && f < kTraceFinishes) trace_sh[kTraceFlags] |= (1ll << f) | (early ? 1ll << (16 + f) : 0);
    }
    if (last_block) {
      // Slots read before any is added, so their L2 latencies overlap: a few
      // at a time while the next tile's fragments fill the registers (fewer
      // as MT grows; MT 4 would spill), more at the block's last slice end.
      auto sum_slots = [&](auto batch) {
        constexpr int kBatch = decltype(batch)::value;
        for (int i = tid; i < m * kVecs; i += kTiledThreads) {
          const int row = i / kVecs;
          const int v = i % kVecs;
          float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          for (int blk0 = lo; blk0 <= hi; blk0 += kBatch) {  // block order, whoever arrived last
            float4 part[kBatch][2];
#pragma unroll
            for (int q = 0; q < kBatch; ++q) {
              const int blk = blk0 + q;
              if (blk > hi) break;
              if (blk == static_cast<int>(blockIdx.x)) {
                part[q][0] = block_sum4(row, 2 * v);
                part[q][1] = block_sum4(row, 2 * v + 1);
              } else {
                const float4* src =
                    reinterpret_cast<const float4*>(slots + (blk + slice) * slot_floats + row * TILE_N + 8 * v);
                part[q][0] = __ldcg(src);
                part[q][1] = __ldcg(src + 1);
              }
            }
#pragma unroll
            for (int q = 0; q < kBatch; ++q)
              if (blk0 + q <= hi) add8(sum, part[q][0], part[q][1]);
          }
          *reinterpret_cast<uint4*>(ys + static_cast<int64_t>(row) * n + 8 * v) = pack8<T>(sum);
        }
      };
      if (final_end) {
        sum_slots(std::integral_constant<int, MT < 4 ? 4 : (TILE_N < 256 ? 2 : 1)>{});
      } else {
        sum_slots(std::integral_constant<int, MT >= 4 ? 1 : (MT == 3 ? 2 : 4)>{});
      }
      if (tid == 0) counters[slice] = 0;  // ready for the next call
    }
    __syncthreads();
    mark_finish(3);
  };

  // Main loop. A warp holds the fragments of its next k16 row while it
  // multiplies the current one (Marlin's register double buffer); before its
  // last row of a stage it refills the oldest stage, waits for the next one
  // and fetches that stage's first row, so stage changes leave no bubble.
  zero_acc();
#pragma unroll
  for (int s = 0; s < kTiledStages - 1; ++s) load_next(s);
  cp_async_wait<kTiledStages - 2>();
  __syncthreads();
  mark(kTraceClkFirst);
  uint32_t a_cur[MT][4], a_nxt[MT][4];
  uint4 b_cur, b_nxt;
  fetch(0, row0, a_cur, b_cur);
  uint4 sc_cur = fetch_scales(0), sc_nxt = sc_cur;
  int s = 0;
  int slice = begin / k_tiles;
  int kr = begin - slice * k_tiles;
  for (int t = begin; t < end; ++t) {
    const int s_next = s + 1 == kTiledStages ? 0 : s + 1;
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
      if (rr + 1 < R) {
        fetch(s, row0 + rr + 1, a_nxt, b_nxt);
      } else {
        load_next(s == 0 ? kTiledStages - 1 : s - 1);  // the stage of tile t - 1, read by everyone
        cp_async_wait<kTiledStages - 2>();             // tile t + 1 has landed
        __syncthreads();
        if (t + 1 < end) {
          fetch(s_next, row0, a_nxt, b_nxt);
          sc_nxt = fetch_scales(s_next);
        }
      }
      mma_row(a_cur, b_cur, sc_cur);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int q = 0; q < 4; ++q) a_cur[mt][q] = a_nxt[mt][q];
      b_cur = b_nxt;
    }
    sc_cur = sc_nxt;
    s = s_next;
    if (kr == k_tiles - 1 || t == end - 1) finish_slice(slice, t == end - 1);
    if (++kr == k_tiles) {
      kr = 0;
      ++slice;
    }
  }
  cp_async_wait<0>();
  if constexpr (kTrace) {
    if (tid == 0) {
      trace_sh[kTraceClkExit] = clock64();
      trace_sh[kTraceGtExit] = globaltimer();
      trace_sh[kTraceTicket] =
          static_cast<long long>(atomicAdd(reinterpret_cast<unsigned long long*>(trace + gridDim.x * kTraceEvents), 1ull));
      for (int e = 0; e < kTraceEvents; ++e) trace[blockIdx.x * kTraceEvents + e] = trace_sh[e];
    }
  }
}

// The tiled weight back to [N, K] in the scales' dtype, for batches too big
// for the GEMM kernels. A block takes one 128 x 64 (k x n) piece: each lane
// dequantizes its 32 codes into a shared tile, which is then written out in
// 16-byte row stores. The arithmetic is dequantize_w4a16_kernel's, so both
// layouts give bit-identical weights.
template <typename T>
__global__ void __launch_bounds__(256) dequantize_w4a16_tiled_kernel(
    const uint4* __restrict__ qweight, const uint4* __restrict__ scales, T* __restrict__ out, int n, int k) {
  using Tr = Traits<T>;
  constexpr int kStride = 128 + 8;  // T per row of the piece
  __shared__ __align__(16) T piece[64 * kStride];
  const int group = blockIdx.x;  // 128 values of K
  const int t = blockIdx.y;      // 64 values of N
  const int units = n / 64;
  const int r = threadIdx.x / 32;  // k16 row of the group
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const uint4 packed = qweight[(static_cast<int64_t>(group * 8 + r) * units + t) * 32 + lane];
  const uint4 packed_scales = scales[(static_cast<int64_t>(group) * units + t) * 8 + g];
  const uint32_t* words = reinterpret_cast<const uint32_t*>(&packed);
  const T* s8 = reinterpret_cast<const T*>(&packed_scales);
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int j = 2 * i + p / 2;  // n8 tile
      const float2 f = Tr::to_float2(as_t2<T>(dequant_pair<T>(words[i], p)));
      const float scale = Tr::to_float(s8[j]);
      T* dst = piece + (8 * j + g) * kStride + 16 * r + 8 * (p % 2) + 2 * c;
      dst[0] = Tr::from_float(f.x * scale);
      dst[1] = Tr::from_float(f.y * scale);
    }
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * 16; i += 256) {
    const int row = i / 16;
    const int v = i % 16;
    *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t * 64 + row) * k + group * 128 + 8 * v) =
        *reinterpret_cast<const uint4*>(piece + row * kStride + 8 * v);
  }
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

// K split across blocks (gridDim.y) and the persistent split-K buffers.
struct Split {
  int64_t splits;
  float* workspace_ptr;  // >= m * n floats, all zero between calls
  int* counters;         // >= gridDim.x ints, all zero between calls
  int per_split(int k) const { return static_cast<int>((k / kMmaGroup + splits - 1) / splits); }
  int grid_y(int k) const { return (k / kMmaGroup + per_split(k) - 1) / per_split(k); }
  float* workspace(int k) const { return grid_y(k) > 1 ? workspace_ptr : nullptr; }
};

// Four n8 tiles per block up to two m16 tiles, two beyond (see the kernel).
constexpr int mma_tiles_n(int mt) { return mt <= 2 ? 4 : 2; }

template <typename T, int MT>
void launch_mma(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                torch::Tensor& y, const Split& split, cudaStream_t stream) {
  constexpr int NT = mma_tiles_n(MT);
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(qweight.size(1) * 8);
  const dim3 grid((n + 8 * NT - 1) / (8 * NT), split.grid_y(k));
  gemm_w4a16_mma_kernel<T, MT, NT><<<grid, kThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()), static_cast<const uint32_t*>(qweight.data_ptr()),
      static_cast<const T*>(scales.data_ptr()), static_cast<T*>(y.data_ptr()),
      split.workspace(k), split.counters, m, n, k, split.per_split(k));
}

template <typename T>
void launch_mma_rows(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                     torch::Tensor& y, const Split& split, cudaStream_t stream) {
  switch ((x.size(0) + 15) / 16) {
    case 1: launch_mma<T, 1>(x, qweight, scales, y, split, stream); break;
    case 2: launch_mma<T, 2>(x, qweight, scales, y, split, stream); break;
    case 3: launch_mma<T, 3>(x, qweight, scales, y, split, stream); break;
    default: launch_mma<T, 4>(x, qweight, scales, y, split, stream); break;
  }
}

template <typename T, int MT>
void launch_wide(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                 torch::Tensor& y, const Split& split, cudaStream_t stream) {
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(qweight.size(1) * 8);
  const dim3 grid((n + kWideCols - 1) / kWideCols, split.grid_y(k));
  gemm_w4a16_wide_kernel<T, MT><<<grid, kThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()), static_cast<const uint32_t*>(qweight.data_ptr()),
      static_cast<const T*>(scales.data_ptr()), static_cast<T*>(y.data_ptr()),
      split.workspace(k), split.counters, m, n, k, split.per_split(k));
}

template <typename T>
void launch_wide_rows(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                      torch::Tensor& y, const Split& split, cudaStream_t stream) {
  switch ((x.size(0) + 15) / 16) {
    case 1: launch_wide<T, 1>(x, qweight, scales, y, split, stream); break;
    case 2: launch_wide<T, 2>(x, qweight, scales, y, split, stream); break;
    case 3: launch_wide<T, 3>(x, qweight, scales, y, split, stream); break;
    default: launch_wide<T, 4>(x, qweight, scales, y, split, stream); break;
  }
}

int sm_count() {
  // Not ATen/cuda/CUDAContext.h: it needs the cuSPARSE/cuBLAS headers, and
  // the build only has nvcc, cudart and cccl.
  int device = 0, sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return sms;
}

Split make_split(int64_t splits, const torch::Tensor& workspace, const torch::Tensor& counters, int64_t m,
                 int64_t n, int64_t cols, const char* name) {
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kFloat32 && workspace.is_contiguous() &&
                  workspace.numel() >= m * n,
              name, ": workspace must be a zeroed, contiguous fp32 CUDA tensor of at least m * n elements");
  TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 && counters.is_contiguous() &&
                  counters.numel() >= (n + cols - 1) / cols,
              name, ": counters must be a zeroed int32 CUDA tensor with one entry per column block");
  return Split{splits, workspace.data_ptr<float>(), counters.data_ptr<int>()};
}

void check_mma_args(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                    int64_t group_size, const char* name) {
  check_weight(qweight, scales, group_size);
  TORCH_CHECK(group_size == kMmaGroup, name, ": group_size must be 128");
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), name, ": contiguous CUDA x expected");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), name, ": x and scales must share a dtype");
  TORCH_CHECK(x.size(1) == qweight.size(1) * 8, name, ": shape mismatch");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64, name, ": 1 to 64 rows supported");
}

template <typename T, int MT, int TILE_K, int TILE_N, bool kTrace>
auto tiled_kernel() {
  using Sh = TiledShape<MT, TILE_K, TILE_N>;
  const auto kernel = gemm_w4a16_tiled_kernel<T, MT, TILE_K, TILE_N, kTrace>;
  static bool configured = false;  // once per instantiation; the first call runs before any capture
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Sh::kSmemBytes));
    configured = true;
  }
  return kernel;
}

template <typename T, int MT, int TILE_K, int TILE_N>
void launch_tiled(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                  const torch::Tensor& counters, torch::Tensor& out, int grid, const c10::optional<torch::Tensor>& trace,
                  cudaStream_t stream) {
  using Sh = TiledShape<MT, TILE_K, TILE_N>;
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(out.size(1));
  const int slices = n / TILE_N;
  const int total = (k / TILE_K) * slices;
  const int iters = (total + grid - 1) / grid;
  const int blocks = (total + iters - 1) / iters;
  // Partial sums of slices shared by several blocks; slot (block + slice).
  torch::Tensor slots = torch::empty({static_cast<int64_t>(blocks + slices) * Sh::kRows * TILE_N},
                                     x.options().dtype(torch::kFloat32));
  const auto args = [&](auto kernel, long long* trace_ptr) {
    kernel<<<blocks, kTiledThreads, Sh::kSmemBytes, stream>>>(
        static_cast<const T*>(x.data_ptr()), static_cast<const uint4*>(qweight.data_ptr()),
        static_cast<const uint4*>(scales.data_ptr()), static_cast<T*>(out.data_ptr()), slots.data_ptr<float>(),
        counters.data_ptr<int>(), m, n, k, iters, trace_ptr);
  };
  if (!trace.has_value()) {
    args(tiled_kernel<T, MT, TILE_K, TILE_N, false>(), nullptr);
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {  // the trace build is bf16 only
    const torch::Tensor& t = *trace;
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt64 && t.is_contiguous() &&
                    t.numel() >= static_cast<int64_t>(blocks + 1) * kTraceEvents,
                "gemm_w4a16_tiled: trace must be a contiguous int64 CUDA tensor of (blocks + 1) x ", kTraceEvents,
                " (blocks = ", blocks, ")");
    C10_CUDA_CHECK(cudaMemsetAsync(t.data_ptr(), 0, t.numel() * sizeof(long long), stream));
    args(tiled_kernel<T, MT, TILE_K, TILE_N, true>(), static_cast<long long*>(t.data_ptr()));
  } else {
    TORCH_CHECK(false, "gemm_w4a16_tiled: the trace build is bf16 only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The (MT, TILE_K, TILE_N) built; w4a16.py's _TILED_TILES_BY_MT lists the same.
template <typename T>
void launch_tiled_config(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                         const torch::Tensor& counters, torch::Tensor& out, int64_t tile_k, int64_t tile_n,
                         int grid, const c10::optional<torch::Tensor>& trace, cudaStream_t stream) {
  const int mt = static_cast<int>((x.size(0) + 15) / 16);
  if (tile_k == 128 && tile_n == 128 && mt == 1) return launch_tiled<T, 1, 128, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 128 && tile_n == 128 && mt == 2) return launch_tiled<T, 2, 128, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 256 && tile_n == 128 && mt == 1) return launch_tiled<T, 1, 256, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 256 && tile_n == 128 && mt == 2) return launch_tiled<T, 2, 256, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 256 && mt == 1) return launch_tiled<T, 1, 64, 256>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 256 && mt == 2) return launch_tiled<T, 2, 64, 256>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 256 && mt == 3) return launch_tiled<T, 3, 64, 256>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 256 && mt == 4) return launch_tiled<T, 4, 64, 256>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 128 && mt == 3) return launch_tiled<T, 3, 64, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  if (tile_k == 64 && tile_n == 128 && mt == 4) return launch_tiled<T, 4, 64, 128>(x, qweight, scales, counters, out, grid, trace, stream);
  TORCH_CHECK(false, "gemm_w4a16_tiled: no kernel built for ", tile_k, "x", tile_n, " tiles and ", x.size(0), " rows");
}

void check_tiled_weight(const torch::Tensor& qweight, const torch::Tensor& scales) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "w4a16 tiled: CUDA tensors expected");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32 && qweight.dim() == 4 && qweight.is_contiguous() &&
                  qweight.size(2) == 32 && qweight.size(3) == 4 && qweight.size(0) % 8 == 0,
              "w4a16 tiled: qweight must be a contiguous int32 [K/16, N/64, 32, 4] tensor, K a multiple of 128");
  TORCH_CHECK(scales.scalar_type() == torch::kBFloat16 || scales.scalar_type() == torch::kHalf,
              "w4a16 tiled: bf16 or fp16 scales expected");
  TORCH_CHECK(scales.dim() == 4 && scales.is_contiguous() && scales.size(0) == qweight.size(0) / 8 &&
                  scales.size(1) == qweight.size(1) && scales.size(2) == 8 && scales.size(3) == 8,
              "w4a16 tiled: scales must be a contiguous [K/128, N/64, 8, 8] tensor");
}

bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

}  // namespace

void gemm_w4a16_tiled(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales, torch::Tensor counters,
                      torch::Tensor out, int64_t tile_k, int64_t tile_n, int64_t grid,
                      c10::optional<torch::Tensor> trace) {
  check_tiled_weight(qweight, scales);
  const int64_t k = qweight.size(0) * 16;
  const int64_t n = qweight.size(1) * 64;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.size(1) == k && aligned16(x),
              "gemm_w4a16_tiled: x must be a contiguous, 16-byte aligned [M, K] CUDA tensor");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64, "gemm_w4a16_tiled: 1 to 64 rows per launch");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "gemm_w4a16_tiled: x and scales must share a dtype");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == x.scalar_type() && out.dim() == 2 &&
                  out.size(0) == x.size(0) && out.size(1) == n && aligned16(out),
              "gemm_w4a16_tiled: out must be a contiguous, 16-byte aligned [M, N] tensor of x's dtype");
  TORCH_CHECK(tile_n > 0 && n % tile_n == 0 && tile_k > 0 && k % tile_k == 0,
              "gemm_w4a16_tiled: TILE_N must divide N and TILE_K must divide K");
  TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 && counters.is_contiguous() &&
                  counters.numel() >= n / tile_n,
              "gemm_w4a16_tiled: counters must be a zeroed int32 CUDA tensor with one entry per slice");
  TORCH_CHECK(grid >= 1, "gemm_w4a16_tiled: grid must be positive");
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kBFloat16) {
    launch_tiled_config<__nv_bfloat16>(x, qweight, scales, counters, out, tile_k, tile_n, static_cast<int>(grid),
                                       trace, stream);
  } else {
    launch_tiled_config<__half>(x, qweight, scales, counters, out, tile_k, tile_n, static_cast<int>(grid), trace,
                                stream);
  }
}

torch::Tensor dequantize_w4a16_tiled(torch::Tensor qweight, torch::Tensor scales) {
  check_tiled_weight(qweight, scales);
  const int64_t k = qweight.size(0) * 16;
  const int64_t n = qweight.size(1) * 64;
  auto out = torch::empty({n, k}, scales.options());
  if (n == 0 || k == 0) return out;
  const dim3 grid(static_cast<unsigned>(k / 128), static_cast<unsigned>(n / 64));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (scales.scalar_type() == torch::kBFloat16) {
    dequantize_w4a16_tiled_kernel<__nv_bfloat16><<<grid, 256, 0, stream>>>(
        static_cast<const uint4*>(qweight.data_ptr()), static_cast<const uint4*>(scales.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n), static_cast<int>(k));
  } else {
    dequantize_w4a16_tiled_kernel<__half><<<grid, 256, 0, stream>>>(
        static_cast<const uint4*>(qweight.data_ptr()), static_cast<const uint4*>(scales.data_ptr()),
        static_cast<__half*>(out.data_ptr()), static_cast<int>(n), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gemm_w4a16_wide(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales, torch::Tensor workspace,
                              torch::Tensor counters, int64_t group_size, int64_t splits) {
  check_mma_args(x, qweight, scales, group_size, "gemm_w4a16_wide");
  const int64_t m = x.size(0);
  const int64_t n = qweight.size(0);
  const int64_t groups = qweight.size(1) * 8 / kMmaGroup;
  if (splits <= 0) {  // about two blocks per SM, at least two groups per split
    const int64_t blocks = (n + kWideCols - 1) / kWideCols;
    splits = std::max<int64_t>(1, std::min<int64_t>((2 * sm_count() + blocks - 1) / blocks, groups / 2));
  }
  const Split split = make_split(splits, workspace, counters, m, n, kWideCols, "gemm_w4a16_wide");
  auto y = torch::empty({m, n}, x.options());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kBFloat16) {
    launch_wide_rows<__nv_bfloat16>(x, qweight, scales, y, split, stream);
  } else {
    launch_wide_rows<__half>(x, qweight, scales, y, split, stream);
  }
  return y;
}

torch::Tensor gemm_w4a16_mma(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales, torch::Tensor workspace,
                             torch::Tensor counters, int64_t group_size, int64_t splits) {
  check_mma_args(x, qweight, scales, group_size, "gemm_w4a16_mma");
  const int64_t m = x.size(0);
  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1) * 8;
  const int64_t cols = 8 * mma_tiles_n(static_cast<int>((m + 15) / 16));
  if (splits <= 0) {  // about 1.5 blocks per SM, at least one group per warp (tune.py, run 5)
    const int64_t blocks = (n + cols - 1) / cols;
    const int64_t wanted = (3 * sm_count() + 2 * blocks - 1) / (2 * blocks);
    splits = std::max<int64_t>(1, std::min<int64_t>(wanted, k / kMmaGroup / kWarps));
  }
  const Split split = make_split(splits, workspace, counters, m, n, cols, "gemm_w4a16_mma");
  auto y = torch::empty({m, n}, x.options());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kBFloat16) {
    launch_mma_rows<__nv_bfloat16>(x, qweight, scales, y, split, stream);
  } else {
    launch_mma_rows<__half>(x, qweight, scales, y, split, stream);
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
             pybind11::arg("workspace"), pybind11::arg("counters"),
             pybind11::arg("group_size") = 128, pybind11::arg("splits") = 0);
  module.def("gemm_w4a16_wide", &gemm_w4a16_wide, "W4A16 tensor-core GEMM for 17-64 rows",
             pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("scales"),
             pybind11::arg("workspace"), pybind11::arg("counters"),
             pybind11::arg("group_size") = 128, pybind11::arg("splits") = 0);
  module.def("dequantize_w4a16", &dequantize_w4a16, "int4 weight -> scales dtype",
             pybind11::arg("qweight"), pybind11::arg("scales"), pybind11::arg("group_size") = 128);
  module.def("gemm_w4a16_tiled", &gemm_w4a16_tiled, "Marlin-style W4A16 GEMM for 1-64 rows (tiled layout)",
             pybind11::arg("x"), pybind11::arg("qweight"), pybind11::arg("scales"), pybind11::arg("counters"),
             pybind11::arg("out"), pybind11::arg("tile_k"), pybind11::arg("tile_n"), pybind11::arg("grid"),
             pybind11::arg("trace") = pybind11::none());
  module.def("dequantize_w4a16_tiled", &dequantize_w4a16_tiled, "tiled int4 weight -> [N, K] in the scales dtype",
             pybind11::arg("qweight"), pybind11::arg("scales"));
}
