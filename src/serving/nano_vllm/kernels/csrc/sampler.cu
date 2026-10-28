// Sampling for decode: one token per row of bf16 or fp16 logits [rows, vocab].
//
// No sort and no full-vocab intermediate. Every kernel reads the logits once,
// split into chunks of 2048-8192 tokens so that even a single row fills the
// GPU, and the chunks of a row meet in a workspace that the row's last block
// to finish (a ticket counter, as in w4a16.cu's finish_split) reads and then
// re-zeroes for the next call.
//
// Temperature does not change the order of the logits, and a 16-bit float has
// 65536 values, so any top-k or top-p cutoff is one 16-bit key: the float's
// bits mapped so that integer order is value order. Two histograms, over the
// key's high 10 bits and then its low 6, find it exactly:
//   K1 stats    each row's max and argmax; greedy rows end here.
//   K2 hist     per high bits: the token count (top-k) and the probability
//               mass (top-p), exp((l - max) / T) in fixed point, so sums are
//               exact whatever the order of the atomics. The last block finds
//               the bin that holds the cutoff.
//   K3 refine   per low bits of that bin: count and mass again, which give the
//               cutoff key.
//   K4          rows with both top-k and top-p whose top-p cutoff lies in a
//               bin above the top-k one: K3 again, on that bin.
//   K5 draw     every token at or above the cutoff races with v - log(E),
//               v = l / T, E ~ Exp(1) from Philox keyed by (seed, sequence id,
//               position, token index), so the token does not depend on how
//               the vocab is split.
// A token equal to a cutoff stays with it (sampling.keep_mask is the same rule
// in PyTorch).
//
// On sm_80 only 32-bit integer shared-memory atomics are native (64-bit and
// float ones are CAS loops, which serialize when a block's tokens crowd into a
// few bins), so the histograms add masses as two u32 halves and the draw keeps
// its best candidate in registers until one global atomic per block.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

using u64 = unsigned long long;

constexpr int kThreads = 256;
constexpr int kVec = 8;  // 16-bit logits per 128-bit load
constexpr int kLowBits = 6;
constexpr int kBins1 = 1 << (16 - kLowBits);  // 1024 bins of the key's high bits
constexpr int kBins2 = 1 << kLowBits;         // 64 low-bit values in one of them
// A token's mass relative to the row's max in fixed point, at most 2^37: a
// row of up to 2^18 tokens sums far below 2^64, and all that rounds to 0
// (under 2^-37 each) adds up to under 2^-19 of the max token's mass. In shared
// memory it is added as halves split at bit 19: over a chunk of up to 8192
// tokens both halves' sums stay below 2^32.
constexpr float kMassScale = 137438953472.0f;  // 2^37
constexpr int kSplit = 19;
constexpr int64_t kMaxVocab = int64_t{1} << 18;

// Workspace words (int64) per row; kernels/sampler.py mirrors the state's.
constexpr int kState = 0;  // 8 words:
constexpr int kStateArgmax = 0;
constexpr int kStateMax = 1;         // float bits of the max logit
constexpr int kStateFlags = 2;       // kDone, kSecond
constexpr int kStateBin = 3;         // high bits the refine pass resolves; -1: no cutoff
constexpr int kStateMassAbove = 4;   // mass of the kept tokens above that bin
constexpr int kStateThreshold = 5;   // keep a value while the mass above it is below this
constexpr int kStateCountAbove = 6;  // tokens above that bin
constexpr int kStateCutKey = 7;      // the lowest key kept, -1: every token
constexpr int kRowBest = 8;          // K1: packed (key, ~index) max
constexpr int kDrawBest = 9;         // K5: packed (score, ~index) max
constexpr int kTickets = 10;         // 2 words: int32 tickets of K1, K2, K3/K4, K5
constexpr int kFarMass = 12;         // K2: mass of the tokens no top-p cutoff reaches
constexpr int kCount1 = 16;                  // u32 per high bits (512 words)
constexpr int kMass1 = kCount1 + kBins1 / 2;  // u64 per high bits
constexpr int kCount2 = kMass1 + kBins1;      // u32 per low bits (32 words)
constexpr int kMass2 = kCount2 + kBins2 / 2;  // u64 per low bits
constexpr int kWords = kMass2 + kBins2;

constexpr int64_t kDone = 1;    // out[row] is written (greedy, no finite max, or drawn)
constexpr int64_t kSecond = 2;  // K4 resolves this row's cutoff

// Rows of params (int32 [6, stride]); sampling.Sampler fills them.
constexpr int kTemperature = 0, kTopP = 1, kTopK = 2, kPosition = 3, kSeqLo = 4, kSeqHi = 5;

template <typename T>
struct Half;
template <>
struct Half<__nv_bfloat16> {
  static constexpr uint32_t kNegInf = 0xFF80;
  static constexpr uint32_t kInf = 0x7F80;
  __device__ static float to_float(uint32_t bits) { return __uint_as_float(bits << 16); }
};
template <>
struct Half<__half> {
  static constexpr uint32_t kNegInf = 0xFC00;
  static constexpr uint32_t kInf = 0x7C00;
  __device__ static float to_float(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
  }
};

// NaN becomes -inf and -0 becomes +0, so equal values have equal keys.
template <typename T>
__device__ __forceinline__ uint32_t sanitize(uint32_t bits) {
  if ((bits & 0x7FFF) > Half<T>::kInf) return Half<T>::kNegInf;
  return bits == 0x8000 ? 0 : bits;
}

// Integer order of the key = value order of the float.
__device__ __forceinline__ uint32_t order_key(uint32_t bits) {
  return (bits & 0x8000) ? (~bits & 0xFFFF) : (bits | 0x8000);
}

__device__ __forceinline__ uint32_t key_bits(uint32_t key) {
  return (key & 0x8000) ? (key & 0x7FFF) : (~key & 0xFFFF);
}

__device__ __forceinline__ uint32_t float_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The max of these is the largest key, the smallest index among equals; it is
// never 0 (the smallest sanitized key is -inf's), so 0 means "none yet".
__device__ __forceinline__ u64 pack(uint32_t key, int index) {
  return (static_cast<u64>(key) << 32) | (0xFFFFFFFFu - static_cast<uint32_t>(index));
}

__device__ __forceinline__ int unpack_index(u64 packed) {
  return static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(packed));
}

// (l - max) / T, the log of a token's mass relative to the max token.
__device__ __forceinline__ float log_mass(float logit, float top, float inv_t) { return (logit - top) * inv_t; }

__device__ __forceinline__ u64 mass_fixed(float log_m) { return __float2ull_rz(__expf(log_m) * kMassScale); }

__device__ __forceinline__ void add_mass(uint32_t* lo, uint32_t* hi, int bin, u64 mass) {
  const uint32_t low = static_cast<uint32_t>(mass) & ((1u << kSplit) - 1);
  const uint32_t high = static_cast<uint32_t>(mass >> kSplit);
  if (low) atomicAdd(&lo[bin], low);
  if (high) atomicAdd(&hi[bin], high);
}

__device__ __forceinline__ u64 joined(uint32_t lo, uint32_t hi) { return (static_cast<u64>(hi) << kSplit) + lo; }

__device__ __forceinline__ u64 ceil_mul(float p, u64 total) {
  return static_cast<u64>(ceil(static_cast<double>(p) * static_cast<double>(total)));
}

// Philox4x32-10.
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c.x);
    const uint32_t lo0 = 0xD2511F53u * c.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z);
    const uint32_t lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// log(E) for E ~ Exp(1) from 23 random bits: u = (m + 0.5) / 2^23 and
// E = -log(u) = -log1p(u - 1), where u - 1 is exact, so E keeps its relative
// precision near 0 (the draws that win) as well as near 16.6 (u near 0).
__device__ __forceinline__ float log_exponential(uint32_t random) {
  const float u = (static_cast<float>(random >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return __logf(-log1pf(u - 1.0f));
}

__device__ __forceinline__ u64 warp_max(u64 v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

__device__ __forceinline__ u64 warp_sum(u64 v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// The block's max, in thread 0.
__device__ __forceinline__ u64 block_max(u64 v, u64* scratch) {
  v = warp_max(v);
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) v = warp_max(threadIdx.x < kThreads / 32 ? scratch[threadIdx.x] : 0);
  return v;
}

// The block's sum, in every thread.
__device__ __forceinline__ u64 block_sum(u64 v) {
  __shared__ u64 scratch[kThreads / 32];
  __shared__ u64 total;
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = warp_sum(threadIdx.x < kThreads / 32 ? scratch[threadIdx.x] : 0);
    if (threadIdx.x == 0) total = v;
  }
  __syncthreads();
  v = total;
  __syncthreads();  // before the next call reuses the scratch
  return v;
}

// The lowest bin i of v[0, n) with v[i] != 0 and base + (sum of v above i) <
// limit, or -1, in every thread; *above gets base + the sum above that bin.
// Bins above a qualifying one qualify too, so this is where the running total
// from the top first reaches limit.
template <typename V>
__device__ int find_cut(const V* v, int n, u64 base, u64 limit, u64* above) {
  __shared__ u64 warp_sums[kThreads / 32];
  __shared__ int best;
  __shared__ u64 best_above;
  const int per = (n + kThreads - 1) / kThreads;
  const int top = n - 1 - static_cast<int>(threadIdx.x) * per;  // this thread's bins, top down
  u64 sum = 0;
  for (int j = 0; j < per && top - j >= 0; ++j) sum += v[top - j];
  // Exclusive scan from the top bins (thread 0) down.
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  u64 inclusive = sum;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const u64 x = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += x;
  }
  if (lane == 31) warp_sums[warp] = inclusive;
  if (threadIdx.x == 0) best = n;
  __syncthreads();
  u64 before = base + inclusive - sum;
  for (int w = 0; w < warp; ++w) before += warp_sums[w];
  int candidate = n;
  u64 candidate_above = 0;
  for (int j = 0; j < per && top - j >= 0; ++j) {
    const int i = top - j;
    if (!v[i]) continue;
    if (before >= limit) break;
    candidate = i;
    candidate_above = before;
    before += v[i];
  }
  if (candidate < n) atomicMin(&best, candidate);
  __syncthreads();
  if (candidate < n && candidate == best) best_above = candidate_above;
  __syncthreads();
  const int result = best;
  if (result < n) *above = best_above;
  __syncthreads();  // before the next call reuses the shared variables
  return result < n ? result : -1;
}

// True in the row's last block to finish, which then sees every other block's
// atomics; it resets the ticket when done.
__device__ __forceinline__ bool last_block(int* ticket) {
  __shared__ bool last;
  __threadfence();  // this block's atomics are visible before the ticket moves
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(ticket, 1) == static_cast<int>(gridDim.x) - 1;
  __syncthreads();
  if (last) __threadfence();
  return last;
}

struct Args {
  const uint16_t* logits;
  int vocab;
  const int* params;
  int stride;  // of params
  uint2 seed;
  int64_t* out;
  int64_t* workspace;
};

__device__ __forceinline__ int param(const Args& a, int field, int row) { return a.params[field * a.stride + row]; }
__device__ __forceinline__ float param_f(const Args& a, int field, int row) {
  return __int_as_float(param(a, field, row));
}

// This thread's G 128-bit loads of the block's chunk (zeros past the vocab,
// which callers skip by index).
template <int G>
__device__ __forceinline__ void load_chunk(const Args& a, int row, uint4 (&v)[G], int (&start)[G]) {
  const uint16_t* x = a.logits + static_cast<int64_t>(row) * a.vocab;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    start[g] = (blockIdx.x * G + g) * kThreads * kVec + threadIdx.x * kVec;
    v[g] = start[g] < a.vocab ? __ldg(reinterpret_cast<const uint4*>(x + start[g])) : make_uint4(0, 0, 0, 0);
  }
}

__device__ __forceinline__ uint32_t half_at(const uint4& v, int i) {
  const uint32_t word = (&v.x)[i / 2];
  return (i % 2) ? (word >> 16) : (word & 0xFFFF);
}

template <typename T, int G>
__global__ void __launch_bounds__(kThreads) sample_stats_kernel(Args a) {
  const int row = blockIdx.y;
  int64_t* ws = a.workspace + static_cast<int64_t>(row) * kWords;
  uint4 v[G];
  int start[G];
  load_chunk<G>(a, row, v, start);
  u64 best = 0;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (start[g] >= a.vocab) continue;
#pragma unroll
    for (int i = 0; i < kVec; ++i) best = max(best, pack(order_key(sanitize<T>(half_at(v[g], i))), start[g] + i));
  }
  __shared__ u64 scratch[kThreads / 32];
  best = block_max(best, scratch);
  u64* row_best = reinterpret_cast<u64*>(ws + kRowBest);
  if (threadIdx.x == 0) atomicMax(row_best, best);
  int* tickets = reinterpret_cast<int*>(ws + kTickets);
  if (!last_block(&tickets[0])) return;
  if (threadIdx.x != 0) return;
  const u64 packed = atomicExch(row_best, 0ull);
  const int argmax = unpack_index(packed);
  const float top = Half<T>::to_float(key_bits(static_cast<uint32_t>(packed >> 32)));
  const bool done = !(param_f(a, kTemperature, row) > 0.f) || isinf(top);
  int64_t* state = ws + kState;
  state[kStateArgmax] = argmax;
  state[kStateMax] = __float_as_int(top);
  state[kStateFlags] = done ? kDone : 0;
  state[kStateBin] = -1;
  state[kStateMassAbove] = 0;
  state[kStateThreshold] = 0;
  state[kStateCountAbove] = 0;
  state[kStateCutKey] = -1;
  if (done) a.out[row] = argmax;
  tickets[0] = 0;
}

template <typename T, int G>
__global__ void __launch_bounds__(kThreads) sample_hist_kernel(Args a) {
  const int row = blockIdx.y;
  int64_t* ws = a.workspace + static_cast<int64_t>(row) * kWords;
  int64_t* state = ws + kState;
  if (state[kStateFlags] & kDone) return;
  const int top_k = param(a, kTopK, row);
  const float top_p = param_f(a, kTopP, row);
  const bool use_k = top_k > 0;
  const bool use_p = top_p < 1.f;
  if (!use_k && !use_p) return;  // K5 races every token
  const float inv_t = 1.f / param_f(a, kTemperature, row);
  const float top = __int_as_float(static_cast<int>(state[kStateMax]));
  // Top-p alone never keeps a token more than log(vocab / (1 - p)) nats below
  // the max: all of those together hold under (1 - p) of the max token's mass,
  // so the mass above any of them is over p of the total. They only add to
  // the total, from a register; the histogram is for the rest.
  const float far = use_p && !use_k ? __logf(static_cast<float>(a.vocab) / (1.f - top_p)) : INFINITY;

  __shared__ uint32_t count[kBins1];
  __shared__ uint32_t lo[kBins1];
  __shared__ uint32_t hi[kBins1];
  for (int b = threadIdx.x; b < kBins1; b += kThreads) count[b] = lo[b] = hi[b] = 0;
  uint4 v[G];
  int start[G];
  load_chunk<G>(a, row, v, start);
  __syncthreads();
  u64 far_mass = 0;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (start[g] >= a.vocab) continue;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const uint32_t bits = sanitize<T>(half_at(v[g], i));
      const int bin = static_cast<int>(order_key(bits) >> kLowBits);
      if (use_k) atomicAdd(&count[bin], 1u);
      if (use_p) {
        const float log_m = log_mass(Half<T>::to_float(bits), top, inv_t);
        const u64 m = mass_fixed(log_m);
        if (log_m < -far)
          far_mass += m;
        else if (m)
          add_mass(lo, hi, bin, m);
      }
    }
  }
  far_mass = block_sum(far_mass);  // its barriers also complete the histogram
  if (threadIdx.x == 0 && far_mass) atomicAdd(reinterpret_cast<u64*>(ws + kFarMass), far_mass);
  uint32_t* count1 = reinterpret_cast<uint32_t*>(ws + kCount1);
  u64* mass1 = reinterpret_cast<u64*>(ws + kMass1);
  for (int b = threadIdx.x; b < kBins1; b += kThreads) {
    if (count[b]) atomicAdd(&count1[b], count[b]);
    const u64 m = joined(lo[b], hi[b]);
    if (m) atomicAdd(&mass1[b], m);
  }
  int* tickets = reinterpret_cast<int*>(ws + kTickets);
  if (!last_block(&tickets[1])) return;

  __shared__ u64 mass[kBins1];
  for (int b = threadIdx.x; b < kBins1; b += kThreads) {
    count[b] = __ldcg(&count1[b]);
    mass[b] = __ldcg(&mass1[b]);
  }
  __syncthreads();
  int bin;
  u64 above = 0, mass_above = 0, threshold = 0;
  if (use_k) {
    // The bin holding the k-th largest token: fewer than k tokens above it.
    bin = find_cut(count, kBins1, 0, top_k, &above);
    if (use_p) {
      u64 sum = 0;
      for (int b = threadIdx.x; b < kBins1; b += kThreads)
        if (b > bin) sum += mass[b];
      mass_above = block_sum(sum);
    }
  } else {
    // The lowest bin whose largest value has less than top_p of the mass above it.
    u64 sum = 0;
    for (int b = threadIdx.x; b < kBins1; b += kThreads) sum += mass[b];
    threshold = ceil_mul(top_p, block_sum(sum) + __ldcg(reinterpret_cast<u64*>(ws + kFarMass)));
    bin = find_cut(mass, kBins1, 0, threshold, &mass_above);
  }
  if (threadIdx.x == 0) {
    state[kStateBin] = bin;
    state[kStateCountAbove] = static_cast<int64_t>(above);
    state[kStateMassAbove] = static_cast<int64_t>(mass_above);
    state[kStateThreshold] = static_cast<int64_t>(threshold);
    tickets[1] = 0;
  }
}

// K3 (kSecondPass false) and K4 (true).
template <typename T, int G, bool kSecondPass>
__global__ void __launch_bounds__(kThreads) sample_refine_kernel(Args a) {
  const int row = blockIdx.y;
  int64_t* ws = a.workspace + static_cast<int64_t>(row) * kWords;
  int64_t* state = ws + kState;
  const int64_t flags = state[kStateFlags];
  if (flags & kDone) return;
  if (kSecondPass && !(flags & kSecond)) return;
  const int bin = static_cast<int>(state[kStateBin]);
  if (bin < 0) return;  // no cutoff
  const int top_k = param(a, kTopK, row);
  const float top_p = param_f(a, kTopP, row);
  // The second pass resolves top-p only: the top-k cutoff lies below its bin.
  const bool use_k = !kSecondPass && top_k > 0;
  const bool use_p = top_p < 1.f;
  const float inv_t = 1.f / param_f(a, kTemperature, row);
  const float top = __int_as_float(static_cast<int>(state[kStateMax]));

  __shared__ uint32_t count[kBins2];
  __shared__ uint32_t lo[kBins2];
  __shared__ uint32_t hi[kBins2];
  if (threadIdx.x < kBins2) count[threadIdx.x] = lo[threadIdx.x] = hi[threadIdx.x] = 0;
  uint4 v[G];
  int start[G];
  load_chunk<G>(a, row, v, start);
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (start[g] >= a.vocab) continue;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const uint32_t bits = sanitize<T>(half_at(v[g], i));
      const uint32_t key = order_key(bits);
      if (static_cast<int>(key >> kLowBits) != bin) continue;
      const int low = static_cast<int>(key & (kBins2 - 1));
      if (use_k) atomicAdd(&count[low], 1u);
      if (use_p) {
        const u64 m = mass_fixed(log_mass(Half<T>::to_float(bits), top, inv_t));
        if (m) add_mass(lo, hi, low, m);
      }
    }
  }
  __syncthreads();
  uint32_t* count2 = reinterpret_cast<uint32_t*>(ws + kCount2);
  u64* mass2 = reinterpret_cast<u64*>(ws + kMass2);
  if (threadIdx.x < kBins2) {
    if (count[threadIdx.x]) atomicAdd(&count2[threadIdx.x], count[threadIdx.x]);
    const u64 m = joined(lo[threadIdx.x], hi[threadIdx.x]);
    if (m) atomicAdd(&mass2[threadIdx.x], m);
  }
  int* tickets = reinterpret_cast<int*>(ws + kTickets);
  if (!last_block(&tickets[2])) return;

  __shared__ u64 mass[kBins2];
  __shared__ u64 mass_high[kBins1];  // top-k + top-p: the high bits above `bin`
  uint32_t* count1 = reinterpret_cast<uint32_t*>(ws + kCount1);
  u64* mass1 = reinterpret_cast<u64*>(ws + kMass1);
  if (threadIdx.x < kBins2) {
    count[threadIdx.x] = __ldcg(&count2[threadIdx.x]);
    mass[threadIdx.x] = __ldcg(&mass2[threadIdx.x]);
  }
  const bool both = use_k && use_p;
  if (both)
    for (int b = threadIdx.x; b < kBins1; b += kThreads) mass_high[b] = b > bin ? __ldcg(&mass1[b]) : 0;
  __syncthreads();
  const u64 mass_above = static_cast<u64>(state[kStateMassAbove]);
  int cut = kBins2;  // the lowest low bits of `bin` kept; kBins2: none
  bool second = false;
  int bin_p = -1;
  u64 above_p = 0, threshold = static_cast<u64>(state[kStateThreshold]);
  u64 above;
  if (use_k) {
    const int c = find_cut(count, kBins2, static_cast<u64>(state[kStateCountAbove]), top_k, &above);
    cut = c < 0 ? kBins2 : c;
    if (use_p) {
      // Top-p within what top-k kept.
      const u64 kept = mass_above + block_sum(threadIdx.x < kBins2 && static_cast<int>(threadIdx.x) >= cut
                                                  ? mass[threadIdx.x]
                                                  : 0);
      threshold = ceil_mul(top_p, kept);
      if (mass_above >= threshold) {  // no value of `bin` has under p above it: K4 refines bin_p
        bin_p = find_cut(mass_high, kBins1, 0, threshold, &above_p);
        second = true;
      } else {
        if (threadIdx.x < kBins2 && static_cast<int>(threadIdx.x) < cut) mass[threadIdx.x] = 0;
        __syncthreads();
        const int c_p = find_cut(mass, kBins2, mass_above, threshold, &above);
        cut = c_p < 0 ? kBins2 : max(cut, c_p);
      }
    }
  } else {
    const int c = find_cut(mass, kBins2, mass_above, threshold, &above);
    cut = c < 0 ? kBins2 : c;
  }
  if (threadIdx.x == 0) {
    if (second) {
      state[kStateFlags] = kSecond;
      state[kStateBin] = bin_p;
      state[kStateMassAbove] = static_cast<int64_t>(above_p);
      state[kStateThreshold] = static_cast<int64_t>(threshold);
    } else {
      state[kStateFlags] = 0;
      state[kStateCutKey] = cut < kBins2 ? (bin << kLowBits | cut) : (bin + 1) << kLowBits;
    }
    tickets[2] = 0;
  }
  if (threadIdx.x < kBins2) {
    count2[threadIdx.x] = 0;
    mass2[threadIdx.x] = 0;
  }
  if (!kSecondPass) {
    for (int b = threadIdx.x; b < kBins1; b += kThreads) {
      count1[b] = 0;
      mass1[b] = 0;
    }
    if (threadIdx.x == 0) ws[kFarMass] = 0;
  }
}

// The launch bound's 2 blocks per SM let ptxas use up to 128 registers;
// without it the G=4 kernel stays at 64 and spills.
template <typename T, int G>
__global__ void __launch_bounds__(kThreads, 2) sample_draw_kernel(Args a) {
  const int row = blockIdx.y;
  int64_t* ws = a.workspace + static_cast<int64_t>(row) * kWords;
  int64_t* state = ws + kState;
  if (state[kStateFlags] & kDone) return;
  const int cut_key = static_cast<int>(state[kStateCutKey]);  // -1: every token
  const float inv_t = 1.f / param_f(a, kTemperature, row);
  const uint32_t position = static_cast<uint32_t>(param(a, kPosition, row));
  const uint32_t seq_lo = static_cast<uint32_t>(param(a, kSeqLo, row));
  const uint32_t seq_hi = static_cast<uint32_t>(param(a, kSeqHi, row));
  uint4 v[G];
  int start[G];
  load_chunk<G>(a, row, v, start);
  u64 best = 0;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (start[g] >= a.vocab) continue;
    uint32_t bits[kVec];
    bool keep[kVec];
    bool any = false;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      bits[i] = sanitize<T>(half_at(v[g], i));
      keep[i] = static_cast<int>(order_key(bits[i])) >= cut_key;
      any |= keep[i];
    }
    if (!any) continue;
    const uint32_t group = static_cast<uint32_t>(start[g]) / 4;
    const uint4 r0 = philox(make_uint4(group, position, seq_lo, seq_hi), a.seed);
    const uint4 r1 = philox(make_uint4(group + 1, position, seq_lo, seq_hi), a.seed);
    const uint32_t random[kVec] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      if (!keep[i]) continue;
      const float score = Half<T>::to_float(bits[i]) * inv_t - log_exponential(random[i]);
      best = max(best, pack(float_key(score), start[g] + i));
    }
  }
  __shared__ u64 scratch[kThreads / 32];
  best = block_max(best, scratch);
  u64* draw_best = reinterpret_cast<u64*>(ws + kDrawBest);
  if (threadIdx.x == 0 && best) atomicMax(draw_best, best);
  int* tickets = reinterpret_cast<int*>(ws + kTickets);
  if (!last_block(&tickets[3])) return;
  if (threadIdx.x != 0) return;
  const u64 winner = atomicExch(draw_best, 0ull);
  a.out[row] = winner ? unpack_index(winner) : state[kStateArgmax];
  state[kStateFlags] = kDone;
  tickets[3] = 0;
}

template <typename T, int G>
void launch(const Args& a, int rows, bool any_sampled, bool any_filter, bool any_both) {
  const int chunk = G * kThreads * kVec;
  const dim3 grid((a.vocab + chunk - 1) / chunk, rows);
  // The current stream, so the kernels can be captured in a CUDA graph.
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sample_stats_kernel<T, G><<<grid, kThreads, 0, stream>>>(a);
  if (!any_sampled) return;
  if (any_filter) {
    sample_hist_kernel<T, G><<<grid, kThreads, 0, stream>>>(a);
    sample_refine_kernel<T, G, false><<<grid, kThreads, 0, stream>>>(a);
    if (any_both) sample_refine_kernel<T, G, true><<<grid, kThreads, 0, stream>>>(a);
  }
  sample_draw_kernel<T, G><<<grid, kThreads, 0, stream>>>(a);
}

template <typename T>
void launch_rows(const Args& a, int rows, bool any_sampled, bool any_filter, bool any_both) {
  // Smaller chunks for fewer rows, so that one row still spreads over ~63 blocks.
  if (rows <= 8)
    launch<T, 1>(a, rows, any_sampled, any_filter, any_both);
  else if (rows <= 32)
    launch<T, 2>(a, rows, any_sampled, any_filter, any_both);
  else
    launch<T, 4>(a, rows, any_sampled, any_filter, any_both);
}

}  // namespace

void sample(torch::Tensor logits, torch::Tensor params, torch::Tensor out, torch::Tensor workspace, int64_t seed,
            bool any_sampled, bool any_filter, bool any_both) {
  TORCH_CHECK(logits.is_cuda() && params.is_cuda() && out.is_cuda() && workspace.is_cuda(),
              "sample: CUDA tensors expected");
  TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 || logits.scalar_type() == torch::kHalf,
              "sample: bf16 or fp16 logits expected");
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(), "sample: contiguous [rows, vocab] logits expected");
  const int64_t rows = logits.size(0);
  const int64_t vocab = logits.size(1);
  TORCH_CHECK(vocab % kVec == 0 && vocab <= kMaxVocab, "sample: vocab must be a multiple of 8, at most 2^18");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0, "sample: 16-byte aligned logits expected");
  TORCH_CHECK(params.scalar_type() == torch::kInt && params.dim() == 2 && params.size(0) == 6 &&
                  params.size(1) >= rows && params.is_contiguous(),
              "sample: int32 params [6, >= rows] expected");
  TORCH_CHECK(out.scalar_type() == torch::kLong && out.is_contiguous() && out.numel() >= rows,
              "sample: int64 out with a slot per row expected");
  TORCH_CHECK(workspace.scalar_type() == torch::kLong && workspace.dim() == 2 && workspace.size(0) >= rows &&
                  workspace.size(1) == kWords && workspace.is_contiguous(),
              "sample: int64 workspace [>= rows, WORDS_PER_ROW] expected");
  if (rows == 0) return;
  Args a;
  a.logits = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  a.vocab = static_cast<int>(vocab);
  a.params = params.data_ptr<int>();
  a.stride = static_cast<int>(params.size(1));
  a.seed = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32));
  a.out = out.data_ptr<int64_t>();
  a.workspace = workspace.data_ptr<int64_t>();
  if (logits.scalar_type() == torch::kBFloat16)
    launch_rows<__nv_bfloat16>(a, static_cast<int>(rows), any_sampled, any_filter, any_both);
  else
    launch_rows<__half>(a, static_cast<int>(rows), any_sampled, any_filter, any_both);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sample", &sample, "one token per row of 16-bit logits: argmax, or top-k / top-p and a draw",
             pybind11::arg("logits"), pybind11::arg("params"), pybind11::arg("out"), pybind11::arg("workspace"),
             pybind11::arg("seed"), pybind11::arg("any_sampled"), pybind11::arg("any_filter"),
             pybind11::arg("any_both"));
  module.attr("WORDS_PER_ROW") = kWords;
  module.attr("KEY_LOW_BITS") = kLowBits;
}
