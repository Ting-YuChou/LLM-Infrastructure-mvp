# Phase 0：Baseline 與 roofline

Llama 3.1 8B Instruct（BF16）在一張 A100 上的 baseline，對照組是 vLLM 和 SGLang，再加上自己的 nano_vllm 當作起點。每個數字都和 roofline 上界比較，之後每個優化都用「離上界多遠」來評分。路線圖在 [Inference Engine 優化路線圖](https://claude.ai/artifact/Goa72xjG2zaX8Lw411ueiU)，執行過程、失敗和修正都記在 [LOG.md](LOG.md)。

- **Run**：`results/20261022-colab-a100-40gb/`，2026-10-22。
- **硬體**：Google Colab A100-SXM4-40GB（driver 580，1,555 GB/s，BF16 312 TFLOPs）。
- **軟體**：vLLM 0.19.1（torch 2.10）、SGLang 0.5.22（torch 2.14、flashinfer 0.7.1）、nsys / ncu 2025.3。

## 重點

1. **batch=1 時，vLLM 和 SGLang 已經貼著 cuBLAS 的下限。**
   - vLLM 每個 decode step 12.62 ms，GPU 有 98% 的時間在跑 kernel，其中 91% 是 GEMV。
   - 只把 cuBLAS 線性層（M=1）的時間加總就是 12.49 ms，也就是說 vLLM 只比這個下限多出 0.13 ms。
   - 剩下的空間在 GEMV 本身：cuBLAS 在 o_proj 只吃到 63% 的頻寬、qkv 73%。如果每個 GEMV 都能跑到 copy 頻寬（1,380 GB/s），就是 10.9 ms，快 14%。
   - 要再快更多就得減少要讀的 bytes，例如 W4A16。
2. **CUDA graph 每個 token 省 10 ms。**
   - vLLM 用 eager 時每個 step 22.7 ms：GPU 只忙 12.9 ms，其餘 15.9 ms 都在等 369 次 kernel launch。
   - 開了 CUDA graph 之後，每個 step 只剩 15 次 launch。
3. **nano_vllm 的起點是 vLLM 的 3.5 倍慢，批次處理更糟。**
   - static b=1：每 token 44.1 ms。每個 step 要 launch 1,573 個 kernel，GPU 只有 19 ms 在工作，40 ms 都在等 CPU。
   - paged b=16（輸入 1024）：每 step 619 ms，只有 26 tok/s。每個 step 有 17,619 個 kernel，70% 的 kernel 時間花在 elementwise 和 copy，也就是逐條 sequence 搬 KV。
4. **batch 大了以後，attention 變得重要。**
   - vLLM 在 b=64、ctx 約 1,060 時，attention（FA2 split-KV）佔 GPU 時間的 30%。用 KV bytes 估算，大約跑到 1.3 TB/s（峰值的 84%）。
   - 這就是 Phase 1 paged decode attention kernel 要追的對照組。
5. **吞吐量**（輸入 1024 / 輸出 256）：
   - vLLM 在 c=64 時最高，1,840 tok/s；SGLang 在 c=32 時最高，1,409 tok/s。
   - SGLang 到 c=64 反而掉到 1,316 tok/s，TPOT 從 18.5 ms 跳到 40 ms。推測是因為它預設的 decode CUDA graph 只 capture 到 batch 32，這還沒驗證。
   - 40GB 卡扣掉權重後，KV cache 只放得下大約 14 萬個 token，所以 c=128 時兩個 engine 都開始排隊。

## Roofline

一個 decode step 要把整份權重從 HBM 讀一次。路線圖原本用全部權重 16.06 GB 去算，但 embedding table 每個 token 只讀一列，lm_head 才是整個讀，所以實際讀的是 **15.01 GB**：

$$t_{\text{token}} \geq \frac{15.01\ \text{GB}}{\text{HBM 頻寬}}$$

另外每個 token 要讀 128 KiB 的 KV cache（2 × 32 層 × 8 個 KV head × 128 × 2 bytes）。`roofline.py` 從模型 shape 算出 bytes 和 FLOPs，取 max(記憶體時間, 計算時間)。

| GPU | 頻寬 | batch=1 下限 | 上限 |
|---|---:|---:|---:|
| A100 80GB（datasheet） | 2,039 GB/s | 7.38 ms | 136 tok/s |
| A100 40GB（datasheet） | 1,555 GB/s | 9.67 ms | 103 tok/s |
| A100 40GB（這張卡實測 1 GiB copy） | 1,380 GB/s | 10.90 ms | 92 tok/s |

batch 和 roofline 的完整對照表在 [roofline.md](results/20261022-colab-a100-40gb/roofline.md)。

## Batch=1 decode

| 量測 | ms/token | tok/s | 佔 roofline |
|---|---:|---:|---:|
| roofline（datasheet） | 9.67 | 103 | 100% |
| roofline（實測 copy 頻寬） | 10.90 | 92 | 89% |
| cuBLAS 線性層加總（M=1，不含 attention） | 12.49 | 80 | 77% |
| SGLang serve，TPOT p50（輸入 128 / 輸出 256） | 12.61 | 79 | 77% |
| vLLM engine step（CUDA graph） | 12.62 | 79 | 77% |
| vLLM serve，TPOT p50（輸入 128 / 輸出 256） | 12.97 | 77 | 75% |
| vLLM engine step（eager） | 22.68 | 44 | 43% |
| nano_vllm static 路徑 | 44.10 | 23 | 22% |
| nano_vllm paged engine | 53.63 | 19 | 18% |

### cuBLAS GEMV 的頻寬（`bandwidth.py`，M=1）

| 層 | shape（out × in） | 權重 | 時間 | 有效頻寬 | 佔峰值 |
|---|---|---:|---:|---:|---:|
| qkv_proj | 6144 × 4096 | 50 MB | 44.1 µs | 1,142 GB/s | 73% |
| o_proj | 4096 × 4096 | 34 MB | 34.4 µs | 976 GB/s | 63% |
| gate_up_proj | 28672 × 4096 | 235 MB | 182.1 µs | 1,290 GB/s | 83% |
| down_proj | 4096 × 14336 | 117 MB | 107.1 µs | 1,096 GB/s | 70% |
| lm_head | 128256 × 4096 | 1,051 MB | 726.5 µs | 1,446 GB/s | 93% |

矩陣越小，cuBLAS 越難把頻寬吃滿。M 一路到 16，線性層的總時間都維持在 12.2–12.6 ms，M=64 是 15.5 ms，M=256 是 20.1 ms。M=4096 時 GEMM 跑到 247–269 TFLOPs，是峰值的 79–86%。

## 吞吐量 sweep

用 `vllm bench serve` 當 client，random dataset，輸入 1024 / 輸出 256，開 `--ignore-eos`，每組都確認 `completed` 和 `total_output_tokens` 等於預期。

| engine | concurrency | 輸出 tok/s | TTFT p50 (ms) | TPOT p50 (ms) | ITL p99 (ms) |
|---|---:|---:|---:|---:|---:|
| vLLM | 1 | 76 | 26 | 13.1 | 13.3 |
| vLLM | 8 | 542 | 75 | 14.5 | 15.9 |
| vLLM | 32 | 1,332 | 234 | 21.4 | 136.4 |
| vLLM | 64 | **1,840** | 606 | 27.3 | 151.8 |
| vLLM | 128 | 1,725 | 737 | 71.5 | 186.2 |
| SGLang | 1 | 78 | 53 | 12.7 | 12.9 |
| SGLang | 8 | 565 | 85 | 13.9 | 14.5 |
| SGLang | 32 | **1,409** | 172 | 18.5 | 19.4 |
| SGLang | 64 | 1,316 | 1,994 | 40.0 | 38.7 |
| SGLang | 128 | 1,658 | 4,398 | 48.4 | 42.8 |

- **ITL**：vLLM 在 c≥16 時 p99 ITL 跳到 64–186 ms，SGLang 一直維持在 43 ms 以下。這是 prefill 打斷 decode 的結果，chunked prefill 的策略不同。Phase 2 要量的就是這個。
- **TTFT**：c=1 時 SGLang 的 TTFT（約 50 ms）是 vLLM（22–26 ms）的兩倍。
- **啟動時間**：vLLM server 86 秒 ready；SGLang 152 秒，其中包含 flashinfer 的 JIT 編譯。

完整表格（含 p99 和總吞吐量）在 [summary.md](results/20261022-colab-a100-40gb/summary.md)，圖在 [plots/](results/20261022-colab-a100-40gb/plots/)：

![throughput](results/20261022-colab-a100-40gb/plots/throughput_vs_concurrency.png)

## 一個 decode step 的時間花在哪裡（nsys）

`vllm_offline.py` 和 `nano_bench.py` 自己驅動 engine 的 step，每個 step 包一個 NVTX range，只錄 prefill 結束之後的 decode step。表中數字是 64 個 step 的中位數。

| capture | batch | ms/step | GPU busy | GPU idle | kernels/step | launch 次數/step | kernel 時間分布 |
|---|---:|---:|---:|---:|---:|---:|---|
| vLLM CUDA graph | 1 | 12.85 | 12.57 | 0.28 | 365 | 15 | GEMM 91%、attention 5% |
| vLLM eager | 1 | 28.74 | 12.87 | 15.87 | 369 | 369 | GEMM 89%、attention 5% |
| vLLM CUDA graph（ctx ≈1,060） | 64 | 22.69 | 22.62 | 0.07 | 463 | 15 | GEMM 65%、attention 31% |
| nano static | 1 | 59.37 | 19.08 | 40.30 | 1,573 | 1,573 | GEMM 73%、elementwise/copy 22% |
| nano paged | 1 | 69.88 | 19.87 | 50.01 | 1,604 | 1,604 | GEMM 72%、elementwise/copy 24% |
| nano paged（ctx ≈1,060） | 16 | 807.21 | 128.20 | 679.02 | 17,619 | 17,619 | elementwise/copy 70%、GEMM 25% |

nsys 本身有一點 overhead：vLLM graph b=1 在 nsys 下是 12.85 ms，不開時是 12.62 ms。eager 和 nano 這種 launch 很多的情況，nsys 的影響比較大（eager：28.7 vs 22.7 ms）。

| vLLM eager b=1（5 ms 放大）：kernel 之間的空檔是 launch | vLLM CUDA graph b=1（3 個 step）：GPU 一直在忙 |
|---|---|
| ![eager](results/20261022-colab-a100-40gb/nsys/vllm_eager_b1_timeline.png) | ![graph](results/20261022-colab-a100-40gb/nsys/vllm_graph_b1_timeline.png) |

| nano static b=1（5 ms 放大） | nano paged b=16（5 ms 放大） |
|---|---|
| ![nano static](results/20261022-colab-a100-40gb/nsys/nano_static_b1_timeline.png) | ![nano paged b16](results/20261022-colab-a100-40gb/nsys/nano_paged_b16_timeline.png) |

## Decode kernel 的硬體 counter（ncu）

只有 Colab 的 ncu 能讀硬體 counter，RunPod 上不行，見路線圖的「Profiling 環境」。這裡錄了 vLLM b=1 完整的一個 decode step（367 個 kernel），以及 nano static b=1 的前 300 個 kernel（大約 10 層）。

設定：
- sections：SpeedOfLight、MemoryWorkloadAnalysis、Occupancy、LaunchStats、WarpStateStats。
- `--clock-control none`，讓 kernel 跑在實際的 boost clock。
- ncu 預設每個 pass 之前會清 cache，所以加總起來（13.39 ms）比 nsys 量到的 12.57 ms 略長。

**vLLM b=1**（一個 step 的 kernel 時間是 13.39 ms，平均 DRAM 頻寬 1,147 GB/s）：

| kernel | 對應的層（從次數和平均時間推測） | 次數 | 時間佔比 | DRAM 佔峰值 | achieved occupancy | 主要 stall |
|---|---|---:|---:|---:|---:|---|
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2…` | gate_up ×32、lm_head | 33 | 47% | 88% | 12% | long_scoreboard |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2…` | qkv、down | 64 | 31% | 75% | 6% | long_scoreboard |
| `cutlass::Kernel2<…s16816gemm…>` | o_proj | 32 | 11% | 77% | 6% | long_scoreboard |
| `flash_fwd_splitkv_kernel` + `combine` | attention | 64 | 5% | 1–4% | 6–7% | long_scoreboard |
| `triton_red_fused_…rsqrt`（residual + RMSNorm） | norm | 64 | 2% | ≈0 | 24% | long_scoreboard |

GEMV 的 occupancy 很低（6–12%），stall 都在 long_scoreboard（等 global memory）。o_proj 那個 cutlass kernel 的 DRAM 流量比它的權重（34 MB）還多。以權重 bytes 計算的有效頻寬只有約 760 GB/s，推測多出來的是 split-K 的 workspace 讀寫。

**nano static b=1**：最大的兩個 GEMV 跑到 1,175–1,232 GB/s，和 vLLM 差不多。但它在每一層多跑了大約 25 個 PyTorch 的 elementwise、index 和 cat kernel（RoPE、repeat_kv、KV cache 讀寫），每個只有 3–8 µs，DRAM 用量接近 0。時間全浪費在 launch 上。

完整的每個 kernel 表格在 [summary.md](results/20261022-colab-a100-40gb/summary.md)。原始報告 `ncu/*.ncu-rep` 有 80–130 MB，沒有進 git。

## 對 Phase 1–2 的意義

照影響大小排：

1. **nano 的 paged decode attention kernel**（Phase 1 第一項）：b=16 時有 70% 的 kernel 時間花在逐條 sequence 搬 KV，GPU 只有 16% 的時間在忙。對照組是 vLLM 的 FA2 split-KV，在 b=64 時跑到約 1.3 TB/s。
2. **nano 的融合 kernel 加上 CUDA graph**（Phase 1 fusion + Phase 2）：
   - 每個 step 1,573 → 300–400 個 kernel，再用 CUDA graph 把 launch 從每 step 1,573 次降到個位數。
   - 目標是從 44 ms 降到 vLLM 的 12.6 ms。
   - vLLM eager vs graph 那組（22.7 → 12.6 ms）已經說明這一步值多少。
3. **打贏 cuBLAS 的 bf16 GEMV**（Phase 1 進階）：o_proj 和 qkv 的形狀只吃到 63–73% 的頻寬。目標是 ≥85%，能把 batch=1 從 12.5 ms 拉到大約 11 ms。
4. **W4A16 GEMV**：線性層讀的 bytes 是四分之一，batch=1 的理論下限從 9.7 ms 降到約 2.4–2.9 ms（差在 lm_head 有沒有一起量化）。這是 batch=1 延遲最大的槓桿。
5. **待驗證**：SGLang c=64 的掉落是不是因為 decode CUDA graph 的 `max_bs=32`。可以用 `--cuda-graph-max-bs` 調大重跑一次確認。如果是預設值在 40GB A100 上選得不好，也許能當 upstream issue 或 PR 的題材。

## 重現

```bash
bash benchmarks/phase0/colab_phase0.sh
```

這會開一台 Colab A100，上傳程式碼，跑完全部步驟，結果下載到 `results/<run_id>/`，最後關掉 VM。整個流程大約 1 小時。

```bash
P0_RUN_ID=<run_id> P0_STEPS="ncu summarize" COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun
```

在同一台 VM 上只重跑某些步驟。venv 和模型都會保留；nsys 和 ncu 已經分析過的 label 會自動跳過。

```bash
python benchmarks/phase0/roofline.py --gpu A100-SXM4-40GB --bandwidth-json benchmarks/phase0/results/<run_id>/bandwidth.json
```

在本機重新算 roofline。

```bash
python benchmarks/phase0/summarize.py benchmarks/phase0/results/<run_id>
```

在本機重新產生 summary.md 和圖，需要 matplotlib。

`run_phase0.sh` 本身不綁平台，在 RunPod 這類 80GB 的機器上也能直接當 pod 的啟動指令跑。設 `P0_REPO` 的話，每一步結束都會把結果上傳到私有 HF dataset repo；不過 RunPod 上 ncu 不能用。

## 檔案

| 檔案 | 內容 |
|---|---|
| `run_phase0.sh` | 主流程：安裝、benchmark、nsys、ncu、彙整。每一步各自寫 log 並記進 `status.tsv`，某一步失敗不會影響後面的步驟 |
| `colab_phase0.sh`、`colab_launcher.py` | 從本機驅動 Colab：上傳、執行、串流 log、下載 |
| `roofline.py`、`bandwidth.py` | 上界計算；實測 copy 頻寬和 cuBLAS 在各個 M 的時間 |
| `vllm_offline.py`、`nano_bench.py` | 自己驅動 engine 的 step 來計時，只 profile decode step |
| `analyze_nsys.py`、`analyze_ncu.py`、`summarize.py` | 把 SQLite 和 raw CSV 整理成 JSON、表格和圖 |
| `results/<run_id>/` | `summary.md`、`serve/`（每組 sweep 的 JSON）、`nsys/`、`ncu/`、`offline/`、`logs/`、`environment.txt`、`status.tsv`，以及當次 run 的程式碼副本 `code/` |

## 限制

- **硬體**：這次是 A100 40GB，不是路線圖原本設定的 80GB。80GB 的頻寬是 2,039 GB/s，KV cache 也多一倍，絕對數字會不同；佔 roofline 的比例比較能直接比。
- **重複次數**：每組設定只跑一次，沒有重複量測，也沒有算變異。從 c=1 的 TPOT 在 latency 和 sweep 兩次量測之間只差 1% 來看，雜訊不大。
- **模型**：用的是 `unsloth/Meta-Llama-3.1-8B-Instruct`，權重和 meta-llama 的官方版相同，只是不需要 token。
- **工作負載**：random token prompt、greedy decode、`ignore_eos`。沒有測 prefix cache 命中、真實長度分布，或 Poisson 到達。
- **SGLang**：沒有錄 nsys，因為它是多 process 的 server。
- **nano_vllm 的 bug**：它的 `silu_mul` Triton kernel 原本不能在 bf16 上編譯。這是 Phase 0 對 repo 唯一的程式碼修改，見 LOG.md。
