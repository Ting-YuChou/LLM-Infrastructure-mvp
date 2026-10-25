# Phase 0 執行紀錄

照時間順序記錄每一步：做了什麼、為什麼、花多少錢、遇到什麼問題。結果和分析在 [README.md](README.md)；路線圖在 [Inference Engine 優化路線圖](https://claude.ai/artifact/Goa72xjG2zaX8Lw411ueiU)。時間一律是 UTC。

## 2026-10-22 10:50 — 準備

**目標**（路線圖 Phase 0）：
- 量出 vLLM 和 SGLang 在 1× A100 80GB 跑 Llama 3.1 8B Instruct（BF16）的 baseline：batch=1 延遲、各 concurrency 的吞吐量、TTFT、ITL。
- 用 nsys 錄 decode，看時間花在 kernel、kernel launch 還是 CPU。
- 算 roofline 上界。

**順便量的**：
- nano_vllm 現在的數字，當作自己 engine 的起點。
- 這張卡實際能達到的 HBM 頻寬，以及 cuBLAS 線性層的下限。

### 寫的腳本（都在這個目錄）

| 檔案 | 做什麼 |
|---|---|
| `run_phase0.sh` | pod 上的主流程：安裝、下載模型、跑每一步，每步結束就上傳結果 |
| `roofline.py` | 從模型 shape 算 bytes/FLOPs 的上界（本機可跑） |
| `bandwidth.py` | 1 GiB copy 頻寬，加上 qkv/o/gate_up/down/lm_head 在 M=1…4096 的 cuBLAS 時間 |
| `vllm_offline.py` | 在同一個 process 裡跑 vLLM，自己呼叫 `LLMEngine.step()` 計時，只 profile prefill 結束後的 64 個 decode step |
| `nano_bench.py` | nano_vllm 的 static 路徑和 paged engine 路徑，每個 step 一個 NVTX range |
| `analyze_nsys.py` | 從 nsys 匯出的 SQLite 算每個 step 的 wall time、GPU busy/idle、kernel 數、launch 次數、kernel 分類，並輸出時間軸 PNG |
| `summarize.py` | 把一次 run 的結果整理成 `summary.md` 和圖 |

### 決定和理由

- **Benchmark 設定**：兩個 engine 都用 `vllm bench serve` 當 client，資料用 random dataset 加 `--ignore-eos`，讓每個 request 的輸入輸出長度完全固定。
  - latency：輸入 128 / 輸出 256，concurrency 1，10 個 request。
  - sweep：輸入 1024 / 輸出 256，concurrency 1–256，request 數取 max(16, 4×concurrency)。
  - 每組都會檢查 `completed` 和 `total_output_tokens` 是否等於預期，確認 ignore_eos 真的生效。
- **版本**：
  - vLLM 用 repo 釘的 0.19.1（`requirements.txt`）。
  - SGLang 用當下 PyPI 的最新版，實際版本記在 `environment.txt`。
  - 兩者各自用一個 uv venv，因為它們釘的 torch 不同，不能共用環境。
- **Server 參數**：只改 `--max-model-len 8192`（SGLang 是 `--context-length 8192`）和 seed，其餘都用預設值。Phase 0 要量的就是開箱即用的 baseline。
- **nsys**：`-t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi`，只錄量測的那一段，載入模型和 CUDA graph capture 不會進到報告裡。
  - vLLM 錄三組：CUDA graph b=1、eager b=1、CUDA graph b=64。
  - nano_vllm 錄三組：static b=1、paged b=1、paged b=16。
  - SGLang 這次不錄：它是多 process 的 server，要另外處理。
- **硬體**：一開始打算用 RunPod A100 80GB，後來改用 Colab，原因和經過見下一段。

### 本機先驗證過的

- `roofline.py` 可以跑，算出的數字見下一段。
- `nano_bench.py --device cpu` 用 `tests/unit/tiny_llama.py` 的 tiny Llama 跑通：static 路徑、paged b=1、paged b=4 都正常。
- `analyze_nsys.py` 用一個人工造的 nsys schema SQLite 測過，per-step 的 busy/idle 和 kernel 分類都正確。
- `summarize.py` 用假資料跑通，三張圖都有產生。
- `bash -n run_phase0.sh` 語法檢查通過。
- `bandwidth.py` 和 `vllm_offline.py` 需要 GPU，本機沒辦法測。
- `analyze_ncu.py` 用人工造的 ncu raw CSV（含單位列、千分位逗號）測過。

### Roofline 修正

路線圖原本寫 16.06 GB / 2.039 TB/s ≈ 7.9 ms。但 decode 時 embedding table 每個 token 只讀一列，不是整個 1.05 GB 的矩陣，所以每個 decode step 實際讀的權重是 **15.01 GB**：

$$t_{\text{token}} \geq \frac{15.01\ \text{GB}}{2.039\ \text{TB/s}} = 7.36\ \text{ms}$$

算上 KV cache 讀取（ctx 256）是 **7.38 ms/token（136 tok/s）**。用之前實測的 1,763 GB/s 換算是 8.5 ms。

## 2026-10-22 11:20 — 改用 Colab A100 40GB

原本要在 RunPod Secure A100 80GB（$1.79/hr）上跑，但遇到兩個問題：

1. **檔案進不去 pod**：腳本還沒 commit，而 GitHub repo 是公開的，pod 沒辦法用 git clone 拿到。
   - 試過把腳本傳到私有 HF repo 給 pod 下載，但本機的 HF token 是唯讀的（`hf repo create` 回 "Cannot access content at api/repos/create"）。
   - 改成讓 pod 在 8888 port 開一個帶金鑰的上傳端點，被 Claude Code 的權限檢查擋下（Expose Local Services）。
2. **使用者建議改用 Colab**：Colab 的 ncu 可以用，kernel 層級的分析在 40GB 上就夠，8B BF16 的權重也放得下。Colab CLI 本來就能直接上傳本機檔案、下載結果，CU 額度也已經有了。

所以改成：
- `colab_phase0.sh` 在本機把 `benchmarks/phase0` 和 `src/serving/nano_vllm` 打包上傳。
- `colab_launcher.py` 在 VM 上跑 `run_phase0.sh`。
- 結果下載到 `results/<run_id>/`。
- `run_phase0.sh` 改成不綁平台（`P0_ROOT`、`P0_REPO_DIR`、`P0_CONCURRENCY`），RunPod 之後還是能用。

和 80GB 的差別：
- **頻寬**：A100 40GB 是 1,555 GB/s，batch=1 的 roofline 是 9.67 ms/token（103 tok/s），不是 7.38 ms。
- **KV cache**：扣掉 16 GB 權重後大約剩 18 GB，約 145k tokens，所以 sweep 只跑到 concurrency 128。輸入 1024、輸出 256 時，再往上只會排隊。
- **模型**：改用免 token 的 `unsloth/Meta-Llama-3.1-8B-Instruct`。權重和官方版相同，這樣 HF token 完全不用經手。
- **ncu（新增）**：vLLM b=1、vLLM b=64 各錄一個完整 decode step；nano_vllm static b=1 錄前 300 個 kernel（大約 10 層）。sections 包含 SpeedOfLight、MemoryWorkloadAnalysis、Occupancy、LaunchStats、WarpStateStats，並設 `--clock-control none`，讓 kernel 跑在實際的 boost clock。

80GB 的 headline 數字如果之後 write-up 需要，再用 RunPod 補跑一次（加 `P0_REPO` 就會把結果上傳到 HF）。

## 2026-10-22 10:57 — 第一次在 Colab 上跑（run `20261022-colab-a100-40gb`）

- `colab new --gpu A100` 拿到 A100-SXM4-40GB。
- 安裝並行進行，都成功：
  - vLLM 0.19.1 venv：196 s
  - SGLang venv：193 s
  - 模型下載（16 GB）：65 s
  - nsys 是 Colab 預裝的
- `bandwidth.py`（4 s）：
  - 1 GiB copy 1,382 GB/s，是 1,555 GB/s 峰值的 89%
  - 最佳 GEMM 270 TFLOPs，是 312 的 87%
  - qkv_proj 在 M=1 時只有 1,115 GB/s，在 M=4096 時是 216 TFLOPs
- **失敗：vLLM server 20 秒就掛了。**
  - 錯誤：`ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend`
  - 原因：Colab 的 Jupyter kernel 設了 `MPLBACKEND=module://matplotlib_inline.backend_inline`（還有 `PYTHONPATH=/env/python`），子 process 會繼承。vLLM 一 import matplotlib 就讀到這個不存在的 backend。
  - 之後每個會載入 vLLM 或 matplotlib 的步驟都會失敗。所以先停掉本機 driver，再用 `colab console` 把遠端的 run 和 SGLang server 也停掉，確認 GPU 記憶體回到 0 MiB。
- **修正**：
  - `colab_launcher.py`：拿掉 `PYTHONPATH`，設 `MPLBACKEND=Agg`。
  - `run_phase0.sh`：也 export `MPLBACKEND=Agg`，平台換了也不會再遇到。
  - `colab_phase0.sh`：加一個 `rerun` 模式，重用同一台 VM（venv 和模型都還在），跳過安裝步驟。
- 用同一個 run id 重跑。`status.tsv` 會保留第一次失敗的那一行。

## 2026-10-22 11:04 — 重跑（同一台 VM，跳過安裝）

- `bandwidth`：4 s，數字和第一次相同。
- **vLLM serve：成功**（499 s）。
  - server 86 秒 ready，佔用 37 GB 顯存。
  - 每組都通過 `completed` 和 `total_output_tokens` 的檢查。
  - latency（輸入 128 / 輸出 256，c=1）：TPOT p50 **12.97 ms**（77 tok/s），TTFT 22 ms。
  - sweep 的輸出吞吐量：c=1 76、c=8 542、c=32 1,332、c=64 **1,840 tok/s**（最高）、c=128 1,725 tok/s。
  - c=128 時 TPOT p50 跳到 71.5 ms：KV 不夠，開始 preempt。
- **SGLang serve：失敗**（45 s）。
  - 錯誤：`FileNotFoundError: 'ninja'`，發生在 flashinfer JIT 編譯 kernel、SGLang capture prefill CUDA graph 的時候。
  - 這版 SGLang 預設 `attention_backend=flashinfer`、`mem_fraction_static=0.823`，KV cache 139,721 tokens。
  - Colab 的 nvcc 是 13.0。
- **處理**：
  - 停掉本機 driver，不讓它跑完就關 VM。遠端照樣跑 nsys、nano、ncu，我每 60 秒下載 `console.log` 盯進度。
  - `install_sglang` 加裝 ninja，venv 已存在時直接沿用。
  - `sglang_serve` 把 venv 的 bin 放進 PATH；預設設定起不來時，自動改用 `--attention-backend triton --sampling-backend pytorch` 再試一次，並把實際用的參數寫進 `serve/sglang_startup.json`。
- 等這輪跑完，用 `P0_STEPS="sglang_serve summarize"` 在同一台 VM 上補跑 SGLang。

## 2026-10-22 11:16 — 又抓到兩個 bug，停掉這輪

- **nsys_vllm 失敗**（166 s）：`statistics.StatisticsError: fmean requires at least one data point`。
  - 原因：`vllm_offline.py` warm-up 時把同一個 `SamplingParams` 傳給了 `llm.generate()`，它會把 `output_kind` 原地改成 `FINAL_ONLY`。之後 `engine.step()` 只在 request 結束時才回傳輸出，程式永遠判斷不出 prefill 結束，所以 decode step 一個都沒記到，profiler 也從來沒啟動。
  - 修正：每次都建一個新的 `SamplingParams(output_kind=DELTA)`，用累加的方式計算 token 數。
- **nano 失敗**（56 s）：`triton.compiler.errors.CompilationError ... Expected dtype ['fp32', 'fp64'] but got bf16`。
  - 原因：`src/serving/nano_vllm/kernels/silu_mul.py` 的 `_silu_mul_kernel` 直接對 bf16 呼叫 `tl.exp`。這個 repo 的 Triton kernel 以前從沒在 GPU 上跑過 bf16，CPU 測試只會走 PyTorch fallback，所以一直沒被發現。
  - 修正：照 `rmsnorm.py` 的做法，load 進來先 `.to(tl.float32)`，store 時轉回 `out_ptr.dtype.element_ty`。`tests/unit/test_nano_vllm_kernels.py` 和 `test_nano_vllm_model.py` 跑出 22 passed、2 skipped。
- **這是 Phase 0 對 repo 唯一的程式碼修改**（`benchmarks/phase0/` 以外）。
- **SGLang 的環境**：
  - 版本：sglang 0.5.22、torch 2.14.1+cu130、flashinfer 0.7.1、sglang-kernel 0.4.9。
  - venv 裡其實已經有 ninja 1.13.2，只是 venv 的 bin 不在 PATH 上。torch 是 cu130，剛好和 Colab 的 nvcc 13.0 對得上，所以應該補上 PATH 就夠了。
- 接下來的 ncu 步驟一定也會碰到這兩個 bug，所以用 `colab console` 先停掉，GPU 回到 0 MiB。
- nsys/ncu 改成會跳過已經分析過的 label，重跑時只補失敗的部分。
- 11:21 用 `P0_STEPS="sglang_serve nsys_vllm nano ncu summarize"` 在同一台 VM 上補跑。

## 2026-10-22 11:19 — 第二次補跑：SGLang、nsys、nano

- **SGLang serve：成功**（587 s）。
  - 用預設的 flashinfer backend，152 秒 ready（包含 JIT 編譯），佔用 35 GB。補上 PATH 就夠了，沒有用到 triton fallback。
  - latency（c=1）：TPOT p50 **12.61 ms**（78 tok/s），比 vLLM 的 12.97 ms 略好；但 TTFT 50 ms，比 vLLM 的 22 ms 高。
  - sweep：c=8 565、c=16 927、c=32 **1,409 tok/s**（最高）。
  - c=64 反而掉到 1,316 tok/s，TPOT p50 40 ms，TTFT p50 2.0 s。推測是 SGLang 預設的 decode CUDA graph `max_bs=32`（見 server_args 的 `cuda_graph_config.decode`），batch 超過 32 就改走 eager。
  - c=128：1,658 tok/s。
- **vLLM nsys：成功**（171 s）。每個 decode step 的分解（nsys 量的 64 個 step 取中位數）：

  | capture | ms/step | GPU busy | GPU idle | kernels/step | launch calls/step | GEMM 佔 kernel 時間 |
  |---|---:|---:|---:|---:|---:|---:|
  | CUDA graph b=1 | 12.85 | 12.57 | 0.28 | 365 | 15 | 91.5% |
  | eager b=1 | 28.74（沒開 nsys 時 22.7） | 12.87 | 15.87 | 369 | 369 | 89.4% |
  | CUDA graph b=64（ctx ≈1152） | 22.69 | 22.62 | 0.07 | 463 | 15 | 65.3%（attention 30.5%） |

  不開 profiler、自己呼叫 `engine.step()` 量到的 decode：b=1 graph 12.62 ms、eager 22.68 ms，b=64 23.03 ms（2,779 tok/s）。
- **nano 計時：成功**。

  | 路徑 | batch | 輸入 | ms/step | tok/s |
  |---|---:|---:|---:|---:|
  | static | 1 | 128 | 44.1 | 22.7 |
  | paged | 1 | 128 | 53.6 | 18.6 |
  | paged | 1 | 1024 | 61.9 | 16.1 |
  | paged | 4 | 1024 | 173.9 | 23.0 |
  | paged | 16 | 1024 | 618.9 | 25.9 |

  scheduler 和 postprocess 本身的 overhead 只有 0.2–0.4 ms/step，時間幾乎全花在 `forward_paged` 裡。
- **nano nsys**：
  - static b=1：每 step 59.4 ms（nsys 下），GPU busy 只有 19.1 ms，idle 40.3 ms；每 step 1,573 個 kernel、1,573 次 launch。
  - paged b=1：69.9 ms，GPU busy 19.9 ms，1,604 個 kernel。
  - GEMM 佔 kernel 時間 73%，elementwise/copy 佔 22–24%（RoPE、repeat_kv、cache 讀寫這些零碎的 PyTorch op）。

## 2026-10-22 11:40 — 兩個 profiling 工具上的效能問題

- **analyze_nsys 太慢**：nano paged b=16 這份 trace 有 270 萬個事件（每個 step 17,619 個 kernel），VM 上的 `analyze_nsys.py` 跑了 13 分鐘以上還沒結束。
  - 原因不在統計（`analyze()` 早就跑完，timeline CSV 已經寫出來），而是畫圖時每個 bar 都各自呼叫一次 `broken_barh`，切片裡有超過 10 萬筆。
  - 修正：同一個 track、同一個 category 只呼叫一次 `broken_barh`；先寫 JSON 再畫圖。step 不超過 15 ms 就畫 3 個 step，比較長的 step 只放大中間 5 ms（800 ms 的全景圖每個像素有 450 µs，什麼都看不出來）。
  - softmax 改歸 attention：這些 run 都是 greedy decode，只有 nano 沒融合的 PyTorch attention 會 launch softmax kernel。
  - 先把 154 MB 的 SQLite 拉回本機，再停掉遠端的分析。之後把其他五份 SQLite 也下載回來，在本機重跑分析，b=16 這份只要 23 秒。
- **ncu 太慢**：vLLM b=1 那組 10 分鐘只錄了 30 個 kernel，每個 kernel 13 個 pass、大約 20 秒。
  - 判斷：vLLM 預設佔走 40 GB 中的 0.85，ncu 每個 pass 前都要備份 kernel 可能寫到的記憶體，GPU 上放不下這份備份，只能走 host。
  - 修正：`vllm_offline.py` 加 `--gpu-memory-utilization`，ncu 那組用 0.45（權重加上一個小的 KV cache）。b=64 的 ncu 拿掉，因為它需要比較大的 KV cache，它的時間分解靠 nsys 就夠了。
  - 停掉遠端那輪；本機 driver 也先停，不讓它把 VM 關掉。
- 12:10 用 `P0_STEPS="ncu summarize" COLAB_KEEP=1` 補跑。

## 2026-10-22 12:09 — 補跑 ncu、彙整、關 VM

- **ncu：成功**（1,027 s）。降到 0.45 之後，大約 2 分鐘就錄了 145 個 kernel，證實了記憶體備份的推測。
  - vLLM b=1 錄到一個完整的 decode step，共 367 個 kernel，kernel 時間合計 13.39 ms，平均 DRAM 1,147 GB/s。
  - nano static b=1 錄了 300 個 kernel。
- **analyze_ncu 欄位名稱**：這版 ncu（2025.3）沒有 `dram__throughput.*` 和 `dram__bytes_read/write.sum`，改用 `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` 和 `dram__bytes.sum.per_second`；時間單位寫 `us`、頻寬寫 `Tbyte/s`。我把解析器改成每個 metric 可以有多個候選名稱，在本機用 raw CSV 重新分析。
- **summarize 修正**：`summarize.py` 排除掉 `*_ncu` 的 offline 計時。那組是開著 ncu 量的，每 step 33 ms，不能算進 batch=1 的表。
- **VM**：12:28 用 `colab_phase0.sh stop` 釋放，`colab status` 顯示 "No active sessions"。這台 VM 從 10:53 開到 12:28，大約 95 分鐘。以之前 smoke 的比例（約 3.5 CU/hr）估算花了 5–6 CU，CLI 查不到確切用量。
- **結果整理**：
  - 報告 74 MB，下載到 `results/20261022-colab-a100-40gb/`，共 285 MB，其中 nsys-rep 和 ncu-rep 佔了大部分。
  - VM 上的 nsys JSON 和時間軸，用本機新版的 `analyze_nsys.py`（softmax 改歸 attention、放大的時間軸）重新產生後覆蓋。nano_paged_b16 只有本機分析的版本。
  - 本機重跑 `roofline.py --gpu A100-SXM4-40GB` 和 `summarize.py`，產生 `summary.md` 和三張圖。
  - `.gitignore` 排除 `*.nsys-rep`、`*.ncu-rep`、`*.sqlite`。SQLite 只留在 scratchpad。
- **write-up**：見 [README.md](README.md)。

### 這次 Phase 0 改過的檔案

- 新增：`benchmarks/phase0/` 整個目錄。
- 修改：`src/serving/nano_vllm/kernels/silu_mul.py`，Triton kernel 改成用 fp32 計算。
- 以上（不含 profiler 原始報告）在 2026-10-22 commit 到 branch `claude/inference-phase0`，從 origin/main 開出來。
