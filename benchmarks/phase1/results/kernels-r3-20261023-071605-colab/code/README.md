# Phase 1：Kernel

幫 nano_vllm 寫 Triton/CUDA kernel，目標是 Llama 3.1 8B Instruct（BF16）在一張 A100 上的 decode。照這個順序做：

1. paged decode attention
2. 融合 RoPE + KV 寫入
3. decode CUDA graph
4. 融合 add + RMSNorm
5. prefill attention
6. bf16 GEMV（先寫 Triton，再改寫成 CUDA）
7. W4A16

每個 milestone 都在 Colab A100 上跑一輪 `tests` → `kbench` → `ncu` → `e2e`，數字和 [Phase 0](../phase0/README.md) 的 baseline 放在一起比。失敗和修正都記在 [LOG.md](LOG.md)。

## 怎麼跑

```bash
BENCH_PHASE=phase1 bash benchmarks/phase0/colab_phase0.sh
```

開一台 Colab A100，上傳程式碼，跑完全部步驟，把結果下載到 `results/<run_id>/`，最後關掉 VM。總表在 `results/<run_id>/summary.md`。

```bash
BENCH_PHASE=phase1 P1_STEPS="tests kbench" COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun
```

開發迴圈：在同一台 VM 上只跑測試和 kernel benchmark。venv、模型和 CUDA toolkit 都留在 VM 上，`COLAB_KEEP=1` 讓 VM 跑完不關。不設 `P1_STEPS` 的話，`rerun` 跑 `tests kbench ncu e2e quality summarize`。

- `P1_KERNELS=rms_norm,linear`：kbench 和 ncu 只跑這幾個 kernel。
- `P1_TUNE=decode,w4a16`：`tune` step 只掃這幾個 kernel（預設全部：decode、prefill、gemv、cuda_gemv、w4a16）。
- `P1_NANO_CONFIGS`：e2e 要跑的 nano 設定，每個是 `label:attention:linear:eager|graph[:weight_quant]`，預設是 milestone 依序疊上去的六組（torch_eager、triton_eager、triton_graph、gemv_graph、cuda_gemv_graph、w4a16_graph）。
- 同一個 `P0_RUN_ID` 重跑時，ncu 和 nsys 已經分析過的 label 會跳過。
- 用完記得 `bash benchmarks/phase0/colab_phase0.sh stop`。

```bash
python benchmarks/phase1/kbench.py --device cpu --out /tmp/kbench
```

在本機跑：只做 torch 實作的正確性檢查，Triton 實作會標成 skipped，大約 80 秒。macOS 沒有 Triton wheel，也沒有 GPU，所以 Triton kernel 和計時都只能在 Colab 上跑。

## Steps（`run_phase1.sh`）

每一步各自寫 `logs/<step>.log`，結果記進 `status.tsv`；某一步失敗不會擋住後面的步驟。

| step | 做什麼 |
|---|---|
| `installs` | vLLM 0.19.1 的 venv（和 Phase 0 共用；VM 上已經有就沿用）加上 pytest、ninja；nsys；`unsloth/Meta-Llama-3.1-8B-Instruct` |
| `cuda129` | apt 安裝和 venv 裡 torch 同版本的 CUDA 12 nvcc、cudart-dev、cccl，寫出 `/content/p0/cuda129.env`（`CUDA_HOME`、`TORCH_CUDA_ARCH_LIST=8.0`、`TORCH_EXTENSIONS_DIR`），後面每一步都會 source。系統的 nvcc 是 13.0，不能拿來編 extension。M6 之前用不到 |
| `tests` | `pytest -q -rs tests/gpu tests/unit/test_nano_vllm_*.py`；只要有一行 `SKIPPED` 就算失敗（和 CI training job 同一條規則）。`tests/gpu` 還沒建立時只跑 unit tests |
| `kbench` | `kbench.py`：每個 kernel 的每個 shape 先對 reference 檢查，再計時，輸出 `kbench/kbench.{json,md}` |
| `ncu` | 每個 kbench kernel 各錄一份 ncu（`--kernel-name regex:<登記的 regex>`，sections 和 Phase 0 相同），再用 Phase 0 的 `analyze_ncu.py` 整理 |
| `tune` | `tune.py`：每個 kernel、每個 shape 掃一遍 launch 設定，和程式目前選的設定並排，輸出 `tune/tune.{json,md}`。不在預設步驟裡 |
| `e2e` | 每組 `P1_NANO_CONFIGS` 各跑一次 `nano_bench.py`（`static:1:128:256,paged:1:128:256,paged:16:1024:256,paged:64:1024:256`）；`vllm_offline.py` 跑同樣的 paged 點；最後一組 nano 設定的 paged b=1 和 b=16 錄 nsys |
| `quality` | `quality.py`：在真的 8B 上，用 wikitext-2 test 的 2048-token 區塊做 teacher forcing，比 torch 路徑、Triton kernel、Triton + W4A16 的 perplexity 和 top-1 一致率 |
| `summarize` | 全部整理成 `summary.md` |

## kbench

每個 kernel 用 `register(Kernel(...))` 登記：

| 欄位 | 內容 |
|---|---|
| `shapes` | shape sweep，每個 shape 是一個 dict |
| `inputs(shape, device)` | 產生一次呼叫的輸入 |
| `impls` | `{名稱: factory(inputs) -> 不帶參數的 callable}`。factory 在計時之外做準備（配置 output、repack 權重）。第一個實作是 baseline |
| `reference(inputs)` | 正確答案，用 fp32 比對 |
| `bytes` 或 `flops` | 一次呼叫要搬的 bytes（memory-bound），或要算的 FLOPs（compute-bound），兩者擇一 |
| `ncu` | ncu step 的 `--kernel-name` regex |

目前登記的 kernel：

- `rms_norm`、`silu_mul`：`kernels/` 現有的 Triton 版對照 torch 版，M = 1、16、64、256、2048。
- `linear`：cuBLAS `F.linear`、Triton GEMV、CUDA GEMV，跑 decode 的五種 shape（qkv 6144×4096、o 4096×4096、gate_up 28672×4096、down 4096×14336、lm_head 128256×4096），M = 1、2、4、8、16。
- `fused_add_rms_norm`、`silu_and_mul`、`rope_cache_write`：M4 和 M2 的融合 kernel 對照 torch 版。
- `paged_decode_attention`：vLLM 的 FA2（`flash_attn_varlen_func` 加 block_table）、torch reference、Triton split-KV，B = 1、16、64、128，ctx = 128、1k、4k。
- `paged_prefill_attention`：vLLM 的 FA2、torch reference、Triton，算 TFLOPs。
- `w4a16_linear`：vLLM 的 `gptq_marlin`、我們的 CUDA GEMV 和 tensor-core 版，以及 cuBLAS 跑反量化後的 bf16 權重；四種 decoder 投影，M = 1–64。所有實作都用同一組 4-bit code（Marlin 的權重由我們的 code 用 vLLM 的 helper 重排），所以 reference 相同。GB/s 一律用 4-bit 權重的 bytes 算，cuBLAS 那列可以讀成「W4A16 要多少頻寬才追得上 bf16」。

**計時方式**：

- 每個實作先用 bf16 容忍度（atol = rtol = 2e-2）對 reference 檢查，通過了才計時。
- 計時用 CUDA graph replay，一個 graph 裡錄至少 100 次呼叫，所以 launch overhead 不會算進去。
- 輸入小於 L2（40 MB）兩倍時，和 `bandwidth.py` 一樣輪替多份輸入，讓 working set 到大約 512 MiB，量到的是 HBM 而不是 L2。
- 沒辦法 capture 的實作（例如有 host sync）會退回一般 launch 迴圈計時，表裡會標出 `eager`。

**評分**：

- memory-bound：GB/s、佔 1,555 GB/s 峰值的比例，以及佔這張卡實測 copy 頻寬的比例。copy 頻寬每次 run 都用 `bandwidth.copy_bandwidth()` 量一次，Phase 0 量到的是 1,380 GB/s。
- compute-bound：TFLOPs 和佔 312 TFLOPs 的比例。

`--ncu-mode` 不計時，也不檢查正確性。每個 shape、每個實作先暖機一次，再在 `cudaProfilerStart/Stop` 之間呼叫 3 次，外面包一個 NVTX range `kernel/impl/shape`。

## 結果

所有數字都來自 Colab A100-SXM4-40GB，模型是 Llama 3.1 8B Instruct。過程（每一輪 run 量到什麼、哪裡失敗、怎麼修）在 [LOG.md](LOG.md)。

### 端到端 decode

ms 是 decode 每個 step 的時間。static 和 paged b=1 的輸入是 128，b=16 和 b=64 的輸入是 1024，輸出都是 256。nano 的設定是逐個 milestone 疊上去的，所以每一列代表「做到這個 milestone 為止」。

| 設定 | 對應 milestone | run | paged b=1 | paged b=16 | paged b=64 | kernels/step（b=1） |
|---|---|---|---:|---:|---:|---:|
| Phase 0 baseline | - | Phase 0 | 53.63 | 618.92 | - | 1,604 |
| torch_eager | M1 engine 重構 | run 3 | 39.60 | 51.12 | 152.52 | |
| triton_eager | + M2 decode attention、RoPE/KV、M4 fusion、M5 prefill | run 3 | 21.56 | 23.00 | 30.17 | |
| triton_graph | + M3 decode CUDA graph | run 3 | 12.61 | 14.87 | 23.48 | |
| gemv_graph | + M6 Triton GEMV | run 4 | 12.32 | **14.63** | 23.77 | 424 |
| cuda_gemv_graph | M6 改用 CUDA GEMV（1–2 列） | run 4 | **12.15** | 14.87 | 23.51 | 328 |
| **vLLM 0.19.1 bf16** | 對照組 | run 2–6 | 12.63 | 14.83 | 23.04 | 365（Phase 0） |
| w4a16_graph | + M7 W4A16 | run 6 | 5.95 | 8.91 | 24.20 | 328 |
| **vLLM 0.19.1 AWQ int4（awq_marlin）** | M7 的對照組 | run 6 | 5.55 | 7.87 | 16.23 | |

- **bf16**：nano 和 vLLM 打平。b=1 快 4%，b=16 快 1%，b=64 慢 2%。e2e 的 run-to-run 差異約 1%（LOG.md，run 4）。
- **M2 目標**（eager 下 b=16 < 60 ms）：23.0 ms，達標。光是 M1 的重構就從 619 降到 51 ms。
- **M3 目標**（b=1 ≤ 13.5 ms）：12.6 ms，達標。
- **M4 目標**（每個 step 約 330 個 kernel）：CUDA GEMV 路徑 328 個，達標；Triton GEMV 多 96 個 split-K reduce kernel，共 424 個。
- **M6 目標**（b=1 約 11 ms）：12.15 ms，沒達標。計畫估的是「GEMV 比 cuBLAS 快很多」，實際上 cuBLAS 在 graph replay 下已經到 76–94% 峰值（Phase 0 用 CUDA event 量到的 63–73% 包含 launch 間隔），GEMV 能贏的空間只有 0–8%。
- **M7**：b=1 時 W4A16 比 bf16 快 2.0 倍，和 vLLM 的 AWQ Marlin 差 7%；b=16 差 13%；b=64 差 49%（見下面的 kernel 表）。
- static path 刻意保留 Phase 0 的寫法（逐層 PyTorch attention、F.linear），所以一直是 40 ms 左右，沒有列出。
- prefill 時間沒有和 vLLM 比：`vllm_offline.py` 的 `prefill_s` 沒有涵蓋整個 prefill（LOG.md，run 2）。

### Kernel

kbench，CUDA graph replay，輸入輪替超過 L2。表裡是 run 5 的結果，所有預設都是 tune 調過的最終版。

| kernel | 對照組 | 目標 | 實測 |
|---|---|---|---|
| paged decode attention | vLLM FA2 split-KV | 每個 (B, ctx) 都 ≥ FA2；B·ctx 大時 ≥ 80% 峰值 | 12 個點裡 11 個 ≥ FA2（1.03–1.44×）；B=1/ctx=128 只有 0.52×（S=32 個 split 只有一個有工作，加上 reduce kernel，每個 token 約多 0.2 ms）。B·ctx 大時 86–93% 峰值 |
| prefill attention | `flash_attn_varlen_func` | 1k–4k 時 ≥ 70% FA2 TFLOPs | 1k–4k：0.80–0.89×（1×4096 是 144 TFLOPs，46% 峰值）；512 時 0.68× |
| bf16 GEMV（M=1） | cuBLAS | 每種 shape ≥ cuBLAS；o_proj、qkv ≥ 80% 峰值 | Triton：gate_up 1.06×、down 1.05×、lm_head 1.01×、qkv 0.99×、o_proj 0.97×。CUDA：1.08×、1.06×、1.01×、1.00×、0.97×。o_proj 73–74%、qkv 78–79% 峰值，都差一點沒達標 |
| W4A16 | vLLM `gptq_marlin`（同一組 4-bit code） | 權重頻寬 ≥ 80% 峰值，和 Marlin 差 10% 以內 | M=1：o_proj 1.33×、qkv 1.06×、gate_up 0.99×、down 0.64×。M=4–16：0.59–1.07×。M=32–64：0.48–0.70×（run 6）。down_proj 和大 batch 沒達標：ncu 顯示我們的 kernel 被記憶體延遲卡住（long_scoreboard），每個 thread 只有一組權重在路上，DRAM 41–52%；Marlin 用多段 `cp.async` pipeline，到 62%（LOG.md，run 7） |
| fused add + RMSNorm、silu_and_mul、RoPE + KV 寫入 | torch | - | 對 torch 快 2.2–7.5 倍。256 和 2048 個 token 時，前兩個到 83–94% 峰值，RoPE + KV 寫入是 42–62% |

### 品質（`quality.py`，wikitext-2 test，288,627 個 token）

| 設定 | 權重 GiB | perplexity | top-1 一致率（對 bf16 torch 路徑） |
|---|---:|---:|---:|
| bf16，PyTorch 路徑 | 15.0 | 7.233 | - |
| bf16，Triton kernel | 15.0 | 7.232 | 98.81% |
| W4A16（RTN，對稱，每 128 個一組） | 5.3 | 8.024 | 85.90% |

計畫要求 bf16 的 top-1 一致率 ≥ 99%，實測 98.81%。perplexity 到小數第三位相同，差異只在 attention（Triton 照 FA2 的做法把 softmax 機率轉成 bf16 再乘 V），不一致的都是接近平手的 token。W4A16 的 RTN 讓 perplexity 增加 11%，和文獻上 Llama 3 8B 的結果同一個量級。

## 檔案

| 檔案 | 內容 |
|---|---|
| `run_phase1.sh` | 主流程。step runner、安裝、nsys、ncu 函式來自 `../phase0/lib.sh` |
| `kbench.py` | kernel micro-benchmark：登記、正確性檢查、CUDA graph 計時、JSON 和 markdown |
| `tune.py` | launch 設定的 sweep，計時沿用 kbench |
| `quality.py` | 8B 上的 perplexity 和 top-1 一致率 |
| `interp/` | 本機用 Docker（linux/arm64）跑 `TRITON_INTERPRET=1` 檢查 Triton kernel 的邏輯：`interp_check.py`，以及證明 interpreter 的 bf16 `tl.dot` 有錯的 `dot_check.py` |
| `summarize.py` | 把一次 run 整理成 `summary.md` |
| `results/<run_id>/` | `summary.md`、`status.tsv`、`logs/`、`tests/`、`kbench/`、`tune/`、`ncu/`、`nsys/`、`offline/`、`quality/`、`environment.txt`，以及當次 run 的程式碼副本 `code/` |

Colab 的 driver 和 launcher 沿用 Phase 0 的 `colab_phase0.sh`、`colab_launcher.py`，用 `BENCH_PHASE=phase1` 切換。tarball 除了 `benchmarks/phase0`、`benchmarks/phase1` 和 `src/serving/nano_vllm`，還會帶 `tests/unit/test_nano_vllm_*.py`、`tiny_llama.py`，以及這些測試在 nano_vllm 以外會 import 或讀取的檔案：`src/monitoring/metrics.py`、`src/api/model_routing.py`、`config/model_routing.nano.json`。`tests/gpu/` 存在的話也會帶上。

## 限制

- 只在 Colab A100-SXM4-40GB 上跑。80GB 卡的峰值會自動改成 2,039 GB/s，其他 GPU 要用 `--peak-gbps` 指定。
- 每個點只量一次：kbench 取 20 次 replay 的中位數，e2e 每組設定跑一次。
- `cuda129` 名稱沿用計畫，但實際裝的是和 venv 裡 torch 相同的 CUDA 版本。Phase 0 記錄到 vLLM 0.19.1 的 torch 是 `2.10.0+cu128`，所以預設會裝 12.8；要強制裝 12.9，設 `P1_CUDA_VERSION=12.9`。
