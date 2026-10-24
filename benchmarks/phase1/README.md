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
| `ncu` | 每個 kbench kernel 各錄一份 ncu（`--kernel-name regex:<登記的 regex>`，sections 和 Phase 0 相同），再用 Phase 0 的 `analyze_ncu.py` 整理。`P1_NCU_SHAPES` 另選 shape，`P1_NCU_SECTIONS` 加錄 section（`SourceCounters` 另外輸出逐指令的 source page） |
| `trace` | `tiled_trace.py`：`gemm_w4a16_tiled` 的 trace build，在冷的權重上拆出每個 block 的時間（prologue、每個 tile、slice 結尾、最後結束的 block），輸出 `trace/trace.{json,md}` 和 Gantt 圖。不在預設步驟裡 |
| `golden` | `golden.py`：第一次把 tiled kernel 對固定輸入的輸出存到 `P1_GOLDEN`，之後的 run 逐 bit 比對。不在預設步驟裡 |
| `tune` | `tune.py`：每個 kernel、每個 shape 掃一遍 launch 設定，和程式目前選的設定並排，輸出 `tune/tune.{json,md}`。不在預設步驟裡 |
| `e2e` | 每組 `P1_NANO_CONFIGS` 各跑一次 `nano_bench.py`（`static:1:128:256,paged:1:128:256,paged:16:1024:256,paged:64:1024:256`）；`vllm_offline.py` 跑同樣的 paged 點；最後一組 nano 設定的 paged b=1、16、64 錄 nsys。最後一組是 W4A16 時，vLLM 的 AWQ 也錄 nsys，再用 `gemm_breakdown.py` 讀出兩邊 decode 裡每個投影的 GEMM 時間 |
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

## Phase 2 之後：Kernel 改進清單第一輪（runs `kernels-r1` 到 `kernels-r4`、`kernels-serve`）

Phase 1 結束時，路線圖 doc 的「Kernel 改進清單」列了 12 項。Phase 2 之後重新評估，這一輪做了 5 項，其餘的改列下一輪或不做。分支 `claude/inference-kernels`，只在本機。

| # | 項目 | 結果 |
|---|---|---|
| 10 | async scheduling 改成預設 | 完成。W4A16 也省 0.23 / 0.42 / 1.01 ms（b=1/16/64） |
| 12 | sync guard 擴到 sampled、eager、mixed | 完成。GPU 上沒有抓到任何同步 |
| 5 | 採樣路徑的量測 | 完成量測。b=64 時 nano 的採樣比 vLLM 貴 0.96 ms，不排序的採樣器列為下一輪 |
| 1 | W4A16 GEMV 暫存器預取 | 負面結果，revert。預取越深越慢 |
| 3 | FP8 KV cache | 完成，選用（`kv_cache_dtype: fp8`）。b=64 快 1.7 ms，perplexity +0.2% |
| 2、6、7 | Marlin 式 W4A16 和它的 epilogue/prologue 融合 | 下一輪。W4A16 在 b=64 仍落後 AWQ 44%（#2 第二輪已完成，見下一節） |
| 4、8、9、11 | reduce 併進 split kernel、prefill tile 順序、bf16 CUDA GEMV、CUDA event 計時 | 不做：收益太小、TTFT 已贏 vLLM、不在預設路徑、async 之後用不到 |

### 端到端 decode（ms/step，b=1 / 16 / 64；b=16、4096 token）

| 設定 | b=1 | b=16 | b=64 | b=16，4096 |
|---|---:|---:|---:|---:|
| nano bf16（async scheduling，現在的預設） | 12.05 | 14.13 | 22.60 | 18.73 |
| nano bf16 + fp8 KV | **12.01** | **13.81** | **20.87** | 17.16 |
| vLLM bf16（FA2） | 12.60 | 14.82 | 23.04 | - |
| vLLM bf16 + fp8 KV（FLASHINFER） | 13.05 | 14.50 | 21.30 | - |
| nano W4A16 | 5.68 | 8.48 | 23.27 | 13.01 |
| nano W4A16 + fp8 KV | **5.65** | 8.25 | 21.52 | 11.60 |
| vLLM AWQ（Marlin） | 5.56 | 7.86 | 16.19 | - |
| vLLM AWQ + fp8 KV（FLASHINFER） | 6.05 | **7.55** | **14.44** | - |

run `kernels-r4`（同一台 VM、同一輪）。vLLM 的 4096 點失敗了（`vllm_offline.py` 的 `max_model_len` 寫死 4096，已修）。

### FP8 KV cache

- **做法**（`kernels/fp8.py`）：Triton 3.6 的 CUDA backend 在 sm80 不接受 `float8e4nv`，A100 也沒有 fp8 轉換指令，所以 kernel 把 cache 當 uint8，用整數運算處理 e4m3：
  - **解碼**：把 7 個 exponent 和 mantissa bit 放到 bf16 的 bit 4–10，得到「原值 × 2^-120」，直接進 `tl.dot`。
  - **scale**：每個 (token, kv head) 一個 fp32 scale（amax / 448）。K 的 scale 乘在 score 上，V 的 scale 乘在機率上，不做逐元素反量化。q 先乘 2^60，輸出最後乘 2^60，中間值都遠離 fp32 的 subnormal。
  - **容量**：每 128 個值佔 132 bytes，bf16 是 256，自動 KV sizing 會給出約 1.9 倍的 block。
- **vLLM 0.19.1 在 A100 上的 fp8 KV**：FLASH_ATTN 直接拒絕；TRITON_ATTN 撞到同樣的 `fp8e4nv` 限制，編譯失敗；只有 FLASHINFER 能跑，但它在 b=1 比 vLLM 自己的 FA2 慢 0.45 ms。
- **kernel**（decode attention，µs，bf16 / fp8；fp8 用自己調過的 launch 級距）：

  | | B=1 | B=16 | B=64 | B=128 |
  |---|---:|---:|---:|---:|
  | ctx=1024 | 14.4 / 12.0 | 62.3 / 53.9 | 201.0 / 147.3 | 379.9 / 273.2 |
  | ctx=4096 | 20.9 / 19.8 | 204.6 / 156.1 | 770.2 / 541.6 | 1484.9 / 1033.8 |

  bytes 少了將近一半，時間少 4–30%。fp8 只到 HBM 峰值的 60–69%（bf16 是 86–93%），整數解碼成了新的上限。
- **品質**（wikitext-2 test，288,627 個 token）：bf16 的 perplexity 從 7.232 變成 7.248（+0.22%），top-1 一致率對 bf16 是 97.55%；W4A16 從 8.024 變成 8.040。
- **線上**（run `kernels-serve`，Phase 2 的 harness，每個 server 依 0.9 的 GPU 記憶體自己配置 KV）：
  - **容量**：nano 從 157,744 增加到 305,920 個 token（1.94 倍），vLLM 從 159,072 增加到 326,352。
  - **W2b**：48 個 4096-token 的 prefix，總長超過 bf16 的容量。fp8 讓兩邊的命中率都從約 48% 升到 70.6%。nano 的 req/s 從 3.33 升到 3.99，TTFT p50 從 405 降到 160 ms；vLLM fp8 是 4.16 req/s、259 ms。
  - **W1 c=64**：ITL p50 依序是 nano fp8 21.00、vLLM fp8 21.38、nano 22.65、vLLM 23.24 ms。
- **門檻 G4**：perplexity 增加 ≤ 1%，通過；top-1 一致率 ≥ 98%，差 0.45 個百分點（bf16 的 Triton 和 torch 路徑之間，光是數值差異就只有 98.81%）；b=64 每步省 2 ms，實測 1.73 ms，也差一點。兩項都只是小幅未達，所以保留成選用，預設維持 `auto`，和 vLLM 一樣。

### 負面結果：GEMV 暫存器預取

假設是「每個 warp 一輪只發一次 load，延遲露出來」。實作成 register ring，每輪發出後面 1、2 或 4 輪的 load，結果每個深度都更慢：down_proj M=1 在深度 0 / 1 / 2 / 4 分別是 37.7 / 39.2 / 39.4 / 48.3 µs，e2e 的 W4A16 b=1 在深度 0 / 2 / 4 是 5.74 / 6.00 / 6.32 ms。ring 吃暫存器、降低 occupancy；深度 0 的 achieved occupancy 已經有 30%，延遲本來就由其他 warp 蓋住。e53207f revert。Marlin 快的原因不只是預取，而是常駐 grid、`cp.async` 多段 pipeline 和條狀切分一起作用，這要整個重寫（下一輪）。

### 採樣路徑

所有 benchmark 之前都是 greedy，但 server 的預設是 temperature 0.7、top-p 0.9：

| sampled 減 greedy（ms/step） | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano | 0.42 | 1.02 | 2.62 |
| vLLM | 0.25 | 1.33 | 1.66 |

nsys（b=16，每步）：`torch.cumsum` 沿 128k vocab 的 innermost scan 要 231 µs，radix sort 約 265 µs，兩次 argmax 共 96 µs（greedy 的 bf16 argmax 每步都算，一列只用一個 block）。線上的 W1s（c=1/16/64）nano 是 78 / 855 / 1,711 tok/s，vLLM 75 / 609 / 1,693；c=64 的 ITL p50 是 25.31 對 25.03 ms，nano 慢 1.1%。下一輪改用不排序的採樣器（FlashInfer 的 rejection sampling，或 DeepInfer 的 radix select）。

## Kernel 改進清單第二輪：Marlin 式 W4A16（runs `marlin-t1` 到 `marlin-t2`）

第一輪結束時，W4A16 在 b=64 還比 vLLM 的 AWQ 慢 44%（23.27 對 16.19 ms），kbench 也顯示差距幾乎都在 GEMM。這一輪自己寫了一個 Marlin 設計的 kernel `gemm_w4a16_tiled`（`kernels/csrc/w4a16.cu`），沒有用 vLLM 的程式碼；預設 layout 改成 tiled。分支同樣是 `claude/inference-kernels`，只在本機。

### 設計

- **tiled layout**（`kernels/w4a16.py`，載入時重排）：
  - qweight 是 int32 `[K/16, N/64, 32, 4]`。每 512 bytes 是 mma.m16n8k16 一個 16 × 64 tile 的 B operand；一個 lane 用一次 16-byte 的 shared load，就拿到 8 個 n8 tile 的 b0/b1。
  - scales 排成 `[K/128, N/64, 8, 8]`，一個 lane 的 8 欄 scale 放在一起。
  - 存成 4 維，舊 kernel 的形狀檢查會拒絕它。
- **pipeline**：
  - 4 段 cp.async，每段放 x tile（swizzle 後用 ldmatrix 讀）、權重 tile 和 scale；
  - 一個 n64 欄的幾個 warp 分攤這一段的 k16 列；
  - 反量化後先乘 scale（只 round 一次），再進 mma。
- **persistent striped grid**（Marlin 的切法）：
  - 所有 (欄 slice, k tile) 依序編號，每個 block 拿連續的一段，不論 shape 都平均；
  - loader 用全域 tile 編號定址，預取跨過 slice 邊界也不會清空 pipeline（Marlin 會清空再重填）。
- **跨 block 相加**：
  - 共用一個 slice 的 block，各自把 fp32 部分和寫進 slot，再 atomicAdd 這個 slice 的 counter；
  - 最後到的 block 依 block 順序加總、寫出結果；
  - 結果 deterministic，也沒有 block 要等別的 block（Marlin 是串行的 lock 鏈）。
  - 同步的 W4A16 不再需要 `splits=1` 才能逐 token 重現。
- **超過 64 列**：每 64 列 launch 一次，最多到 `tiled_max_rows`（128）；再多就用 tiled 的反量化 kernel 加 F.linear。tiled 反量化和 rows 的結果 bitwise 相同，速度也一樣（78–80% 峰值）。

### 怎麼追上 Marlin

| run | 改了什麼 | 四層加總對 Marlin（M=1 / 16 / 64） |
|---|---|---|
| t1 | 第一版 | 1.21 / 1.15 / 1.11 |
| t1c | 位址遞增、fragment double buffer（跨 k16 步和跨 stage）、256x128 tile | 1.03 / 0.98 / 1.11 |
| t1d | MT≥3 用 64x128 tile、slot 相加的 fence 減到一次 | 1.02 / 0.98 / 1.04 |
| t1e | 反量化的 AND 和 OR 合成一個 `lop3` | **0.96 / 0.96 / 0.98** |

- **ncu（t1b）**：
  - M=1 不是頻寬限制而是指令限制：同樣的工作，第一版發出的指令多約 45%；
  - 從 base clock 換到 boost clock，Marlin 快了 43%，我們只快 6%。（第三輪更正：ncu 一直是 `--clock-control none`，也就是 boost clock；這 43% 是 ncu 每次清 cache 的冷權重，換成 kbench 裡留在 L2 的暖權重的差別，見第三輪。）
- **nvdisasm（t1d）**：
  - nvcc 把 `(w >> 4s) & 0x000F000F | magic` 編成兩個 LOP3，每對 code 要 5 個指令；
  - 改用 inline `lop3.b32 …, 0xEA` 之後是 4 個；
  - 光這一項，M=1 的加總就從 121 降到 115 µs。
- **tune（t1c）**：
  - M ≤ 32 用 256x128（每次 barrier 搬 16 KB 權重）；
  - M=64 用 64x128，N ≥ 16384 時（gate_up）用 64x256；
  - 每個 SM 2 個 block 一律比較慢。

### kernel（kbench，run `marlin-t2-20261023-104627-colab`，µs，tiled / vLLM `gptq_marlin`，同一組 4-bit code）

| M | qkv | o_proj | gate_up | down | 加總 | 加總，舊 kernel |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 17.5 / 19.1 | 15.8 / 17.4 | 52.4 / 59.1 | 28.8 / 23.3 | **114.5** / 118.9 | 126.3 |
| 16 | 18.2 / 21.2 | 16.1 / 19.6 | 62.6 / 65.3 | 31.0 / 27.5 | **127.9** / 133.6 | 149.9 |
| 32 | 23.0 / 27.5 | 20.6 / 22.2 | 72.1 / 72.1 | 41.4 / 37.4 | **157.1** / 159.2 | 258.6 |
| 64 | 40.1 / 45.2 | 33.5 / 35.9 | 105.5 / 113.6 | 73.1 / 63.1 | **252.2** / 257.8 | 449.9 |

down_proj（N=4096、K=14336）在每個 M 都還是 Marlin 的 1.1–1.3 倍，其他三層都追平或更快。（第三輪更正：kbench 量 down_proj 時只有一份權重，留在 L2 裡；冷的時候 down 在每個 M 都比 Marlin 快，見第三輪。）

### 端到端 decode（run `marlin-t2-20261023-104627-colab`，ms/step）

| 設定 | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 tiled（新預設） | **5.35** | **7.31** | 16.82 |
| nano W4A16 rows（舊 kernel） | 5.70 | 8.29 | 23.03 |
| vLLM AWQ（awq_marlin） | 5.56 | 7.85 | **16.23** |
| nano bf16 | 12.06 | 14.12 | 22.61 |
| vLLM bf16 | 12.61 | 14.84 | 23.07 |

- b=1 和 b=16 比 vLLM AWQ 快 4% 和 7%；b=64 從慢 42% 縮小到慢 4%。
- b=64 剩下的 0.59 ms 不在 kbench 量的 GEMM 上：四層加總 252 對 258 µs。拿 bf16 的 e2e 扣掉 cuBLAS 的時間，vLLM 的 AWQ GEMM 在 e2e 裡每層約 224 µs，比 kbench 的 `gptq_marlin` 少，原因沒查。
- nsys（b=1 / b=16）：GPU 空檔 0.04 ms，每步 330 個 kernel，GEMM 佔 84% / 66%。

### 門檻

- **G1 正確性**：通過。
  - GPU 測試 629 passed，沒有 skip，包含 tiled 的排程、fp16、CUDA graph，以及 model、graph、async 測試在 tiled 下跑；
  - 同一個 config 連跑 bitwise 相同；
  - tiled 和 rows 的反量化 bitwise 相同，所以 prefill 和 `quality.py`（teacher forcing 走反量化）的 perplexity 不變，這一輪沒有重跑 quality。
- **G2 kernel**：
  - 四層加總在 M=1、16、64 都 ≤ Marlin 的 1.05 倍，通過；
  - 單點不慢於 1.25 倍：down_proj M=8 是 1.29 倍，沒過。
- **G3 e2e**：b=64 ≤ 17.5、b=16 ≤ 8.1、b=1 ≤ 5.68 ms，三項都過，所以預設 layout 改成 tiled。

## Kernel 改進清單第三輪：down_proj（runs `down-d1` 到 `down-d3`）

第二輪結束時 kbench 顯示 down_proj 在每個 M 都是 Marlin 的 1.10–1.29 倍。這一輪先查原因，結果大部分是量測的問題：冷的權重下 down 在 M ≤ 32 已經比 Marlin 快，真正的差距只在 M=64（b=64 decode）。修正之後，W4A16 的 decode 在 b=1、16、64 都比 vLLM 的 AWQ 快。分支同樣是 `claude/inference-kernels`，只在本機。

### 量測的修正

- **kbench 讓 down_proj 的權重留在 L2**：`copies_for` 依輸入 tensor 的大小決定輪替幾份權重，輸入裡是 bf16 的權重（down 117 MB），超過 2 倍 L2，於是 `copies=1`，100 次呼叫一直讀同一份 29 MB 的 4-bit 權重，大部分留在 40 MB 的 L2。
  - 現在 `Kernel.rotate_bytes` 給出每次呼叫實際讀的 bytes，W4A16 依 4-bit 權重輪替（down 18 份、gate_up 9 份）。
  - gate_up 的 4-bit 權重 59 MB 比 L2 大，第二輪本來就近乎冷的；qkv、o_proj 本來就有輪替；`tune.py` 一直依 4-bit 大小輪替。
  - Marlin 暖的時候快 43%（33.3 → 23.3 µs），我們只快 4–7%，所以第二輪的 down 差距幾乎全是這個。
- **ncu 是 boost clock**：`ncu_run` 從一開始就用 `--clock-control none`。第二輪寫的「base clock 換到 boost clock，Marlin 快 43%」其實是冷權重換成暖權重（上一節和 LOG 已加註）。
- **三種量法對 Marlin 的結論不同**：kbench（冷、CUDA graph 連續呼叫）在 M ≥ 16 量到的 Marlin 比 decode 裡的 AWQ Marlin 慢 15–23%（down M=64：81.5 對 66.0 µs），ncu（單次、清 cache）和 decode 一致（68 µs），我們的 kernel 三種量法都一樣。所以和 Marlin 的比較以 decode 裡的時間為準：e2e 時 nano 和 vLLM 的 AWQ 都錄 nsys，`gemm_breakdown.py` 依位置認出每個投影（SiLU 之後的 GEMM 是 down_proj，前三個是 gate_up、o_proj、qkv）。

### 工具

- **trace build**：`gemm_w4a16_tiled` 加 `template <bool kTrace>`，一般版的 SASS 不變（和改之前的 build 逐指令比對過）。thread 0 把 `clock64` 時間戳寫在 shared memory，結束時才寫回：入口、第一個 tile 到齊、每次 slice 結尾的四個點、出口，加上 globaltimer、最後到的 block 等旗標和一個 atomic 取的結束順序。
- **觀察**：
  - globaltimer 每 1,024 ns 才跳一次，所以 block 內用 clock，跨 block 用結束順序。
  - 同一個 VM 上，M ≥ 8 的 trace run 大多只有約 1.08 GHz，M=1 是 1.3–1.4 GHz；M=1 和 M=8 是同一個 kernel、同樣的指令，差別只在 x 有幾列不是 0，看起來是 400 W 的功耗上限。跨 run 的比較只看同 clock 的列，版本之間的比較用 kbench 和 e2e。
  - 每次 GEMM 的關鍵路徑上，prologue（第一批 tile 從冷的 DRAM 到）約 3–4 µs，slice 結尾約 4–6 µs；穩態每個 tile 的時間和它分到的 DRAM 時間差不多。

### 做了什麼

| 改動 | commit | 結果 |
|---|---|---|
| stage 對齊 128 bytes | 17a3038 | static 的 `__shared__ int last_block` 把 dynamic shared memory 推到位址 16，每條權重 `cp.async` 的 L2 sector 是理想的 2 倍（ncu SourceCounters：50% excessive）。改放 dynamic allocation 尾端之後，down M=1 從 1.95M 降到 0.98M 個 sector，和 Marlin 一樣；kbench 四層加總 −1.2 到 −9.2%。輸出 bitwise 相同 |
| slice 結尾 | 7a0072e | thread 0 一開始就 `ld.acquire` 讀 counter，看到其他 block 都到了就不寫 slot、不做 atomic（down 的最後到達有 59–81% 走這條）；其餘用一個 `atom.acq_rel`；block 最後一次 slice 結尾在已排空的 stage 裡同時寫各 warp 的部分和。四層加總 −0.3 到 −2.7%，bitwise 相同 |
| 平行的 warp 相加 | 7e30049、8537411 | 放得下時前 n−1 個 warp 同時寫自己的 buffer，最後一個依序加上自己的。MT=4 時 warp 相加從 2.2 降到 1.0 µs，但主迴圈每個 tile 慢 5%（255 個暫存器），down M=64 72 → 81 µs，所以只留給 MT 1–2（約 −1%） |
| 33–64 列用 128x128 tile | de05b0f、eb6b2a0 | 每個 warp 每個 stage 算兩個 k16 列（64x128 是一個），每 byte 的 slice 結尾減半。tune（冷）M=64：qkv 38.2 → 34.7、o_proj 31.6 → 27.9、down 72.0 → 62.4 µs；M=48 down 62.9 → 52.9。gate_up 留在 64x256 |

沒做的兩項：
- **下一個 GEMM 的 L2 預取（F5）**：`tiled_trace.py --prefetch-tiles 3` 先把每個 block 前 3 個 tile 拉進 L2 再量，短的 GEMM（qkv、o_proj 的 M=1）約快 2 µs，gate_up、down 只快 0–1 µs，因為 prologue 時 DRAM 本來就在搬所有 block 的前 3 個 tile；要改 model 的呼叫介面，對 down 沒有幫助，所以不做。
- **`mma_trans`（F2）**：boost clock 下穩態每個 tile 不比 DRAM 慢，所以沒做。

### kernel（kbench，冷，run `down-d3`，µs，tiled / Marlin）

| M | qkv | o_proj | gate_up | down | 加總 | 加總，第三輪開始時（`down-d1`） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16.2 / 18.9 | 14.5 / 17.4 | 51.6 / 59.4 | 30.1 / 33.5 | **112.4** / 129.2 | 116.0 |
| 16 | 16.0 / 21.3 | 15.0 / 19.5 | 54.5 / 65.3 | 31.5 / 36.9 | **117.0** / 143.0 | 125.3 |
| 32 | 19.7 / 27.3 | 18.5 / 21.8 | 65.5 / 71.8 | 38.2 / 48.3 | **141.9** / 169.2 | 157.0 |
| 64 | 34.6 / 44.9 | 27.9 / 35.6 | 101.6 / 111.1 | 62.4 / 81.5 | **226.5** / 273.1 | 253.2 |

（kbench 的 Marlin 在 M ≥ 16 偏慢，見上面；和 Marlin 的比較看下一張表。）

### decode 裡每個投影的 GEMM（nsys，run `down-d3`，µs，nano / vLLM AWQ）

| b | qkv | o_proj | gate_up | down | 每層 |
|---|---:|---:|---:|---:|---:|
| 1 | 16.6 / 16.4 | 14.8 / 14.6 | 52.3 / 55.6 | **30.4** / 31.0 | **114.2** / 117.6 |
| 16 | 16.4 / 17.8 | 15.2 / 16.3 | 55.0 / 58.7 | **31.7** / 33.3 | **118.3** / 126.0 |
| 64 | 34.6 / 34.8 | 28.3 / 30.4 | 102.0 / 89.6 | **62.0** / 66.0 | 227.0 / 220.8 |

- down_proj 在 b=1、16、64 是 vLLM 的 0.98、0.95、0.94 倍；第三輪開始時 b=64 是 1.09 倍（71.7 對 65.9，run `down-d2b`）。
- b=64 只剩 gate_up 慢（102.0 對 89.6 µs，每步約 0.4 ms）。

### 端到端 decode（run `down-d3`，ms/step）

| 設定 | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 tiled（第三輪） | **5.19** | **7.10** | **15.94** |
| nano W4A16 tiled（第二輪，`marlin-t2`） | 5.35 | 7.31 | 16.82 |
| nano W4A16 rows（舊 kernel） | 5.68 | 8.29 | 22.94 |
| vLLM AWQ（awq_marlin） | 5.55 | 7.85 | 16.19 |
| nano bf16 | 12.04 | 14.12 | 22.57 |
| vLLM bf16 | 12.64 | 14.84 | 23.04 |

W4A16 在 b=1、16、64 分別比 vLLM AWQ 快 6.5%、9.5%、1.5%；第二輪 b=64 還慢 4%。

### 門檻

- **G0 量測**：冷的 down 在 kbench 是 Marlin 的 0.85–0.94 倍（`down-d1`），decode 裡 b=1/16 是 0.99/0.96 倍、b=64 是 1.09 倍（`down-d2b`）。第二輪 G2 的 down 單點（M=8 的 1.29 倍）改用冷的數字判定：通過。
- **G1 正確性**：
  - GPU 測試 645 passed，沒有 skip；
  - 對齊、slice 結尾、平行相加之後，golden 16/16 bitwise 相同（`down-d2a` 到 `down-d2d`）；
  - 128x128 的預設改了 M=64 三個 shape 的加總順序：`down-d3` 的 golden 13/16 相同，不同的是 qkv、o_proj、down 的 M=64，最大差 0.0078（bf16 在 1–2 附近的 1 ulp），每個 64×N 的輸出裡 51–148 個元素；trace 版和一般版 bitwise 相同。
- **G2 kernel（冷）**：
  - down 在 M=1、16 比 `down-d1` 快 ≥ 10%：沒過（−2%、−4%）；M=64 快 15%；
  - down 在每個 M 都 ≤ Marlin 的 1.05 倍：通過（kbench 和 decode 都是）；
  - 四層加總在任何 M 都不比 `down-d1` 慢：通過（−3.1 到 −10.5%）。
- **G3 e2e**：沒有退步，b=1 5.19 ms ≤ 5.20：通過。

### 還沒做

- gate_up 在 M=64：102 對 89.6 µs，是 b=64 剩下最大的差距。
- 功耗上限下 M ≥ 8 的 SM clock 約 1.08 GHz（trace），e2e 時是多少沒量。
- kbench 量 Marlin 在 M ≥ 16 比 ncu、decode 慢的原因沒查。

## 大 batch：b=128、256（run `concur-c1`）

第三輪之後在 b=128、256 量 decode。W4A16 在 b=64 和 vLLM AWQ 打平，過了 64 列差距隨 batch 變大，而且全部來自 GEMM：attention 兩邊一樣。

- **跑法**：nano 只跑 W4A16 tiled（async），兩種 graph：
  - 預設：decode graph 到 64 列，再大的 batch 用 piecewise graph，attention 在 graph 外；
  - `fullgraph`（`nano_bench.py --full-graphs`）：decode graph 到最大的 batch，和 vLLM 一樣。
- **context**：b=256 用 512 個 prompt token（加 256 個輸出），因為 1k 的 KV 放不下 40 GB；b=128 兩種長度都跑。
- **nano 的 GEMM 路徑**：
  - 65–128 列：tiled kernel 每 64 列 launch 一次，每次都把整份權重讀一遍；
  - 超過 `tiled_max_rows`（128）：每個投影先把整份權重反量化成 bf16，再呼叫 cuBLAS。

### 端到端 decode（ms/step，tok/s）

| b / prompt | nano 預設 | nano fullgraph | vLLM AWQ | nano / vLLM | tok/s：nano / vLLM |
|---|---:|---:|---:|---:|---:|
| 64 / 1024 | 16.02 | 16.04 | 16.20 | 0.99 | 3,989 / 3,951 |
| 128 / 1024 | 30.57 | 30.20 | 27.45 | 1.10 | 4,239 / 4,663 |
| 128 / 512 | 24.57 | 24.21 | 21.42 | 1.13 | 5,286 / 5,974 |
| 256 / 512 | 50.44 | 49.99 | 39.04 | 1.28 | 5,121 / 6,558 |

- **graph**：fullgraph 只快 0.1–1.6%。
- **吞吐量**：nano 從 b=128 到 b=256 反而略降（5,286 → 5,121 tok/s），vLLM 多 10%。
- **vLLM bf16 不能比**：KV cache 只有 142–144k token，b=128/1k（164k）和 b=256/512（196k）都放不下，會 preempt，所以這兩點沒列。AWQ 有 221–223k，兩點都放得下。
- **prefill_s 不能比**：vLLM 先用同一批 prompt 暖機，正式量測時命中 prefix cache。

### decode 裡每個投影的 GEMM（nsys，µs，nano fullgraph / vLLM AWQ）

| b | qkv | o_proj | gate_up | down | 每層 | 每步（×32） |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 34.7 / 34.8 | 28.3 / 30.5 | 102.1 / 89.5 | 62.0 / 66.5 | 227.1 / 221.4 | 7.3 / 7.1 ms |
| 128 | 69.1 / 49.8 | 56.7 / 41.6 | 203.7 / 167.1 | 124.7 / 94.7 | 454.2 / 353.1 | 14.5 / 11.3 ms |
| 256 | 110.7 / 80.7 | 78.5 / 59.0 | 556.8 / 326.6 | 262.1 / 166.1 | 1008.0 / 632.4 | 32.3 / 20.2 ms |

（`gemm_breakdown.py` 把同一個投影連續的幾次 launch 加成一次呼叫；超過 128 列時反量化 kernel 算進後面的 cuBLAS。）

- **差距的來源**：
  - b=128：GEMM 多 3.2 ms/步，e2e 差 2.75 ms；
  - b=256：GEMM 多 12.0 ms/步，e2e 差 11.0 ms。
- **attention 一樣**：nano 的 `_paged_decode_split_kernel` 在 b=256/512 每層 416 µs，vLLM 的 `flash_fwd_splitkv_kernel` 419 µs。
- **b=128**：nano 正好是 b=64 的 2 倍（兩次 launch，權重讀兩遍）；vLLM 的 Marlin 一次 launch 處理所有 64 列一組的 M 區塊，只要 1.6 倍。
- **b=256**：
  - 反量化每層 4 次、平均 104 µs，每層 418 µs、每步 13.4 ms，全是多出來的；
  - cuBLAS 本身每層約 590 µs，已經比 Marlin 的 632 µs 快。
  - `down-d2e` 的 tune 量過 M=256 時改用 4 次 tiled launch：gate_up 416 對 554 µs、down 249 對 258 µs 比較快，qkv、o_proj 還是反量化比較快。
- **離上限還遠**：M=256 的 down 是 30 GFLOP，A100 實際約 280 TFLOPS 時要 107 µs，權重讀一次 21 µs；Marlin 是 166 µs，nano 是 262 µs。

### 能做的

1. **一次 launch 處理 65–256 列**：
   - 做法：tiled kernel 的 persistent 排程加一個 M 區塊的維度，同一個權重 tile 給不同 M 區塊的 block 同時處理，第二次以後從 L2 讀；
   - 預期：b=128 從 2 倍降到 Marlin 的 1.6 倍左右，b=256 也不必反量化。
   - 這是最大的一項。
2. **權重 tile 在一個 block 裡用給 128 列**：B fragment 解一次、用兩次。MT=4 已經用滿 255 個暫存器，要把 warp 分成 M 和 K 兩組。
3. **暫時的做法**：超過 128 列時依投影選路徑，gate_up 和 down 用 tiled，qkv 和 o_proj 用反量化。依 `down-d2e` 的 tune，b=256 每層約省 160 µs、每步約 5 ms，但仍比 vLLM 慢。

第 1 項在下一節做了（第 3 項因此不需要）；第 2 項沒做，因為每個 tile 的指令數才是主因。

## Kernel 改進清單第四輪：大 batch（runs `chunk-g0` 到 `chunk-c7`）

目標是上一節的 b=128、256。三個改動之後，W4A16 的 decode 在 b=1 到 256 都比 vLLM 的 AWQ 快 6–9%：b=256 從 1.28 倍變成 0.93 倍，b=64 也從 15.94 降到 15.03 ms。分支同樣是 `claude/inference-kernels`，只在本機。

### 做了什麼

| 改動 | commit | 結果 |
|---|---|---|
| 一次 launch 處理多個列區塊 | d43d122、c655d29 | 超過 64 列時，列分成 16 MT 列一組的 chunk（Marlin 的 `par`）：chunk c 拿 grid / chunks 個 block，各自把整份權重分 stripe，有自己的 slot 和 counter。每個 chunk 的第 b 個 block 同時走同樣的 tile，權重從 DRAM 讀一次，其他 chunk 從 L2 讀。列數平均分（96 列是兩個 48 列）。`tiled_max_rows` 從 128 改成 256，65–256 列不再反量化；3 個以上 chunk 用 64x256 tile。b=256 的 GEMM 每層 1008 → 723 µs（`chunk-c3`） |
| 較便宜的 load stream | 421dcda | ncu 的 SourceCounters 顯示我們執行的指令裡 HMMA 只佔 14%（Marlin 24%）。SASS 裡 x tile 每次 cp.async 前有約 20 個指令的索引運算：runtime 上限的迴圈、有號除法、swizzle、64-bit 位址。改成每個 thread 固定複製同一個 chunk、相隔固定列數的幾列，位址只算一次；x 和權重指標每個 stage 前進一個 k tile；下一個 stage 的 fragment 無條件讀取，省掉 barrier 前約 20 個暫存器複製。MT 4、128x128 的主迴圈從 497 個指令（64 個 HMMA）降到 335 個。kbench 四層加總：M=64 −11.5%、M=128 −17.1%、M=256 −21.0% |
| 最後一次 slice 結尾移到主迴圈之後 | 8bcb99f | block 的最後一次 slice 結尾在主迴圈之後執行，slot 每次讀 4 個也不會擠到主迴圈的暫存器。在迴圈裡這樣做會讓 MT 3–4 慢 2–4%（`chunk-c5`）。kbench：M=64 −2.2%、M=128 −1.7% |

三項都不改加總順序：64 列以內的輸出和改動前 bitwise 相同（golden 16/16，`chunk-g0` 存、`chunk-c2` 到 `chunk-c7` 檢查）。

### 端到端 decode（run `chunk-c7`，ms/step）

| b / prompt | nano W4A16 tiled | vLLM AWQ | nano / vLLM | tok/s：nano / vLLM | 第四輪之前的 nano |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | **5.24** | 5.55 | 0.94 | 191 / 180 | 5.19（`down-d3`） |
| 16 / 1024 | **7.11** | 7.85 | 0.91 | 2,249 / 2,039 | 7.10（`down-d3`） |
| 64 / 1024 | **15.03** | 16.18 | 0.93 | 4,257 / 3,956 | 15.94（`down-d3`） |
| 128 / 1024 | **25.83** | 27.48 | 0.94 | 4,955 / 4,658 | 30.20（`concur-c1`） |
| 128 / 512 | **19.86** | 21.43 | 0.93 | 6,447 / 5,973 | 24.21（`concur-c1`） |
| 256 / 512 | **36.14** | 39.05 | 0.93 | 7,083 / 6,555 | 49.99（`concur-c1`） |

- nano 用 `fullgraph`（decode CUDA graph 到 256 列），和 vLLM 一樣；預設的 piecewise graph 約慢 1%（`concur-c1`）。
- vLLM bf16 在 b=128/1k、256/512 會 preempt，不列（上一節）。

### decode 裡每個投影的 GEMM（nsys，run `chunk-c7`，µs，nano / vLLM AWQ）

| b | qkv | o_proj | gate_up | down | 每層 |
|---|---:|---:|---:|---:|---:|
| 1 | 16.1 / 16.4 | 14.8 / 14.6 | 51.5 / 55.6 | 30.0 / 31.0 | **112.4** / 117.6 |
| 16 | 16.3 / 17.8 | 15.3 / 16.3 | 55.0 / 58.7 | 32.0 / 33.2 | **118.5** / 126.0 |
| 64 | 31.7 / 34.9 | 27.5 / 30.3 | 85.2 / 89.4 | 53.7 / 65.9 | **198.2** / 220.5 |
| 128 | 45.2 / 49.5 | 35.9 / 41.5 | 155.1 / 165.9 | 84.2 / 94.1 | **320.4** / 351.0 |
| 256 | 74.3 / 80.1 | 56.6 / 58.4 | 291.4 / 322.8 | 149.0 / 164.3 | **571.4** / 625.5 |

- 第三輪剩下的 gate_up M=64（102.0 對 89.6）現在是 85.2 對 89.4。
- 第四輪之前 b=128、256 每層是 454.2、1008.0 µs（`concur-c1`）。

### 量測上的教訓

- **Colab 的 rerun 可能載入舊的 kernel**：tarball 保留本機的 mtime，比 VM 上前一次 build 早的話 ninja 不會重 build（`chunk-c2` 第一次，68 個測試失敗才發現）。`colab_launcher.py` 現在解開後把 mtime 設成現在。
- **trace 的 SM clock 每次不同**（1.08–1.41 GHz），每個 tile 的時間要換成 cycle 才能比。換算後，128x128 tile 在 MT 4 從約 1,900 降到約 1,600 cycle，64x256 在 M=256 約 1,430 cycle（tensor core 下限 1,024）。

### 還沒做

- slice 結尾仍佔 M=64 關鍵路徑的約 25%：128x128 在 MT 3–4 是 4 個 warp 依序相加（約 2.3–4.6 µs）。成對相加要多一份 fp32 tile，shared memory 只剩 29 KB，得少一個 stage。
- 每個 tile 仍比 tensor core 下限多 40–55%：dequant 和 HMMA 沒有交錯（ptxas 把一列的 dequant 全排在 HMMA 前），每兩個 k16 列就一次 barrier。
- nano 的預設 `cuda_graph_max_bs` 是 64；大 batch 時改成全 graph 約快 1%。

## 檔案

| 檔案 | 內容 |
|---|---|
| `run_phase1.sh` | 主流程。step runner、安裝、nsys、ncu 函式來自 `../phase0/lib.sh` |
| `kbench.py` | kernel micro-benchmark：登記、正確性檢查、CUDA graph 計時、JSON 和 markdown |
| `tune.py` | launch 設定的 sweep，計時沿用 kbench |
| `tiled_trace.py` | `gemm_w4a16_tiled` 的 trace build：每個 block 的時間拆解、最後結束的 block、`--prefetch-tiles` 量 L2 預取的上限；超過 64 列時逐 chunk 套同一個排程 |
| `golden.py` | tiled kernel 輸出的逐 bit 比對（存一次，之後檢查） |
| `gemm_breakdown.py` | 從 nsys 的 SQLite 讀出 decode 裡每個投影的 GEMM 時間，nano 和 vLLM 都適用；同一投影連續的 launch 合成一次，`:prelude` regex（例如反量化）算進下一次 GEMM |
| `quality.py` | 8B 上的 perplexity 和 top-1 一致率 |
| `interp/` | 本機用 Docker（linux/arm64）跑 `TRITON_INTERPRET=1` 檢查 Triton kernel 的邏輯：`interp_check.py`，以及證明 interpreter 的 bf16 `tl.dot` 有錯的 `dot_check.py` |
| `summarize.py` | 把一次 run 整理成 `summary.md` |
| `results/<run_id>/` | `summary.md`、`status.tsv`、`logs/`、`tests/`、`kbench/`、`tune/`、`ncu/`、`nsys/`、`offline/`、`quality/`、`trace/`、`environment.txt`，以及當次 run 的程式碼副本 `code/` |

Colab 的 driver 和 launcher 沿用 Phase 0 的 `colab_phase0.sh`、`colab_launcher.py`，用 `BENCH_PHASE=phase1` 切換。tarball 除了 `benchmarks/phase0`、`benchmarks/phase1` 和 `src/serving/nano_vllm`，還會帶 `tests/unit/test_nano_vllm_*.py`、`tiny_llama.py`，以及這些測試在 nano_vllm 以外會 import 或讀取的檔案：`src/monitoring/metrics.py`、`src/api/model_routing.py`、`config/model_routing.nano.json`。`tests/gpu/` 存在的話也會帶上。

## 限制

- 只在 Colab A100-SXM4-40GB 上跑。80GB 卡的峰值會自動改成 2,039 GB/s，其他 GPU 要用 `--peak-gbps` 指定。
- 每個點只量一次：kbench 取 20 次 replay 的中位數，e2e 每組設定跑一次。
- `cuda129` 名稱沿用計畫，但實際裝的是和 venv 裡 torch 相同的 CUDA 版本。Phase 0 記錄到 vLLM 0.19.1 的 torch 是 `2.10.0+cu128`，所以預設會裝 12.8；要強制裝 12.9，設 `P1_CUDA_VERSION=12.9`。
