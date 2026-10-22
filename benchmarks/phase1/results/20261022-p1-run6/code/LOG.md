# Phase 1 執行紀錄

照時間順序記錄每一步：做了什麼、為什麼、遇到什麼問題、怎麼修。每個失敗都要記，包含原因和修正。結果和分析在 [README.md](README.md)。時間一律是 UTC。

## 2026-10-22 19:20 — M0：harness（本機）

### 做了什麼

- **`benchmarks/phase0/lib.sh`**：把 `run_phase0.sh` 裡兩個 phase 都用得到的部分抽出來：
  - `run_step`、`want`；
  - `install_vllm`、`install_nsys`、`download_model`、`find_nsys`、`find_ncu`；
  - `nsys_run`、`ncu_run`；
  - NVIDIA apt keyring 的邏輯，改成獨立的 `cuda_apt_repo <探測用的套件>`，給 `install_nsys` 和 `cuda129` 共用。

  `ncu_run` 多了一個用法：`ncu_run label count [ncu 選項... --] command...`，`--` 前面的參數會直接傳給 ncu。Phase 0 的呼叫沒有 `--`，行為和原本一樣。分析腳本的路徑改成相對於 `lib.sh` 本身，這樣從 phase1 source 進來也找得到 `analyze_*.py`。
- **`colab_phase0.sh`、`colab_launcher.py` 支援 `BENCH_PHASE`**（預設 phase0）：
  - 跑 `benchmarks/${BENCH_PHASE}/run_${BENCH_PHASE}.sh`；
  - 結果下載到 `benchmarks/${BENCH_PHASE}/results/`；
  - 轉送所有有設定的 `P1_*` 變數；
  - `rerun` 的預設 steps 依 phase 決定。
- **`run_phase1.sh`**：steps 有 `installs`、`cuda129`、`tests`、`kbench`、`ncu`、`e2e`、`summarize`，說明見 README。
- **`kbench.py`**：kernel 登記、正確性檢查、CUDA graph replay 計時、輸入輪替、JSON 和 markdown 輸出。第一批登記的是 `rms_norm`、`silu_mul`（Triton 對照 torch）和 `linear`（cuBLAS，GEMV baseline）。
- **`summarize.py`**：把 status、測試、kbench、ncu、e2e、nsys 整理成一份 `summary.md`。

### 本機驗證

- **Shell 語法**：`bash -n` 檢查過全部 shell 腳本。shellcheck 結果見下面「檢查」一段。
- **`kbench.py --device cpu`**：全部 kernel 跑完大約 80 秒，大部分時間花在 CPU 上產生 lm_head 那 1 GB 的權重。每個 torch 實作都和 reference 一致；Triton 實作標成 `skipped: needs Triton and a GPU`。
  - **抓到的 bug**：`kernels/__init__.py` re-export 了 `silu_mul` 這個函式，蓋掉了同名的 submodule。所以 `from src.serving.nano_vllm.kernels import silu_mul` 拿到的是函式，不是模組，silu_mul 的 reference 全部報錯。
  - **修正**：改成從 submodule 直接 import 函式，`from ...kernels.silu_mul import silu_mul, silu_mul_reference`。
- **GPU 路徑**：本機沒有 GPU，所以把 `torch.cuda` 的 Stream、graph、Event 換成 mock，用一個假 kernel 跑 `bench_kernel`。確認了這幾件事：
  - graph 計時正常；capture 失敗時會退回 eager，表裡也會標出來；
  - 結果不符、丟 exception、Triton 不可用，各自都有正確的紀錄；
  - x base 只拿第一個實作當基準。
- **`summarize.py`**：拿 Phase 0 的 nsys、ncu、offline 結果，加上 CPU kbench 的輸出，組成一個假的 run 目錄來測，各段表格都有正確產生。
- **Phase 0 行為沒變**：方法和結果見下面「檢查」一段。

### 檢查

- **shellcheck**：本機沒裝，改用 Docker 跑 `koalaman/shellcheck:stable -x`。
  - 新寫的程式沒有任何 finding。
  - 剩下的 5 個 finding 和 HEAD 的 `run_phase0.sh` 完全相同，都在搬到 `lib.sh` 或原本就沒動的程式碼裡：`SC2012`（用 ls 找 nsys 和 ncu）、`SC1091`（`/etc/os-release`）、`SC2034`（`stop_server` 的迴圈變數 `i`）、`SC2155`（`serve_engine` 的 `local start=`）。
- **Phase 0 行為沒變**：
  - **做法**：在 Linux container（bash 5、GNU coreutils，和 Colab 相同）裡，用同一個路徑先後跑 HEAD 和新版的 `run_phase0.sh`。steps 是 `installs bandwidth nsys_vllm nano ncu summarize`；nvidia-smi、uv、apt、nsys、ncu、python 都換成 stub，其中 apt stub 會強制走 NVIDIA keyring 那條路徑。
  - **結果**：console、每個 step 的 log、status.tsv 和輸出的檔案清單完全相同。只有 console 時間戳和 step 秒數先正規化，背景安裝的輸出順序會交錯，所以 console 排序後再比。
- **Phase 1 整條 Colab 流程的 dry run**：
  - **做法**：在 Linux container 裡把整條流程串起來跑：真的 `colab_phase0.sh rerun`（`BENCH_PHASE=phase1`）→ stub 的 `colab` CLI → 真的 `colab_launcher.py`（在 `/content` 底下）→ 真的 `run_phase1.sh`。
  - **用真的程式跑的部分**：pytest、`kbench.py --device cpu`、`analyze_ncu.py`、`summarize.py`、tar（bsdtar，和 macOS 一樣）。
  - **換成 stub 的部分**：ncu、nsys、apt、nvidia-smi、`nano_bench.py`、`vllm_offline.py`。
  - **確認了這些事**：
    - `P1_*` 有轉送到 VM 上；
    - `cuda129` 從 venv 的 torch 讀到 12.8，env 檔寫的是 `CUDA_HOME=/usr/local/cuda-12.8`；
    - ncu 的參數是 `--kernel-name regex:_rms_norm_kernel --nvtx -- ... kbench.py --ncu-mode --kernels rms_norm`，之後接 `analyze_ncu.py`；
    - e2e 對 `P1_E2E_RUNS` 裡的每個 paged 點各跑一次 vLLM，nsys 的參數也正確；
    - 結果有下載到 `benchmarks/phase1/results/<run_id>/`，`summary.md` 各段都有產生。
  - **抓到的 bug**：`tests` step 在收集測試時就失敗了。`nano_vllm/server.py` 會 import `src.monitoring.metrics`，但 tarball 沒有帶這個檔案，所以到 Colab 上 `test_nano_vllm_server.py` 一定會壞。
  - **修正**：tarball 加上 `src/monitoring/metrics.py`。它需要的 `prometheus_client` 是 vLLM 的依賴，venv 裡本來就有。
  - **修正後**：81 個測試通過，兩個 Triton 測試在沒有 GPU 的 container 裡被 skip。`tests` step 照設計判定失敗，到了 GPU 上這兩個測試就會跑。

### 發現：venv 的 torch 是 cu128，不是 cu129

- **原本的假設**：計畫寫的是「venv 的 torch 是 cu129，所以另外裝 CUDA 12.9」。
- **實際情況**：Phase 0 的 `environment.txt` 記錄到 vLLM 0.19.1 帶的是 `torch 2.10.0+cu128`。
- **影響**：用 12.9 的 nvcc 編 extension 的話，torch 的 cpp_extension 只會發出 minor 版本不一致的警告，大概還是能用；但最穩的做法是讓版本完全一致。
- **做法**：`cuda129` 這個 step 名稱不變，預設改成裝和 `torch.version.cuda` 相同的版本，在 Colab 上就是 12.8，env 檔裡會寫 `CUDA_HOME=/usr/local/cuda-12.8`。要強制裝 12.9 的話，設 `P1_CUDA_VERSION=12.9`。
- step 會在 log 裡印出 `/usr/local/cuda` 安裝前後指到哪裡，確認 apt 沒有把系統預設的 13.0 換掉。

### 選做：Docker linux/arm64 上的 `TRITON_INTERPRET=1`（timebox 15 分鐘，實際用了大約 5 分鐘）

**結果：可行。**

- **環境**：
  - Docker 29.8.2，`python:3.12-slim`（linux/arm64）；
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu` 裝到 torch 2.14.1+cpu；
  - `pip install triton` 裝到 triton 3.8.0，PyPI 有 aarch64 wheel。
- **測法**：repo 的 `kernels/` 以唯讀方式掛進容器，直接載入 `rmsnorm.py` 和 `silu_mul.py` 兩個檔案（不經過 package `__init__`，避免把整個 engine 拉進來），用 CPU tensor launch `@triton.jit` kernel。
- **fp32**：兩個 kernel 和 reference 的誤差都在 1e-6 以下。
- **bf16**：
  - rms_norm 最大誤差 3.1e-2，剛好是 [4, 8) 區間的一個 bf16 ulp：Triton 版最後只 round 一次，reference 先 round normalized 再乘 weight，round 了兩次。
  - silu_mul 最大誤差 6.25e-2，是 [8, 16) 的一個 ulp。
  - 用 kbench 的規則（atol + rtol·|ref|，都是 2e-2）判斷，兩個都通過。
  - 一開始驗證腳本用的是單純的絕對誤差 < 2e-2，所以 bf16 rms_norm 被誤判成失敗；改成 kbench 的規則之後就通過了。
- **時間**：每次大約 46 秒，幾乎都花在 pip 安裝上；kernel 本身在 interpreter 裡跑 5–93 ms。
- **用途**：之後寫新 kernel 時，可以先在本機用 interpreter 抓 indexing 和 mask 的邏輯錯誤，再上 Colab，省 CU。
- **限制**：
  - interpreter 量不到效能，也測不出 `tl.dot` 在 GPU 上的精度、race 或 shared memory 的問題。
  - nano_vllm 的 wrapper（`rms_norm(..., use_triton=True)`）在 CPU tensor 上會直接走 torch 版，所以要測 kernel 必須直接 launch `@triton.jit` 函式。
  - 要常用的話，可以做一個預先裝好 torch 和 triton 的 image，省掉每次的安裝時間。
- **路線圖要修正的地方**：原本寫「macOS 可以 TRITON_INTERPRET」，實際上 macOS 沒有 Triton wheel，要透過 Docker linux/arm64 才行。

### 第一次上 GPU 時要確認的事

- `cuda129` 會不會動到 `/usr/local/cuda` 的 symlink，看 log 裡印出的 before 和 after。
- kbench 的 CUDA graph capture：cuBLAS 的 workspace、Triton launcher 在 capture 底下能不能正常運作。表裡有任何一列標成 `eager`，就代表 capture 失敗了。
- `tests` step 不能有 `SKIPPED`：`test_nano_vllm_kernels.py` 的兩個 Triton 測試只在有 CUDA 和 Triton 時才會跑，到了 GPU 上應該都要跑到。
- e2e 的 nano paged b=64 會很慢：Phase 0 量到 b=16 是 619 ms/step，照比例推估，M1 之前的 b=64 大約 2.5 s/step，256 個 step 要 10 分鐘以上。這是推估，還沒量過。
- ncu 用 `--kernel-name regex:`，搭配 kbench 登記的 regex：rms_norm 和 silu_mul 只錄 Triton kernel，linear 錄全部（cuBLAS 的 kernel 名稱會隨 shape 改變）。

## 2026-10-22 — M1–M6 的程式碼（本機，主 session）

M0 在另一個 session 做的同時，主 session 寫了 nano_vllm 這邊的程式碼。每個 milestone 一個 commit，全部只在 CPU 上測過。

| commit | milestone | 內容 |
|---|---|---|
| `270f7d5` | M1 | `BatchMetadata`（device 上的 int32 block table 加 null block、`cu_seqlens_q`、`logits_indices`，經 pinned memory 兩次非同步 H2D）、`ModelRunner`、一次 sampling 整個 batch（`sample_tokens`）、載入時融合 QKV 和 gate_up、cos/sin table、整個 batch 一起做 RoPE、sync-free 的 PyTorch paged attention reference |
| `1c9ffa0` | M2 | Triton split-KV paged decode attention（每個 program 處理一個 kv head 的 4 個 q head，pad 到 16 行用 `tl.dot`）、融合的 RoPE + KV 寫入 |
| `4c9277f` | M3 | 每個 batch size 一個 decode CUDA graph（`ModelRunner.capture_decode_graphs`），新增 `enforce_eager`、`cuda_graph_max_bs` 兩個設定，`nano_bench.py` 加 `--attention-backend`、`--enforce-eager` |
| `586413a` | M4 | `fused_add_rms_norm`、`silu_and_mul`；Triton RMSNorm 的捨入改成和 transformers 一致 |
| `76d90a0` | M5 | Triton varlen causal prefill attention，直接讀 paged cache |
| `4f3d368` | M6（前半） | Triton GEMV（rows ≤ 16、split-K），新增 `linear_backend` 設定 |

驗證：

- **CPU 測試**：`pytest tests/unit/test_nano_vllm_*.py` 81 個通過，包括 static 和 paged 對 transformers 的 HF parity、engine 對 HF 的 greedy decode。`tests/gpu` 加了 126 個 GPU 測試，本機全部 skip。
- **nano_bench**：用 CPU 加 tiny 模型從頭跑過一次（`--attention-backend triton`，在 CPU 上會退回 reference）。

## 2026-10-22 — Triton interpreter 檢查（本機 Docker）

M0 確認 linux/arm64 container 裡可以跑 `TRITON_INTERPRET=1` 之後，做了一個固定的 image 和檢查腳本，放在 `interp/`：

```bash
docker build -t nano-triton-interp benchmarks/phase1/interp
docker run --rm -v "$PWD":/repo:ro nano-triton-interp python /repo/benchmarks/phase1/interp/interp_check.py
```

它直接 launch 每個 `@triton.jit` kernel，和 reference 比對。測了 decode attention（含 ctx=0、多個 split、page size 4 和 16、tiny 和 8B 的 head 配置）、prefill attention（含 prefix 和 chunk）、RoPE + KV 寫入（含 null slot）、rms_norm、fused_add_rms_norm、silu_and_mul、GEMV（含 split-K 和不整除的 K）。

- **第一次跑：22/29 通過**。fp32 的 decode、prefill 和 rope 全部通過，誤差都在 1e-6 以下。失敗的全是 bf16 加 `tl.dot` 的組合，誤差 1e8–1e11。
- **原因是 interpreter**：`interp/dot_check.py` 用一個最小的 `tl.dot` kernel 測試，fp32 和 fp16 都完全正確，bf16 的誤差有 4.6e10。triton 3.8.0 的 interpreter 對 bf16 的 dot 有 bug，不是我們的 kernel。
- **修正**：用到 `tl.dot` 的 kernel 改用 fp16 測，結果 29/29 通過。
- **同一個坑又踩一次**：檢查腳本用 `from kernels import linear` 時，拿到的是 re-export 的函式，不是 submodule（和 M0 碰到 `silu_mul` 時一樣）。改成從 `sys.modules["kernels.linear"]` 取。
- **interpreter 測不到的**：bf16 在 GPU 上的實際精度、效能、CUDA graph capture，這些要到 Colab 才能確認。

## 2026-10-22 19:37 — 第一次上 GPU（run `20261022-p1-run1`，Colab A100-SXM4-40GB）

`BENCH_PHASE=phase1 P1_STEPS="installs tests kbench summarize" COLAB_KEEP=1 ... colab_phase0.sh start`。VM 是新開的，vLLM venv 和模型都重新裝；`install_vllm` 84 秒，`download_model` 70 秒。

### tests：198 passed，11 failed

- **`tests/gpu` 全部通過**：Triton decode 和 prefill attention、RoPE + KV 寫入、fused norm、silu_and_mul、GEMV 對 reference；CUDA graph replay 對 eager（包含 batch 超過 graph 上限、torch backend）；`set_sync_debug_mode("error")` 底下的 decode forward 沒有 sync；bf16 + Triton 對 fp32 的端到端 parity。
- **失敗的 11 個全是和 transformers 對照的測試**：
  - **原因**：VM 的 vLLM venv 裝的是 transformers 5.20.2（vLLM 0.19.1 的依賴），不是 repo pin 的 4.57.6。transformers 5 存 config 時把 `rope_theta` 和 `rope_scaling` 改成放在 `rope_parameters` 底下，nano 的 `NanoLlamaConfig.from_pretrained` 讀不到，於是用了預設的 theta 10000、也沒有 scaling，結果 logits 和 HF 對不起來。`test_rope_matches_transformers_rotate_half_layout` 則是測試本身讀了 `config.rope_theta`，transformers 5 已經沒有這個屬性。
  - **修正**：`NanoLlamaConfig.from_pretrained` 改成也能讀 `rope_parameters`（verl 的 checkpoint 也是 transformers 5 存的，所以這本來就該支援），並加了一個測試。kernel 測試改用常數 500000.0，不讀 `config.rope_theta`。
  - **為什麼本機沒抓到**：本機和 CI 的 transformers 都是 4.57。

### kbench：兩類假失敗

- **`rope_cache_write` 的 Triton 版每個 shape 都 mismatch**，最大誤差 12.6。
  - **原因**：kbench 的 cos/sin 是隨機數，前後兩半不相同；kernel 照約定只讀前半（`rope_cos_sin` 的 table 兩半一樣），reference 則讀了整條 128 維。GPU 測試用的是真的 table，所以有過。
  - **修正**：kbench 改成用 `cat(angles, angles)` 算出真的 cos/sin。
- **「reference 太慢、刻意不計時」被算成失敗**：kbench 在 GPU 上把所有 `Unavailable` 都當失敗。
  - **修正**：新增 `NotApplicable`（`Unavailable` 的子類別），刻意略過的 shape 會在列上標 `by_design`，不算失敗。CUDA GEMV 超過 8 列的情況也用它。

### 第一批數字（kbench，CUDA graph replay，輸入輪替超過 L2）

**Decode attention**（對照 vLLM 的 FA2 `flash_attn_varlen_func`，block_table 和 vLLM 的 backend 呼叫方式相同）：

| B | ctx | FA2 µs | Triton µs | Triton 佔峰值 | 對 FA2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 6.9 | 12.0 | 3% | 0.58× |
| 1 | 1024 | 15.7 | 13.6 | 20% | 1.15× |
| 1 | 4096 | 30.2 | 22.4 | 48% | 1.35× |
| 16 | 1024 | 75.7 | 64.4 | 67% | 1.17× |
| 16 | 4096 | 225 | 222 | 78% | 1.01× |
| 64 | 1024 | 221 | 222 | 78% | 1.00× |
| 64 | 4096 | 813 | 862 | 80% | 0.94× |
| 128 | 1024 | 404 | 389 | 89% | 1.04× |
| 128 | 4096 | 1524 | 1510 | 91% | 1.01× |

- 大部分點和 FA2 打平或更快。B=1、ctx 夠長時，split-KV 讓 Triton 版快到 1.35×。
- B=1/ctx=128 比較慢：S=32 的 split 只有一個有工作，還要多跑一個 reduce kernel。換算成每個 token 大約多 0.17 ms，約佔 1%。
- PyTorch reference 只有 FA2 的 0.06–0.18×（也就是 M1 的 torch 路徑）。

**Prefill attention**（TFLOPs，佔 312 的比例）：Triton 版只有 FA2 的 0.55–0.67×。例如 1×4096 時，FA2 是 162 TFLOPs（52%），Triton 是 90（29%）。沒達到 0.7× 的目標，所以改成兩段迴圈（只有對角線那幾個 tile 做 mask），並把 launch 設定改成可以調，交給 `tune.py` 去掃。

**GEMV（M=1）**：

| layer | cuBLAS | Triton | 對 cuBLAS |
|---|---:|---:|---:|
| qkv | 79% | 77% | 0.97× |
| o_proj | 76% | 70% | 0.92× |
| gate_up | 85% | 88% | 1.04× |
| down | 81% | 83% | 1.02× |
| lm_head | 94% | 95% | 1.02× |

- kbench 量到的 cuBLAS o_proj 是 76%，比 Phase 0 的 63% 高。Phase 0 用的是一般 launch 加上 CUDA event 計時，kbench 是 graph replay，launch 的間隔不會算進去。
- Triton GEMV 在短而寬的 qkv 和 o_proj 上輸給 cuBLAS，接下來交給 `tune.py` 調，另外再寫 CUDA 版對照。

**小 kernel**：fused_add_rms_norm 和 rms_norm 的 Triton 版是 torch 的 6–12×；silu_and_mul 是 2–3.4×。M=2048 時都到峰值的 83–89%。

## 2026-10-22 19:30 — run `20261022-p1-run2`（同一台 VM，`COLAB_KEEP=1`）

`P1_STEPS="cuda129 tests kbench tune e2e summarize"`。這一輪帶了 run 1 的修正、兩段式 prefill 迴圈、CUDA GEMV v1（每個 warp 4 列）和第一版 `tune.py`。

### tests：225 passed，1 failed

- `cuda129` 裝好 CUDA 12.8 的 nvcc，`test_cuda_extension_builds` 通過，CUDA GEMV 的 JIT 編譯在 VM 上可以用。
- **剩下的 1 個失敗**：transformers 5 會直接改寫傳進去的 `rope_scaling` dict，測試 helper 的 module 層級常數 `LLAMA3_SCALING` 被改掉，之後的測試就讀到錯的 scaling。只在測試順序剛好時出現。
  - **修正**：`_save_tiny_hf_llama` 改成 `copy.deepcopy(overrides)` 之後再傳。

### kbench

- **prefill 兩段迴圈**：從 FA2 的 0.55–0.67× 變成 0.63–0.70×，1×4096 從 29% 升到 33% 峰值，還是沒到 0.7×。
- **CUDA GEMV v1**：M=1 時和 cuBLAS 打平（qkv 1.00×、gate_up 1.06×、lm_head 1.00×），但 o_proj 0.92×、down 0.93×；M=8 時掉到 0.73–0.96×。
  - **推測原因**：一是每個 x 列都重新把權重從 bf16 轉成 fp32，M=8 時轉換次數是 M=1 的 8 倍；二是每個 block 管 16 列，o_proj 只有 256 個 block（108 個 SM 跑不滿 3 個 wave），最後一個 wave 拖慢。
  - **v2 的改法**：權重每個元素只轉換一次；每個 warp 的列數改成 template 參數（1、2、4），交給 `tune.py` 掃。

### tune：decode 和 prefill 跑完，GEMV 崩潰

- **崩潰原因**：`tune.py` 用 `from src.serving.nano_vllm.kernels import linear as linear_mod`，拿到的是 re-export 的 `linear()` 函式，不是 submodule。這是第三次踩到同一個坑（M0 的 `silu_mul`、interpreter 檢查的 `linear`）。
  - **修正**：改用 `importlib.import_module("src.serving.nano_vllm.kernels.linear")`。另外 `tune.py` 改成每掃完一個 kernel 就寫一次 `tune.json`，每個 shape 的 log 行也帶上所有 config 的時間，之後再崩潰也不會丟掉已經跑完的部分。
- **decode**（µs，舊預設是 block_n=64、4 warps、2 stages）：

  | B | ctx | 最佳 config | 最佳 | 舊預設 | 加速 |
  |---:|---:|---|---:|---:|---:|
  | 1 | 1024 | 32 / 4 / 1 | 14.6 | 16.4 | 1.13× |
  | 1 | 4096 | 64 / 8 / 1 | 20.9 | 22.4 | 1.07× |
  | 16 | 1024 | 32 / 4 / 3 | 61.3 | 64.6 | 1.05× |
  | 64 | 1024 | 128 / 4 / 3 | 200.1 | 219.4 | 1.10× |
  | 64 | 4096 | 128 / 4 / 3 | 767.3 | 860.7 | 1.12× |
  | 128 | 1024 | 128 / 4 / 3 | 379.9 | 386.4 | 1.02× |

  B 小的時候每條 sequence 切成很多 split，短的 tile 讓每個 split 都有事做；B≥64 時 S=1，一個 program 掃完整條 context，長 tile 加 3 段 pipeline 比較快。B=64/ctx=4096 從 FA2 的 0.94× 變成約 1.06×。

- **prefill**（µs，舊預設是 block_m=64、block_n=64、4 warps、2 stages）：

  | shape | 最佳 config | 最佳 | 舊預設 | 加速 | 換算對 FA2 |
  |---|---|---:|---:|---:|---:|
  | 1×1024 | 128 / 64 / 8 / 2 | 105.3 | 129.6 | 1.23× | 0.80× |
  | 1×4096 | 128 / 128 / 8 / 2 | 940.8 | 1333.5 | 1.42× | 0.90× |
  | 4×1024 | 128 / 32 / 4 / 2 | 295.3 | 402.5 | 1.36× | 0.88× |

  block_m=128 是主要差別：每個 program 載入一次 K/V tile，給兩倍的 query 列用，K/V 的讀取量減半。

- **新的預設**（`paged_attention.py`）：
  - `decode_config(B)`：B<32 用 32 / 4 / 3，B≥32 用 128 / 4 / 3。只看 batch 大小，所以每個 CUDA graph 的設定是固定的。
  - `PREFILL_CONFIG` 改成 128 / 64 / 8 / 2。
  - 這是從 run 2 的「最佳值」挑出來的規則，沒有每個點都量過，run 3 的 `tune.py` 會把「目前規則」和最佳值並排列出來。interpreter 檢查多了大 batch config 的 decode case，31/31 通過。

### e2e：開 graph 之後和 vLLM 打平（run 2，用的是調參前的預設）

decode ms/step（`nano_bench.py` 和 `vllm_offline.py`，相同的 batch / in / out；nano 是 paged path）：

| config | b=1 in=128 | b=16 in=1024 | b=64 in=1024 |
|---|---:|---:|---:|
| torch_eager（M1 重構後的 torch 路徑） | 37.59 | 51.07 | 152.42 |
| triton_eager（M2/M4/M5 的 kernel，不開 graph） | 21.95 | 23.66 | 29.35 |
| triton_graph（再加 M3 的 CUDA graph） | 12.67 | 14.96 | 24.08 |
| gemv_graph（再加 M6 的 Triton GEMV） | **12.47** | 14.99 | 24.07 |
| vLLM 0.19.1（CUDA graph） | 12.63 | 14.83 | 23.04 |
| nano / vLLM | **0.99×** | 1.01× | 1.04× |

- **起點對照**：Phase 0 的 nano 是 b=1 44.1 ms、b=16 619 ms。M1 純重構（拿掉逐 sequence 迴圈和 host sync）就讓 b=16 從 619 降到 51 ms；Triton kernel 再降到 24 ms；graph 把 b=1 從 22 降到 12.7 ms，達到 M3 的 ≤13.5 ms 目標。
- **b=1 比 vLLM 快 1%**：nsys（gemv_graph b=1）每個 step 424 個 kernel（Phase 0 是 1,604），2 次 launch API 呼叫（graph replay 和 sampling），GPU 忙 12.29 ms、閒 0.76 ms；GEMM 佔 kernel 時間的 93%。
- **b=64 還慢 4.5%**：b=16 時 attention 佔 14%，b=64 時更多；run 2 的 decode 預設在 B=64 比最佳設定慢 10%，run 3 用調過的設定再量。
- **GEMV 在 e2e 上只差 0.2 ms**：b=1 時 Triton GEMV 比 cuBLAS 快 1.6%，和 kbench 的結果一致（gate_up、down、lm_head 稍快，qkv、o_proj 稍慢）。b≥16 時列數超過 16，GEMV 不會被用到，所以和 triton_graph 一樣。
- **static path 沒變**：那條路徑是 Phase 0 的對照組，刻意保留 F.linear 和逐層的 PyTorch attention。
- **prefill 時間不能拿來比**：`vllm_offline.py` 的 `prefill_s` 是把非 decode step 的時間加總。b=16×1024 時只有 0.042 s，換算超過 A100 的算力上限（16K token × 16 GFLOP ≈ 262 TFLOP，至少 0.84 s），所以 vLLM 的 step 計時沒有涵蓋整個 prefill。nano 的 1.02 s 約等於 257 TFLOPs，是峰值的 82%。這兩個數字要等改用 TTFT 量法之後才能並排比較。

## 2026-10-22 20:13 — run `20261022-p1-run3`（同一台 VM）

`P1_STEPS="tests kbench tune e2e summarize"`。帶了調過的 decode/prefill 預設、CUDA GEMV v2、修好的 `tune.py`，e2e 多一組 `cuda_gemv_graph`。

- **tests：226 passed**，沒有 skip。
- **CUDA GEMV v2**（每個 warp 2 列、每個權重只轉換一次）：

  | layer | M=1 | M=2 | M=4 | M=8 |
  |---|---:|---:|---:|---:|
  | qkv | 1.00× | 1.00× | 0.95× | 0.69× |
  | o_proj | 0.98× | 1.01× | 0.92× | 0.59× |
  | gate_up | **1.08×**（92% 峰值） | 1.07× | 1.05× | 0.77× |
  | down | 1.04× | 1.00× | 0.93× | 0.57× |
  | lm_head | 1.01× | 1.00× | 0.99× | 0.73× |

  （對 cuBLAS 的速度比。）M≤2 時每種 shape 都和 cuBLAS 打平或更快，M=8 反而比 v1 更慢。
  - **原因**：CUDA core 的 fp32 FMA 不夠。M=8 時每個權重要做 8 次 FMA，bf16 權重在 1.4 TB/s 下是每秒 0.7e12 個，需要 5.6e12 FMA/s，再加上 x 的轉換和 shared memory 讀取，已經接近 A100 的 9.7e12 FMA/s。Triton 版用 `tl.dot` 走 tensor core，所以 M=8 時還能維持 0.93–1.05×。
  - `tune.py` 顯示 M=8 時每個 warp 4 列能快 1.2–1.3 倍（例如 down 從 162 µs 降到 124 µs），但還是輸給 cuBLAS 的 93 µs。
  - **結論**：CUDA core 的 GEMV 只適合 1–2 列；列數再多就需要 tensor core（M7 的 W4A16 也是同一個道理，所以接著寫 `mma.sync` 版）。
- **tune**（每個 shape 的「目前規則」和最佳值並排）：
  - decode：B≥16 時規則就是最佳或差不到 2%；B=1 時 8 個 warp、1 段 pipeline 還能快 1.08–1.19 倍。
  - prefill：128 / 128 / 8 / 2 在 1×1024 和 1×4096 最快（1×4096 比目前的 128 / 64 快 1.08 倍）；4×1024 則是 128 / 32 / 4 / 2 最快。
  - Triton GEMV：qkv、o_proj 用 block_n=32、block_k=128、4 warps、3 stages 快 2–6%；gate_up 和 down 各有不同的最佳值，差 3–4%。
- **e2e**（decode ms/step；run 3 用了 run 2 調出來的 decode/prefill 預設）：

  | config | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | triton_graph | 12.61 | 14.87 | 23.48 |
  | gemv_graph（Triton GEMV） | 12.45 | 14.89 | 23.50 |
  | cuda_gemv_graph（CUDA GEMV） | **12.09** | 14.86 | 23.50 |
  | vLLM 0.19.1 | 12.63 | 14.83 | 23.04 |
  | nano / vLLM（最佳那組） | **0.96×** | 1.00× | 1.02× |

  - b=64 從 run 2 的 24.07 降到 23.50 ms，這是 decode 新設定的效果（kbench：B=64/ctx=1024 從 219 µs 降到 200 µs）。
  - b=1 時 CUDA GEMV 比 Triton GEMV 快 0.36 ms：nsys 顯示每個 step 328 個 kernel（Triton GEMV 是 424 個），少掉的 96 個正是 Triton split-K 的 reduce kernel，每個約 2.6 µs。CUDA 版沒有 split-K，一個 kernel 就寫出結果。
  - b≥16 時兩種 GEMV 都用不到（列數超過上限），三組設定一樣。
- **nsys 順便看到的**：b=16 每個 step 有一個 PyTorch argmax（`reduce_kernel<ArgMaxOps>`）花 56 µs，讀 4 MB 照理只要 3 µs；PyTorch 對「列很少、每列很長」的 reduce 平行度不夠。留給 Phase 2。
- **套用 run 3 的完整 sweep**（`tune.json` 裡每個 config 的時間）：
  - decode 分三段：B<16 用 64 / 8 warps / 1 stage，16≤B<64 用 32 / 4 / 2，B≥64 用 128 / 4 / 3。每段在每個量過的 (B, ctx) 都在最佳值的 3% 以內；B=1/ctx=4096 從 22.6 降到 20.9 µs。
  - prefill 改成單一的 128 / 32 / 4 warps / 2 stages：在三個 shape 上都在最佳值的 1.3% 以內（4×1024 從 319.5 降到 295.7 µs）。
  - Triton GEMV 改成 block_n=32、block_k=128、4 warps、3 stages：每個 shape 都在最佳值的 2% 以內。
  - CUDA GEMV 的分派上限從 8 列改成 2 列，只在贏 cuBLAS 的地方用。kernel 本身還是支援 8 列，GPU 測試直接呼叫它測 3 和 8 列。

## 2026-10-22 20:38 — run `20261022-p1-run4`：W4A16 第一次上 GPU，三個問題

`P1_STEPS="tests kbench tune e2e quality summarize"`，`P1_TUNE=w4a16`，e2e 只跑 gemv_graph、cuda_gemv_graph、w4a16_graph。

1. **tests 在收集階段就失敗**：`tests/unit/test_nano_vllm_w4a16.py` 和 `tests/gpu/test_nano_vllm_w4a16.py` 同名，兩個目錄都沒有 `__init__.py`，pytest 把它們當成同一個 module（`import file mismatch`）。一個測試都沒跑。
   - **修正**：GPU 那個改名成 `test_nano_vllm_w4a16_kernels.py`。
2. **`w4a16.cu` 編不過**：`fatal error: cusparse.h: No such file or directory`。為了拿 SM 數量 include 了 `ATen/cuda/CUDAContext.h`，它會連帶 include cuSPARSE 和 cuBLAS 的 header，可是 `cuda129` step 只裝了 nvcc、cudart-dev 和 cccl。
   - **修正**：改用 `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`。
   - **以後不必上 GPU 才發現編譯錯誤**：新增 `benchmarks/phase1/nvcc/`，用 Docker（linux/arm64）裝 NVIDIA 的 arm64 nvcc 12.8（和 VM 一樣只裝 nvcc、cudart-dev、cccl）加上 CPU 版 torch 2.10.0 的 header，用 cpp_extension 的參數對 sm_80 跑 `nvcc -c`，並用 `-Xptxas -v` 列出暫存器和 spill。CPU wheel 少一個生成的 header（`c10/cuda/impl/cuda_cmake_macros.h`），腳本會補一個一行的 stub。
   - **ptxas 抓到的問題**：tensor-core kernel 在 3–4 個 m16 tile 時用滿 255 個暫存器，spill 192–208 bytes。原因是每個 thread 有兩組 fp32 累加器（總和和這一組的），各 MT×NT×4 個。改成 MT≥3 時每個 block 只放 2 個 n8 tile（16 欄），暫存器降到 140–188，沒有 spill。
3. **w4a16 的 e2e 在 CUDA graph capture 時失敗**（`operation failed due to a previous error during capture`）：extension 沒編起來，`dequantize_w4a16` 退回 PyTorch 版，而 `unpack_w4a16` 用 Python list 做 index，torch 會先在 host 建 index tensor 再複製到 GPU，capture 期間不允許。
   - **修正**：shift 量改用 `torch.arange` 直接在 device 上算（k=2i 在 bit 4i，k=2i+1 在 bit 16+4i），不再有 host-to-device 複製。
4. **quality 跑不起來**：vLLM 的 venv 沒有 `datasets`，也沒有 `pyarrow`。改成 quality step 需要時才用 uv 裝 `pyarrow`，再從 HF hub 下載 wikitext-2 的 parquet。

**bf16 的 e2e 照常跑完**，這是第一次用 run 3 調出來的全部預設值：gemv_graph b=1 12.32 / b=16 **14.63（vLLM 14.83，0.99×）**/ b=64 23.77 ms；cuda_gemv_graph b=1 12.15 / 14.87 / 23.51 ms。b=64 這兩組的 kernel 完全一樣（列數都超過 GEMV 上限），卻差 0.26 ms，所以 e2e 的 run-to-run 差異大約是 1%。
