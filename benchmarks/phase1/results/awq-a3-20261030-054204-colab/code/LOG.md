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

## 2026-10-22 20:50 — run `20261022-p1-run5`：W4A16 第一次跑通

- **tests：315 passed**，沒有 skip。兩個 W4A16 kernel 對反量化矩陣乘法都正確（1–64 列、bf16/fp16、K 切分、graph capture），W4A16 模型對 fp32 參考在 bf16 容忍度內，graph replay 等於 eager。
- **kbench**（對照 vLLM `gptq_marlin`，用同一組 4-bit code；µs）：

  | layer | M | Marlin | 我們 GEMV | 我們 mma | cuBLAS bf16 |
  |---|---:|---:|---:|---:|---:|
  | qkv | 1 | 19.2 | 23.0 | 23.9 | 41.0 |
  | o_proj | 1 | 17.5 | 19.5 | 21.8 | 28.5 |
  | gate_up | 1 | 59.6（65% 峰值） | 71.6 | 70.4 | 178.3 |
  | down | 1 | 23.5（83%） | 43.2 | 36.5 | 92.9 |
  | o_proj | 16 | 19.5 | - | 19.5 | 29.0 |
  | gate_up | 16 | 66.1 | - | 78.1 | 182.9 |
  | gate_up | 64 | 113.9 | - | 312.5 | 264.7 |
  | down | 64 | 62.8 | - | 187.5 | 99.3 |

  - M≤16 時 mma 版是 Marlin 的 0.64–1.00×，比 bf16 cuBLAS 快 1.5–3.3 倍。
  - M=32–64 時掉到 Marlin 的 0.34–0.73×，M=64 甚至比 bf16 cuBLAS 慢。**原因**：每個 block 只管 32 欄（M≥33 時 16 欄），卻要從 L2 讀整份 x 的 K 範圍；M=64 時 x 是 512 KB，o_proj 的 256 個 block 加起來從 L2 讀約 128 MB，權重只有 8 MB。
  - kbench 和 tune 量到的同一個設定差到 20%（qkv M=1 GEMV：kbench 23.0、tune 18.7 µs）。兩邊都輪替輸入超過 L2，但 kbench 每份輸入還帶著 bf16 權重（給 cuBLAS 那列用），working set 和配置都不同。所以不同實作之間只拿同一個工具的數字比：Marlin 的對照看 kbench，設定的選擇看 tune。
- **tune**：
  - GEMV 在 1–2 列時每個 warp 2 列最好，4 列反而慢到 1.27 倍（o_proj、down）。M=2 時 GEMV 還是比 mma 快，M≥3 改用 mma。
  - mma 的 K 切分：自動值（目標每個 SM 4 個 block）對 qkv 切太多（切 3 份比不切慢 14%），每多切一次就多一個 memset、一個 cast kernel 和 atomic 的成本。改成目標每個 SM 1.5 個 block。
- **e2e**（decode ms/step）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | nano W4A16 | **6.48** | **9.31** | 30.32 |
  | nano bf16（run 4 最佳） | 12.15 | 14.63 | 23.51 |
  | vLLM bf16 | 12.63 | 14.83 | 23.04 |

  b=1 比 bf16 快 1.9 倍，b=16 快 1.6 倍；b=64 反而慢，就是上面 M=64 的問題。nsys（b=16）每個 step 520 個 kernel，比 bf16 多 192 個：3 個要切 K 的 linear × 32 層 × 2（memset 和 cast）。
- **quality**（wikitext-2 test，288,627 個 token，2048-token 區塊，teacher forcing）：

  | config | 權重 GiB | perplexity | top-1 一致率（對 bf16 torch） |
  |---|---:|---:|---:|
  | bf16，torch 路徑 | 15.0 | 7.233 | - |
  | bf16，Triton kernel | 15.0 | 7.232 | 98.81% |
  | W4A16（RTN g128 對稱），Triton | 5.3 | 8.024 | 85.90% |

  - Triton 和 torch 的 perplexity 到小數第三位都一樣；top-1 一致率 98.81%，比計畫的 99% 低一點。兩邊的 linear 和 norm 走同一條路（prefill 列數超過 GEMV 上限），只差在 attention：Triton kernel 照 FA2 的做法把 P 轉成 bf16 再乘 V，reference 全程 fp32。不一致的位置是 bf16 捨入剛好翻轉的接近平手的 token，不是錯誤。
  - RTN W4A16 讓 perplexity 增加 11%，和文獻上 Llama 3 8B 的 RTN g128 結果同一個量級（Llama 3 比 Llama 2 難量化）。要更好的品質得用 GPTQ/AWQ 或非對稱量化，這是 Phase 2 的事。
- **針對 M>16 和 K 切分的修正**（run 6 驗證）：
  - 新的 `gemm_w4a16_wide`（17–64 列）：block 裡 4 個 warp 各管不同的 32 欄（一個 block 128 欄），共用一份放在 shared memory 的 x（`cp.async`，先載下一組），讀取時用 XOR swizzle 避開 bank conflict。group scale 乘進反量化後的 B fragment（q−8 是精確值，乘積只在 bf16 捨入一次，和 Marlin 一樣），所以每個 thread 只要一組 fp32 累加器。ptxas：120–190 個暫存器，沒有 spill。
  - K 切分改成「最後一個抵達的 block 收尾」：每個 split 把部分和 atomicAdd 進 fp32 workspace，用每個 column tile 的計數器判斷誰最後到；最後到的 block 寫出 bf16，並把 workspace 和計數器歸零。workspace 常駐（`w4a16_workspace`，模型載入時就配好，在 capture 之前），所以不再需要 memset 和 cast kernel。

## 2026-10-22 21:07 — run `20261022-p1-run6`：wide kernel、K 切分收尾、vLLM AWQ 對照

- **tests：348 passed**，包含 wide kernel 的 1–64 列、K 切分、graph capture，以及「呼叫後 workspace 和計數器都歸零」。
- **kbench**（自動分派，對 Marlin 的速度比）：

  | layer | M=1 | M=4 | M=8 | M=16 | M=32 | M=64 |
  |---|---:|---:|---:|---:|---:|---:|
  | qkv | 1.06 | 0.89 | 0.89 | 0.97 | 0.68 | 0.70 |
  | o_proj | **1.33** | 1.03 | 1.00 | 1.07 | 0.62 | 0.62 |
  | gate_up | 0.99 | 0.86 | 0.85 | 0.88 | 0.61 | 0.51 |
  | down | 0.64 | 0.60 | 0.59 | 0.66 | 0.52 | 0.48 |

  wide kernel 讓 M=64 的時間大約減半（gate_up 311 → 204 µs，down 237 → 131 µs），M=32 的 gate_up 和 down 也變快，但 qkv 和 o_proj 在 M=32 時還是 mma kernel 快（tune：1.19–1.30 倍）。e2e 量的是 b=1、16、64，所以自動分派維持 16 列以上改用 wide。
- **e2e**：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | nano W4A16 | 5.95 | 8.91 | 24.20 |
  | vLLM AWQ int4（`awq_marlin`） | 5.55 | 7.87 | 16.23 |
  | nano / vLLM | 1.07× | 1.13× | 1.49× |

  vLLM 跑的是 `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`，量化方法（AWQ，非對稱）和我們的 RTN 不同，比的是速度。b=16 每個 step 的 kernel 從 520 降到 328，K 切分不再需要 memset 和 cast。b=64 從 30.32 降到 24.20 ms。

## 2026-10-22 21:21 — run `20261022-p1-run7`：W4A16 的 ncu 證據

`P1_STEPS="ncu summarize" P1_KERNELS=w4a16_linear P1_KBENCH_SHAPES="down_proj M=(1|16|64) "`，只看最弱的 down_proj（N=4096，K=14336）。新增的 `kbench.py --shapes` 讓 ncu 只錄這幾個 shape。

- 第一次失敗：ncu mode 把 GEMV 在 M=64 的 `NotApplicable`（刻意不支援）算成錯誤，kbench 回傳 1，ncu step 就沒有做 export 和分析。改成和計時模式一樣當成 skip（run 7b）。
- **結果**（ncu，`--clock-control none`，每組三次取中位數）：

  | M | kernel | µs | DRAM 佔峰值 | occupancy | 主要 stall（每個 issue 的 warp 數） |
  |---:|---|---:|---:|---:|---|
  | 1 | Marlin | 33.6 | 62% | 12%（grid 108，每個 SM 一個 block） | long_scoreboard 1.4 |
  | 1 | GEMV | 40.1 | 52% | 26% | **long_scoreboard 3.8** |
  | 1 | mma | 46.1 | 46% | 15% | long_scoreboard 2.5 |
  | 1 | wide | 51.0 | 41% | 13% | long_scoreboard 2.5 |
  | 16 | Marlin | 34.5 | 62% | 12% | long_scoreboard 0.8 |
  | 16 | mma | 50.4 | 42% | 15% | long_scoreboard 2.9 |
  | 64 | Marlin | 67.4 | 34%（SM 36%） | 6% | wait 1.8 |
  | 64 | wide | 134.0 | 17%（SM 21%） | 12% | long_scoreboard 1.4、wait 1.3 |

- **結論**：我們的三個 kernel 都既不是頻寬瓶頸，也不是算力瓶頸，而是被記憶體延遲卡住。warp 大多在等 global load（long_scoreboard），因為每個 thread 同時只有大約一組（64 B）權重在路上；多開 warp 也沒用，GEMV 的 occupancy 是 Marlin 的兩倍，DRAM 還是比較低。Marlin 每個 SM 只放一個常駐 block，用多段 `cp.async` pipeline 讓權重和 x 都提早好幾段載入，occupancy 只有 12% 就能到 62% DRAM。M=64 時 wide 的 SM 使用率只有 21%，除了等記憶體，還有 mma 之間的相依（wait）。
- **下一步**（Phase 1 先停在這裡）：權重也經過 shared memory，用 3–4 段 `cp.async` pipeline；grid 改成常駐、跨 SM 條狀分配 K（Marlin 的 striped partition），取代現在的 K 切分加 atomic；M≥32 時用 `ldmatrix` 讀 A，並讓每個 warp 管更多欄來攤平 A 的讀取。
- 跑完之後把 Colab VM 關掉了。

## 2026-10-23 06:18 — run `kernels-r1-20261023-061836-colab`：Kernel 改進清單第一輪

Phase 2 之後重新評估路線圖 doc 的「Kernel 改進清單」（計畫在 `~/.claude/plans/abundant-shimmying-crab.md`），這一輪做 async 預設（#10）、sync guard（#12）、採樣路徑的量測（#5 的前半）、GEMV 暫存器預取（#1）和 FP8 KV（#3）。分支 `claude/inference-kernels`（從 `claude/inference-phase2` 開，只在本機）。

設定：`P1_STEPS="installs cuda129 tests kbench tune ncu e2e summarize" P1_KERNELS=w4a16_linear P1_TUNE=w4a16`，e2e 是 b=1/16/64 的 bf16 和 W4A16，sync 和 async，greedy 和 sampled（temperature 0.7、top-p 0.9，nano server 的預設），W4A16 另外跑預取深度 2 和 4；vLLM 跑 bf16 和 AWQ，greedy 和 sampled。

- **tests**：455 passed、27 skipped，step 算失敗。skip 全是 C++ radix tree 的測試：Phase 1 的 installs 沒裝 nanobind（Phase 2 的有）。新的 GPU 測試全過：async 的 W4A16 和中途加入的 request、sync guard（greedy/sampled × graph/eager × decode/mixed，沒有抓到任何同步）、fp8 codec（每個 code 的解碼、tie、subnormal、飽和都和 torch 逐 bit 相同）、GEMV 預取的每個深度。修正：tests step 前先裝 nanobind（72e64ed）。
- **GEMV 預取（#1）是負面結果**。tune（µs，r = rows_per_warp，p = 預取深度）：

  | shape | r1 p0 | r1 p1 | r1 p2 | r1 p4 | r2 p0 | r2 p2 | r2 p4 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | down M=1 | **37.7** | 39.2 | 39.4 | 48.3 | 39.0 | 40.1 | 40.9 |
  | gate_up M=1 | 63.4 | 66.4 | 68.9 | 73.7 | **60.1** | 64.1 | 70.7 |
  | qkv M=1 | 21.0 | 20.6 | 19.7 | 21.7 | **18.1** | 19.1 | 21.0 |
  | o_proj M=1 | **13.1** | 14.5 | 14.3 | 17.9 | 13.5 | 14.0 | 15.1 |

  e2e 的 W4A16 async b=1：深度 0 / 2 / 4 是 5.74 / 6.00 / 6.32 ms。原因：ring 吃暫存器、降低 occupancy（ncu 的 M=8 版本 achieved occupancy 22% → 16% → 12%）；M=1、深度 0 已經有 30% achieved（理論 62%），延遲本來就由其他 warp 蓋住，單一 warp 的 load 串行不是瓶頸。G1 不通過，revert（e53207f）。ncu 的 300 個 kernel 上限沒錄到 M=1 的預取版本，但 tune 和 e2e 已經夠下結論。
- **e2e**（ms/step）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | nano bf16，sync | 12.28 | 14.55 | 23.59 |
  | nano bf16，async | **12.05** | **14.12** | **22.60** |
  | vLLM bf16 | 12.62 | 14.82 | 23.11 |
  | nano bf16，async，sampled | 12.47 | 15.14 | 25.22 |
  | vLLM bf16，sampled | 12.87 | 16.15 | 24.77 |
  | nano W4A16，sync | 5.97 | 8.89 | 24.28 |
  | nano W4A16，async | 5.74 | 8.47 | 23.27 |
  | vLLM AWQ（Marlin） | **5.54** | **7.86** | **16.19** |
  | nano W4A16，async，sampled | 6.16 | 9.48 | 25.87 |
  | vLLM AWQ，sampled | 5.82 | 9.34 | 18.10 |

  - async 在 W4A16 上省 0.23 / 0.42 / 1.01 ms，和 bf16 差不多；W4A16 還是落後 AWQ 3.6% / 7.8% / 44%。b=64 落後超過 10%，Marlin 式 kernel（#2）列為下一輪。
  - **採樣的成本**（sampled 減 greedy）：nano 0.42 / 1.02 / 2.62 ms，vLLM 0.25 / 1.33 / 1.66 ms。b=16 nano 比較便宜，b=64 貴 0.96 ms，超過 G3 的 0.3 ms 門檻：不排序的採樣器列為下一輪。nsys（W4A16 async sampled b=16，每步）：`torch.cumsum` 的 innermost scan 231 µs、float radix sort 4 × 48 µs 加 `sort_postprocess` 41 µs 加 index 的 long sort 34 µs、兩次 argmax 55（greedy 的 bf16 argmax，每步都算）+ 41 µs（multinomial 的 exponential trick），合計約 0.7 ms。greedy 的 argmax 55 µs 就是 Phase 1 留下的那個缺口，`reduce_kernel` 一列一個 block，16 列只用 16 個 SM。
- **kbench**（W4A16，`cuda_auto` / `vllm_marlin` 的時間比）：down_proj 在 M=1–64 都是 1.5–2.1 倍，gate_up 1.0–1.8 倍，M≥32 時四個 layer 都是 1.4–2.1 倍；o_proj 和 qkv 在 M≤16 時和 Marlin 打平或更快。這是下一輪 Marlin 式 kernel 的起點。

## 2026-10-23 06:55 — run `kernels-r2-20261023-065546-colab`：FP8 KV cache 第一次上 GPU（同一台 VM）

FP8 KV（#3）在 R1 跑的同時寫完（540cc8d）。Triton 3.6 的 CUDA backend 在 sm80 只接受 fp8e5 和 fp8e4b15，`float8e4nv` 要 sm89 以上，A100 也沒有 fp8 轉換指令，所以 kernel 把 cache 當 uint8，用整數運算編碼和解碼 e4m3（`kernels/fp8.py`）：解碼把 7 個 bit 放到 bf16 的 bit 4–10，得到「原值 × 2^-120」直接進 `tl.dot`；每個 (token, kv head) 一個 fp32 scale，K 的 scale 乘在 score、V 的 scale 乘在機率上，q 先乘 2^60、輸出最後乘 2^60，中間值都離 fp32 的 subnormal 很遠。本機先用 interpreter 檢查（它的 bf16 `tl.dot` 是壞的，所以 fp32 query 時解碼成 fp32，2^-120 的關係一樣成立）：59/59 通過。

設定：`P1_STEPS="tests kbench e2e quality summarize"`，kbench 跑 bf16 和 fp8 的 decode attention，e2e 是 bf16 / W4A16 × bf16 / fp8 KV（async），加上 b=16、4096 token 的點，vLLM 跑預設和 `--kv-cache-dtype fp8 --attention-backend TRITON_ATTN`；quality 比四組。

- **tests**：535 passed、9 failed，全是新的 fp8 測試的容差，沒有結構性錯誤。kernel 測試的 K/V 刻意跨兩個數量級，輸出也跟著放大，1–9 個元素（最多 2M 個）超出絕對容差（fp32 最多 1.6e-4，bf16 最多 0.059，都不到最大輸出的 1%）；模型層級的 fp32 嚴格比對通過，bf16 模型量化的是 bf16 捨入過的 K，有些 code 和 fp32 模型差一格；fp8 對全精度的 cosine 0.9985，而這個隨機權重的模型連 fp32 reference 路徑都只有 0.9986（CPU 上量）。修正：kernel 測試改成「誤差不超過輸出最大值的某個比例」，bf16 模型層級改比 cosine，門檻 0.995（2b5daa7）。
- **kbench**（decode attention，µs，bf16 / fp8，同一個 Triton kernel 和 launch 設定）：

  | | B=1 | B=16 | B=64 | B=128 |
  |---|---:|---:|---:|---:|
  | ctx=1024 | 14.4 / 14.2 | 62.3 / 58.7 | 201.0 / 147.9 | 379.9 / 273.3 |
  | ctx=4096 | 20.9 / 23.1 | 204.6 / 188.5 | 770.2 / 543.3 | 1484.9 / 1032.1 |

  B≥64 快 26–30%，B≤16 幾乎沒差，B=1、4096 反而慢 10%。bytes 少了將近一半，但 fp8 只到 HBM 峰值的 60–69%（bf16 是 86–93%）：launch 設定是照 bf16 調的，加上整數解碼。tune.py 加了 `decode_fp8` 的掃描（b3dc532）。
- **e2e 失敗兩處**：
  - nano 的兩組 bf16 KV config OOM：nano_bench 用 max_batch × max_len 配置 KV，b=64 加上 4096 的點等於要 281k token、36 GB。改成依最大的那個 run 配置（fd8102c）。
  - vLLM 的 fp8 run 全部失敗：torch inductor 編出的 Triton kernel 報 `type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')`，和我們碰到的限制一樣。vLLM 0.19.1 在 A100 上用 TRITON_ATTN 開不了 fp8 KV（FLASH_ATTN 本來就拒絕）。下一輪試 FLASHINFER。
- **e2e 的 nano fp8**（ms/step，括號是 R1 的 bf16 KV）：bf16 權重 12.09（12.05）/ 14.01（14.12）/ 20.87（22.60），W4A16 5.79（5.74）/ 8.46（8.47）/ 21.50（23.27）；b=16、4096 token：bf16 18.07、W4A16 12.56。b=64 省 1.73–1.77 ms，比 G4 的 2 ms 少一點；b≤16 沒有差別，和 kbench 一致。
- **quality**（wikitext-2 test，288,627 個 token）：

  | 設定 | perplexity | top-1 一致率（對 bf16） |
  |---|---:|---:|
  | bf16 | 7.232 | - |
  | bf16 + fp8 KV | 7.248（+0.22%） | 97.55% |
  | W4A16 | 8.024 | 85.88% |
  | W4A16 + fp8 KV | 8.040（+0.20%） | 85.79% |

  perplexity 遠低於 G4 的 1%，top-1 97.55% 比 98% 低 0.45 個百分點。參考：Phase 1 的 bf16 Triton 對 torch 路徑，光是數值差異就是 98.81%。

## 2026-10-23 07:16 — run `kernels-r3-20261023-071605-colab`：fp8 decode 的 launch 設定（同一台 VM）

`P1_STEPS="tune summarize" P1_TUNE=decode_fp8`：block_n 32–256 × warps 4/8 × stages 1–4，在 fp8 cache 上掃（µs，目前設定 → 最佳）：

| shape | 目前（bf16 的級距） | 最佳 | 最佳設定 |
|---|---:|---:|---|
| B=1 ctx=1024 | 14.2 | 12.0 | block_n 32、4 warps、2 段 |
| B=1 ctx=4096 | 22.9 | 19.3 | block_n 64、4 warps、1 段 |
| B=16 ctx=1024 | 58.5 | 51.3 | block_n 128、4 warps、2 段 |
| B=16 ctx=4096 | 188.7 | 154.5 | block_n 128、4 warps、4 段 |
| B=64 ctx=1024 | 147.9 | 147.7 | bf16 的級距 |
| B=64 ctx=4096 | 541.9 | 541.9 | bf16 的級距 |
| B=128 ctx=1024 | 271.9 | 271.9 | bf16 的級距 |

設定只能看 batch size（CUDA graph 要固定），所以每一級挑兩種 ctx 下最差情況最小的：B<16 用 block_n 32、4 warps、2 段（最差離最佳 2.3%），B≥16 用 block_n 128、4 warps、3 段（最差 4.8%，B≥64 就是原本的級距）。`DECODE_CONFIGS_FP8`、`decode_config(batch, kv_fp8)`（c235073）。

## 2026-10-23 07:21 — run `kernels-r4-20261023-072114-colab`：fp8 KV 的 e2e，vLLM 改用 FLASHINFER（同一台 VM）

`P1_STEPS="kbench e2e summarize"`，nano_bench 改成依最大的 run 配置 KV（fd8102c），fp8 用新的 decode 級距（c235073），vLLM 的 fp8 改用 `--attention-backend FLASHINFER`。

- `colab exec` 在 run 中途斷線（"colab exec ended (exit 1) before the run finished"），run 在 VM 上照常跑完，之後用 `tail` 確認、`fetch` 下載。
- **kbench**（fp8，新級距）：B=1 12.0 / 19.8 µs（ctx 1024 / 4096，bf16 是 14.4 / 20.9），B=16 53.9 / 156.1 µs（bf16 62.3 / 204.6），B≥64 不變。
- **FLASHINFER 讓 vLLM 在 A100 上開得了 fp8 KV**（JIT 用 cuda129 的 nvcc）。
- **e2e**（ms/step）：

  | | b=1 | b=16 | b=64 | b=16，4096 token |
  |---|---:|---:|---:|---:|
  | nano bf16 | 12.05 | 14.13 | 22.60 | 18.73 |
  | nano bf16 + fp8 KV | **12.01** | **13.81** | **20.87** | 17.16 |
  | vLLM bf16（FA2） | 12.60 | 14.82 | 23.04 | - |
  | vLLM bf16 + fp8 KV（FLASHINFER） | 13.05 | 14.50 | 21.30 | - |
  | nano W4A16 | 5.68 | 8.48 | 23.27 | 13.01 |
  | nano W4A16 + fp8 KV | **5.65** | 8.25 | 21.52 | 11.60 |
  | vLLM AWQ | 5.56 | 7.86 | 16.19 | - |
  | vLLM AWQ + fp8 KV（FLASHINFER） | 6.05 | **7.55** | **14.44** | - |

  - nano 的 fp8 KV：b=64 省 1.73–1.75 ms，b=16、4096 token 省 1.41–1.57 ms，b=16、1024 省 0.23–0.32 ms，b=1 不變。G4 的速度門檻（b=64 省 2 ms）差 0.27 ms。
  - 同樣開 fp8 KV，nano bf16 比 vLLM 快 8% / 5% / 2%；vLLM 的 FLASHINFER 在 b=1 比它自己的 FA2 慢 0.45 ms，所以 nano W4A16 + fp8 KV 在 b=1 反而比 vLLM AWQ + fp8 KV 快。
  - vLLM 的 4096 點全部失敗：`vllm_offline.py` 的 `max_model_len` 寫死 4096，裝不下 4096 + 輸出。改成依 run 設定。

## 2026-10-23 07:53 — run `kernels-serve-20261023-075342-colab`：線上，greedy 和 sampled，bf16 和 fp8 KV（同一台 VM，Phase 2 的 harness）

`BENCH_PHASE=phase2 P2_STEPS="tests serve summarize" P2_CONCURRENCY="1 16 64"`；nano 用 Phase 2 最後一輪的設定（core process、async、2048-token chunk），KV 改成依 0.9 的 GPU 記憶體自動配置（`NUM_KVCACHE_BLOCKS=0`，和 vLLM 的預設一樣），vLLM 的 fp8 用 FLASHINFER。

- **tests**：514 passed，沒有 skip、沒有失敗（含修過容差的 fp8 測試；比 R2 少的 30 個是 revert 掉的 GEMV 預取測試）。
- **KV 容量**：nano 157,744 / fp8 305,920 token（1.94 倍），vLLM 159,072 / fp8 326,352（2.05 倍，vLLM 的 fp8 是每個 tensor 一個 scale，不佔額外 bytes）。
- **W1**（output tok/s · ITL p50 ms）：

  | | c=1 | c=16 | c=64 | TTFT p50 c=64 |
  |---|---:|---:|---:|---:|
  | vLLM | 78 · 12.80 | 940 · 14.87 | 1,777 · 23.24 | 495 ms |
  | vLLM fp8 | 76 · 13.12 | 952 · 14.57 | 1,915 · 21.38 | 470 ms |
  | nano | 82 · 12.08 | 976 · 14.13 | 1,839 · 22.65 | 250 ms |
  | nano fp8 | 83 · 12.04 | 985 · 13.81 | 1,880 · 21.00 | 259 ms |

  c=64 的吞吐量兩邊都比 Phase 2 最後一輪低（vLLM 1,976、nano 1,997），c=64 本來就隨 request 的排列浮動；同一輪裡 nano fp8 的 ITL 最低。
- **W1s（temperature 0.7、top-p 0.9）**：nano 78 / 855 / 1,711 tok/s，vLLM 75 / 609 / 1,693；ITL p50 nano 12.50 / 15.17 / 25.31 ms，vLLM 13.07 / 16.28 / 25.03 ms。c=64 時 nano 的 ITL 比 vLLM 高 1.1%，沒有到 G3 的線上門檻（3%）；G3 由 offline 的 b=64 成立（採樣多 0.96 ms）。vLLM 在 c=16 sampled 只有 609 tok/s（greedy 940），TTFT p50 489 ms，原因沒查。
- **W2b**（48 個 4096-token prefix，總長超過 bf16 的 KV 容量，c=16）：

  | | req/s | TTFT p50 | 命中率 |
  |---|---:|---:|---:|
  | nano | 3.33 | 405 ms | 48.2% |
  | nano fp8 | 3.99 | 160 ms | 70.6% |
  | vLLM | 3.22 | 544 ms | 48.9% |
  | vLLM fp8 | 4.16 | 259 ms | 70.6% |

  fp8 讓兩邊的命中率都從約 48% 升到 70.6%；nano fp8 的 req/s 比 vLLM fp8 低 4%，TTFT p50 低 38%。
- 跑完把 Colab VM 關掉了。

## 2026-10-23 09:22 — run `marlin-t1-20261023-092256-colab`：Marlin 式 W4A16 第一版（新 VM `marlin`）

清單第二輪（#2）。`gemm_w4a16_tiled`（5feb015）：tiled layout、4 段 cp.async、persistent striped grid、slot 式跨 block 相加。`P1_STEPS="installs cuda129 tests kbench summarize"`，`P1_TESTS` 只跑 `test_nano_vllm_w4a16_kernels.py`。

- **tests**：197 passed，第一次就全過。包括 grid = slice 數（沒有跨 block 相加）、grid 1/3/7（stripe 跨 slice）、fp16、連跑 3 次 bitwise 相同、CUDA graph、tiled 反量化和 rows 的 bitwise 相同。
- **kbench**（µs，tiled / vLLM Marlin；每層 4 個投影加總）：

  | M | qkv | o_proj | gate_up | down | 加總 | 舊 kernel（auto）加總 |
  |---|---:|---:|---:|---:|---:|---:|
  | 1 | 21.0 / 18.8 | 16.3 / 17.4 | 68.4 / 58.8 | 37.0 / 23.3 | 142.7 / 118.3（1.21×） | 126.1 |
  | 16 | 22.0 / 21.4 | 17.3 / 19.4 | 73.7 / 65.1 | 40.2 / 27.7 | 153.2 / 133.6（1.15×） | 156.0 |
  | 32 | 27.2 / 27.4 | 21.8 / 22.0 | 88.4 / 71.9 | 48.9 / 37.0 | 186.3 / 158.3（1.18×） | 263.7 |
  | 64 | 44.9 / 45.0 | 40.1 / 35.7 | 120.6 / 113.3 | 78.8 / 62.7 | 284.4 / 256.7（1.11×） | 457.2 |

  - M≥32 大幅改善：M=64 從 457 降到 284 µs。qkv、o_proj 在 M ≤ 32 已經追平或超過 Marlin。
  - down_proj 在每個 M 都只有 Marlin 的 63–80%，gate_up 在 M ≤ 16 也差 12–14%。M=1 反而比舊的 GEMV 慢。

## 2026-10-23 09:30 — run `marlin-t1b-20261023-093008-colab`：ncu（tiled 對 Marlin）

`P1_STEPS="tune ncu"`，ncu 只錄 down、gate_up、o_proj 的 M=1 和 M=64。tune 因為 `tune.py` 的 default 函式少傳 `tiled_grid` 而失敗（已修）。

ncu（base clock，每次 launch 前清 cache）：（更正：`ncu_run` 用 `--clock-control none`，其實是 boost clock，見 run `down-d1`）

| | µs | DRAM % | tensor % | issue/cycle | 前幾名 stall |
|---|---:|---:|---:|---:|---|
| tiled down M=1 | 39.4 | 53.0 | 15.4 | 0.43 | wait、barrier、math_pipe_throttle |
| Marlin down M=1 | 33.3 | 62.6 | 9.5 | 0.35 | long_scoreboard、wait、barrier |
| tiled down M=64 | 81.3 | 27.7 | 30.6 | 0.34 | wait、math_pipe_throttle |
| Marlin down M=64（128 threads） | 67.9 | 33.5 | 36.0 | 0.24 | wait、barrier |
| tiled gate_up M=64 | 121.4 | 42.0 | 40.0 | 0.34 | wait、math_pipe_throttle |
| Marlin gate_up M=64 | 90.7 | 51.5 | 53.4 | 0.30 | wait、math_pipe_throttle |

- **M=1 不是頻寬限制，是指令和延遲**：同樣的工作，tiled 發出的指令約多 45%（issue 0.43 × 39.4 µs 對 0.35 × 33.3 µs）。Marlin 從 ncu 的 base clock 換到 kbench 的 boost clock 快了 43%（33.3 → 23.3 µs），tiled 只快 6%（39.4 → 37.0）。（更正：兩邊都是 boost clock；差別是 ncu 清 cache 的冷權重對 kbench 留在 L2 的暖權重，kbench 量 down_proj 時只有 1 份權重，見 run `down-d1`。）
- **指令多在哪**：每個 tile 都要算 `t / k_tiles` 的除法和 64 位元位址；每個 warp 每個 tile 只算 2 個 k16 列，fragment 在 stage 之間要先等 ldmatrix 和 LDS 回來才能打 mma。
- **改法**：
  - 位址改成遞增：每個 thread 的權重偏移只算一次，載入端和計算端各自維護 (slice, k tile) 游標；
  - fragment double buffer：下一列的 fragment 在打這一列的 mma 之前就先讀；每個 stage 最後一列之前先補下一個 stage、等它到，再讀它的第一列（Marlin 的做法）；
  - 小 M 加 256x128 tile：每次 barrier 搬 16 KB，在路上的權重也加倍。

## 2026-10-23 10:12 — run `marlin-t1c-20261023-101227-colab`：double buffer、遞增位址、256x128（新 VM）

第一次跑 T1c 時 Colab VM 中途消失（`colab exec` 斷線，之後 session 404），改開新 VM 從 installs 重跑。

- **tests**：207 passed（多了 256x128 的排程案例）。
- **kbench**（µs，tiled / Marlin；加總對 Marlin）：

  | M | qkv | o_proj | gate_up | down | 加總 |
  |---|---:|---:|---:|---:|---:|
  | 1 | 18.6 / 19.1 | 16.5 / 17.4 | 55.9 / 58.8 | 30.9 / 23.3 | 1.03 |
  | 16 | 19.2 / 21.2 | 16.6 / 19.8 | 62.0 / 65.2 | 33.1 / 27.6 | 0.98 |
  | 32 | 24.2 / 27.5 | 21.7 / 22.2 | 77.3 / 72.1 | 43.9 / 37.4 | 1.05 |
  | 64 | 43.9 / 45.2 | 40.0 / 35.9 | 112.5 / 100.9 | 76.1 / 63.0 | 1.11 |

  M ≤ 16 追平 Marlin，gate_up 在 M ≤ 16 比 Marlin 快。
- **tune**（`w4a16_tiled`）：
  - M ≤ 32 每一層都是 256x128、每個 SM 1 個 block 最快；每個 SM 2 個 block 一律比較慢，MT=1 已經用了 168 個以上的暫存器，兩個 block 也擺不下。
  - M=64：qkv、o_proj 用 64x128 比較快（41.9 / 33.8 µs，64x256 是 44.1 / 39.9），gate_up 還是 64x256 快（112.6 對 141.2），down 兩者差不多。
  - 超過 64 列：每 64 列 launch 一次，M=128 時四層加總 531 µs，dequant + F.linear 是 913 µs；M=256 時 1,062 對 1,000，差不多。
  - 所以預設改成 MT≥3 用 64x128（N ≥ 16384 才用 64x256），`tiled_max_rows` 改成 128。
- **down_proj 仍慢**：用 tiles/block 回歸，每個 kernel 約有 6 µs 的固定時間，懷疑是最後的 slot 相加，所以改成：
  - release 只由 thread 0 在 barrier 後做一次 fence（CUTLASS semaphore、Marlin lock 的作法）；
  - 讀 slot 時一次發 4 個 load（MT=3 發 2 個；MT=4 一次 1 個，否則會 spill）；
  - scale 改成 packed 存（每份 4 個暫存器），用的時候才 broadcast，MT=4 才不會 spill。

## 2026-10-23 10:31 — run `marlin-t1d-20261023-103133-colab`：slot 相加和預設 tile

- **tests**：207 passed。
- **kbench** 加總對 Marlin：M=1 1.02、M=4 1.01、M=8 1.02、M=16 0.98、M=32 1.01、M=64 1.04。o_proj M=64 用 64x128 之後是 35.6 µs（Marlin 35.8）。
- **down_proj 沒變**：M=1 30.7 對 23.6 µs，M=64 80.9 對 63.0。固定時間不在 slot 相加。
- **改看 SASS**：把 cubin 用 nvdisasm 反組譯（Docker 裡另裝 `cuda-nvdisasm-12-8`）：
  - MT=1 256x128 的主迴圈裡，每對 code 要 5 個指令：SHF、LOP3（AND）、LOP3（OR）、HFMA2（減 136）、HFMA2（乘 scale）；
  - nvcc 沒有把 `(w >> 4s) & 0x000F000F | magic` 合成一個三輸入的 LOP3。
  - 拿 o_proj（K=4096）和 down（K=14336）相減，增加的部分約 1.46 TB/s，所以 down 慢的原因不是頻寬，而是每個 16 KB tile 要約 1.5 µs 的指令時間。
- **改法**：用 inline `lop3.b32 …, 0xEA`（(a & b) | c），和 Marlin 一樣。LOP3 從 165 個降到 101 個，全部 instantiation 仍然 0 spill。

## 2026-10-23 10:52 — run `marlin-t1e-20261023-104222-colab`：lop3

- **tests**：207 passed。
- **kbench**（µs，tiled / Marlin）：

  | M | qkv | o_proj | gate_up | down | 加總對 Marlin |
  |---|---:|---:|---:|---:|---:|
  | 1 | 17.5 / 19.1 | 15.8 / 17.5 | 52.4 / 59.1 | 28.8 / 23.3 | 0.96 |
  | 4 | 17.6 / 19.1 | 15.8 / 17.6 | 53.7 / 59.1 | 28.9 / 23.3 | 0.97 |
  | 8 | 17.9 / 19.1 | 15.7 / 17.6 | 54.7 / 59.7 | 29.9 / 23.2 | 0.99 |
  | 16 | 18.2 / 21.2 | 16.1 / 19.6 | 62.6 / 65.4 | 30.9 / 27.5 | 0.96 |
  | 32 | 22.9 / 27.5 | 20.6 / 22.2 | 72.4 / 72.0 | 41.3 / 37.4 | 0.99 |
  | 64 | 40.0 / 45.1 | 33.4 / 35.8 | 105.5 / 112.1 | 73.1 / 63.0 | 0.98 |

  - 一個 lop3 就讓 M=1 的加總從 121.2 降到 114.5 µs，代表 M=1 確實是被指令數卡住。
  - G2 的加總門檻在 M=1、16、64 都過了。單點方面，down_proj 是 Marlin 的 1.10–1.29 倍，M=8 的 1.29 超過 1.25 倍的門檻。
- commit f541f3a。

## 2026-10-23 10:46 — run `marlin-t2-20261023-104627-colab`：全套測試、kbench、e2e（同一台 VM）

`P1_STEPS="tests kbench e2e summarize"`，nano 設定是 bf16_async、w4a16_rows（`async+rows`）、w4a16_tiled（`async+tiled`），vLLM 跑 bf16 和 AWQ。

- **tests**：629 passed，沒有 skip。
- **kbench**：和 t1e 相同（加總對 Marlin 0.96 / 0.97 / 0.99 / 0.96 / 0.99 / 0.98）。tiled 反量化和 rows 一樣快：qkv 52.6 對 52.4 µs，gate_up 242.2 對 238.7 µs。
- **e2e**（ms/step）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | nano bf16 | 12.06 | 14.12 | 22.61 |
  | vLLM bf16 | 12.61 | 14.84 | 23.07 |
  | nano W4A16 rows | 5.70 | 8.29 | 23.03 |
  | nano W4A16 tiled | **5.35** | **7.31** | 16.82 |
  | vLLM AWQ | 5.56 | 7.85 | **16.23** |

  - G3 的三個門檻都過了，預設 layout 改成 tiled。
  - b=64 還慢 vLLM AWQ 0.59 ms，但 kbench 的 GEMM 加總已經比 Marlin 少：拿 bf16 的 e2e 扣掉 cuBLAS 的時間，vLLM 的 AWQ GEMM 在 e2e 裡每層約 224 µs，比 kbench 的 `gptq_marlin` 少 30 µs，原因沒查。
- **nsys**（tiled，b=1 / 16）：5.38 / 7.18 ms/step，GPU 空檔 0.04 ms，每步 330 / 332 個 kernel，GEMM 佔 84% / 66%。
- 跑完把 Colab VM 關掉了。

## 2026-10-23 23:33 — run `down-d1-20261023-233303-colab`：down_proj 研究，冷的量測（新 VM `down`）

清單第三輪（down_proj）。kbench 的 `copies` 改依每次呼叫實際讀的 4-bit 權重決定（0be78c1），kernel 加 trace build（b653bb7），新增 `tiled_trace.py`、`golden.py`（99384a0）。`P1_STEPS="installs cuda129 tests kbench ncu trace golden summarize"`，`P1_KERNELS=w4a16_linear`，ncu 只錄 down 的 M=1/8/64 和 o_proj、gate_up 的 M=1，加錄 `MemoryWorkloadAnalysis_Tables`、`SourceCounters`。

- **tests**：213 passed（多了 6 個 trace 的測試）。
- **kbench（冷）**：down_proj 現在輪替 18 份權重。tiled / Marlin（µs）：

  | M | 1 | 4 | 8 | 16 | 32 | 64 |
  |---|---:|---:|---:|---:|---:|---:|
  | down，第三輪冷 | 30.7 / 33.6 | 31.2 / 33.5 | 31.5 / 33.7 | 32.8 / 36.9 | 41.1 / 48.3 | 73.3 / 81.5 |
  | down，第二輪暖（T2） | 28.8 / 23.3 | | 30.0 / 23.3 | 31.0 / 27.5 | 41.4 / 37.4 | 73.1 / 63.1 |

  - 冷的時候 down 在每個 M 都比 Marlin 快（0.85–0.94 倍）；第二輪的 1.1–1.3 倍是 Marlin 吃到 L2 裡的暖權重。我們的 kernel 暖的時候只快 4–7%。
  - 四層加總對 Marlin：0.90 / 0.91 / 0.92 / 0.88 / 0.94 / 0.93（M=1/4/8/16/32/64）。gate_up 的 4-bit 權重 59 MB 比 L2 大，第二輪本來就近乎冷的，數字沒變。
- **ncu（現在的程式，boost clock，清 cache）**：
  - down M=1：tiled 34.9、Marlin 33.2 µs；DRAM 讀的 bytes 一樣（30.3 MB）。
  - **L2→SM 的 sector 是 Marlin 的 2 倍**：down M=1 讀了 1.95M 個 sector、732K 個 request，Marlin 是 989K、248K。SourceCounters 指到 4 條權重的 `LDGSTS`，各有 50% 的「L2 Theoretical Sectors Global Excessive」，Marlin 同樣的 `LDGSTS.E.BYPASS.128` 是 0。
  - 原因：kernel 裡的 static `__shared__ int last_block` 把 dynamic shared memory 推到位址 16（SASS 的 stage 基底是 0x10），stage 不在 128 bytes 邊界上。
- **trace**：大部分 run 的 SM clock 只有約 1.08 GHz（每次 launch 後都回 host 一次，GPU 閒下來就降頻），只有 o_proj M=1/8、gate_up M=1 在 1.39–1.40 GHz。這三列和 kbench 一致（15.4 對 15.6、53.3 對 52.3 µs）：
  - globaltimer 每 1,024 ns 才跳一次；
  - o_proj M=1：每個 block 的第一個 tile 在 2.3–5 µs 之間才到；slice 結尾在關鍵路徑上約 4.3 µs，最後到的 block 的相加（紅色）決定 kernel 結束；
  - 每個 tile 1.16 µs，對每個 tile 分到的 DRAM 時間 1.29 µs。
- golden 存了 16 個輸出（四層 × M=1/8/16/64）。

## 2026-10-23 23:55 — run `down-d2a-20261023-235500-colab`：stage 對齊 128 bytes、trace 不降頻

- **改了什麼**（17a3038、cb241d1）：`last_block` 和 trace 的時間戳改放 dynamic allocation 的尾端，不再有 static shared 變數，stage 從位址 0 開始；trace 改成全部 launch 完才同步一次，前面先熱身。
- **tests**：213 passed。**golden**：16/16 bitwise 相同。
- **ncu**：down M=1 降到 978K sector、245K request，和 Marlin（989K、248K）一樣；DRAM bytes 不變。
- **kbench（冷）**：四層加總 −1.2 / −2.2 / −2.3 / −4.6 / −9.2 / −1.7%（M=1/4/8/16/32/64）；qkv 在 M ≤ 16 快 6–10%；down 幾乎沒變（30.7 → 30.6 µs）。
- **trace**：熱身之後 M ≥ 8 的 run 仍然只有約 1.08 GHz，M=1 是 1.3–1.4 GHz。M=1 和 M=8 是同一個 MT=1 kernel、同樣的指令數，差別只在 x 有幾列不是 0，看起來是 400 W 的功耗上限（tensor core 乘隨機資料比乘 0 耗電）。trace 的跨 run 比較因此只看同 clock 的列。

## 2026-10-24 00:12 — run `down-d2b-20261024-001228-colab`：slice 結尾、L2 預取的上限、e2e 對照

- **改了什麼**（7a0072e、f2e7ab9、c16b343）：
  - slice 結尾一開始 thread 0 就用 `ld.acquire` 讀 counter（和 warp 相加重疊），看到其他 block 都到了就直接加總，不寫自己的 slot、不做 atomic；否則一個 `atom.acq_rel` 取代 fence + atomicAdd + fence；block 最後一次 slice 結尾用已排空的 stage 放每個 warp 的部分和；
  - `tiled_trace.py --prefetch-tiles 0,3`：先把每個 block 前 3 個 tile 拉進 L2 再量；
  - e2e 對 vLLM AWQ 也跑 nsys，`gemm_breakdown.py` 從兩邊的 trace 讀出每個投影的 GEMM 時間。
- **tests**：213 passed。**golden**：16/16 bitwise 相同。
- **kbench（冷）**：四層加總比 D1 −2.8 / −3.6 / −3.8 / −5.8 / −8.7 / −4.0%；down 30.7 → 30.3（M=1）、32.8 → 31.7（M=16）。
- **trace**：最後到的 block 有 72–81%（down，M ≤ 16）走提早的路。前 3 個 tile 在 L2 時（只比同 clock 的列）：qkv、o_proj 的 M=1 快約 2 µs（19.5 → 17.4、14.3 → 12.3），gate_up、down 只快 0–1 µs，因為 prologue 時 DRAM 本來就在搬所有 block 的前 3 個 tile。
- **e2e**（ms/step）：nano W4A16 tiled 5.24 / 7.14 / 16.50，vLLM AWQ 5.54 / 7.85 / 16.18（b=1/16/64）。
- **decode 裡每個投影的 GEMM**（nsys，µs，nano / vLLM AWQ）：

  | b | qkv | o_proj | gate_up | down | 每層 |
  |---|---:|---:|---:|---:|---:|
  | 1 | 16.5 / 16.4 | 15.0 / 14.7 | 52.2 / 55.6 | 30.6 / 31.0 | 114.3 / 117.7 |
  | 16 | 16.5 / 17.8 | 15.7 / 16.3 | 55.2 / 58.8 | 32.1 / 33.3 | 119.6 / 126.2 |
  | 64 | 38.2 / 34.8 | 32.2 / 30.3 | 102.1 / 89.5 | 71.7 / 65.9 | 244.3 / 220.5 |

  down 在 b=1 打平、b=16 快 4%，b=64 慢 9%；b=64 四層都比 vLLM 慢，每步 0.76 ms。kbench 的 Marlin 在 M=64 比 e2e 裡的 AWQ Marlin 慢 15–23%（例如 down 81.6 對 65.9），我們的 kernel 兩邊一樣，所以 M ≥ 16 的 Marlin 對照以 e2e 和 ncu 為準（ncu 的 Marlin down M=64 是 68 µs）。

## 2026-10-24 00:37 — run `down-d2c-20261024-003658-colab`：M=64 的 slice 結尾（退回）

- **改了什麼**（7e30049）：放得下的 tile 設定讓一欄的前 n−1 個 warp 同時寫自己的 buffer、最後一個 warp 依序加上自己的；最後一次 slice 結尾一次讀 4 個（MT=4 讀 2 個）slot。
- **tests**：213 passed。**golden**：16/16 bitwise 相同。
- **M=64 變慢**：trace（down，同 clock）的 warp 相加從 2.17 降到 1.05 µs，但每個 tile 從 1.19 變 1.25 µs，最後一個 block 的相加也從 6.2 變 6.9 µs；kbench down M=64 72.0 → 81.3 µs，e2e b=64 16.50 → 16.93 ms（down 81.4 µs）。MT=4 卡在 255 個暫存器，主迴圈的排程變差。
- M=1–32 的 kbench 加總快 0.4–1%，e2e b=16 7.10 ms。
- **處理**（8537411）：這兩項只留給 MT 1–2，MT 3–4 回到 D2b 的寫法（SASS 指令數相同，只有排程順序不同）。另外加了 MT 3–4 的 128x128 tile（de05b0f，不是預設），給 tune 比較。

## 2026-10-24 00:53 — run `down-d2d-20261024-005304-colab`：MT 3–4 退回之後、加上 128x128

- **tests**：223 passed（多了 128x128 在 40、64 列的排程案例）。**golden**：16/16 bitwise 相同。
- **kbench（冷）**：M=64 回到 D2b 的水準（四層加總 243.2 µs，down 72.0）；M=1–32 比 D2b 快 0.4–1%，比 D1 快 3.2–9.6%。
- **e2e**：nano W4A16 tiled 5.25 / 7.10 / 16.49 ms，vLLM AWQ 5.55 / 7.85 / 16.17 ms。decode 裡的 GEMM 和 D2b 相同（b=64 每層 244.1 對 220.9 µs）。
- **tune 失敗**：`default()` 把 M=128 直接交給 `tiled_launch_config`，m16 數是 8，沒有 tile（KeyError）；`tiled_max_rows` 從 256 改成 128 之後才出現。崩潰前 qkv M=64 已經顯示 128x128 34.7 對 64x128 38.2 µs。修正見 754b882。

## 2026-10-24 01:09 — run `down-d2e-20261024-010900-colab`：tune（冷）

`P1_STEPS="tune summarize"`。每個 SM 1 個 block、tiled_max_rows 256 的 µs：

| M | qkv：128x128 / 64x256 / 64x128 | o_proj | gate_up | down |
|---|---|---|---|---|
| 64 | **34.7** / 41.4 / 38.2 | **27.9** / 38.0 / 31.6 | 111.8 / **102.5** / 128.6 | **62.4** / 72.3 / 72.0 |
| 128（兩次 launch） | **68.6** / 82.6 / 75.9 | **55.1** / 75.8 / 63.6 | 223.6 / **206.0** / 257.1 | **124.7** / 144.6 / 143.9 |

- M ≤ 32 仍然是 256x128 最快（o_proj M=16 的 128x128 只差 0.1 µs）。
- M=64 的 128x128 對 vLLM AWQ 在 decode 裡的時間：qkv 34.7 對 34.9、o_proj 27.9 對 30.4、down 62.4 對 66.0。gate_up 留在 64x256（102.5 對 89.5），是 b=64 剩下最大的差距。
- 預設改成 MT 3–4 用 128x128，N ≥ 16384 才用 64x256（eb6b2a0）；M=64 的輸出因此換了加總順序。
- M=256 時 down、gate_up 的 tiled 仍比反量化 + F.linear 快（249 對 258、416 對 554 µs），qkv、o_proj 則是反量化快；`tiled_max_rows` 沒改。

## 2026-10-24 01:12 — run `down-d3-20261024-011151-colab`：全套測試、tune、e2e（同一台 VM）

`P1_STEPS="tests golden kbench tune trace e2e summarize"`，nano 設定和第二輪 T2 相同（bf16_async、w4a16_rows、w4a16_tiled，都開 async），paged b=1/16/64。

- **tests**：645 passed，沒有 skip。
- **golden**：13/16 bitwise 相同；不同的是 qkv、o_proj、down 的 M=64，因為預設 tile 換成 128x128（最大差 0.0078，51–148 個元素）。這一步的 rc=1 是預期的。
- **kbench（冷）**：四層加總比 `down-d1` 快 3.1 / 4.2 / 4.1 / 6.6 / 9.6 / 10.5%（M=1/4/8/16/32/64）；down 30.7 → 30.1、32.8 → 31.5、73.3 → 62.4 µs（M=1/16/64）。
- **tune**：M=48 也是 128x128 最快（qkv 29.0、o_proj 24.0、down 52.9 µs；gate_up 64x256 85.8）；其他和 `down-d2e` 相同。
- **trace**：down M=64（128x128，1.08 GHz）：prologue 2.1 µs、每個 tile 1.72 µs、slice 結尾 5.4 µs；down M=1（1.25 GHz）：kernel 30.7 µs，最後到達的 59% 走提早的路。
- **e2e**（ms/step）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | nano bf16 | 12.04 | 14.12 | 22.57 |
  | vLLM bf16 | 12.64 | 14.84 | 23.04 |
  | nano W4A16 rows | 5.68 | 8.29 | 22.94 |
  | nano W4A16 tiled | **5.19** | **7.10** | **15.94** |
  | vLLM AWQ | 5.55 | 7.85 | 16.19 |

- **decode 裡的 GEMM**（nano / vLLM AWQ，µs）：down 30.4 / 31.0、31.7 / 33.3、62.0 / 66.0（b=1/16/64）；每層 114.2 / 117.6、118.3 / 126.0、227.0 / 220.8。b=64 只剩 gate_up 慢（102.0 / 89.6）。
- 跑完把 Colab VM（`down`）關掉了。

## 2026-10-24 02:58 — run `concur-c1-20261024-025830-colab`：e2e 在 b=128、256（新 VM `concur`）

`P1_STEPS="installs cuda129 e2e summarize"`，`P1_NANO_CONFIGS="w4a16_tiled:...:async+tiled w4a16_tiled_full:...:async+tiled+fullgraph"`，`P1_E2E_RUNS="paged:64:1024:256,paged:128:1024:256,paged:128:512:256,paged:256:512:256"`，`P1_NSYS_RUNS="64:1024 128:1024 256:512"`。程式在 bb80af4（`--full-graphs`、`P1_NSYS_RUNS`、`gemm_breakdown` 合併連續 launch 和反量化前置 kernel）。

- **e2e**（ms/step，nano 預設 / fullgraph / vLLM AWQ）：
  - b=64/1k：16.02 / 16.04 / 16.20（和 `down-d3` 的 15.94、16.19 一致）；
  - b=128/1k：30.57 / 30.20 / 27.45；
  - b=128/512：24.57 / 24.21 / 21.42；
  - b=256/512：50.44 / 49.99 / 39.04。
- **vLLM bf16**：27.28、25.35、32.86 ms（b=128/1k、128/512、256/512）。KV cache 只有 142–144k token，b=128/1k 和 b=256/512 會 preempt，不能比。
- **decode 裡的 GEMM**（每層 µs，nano / vLLM AWQ）：b=64 227.1 / 221.4，b=128 454.2 / 353.1，b=256 1008.0 / 632.4。
- **b=256 的 nano**：反量化 kernel 每次平均 104 µs（每步 13.4 ms），cuBLAS 每層約 590 µs。
- **attention**：每層約 404–416 µs，和 vLLM 的 419 µs 相同。
- **VM**：跑完把 Colab VM（`concur`）關掉了。

## 2026-10-24 03:51 — run `chunk-g0-20261024-035150-colab`：舊 kernel 的 golden（新 VM `chunk`）

在 09e2a72（改動之前）的 worktree 上跑 `P1_STEPS="installs cuda129 golden"`，把四層 × M=1/8/16/64 的輸出存到 VM 上的 `w4a16_golden.pt`，之後的 run 用它檢查 64 列以內 bitwise 不變。

## 2026-10-24 03:56 — run `chunk-c2-20261024-035612-colab`：作廢

68 個測試失敗，訊息是舊 kernel 的「1 to 64 rows per launch」。tarball 保留本機檔案的 mtime，而我改程式的時間比 VM 上剛 build 的舊 kernel 早，ninja 認為 extension 是新的，載入了舊版。手動停掉，`colab_launcher.py` 解開 tarball 後把所有檔案的 mtime 設成現在（834a510），之後每次 rerun 都會重新 build 改過的 kernel。

## 2026-10-24 04:02 — run `chunk-c2-20261024-040251-colab`：一次 launch 處理多個列區塊（d43d122）

`P1_STEPS="tests golden kbench tune trace ncu summarize"`，trace 在 down、gate_up、qkv 的 M=64/128/256，ncu 錄 down M=64/128 的 WarpStateStats、SchedulerStats、SourceCounters。

- **tests**：711 passed，沒有 skip（多了 65–512 列的 chunk 案例）。**golden**：16/16 bitwise 相同。
- **kbench**（冷，µs，tiled / Marlin / 已反量化的 cuBLAS bf16）：
  - M=128：qkv 55.2 / 62.1 / 56.1，o_proj 41.4 / 52.7 / 41.3，gate_up 193.7 / 170.4 / 271.0，down 110.3 / 109.7 / 111.7；
  - M=256：qkv 96.4 / 82.5 / 61.1，o_proj 69.0 / 59.4 / 54.2，gate_up 370.2 / 322.3 / 320.1，down 202.3 / 166.7 / 144.4。
- **tune**：chunk ≥ 3（192、256 列）時 64x256 比 128x128 快 2–5%（qkv、o_proj、down）；反量化 + cuBLAS 在 65 列以上都比較慢。
- **trace**：128x128 每個 tile 在 1.40 GHz 約 1,900 cycle（down M=64 和 M=256 一樣），tensor core 下限是 1,024。
- **ncu**：我們的 top stall 是 wait、selected，Marlin 是 wait、math_pipe_throttle；執行的指令裡 HMMA 佔 14%（Marlin 24%），每個 HMMA 多了約 3 個 IMAD、LEA、SHF。SASS 顯示 x tile 每次 cp.async 前有約 20 個指令的索引運算（runtime 上限的迴圈、有號除法、swizzle、64-bit 位址），stage 交界還有約 20 個暫存器複製。

## 2026-10-24 04:16 — run `chunk-c3-20261024-041652-colab`：e2e（d43d122 + 64x256 規則）

nano 只跑 `w4a16_tiled_full`（fullgraph），其他和 `concur-c1` 相同。

- **e2e**（ms/step，nano / vLLM AWQ）：b=64/1k 16.15 / 16.18，b=128/1k 28.37 / 27.47，b=128/512 22.40 / 21.48，b=256/512 40.96 / 39.09。
- **decode 裡的 GEMM**（每層 µs）：b=64 231.8 / 220.6，b=128 399.7 / 351.2，b=256 722.7 / 624.9。

## 2026-10-24 04:36 — run `chunk-c4-20261024-043613-colab`：較便宜的 load stream（421dcda）

`P1_STEPS="tests golden kbench ncu trace e2e summarize"`。

- **tests**：711 passed。**golden**：16/16 bitwise 相同。
- **kbench**（四層加總，µs，相對 `chunk-c2`）：M=1 −0.9%，M=32 −6.3%，M=64 231.5 → 204.9（−11.5%），M=128 400.6 → 332.0（−17.1%），M=256 737.9 → 582.8（−21.0%）；gate_up M=64 103.7 → 86.5。
- **ncu**：top stall 變成 wait、math_pipe_throttle，SM throughput 48%（Marlin 50%）。
- **trace**：64x256 每個 tile 在 1.40 GHz 約 1,430 cycle（down、qkv M=256）；slice 結尾佔關鍵路徑 27–42%（M=64），128x128 時 4 個 warp 依序相加要 4.6 µs，最後到的 block 逐一讀 slot 5–10 µs。
- **e2e**（ms/step，nano / vLLM AWQ）：b=64 15.20 / 16.17，b=128/1k 26.05 / 27.48，b=128/512 20.08 / 21.43，b=256/512 36.16 / 39.07。
- **decode 裡的 GEMM**（每層 µs）：b=64 202.4 / 220.5，b=128 327.8 / 352.2，b=256 573.4 / 630.8。

## 2026-10-24 05:03 — run `chunk-c5-20261024-050307-colab`：MT 3–4 的 slot 一次讀 4 個（沒保留）

`P1_STEPS="tests golden kbench tune trace summarize"`，只跑 w4a16 的 kernel 測試。

- **kbench**：M=64 +3.6%，M=128 +2.0%，M=256 −0.8%。主迴圈的暫存器多了（MT 4、128x128：245 → 252），和第三輪 `down-d2c` 一樣變慢，revert。
- **tune**：現在的 tile 規則在每個 shape 都是最快或差 2% 以內（qkv M=64 的 64x128、M=128 的 64x256 快 2%）。

## 2026-10-24 05:13 — run `chunk-c6-20261024-051347-colab`：最後一次 slice 結尾移到主迴圈之後（8bcb99f）

- **tests**：276 passed（w4a16 kernel 測試）。**golden**：16/16 bitwise 相同。
- **kbench**（四層加總，相對 `chunk-c4`）：M=1–16 +0.3–0.5%，M=32 −0.7%，M=64 204.9 → 200.3（−2.2%），M=128 332.0 → 326.2（−1.7%），M=256 −0.3%；gate_up 慢約 1%，其他三層快。保留。

## 2026-10-24 05:22 — run `chunk-c7-20261024-052230-colab`：全套測試、e2e 在 b=1 到 256（8bcb99f）

`P1_STEPS="tests golden e2e summarize"`，nano 只跑 `w4a16_tiled_full`，`P1_E2E_RUNS` 加上 static b=1、paged b=1/16，`P1_NSYS_RUNS="1:128 16:1024 64:1024 128:1024 256:512"`。

- **tests**：711 passed，沒有 skip。**golden**：16/16 bitwise 相同。
- **e2e**（ms/step，nano / vLLM AWQ）：b=1 5.24 / 5.55，b=16 7.11 / 7.85，b=64 15.03 / 16.18，b=128/1k 25.83 / 27.48，b=128/512 19.86 / 21.43，b=256/512 36.14 / 39.05。
- **decode 裡的 GEMM**（每層 µs）：b=1 112.4 / 117.6，b=16 118.5 / 126.0，b=64 198.2 / 220.5，b=128 320.4 / 351.0，b=256 571.4 / 625.5。
- **VM**：跑完把 Colab VM（`chunk`）關掉了。

## 2026-10-28 11:21 — 採樣器：本機實作（1cb75e3、e98f015）

到目前為止的 benchmark 都是 greedy，但 nano server 的預設是 temperature 0.7、top-p 0.9。第一輪（`kernels-r1`）量到 nano BF16 採樣的成本是 0.42 / 1.02 / 2.62 ms（b=1/16/64），vLLM 是 0.25 / 1.33 / 1.66。這一輪的目標是做一個不排序的採樣器，在每個 batch size 都比 vLLM 快。分支是從 `claude/inference-kernels`（96bb209）開的本機分支 `claude/fast-sampler`。

### 做了什麼

- **vLLM 0.19.1 的採樣路徑**（讀 v0.19.1 的原始碼）：
  - logits 轉成 fp32，就地除以 temperature；
  - `apply_top_k_top_p`：B ≥ 8 時走 Triton 的 pivot 搜尋，一個 program 處理一列，grid 是 min(SM 數, B)，最多 18 次；B < 8 時走 PyTorch，整列 sort 加 cumsum；
  - softmax 之後 `random_sample`（`exponential_`、除法、argmax）；
  - greedy 是 fp32 副本的 argmax；FlashInfer 的 sampler 要設 `VLLM_USE_FLASHINFER_SAMPLER=1` 才會開。
- **`kernels/csrc/sampler.cu`**：
  - bf16/fp16 的 logits 有保序的 16-bit key，除以正的 temperature 不改變順序，所以 top-k 和 top-p 的截斷點都是一個 key 值，用兩層 histogram（高 8 位 256 bin，再加邊界 bin 的低 8 位）就能精確找到，不用排序；
  - grid 是 (chunks, rows)，每列切成 2048·G 個 token（列數 ≤ 8 時 G=1，≤ 32 時 G=2，其他 G=4）；同一列的 chunk 透過 global atomics 在每列的 workspace 匯合，最後到的 block（ticket，沿用 `w4a16.cu` 的做法）收尾並把 workspace 歸零；
  - 四個 kernel：stats（max、argmax，greedy 列在這裡結束）、hist（count 和質量，質量是 2^45 的 u64 定點）、refine 兼抽樣、只給 top-k + top-p 列的第二次 refine；
  - 抽樣是 exponential race，在保留的 token 中取 v − log E 的 argmax，E 由 Philox4x32-10 產生，key 是 engine seed，counter 是（token index / 4、position、seq id 的低 32 位和高 32 位），所以同一個 request 的結果和它所在的 batch 無關。
- **`sampling.py`**：
  - PyTorch 路徑（CPU、nvcc 不可用時）改成同值保留的規則：top-k 保留比它大的 token 少於 k 個的 token，top-p 保留比它大的 token 質量和小於 p 倍總和的 token（`keep_mask`），抽樣也改成 exponential race（`sample_rows`）；
  - `Sampler`：每列的設定用 pinned staging 兩個 slot 輪流一次上傳，和 decode graph 的輸入一樣。
- **`model_runner.py`**：`launch` 改用 `Sampler`，position 用 `num_tokens + num_inflight`；`config.py` 加 `engine.seed`（`SEED`，預設 0，和 vLLM 相同）。
- **kbench `sampler`**（e98f015）：
  - `Kernel` 加可選的 `check`：抽到的 token 要在 float64 reference 的保留集合內（p 加 1e-4 的容差），greedy 列要等於 argmax；
  - impls 是 vLLM（fp32 副本、temperature、`apply_top_k_top_p`、softmax、`random_sample`）、`nano_torch`（`sample_rows`）、`nano_cuda`（`Sampler`）；
  - shapes 是 peaked 和 flat 兩種分布 × greedy、(T 0.7, p 0.9)、(T 1)、(T 0.7, p 0.9, k 50) × B = 1–256。
- **測試**：
  - CPU：同值規則、top-p 在 top-k 之內、逐 token 的規則（隨機）、`sample_rows` 的分布、`Sampler` 的 staging；
  - GPU 新檔 `tests/gpu/test_nano_vllm_sampler.py`：保留集合（peaked、flat、ties、−inf）、列數和 vocab 大小、fp16、top-k + top-p 的兩種分支、greedy 取第一個最大值、degenerate 列、deterministic、chi-square、graph replay、fp32 走 PyTorch 路徑；
  - `test_nano_vllm_async.py` 加上 sampled 的 token 在 sync 和 async 之間相同。
- **編譯**：本機 docker 裡的 nvcc 編過，0 spill。

## 2026-10-28 11:23 — run `sampler-s1-20261028-112305-colab`：第一版（新 VM `sampler`）

`P1_STEPS="installs cuda129 tests kbench e2e summarize" P1_KERNELS=sampler`；e2e 的 nano 跑 bf16 和 W4A16 tiled，greedy 和 sampled，b=1/16/64；vLLM 跑 bf16 和 AWQ，greedy 和 sampled；nsys 錄 W4A16 sampled 的 b=1/16/64。

- **tests**：776 passed、1 failed，沒有 skip。失敗的是 `test_top_k_and_top_p_cutoffs_in_both_branches`：測試假設每個 peaked 列的 top-p 截斷點都在 top-k 之上的 bin，不成立。改成給前 32 列一個很大的 token（`logits[:32, 0] = 40`），並比較 `key >> KEY_LOW_BITS`。保留集合的測試全過。
- **kbench**：sampled 的列全部出錯（`Offset increment outside graph capture encountered unexpectedly`）。`nano_torch` 在計時的 lambda 裡做 pinned 上傳，graph capture 不了；capture 失敗後 CUDA 的 RNG generator 停在 capture 狀態，之後每一列都失敗。改成在 factory 裡上傳設定。greedy 的列正常（µs，`nano_cuda` / vLLM）：B=1 15.6 / 41.1，B=16 20.4 / 51.0，B=64 34.9 / 106.0，B=256 78.9 / 297.1。
- **e2e**（ms/step，nano / vLLM）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | bf16 greedy | 12.01 / 12.62 | 14.08 / 14.82 | 22.55 / 23.02 |
  | bf16 sampled | 12.10 / 12.87 | 14.44 / 16.14 | 23.64 / 24.76 |
  | W4A16 / AWQ greedy | 5.09 / 5.54 | 7.06 / 7.85 | 15.00 / 16.21 |
  | W4A16 / AWQ sampled | 5.16 / 5.81 | 7.41 / 9.33 | 16.22 / 18.11 |

  - 採樣的成本（sampled 減 greedy，W4A16 / AWQ）：0.07 / 0.27、0.35 / 1.48、1.22 / 1.90 ms。已經在每個 b 都比 vLLM 低，但 b=64 遠高於設計的幾十 µs。
- **nsys**（每步的 sampler kernel，µs）：hist 72 / 323 / 1,382，refine 20 / 51 / 143，stats 5 / 11 / 23（b=1/16/64）。
- **原因**（看 SASS）：sm_80 上 shared memory 的 u64 `atomicAdd`、`atomicMax` 和 f32 `atomicAdd` 都編成 compare-and-swap 迴圈（`ATOMS.CAST.SPIN`），只有 32-bit 整數的 add、min、max 是原生指令（`ATOMS.ADD`、`ATOMS.MIN.S32`、`ATOMS.MAX`）；global 的 u64 add、max 是原生的（`RED.E.ADD.64`、`RED.E.MAX.64`）。key 的高 8 位只有 sign 和 7 個 exponent bit，每個 bin 涵蓋 4 倍的數值範圍，所以一列的 logits 集中在少數幾個 bin，CAS 迴圈互相衝突。

## 2026-10-28 12:08 — run `sampler-s2-20261028-120816-colab`：原生 32-bit shared atomics、更細的 bin、top-p 的界、獨立的 draw（a926db2，同一台 VM）

### 改了什麼

- 質量的定點 scale 改成 2^37，加進 shared memory 時拆成兩個 u32（在 bit 19 切開），進 global memory 時合成 u64；一個 block 最多 8192 個 token，兩半都不會溢位。
- 第一層 bin 從 256 改成 1024（key 的高 10 位），第二層是 64。
- 只有 top-p 的列：比 max 低超過 ln(V / (1 − p)) nats 的 token 一定不會保留（它們全部加起來不到 max token 質量的 1 − p），只用 register 加進總和，不進 histogram。
- 抽樣拆成自己的 kernel（draw），不再需要 shared 的 u64 max；draw 加 `__launch_bounds__(256, 2)`，G=4 的版本才不會 spill。
- 本機用 Python 逐行模擬新的截斷邏輯，3,600 個案例和 reference 沒有差異（包含 top-k + top-p 的兩種分支）。

### 結果

`P1_STEPS="tests kbench e2e summarize"`，其他和 S1 相同。

- **tests**：777 passed，沒有 skip。
- **kbench**（µs，`nano_cuda` / vLLM，每一列的 check 都過）：
  - greedy：B=1 15.7 / 41.1，B=16 19.6 / 51.0，B=64 30.5 / 105.8，B=256 79.0 / 296.3；
  - (T 0.7, p 0.9) peaked：32.8 / 269.0、51.9 / 550.9、104.5 / 882.2、301.1 / 2,577.1；
  - (T 0.7, p 0.9) flat：35.0 / 270.8、62.8 / 1,456.7、149.1 / 2,064.3、459.5 / 7,154.4；
  - (T 1) peaked：21.1 / 103.5、36.9 / 136.7、90.2 / 362.9、273.7 / 1,296.0；
  - (T 0.7, p 0.9, k 50) peaked：39.9 / 304.8、62.9 / 426.2、128.0 / 719.0、382.4 / 2,292.5；
  - 8 個 B、全部設定和兩種分布都比 vLLM 快：greedy 2.6–3.8 倍，採樣 3.5–30 倍。`nano_torch`（PyTorch 路徑）0.3 ms（B=1）到 10.7 ms（B=256）。
- **e2e**（ms/step，nano / vLLM）：

  | | b=1 | b=16 | b=64 |
  |---|---:|---:|---:|
  | bf16 greedy | 12.01 / 12.61 | 14.08 / 14.82 | 22.55 / 23.02 |
  | bf16 sampled | 12.03 / 12.87 | 14.14 / 16.15 | 22.68 / 24.76 |
  | W4A16 / AWQ greedy | 5.09 / 5.54 | 7.06 / 7.84 | 15.00 / 16.20 |
  | W4A16 / AWQ sampled | 5.11 / 5.81 | 7.11 / 9.33 | 15.13 / 18.11 |

  - 採樣的成本（sampled 減 greedy）：W4A16 0.02 / 0.05 / 0.13 ms，AWQ 0.27 / 1.49 / 1.91；bf16 0.02 / 0.06 / 0.13，vLLM bf16 0.26 / 1.33 / 1.74。
  - nano sampled 比 vLLM sampled 快：W4A16 12% / 24% / 16%，bf16 7% / 12% / 8%。
- **nsys**（每步的 sampler kernel，µs，b=1/16/64）：hist 9.5 / 25.8 / 67.8，draw 5.1 / 17.4 / 51.7，refine 6.1 / 12.9 / 29.7，stats 5.1 / 10.4 / 22.9；合計 26 / 67 / 172（S1 是 96 / 385 / 1,548）。
- **門檻**：G1 過；G2 的「每個 B 都比 vLLM 快」和「greedy 比原本的 argmax 快」過，「b=64 (0.7, 0.9) ≤ 60 µs」沒過（peaked 104.5、flat 149.1）；G3 的「sampled 減 greedy 低於 vLLM」和「sampled 比 vLLM 快」過，「≤ 0.1 ms」在 b=64 沒過（0.13）。
- **b=64 剩下的時間**：只讀 logits 的 stats 是 23 µs，四個 kernel 合計 172 µs。benchmark 的 prompt 是隨機 token，推測每列的分布很平（e2e 的 kernel 時間比 kbench 的 flat 還長）：hist 裡很多 token 落在同幾個 bin，同一個 warp 的 lane 在 u32 atomics 上仍然衝突重播；draw 的 nucleus 很大，幾乎每個 warp step 都有保留的 token，每個 token 付 Philox、`log1pf`、`__logf` 約 50 個指令。改善方向：warp-aggregated atomics（`__match_any_sync` 加 `__reduce_add_sync`），以及每列只用一個亂數的 inverse-CDF 抽樣。這次沒做。

## 2026-10-28 12:47 — run `sampler-serve-20261028-124708-colab`：線上，greedy 和 sampled（同一台 VM，Phase 2 的 harness）

`BENCH_PHASE=phase2 P2_STEPS="serve summarize" P2_CONCURRENCY="1 16 64"`，`P2_RUNS="vllm:vllm:w1,w1s nano:nano:w1,w1s:ENGINE_MULTIPROCESS=true,MAX_NUM_BATCHED_TOKENS=2048,NUM_KVCACHE_BLOCKS=0"`，和 `kernels-serve` 的設定相同；tests 已經在 S2 跑過，這次不跑。程式在 a926db2。

- **W1**（greedy，output tok/s · ITL p50 ms）：nano 83 · 12.03、979 · 14.08、1,840 · 22.62；vLLM 78 · 12.78、937 · 14.83、1,780 · 23.22（c=1/16/64）。
- **W1s**（temperature 0.7、top-p 0.9）：nano 81 · 12.05、904 · 14.16、1,833 · 22.73；vLLM 75 · 13.05、612 · 16.27、1,692 · 25.00。
  - 採樣讓 ITL p50 多 nano 0.02 / 0.08 / 0.11 ms，vLLM 0.27 / 1.44 / 1.78 ms。
  - `kernels-serve` 時 nano 的 W1s 是 78 / 855 / 1,711 tok/s，ITL p50 12.50 / 15.17 / 25.31 ms；c=64 的 ITL p50 從落後 vLLM 1.1% 變成領先 9%。
  - vLLM 在 c=16 sampled 又只有 612 tok/s（greedy 937），TTFT p50 498 ms，和 `kernels-serve` 一樣，原因沒查。
- **G4**：過。W1s 在 c=1/16/64，nano 的 ITL p50 都比 vLLM 低，throughput 都比 vLLM 高。
- **VM**：跑完把 Colab VM（`sampler`）關掉了。

## 2026-10-29 — AWQ 權重、int8 lm_head、大 M kernel：本機實作（45cea19 到 78d6708）

nano 原本的 W4A16 是載入時做的 RTN（對稱、group 128、沒有 zero point），wikitext-2 的 PPL 是 8.023，比 bf16 的 7.232 高 10.9%。vLLM 那邊跑的是 `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`，所以兩邊的權重本來就不同。這一輪讓 nano 直接載入同一份 AutoAWQ checkpoint，再試兩項加速：int8 lm_head 和大 M 的 prefill kernel。分支是從 `claude/w4a16-serve`（77e6d5f）開的本機分支 `claude/nano-awq`。

### 做了什麼

- **AWQ 載入**（45cea19）：
  - `config.json` 的 `quantization_config` 必須是 `{quant_method: awq, version: gemm, bits: 4, group_size: 128, zero_point: true}`，其他組合報錯；
  - `weight_quant` 新增 `"awq"`：AWQ checkpoint 配 `"none"` 時自動改用 awq，AWQ 配 `"w4a16"` 或 bf16 checkpoint 配 `"awq"` 都報錯；
  - `awq_unpack`：qweight `[K, N/8]` 沿 N 打包，nibble 順序 `(0,2,4,6,1,3,5,7)`，解出 codes、zeros、scales，再用既有的 `_pack_tiled` 排成 tiled layout；
  - q/k/v 和 gate/up 在 packed 的 dim 1 上直接串接；scales 從 fp16 轉成模型的 dtype。
- **kernel 的 zero point**（e64c862）：`dequant_pair` 多一個 bias（magic + z），tiled kernel 和 `dequantize_w4a16_tiled` 跟著 scales 一起讀 zeros；RTN 走同一個 kernel，z 一律是 8，結果和原本逐 bit 相同。
- **kbench 與品質工具**（7303761）：
  - `w4a16_linear` 改用非對稱的隨機 codes、zeros、scales，新 impl `vllm_awq_marlin`（照 vLLM 0.19.1 的 repack 和 apply）；
  - 新 kernel `awq_dequant`：真實 checkpoint 的第 0 層，nano 解包後 dequant，和 vLLM 的 `awq_dequantize_triton` 比 atol=rtol=0；
  - `quality.py` 加 `--awq-model`，chunks 和每個 config 的 top-1 寫成檔案；新的 `quality_vllm.py` 在 vLLM venv 用同樣的 chunks 算 vLLM 的 NLL 和 top-1。
- **int8 lm_head**（4c3a72f）：`W8A16Weight`，每列對稱 int8、fp32 scale；Triton kernel 在 K 迴圈把 int8 轉 bf16 做 `tl.dot`，最後乘 scale。設定是 `lm_head_quant: auto | none | int8`（env `LM_HEAD_QUANT`），`auto` 在 G3 過之前等於 none。
- **`gemm_w4a16_large`**（836af08）：
  - 和 tiled kernel 相同的持久 striped grid（fp32 slots 加 counters），每個 chunk 64·WM 列；
  - 8 個 warp 依列 × 欄排列，每個 warp 負責 64×64（MT 4）；BK 64，4 個 stage；block tile 128×256 或 256×128；
  - 計畫寫的是「依輸出 tile 平行、不需要 slots」，實作改成沿用 tiled 的排程，因為 qkv、o_proj 在 512 列時的 tile 不滿一個 wave。
- **Harness**（87480b9、78d6708）：phase1 的 nano W4 config 改用 `--weight-quant awq` 載入 AWQ checkpoint；phase2 的 nano 在 `WEIGHT_QUANT=awq` 時載入 `P2_VLLM_W4_MODEL`；`quality.py` 的 config 多一個 prefill 欄位。
- **編譯**：本機 docker 的 nvcc，0 spill（只有 trace build 有 4 B，之前就有）。

## 2026-10-30 03:42 — run `awq-a1-20261030-034248-colab`：AWQ 第一輪（新 VM `awq`）

`P1_STEPS="installs cuda129 tests kbench e2e quality summarize"`；e2e 的 nano 跑 w4a16（RTN）、awq、awq + int8 lm_head，各有 greedy 和 sampled，b=1/16/64/128/256；nsys 錄 awq 的 b=1/16/64；quality 是 bf16、RTN、AWQ、AWQ 的 dequant prefill、AWQ + int8 lm_head，再加 vLLM bf16 和 vLLM AWQ。程式在 78d6708。

- **tests**：沒跑到。`tests/gpu/test_nano_vllm_w8a16.py` 和 `tests/unit/test_nano_vllm_w8a16.py` 同名，pytest 收集時出錯，整個 step 中斷。GPU 測試檔改名成 `test_nano_vllm_w8a16_kernels.py`（7b71fd8）。
- **品質**（wikitext-2 test，288,627 個 token，2048 一段）：

  | config | PPL | top-1 agreement |
  |---|---:|---:|
  | nano bf16 | 7.232 | baseline |
  | nano W4A16 RTN | 8.023（+10.9%） | 85.91% |
  | nano AWQ | 7.562（+4.6%） | 90.10% |
  | nano AWQ，dequant prefill | 7.563 | 90.11% |
  | nano AWQ + int8 lm_head | 7.563 | 90.06% |
  | vLLM bf16 | 7.232 | 98.81% |
  | vLLM AWQ（awq_marlin） | 7.560 | 90.13% |

  - nano AWQ 和 vLLM AWQ 差 +0.04%（7.5625 對 7.5595），在 ±0.5% 內；
  - int8 lm_head 對 AWQ：PPL +0.001%（7.5626），top-1 99.01%，兩項都過 G3 的品質門檻；
  - large kernel 和 dequant prefill 差 0.0002，過 ≤ 0.01。
- **`awq_dequant`**：四種投影 nano 和 vLLM 的 `awq_dequantize_triton` 逐 bit 相同，速度也相同（qkv 53.1 對 53.7 µs）。
- **kbench decode**（µs，nano tiled / Marlin）：M=1 qkv 20.0 / 19.1、o 14.8 / 14.6、gate_up 52.4 / 54.9；M=64 qkv 33.5 / 35.6、o 28.1 / 28.3、gate_up 92.7 / 91.4。
- **kbench 大 M**（µs）：

  | 投影 | M | large 128x256 | large 256x128 | dequant + cuBLAS | cuBLAS bf16 | Marlin |
  |---|---:|---:|---:|---:|---:|---:|
  | qkv | 2048 | 602 | - | 468 | 429 | 538 |
  | gate_up | 2048 | 2,767 | 2,636 | 2,080 | 1,873 | 2,553 |
  | down | 2048 | 1,378 | - | 1,057 | - | - |
  | o_proj | 2048 | 435 | 423 | 336 | 313 | 353 |

  - large kernel 在 1,024–2,048 列大約 182 TFLOPs，cuBLAS 是 257，比 dequant + cuBLAS 慢 20–30%，G4 沒過；64–256 列也比 tiled 慢；
  - Marlin 在 2,048 列同樣比 dequant + cuBLAS 慢；
  - tiled kernel 一次跑 512 列（`cuda_tiled_512`）在 384 和 512 列反而最快：四個投影合計 911 和 1,225 µs，dequant + cuBLAS 是 1,167 和 1,386。
- **kbench lm_head**（µs）：int8 在 ≤ 16 列 369–386，bf16 713–736（1.9 倍）；64 列 641 對 779；256 列 2,526 對 cuBLAS 1,164。256 列慢的原因是 grid 先跑完一個 row block 的所有 column tile 才換下一個，每個 row block 都從 HBM 重讀一次 int8 權重。
- **e2e**（ms/step，greedy，vLLM AWQ 是 awq_marlin）：

  | config | b=1 | b=16 | b=64 | b=128 | b=256 |
  |---|---:|---:|---:|---:|---:|
  | nano RTN | 5.20 | 7.15 | 15.59 | 27.32 | 38.42 |
  | nano AWQ | 5.19 | 7.15 | 15.57 | 27.29 | 38.34 |
  | nano AWQ + int8 lm_head | 4.82 | 6.78 | 15.41 | 27.79 | 39.87 |
  | vLLM AWQ | 5.56 | 7.86 | 16.22 | 27.27 | 38.76 |

  - AWQ 和 RTN 在同一次 run 裡幾乎一樣，因為 RTN 現在也走帶 zero point 的 kernel（z=8）；兩者都比 `sampler-s2` 的 RTN（舊 kernel，5.09 / 7.06 / 15.00）慢 2–4%。b=64 的 GEMM 多了 8.5%，來自 MT 3–4 每個 k16 row 都解一次 zero point（7b71fd8 改成每個 tile 解一次）；
  - int8 lm_head 在 b=1 快 0.37 ms、b=16 快 0.37 ms，b=128 和 b=256 反而慢 0.5 和 1.5 ms（上面的 kernel 問題）；
  - sampled：nano AWQ 5.21 / 7.20 / 15.69 / 27.47 / 38.70，vLLM 5.84 / 9.33 / 18.11 / 31.07 / 44.68。
- **nsys GEMM**（每層的四個投影，µs，nano / vLLM AWQ）：b=1 114.0 / 117.7，b=16 120.8 / 126.6，b=64 214.8 / 221.3。
- **門檻**：G1 的品質部分過，測試沒跑到；G2 和 vLLM 比過，和 s2 RTN 比沒過（要在同一台 VM 重比）；G3 品質過，速度在 b ≥ 128 沒過；G4 沒過。
- **VM**：跑完把 `awq` 關掉了。
