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
