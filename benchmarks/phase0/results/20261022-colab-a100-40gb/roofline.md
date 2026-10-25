### A100-SXM4-40GB (datasheet): 1555 GB/s, 312 TFLOPs

- params 8.030 B, all weights 16.06 GB; one decode step reads 15.01 GB (embedding table excluded: one row per token)
- KV cache 128 KiB per token
- batch=1 decode bound: **9.67 ms/token** (103 tok/s)

| batch | ctx 256: ms/step | tok/s | ctx 1152: ms/step | tok/s | bound | serve 1024/256: out tok/s | bound |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 9.67 | 103 | 9.75 | 103 | memory | 103 | memory |
| 2 | 9.70 | 206 | 9.85 | 203 | memory | 203 | memory |
| 4 | 9.74 | 411 | 10.04 | 398 | memory | 398 | memory |
| 8 | 9.83 | 814 | 10.43 | 767 | memory | 767 | memory |
| 16 | 10.00 | 1600 | 11.21 | 1428 | memory | 1427 | memory |
| 32 | 10.34 | 3094 | 12.76 | 2508 | memory | 2506 | memory |
| 64 | 11.03 | 5800 | 15.87 | 4033 | memory | 4029 | memory |
| 128 | 12.42 | 10310 | 22.08 | 5796 | memory | 4066 | compute |
| 256 | 15.18 | 16866 | 34.51 | 7418 | memory | 4066 | compute |

### A100-SXM4-40GB (measured on this pod): 1380 GB/s, 269 TFLOPs

- params 8.030 B, all weights 16.06 GB; one decode step reads 15.01 GB (embedding table excluded: one row per token)
- KV cache 128 KiB per token
- batch=1 decode bound: **10.90 ms/token** (92 tok/s)

| batch | ctx 256: ms/step | tok/s | ctx 1152: ms/step | tok/s | bound | serve 1024/256: out tok/s | bound |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 10.90 | 92 | 10.98 | 91 | memory | 91 | memory |
| 2 | 10.92 | 183 | 11.09 | 180 | memory | 180 | memory |
| 4 | 10.97 | 365 | 11.31 | 354 | memory | 354 | memory |
| 8 | 11.07 | 723 | 11.75 | 681 | memory | 681 | memory |
| 16 | 11.26 | 1420 | 12.63 | 1267 | memory | 1267 | memory |
| 32 | 11.65 | 2746 | 14.38 | 2226 | memory | 2224 | memory |
| 64 | 12.43 | 5148 | 17.88 | 3580 | memory | 3504 | compute |
| 128 | 13.99 | 9151 | 24.88 | 5145 | memory | 3504 | compute |
| 256 | 17.10 | 14971 | 38.88 | 6584 | memory | 3504 | compute |

