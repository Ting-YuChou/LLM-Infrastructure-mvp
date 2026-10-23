# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16 | layer=qkv M=1 N=6144 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2, 'gemv_prefetch': 0} | 18.1 | 18.1 | 1.00x |
| w4a16 | layer=qkv M=2 N=6144 K=4096 | {'kernel': 'wide', 'splits': 8} | 19.9 | 20.7 | 1.04x |
| w4a16 | layer=qkv M=4 N=6144 K=4096 | {'kernel': 'wide', 'splits': 8} | 20.9 | 21.1 | 1.01x |
| w4a16 | layer=qkv M=8 N=6144 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 21.2 | 21.2 | 1.00x |
| w4a16 | layer=qkv M=16 N=6144 K=4096 | {'kernel': 'mma', 'splits': 0} | 22.0 | 22.0 | 1.00x |
| w4a16 | layer=qkv M=32 N=6144 K=4096 | {'kernel': 'mma', 'splits': 0} | 31.1 | 40.2 | 1.29x |
| w4a16 | layer=qkv M=64 N=6144 K=4096 | {'kernel': 'wide', 'splits': 4} | 63.7 | 65.2 | 1.02x |
| w4a16 | layer=o_proj M=1 N=4096 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 1, 'gemv_prefetch': 0} | 13.1 | 13.5 | 1.03x |
| w4a16 | layer=o_proj M=2 N=4096 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2, 'gemv_prefetch': 0} | 15.4 | 15.5 | 1.00x |
| w4a16 | layer=o_proj M=4 N=4096 K=4096 | {'kernel': 'mma', 'splits': 0} | 16.8 | 16.8 | 1.00x |
| w4a16 | layer=o_proj M=8 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 17.2 | 17.2 | 1.00x |
| w4a16 | layer=o_proj M=16 N=4096 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 18.2 | 18.2 | 1.00x |
| w4a16 | layer=o_proj M=32 N=4096 K=4096 | {'kernel': 'mma', 'splits': 1} | 29.8 | 35.9 | 1.21x |
| w4a16 | layer=o_proj M=64 N=4096 K=4096 | {'kernel': 'wide', 'splits': 0} | 57.1 | 57.1 | 1.00x |
| w4a16 | layer=gate_up M=1 N=28672 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 60.0 | 60.0 | 1.00x |
| w4a16 | layer=gate_up M=2 N=28672 K=4096 | {'kernel': 'mma', 'splits': 0} | 68.9 | 71.0 | 1.03x |
| w4a16 | layer=gate_up M=4 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 69.5 | 69.6 | 1.00x |
| w4a16 | layer=gate_up M=8 N=28672 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 70.6 | 70.6 | 1.00x |
| w4a16 | layer=gate_up M=16 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 73.9 | 74.0 | 1.00x |
| w4a16 | layer=gate_up M=32 N=28672 K=4096 | {'kernel': 'wide', 'splits': 1} | 117.9 | 117.9 | 1.00x |
| w4a16 | layer=gate_up M=64 N=28672 K=4096 | {'kernel': 'wide', 'splits': 1} | 203.4 | 203.4 | 1.00x |
| w4a16 | layer=down M=1 N=4096 K=14336 | {'kernel': 'gemv', 'rows_per_warp': 1, 'gemv_prefetch': 0} | 37.7 | 39.0 | 1.03x |
| w4a16 | layer=down M=2 N=4096 K=14336 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 43.7 | 43.7 | 1.00x |
| w4a16 | layer=down M=4 N=4096 K=14336 | {'kernel': 'mma', 'splits': 0} | 44.1 | 44.1 | 1.00x |
| w4a16 | layer=down M=8 N=4096 K=14336 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0, 'gemv_prefetch': 0} | 45.2 | 45.2 | 1.00x |
| w4a16 | layer=down M=16 N=4096 K=14336 | {'kernel': 'mma', 'splits': 2} | 47.7 | 47.7 | 1.00x |
| w4a16 | layer=down M=32 N=4096 K=14336 | {'kernel': 'wide', 'splits': 8} | 68.5 | 74.7 | 1.09x |
| w4a16 | layer=down M=64 N=4096 K=14336 | {'kernel': 'wide', 'splits': 4} | 126.8 | 132.5 | 1.05x |
