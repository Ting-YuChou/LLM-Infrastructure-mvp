# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16 | layer=qkv M=1 N=6144 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0} | 17.9 | 17.9 | 1.00x |
| w4a16 | layer=qkv M=2 N=6144 K=4096 | {'kernel': 'wide', 'splits': 8} | 19.8 | 20.7 | 1.05x |
| w4a16 | layer=qkv M=4 N=6144 K=4096 | {'kernel': 'wide', 'splits': 8} | 20.8 | 21.1 | 1.02x |
| w4a16 | layer=qkv M=8 N=6144 K=4096 | {'kernel': 'mma', 'splits': 0} | 21.3 | 21.3 | 1.00x |
| w4a16 | layer=qkv M=16 N=6144 K=4096 | {'kernel': 'mma', 'splits': 0} | 22.1 | 22.1 | 1.00x |
| w4a16 | layer=qkv M=32 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 31.2 | 40.4 | 1.30x |
| w4a16 | layer=qkv M=64 N=6144 K=4096 | {'kernel': 'wide', 'splits': 4} | 63.6 | 64.9 | 1.02x |
| w4a16 | layer=o_proj M=1 N=4096 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 1} | 13.1 | 13.2 | 1.01x |
| w4a16 | layer=o_proj M=2 N=4096 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0} | 15.4 | 15.4 | 1.00x |
| w4a16 | layer=o_proj M=4 N=4096 K=4096 | {'kernel': 'mma', 'splits': 0} | 16.8 | 16.8 | 1.00x |
| w4a16 | layer=o_proj M=8 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 17.2 | 17.2 | 1.00x |
| w4a16 | layer=o_proj M=16 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 18.2 | 18.2 | 1.00x |
| w4a16 | layer=o_proj M=32 N=4096 K=4096 | {'kernel': 'mma', 'splits': 1} | 30.0 | 35.6 | 1.19x |
| w4a16 | layer=o_proj M=64 N=4096 K=4096 | {'kernel': 'wide', 'splits': 0} | 57.3 | 57.3 | 1.00x |
| w4a16 | layer=gate_up M=1 N=28672 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 59.6 | 59.6 | 1.00x |
| w4a16 | layer=gate_up M=2 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 69.2 | 70.7 | 1.02x |
| w4a16 | layer=gate_up M=4 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 69.7 | 69.8 | 1.00x |
| w4a16 | layer=gate_up M=8 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 70.7 | 70.8 | 1.00x |
| w4a16 | layer=gate_up M=16 N=28672 K=4096 | {'kernel': 'mma', 'splits': 0} | 74.2 | 74.2 | 1.00x |
| w4a16 | layer=gate_up M=32 N=28672 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0} | 118.3 | 118.3 | 1.00x |
| w4a16 | layer=gate_up M=64 N=28672 K=4096 | {'kernel': 'wide', 'splits': 0} | 204.3 | 204.4 | 1.00x |
| w4a16 | layer=down M=1 N=4096 K=14336 | {'kernel': 'gemv', 'rows_per_warp': 1} | 37.8 | 38.4 | 1.02x |
| w4a16 | layer=down M=2 N=4096 K=14336 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0} | 43.4 | 43.4 | 1.00x |
| w4a16 | layer=down M=4 N=4096 K=14336 | {'kernel': 'auto', 'auto_gemv_rows': 2, 'auto_mma_rows': 16, 'rows_per_warp': 2, 'splits': 0} | 44.4 | 44.4 | 1.00x |
| w4a16 | layer=down M=8 N=4096 K=14336 | {'kernel': 'mma', 'splits': 0} | 45.5 | 45.5 | 1.00x |
| w4a16 | layer=down M=16 N=4096 K=14336 | {'kernel': 'mma', 'splits': 2} | 47.9 | 47.9 | 1.00x |
| w4a16 | layer=down M=32 N=4096 K=14336 | {'kernel': 'wide', 'splits': 8} | 69.0 | 74.8 | 1.08x |
| w4a16 | layer=down M=64 N=4096 K=14336 | {'kernel': 'wide', 'splits': 4} | 127.3 | 132.7 | 1.04x |
