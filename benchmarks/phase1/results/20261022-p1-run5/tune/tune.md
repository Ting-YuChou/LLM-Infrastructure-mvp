# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16 | layer=qkv M=1 N=6144 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 17.8 | 18.7 | 1.05x |
| w4a16 | layer=qkv M=2 N=6144 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 20.7 | 20.9 | 1.01x |
| w4a16 | layer=qkv M=4 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 20.9 | 26.6 | 1.27x |
| w4a16 | layer=qkv M=8 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 21.1 | 24.6 | 1.17x |
| w4a16 | layer=qkv M=16 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 21.9 | 26.6 | 1.21x |
| w4a16 | layer=qkv M=32 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 30.8 | 39.1 | 1.27x |
| w4a16 | layer=qkv M=64 N=6144 K=4096 | {'kernel': 'mma', 'splits': 1} | 75.0 | 83.2 | 1.11x |
| w4a16 | layer=o_proj M=1 N=4096 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 1} | 13.1 | 15.9 | 1.21x |
| w4a16 | layer=o_proj M=2 N=4096 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 15.4 | 17.6 | 1.15x |
| w4a16 | layer=o_proj M=4 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 17.9 | 22.1 | 1.24x |
| w4a16 | layer=o_proj M=8 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 18.2 | 18.6 | 1.02x |
| w4a16 | layer=o_proj M=16 N=4096 K=4096 | {'kernel': 'mma', 'splits': 2} | 18.6 | 19.5 | 1.05x |
| w4a16 | layer=o_proj M=32 N=4096 K=4096 | {'kernel': 'mma', 'splits': 1} | 29.6 | 30.2 | 1.02x |
| w4a16 | layer=o_proj M=64 N=4096 K=4096 | {'kernel': 'mma', 'splits': 4} | 61.9 | 63.4 | 1.02x |
| w4a16 | layer=gate_up M=1 N=28672 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 59.5 | 60.4 | 1.02x |
| w4a16 | layer=gate_up M=2 N=28672 K=4096 | {'kernel': 'gemv', 'rows_per_warp': 2} | 70.7 | 72.4 | 1.02x |
| w4a16 | layer=gate_up M=4 N=28672 K=4096 | {'kernel': 'mma', 'splits': 0} | 71.7 | 98.2 | 1.37x |
| w4a16 | layer=gate_up M=8 N=28672 K=4096 | {'kernel': 'mma', 'splits': 1} | 72.7 | 72.7 | 1.00x |
| w4a16 | layer=gate_up M=16 N=28672 K=4096 | {'kernel': 'auto', 'auto_gemv_rows': 4, 'rows_per_warp': 4, 'splits': 0} | 77.7 | 77.7 | 1.00x |
| w4a16 | layer=gate_up M=32 N=28672 K=4096 | {'kernel': 'mma', 'splits': 0} | 136.5 | 136.5 | 1.00x |
| w4a16 | layer=gate_up M=64 N=28672 K=4096 | {'kernel': 'mma', 'splits': 0} | 313.4 | 313.4 | 1.00x |
| w4a16 | layer=down M=1 N=4096 K=14336 | {'kernel': 'gemv', 'rows_per_warp': 1} | 37.8 | 47.9 | 1.27x |
| w4a16 | layer=down M=2 N=4096 K=14336 | {'kernel': 'gemv', 'rows_per_warp': 2} | 43.4 | 53.6 | 1.23x |
| w4a16 | layer=down M=4 N=4096 K=14336 | {'kernel': 'mma', 'splits': 8} | 45.4 | 67.2 | 1.48x |
| w4a16 | layer=down M=8 N=4096 K=14336 | {'kernel': 'mma', 'splits': 8} | 46.0 | 47.6 | 1.03x |
| w4a16 | layer=down M=16 N=4096 K=14336 | {'kernel': 'mma', 'splits': 8} | 48.3 | 50.6 | 1.05x |
| w4a16 | layer=down M=32 N=4096 K=14336 | {'kernel': 'mma', 'splits': 8} | 77.3 | 79.5 | 1.03x |
| w4a16 | layer=down M=64 N=4096 K=14336 | {'kernel': 'mma', 'splits': 4} | 167.3 | 189.6 | 1.13x |
