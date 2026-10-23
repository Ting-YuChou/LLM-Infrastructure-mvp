# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16_tiled | layer=qkv M=1 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 18.5 | 18.5 | 1.00x |
| w4a16_tiled | layer=qkv M=4 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 18.7 | 18.7 | 1.00x |
| w4a16_tiled | layer=qkv M=16 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 19.1 | 19.1 | 1.00x |
| w4a16_tiled | layer=qkv M=32 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 24.6 | 24.6 | 1.00x |
| w4a16_tiled | layer=qkv M=64 N=6144 K=4096 | {'tiled_tile': '64x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 41.9 | 44.1 | 1.05x |
| w4a16_tiled | layer=qkv M=128 N=6144 K=4096 | {'tiled_tile': '64x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 83.6 | 102.7 | 1.23x |
| w4a16_tiled | layer=qkv M=256 N=6144 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 107.7 | 107.7 | 1.00x |
| w4a16_tiled | layer=o_proj M=1 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.2 | 16.2 | 1.00x |
| w4a16_tiled | layer=o_proj M=4 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.3 | 16.3 | 1.00x |
| w4a16_tiled | layer=o_proj M=16 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.4 | 16.4 | 1.00x |
| w4a16_tiled | layer=o_proj M=32 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 21.6 | 21.6 | 1.00x |
| w4a16_tiled | layer=o_proj M=64 N=4096 K=4096 | {'tiled_tile': '64x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 33.8 | 39.9 | 1.18x |
| w4a16_tiled | layer=o_proj M=128 N=4096 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 64.7 | 64.7 | 1.00x |
| w4a16_tiled | layer=o_proj M=256 N=4096 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 78.3 | 78.3 | 1.00x |
| w4a16_tiled | layer=gate_up M=1 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 55.9 | 55.9 | 1.00x |
| w4a16_tiled | layer=gate_up M=4 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 55.8 | 55.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=16 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 59.1 | 59.1 | 1.00x |
| w4a16_tiled | layer=gate_up M=32 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 77.1 | 77.1 | 1.00x |
| w4a16_tiled | layer=gate_up M=64 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 112.6 | 112.6 | 1.00x |
| w4a16_tiled | layer=gate_up M=128 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 225.3 | 513.9 | 2.28x |
| w4a16_tiled | layer=gate_up M=256 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 450.6 | 555.3 | 1.23x |
| w4a16_tiled | layer=down M=1 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 32.2 | 32.2 | 1.00x |
| w4a16_tiled | layer=down M=4 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 32.3 | 32.3 | 1.00x |
| w4a16_tiled | layer=down M=16 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 34.0 | 34.0 | 1.00x |
| w4a16_tiled | layer=down M=32 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 44.2 | 44.2 | 1.00x |
| w4a16_tiled | layer=down M=64 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 77.1 | 77.1 | 1.00x |
| w4a16_tiled | layer=down M=128 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 154.4 | 231.4 | 1.50x |
| w4a16_tiled | layer=down M=256 N=4096 K=14336 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 259.2 | 259.2 | 1.00x |
