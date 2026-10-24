# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16_tiled | layer=qkv M=1 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.3 | 16.3 | 1.00x |
| w4a16_tiled | layer=qkv M=4 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.6 | 16.6 | 1.00x |
| w4a16_tiled | layer=qkv M=16 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.2 | 16.2 | 1.00x |
| w4a16_tiled | layer=qkv M=32 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 20.0 | 20.0 | 1.00x |
| w4a16_tiled | layer=qkv M=64 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 34.7 | 38.2 | 1.10x |
| w4a16_tiled | layer=qkv M=128 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 68.6 | 75.9 | 1.11x |
| w4a16_tiled | layer=qkv M=256 N=6144 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 107.1 | 107.1 | 1.00x |
| w4a16_tiled | layer=o_proj M=1 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.8 | 14.8 | 1.00x |
| w4a16_tiled | layer=o_proj M=4 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.7 | 14.7 | 1.00x |
| w4a16_tiled | layer=o_proj M=16 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.9 | 15.0 | 1.01x |
| w4a16_tiled | layer=o_proj M=32 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 18.6 | 18.6 | 1.00x |
| w4a16_tiled | layer=o_proj M=64 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 27.9 | 31.6 | 1.13x |
| w4a16_tiled | layer=o_proj M=128 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 55.1 | 63.6 | 1.15x |
| w4a16_tiled | layer=o_proj M=256 N=4096 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 128} | 77.4 | 77.4 | 1.00x |
| w4a16_tiled | layer=gate_up M=1 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 51.5 | 51.5 | 1.00x |
| w4a16_tiled | layer=gate_up M=4 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 52.2 | 52.2 | 1.00x |
| w4a16_tiled | layer=gate_up M=16 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 54.5 | 54.5 | 1.00x |
| w4a16_tiled | layer=gate_up M=32 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 66.3 | 66.3 | 1.00x |
| w4a16_tiled | layer=gate_up M=64 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 102.5 | 102.5 | 1.00x |
| w4a16_tiled | layer=gate_up M=128 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 206.0 | 206.0 | 1.00x |
| w4a16_tiled | layer=gate_up M=256 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 416.3 | 553.5 | 1.33x |
| w4a16_tiled | layer=down M=1 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 30.0 | 30.0 | 1.00x |
| w4a16_tiled | layer=down M=4 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 30.0 | 30.0 | 1.00x |
| w4a16_tiled | layer=down M=16 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 31.3 | 31.3 | 1.00x |
| w4a16_tiled | layer=down M=32 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 38.2 | 38.2 | 1.00x |
| w4a16_tiled | layer=down M=64 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 62.4 | 72.0 | 1.15x |
| w4a16_tiled | layer=down M=128 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 124.7 | 143.9 | 1.15x |
| w4a16_tiled | layer=down M=256 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 249.4 | 258.5 | 1.04x |
