# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16_tiled | layer=qkv M=1 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.3 | 16.3 | 1.00x |
| w4a16_tiled | layer=qkv M=4 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.6 | 16.6 | 1.00x |
| w4a16_tiled | layer=qkv M=16 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 16.2 | 16.2 | 1.00x |
| w4a16_tiled | layer=qkv M=32 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 20.0 | 20.0 | 1.00x |
| w4a16_tiled | layer=qkv M=48 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 29.0 | 29.0 | 1.00x |
| w4a16_tiled | layer=qkv M=64 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 34.7 | 34.7 | 1.00x |
| w4a16_tiled | layer=qkv M=128 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 68.6 | 68.6 | 1.00x |
| w4a16_tiled | layer=qkv M=256 N=6144 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 64} | 107.6 | 107.8 | 1.00x |
| w4a16_tiled | layer=o_proj M=1 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.8 | 14.8 | 1.00x |
| w4a16_tiled | layer=o_proj M=4 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.7 | 14.7 | 1.00x |
| w4a16_tiled | layer=o_proj M=16 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 14.9 | 15.1 | 1.01x |
| w4a16_tiled | layer=o_proj M=32 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 18.6 | 18.6 | 1.00x |
| w4a16_tiled | layer=o_proj M=48 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 24.0 | 24.0 | 1.00x |
| w4a16_tiled | layer=o_proj M=64 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 27.9 | 27.9 | 1.00x |
| w4a16_tiled | layer=o_proj M=128 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 55.1 | 55.1 | 1.00x |
| w4a16_tiled | layer=o_proj M=256 N=4096 K=4096 | {'tiled_tile': 'auto', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 128} | 77.4 | 77.4 | 1.00x |
| w4a16_tiled | layer=gate_up M=1 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 51.5 | 51.5 | 1.00x |
| w4a16_tiled | layer=gate_up M=4 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 52.3 | 52.3 | 1.00x |
| w4a16_tiled | layer=gate_up M=16 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 54.6 | 54.6 | 1.00x |
| w4a16_tiled | layer=gate_up M=32 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 66.5 | 66.5 | 1.00x |
| w4a16_tiled | layer=gate_up M=48 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 85.8 | 85.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=64 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 102.8 | 102.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=128 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 206.6 | 206.6 | 1.00x |
| w4a16_tiled | layer=gate_up M=256 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 414.8 | 554.6 | 1.34x |
| w4a16_tiled | layer=down M=1 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 30.3 | 30.3 | 1.00x |
| w4a16_tiled | layer=down M=4 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 30.2 | 30.2 | 1.00x |
| w4a16_tiled | layer=down M=16 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 31.5 | 31.5 | 1.00x |
| w4a16_tiled | layer=down M=32 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 38.3 | 38.3 | 1.00x |
| w4a16_tiled | layer=down M=48 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 52.9 | 52.9 | 1.00x |
| w4a16_tiled | layer=down M=64 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 62.5 | 62.5 | 1.00x |
| w4a16_tiled | layer=down M=128 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 125.1 | 125.1 | 1.00x |
| w4a16_tiled | layer=down M=256 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 256} | 249.7 | 258.5 | 1.04x |
