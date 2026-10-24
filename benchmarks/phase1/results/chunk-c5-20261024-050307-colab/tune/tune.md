# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16_tiled | layer=qkv M=1 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.0 | 16.0 | 1.00x |
| w4a16_tiled | layer=qkv M=4 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.1 | 16.1 | 1.00x |
| w4a16_tiled | layer=qkv M=16 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.0 | 16.0 | 1.00x |
| w4a16_tiled | layer=qkv M=32 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 19.8 | 19.8 | 1.00x |
| w4a16_tiled | layer=qkv M=48 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 27.3 | 27.3 | 1.00x |
| w4a16_tiled | layer=qkv M=64 N=6144 K=4096 | {'tiled_tile': '64x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 34.2 | 34.8 | 1.02x |
| w4a16_tiled | layer=qkv M=96 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 37.8 | 37.8 | 1.00x |
| w4a16_tiled | layer=qkv M=128 N=6144 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 48.5 | 49.7 | 1.02x |
| w4a16_tiled | layer=qkv M=192 N=6144 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 61.6 | 61.6 | 1.00x |
| w4a16_tiled | layer=qkv M=256 N=6144 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 74.5 | 74.5 | 1.00x |
| w4a16_tiled | layer=o_proj M=1 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 13.9 | 14.8 | 1.07x |
| w4a16_tiled | layer=o_proj M=4 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 14.3 | 14.9 | 1.04x |
| w4a16_tiled | layer=o_proj M=16 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 14.8 | 15.1 | 1.02x |
| w4a16_tiled | layer=o_proj M=32 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 18.7 | 18.7 | 1.00x |
| w4a16_tiled | layer=o_proj M=48 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 24.2 | 24.2 | 1.00x |
| w4a16_tiled | layer=o_proj M=64 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.0 | 30.0 | 1.00x |
| w4a16_tiled | layer=o_proj M=96 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.1 | 30.1 | 1.00x |
| w4a16_tiled | layer=o_proj M=128 N=4096 K=4096 | {'tiled_tile': '64x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 38.8 | 39.0 | 1.00x |
| w4a16_tiled | layer=o_proj M=192 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 48.9 | 48.9 | 1.00x |
| w4a16_tiled | layer=o_proj M=256 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 55.8 | 55.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=1 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 52.0 | 52.0 | 1.00x |
| w4a16_tiled | layer=gate_up M=4 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 52.7 | 52.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=16 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 54.7 | 54.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=32 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 60.7 | 60.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=48 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 73.7 | 73.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=64 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 88.3 | 88.3 | 1.00x |
| w4a16_tiled | layer=gate_up M=96 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 129.7 | 129.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=128 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 159.8 | 159.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=192 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 228.9 | 228.9 | 1.00x |
| w4a16_tiled | layer=gate_up M=256 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 303.2 | 303.2 | 1.00x |
| w4a16_tiled | layer=down M=1 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.0 | 30.4 | 1.01x |
| w4a16_tiled | layer=down M=4 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.4 | 30.7 | 1.01x |
| w4a16_tiled | layer=down M=16 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 31.7 | 31.9 | 1.01x |
| w4a16_tiled | layer=down M=32 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 36.5 | 36.5 | 1.00x |
| w4a16_tiled | layer=down M=48 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 45.7 | 45.7 | 1.00x |
| w4a16_tiled | layer=down M=64 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 59.5 | 59.5 | 1.00x |
| w4a16_tiled | layer=down M=96 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 71.5 | 71.5 | 1.00x |
| w4a16_tiled | layer=down M=128 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 93.4 | 94.1 | 1.01x |
| w4a16_tiled | layer=down M=192 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 123.2 | 123.2 | 1.00x |
| w4a16_tiled | layer=down M=256 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 155.8 | 155.8 | 1.00x |
