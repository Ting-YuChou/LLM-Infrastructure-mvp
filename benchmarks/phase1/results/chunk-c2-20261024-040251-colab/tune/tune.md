# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| w4a16_tiled | layer=qkv M=1 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.8 | 16.8 | 1.00x |
| w4a16_tiled | layer=qkv M=4 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.8 | 16.8 | 1.00x |
| w4a16_tiled | layer=qkv M=16 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 16.3 | 16.3 | 1.00x |
| w4a16_tiled | layer=qkv M=32 N=6144 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 20.4 | 20.4 | 1.00x |
| w4a16_tiled | layer=qkv M=48 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 29.2 | 29.2 | 1.00x |
| w4a16_tiled | layer=qkv M=64 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 35.6 | 35.6 | 1.00x |
| w4a16_tiled | layer=qkv M=96 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 44.3 | 44.3 | 1.00x |
| w4a16_tiled | layer=qkv M=128 N=6144 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 55.2 | 55.2 | 1.00x |
| w4a16_tiled | layer=qkv M=192 N=6144 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 74.9 | 76.5 | 1.02x |
| w4a16_tiled | layer=qkv M=256 N=6144 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 93.1 | 96.5 | 1.04x |
| w4a16_tiled | layer=o_proj M=1 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 14.8 | 15.0 | 1.02x |
| w4a16_tiled | layer=o_proj M=4 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 14.7 | 15.0 | 1.02x |
| w4a16_tiled | layer=o_proj M=16 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 14.9 | 15.2 | 1.02x |
| w4a16_tiled | layer=o_proj M=32 N=4096 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 18.9 | 18.9 | 1.00x |
| w4a16_tiled | layer=o_proj M=48 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 24.1 | 24.1 | 1.00x |
| w4a16_tiled | layer=o_proj M=64 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 28.5 | 28.5 | 1.00x |
| w4a16_tiled | layer=o_proj M=96 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 33.4 | 33.4 | 1.00x |
| w4a16_tiled | layer=o_proj M=128 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 41.4 | 41.4 | 1.00x |
| w4a16_tiled | layer=o_proj M=192 N=4096 K=4096 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 55.4 | 55.4 | 1.00x |
| w4a16_tiled | layer=o_proj M=256 N=4096 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 67.5 | 68.9 | 1.02x |
| w4a16_tiled | layer=gate_up M=1 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 51.7 | 51.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=4 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 52.4 | 52.4 | 1.00x |
| w4a16_tiled | layer=gate_up M=16 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 54.8 | 54.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=32 N=28672 K=4096 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 67.3 | 67.3 | 1.00x |
| w4a16_tiled | layer=gate_up M=48 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 84.4 | 84.4 | 1.00x |
| w4a16_tiled | layer=gate_up M=64 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 103.8 | 103.8 | 1.00x |
| w4a16_tiled | layer=gate_up M=96 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 156.0 | 156.0 | 1.00x |
| w4a16_tiled | layer=gate_up M=128 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 194.0 | 194.0 | 1.00x |
| w4a16_tiled | layer=gate_up M=192 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 279.7 | 279.7 | 1.00x |
| w4a16_tiled | layer=gate_up M=256 N=28672 K=4096 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 370.4 | 370.4 | 1.00x |
| w4a16_tiled | layer=down M=1 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.3 | 30.3 | 1.00x |
| w4a16_tiled | layer=down M=4 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 30.5 | 30.5 | 1.00x |
| w4a16_tiled | layer=down M=16 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 31.6 | 31.6 | 1.00x |
| w4a16_tiled | layer=down M=32 N=4096 K=14336 | {'tiled_tile': '256x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 39.2 | 39.2 | 1.00x |
| w4a16_tiled | layer=down M=48 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 53.1 | 53.1 | 1.00x |
| w4a16_tiled | layer=down M=64 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 64.0 | 64.0 | 1.00x |
| w4a16_tiled | layer=down M=96 N=4096 K=14336 | {'tiled_tile': '128x128', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 88.0 | 88.0 | 1.00x |
| w4a16_tiled | layer=down M=128 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 108.7 | 110.4 | 1.02x |
| w4a16_tiled | layer=down M=192 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 150.7 | 156.6 | 1.04x |
| w4a16_tiled | layer=down M=256 N=4096 K=14336 | {'tiled_tile': '64x256', 'tiled_blocks_per_sm': 1, 'tiled_max_rows': 512} | 192.7 | 202.6 | 1.05x |
