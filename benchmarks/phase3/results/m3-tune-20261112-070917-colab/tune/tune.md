# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| block | B=1 Q=10 ctx=256 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1} | 19.8 | 36.0 | 1.82x |
| block | B=1 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1} | 21.7 | 37.2 | 1.71x |
| block | B=1 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1} | 35.5 | 48.2 | 1.36x |
| block | B=4 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1} | 32.6 | 43.1 | 1.32x |
| block | B=4 Q=10 ctx=4096 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 1} | 86.7 | 99.4 | 1.15x |
| block | B=16 Q=10 ctx=256 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 1} | 31.7 | 35.7 | 1.13x |
| block | B=16 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 1} | 85.6 | 89.4 | 1.04x |
| block | B=16 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 0} | 270.3 | 270.3 | 1.00x |
| block | B=64 Q=10 ctx=256 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 0} | 76.2 | 76.2 | 1.00x |
| block | B=64 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 0} | 246.7 | 246.7 | 1.00x |
| block | B=64 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 0} | 930.0 | 930.0 | 1.00x |
