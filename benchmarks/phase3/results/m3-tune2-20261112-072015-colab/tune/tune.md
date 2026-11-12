# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| block | B=1 Q=10 ctx=256 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 2, 'reduce': 'heads'} | 9.6 | 19.2 | 2.00x |
| block | B=1 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1, 'reduce': 'heads'} | 13.5 | 20.4 | 1.51x |
| block | B=1 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 1, 'reduce': 'heads'} | 26.6 | 31.4 | 1.18x |
| block | B=4 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 2, 'reduce': 'heads'} | 26.8 | 34.9 | 1.30x |
| block | B=4 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 2, 'reduce': 'heads'} | 76.4 | 88.1 | 1.15x |
| block | B=16 Q=10 ctx=256 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 2, 'reduce': 'heads'} | 25.0 | 35.7 | 1.43x |
| block | B=16 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 16, 'splits_shift': 2, 'reduce': 'heads'} | 75.6 | 89.2 | 1.18x |
| block | B=16 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 0, 'reduce': 'loop'} | 268.8 | 268.8 | 1.00x |
| block | B=64 Q=10 ctx=256 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 2, 'reduce': 'loop'} | 76.2 | 76.4 | 1.00x |
| block | B=64 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 1, 'reduce': 'loop'} | 247.7 | 249.0 | 1.01x |
| block | B=64 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'block_r': 64, 'splits_shift': 2, 'reduce': 'loop'} | 928.9 | 930.4 | 1.00x |
