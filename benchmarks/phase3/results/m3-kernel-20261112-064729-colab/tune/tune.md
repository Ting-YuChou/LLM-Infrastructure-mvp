# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| block | B=1 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 8, 'num_stages': 1, 'splits_shift': 2} | 18.5 | 19.7 | 1.06x |
| block | B=1 Q=10 ctx=4096 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 32.0 | 44.4 | 1.39x |
| block | B=4 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 39.2 | 50.1 | 1.28x |
| block | B=16 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 1} | 89.4 | 110.9 | 1.24x |
| block | B=16 Q=10 ctx=4096 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 1} | 284.6 | 305.5 | 1.07x |
| block | B=64 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 247.9 | 298.6 | 1.20x |
| block | B=64 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 929.2 | 1134.4 | 1.22x |
