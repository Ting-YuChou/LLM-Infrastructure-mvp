# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| decode_fp8 | B=1 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2} | 12.0 | 14.2 | 1.18x |
| decode_fp8 | B=1 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 1} | 19.3 | 22.9 | 1.18x |
| decode_fp8 | B=16 ctx=1024 | {'block_n': 128, 'num_warps': 4, 'num_stages': 2} | 51.3 | 58.5 | 1.14x |
| decode_fp8 | B=16 ctx=4096 | {'block_n': 128, 'num_warps': 4, 'num_stages': 4} | 154.5 | 188.7 | 1.22x |
| decode_fp8 | B=64 ctx=1024 | {'block_n': 128, 'num_warps': 4, 'num_stages': 4} | 147.7 | 147.9 | 1.00x |
| decode_fp8 | B=64 ctx=4096 | {'block_n': 128, 'num_warps': 4, 'num_stages': 3} | 541.9 | 541.9 | 1.00x |
| decode_fp8 | B=128 ctx=1024 | {'block_n': 128, 'num_warps': 4, 'num_stages': 3} | 271.9 | 271.9 | 1.00x |
