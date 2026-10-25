# Triton launch-setting sweep

| kernel | shape | best config | best us | default us | speedup |
|---|---|---|---:|---:|---:|
| decode | B=1 ctx=1024 | {'block_n': 32, 'num_warps': 8, 'num_stages': 1} | 12.4 | 14.7 | 1.19x |
| decode | B=1 ctx=4096 | {'block_n': 64, 'num_warps': 8, 'num_stages': 1} | 20.9 | 22.6 | 1.08x |
| decode | B=16 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 1} | 61.2 | 61.3 | 1.00x |
| decode | B=32 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2} | 108.6 | 111.1 | 1.02x |
| decode | B=64 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2} | 200.3 | 200.4 | 1.00x |
| decode | B=64 ctx=4096 | {'block_n': 128, 'num_warps': 4, 'num_stages': 3} | 767.5 | 767.5 | 1.00x |
| decode | B=128 ctx=1024 | {'block_n': 128, 'num_warps': 4, 'num_stages': 3} | 380.0 | 380.0 | 1.00x |
| prefill | seqs=1 qlen=1024 | {'block_m': 128, 'block_n': 128, 'num_warps': 8, 'num_stages': 2} | 104.9 | 105.3 | 1.00x |
| prefill | seqs=1 qlen=4096 | {'block_m': 128, 'block_n': 128, 'num_warps': 8, 'num_stages': 2} | 940.7 | 1019.7 | 1.08x |
| prefill | seqs=4 qlen=1024 | {'block_m': 128, 'block_n': 32, 'num_warps': 4, 'num_stages': 2} | 295.7 | 319.5 | 1.08x |
| gemv | layer=qkv N=6144 K=4096 | {'block_n': 32, 'block_k': 128, 'num_warps': 4, 'num_stages': 3} | 41.5 | 42.3 | 1.02x |
| gemv | layer=o_proj N=4096 K=4096 | {'block_n': 32, 'block_k': 128, 'num_warps': 4, 'num_stages': 3} | 29.4 | 31.0 | 1.06x |
| gemv | layer=gate_up N=28672 K=4096 | {'block_n': 32, 'block_k': 512, 'num_warps': 4, 'num_stages': 4} | 165.4 | 170.8 | 1.03x |
| gemv | layer=down N=4096 K=14336 | {'block_n': 64, 'block_k': 128, 'num_warps': 4, 'num_stages': 3} | 87.8 | 91.1 | 1.04x |
| gemv | layer=lm_head N=128256 K=4096 | {'block_n': 128, 'block_k': 256, 'num_warps': 4, 'num_stages': 3} | 707.1 | 709.4 | 1.00x |
| cuda_gemv | layer=qkv M=1 N=6144 K=4096 | {'rows_per_warp': 4} | 41.3 | 43.4 | 1.05x |
| cuda_gemv | layer=qkv M=4 N=6144 K=4096 | {'rows_per_warp': 4} | 42.5 | 43.3 | 1.02x |
| cuda_gemv | layer=qkv M=8 N=6144 K=4096 | {'rows_per_warp': 4} | 49.4 | 59.2 | 1.20x |
| cuda_gemv | layer=o_proj M=1 N=4096 K=4096 | {'rows_per_warp': 1} | 29.2 | 29.3 | 1.00x |
| cuda_gemv | layer=o_proj M=4 N=4096 K=4096 | {'rows_per_warp': 2} | 31.4 | 31.4 | 1.00x |
| cuda_gemv | layer=o_proj M=8 N=4096 K=4096 | {'rows_per_warp': 4} | 39.5 | 48.4 | 1.23x |
| cuda_gemv | layer=gate_up M=1 N=28672 K=4096 | {'rows_per_warp': 1} | 165.0 | 165.1 | 1.00x |
| cuda_gemv | layer=gate_up M=4 N=28672 K=4096 | {'rows_per_warp': 2} | 170.7 | 170.7 | 1.00x |
| cuda_gemv | layer=gate_up M=8 N=28672 K=4096 | {'rows_per_warp': 4} | 189.5 | 233.7 | 1.23x |
| cuda_gemv | layer=down M=1 N=4096 K=14336 | {'rows_per_warp': 2} | 88.8 | 88.8 | 1.00x |
| cuda_gemv | layer=down M=4 N=4096 K=14336 | {'rows_per_warp': 2} | 98.4 | 98.4 | 1.00x |
| cuda_gemv | layer=down M=8 N=4096 K=14336 | {'rows_per_warp': 4} | 124.0 | 162.3 | 1.31x |
| cuda_gemv | layer=lm_head M=1 N=128256 K=4096 | {'rows_per_warp': 2} | 711.2 | 711.2 | 1.00x |
| cuda_gemv | layer=lm_head M=4 N=128256 K=4096 | {'rows_per_warp': 4} | 731.6 | 735.4 | 1.01x |
| cuda_gemv | layer=lm_head M=8 N=128256 K=4096 | {'rows_per_warp': 4} | 789.5 | 1000.7 | 1.27x |
