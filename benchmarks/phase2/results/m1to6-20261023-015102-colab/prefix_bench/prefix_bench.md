# Prefix cache host cost (9000 blocks of 16, 192 requests)

| workload | impl | hit rate | admit p50 µs | admit p99 µs | prefill commit p50 µs | decode commit mean µs |
|---|---|---:|---:|---:|---:|---:|
| shared | hash (Phase 1) | 93.6% | 18467 | 19136 | 52 | 0.8 |
| shared | radix (Python) | 93.6% | 167 | 206 | 4 | 0.5 |
| shared | radix (C++) | 93.6% | 16 | 27 | 2 | 0.5 |
| many | hash (Phase 1) | 47.2% | 3021 | 36749 | 935 | 0.9 |
| many | radix (Python) | 47.2% | 313 | 387 | 21 | 1.1 |
| many | radix (C++) | 47.2% | 32 | 46 | 5 | 0.6 |
