# Prefix cache host cost (9000 blocks of 16, 192 requests)

| workload | impl | hit rate | admit p50 µs | admit p99 µs | prefill commit p50 µs | decode commit mean µs |
|---|---|---:|---:|---:|---:|---:|
| shared | hash (Phase 1) | 93.6% | 18542 | 20159 | 54 | 0.8 |
| shared | radix (Python) | 93.6% | 168 | 180 | 4 | 0.6 |
| shared | radix (C++) | 93.6% | 16 | 24 | 2 | 0.5 |
| many | hash (Phase 1) | 47.2% | 3065 | 36837 | 977 | 0.9 |
| many | radix (Python) | 47.2% | 314 | 387 | 21 | 1.1 |
| many | radix (C++) | 47.2% | 33 | 50 | 5 | 0.6 |
