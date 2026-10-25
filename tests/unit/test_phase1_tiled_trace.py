"""tiled_trace.py's analysis on a hand-built trace of gemm_w4a16_tiled's trace build."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tiled_trace  # noqa: E402
from serving.nano_vllm.kernels.w4a16 import TILED_TRACE_EVENTS, tiled_schedule  # noqa: E402

RATE = 2.0  # cycles per ns
N, K, TILE_K, TILE_N, GRID = 1024, 2048, 256, 128, 3  # 8 slices of 8 tiles, stripes of 22


def _row(begin, end, entry_ns, prologue, per_tile, ends, ticket, last=(), sole=()):
    """A block's trace row: `ends` gives (sum warps, handoff, tail) ns per slice end;
    a slice end follows that slice's tiles of the stripe."""
    k_tiles = K // TILE_K
    row = [0] * TILED_TRACE_EVENTS
    clk = 10_000
    row[tiled_trace.GT_ENTRY], row[tiled_trace.CLK_ENTRY] = 1_000_000 + entry_ns, clk
    clk += prologue * RATE
    row[tiled_trace.CLK_FIRST] = clk
    row[tiled_trace.TILES] = end - begin
    t = begin
    for f, (sum_warps, handoff, tail) in enumerate(ends):
        tiles = min(end, (t // k_tiles + 1) * k_tiles) - t
        t += tiles
        clk += tiles * per_tile * RATE
        stamps = [clk]
        for d in (sum_warps, handoff, tail):
            clk += d * RATE
            stamps.append(clk)
        row[tiled_trace.FINISH0 + 4 * f : tiled_trace.FINISH0 + 4 * f + 4] = stamps
    clk += 50 * RATE
    row[tiled_trace.CLK_EXIT] = clk
    row[tiled_trace.GT_EXIT] = row[tiled_trace.GT_ENTRY] + round((clk - 10_000) / RATE)
    row[tiled_trace.FLAGS] = sum(1 << f for f in last) | sum(1 << (8 + f) for f in sole)
    row[tiled_trace.TICKET] = ticket
    return row


def _trace():
    schedule = tiled_schedule(N, K, TILE_K, TILE_N, GRID)
    assert schedule["ranges"] == [(0, 22), (22, 44), (44, 64)]
    rows = [
        # block 0: slices 0, 1 alone, then the start of slice 2 (shared)
        _row(0, 22, 0, 600, 1000, [(100, 0, 0), (100, 0, 0), (100, 400, 0)], ticket=0, sole=(0, 1)),
        # block 1: the rest of slice 2 (last to arrive), slices 3, 4 alone, start of 5
        _row(22, 44, 200, 700, 1100, [(100, 400, 900), (100, 0, 0), (100, 0, 0), (100, 400, 0)],
             ticket=2, last=(0,), sole=(1, 2)),
        # block 2: the rest of slice 5 (last), slices 6, 7 alone
        _row(44, 64, 100, 500, 1000, [(100, 400, 800), (100, 0, 0), (100, 0, 0)], ticket=1, last=(0,), sole=(1, 2)),
    ]
    return rows, schedule


def test_block_segments_recover_the_stamped_times():
    rows, schedule = _trace()
    assert tiled_trace.clock_rate([rows]) == pytest.approx(RATE, rel=1e-3)
    seg = tiled_trace.block_segments(rows[1], schedule["ranges"][1], K // TILE_K, RATE)
    assert seg["tiles"] == 22 and seg["prologue"] == 700 and seg["per_tile"] == pytest.approx(1100)
    assert [f["total"] for f in seg["finishes"]] == [1400, 100, 100, 500]
    assert [(f["last"], f["sole"], f["final"]) for f in seg["finishes"]] == [
        (True, False, False), (False, True, False), (False, True, False), (False, False, True)]
    assert seg["exit"] == 50


def test_analyze_run_finds_the_last_block_and_its_path():
    rows, schedule = _trace()
    run = tiled_trace.analyze_run(rows, schedule, K // TILE_K, RATE)
    crit = run["critical"]
    assert crit["block"] == 1 and crit["tiles"] == 22 and crit["finishes"] == 4
    assert not crit["last_arriver"]  # its final slice end hands off; block 2 sums slice 5
    assert crit["start"] == 200 and crit["prologue"] == 700
    assert crit["streaming"] == pytest.approx(22 * 1100) and crit["finishes_ns"] == 2100
    assert run["launch_skew"] == 200
    assert run["kernel"] == rows[1][tiled_trace.GT_EXIT] - rows[0][tiled_trace.GT_ENTRY]
    assert run["last_tail"] == 850  # median of the two last arrivals' slot sums
    assert run["finish_final_last"] == 0.0  # no block's final slice end was a last arrival
    summary = tiled_trace.summarize([run, run])
    assert summary["critical"]["streaming"] == crit["streaming"]


def test_tile_dram_share():
    # 16 KB of weights, 512 B of scales, 512 B of x per 256x128 tile, 100 blocks at 1,000 GB/s
    assert tiled_trace.tile_dram_ns(1, 256, 128, 100, 1000.0) == pytest.approx((16384 + 512 + 512) / 10)


def test_first_tile_units_cover_each_stripe_start():
    schedule = tiled_schedule(N, K, TILE_K, TILE_N, GRID)
    units = tiled_trace.first_tile_units(N, K, TILE_K, TILE_N, schedule, 2)
    # 3 blocks x 2 tiles x (16 k16 rows x 2 n64 units); block 0 starts at slice 0, k tile 0
    assert len(units) == 3 * 2 * 16 * 2 and len(set(units)) == len(units)
    assert units[:2] == [0, 1] and units[2] == N // 64  # next k16 row, same columns
    # block 1 starts at tile 22: slice 2, k tile 6 -> k16 row 96, units 4 and 5
    assert units[64:66] == [96 * (N // 64) + 4, 96 * (N // 64) + 5]
