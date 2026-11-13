#!/usr/bin/env python3
"""summary.md for one Phase 3 run directory (run_phase3.sh's summarize step).

Reads the P3_SERVE lines (one per `vllm bench serve` run) and P3_TOKENS lines
from the run's console, the speculative-decoding counters saved around each
run (serve/*.metrics.txt), accept/accept.json and status.tsv, and writes:
  serving   output tok/s, TPOT and TTFT by concurrency per server and dataset,
            and each drafter server's speedup over the same server without one
  drafts    accepted tokens per draft as the servers count them
  accept    the acceptance study: tokens per cycle per arm, survival by depth
  gate      nano's projected tokens/s at c=1 against the best DFlash baseline
  parity    nano's drafter against the checkpoint's own code (parity.json)
  texts     completions of a drafter server against the same server without
            one (texts.py), next to that server's own batch-size divergence
  M4 gates  the measured cycle at batch 1 (spec_check.py) and nano with the
            drafter at c=1 against the DFlash baselines, which are this run's
            or, with a second argument, another run's (the M0 run)
  M6        the step time of each CUDA graph (spec_check.py), the verify
            lengths the policy ran per serving run, and nano with the drafter
            at every concurrency against itself without one and against the
            DFlash baselines

  python summarize.py <run dir> [<baseline run dir>]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASETS = ("alpaca", "humaneval", "gsm8k", "long")
# M0's estimate of a nano cycle at batch 1, in ms (to be replaced by
# measurements from M3 on): verify = the engine's own decode step at c=1 times
# VERIFY_FACTOR (10 rows instead of 1: the GEMMs and the KV read are
# bandwidth-bound), plus one drafter pass and the rest of the cycle.
VERIFY_FACTOR = 1.08
DRAFT_MS = {"bf16": 2.1, "rtn4": 1.0}
OTHER_MS = 0.3
GATE = 1.2  # projected nano / best DFlash baseline at c=1


def console_rows(out: Path, tag: str) -> list[dict]:
    """The '<tag> {json}' lines of the run's console and step logs."""
    rows, seen = [], set()
    consoles = [out / "console.log", out.resolve().parents[1] / "console.log"]
    for path in [*consoles, *sorted((out / "logs").glob("*.log"))]:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(tag + " {") and line not in seen:
                seen.add(line)
                rows.append(json.loads(line[len(tag) + 1 :]))
    return rows


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def by_run(rows: list[dict]) -> dict[tuple[str, str, bool], dict[str, dict[int, dict]]]:
    """(dataset, sampled) -> engine -> concurrency -> row."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        m = re.fullmatch(r"([a-z0-9]+)_c(\d+)(s?)", row["run"])
        if m:
            out[(m.group(1), bool(m.group(3)))][row["engine"]][int(m.group(2))] = row
    return out


def base_of(engine: str) -> str | None:
    """The same server without a drafter (``_dflash4``: a 4-bit drafter; a
    final ``g``: drafts are the drafter's argmax for sampled requests too; a
    final ``f``: every cycle runs the whole block; a final ``s``: every
    cycle is collected before the next)."""
    for suffix in (
        "_dflash", "_dflash4", "_dflashg", "_dflash4g", "_dflashf", "_dflash4f", "_dflashs", "_dflash4s"
    ):
        if engine.endswith(suffix):
            return engine[: -len(suffix)]
    return None


def serving(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    runs = by_run(rows)
    for sampled in (False, True):
        for dataset in DATASETS:
            engines = runs.get((dataset, sampled))
            if not engines:
                continue
            levels = sorted({c for per in engines.values() for c in per})
            what = "temperature 0.7, top-p 0.9" if sampled else "temperature 0"
            lines += [f"### {dataset}, {what}", ""]
            for title, key, digits in (
                ("output tok/s", "out_tok_s", 0),
                ("TPOT p50 (ms)", "tpot_p50_ms", 2),
                ("TTFT p50 (ms)", "ttft_p50_ms", 1),
            ):
                body = [
                    [engine] + [fmt(float(per[c][key]), digits) if c in per and key in per[c] else "-" for c in levels]
                    for engine, per in engines.items()
                ]
                lines += [f"**{title}**", ""] + table(["server"] + [f"c={c}" for c in levels], body)
            body = []
            for engine, per in engines.items():
                base = engines.get(base_of(engine) or "")
                if not base:
                    continue
                body.append(
                    [engine]
                    + [
                        f"{per[c]['out_tok_s'] / base[c]['out_tok_s']:.2f}x"
                        if c in per and c in base and base[c]["out_tok_s"]
                        else "-"
                        for c in levels
                    ]
                )
            if body:
                lines += ["**output tok/s over the same server without a drafter**", ""]
                lines += table(["server"] + [f"c={c}" for c in levels], body)
            failed = [f"{e} c={c}" for e, per in engines.items() for c, r in per.items() if not r.get("ok")]
            if failed:
                lines += [f"Not every request completed: {', '.join(failed)}.", ""]
    return lines


def metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        m = re.match(r"([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$", line.strip())
        if m:
            values[m.group(1)] += float(m.group(3))
    return values


def draft_counts(out: Path, row: dict) -> tuple[float, float]:
    """(drafts, accepted tokens) a server counted during one run."""
    stem = f"{row['engine']}_{row['run']}"
    before = metrics(out / "serve" / f"{stem}_before.metrics.txt")
    after = metrics(out / "serve" / f"{stem}_after.metrics.txt")
    delta = {name: after[name] - before.get(name, 0.0) for name in after}
    num = next((v for k, v in delta.items() if "num_drafts" in k), 0.0)
    accepted = next((v for k, v in delta.items() if "num_accepted_tokens" in k and "per_pos" not in k), 0.0)
    return num, accepted


def drafts(out: Path, rows: list[dict]) -> list[str]:
    """Accepted tokens per draft from the counters around each run."""
    body = []
    for row in rows:
        num, accepted = draft_counts(out, row)
        if num > 0:
            body.append([row["engine"], row["run"], f"{int(num)}", f"{accepted / num:.2f}", f"{accepted / num + 1:.2f}"])
    if not body:
        return []
    return ["## Drafts as the servers count them", ""] + table(
        ["server", "run", "drafts", "accepted per draft", "tokens per cycle"], body
    )


def verify_lens(out: Path, rows: list[dict]) -> list[str]:
    """M6: the share of each server's decode rows run at each verify length
    (1: a plain decode step), and the tokens a decode row yielded."""
    body, lens = [], set()
    per_run = []
    for row in rows:
        stem = f"{row['engine']}_{row['run']}"
        before = metrics(out / "serve" / f"{stem}_before.metrics.txt")
        after = metrics(out / "serve" / f"{stem}_after.metrics.txt")
        counts = {
            int(name.rsplit("_", 1)[1]): after[name] - before.get(name, 0.0)
            for name in after if "spec_rows_len_" in name
        }
        if sum(counts.values()) <= 0:
            continue
        num, accepted = draft_counts(out, row)
        lens.update(length for length, count in counts.items() if count > 0)
        per_run.append((row, counts, num, accepted))
    for row, counts, num, accepted in per_run:
        total = sum(counts.values())
        body.append(
            [row["engine"], row["run"]]
            + [f"{100 * counts.get(length, 0.0) / total:.0f}%" for length in sorted(lens)]
            + [f"{(counts.get(1, 0.0) + num + accepted) / total:.2f}"]
        )
    if not body:
        return []
    return [
        "## Verify lengths the policy ran", "",
        "The share of a server's decode rows (a row per sequence and step) at each verify length: "
        "1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a "
        "decode row yielded on average.", "",
    ] + table(["server", "run", *(f"Q={length}" for length in sorted(lens)), "tokens per row"], body)


def coupling(out: Path, rows: list[dict]) -> list[str]:
    """M5: tokens per cycle of sampled requests with coupled drafts against
    the same server drafting its argmax (label + ``g``)."""
    by_engine = {(row["engine"], row["run"]): row for row in rows}
    body = []
    for (engine, run), row in by_engine.items():
        other = by_engine.get((engine + "g", run))
        if other is None or not run.endswith("s"):
            continue
        (num, accepted), (other_num, other_accepted) = draft_counts(out, row), draft_counts(out, other)
        if not num or not other_num:
            continue
        coupled, argmax = accepted / num + 1, other_accepted / other_num + 1
        body.append([
            f"{engine} {run}", f"{coupled:.2f}", f"{argmax:.2f}", f"{coupled / argmax:.3f}x",
            fmt(row["out_tok_s"], 0), fmt(other["out_tok_s"], 0),
        ])
    if not body:
        return []
    return [
        "## Coupled sampling (temperature 0.7, top-p 0.9)", "",
        "Tokens per cycle and output tok/s with the drafts drawn with the target's noise (coupled) and "
        "with the drafter's argmax as drafts, same seed: the emitted tokens are the same requests' same "
        "streams up to the logits' batch noise.", "",
    ] + table(
        ["server, run", "coupled tokens/cycle", "argmax tokens/cycle", "ratio", "coupled tok/s", "argmax tok/s"],
        body,
    )


def accept(out: Path) -> tuple[list[str], list[dict]]:
    path = out / "accept" / "accept.json"
    if not path.exists():
        return [], []
    rows = json.loads(path.read_text())
    lines = ["## Acceptance study (accept_study.py)", ""]
    lines += ["Tokens per cycle (accepted drafts + the bonus token), `drafter-draw-block` per column.", ""]
    for target in ("bf16", "awq"):
        for mode in ("greedy", "sampled"):
            group = [r for r in rows if r["target"] == target and r["mode"] == mode and r.get("cycles")]
            if not group:
                continue
            arms = list(dict.fromkeys(f"{r['drafter']}-{r['draw']}-b{r['block']}" for r in group))
            body = []
            for dataset in DATASETS:
                per = {f"{r['drafter']}-{r['draw']}-b{r['block']}": r for r in group if r["dataset"] == dataset}
                if per:
                    body.append([dataset] + [fmt(per[a]["tau"], 2) if a in per else "-" for a in arms])
            lines += [f"**{target} target, {mode}**", ""] + table(["dataset"] + arms, body)
    reference = [r for r in rows if (r["drafter"], r["draw"], r["block"]) == ("bf16", "greedy", 10) and r.get("cycles")]
    if reference:
        lines += ["Survival by depth (the share of cycles that accept at least that many drafts), "
                  "bf16 drafter, greedy drafts, block 10; the last columns: how often the drafter's second "
                  "choice was the target's token at the first rejected depth, and how many prompts' replay "
                  "matched the live loop.", ""]
        body = [
            [f"{r['target']} {r['mode']} {r['dataset']}"]
            + [fmt(v, 2) for v in r["survival"]]
            + [fmt(r.get("second_at_first_miss"), 2), r.get("replay_matches_live", "-")]
            for r in reference
        ]
        lines += table(["stream"] + [str(d) for d in range(1, 10)] + ["2nd", "replay"], body)
    return lines, rows


def parity(out: Path) -> list[str]:
    path = out / "parity" / "parity.json"
    if not path.exists():
        return []
    p = json.loads(path.read_text())
    by_position = ", ".join(f"{v:.3f}" for v in p["top1_agreement_by_position"])
    return [
        "## Drafter parity (parity.py)", "",
        f"nano's DFlashDrafter against the checkpoint's own code over {p['blocks']} blocks "
        f"({p['draft_positions']} draft positions): top-1 agreement {p['top1_agreement']:.4f} "
        f"(gate {p['gate']}: {'met' if p['passed'] else 'NOT met'}), blocks identical at every position "
        f"{p['blocks_identical']:.3f}, largest logit difference {p['max_logit_diff']:.3f}, "
        f"largest reference top-1 margin where they disagree {p['max_reference_margin_at_disagreement']:.3f}. "
        f"By draft position: {by_position}. The reference accepted "
        f"{p['reference_mean_accepted_per_cycle']:.2f} tokens per cycle.", "",
    ]


def texts(out: Path) -> list[str]:
    """Identical completions between passes of texts.py."""
    runs = {
        path.name[: -len("_texts.json")]: json.loads(path.read_text())
        for path in sorted((out / "serve").glob("*_texts.json"))
    }
    if not runs:
        return []

    def same(a: list[dict], b: list[dict]) -> str:
        equal = sum(x["text"] == y["text"] for x, y in zip(a, b))
        return f"{equal}/{len(a)}"

    def speed(rows: list[dict]) -> str:
        return fmt(sum(r["completion_tokens"] for r in rows) / sum(r["seconds"] for r in rows), 0)

    body = []
    for name, run in runs.items():
        base = runs.get(base_of(name) or "")
        body.append([
            name,
            same(run["c1"], run["c16"]),
            same(run["c1"], base["c1"]) if base else "-",
            same(run["c16"], base["c1"]) if base else "-",
            speed(run["c1"]),
        ])
    return [
        "## Completions at temperature 0 (texts.py)", "",
        "Completions with identical text: a server's 16-at-a-time pass against its own one-at-a-time "
        "pass (logits depend slightly on the batch size), and a drafter server's passes against the "
        "one-at-a-time pass of the same server without a drafter. Last column: output tok/s of the "
        "one-at-a-time pass.", "",
    ] + table(["server", "c16 = own c1", "c1 = base c1", "c16 = base c1", "c1 tok/s"], body)


def spec_check(out: Path) -> list[str]:
    rows = [row for path in sorted((out / "spec_check").glob("*.json")) for row in json.loads(path.read_text())]
    if not rows:
        return []
    body = [
        [f"{r['label']} {r['mode']} b={r['batch']}", str(r["tokens"]), str(r["not_argmax"]),
         f"{100 * r['not_argmax'] / r['tokens']:.2f}%", fmt(r["max_margin"], 3), str(r["margins_over_0.5"]),
         str(r["margins_over_1"]), fmt(r["tokens_per_cycle"], 2), fmt(r["out_tok_s"], 0),
         fmt(r.get("step_ms_median"), 2), fmt(r.get("step_ms_mean"), 2),
         " ".join(f"{length}:{count}" for length, count in sorted(
             (r.get("rows_by_len") or {}).items(), key=lambda item: int(item[0])) if count) or "-"]
        for r in rows
    ]
    lines = [
        "## Emitted tokens against the target's argmax (spec_check.py)", "",
        "Every generated token against the argmax of one teacher-forced pass of the target over the "
        "finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall "
        "time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). "
        "rows by length: decode rows per verify length of their step (1: no drafts).", "",
    ] + table(
        ["target, mode, batch", "tokens", "not argmax", "share", "largest margin", "margins > 0.5",
         "margins > 1", "tokens/cycle", "tok/s", "ms/step median", "ms/step mean", "rows by length"],
        body,
    )
    single = [r for r in rows if r["batch"] == 1 and r.get("step_ms_median_by_dataset")]
    if single:
        lines += ["Median ms/step at batch 1 per dataset:", ""] + table(
            ["target, mode", *DATASETS],
            [[f"{r['label']} {r['mode']}"] + [fmt(r["step_ms_median_by_dataset"].get(d), 2) for d in DATASETS]
             for r in single],
        )
    return lines


def step_times(out: Path) -> list[str]:
    """M6: milliseconds of one step of each CUDA graph on an empty context,
    as the speculative policy measures them at startup (spec_check.py)."""
    lines = []
    for path in sorted((out / "spec_check").glob("*_times.jsonl")):
        for text in path.read_text().splitlines():
            entry = json.loads(text)
            times = {(size, length): ms for size, length, ms in entry["times"]}
            if not times:
                continue
            sizes = sorted({size for size, _ in times})
            lens = sorted({length for _, length in times})
            lines += [f"**{entry['label']} {entry['mode']}**", ""] + table(
                ["sequences", *(f"Q={length}" for length in lens)],
                [[str(size)] + [fmt(times.get((size, length)), 2) for length in lens] for size in sizes],
            )
            if len(lens) > 1:
                # Tokens per ms relative to a plain step at the M0 study's
                # acceptance (alpaca, greedy): where a cycle pays on an empty context.
                survival = [0.81, 0.58, 0.40, 0.27, 0.18, 0.12, 0.08, 0.05, 0.03]
                body = []
                for size in sizes:
                    plain = times.get((size, 1))
                    if not plain:
                        continue
                    body.append([str(size)] + [
                        f"{(1 + sum(survival[: length - 1])) / times[(size, length)] * plain:.2f}x"
                        if (size, length) in times else "-" for length in lens
                    ])
                lines += ["Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):", ""]
                lines += table(["sequences", *(f"Q={length}" for length in lens)], body)
    if not lines:
        return []
    return [
        "## Step time of each CUDA graph (ms, empty context)", "",
        "One replay of the graph for that many sequences and verify length Q (1: a decode step with "
        "its sampling; with a drafter it also writes the drafter's context), on the null block.", "",
    ] + lines


MIN_OVER_PLAIN = 0.98  # M6: with a drafter, never below this share of the same server without


def m6_gates(out: Path, serve_rows: list[dict], baseline_rows: list[dict], baseline_name: str) -> list[str]:
    """Nano with a drafter at every concurrency: at least MIN_OVER_PLAIN of
    the same server without one (this run's), and above every DFlash
    baseline on the same weights."""
    del out
    runs = by_run(serve_rows + baseline_rows)
    body, verdicts = [], defaultdict(lambda: [0, 0, 0, 0])
    for (dataset, sampled), engines in sorted(runs.items(), key=lambda item: (item[0][1], DATASETS.index(item[0][0]))):
        for engine in sorted(engines):
            base = base_of(engine)
            if not engine.startswith("nano") or base is None or base not in engines:
                continue
            for level in sorted(engines[engine]):
                mine = engines[engine][level].get("out_tok_s")
                plain = engines[base].get(level, {}).get("out_tok_s")
                theirs = {
                    name: engines[name][level]["out_tok_s"]
                    for name in BASELINES.get(base, ()) if level in engines.get(name, {})
                }
                if not mine or not plain:
                    continue
                best = max(theirs, key=theirs.get) if theirs else None
                over_plain = mine / plain
                tally = verdicts[engine]
                tally[0] += over_plain >= MIN_OVER_PLAIN
                tally[1] += 1
                if best:
                    tally[2] += mine > theirs[best]
                    tally[3] += 1
                body.append([
                    f"{dataset}{' sampled' if sampled else ''} c={level} / {engine}", fmt(mine, 0),
                    fmt(plain, 0), f"{over_plain:.2f}x{'' if over_plain >= MIN_OVER_PLAIN else ' NOT met'}",
                    f"{best} {theirs[best]:.0f}" if best else "-",
                    (f"{mine / theirs[best]:.2f}x{'' if mine > theirs[best] else ' NOT met'}") if best else "-",
                ])
    if not body:
        return []
    source = f" (baselines from {baseline_name})" if baseline_rows else ""
    lines = [
        "## M6 gate: with a drafter at every concurrency", "",
        f"Output tok/s against the same nano server without a drafter in this run (at least "
        f"{MIN_OVER_PLAIN}x) and against the fastest DFlash server on the same weights{source} (above).", "",
    ] + table(
        ["workload / server", "tok/s", "no drafter", "over no drafter", "best baseline", "over baseline"], body
    )
    for engine, (ok_plain, total, ok_base, total_base) in sorted(verdicts.items()):
        lines.append(
            f"- {engine}: at least {MIN_OVER_PLAIN}x its own server without a drafter on {ok_plain} of "
            f"{total} workloads; above the best DFlash baseline on {ok_base} of {total_base}."
        )
    return lines + [""]


def m7_gates(out: Path, serve_rows: list[dict]) -> list[str]:
    """A cycle launched behind the one in flight against each cycle
    collected before the next: the same tokens, more of them per second.
    In-process (spec_check.py, both in one run) and, where a server ran both
    ways (the label ending in ``s`` collects every cycle), while serving."""
    lines: list[str] = []
    checks = [row for path in sorted((out / "spec_check").glob("*.json")) for row in json.loads(path.read_text())]
    by_mode = {(r["label"], r["mode"], r["batch"]): r for r in checks}
    body = []
    for (label, mode, batch), chained in by_mode.items():
        if not mode.endswith(", async"):
            continue
        alone = by_mode.get((label, mode[: -len(", async")], batch))
        if alone is None:
            continue
        same = chained.get("same_as_sync")
        body.append([
            f"{label} {mode[: -len(', async')]} b={batch}", fmt(alone["out_tok_s"], 0), fmt(chained["out_tok_s"], 0),
            f"{chained['out_tok_s'] / alone['out_tok_s']:.3f}x",
            fmt(alone.get("step_ms_median"), 2), fmt(chained.get("step_ms_median"), 2),
            "-" if same is None else f"{same} of {chained.get('requests', '?')}",
            str(chained.get("discarded", "-")),
        ])
    if body:
        lines += [
            "## M7: cycles launched behind the cycle in flight", "",
            "The engine core in one process (spec_check.py), each cycle collected before the next "
            "(sync) and launched behind the one in flight (async). Same tokens: requests whose "
            "tokens are the sync run's (greedy streams can split at a near-tie when a discarded "
            "row changes a step's size). Discarded: rows that landed after their sequence had ended.", "",
        ] + table(
            ["target, drafter, batch", "sync tok/s", "async tok/s", "async / sync", "sync ms/step",
             "async ms/step", "same tokens", "discarded"],
            body,
        )
    runs = by_run(serve_rows)
    body = []
    for (dataset, sampled), engines in sorted(runs.items(), key=lambda item: (item[0][1], DATASETS.index(item[0][0]))):
        for engine in sorted(engines):
            alone = engines.get(engine + "s")
            if not engine.startswith("nano") or base_of(engine) is None or not alone:
                continue
            for level in sorted(engines[engine]):
                if level not in alone:
                    continue
                mine, theirs = engines[engine][level], alone[level]
                body.append([
                    f"{dataset}{' sampled' if sampled else ''} c={level} / {engine}",
                    fmt(float(theirs["out_tok_s"]), 0), fmt(float(mine["out_tok_s"]), 0),
                    f"{mine['out_tok_s'] / theirs['out_tok_s']:.3f}x",
                    fmt(float(theirs["ttft_p50_ms"]), 1), fmt(float(mine["ttft_p50_ms"]), 1),
                ])
    if body:
        lines += [
            "**Serving: the same server with SPECULATIVE_CHAIN=0 (each cycle collected) and as it is**", "",
        ] + table(
            ["workload / server", "collected tok/s", "chained tok/s", "chained / collected",
             "collected TTFT p50 (ms)", "chained TTFT p50 (ms)"],
            body,
        )
    return lines


CYCLE_GATE_MS = 8.0  # G3: an AWQ cycle at batch 1
BASELINES = {  # nano's servers with a drafter -> the DFlash servers on the same weights
    "nano_awq": ("vllmw4_dflash",),
    "nano": ("vllm_dflash", "sglang_dflash"),
}


def m4_gates(out: Path, serve_rows: list[dict], baseline_rows: list[dict], baseline_name: str) -> list[str]:
    """G3: the AWQ cycle at batch 1 within CYCLE_GATE_MS. G4: nano with the
    drafter above every DFlash baseline at c=1, temperature 0."""
    lines = []
    checks = [row for path in sorted((out / "spec_check").glob("*.json")) for row in json.loads(path.read_text())]
    cycles = [
        r for r in checks
        if r["label"] == "awq" and r["mode"].startswith("spec") and r["batch"] == 1 and r.get("step_ms_median")
    ]
    if cycles:
        verdicts = [
            f"{r['mode']}: {r['step_ms_median']:.2f} ms ({'met' if r['step_ms_median'] <= CYCLE_GATE_MS else 'NOT met'})"
            for r in cycles
        ]
        lines += [f"G3 (AWQ target, batch 1, a cycle within {CYCLE_GATE_MS} ms, median): " + "; ".join(verdicts) + ".", ""]
    runs = by_run(serve_rows + baseline_rows)
    source = f" (baselines from {baseline_name})" if baseline_rows else ""
    for sampled, name in ((False, "G4: output tok/s at c=1, temperature 0"),
                          (True, "M5: output tok/s at c=1, temperature 0.7, top-p 0.9")):
        body = []
        for dataset in DATASETS:
            engines = runs.get((dataset, sampled), {})
            for engine in sorted(engines):
                names = BASELINES.get(base_of(engine) or "")
                mine = engines[engine].get(1, {}).get("out_tok_s")
                theirs = {n: engines[n][1]["out_tok_s"] for n in names or () if 1 in engines.get(n, {})}
                if not engine.startswith("nano") or mine is None or not theirs:
                    continue
                best = max(theirs, key=theirs.get)
                body.append([
                    f"{dataset} / {engine}", fmt(mine, 0), f"{best} {theirs[best]:.0f}",
                    f"{mine / theirs[best]:.2f}x", "met" if mine > theirs[best] else "NOT met",
                ])
        if body:
            lines += [f"{name}, against the fastest DFlash server on the same weights{source}:", ""]
            lines += table(["dataset / server", "nano tok/s", "best baseline", "ratio", "above"], body)
    return ["## M4 and M5 gates", ""] + lines if lines else []


def block_kernel(out: Path) -> list[str]:
    """The verify rows' attention per kernel, and the M3 gate: the block
    kernel within 1.3x (one sequence) and 1.5x (64) of a decode step's
    attention at the same context, for a block of 10."""
    path = out / "kbench" / "kbench.json"
    if not path.exists():
        return []
    rows = [r for r in json.loads(path.read_text())["rows"] if r["kernel"] == "paged_block_attention"]
    shapes: dict[str, dict[str, dict]] = {}
    for r in rows:
        shapes.setdefault(r["label"], {"shape": r["shape"]})[r["impl"]] = r
    impls = ["decode", "vllm_fa2", "prefill", "block"]

    def cell(r: dict | None) -> str:
        if r is None:
            return "-"
        if "us" in r:
            return fmt(r["us"])
        return "skipped" if "skipped" in r else "FAILED"

    body, gates = [], []
    for label, by_impl in shapes.items():
        decode, block = by_impl.get("decode", {}), by_impl.get("block", {})
        ratio = block["us"] / decode["us"] if "us" in block and "us" in decode else None
        body.append([label] + [cell(by_impl.get(i)) for i in impls] + [fmt(ratio, 2)])
        shape = by_impl["shape"]
        limit = {1: 1.3, 64: 1.5}.get(shape["B"]) if shape["Q"] == 10 else None
        if limit is not None:
            gates.append((label, ratio, limit))
    lines = [
        "## Verify attention (kbench paged_block_attention)", "",
        "Microseconds per call for Q causal queries after each sequence's context, bf16. decode: the "
        "decode kernel on one token per sequence at the same context, what a plain decode step "
        "pays. vllm_fa2: vLLM's flash-attention. prefill: nano's varlen kernel, the verify path "
        "before M3. block: the block kernel.", "",
    ] + table(["shape", *impls, "block / decode"], body)
    if gates:
        met = all(ratio is not None and ratio <= limit for _, ratio, limit in gates)
        lines += [f"Gate (block <= 1.3x decode at B=1, <= 1.5x at B=64, Q=10): {'met' if met else 'NOT met'}: "
                  + ", ".join(f"{label} {fmt(ratio, 2)}x (limit {limit})" for label, ratio, limit in gates)
                  + ".", ""]
    write = [r for r in json.loads(path.read_text())["rows"] if r["kernel"] == "norm_rope_cache_write"]
    if write:
        labels = list(dict.fromkeys(r["label"] for r in write))
        by = {(r["label"], r["impl"]): r for r in write}
        lines += ["## Drafter norm + RoPE + KV write (kbench norm_rope_cache_write)", ""] + table(
            ["shape", "torch us", "triton us"],
            [[label, cell(by.get((label, "torch"))), cell(by.get((label, "triton")))] for label in labels],
        )
    tune = out / "tune" / "tune.json"
    if tune.exists():
        sweep = [r for r in json.loads(tune.read_text()) if r["kernel"] == "block"]
        lines += ["## Block kernel launch settings (tune.py)", ""] + table(
            ["shape", "best config", "best us", "current us", "current / best"],
            [[" ".join(f"{k}={v}" for k, v in r["shape"].items()),
              str(r["best"]["config"]) if r["best"] else "-",
              fmt(r["best"]["us"]) if r["best"] else "-",
              fmt(r["default"]["us"]) if r["default"] else "-",
              fmt(r["speedup_vs_default"], 2)] for r in sweep],
        )
    return lines


def gate(serve_rows: list[dict], accept_rows: list[dict]) -> list[str]:
    """Projected nano tokens/s at c=1 against the best DFlash baseline."""
    runs = by_run(serve_rows)
    body = []
    worst = None
    for target, nano, baselines in (
        ("bf16", "nano", ("vllm_dflash", "sglang_dflash")),
        ("awq", "nano_awq", ("vllmw4_dflash",)),
    ):
        for dataset in DATASETS:
            engines = runs.get((dataset, False), {})
            step = engines.get(nano, {}).get(1, {}).get("tpot_p50_ms")
            best = max((engines.get(b, {}).get(1, {}).get("out_tok_s", 0.0) for b in baselines), default=0.0)
            for drafter in ("bf16", "rtn4"):
                tau = next(
                    (
                        r["tau"]
                        for r in accept_rows
                        if (r["target"], r["mode"], r["dataset"], r["drafter"], r["block"])
                        == (target, "greedy", dataset, drafter, 10)
                        and r.get("cycles")
                    ),
                    None,
                )
                if step is None or tau is None:
                    continue
                cycle = VERIFY_FACTOR * step + DRAFT_MS[drafter] + OTHER_MS
                projected = 1000.0 * tau / cycle
                ratio = projected / best if best else None
                if ratio is not None and drafter == "rtn4":
                    worst = ratio if worst is None else min(worst, ratio)
                body.append(
                    [f"{target} / {dataset} / {drafter} drafter", fmt(step, 2), fmt(tau, 2), fmt(cycle, 2),
                     fmt(projected, 0), fmt(best, 0) if best else "-", f"{ratio:.2f}x" if ratio else "-"]
                )
    if not body:
        return []
    lines = ["## M0 gate: projected nano at c=1", ""]
    lines += [
        f"Cycle estimate: {VERIFY_FACTOR} x nano's own TPOT at c=1 (verify) + {DRAFT_MS['bf16']} ms (bf16 drafter) "
        f"or {DRAFT_MS['rtn4']} ms (4-bit drafter) + {OTHER_MS} ms; tokens per cycle from the acceptance study "
        "(greedy, block 10). The baseline is the faster DFlash server's measured output tok/s at c=1.", "",
    ]
    lines += table(
        ["target / dataset / drafter", "nano TPOT (ms)", "tokens/cycle", "cycle (ms)", "projected tok/s",
         "best DFlash tok/s", "ratio"],
        body,
    )
    if worst is not None:
        verdict = "met" if worst >= GATE else "NOT met"
        lines += [f"Gate (>= {GATE}x on every row with the 4-bit drafter): {verdict}, lowest {worst:.2f}x.", ""]
    return lines


def main() -> None:
    out = Path(sys.argv[1])
    serve_rows = console_rows(out, "P3_SERVE")
    # Baseline servers this run did not start, from another run's console.
    baseline = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    mine = {row["engine"] for row in serve_rows}
    baseline_rows = [] if baseline is None else [
        row for row in console_rows(baseline, "P3_SERVE") if row["engine"] not in mine
    ]
    lines = [f"# Phase 3 run {out.name}", ""]
    environment = out / "environment.txt"
    if environment.exists():
        lines += ["```", environment.read_text().strip(), "```", ""]
    status = out / "status.tsv"
    if status.exists():
        body = [line.split("\t") for line in status.read_text().splitlines() if line.strip()]
        lines += ["## Steps", ""] + table(["step", "rc", "seconds"], body)
    tokens = console_rows(out, "P3_TOKENS")
    if tokens:
        lines += ["## Prompt tokens each server counts (one prompt per dataset)", ""]
        lines += table(["server", *DATASETS], [[t["engine"]] + [str(t.get(d, "-")) for d in DATASETS] for t in tokens])
        same = all(len({str(t.get(d)) for t in tokens}) == 1 for d in DATASETS)
        lines += ["Every server tokenizes the prompts the same." if same else
                  "The servers do NOT count the same prompt tokens: the runs are not comparable as they are.", ""]
    if serve_rows:
        lines += ["## Serving", ""] + serving(serve_rows) + drafts(out, serve_rows)
        lines += verify_lens(out, serve_rows) + coupling(out, serve_rows)
    accept_lines, accept_rows = accept(out)
    lines += texts(out) + spec_check(out) + step_times(out) + parity(out) + block_kernel(out)
    lines += accept_lines + gate(serve_rows, accept_rows)
    lines += m4_gates(out, serve_rows, baseline_rows, baseline.name if baseline else "")
    lines += m6_gates(out, serve_rows, baseline_rows, baseline.name if baseline else "")
    lines += m7_gates(out, serve_rows)
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
