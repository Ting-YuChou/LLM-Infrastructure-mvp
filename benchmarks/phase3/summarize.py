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

  python summarize.py <run dir>
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
    return engine[: -len("_dflash")] if engine.endswith("_dflash") else None


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
        m = re.match(r"([A-Za-z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$", line.strip())
        if m:
            values[m.group(1)] += float(m.group(3))
    return values


def drafts(out: Path, rows: list[dict]) -> list[str]:
    """Accepted tokens per draft from the counters around each run."""
    body = []
    for row in rows:
        stem = f"{row['engine']}_{row['run']}"
        before = metrics(out / "serve" / f"{stem}_before.metrics.txt")
        after = metrics(out / "serve" / f"{stem}_after.metrics.txt")
        delta = {name: after[name] - before.get(name, 0.0) for name in after}
        num = next((v for k, v in delta.items() if "num_drafts" in k), 0.0)
        accepted = next((v for k, v in delta.items() if "num_accepted_tokens" in k and "per_pos" not in k), 0.0)
        if num > 0:
            body.append([row["engine"], row["run"], f"{int(num)}", f"{accepted / num:.2f}", f"{accepted / num + 1:.2f}"])
    if not body:
        return []
    return ["## Drafts as the servers count them", ""] + table(
        ["server", "run", "drafts", "accepted per draft", "tokens per cycle"], body
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
         str(r["margins_over_1"]), fmt(r["tokens_per_cycle"], 2), fmt(r["out_tok_s"], 0)]
        for r in rows
    ]
    return [
        "## Emitted tokens against the target's argmax (spec_check.py)", "",
        "Every generated token against the argmax of one teacher-forced pass of the target over the "
        "finished sequence; the margin is the top logit minus the emitted token's.", "",
    ] + table(
        ["target, mode, batch", "tokens", "not argmax", "share", "largest margin", "margins > 0.5",
         "margins > 1", "tokens/cycle", "tok/s"],
        body,
    )


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
    accept_lines, accept_rows = accept(out)
    lines += texts(out) + spec_check(out) + parity(out) + accept_lines + gate(serve_rows, accept_rows)
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
