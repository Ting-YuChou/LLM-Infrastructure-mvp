"""
Performance Benchmarking Suite
Comprehensive benchmarking for LLM inference systems
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FIELD_ALIASES = {
    "concurrent": "concurrent_requests",
    "concurrency": "concurrent_requests",
    "model": "request_model",
    "endpoint": "endpoint_url",
    "path": "request_path",
}

FIELD_PARSERS = {
    "endpoint_url": str,
    "request_path": str,
    "auth_path": str,
    "num_requests": int,
    "concurrent_requests": int,
    "max_tokens": int,
    "temperature": float,
    "top_p": float,
    "timeout": float,
    "request_model": str,
    "username": str,
    "password": str,
    "bearer_token": str,
}


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    endpoint_url: str
    request_path: str = "/v1/completions"
    auth_path: str = "/auth/token"
    num_requests: int = 100
    concurrent_requests: int = 10
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: float = 300.0
    request_model: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    bearer_token: Optional[str] = None

    @property
    def base_url(self) -> str:
        return self.endpoint_url.rstrip("/")

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a serialization-safe config snapshot without secrets."""
        config = asdict(self)
        config["has_login_credentials"] = bool(self.username and self.password)
        config["has_bearer_token"] = bool(self.bearer_token)
        config.pop("password", None)
        config.pop("bearer_token", None)
        return config


@dataclass
class BenchmarkResult:
    """Benchmark results."""

    mean_latency: float
    median_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    std_latency: float
    requests_per_second: float
    tokens_per_second: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_tokens: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    total_duration: float
    timestamp: str
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Results saved to: %s", path)


@dataclass
class BenchmarkRunRecord:
    """A persisted benchmark run entry."""

    label: str
    config: Dict[str, Any]
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "config": self.config,
            "result": self.result,
        }


class LatencyTracker:
    """Track latency metrics."""

    def __init__(self) -> None:
        self.latencies: List[float] = []
        self.start_times: Dict[str, float] = {}

    def start(self, request_id: str) -> None:
        self.start_times[request_id] = time.time()

    def end(self, request_id: str) -> float:
        if request_id not in self.start_times:
            return 0.0
        latency = time.time() - self.start_times[request_id]
        self.latencies.append(latency)
        del self.start_times[request_id]
        return latency

    def get_percentile(self, percentile: float) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, percentile))

    def get_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        return {
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "min": min(self.latencies),
            "max": max(self.latencies),
            "std": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0,
        }


class InferenceBenchmark:
    """
    Benchmark LLM inference endpoints.

    Measures:
    - Latency (P50, P95, P99)
    - Throughput (requests/sec, tokens/sec)
    - Success rate
    - Token usage
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.latency_tracker = LatencyTracker()
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.prompt_tokens_list: List[int] = []
        self.completion_tokens_list: List[int] = []
        self.last_error: Optional[str] = None

        logger.info("Initialized benchmark")
        logger.info("Endpoint: %s", config.base_url)
        logger.info("Request path: %s", config.request_path)
        logger.info("Total requests: %s", config.num_requests)
        logger.info("Concurrent requests: %s", config.concurrent_requests)

    async def _resolve_auth_headers(self, client: httpx.AsyncClient) -> Dict[str, str]:
        """Fetch bearer auth if requested."""
        if self.config.bearer_token:
            return {"Authorization": f"Bearer {self.config.bearer_token}"}

        if not (self.config.username and self.config.password):
            return {}

        response = await client.post(
            self.config.auth_path,
            json={"username": self.config.username, "password": self.config.password},
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
        }
        if self.config.request_model:
            payload["model"] = self.config.request_model
        return payload

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        request_id: int,
        prompt: str,
    ) -> Dict[str, Any]:
        request_id_str = f"req_{request_id}"
        payload = self._build_payload(prompt)
        self.latency_tracker.start(request_id_str)

        try:
            response = await client.post(
                self.config.request_path,
                json=payload,
                timeout=self.config.timeout,
            )
            latency = self.latency_tracker.end(request_id_str)
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                self.prompt_tokens_list.append(prompt_tokens)
                self.completion_tokens_list.append(completion_tokens)
                self.total_tokens += prompt_tokens + completion_tokens
                self.successful_requests += 1
                return {
                    "success": True,
                    "latency": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            self.failed_requests += 1
            self.last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            logger.warning("Request %s failed: %s", request_id, self.last_error)
            return {"success": False, "latency": latency, "status_code": response.status_code}
        except Exception as exc:  # pragma: no cover - network failures only
            self.latency_tracker.end(request_id_str)
            self.failed_requests += 1
            self.last_error = str(exc)
            logger.error("Request %s error: %s", request_id, exc)
            return {"success": False, "error": str(exc)}

    async def _run_batch(
        self,
        client: httpx.AsyncClient,
        request_ids: List[int],
        prompts: List[str],
    ) -> None:
        tasks = [
            self._send_request(client, request_id, prompt)
            for request_id, prompt in zip(request_ids, prompts)
        ]
        await asyncio.gather(*tasks)

    async def run_async(self, prompts: List[str]) -> BenchmarkResult:
        logger.info("Starting benchmark...")
        start_time = time.time()
        limits = httpx.Limits(
            max_connections=max(100, self.config.concurrent_requests * 2),
            max_keepalive_connections=max(20, self.config.concurrent_requests),
        )

        async with httpx.AsyncClient(base_url=self.config.base_url, limits=limits) as client:
            auth_headers = await self._resolve_auth_headers(client)
            if auth_headers:
                client.headers.update(auth_headers)

            batch_size = self.config.concurrent_requests
            for start in tqdm(
                range(0, self.config.num_requests, batch_size),
                desc="Benchmarking",
            ):
                batch_end = min(start + batch_size, self.config.num_requests)
                batch_ids = list(range(start, batch_end))
                batch_prompts = [prompts[index % len(prompts)] for index in batch_ids]
                await self._run_batch(client, batch_ids, batch_prompts)

        total_duration = time.time() - start_time
        latency_stats = self.latency_tracker.get_stats()
        result = BenchmarkResult(
            mean_latency=latency_stats["mean"],
            median_latency=latency_stats["median"],
            p50_latency=latency_stats["p50"],
            p95_latency=latency_stats["p95"],
            p99_latency=latency_stats["p99"],
            min_latency=latency_stats["min"],
            max_latency=latency_stats["max"],
            std_latency=latency_stats["std"],
            requests_per_second=self.successful_requests / total_duration if total_duration > 0 else 0,
            tokens_per_second=self.total_tokens / total_duration if total_duration > 0 else 0,
            total_requests=self.config.num_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            success_rate=(self.successful_requests / self.config.num_requests * 100) if self.config.num_requests else 0,
            total_tokens=self.total_tokens,
            avg_prompt_tokens=statistics.mean(self.prompt_tokens_list) if self.prompt_tokens_list else 0,
            avg_completion_tokens=statistics.mean(self.completion_tokens_list) if self.completion_tokens_list else 0,
            total_duration=total_duration,
            timestamp=datetime.now().isoformat(),
            last_error=self.last_error,
        )
        logger.info("Benchmark complete")
        self._print_results(result)
        return result

    def run(self, prompts: List[str]) -> BenchmarkResult:
        return asyncio.run(self.run_async(prompts))

    def _print_results(self, result: BenchmarkResult) -> None:
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print("\nLatency")
        print(f"  Mean:     {result.mean_latency:.3f}s")
        print(f"  Median:   {result.median_latency:.3f}s")
        print(f"  P95:      {result.p95_latency:.3f}s")
        print(f"  P99:      {result.p99_latency:.3f}s")
        print("\nThroughput")
        print(f"  Requests/sec:  {result.requests_per_second:.2f}")
        print(f"  Tokens/sec:    {result.tokens_per_second:.2f}")
        print("\nSuccess")
        print(f"  Successful:    {result.successful_requests}/{result.total_requests}")
        print(f"  Success rate:  {result.success_rate:.1f}%")
        if result.last_error:
            print(f"  Last error:    {result.last_error}")
        print("\n" + "=" * 70)


def get_default_prompts() -> List[str]:
    """Get default test prompts."""
    return [
        "Explain quantum computing in simple terms.",
        "What are the benefits of machine learning?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is climate change?",
        "How do neural networks learn?",
        "Explain the theory of relativity.",
        "What is blockchain technology?",
        "How does the human brain work?",
        "What causes earthquakes?",
    ]


def load_prompts(prompts_file: Optional[str]) -> List[str]:
    """Load prompts from disk or fall back to defaults."""
    if not prompts_file:
        prompts = get_default_prompts()
        logger.info("Using %s default prompts", len(prompts))
        return prompts

    with open(prompts_file, "r") as file:
        data = json.load(file)

    if isinstance(data, list) and data and isinstance(data[0], dict):
        prompts = [item["prompt"] for item in data]
    elif isinstance(data, list):
        prompts = [str(item) for item in data]
    else:
        raise ValueError("Prompts file must contain a JSON array.")

    logger.info("Loaded %s prompts from file", len(prompts))
    return prompts


def normalize_field_name(name: str) -> str:
    field_name = FIELD_ALIASES.get(name, name)
    if field_name not in FIELD_PARSERS:
        raise ValueError(f"Unsupported sweep field: {name}")
    return field_name


def parse_sweep_value(field_name: str, raw_value: str) -> Any:
    parser = FIELD_PARSERS[field_name]
    return parser(raw_value)


def parse_sweep_arguments(raw_sweeps: List[str]) -> Dict[str, List[Any]]:
    """Parse repeated --sweep field=v1,v2 arguments."""
    sweep_definitions: Dict[str, List[Any]] = {}
    for raw_sweep in raw_sweeps:
        if "=" not in raw_sweep:
            raise ValueError(f"Invalid sweep argument: {raw_sweep}")
        raw_field, raw_values = raw_sweep.split("=", 1)
        field_name = normalize_field_name(raw_field.strip())
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError(f"No values provided for sweep field: {field_name}")
        sweep_definitions[field_name] = [parse_sweep_value(field_name, value) for value in values]
    return sweep_definitions


def build_run_matrix(
    base_config: BenchmarkConfig,
    sweep_definitions: Dict[str, List[Any]],
) -> List[tuple[str, BenchmarkConfig]]:
    """Build benchmark run configs from sweep definitions."""
    if not sweep_definitions:
        return [("baseline", base_config)]

    field_names = sorted(sweep_definitions)
    run_matrix: List[tuple[str, BenchmarkConfig]] = []
    for values in itertools.product(*(sweep_definitions[field] for field in field_names)):
        overrides = dict(zip(field_names, values))
        label = "__".join(f"{field}={value}" for field, value in overrides.items())
        run_matrix.append((label, replace(base_config, **overrides)))
    return run_matrix


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "run"


def ensure_output_dir(output_dir: Optional[str]) -> Path:
    """Create a run output directory."""
    if output_dir:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("outputs") / "benchmarks" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_run_records(
    output_dir: Path,
    records: List[BenchmarkRunRecord],
    prompts: List[str],
    sweep_definitions: Dict[str, List[Any]],
) -> None:
    """Persist benchmark artifacts for one or more runs."""
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"

    summary_payload = [record.to_dict() for record in records]
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_runs": len(records),
        "num_prompts": len(prompts),
        "sweeps": {field: values for field, values in sweep_definitions.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    fieldnames = [
        "label",
        "endpoint_url",
        "request_path",
        "request_model",
        "num_requests",
        "concurrent_requests",
        "max_tokens",
        "temperature",
        "top_p",
        "mean_latency",
        "p95_latency",
        "requests_per_second",
        "tokens_per_second",
        "success_rate",
        "successful_requests",
        "failed_requests",
        "total_duration",
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "label": record.label,
                **{
                    key: record.config.get(key)
                    for key in (
                        "endpoint_url",
                        "request_path",
                        "request_model",
                        "num_requests",
                        "concurrent_requests",
                        "max_tokens",
                        "temperature",
                        "top_p",
                    )
                },
                **{
                    key: record.result.get(key)
                    for key in (
                        "mean_latency",
                        "p95_latency",
                        "requests_per_second",
                        "tokens_per_second",
                        "success_rate",
                        "successful_requests",
                        "failed_requests",
                        "total_duration",
                    )
                },
            }
            writer.writerow(row)

    for index, record in enumerate(records, start=1):
        record_path = output_dir / f"run-{index:03d}-{slugify(record.label)}.json"
        record_path.write_text(json.dumps(record.to_dict(), indent=2))

    logger.info("Saved benchmark artifacts to %s", output_dir)


def print_sweep_summary(records: List[BenchmarkRunRecord]) -> None:
    """Print a compact run comparison table."""
    if len(records) <= 1:
        return

    print("\nSweep Summary")
    print("-" * 100)
    print(f"{'Label':40} {'P95(s)':>10} {'Req/s':>10} {'Tok/s':>10} {'Success%':>10}")
    print("-" * 100)
    for record in records:
        result = record.result
        print(
            f"{record.label[:40]:40} "
            f"{result['p95_latency']:>10.3f} "
            f"{result['requests_per_second']:>10.2f} "
            f"{result['tokens_per_second']:>10.2f} "
            f"{result['success_rate']:>10.1f}"
        )
    print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8080", help="Base endpoint URL")
    parser.add_argument("--path", default="/v1/completions", help="Completions request path")
    parser.add_argument("--auth-path", default="/auth/token", help="Auth token endpoint path")
    parser.add_argument("--num-requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument("--model", help="Optional request model field")
    parser.add_argument("--username", help="Fetch a bearer token from the auth endpoint")
    parser.add_argument("--password", help="Password for the auth endpoint")
    parser.add_argument("--bearer-token", help="Use a pre-issued bearer token directly")
    parser.add_argument("--prompts-file", help="JSON file with custom prompts")
    parser.add_argument("--output", help="Save a single run to this JSON file")
    parser.add_argument("--output-dir", help="Directory for sweep artifacts and summary files")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Parameter sweep in the form field=v1,v2. "
        "Supported fields: concurrent_requests, max_tokens, temperature, "
        "top_p, num_requests, timeout, request_model, endpoint_url, request_path.",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    base_config = BenchmarkConfig(
        endpoint_url=args.endpoint,
        request_path=args.path,
        auth_path=args.auth_path,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        request_model=args.model,
        username=args.username,
        password=args.password,
        bearer_token=args.bearer_token,
    )

    sweep_definitions = parse_sweep_arguments(args.sweep)
    run_matrix = build_run_matrix(base_config, sweep_definitions)
    output_dir = ensure_output_dir(args.output_dir) if (args.output_dir or len(run_matrix) > 1) else None

    records: List[BenchmarkRunRecord] = []
    for label, config in run_matrix:
        logger.info("Running benchmark scenario: %s", label)
        benchmark = InferenceBenchmark(config)
        result = benchmark.run(prompts)
        record = BenchmarkRunRecord(
            label=label,
            config=config.to_public_dict(),
            result=result.to_dict(),
        )
        records.append(record)

        if args.output and len(run_matrix) == 1:
            result.to_json(args.output)

    if output_dir is not None:
        persist_run_records(output_dir, records, prompts, sweep_definitions)

    print_sweep_summary(records)


if __name__ == "__main__":
    main()
