"""
Performance Benchmarking Suite
Comprehensive benchmarking for LLM inference systems
"""

import time
import asyncio
import statistics
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import httpx
import numpy as np
from tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    endpoint_url: str
    model: str = "default"
    num_requests: int = 100
    concurrent_requests: int = 10
    max_tokens: int = 256
    temperature: float = 0.7
    timeout: float = 300.0


@dataclass
class BenchmarkResult:
    """Benchmark results"""
    # Latency metrics (seconds)
    mean_latency: float
    median_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    std_latency: float
    
    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float
    
    # Success metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # Token metrics
    total_tokens: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    
    # Time metrics
    total_duration: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {filepath}")


class LatencyTracker:
    """Track latency metrics"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.start_times: Dict[str, float] = {}
    
    def start(self, request_id: str):
        """Start timing a request"""
        self.start_times[request_id] = time.time()
    
    def end(self, request_id: str) -> float:
        """End timing and record latency"""
        if request_id not in self.start_times:
            return 0.0
        
        latency = time.time() - self.start_times[request_id]
        self.latencies.append(latency)
        del self.start_times[request_id]
        return latency
    
    def get_percentile(self, p: float) -> float:
        """Get percentile value"""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, p)
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {
                'mean': 0.0,
                'median': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
            }
        
        return {
            'mean': statistics.mean(self.latencies),
            'median': statistics.median(self.latencies),
            'p50': self.get_percentile(50),
            'p95': self.get_percentile(95),
            'p99': self.get_percentile(99),
            'min': min(self.latencies),
            'max': max(self.latencies),
            'std': statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0,
        }


class InferenceBenchmark:
    """
    Benchmark LLM inference endpoints
    
    Measures:
    - Latency (P50, P95, P99)
    - Throughput (requests/sec, tokens/sec)
    - Success rate
    - Token usage
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.latency_tracker = LatencyTracker()
        
        # Results tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.prompt_tokens_list = []
        self.completion_tokens_list = []
        
        logger.info("Initialized benchmark")
        logger.info(f"Endpoint: {config.endpoint_url}")
        logger.info(f"Total requests: {config.num_requests}")
        logger.info(f"Concurrent requests: {config.concurrent_requests}")
    
    async def _send_request(
        self,
        client: httpx.AsyncClient,
        request_id: int,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Send single inference request
        
        Args:
            client: HTTP client
            request_id: Unique request ID
            prompt: Input prompt
            
        Returns:
            Response data
        """
        request_id_str = f"req_{request_id}"
        
        # Prepare request
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False
        }
        
        # Start timing
        self.latency_tracker.start(request_id_str)
        
        try:
            # Send request
            response = await client.post(
                f"{self.config.endpoint_url}/v1/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            # End timing
            latency = self.latency_tracker.end(request_id_str)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                # Track tokens
                usage = data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                
                self.prompt_tokens_list.append(prompt_tokens)
                self.completion_tokens_list.append(completion_tokens)
                self.total_tokens += prompt_tokens + completion_tokens
                
                self.successful_requests += 1
                
                return {
                    'success': True,
                    'latency': latency,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                }
            else:
                self.failed_requests += 1
                logger.warning(f"Request {request_id} failed: {response.status_code}")
                return {'success': False, 'latency': latency}
        
        except Exception as e:
            self.latency_tracker.end(request_id_str)
            self.failed_requests += 1
            logger.error(f"Request {request_id} error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_batch(
        self,
        client: httpx.AsyncClient,
        request_ids: List[int],
        prompts: List[str]
    ):
        """Run batch of concurrent requests"""
        tasks = [
            self._send_request(client, req_id, prompt)
            for req_id, prompt in zip(request_ids, prompts)
        ]
        
        await asyncio.gather(*tasks)
    
    async def run_async(self, prompts: List[str]) -> BenchmarkResult:
        """
        Run async benchmark
        
        Args:
            prompts: List of test prompts
            
        Returns:
            Benchmark results
        """
        logger.info("Starting benchmark...")
        start_time = time.time()
        
        # Create HTTP client
        async with httpx.AsyncClient() as client:
            
            # Process in batches for concurrency control
            batch_size = self.config.concurrent_requests
            
            for i in tqdm(range(0, self.config.num_requests, batch_size), desc="Benchmarking"):
                batch_end = min(i + batch_size, self.config.num_requests)
                batch_ids = list(range(i, batch_end))
                batch_prompts = [prompts[j % len(prompts)] for j in batch_ids]
                
                await self._run_batch(client, batch_ids, batch_prompts)
        
        total_duration = time.time() - start_time
        
        # Compute metrics
        latency_stats = self.latency_tracker.get_stats()
        
        result = BenchmarkResult(
            # Latency
            mean_latency=latency_stats['mean'],
            median_latency=latency_stats['median'],
            p50_latency=latency_stats['p50'],
            p95_latency=latency_stats['p95'],
            p99_latency=latency_stats['p99'],
            min_latency=latency_stats['min'],
            max_latency=latency_stats['max'],
            std_latency=latency_stats['std'],
            
            # Throughput
            requests_per_second=self.successful_requests / total_duration if total_duration > 0 else 0,
            tokens_per_second=self.total_tokens / total_duration if total_duration > 0 else 0,
            
            # Success
            total_requests=self.config.num_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            success_rate=self.successful_requests / self.config.num_requests * 100,
            
            # Tokens
            total_tokens=self.total_tokens,
            avg_prompt_tokens=statistics.mean(self.prompt_tokens_list) if self.prompt_tokens_list else 0,
            avg_completion_tokens=statistics.mean(self.completion_tokens_list) if self.completion_tokens_list else 0,
            
            # Time
            total_duration=total_duration,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("Benchmark complete!")
        self._print_results(result)
        
        return result
    
    def run(self, prompts: List[str]) -> BenchmarkResult:
        """
        Run benchmark (sync wrapper)
        
        Args:
            prompts: List of test prompts
            
        Returns:
            Benchmark results
        """
        return asyncio.run(self.run_async(prompts))
    
    def _print_results(self, result: BenchmarkResult):
        """Print formatted results"""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        
        print("\n📊 LATENCY METRICS (seconds)")
        print(f"  Mean:     {result.mean_latency:.3f}s")
        print(f"  Median:   {result.median_latency:.3f}s")
        print(f"  P95:      {result.p95_latency:.3f}s")
        print(f"  P99:      {result.p99_latency:.3f}s")
        print(f"  Min:      {result.min_latency:.3f}s")
        print(f"  Max:      {result.max_latency:.3f}s")
        print(f"  Std Dev:  {result.std_latency:.3f}s")
        
        print("\n🚀 THROUGHPUT METRICS")
        print(f"  Requests/sec:  {result.requests_per_second:.2f}")
        print(f"  Tokens/sec:    {result.tokens_per_second:.2f}")
        
        print("\n✅ SUCCESS METRICS")
        print(f"  Total requests:      {result.total_requests}")
        print(f"  Successful:          {result.successful_requests}")
        print(f"  Failed:              {result.failed_requests}")
        print(f"  Success rate:        {result.success_rate:.1f}%")
        
        print("\n🎯 TOKEN METRICS")
        print(f"  Total tokens:        {result.total_tokens}")
        print(f"  Avg prompt tokens:   {result.avg_prompt_tokens:.1f}")
        print(f"  Avg completion:      {result.avg_completion_tokens:.1f}")
        
        print("\n⏱️  TIME METRICS")
        print(f"  Total duration:      {result.total_duration:.2f}s")
        print(f"  Timestamp:           {result.timestamp}")
        
        print("\n" + "=" * 70)


def get_default_prompts() -> List[str]:
    """Get default test prompts"""
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


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Inference endpoint URL"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "default"),
        help="Model name/path to send in the completion request"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Total number of requests"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per request"
    )
    parser.add_argument(
        "--prompts-file",
        help="JSON file with custom prompts"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            data = json.load(f)
            prompts = [item['prompt'] for item in data]
        logger.info(f"Loaded {len(prompts)} prompts from file")
    else:
        prompts = get_default_prompts()
        logger.info(f"Using {len(prompts)} default prompts")
    
    # Create benchmark config
    config = BenchmarkConfig(
        endpoint_url=args.endpoint,
        model=args.model,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
        max_tokens=args.max_tokens,
    )
    
    # Run benchmark
    benchmark = InferenceBenchmark(config)
    results = benchmark.run(prompts)
    
    # Save results if requested
    if args.output:
        results.to_json(args.output)


if __name__ == "__main__":
    main()
