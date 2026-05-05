"""
Load Testing with Locust
Simulates realistic user traffic patterns for LLM inference endpoints
"""

import random
import time
import os
from typing import List, Dict
import json

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

# Configure Locust logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the key differences between machine learning and deep learning?",
    "How does blockchain technology work?",
    "Describe the process of photosynthesis.",
    "What causes climate change?",
    "Explain the theory of relativity.",
    "How do neural networks learn?",
    "What is the water cycle?",
    "Describe how the internet works.",
    "What is artificial intelligence?",
    "How does DNA replication occur?",
    "Explain the greenhouse effect.",
    "What is quantum entanglement?",
    "How do vaccines work?",
    "Describe the structure of an atom.",
]

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "default")


# =============================================================================
# Custom Metrics Tracking
# =============================================================================

class MetricsCollector:
    """Collect custom metrics during load testing"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
        self.error_count = 0
        self.ttft_samples = []  # Time to first token
    
    def record_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        ttft: float = None
    ):
        """Record successful request metrics"""
        self.request_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        
        if ttft is not None:
            self.ttft_samples.append(ttft)
    
    def record_error(self):
        """Record failed request"""
        self.error_count += 1
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'total_tokens': self.total_tokens,
            'avg_prompt_tokens': self.total_prompt_tokens / max(1, self.request_count),
            'avg_completion_tokens': self.total_completion_tokens / max(1, self.request_count),
            'avg_ttft': sum(self.ttft_samples) / max(1, len(self.ttft_samples)) if self.ttft_samples else 0,
        }


# Global metrics collector
metrics_collector = MetricsCollector()


# =============================================================================
# Locust User Classes
# =============================================================================

class LLMUser(HttpUser):
    """
    Simulated user for LLM inference endpoint
    
    Behavior:
    - Sends completion requests with varying prompts
    - Waits between requests (think time)
    - Tracks token usage and latency
    """
    
    # Wait time between requests (simulates user think time)
    wait_time = between(1, 5)  # 1-5 seconds
    
    def on_start(self):
        """Called when user starts"""
        # Authenticate if needed
        self.auth_token = self.get_auth_token()
    
    def get_auth_token(self) -> str:
        """Get authentication token"""
        try:
            response = self.client.post(
                "/auth/token",
                json={
                    "username": "loadtest",
                    "password": "loadtest123"
                },
                name="/auth/token"
            )
            
            if response.status_code == 200:
                return response.json().get('access_token', '')
            else:
                logger.warning(f"Auth failed: {response.status_code}")
                return ''
        
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return ''
    
    @task(10)
    def short_completion(self):
        """Short completion request (10x weight)"""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        self._send_completion_request(
            prompt=prompt,
            max_tokens=128,
            name="/v1/completions [short]"
        )
    
    @task(5)
    def medium_completion(self):
        """Medium completion request (5x weight)"""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        self._send_completion_request(
            prompt=prompt,
            max_tokens=256,
            name="/v1/completions [medium]"
        )
    
    @task(2)
    def long_completion(self):
        """Long completion request (2x weight)"""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        self._send_completion_request(
            prompt=prompt,
            max_tokens=512,
            name="/v1/completions [long]"
        )
    
    @task(1)
    def streaming_completion(self):
        """Streaming completion request (1x weight)"""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        self._send_completion_request(
            prompt=prompt,
            max_tokens=256,
            stream=True,
            name="/v1/completions [stream]"
        )
    
    def _send_completion_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        stream: bool = False,
        name: str = "/v1/completions"
    ):
        """
        Send completion request
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Enable streaming
            name: Request name for Locust stats
        """
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        payload = {
            "model": DEFAULT_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": stream
        }
        
        start_time = time.time()
        
        try:
            with self.client.post(
                "/v1/completions",
                json=payload,
                headers=headers,
                catch_response=True,
                name=name
            ) as response:
                
                if response.status_code == 200:
                    # Parse response
                    data = response.json()
                    
                    # Extract token usage
                    usage = data.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    
                    # Record metrics
                    metrics_collector.record_request(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
                    
                    response.success()
                
                elif response.status_code == 429:
                    # Rate limited
                    response.failure("Rate limited (429)")
                    metrics_collector.record_error()
                
                else:
                    # Other error
                    response.failure(f"HTTP {response.status_code}")
                    metrics_collector.record_error()
        
        except Exception as e:
            logger.error(f"Request error: {e}")
            metrics_collector.record_error()


class HighVolumeUser(HttpUser):
    """
    High-volume user for stress testing
    
    Sends requests more aggressively with minimal wait time
    """
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    @task
    def rapid_fire_requests(self):
        """Send requests rapidly"""
        prompt = random.choice(SAMPLE_PROMPTS)
        
        payload = {
            "model": DEFAULT_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.7
        }
        
        self.client.post("/v1/completions", json=payload, name="/v1/completions [rapid]")


# =============================================================================
# Event Handlers
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    logger.info("=" * 70)
    logger.info("Load test starting")
    logger.info(f"Target host: {environment.host}")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    logger.info("=" * 70)
    logger.info("Load test completed")
    
    # Print custom metrics
    summary = metrics_collector.get_summary()
    
    logger.info("\nCustom Metrics Summary:")
    logger.info(f"  Total requests:          {summary['total_requests']}")
    logger.info(f"  Total errors:            {summary['total_errors']}")
    logger.info(f"  Total tokens processed:  {summary['total_tokens']}")
    logger.info(f"  Avg prompt tokens:       {summary['avg_prompt_tokens']:.1f}")
    logger.info(f"  Avg completion tokens:   {summary['avg_completion_tokens']:.1f}")
    
    if summary['avg_ttft'] > 0:
        logger.info(f"  Avg TTFT:                {summary['avg_ttft']:.3f}s")
    
    logger.info("=" * 70)
    
    # Save metrics to file
    with open('load_test_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Metrics saved to: load_test_metrics.json")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called on each request (for additional logging if needed)"""
    pass


# =============================================================================
# Usage Instructions
# =============================================================================

"""
Run load test with Locust:

# Start Locust web UI
locust -f scripts/load_test.py --host=http://localhost:8000

# Run headless (command-line mode)
locust -f scripts/load_test.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m \
    --headless

# Distributed load testing (master)
locust -f scripts/load_test.py \
    --host=http://localhost:8000 \
    --master

# Distributed load testing (worker)
locust -f scripts/load_test.py \
    --worker \
    --master-host=localhost

# Export results
locust -f scripts/load_test.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --html=load_test_report.html \
    --csv=load_test_results

Parameters:
  --users: Total number of concurrent users
  --spawn-rate: Users spawned per second
  --run-time: Test duration (e.g., 10m, 1h)
  --host: Target endpoint URL
"""
