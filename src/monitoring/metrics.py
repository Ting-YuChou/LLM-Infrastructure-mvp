"""
Monitoring and Metrics Collection
Prometheus metrics for LLM infrastructure monitoring
"""

import time
import logging
from typing import Dict, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Registry
# ============================================================================

# Create custom registry for our metrics
registry = CollectorRegistry()


# ============================================================================
# Request Metrics
# ============================================================================

# Total number of requests
request_counter = Counter(
    'llm_requests_total',
    'Total number of LLM inference requests',
    ['model', 'endpoint', 'status'],
    registry=registry
)

# Request duration histogram
request_duration = Histogram(
    'llm_request_duration_seconds',
    'Request duration in seconds',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry
)

# Active requests gauge
active_requests = Gauge(
    'llm_active_requests',
    'Number of requests currently being processed',
    ['model'],
    registry=registry
)

# Request queue size
request_queue_size = Gauge(
    'llm_request_queue_size',
    'Number of requests waiting in queue',
    ['model'],
    registry=registry
)


# ============================================================================
# Token Metrics
# ============================================================================

# Total tokens processed
tokens_processed = Counter(
    'llm_tokens_processed_total',
    'Total number of tokens processed',
    ['model', 'type'],  # type: prompt or completion
    registry=registry
)

# Tokens per second
tokens_per_second = Gauge(
    'llm_tokens_per_second',
    'Current tokens processed per second',
    ['model'],
    registry=registry
)

# Time to first token
time_to_first_token = Histogram(
    'llm_time_to_first_token_seconds',
    'Time to generate first token',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    registry=registry
)

# Inter-token latency
inter_token_latency = Histogram(
    'llm_inter_token_latency_seconds',
    'Time between consecutive tokens',
    ['model'],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    registry=registry
)


# ============================================================================
# Model Metrics
# ============================================================================

# Model load time
model_load_duration = Summary(
    'llm_model_load_duration_seconds',
    'Time taken to load model',
    ['model'],
    registry=registry
)

# Model memory usage
model_memory_usage = Gauge(
    'llm_model_memory_bytes',
    'Model memory usage in bytes',
    ['model', 'type'],  # type: gpu or cpu
    registry=registry
)

# KV cache usage
kv_cache_usage = Gauge(
    'llm_kv_cache_usage_percent',
    'KV cache usage percentage',
    ['model'],
    registry=registry
)


# ============================================================================
# GPU Metrics
# ============================================================================

# GPU utilization
gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'model'],
    registry=registry
)

# GPU memory usage
gpu_memory_used = Gauge(
    'llm_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id', 'model'],
    registry=registry
)

# GPU temperature
gpu_temperature = Gauge(
    'llm_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id'],
    registry=registry
)


# ============================================================================
# Error Metrics
# ============================================================================

# Errors counter
error_counter = Counter(
    'llm_errors_total',
    'Total number of errors',
    ['model', 'error_type'],
    registry=registry
)

# Timeout counter
timeout_counter = Counter(
    'llm_timeouts_total',
    'Total number of request timeouts',
    ['model'],
    registry=registry
)


# ============================================================================
# Training Metrics
# ============================================================================

# Training step
training_step = Gauge(
    'llm_training_step',
    'Current training step',
    ['model', 'stage'],  # stage: sft, reward, rlhf
    registry=registry
)

# Training loss
training_loss = Gauge(
    'llm_training_loss',
    'Current training loss',
    ['model', 'stage'],
    registry=registry
)

# Learning rate
learning_rate = Gauge(
    'llm_learning_rate',
    'Current learning rate',
    ['model', 'stage'],
    registry=registry
)


# ============================================================================
# Metrics Decorators and Context Managers
# ============================================================================

def track_request_metrics(model_name: str, endpoint: str):
    """
    Decorator to track request metrics
    
    Usage:
        @track_request_metrics("llama2-7b", "/v1/completions")
        def handle_request():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Increment active requests
            active_requests.labels(model=model_name).inc()
            
            # Track duration
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_counter.labels(
                    model=model_name,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                # Record duration
                duration = time.time() - start_time
                request_duration.labels(
                    model=model_name,
                    endpoint=endpoint
                ).observe(duration)
                
                # Increment counter
                request_counter.labels(
                    model=model_name,
                    endpoint=endpoint,
                    status=status
                ).inc()
                
                # Decrement active requests
                active_requests.labels(model=model_name).dec()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Increment active requests
            active_requests.labels(model=model_name).inc()
            
            # Track duration
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_counter.labels(
                    model=model_name,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                # Record duration
                duration = time.time() - start_time
                request_duration.labels(
                    model=model_name,
                    endpoint=endpoint
                ).observe(duration)
                
                # Increment counter
                request_counter.labels(
                    model=model_name,
                    endpoint=endpoint,
                    status=status
                ).inc()
                
                # Decrement active requests
                active_requests.labels(model=model_name).dec()
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def track_gpu_metrics(gpu_id: int, model_name: str):
    """
    Context manager to track GPU metrics
    
    Usage:
        with track_gpu_metrics(0, "llama2-7b"):
            # GPU-intensive operation
            ...
    """
    try:
        import pynvml
        
        # Initialize NVML
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        # Get initial metrics
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Update metrics
        gpu_memory_used.labels(gpu_id=gpu_id, model=model_name).set(info.used)
        gpu_utilization.labels(gpu_id=gpu_id, model=model_name).set(util.gpu)
        gpu_temperature.labels(gpu_id=gpu_id).set(temp)
        
        yield
        
        # Get final metrics
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Update metrics
        gpu_memory_used.labels(gpu_id=gpu_id, model=model_name).set(info.used)
        gpu_utilization.labels(gpu_id=gpu_id, model=model_name).set(util.gpu)
        gpu_temperature.labels(gpu_id=gpu_id).set(temp)
        
    except ImportError:
        logger.warning("pynvml not installed, GPU metrics disabled")
        yield
    except Exception as e:
        logger.error(f"Error tracking GPU metrics: {e}")
        yield


def track_tokens(model_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Track token metrics
    
    Args:
        model_name: Name of the model
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    """
    tokens_processed.labels(model=model_name, type='prompt').inc(prompt_tokens)
    tokens_processed.labels(model=model_name, type='completion').inc(completion_tokens)


def record_time_to_first_token(model_name: str, seconds: float):
    """Record time to first streamed output token/chunk."""
    time_to_first_token.labels(model=model_name).observe(max(0.0, seconds))


def record_inter_token_latency(model_name: str, seconds: float):
    """Record latency between streamed output token/chunk events."""
    inter_token_latency.labels(model=model_name).observe(max(0.0, seconds))


def update_tokens_per_second(model_name: str, total_tokens: int, duration: float):
    """Update current token throughput gauge from a completed request."""
    if total_tokens <= 0 or duration <= 0:
        return
    tokens_per_second.labels(model=model_name).set(total_tokens / duration)


def start_request_tracking(model_name: str) -> float:
    """
    Mark a request as active and return the start timestamp.

    Args:
        model_name: Name of the model/service handling the request

    Returns:
        Start time as a Unix timestamp
    """
    active_requests.labels(model=model_name).inc()
    request_queue_size.labels(model=model_name).set(0)
    return time.time()


def finish_request_tracking(
    model_name: str,
    endpoint: str,
    start_time: float,
    status: str = "success",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error_type: Optional[str] = None,
):
    """
    Record request completion metrics and clear active-request state.

    Args:
        model_name: Name of the model/service handling the request
        endpoint: Request endpoint label
        start_time: Timestamp returned by start_request_tracking
        status: success or error
        prompt_tokens: Prompt tokens processed for the request
        completion_tokens: Completion tokens processed for the request
        error_type: Optional exception name for error accounting
    """
    duration = max(0.0, time.time() - start_time)
    request_duration.labels(model=model_name, endpoint=endpoint).observe(duration)
    request_counter.labels(
        model=model_name,
        endpoint=endpoint,
        status=status,
    ).inc()
    active_requests.labels(model=model_name).dec()
    request_queue_size.labels(model=model_name).set(0)

    if error_type:
        error_counter.labels(model=model_name, error_type=error_type).inc()

    if prompt_tokens or completion_tokens:
        track_tokens(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        update_tokens_per_second(
            model_name=model_name,
            total_tokens=prompt_tokens + completion_tokens,
            duration=duration,
        )


def track_training_metrics(
    model_name: str,
    stage: str,
    step: int,
    loss: float,
    lr: float
):
    """
    Track training metrics
    
    Args:
        model_name: Name of the model
        stage: Training stage (sft, reward, rlhf)
        step: Current training step
        loss: Current loss value
        lr: Current learning rate
    """
    training_step.labels(model=model_name, stage=stage).set(step)
    training_loss.labels(model=model_name, stage=stage).set(loss)
    learning_rate.labels(model=model_name, stage=stage).set(lr)


# ============================================================================
# Metrics Server
# ============================================================================

class MetricsServer:
    """Prometheus metrics HTTP server"""
    
    def __init__(self, port: int = 9090):
        """
        Initialize metrics server
        
        Args:
            port: Port to serve metrics on
        """
        self.port = port
        logger.info(f"Metrics server initialized on port {port}")
    
    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format
        
        Returns:
            Metrics as bytes
        """
        return generate_latest(registry)
    
    def start(self):
        """Start metrics server"""
        from prometheus_client import start_http_server
        
        start_http_server(self.port, registry=registry)
        logger.info(f"Metrics server started on port {self.port}")
        logger.info(f"Metrics available at http://localhost:{self.port}/metrics")


# ============================================================================
# FastAPI Integration
# ============================================================================

def add_metrics_endpoint(app):
    """
    Add metrics endpoint to FastAPI app
    
    Args:
        app: FastAPI application
    """
    from fastapi import Response
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info("Metrics endpoint added: /metrics")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Track a request
    @track_request_metrics("llama2-7b", "/v1/completions")
    def example_request():
        time.sleep(0.5)  # Simulate processing
        return "response"
    
    # Make some requests
    for _ in range(10):
        example_request()
    
    # Track tokens
    track_tokens("llama2-7b", prompt_tokens=100, completion_tokens=50)
    
    # Track training
    track_training_metrics(
        model_name="llama2-7b",
        stage="sft",
        step=1000,
        loss=0.5,
        lr=2e-5
    )
    
    # Print metrics
    print(generate_latest(registry).decode('utf-8'))
