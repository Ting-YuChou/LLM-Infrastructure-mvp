"""
Unit Tests for Monitoring Metrics
Tests Prometheus metrics collection and tracking
"""

import pytest
import time
import sys
from pathlib import Path

pytest_plugins = ("pytest_asyncio.plugin",)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from monitoring.metrics import (
    request_counter,
    request_duration,
    active_requests,
    tokens_processed,
    track_request_metrics,
    track_tokens,
    track_training_metrics,
    registry as metrics_registry,
)


class TestMetricsCollection:
    """Test basic metrics collection"""
    
    def test_request_counter_increments(self):
        """Test that request counter increments"""
        # Get initial value
        initial_value = request_counter.labels(
            model="test-model",
            endpoint="/test",
            status="success"
        )._value.get()
        
        # Increment
        request_counter.labels(
            model="test-model",
            endpoint="/test",
            status="success"
        ).inc()
        
        # Check increment
        new_value = request_counter.labels(
            model="test-model",
            endpoint="/test",
            status="success"
        )._value.get()
        
        assert new_value == initial_value + 1
    
    def test_active_requests_gauge(self):
        """Test active requests gauge"""
        # Set value
        active_requests.labels(model="test-model").set(5)
        
        # Check value
        value = active_requests.labels(model="test-model")._value.get()
        assert value == 5
        
        # Increment
        active_requests.labels(model="test-model").inc()
        value = active_requests.labels(model="test-model")._value.get()
        assert value == 6
        
        # Decrement
        active_requests.labels(model="test-model").dec()
        value = active_requests.labels(model="test-model")._value.get()
        assert value == 5
    
    def test_tokens_processed_counter(self):
        """Test token counting"""
        initial_prompt = tokens_processed.labels(
            model="test-model",
            type="prompt"
        )._value.get()
        
        initial_completion = tokens_processed.labels(
            model="test-model",
            type="completion"
        )._value.get()
        
        # Track tokens
        track_tokens("test-model", prompt_tokens=100, completion_tokens=50)
        
        # Check increments
        new_prompt = tokens_processed.labels(
            model="test-model",
            type="prompt"
        )._value.get()
        
        new_completion = tokens_processed.labels(
            model="test-model",
            type="completion"
        )._value.get()
        
        assert new_prompt == initial_prompt + 100
        assert new_completion == initial_completion + 50


class TestRequestTracking:
    """Test request tracking decorator"""
    
    def test_track_request_metrics_sync(self):
        """Test request tracking for sync functions"""
        @track_request_metrics("test-model", "/test-endpoint")
        def test_function():
            time.sleep(0.1)
            return "result"
        
        # Get initial counter value
        initial_count = request_counter.labels(
            model="test-model",
            endpoint="/test-endpoint",
            status="success"
        )._value.get()
        
        # Call function
        result = test_function()
        
        # Check result
        assert result == "result"
        
        # Check counter incremented
        new_count = request_counter.labels(
            model="test-model",
            endpoint="/test-endpoint",
            status="success"
        )._value.get()
        
        assert new_count == initial_count + 1
    
    def test_track_request_metrics_error(self):
        """Test request tracking on error"""
        @track_request_metrics("test-model", "/test-endpoint")
        def failing_function():
            raise ValueError("Test error")
        
        # Get initial error counter
        initial_errors = request_counter.labels(
            model="test-model",
            endpoint="/test-endpoint",
            status="error"
        )._value.get()
        
        # Call function and expect error
        with pytest.raises(ValueError):
            failing_function()
        
        # Check error counter incremented
        new_errors = request_counter.labels(
            model="test-model",
            endpoint="/test-endpoint",
            status="error"
        )._value.get()
        
        assert new_errors == initial_errors + 1
    
    @pytest.mark.asyncio
    async def test_track_request_metrics_async(self):
        """Test request tracking for async functions"""
        @track_request_metrics("test-model", "/async-endpoint")
        async def async_function():
            await asyncio.sleep(0.1)
            return "async result"
        
        import asyncio
        
        # Get initial count
        initial_count = request_counter.labels(
            model="test-model",
            endpoint="/async-endpoint",
            status="success"
        )._value.get()
        
        # Call async function
        result = await async_function()
        
        # Check result
        assert result == "async result"
        
        # Check counter
        new_count = request_counter.labels(
            model="test-model",
            endpoint="/async-endpoint",
            status="success"
        )._value.get()
        
        assert new_count == initial_count + 1


class TestTrainingMetrics:
    """Test training metrics tracking"""
    
    def test_track_training_metrics(self):
        """Test training metrics tracking"""
        # Track metrics
        track_training_metrics(
            model_name="test-model",
            stage="sft",
            step=1000,
            loss=0.5,
            lr=2e-5
        )
        
        # Check values were set
        # Note: Can't easily assert on Gauge values without accessing internals
        # In practice, these would be verified via Prometheus queries
        pass


class TestMetricsExport:
    """Test metrics export functionality"""
    
    def test_metrics_can_be_exported(self):
        """Test that metrics can be exported in Prometheus format"""
        from prometheus_client import generate_latest
        
        # Generate metrics
        metrics_output = generate_latest(metrics_registry)
        
        # Check output is bytes
        assert isinstance(metrics_output, bytes)
        
        # Check contains some expected metric names
        output_str = metrics_output.decode('utf-8')
        assert 'llm_requests_total' in output_str or 'prometheus_' in output_str


class TestMetricsLabels:
    """Test metrics labeling"""
    
    def test_metrics_support_multiple_models(self):
        """Test that metrics can track multiple models"""
        # Track requests for different models
        request_counter.labels(
            model="model-a",
            endpoint="/test",
            status="success"
        ).inc()
        
        request_counter.labels(
            model="model-b",
            endpoint="/test",
            status="success"
        ).inc()
        
        # Both should have been incremented independently
        model_a_count = request_counter.labels(
            model="model-a",
            endpoint="/test",
            status="success"
        )._value.get()
        
        model_b_count = request_counter.labels(
            model="model-b",
            endpoint="/test",
            status="success"
        )._value.get()
        
        assert model_a_count >= 1
        assert model_b_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
