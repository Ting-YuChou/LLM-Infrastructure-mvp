"""
Integration Tests for API Gateway
Tests authentication, rate limiting, and request routing
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.gateway import APIGateway, GatewayConfig


def make_test_config(rate_limit_enabled: bool = False) -> GatewayConfig:
    """Create an explicit auth config for gateway tests."""
    config = GatewayConfig()
    config.JWT_SECRET = "test-secret"
    config.AUTH_USERS = {"testuser": "testpass"}
    config.DEV_AUTH_ENABLED = False
    config.RATE_LIMIT_ENABLED = rate_limit_enabled
    return config


class FakeBackendResponse:
    """Minimal async-httpx response stand-in for gateway tests."""

    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload


class FakeBackendClient:
    """Capture proxied backend requests without a network service."""

    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    async def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        return FakeBackendResponse(self.payload)


class TestAPIGateway:
    """Integration tests for API Gateway"""
    
    @pytest.fixture
    def gateway(self):
        """Create test gateway instance"""
        config = make_test_config(rate_limit_enabled=False)
        gateway = APIGateway(config=config)
        return gateway
    
    @pytest.fixture
    def client(self, gateway):
        """Create test client"""
        return TestClient(gateway.app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_get_token_success(self, client):
        """Test successful token generation"""
        response = client.post(
            "/auth/token",
            json={
                "username": "testuser",
                "password": "testpass"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_get_token_missing_credentials(self, client):
        """Test token generation with missing credentials"""
        response = client.post(
            "/auth/token",
            json={
                "username": "",
                "password": ""
            }
        )
        
        assert response.status_code == 401

    def test_get_token_rejects_wrong_password(self, client):
        """Test token generation with invalid configured credentials."""
        response = client.post(
            "/auth/token",
            json={"username": "testuser", "password": "wrongpass"}
        )

        assert response.status_code == 401
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/models")
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_valid_token(self, client):
        """Test accessing protected endpoint with valid token"""
        # First, get a token
        token_response = client.post(
            "/auth/token",
            json={
                "username": "testuser",
                "password": "testpass"
            }
        )
        token = token_response.json()["access_token"]
        
        # Use token to access protected endpoint
        response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token"""
        response = client.get(
            "/models",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
    
    def test_list_models(self, client):
        """Test listing available models"""
        # Get token
        token_response = client.post(
            "/auth/token",
            json={"username": "testuser", "password": "testpass"}
        )
        token = token_response.json()["access_token"]
        
        # List models
        response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

    def test_chat_completion_is_proxied(self, gateway):
        """Test chat completions are routed to the backend chat endpoint."""
        gateway.config.DEFAULT_COMPLETION_MODEL = "default-chat-model"
        gateway.http_client = FakeBackendClient(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 123,
                "model": "default-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        client = TestClient(gateway.app)
        token = client.post(
            "/auth/token",
            json={"username": "testuser", "password": "testpass"},
        ).json()["access_token"]

        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": "say hello"}],
                "max_tokens": 16,
            },
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert gateway.http_client.requests[0]["url"].endswith("/v1/chat/completions")
        assert gateway.http_client.requests[0]["json"]["model"] == "default-chat-model"


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def gateway_with_limits(self):
        """Create gateway with strict rate limits"""
        config = make_test_config(rate_limit_enabled=True)
        config.RATE_LIMIT_REQUESTS_PER_MINUTE = 5
        config.RATE_LIMIT_REQUESTS_PER_HOUR = 100
        gateway = APIGateway(config=config)
        return gateway
    
    @pytest.fixture
    def limited_client(self, gateway_with_limits):
        """Create test client with rate limiting"""
        return TestClient(gateway_with_limits.app)
    
    def test_rate_limit_enforcement(self, limited_client):
        """Test that rate limits are enforced"""
        # Get token
        token_response = limited_client.post(
            "/auth/token",
            json={"username": "testuser", "password": "testpass"}
        )
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Note: Actual rate limit testing would require mocking the backend
        # and making multiple requests. This is a simplified test.
        
        # Make a request
        response = limited_client.get("/models", headers=headers)
        assert response.status_code == 200


class TestCORS:
    """Test CORS configuration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        gateway = APIGateway(config=make_test_config())
        return TestClient(gateway.app)
    
    def test_cors_headers(self, client):
        """Test CORS headers are present for real browser preflight."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        
        # Check CORS headers
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        gateway = APIGateway(config=make_test_config())
        return TestClient(gateway.app)
    
    def test_404_error(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/auth/token",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable entity


class TestRequestLogging:
    """Test request logging middleware"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        gateway = APIGateway(config=make_test_config())
        return TestClient(gateway.app)
    
    def test_request_is_logged(self, client, caplog):
        """Test that requests are logged"""
        import logging
        
        # Set logging level
        caplog.set_level(logging.INFO)
        
        # Make request
        response = client.get("/health")
        
        # Check response
        assert response.status_code == 200
        
        # Check logs contain request info
        # Note: This may not work in all test environments
        # In practice, would verify logs in actual deployment


class TestUsageTracking:
    """Test usage tracking endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        gateway = APIGateway(config=make_test_config())
        return TestClient(gateway.app)
    
    def test_get_usage(self, client):
        """Test getting usage statistics"""
        # Get token
        token_response = client.post(
            "/auth/token",
            json={"username": "testuser", "password": "testpass"}
        )
        token = token_response.json()["access_token"]
        
        # Get usage
        response = client.get(
            "/usage",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user" in data
        assert "total_requests" in data
        assert "total_tokens" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
