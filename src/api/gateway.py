"""
API Gateway for LLM Infrastructure
Provides authentication, rate limiting, and request routing
"""

import os
import time
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import jwt
import redis
from collections import defaultdict
import httpx

try:
    from src.monitoring.metrics import (
        add_metrics_endpoint,
        finish_request_tracking,
        start_request_tracking,
    )
except ModuleNotFoundError:
    from monitoring.metrics import (  # type: ignore[no-redef]
        add_metrics_endpoint,
        finish_request_tracking,
        start_request_tracking,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_env_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variables safely."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    """Parse integer environment variables safely."""
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_env_list(name: str, default: List[str]) -> List[str]:
    """Parse comma-delimited list environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or default


def _get_env_users(name: str = "AUTH_USERS") -> Dict[str, str]:
    """Parse comma-delimited username:password pairs."""
    value = os.getenv(name)
    if not value:
        return {}

    users: Dict[str, str] = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        if ":" not in pair:
            raise ValueError(f"{name} entries must use username:password format")
        username, password = pair.split(":", 1)
        username = username.strip()
        password = password.strip()
        if username and password:
            users[username] = password
    return users


# ============================================================================
# Configuration
# ============================================================================

class GatewayConfig:
    """API Gateway configuration"""
    
    # JWT Settings
    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    AUTH_USERS = _get_env_users()
    DEV_AUTH_ENABLED = _get_env_bool("DEV_AUTH_ENABLED", False)
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = _get_env_bool("RATE_LIMIT_ENABLED", True)
    RATE_LIMIT_REQUESTS_PER_MINUTE = _get_env_int("RATE_LIMIT_REQUESTS_PER_MINUTE", 60)
    RATE_LIMIT_REQUESTS_PER_HOUR = _get_env_int("RATE_LIMIT_REQUESTS_PER_HOUR", 1000)
    
    # Redis for distributed rate limiting
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Backend services
    VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://localhost:8000")
    DEFAULT_COMPLETION_MODEL = os.getenv("DEFAULT_COMPLETION_MODEL")
    
    # CORS
    CORS_ORIGINS = _get_env_list("CORS_ORIGINS", ["*"])  # Configure for production
    
    # Security
    ALLOWED_HOSTS = _get_env_list("ALLOWED_HOSTS", ["*"])  # Configure for production


# ============================================================================
# Request/Response Models
# ============================================================================

class TokenRequest(BaseModel):
    """Request for API token"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Response with API token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CompletionRequest(BaseModel):
    """Completion request model"""
    model: Optional[str] = None
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    """Chat message model for OpenAI-compatible chat requests."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class UsageInfo(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Completion response model"""
    id: str
    model: str
    text: str
    usage: UsageInfo
    created_at: str


# ============================================================================
# Authentication
# ============================================================================

class AuthManager:
    """Handles JWT-based authentication"""
    
    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm
    
    def create_token(self, username: str, expires_hours: int = 24) -> str:
        """
        Create JWT token
        
        Args:
            username: Username
            expires_hours: Token expiration in hours
            
        Returns:
            JWT token string
        """
        expiration = datetime.utcnow() + timedelta(hours=expires_hours)
        
        payload = {
            "sub": username,
            "exp": expiration,
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict:
        """
        Verify JWT token
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload
            
        Raises:
            HTTPException if token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Redis-based distributed rate limiter
    
    Implements sliding window rate limiting with multiple time windows
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter
        
        Args:
            redis_client: Redis client for distributed limiting
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        self.redis = redis_client
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        
        # Fallback to in-memory if Redis unavailable
        self.memory_store: Dict[str, List[float]] = defaultdict(list)
    
    def _get_current_minute_key(self, user_id: str) -> str:
        """Get Redis key for current minute"""
        minute = int(time.time() / 60)
        return f"ratelimit:{user_id}:minute:{minute}"
    
    def _get_current_hour_key(self, user_id: str) -> str:
        """Get Redis key for current hour"""
        hour = int(time.time() / 3600)
        return f"ratelimit:{user_id}:hour:{hour}"
    
    def check_rate_limit(self, user_id: str) -> tuple[bool, Dict[str, int]]:
        """
        Check if user is within rate limits
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (allowed: bool, remaining: dict)
        """
        current_time = time.time()
        
        if self.redis:
            # Use Redis for distributed rate limiting
            return self._check_redis_rate_limit(user_id, current_time)
        else:
            # Use in-memory rate limiting
            return self._check_memory_rate_limit(user_id, current_time)
    
    def _check_redis_rate_limit(
        self,
        user_id: str,
        current_time: float
    ) -> tuple[bool, Dict[str, int]]:
        """Check rate limit using Redis"""
        minute_key = self._get_current_minute_key(user_id)
        hour_key = self._get_current_hour_key(user_id)
        
        # Increment counters
        minute_count = self.redis.incr(minute_key)
        hour_count = self.redis.incr(hour_key)
        
        # Set expiration if this is first request in window
        if minute_count == 1:
            self.redis.expire(minute_key, 60)
        if hour_count == 1:
            self.redis.expire(hour_key, 3600)
        
        # Check limits
        minute_allowed = minute_count <= self.rpm_limit
        hour_allowed = hour_count <= self.rph_limit
        
        allowed = minute_allowed and hour_allowed
        
        remaining = {
            "requests_per_minute": max(0, self.rpm_limit - minute_count),
            "requests_per_hour": max(0, self.rph_limit - hour_count),
        }
        
        return allowed, remaining
    
    def _check_memory_rate_limit(
        self,
        user_id: str,
        current_time: float
    ) -> tuple[bool, Dict[str, int]]:
        """Check rate limit using in-memory store"""
        # Clean old requests
        timestamps = self.memory_store[user_id]
        timestamps = [t for t in timestamps if current_time - t < 3600]
        
        # Count requests in windows
        minute_count = sum(1 for t in timestamps if current_time - t < 60)
        hour_count = len(timestamps)
        
        # Check limits
        allowed = (minute_count < self.rpm_limit and hour_count < self.rph_limit)
        
        if allowed:
            timestamps.append(current_time)
            self.memory_store[user_id] = timestamps
        
        remaining = {
            "requests_per_minute": max(0, self.rpm_limit - minute_count),
            "requests_per_hour": max(0, self.rph_limit - hour_count),
        }
        
        return allowed, remaining


# ============================================================================
# API Gateway Application
# ============================================================================

class APIGateway:
    """Main API Gateway application"""
    
    def __init__(self, config: GatewayConfig = GatewayConfig()):
        """
        Initialize API Gateway
        
        Args:
            config: Gateway configuration
        """
        self.config = config

        if not self.config.JWT_SECRET:
            raise RuntimeError(
                "JWT_SECRET must be set. Configure AUTH_USERS for static "
                "credentials, or set DEV_AUTH_ENABLED=true only for local "
                "development."
            )
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="LLM Infrastructure API Gateway",
            description="Authentication, rate limiting, and request routing",
            version="1.0.0"
        )
        add_metrics_endpoint(self.app)
        
        # Initialize auth manager
        self.auth_manager = AuthManager(
            secret=config.JWT_SECRET,
            algorithm=config.JWT_ALGORITHM
        )
        
        # Initialize rate limiter
        try:
            redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True
            )
            # Test connection
            redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory rate limiting: {e}")
            redis_client = None
        
        self.rate_limiter = RateLimiter(
            redis_client=redis_client,
            requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE,
            requests_per_hour=config.RATE_LIMIT_REQUESTS_PER_HOUR
        )
        
        # HTTP client for backend requests
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("API Gateway initialized")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token counts when the backend does not provide usage."""
        return max(1, len(text.split())) if text else 0

    def _estimate_chat_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate chat prompt tokens when backend usage is unavailable."""
        return sum(self._estimate_tokens(message.content) + 1 for message in messages)

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate token credentials against configured auth mode."""
        if not username or not password:
            return False

        if self.config.AUTH_USERS:
            return self.config.AUTH_USERS.get(username) == password

        return self.config.DEV_AUTH_ENABLED

    async def _proxy_streaming_response(
        self,
        backend_response: httpx.Response,
        model_name: str,
        endpoint: str,
        request_start: float,
        prompt_tokens: int,
    ):
        """Proxy backend SSE bytes while recording gateway metrics."""
        status = "success"
        error_type: Optional[str] = None

        try:
            async for chunk in backend_response.aiter_raw():
                yield chunk
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            await backend_response.aclose()
            finish_request_tracking(
                model_name=model_name,
                endpoint=endpoint,
                start_time=request_start,
                status=status,
                prompt_tokens=prompt_tokens,
                error_type=error_type,
            )
    
    def _setup_middleware(self):
        """Configure middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted hosts
        if self.config.ALLOWED_HOSTS != ["*"]:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.ALLOWED_HOSTS
            )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Log
            duration = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} "
                f"- {response.status_code} - {duration:.3f}s"
            )
            
            return response
    
    def _get_current_user(self, authorization: str = Header(None)) -> str:
        """
        Dependency to get current user from JWT token
        
        Args:
            authorization: Authorization header
            
        Returns:
            Username
        """
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization header")
        
        # Extract token
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = parts[1]
        
        # Verify token
        payload = self.auth_manager.verify_token(token)
        return payload["sub"]
    
    def _check_rate_limit(self, user_id: str) -> None:
        """
        Dependency to check rate limit
        
        Args:
            user_id: User identifier
            
        Raises:
            HTTPException if rate limit exceeded
        """
        if not self.config.RATE_LIMIT_ENABLED:
            return
        
        allowed, remaining = self.rate_limiter.check_rate_limit(user_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Remaining-Minute": str(remaining["requests_per_minute"]),
                    "X-RateLimit-Remaining-Hour": str(remaining["requests_per_hour"]),
                }
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.post("/auth/token", response_model=TokenResponse)
        async def get_token(request: TokenRequest):
            """
            Get API token
            
            Validates credentials against AUTH_USERS unless DEV_AUTH_ENABLED
            is explicitly set for local-only development.
            """
            if not self.config.AUTH_USERS and not self.config.DEV_AUTH_ENABLED:
                raise HTTPException(
                    status_code=503,
                    detail="Authentication is not configured",
                )

            if not self._validate_credentials(request.username, request.password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Create token
            token = self.auth_manager.create_token(
                username=request.username,
                expires_hours=self.config.JWT_EXPIRATION_HOURS
            )
            
            return TokenResponse(
                access_token=token,
                expires_in=self.config.JWT_EXPIRATION_HOURS * 3600
            )
        
        @self.app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(
            request: CompletionRequest,
            user: str = Depends(self._get_current_user)
        ):
            """
            Create text completion
            
            Requires authentication and respects rate limits
            """
            # Check rate limit
            self._check_rate_limit(user)

            # Forward request to vLLM backend
            payload = request.model_dump(exclude_none=True)
            if "model" not in payload and self.config.DEFAULT_COMPLETION_MODEL:
                payload["model"] = self.config.DEFAULT_COMPLETION_MODEL

            model_name = payload.get("model") or "gateway"
            prompt_tokens = self._estimate_tokens(request.prompt)
            request_start: Optional[float] = None

            try:
                if request.stream:
                    request_start = start_request_tracking(model_name)
                    backend_request = self.http_client.build_request(
                        "POST",
                        f"{self.config.VLLM_BACKEND_URL}/v1/completions",
                        json=payload,
                    )
                    backend_response = await self.http_client.send(
                        backend_request,
                        stream=True,
                    )

                    if backend_response.status_code != 200:
                        backend_error = (await backend_response.aread()).decode(
                            "utf-8",
                            errors="replace",
                        )
                        await backend_response.aclose()
                        finish_request_tracking(
                            model_name=model_name,
                            endpoint="/v1/completions",
                            start_time=request_start,
                            status="error",
                            prompt_tokens=prompt_tokens,
                            error_type="HTTPException",
                        )
                        raise HTTPException(
                            status_code=backend_response.status_code,
                            detail=f"Backend error: {backend_error}",
                        )

                    return StreamingResponse(
                        self._proxy_streaming_response(
                            backend_response=backend_response,
                            model_name=model_name,
                            endpoint="/v1/completions",
                            request_start=request_start,
                            prompt_tokens=prompt_tokens,
                        ),
                        media_type=backend_response.headers.get(
                            "content-type",
                            "text/event-stream",
                        ),
                    )

                request_start = start_request_tracking(model_name)
                response = await self.http_client.post(
                    f"{self.config.VLLM_BACKEND_URL}/v1/completions",
                    json=payload,
                    timeout=300.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Backend error: {response.text}"
                    )
                
                result = response.json()
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Transform response
                gateway_response = CompletionResponse(
                    id=result.get("id", "unknown"),
                    model=result.get("model", "unknown"),
                    text=result["choices"][0]["text"],
                    usage=UsageInfo(
                        prompt_tokens=usage.get("prompt_tokens", prompt_tokens),
                        completion_tokens=completion_tokens,
                        total_tokens=usage.get(
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        )
                    ),
                    created_at=datetime.utcnow().isoformat()
                )
                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/completions",
                    start_time=request_start,
                    prompt_tokens=gateway_response.usage.prompt_tokens,
                    completion_tokens=gateway_response.usage.completion_tokens,
                )
                return gateway_response

            except HTTPException as exc:
                if request_start is not None and not request.stream:
                    finish_request_tracking(
                        model_name=model_name,
                        endpoint="/v1/completions",
                        start_time=request_start,
                        status="error",
                        prompt_tokens=prompt_tokens,
                        error_type=type(exc).__name__,
                    )
                raise
                
            except httpx.RequestError as e:
                if request_start is not None:
                    finish_request_tracking(
                        model_name=model_name,
                        endpoint="/v1/completions",
                        start_time=request_start,
                        status="error",
                        prompt_tokens=prompt_tokens,
                        error_type=type(e).__name__,
                    )
                logger.error(f"Backend request failed: {e}")
                raise HTTPException(status_code=503, detail="Backend service unavailable")

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(
            request: ChatCompletionRequest,
            user: str = Depends(self._get_current_user)
        ):
            """
            Create chat completion.

            Proxies the backend OpenAI-compatible chat response without
            reshaping the payload.
            """
            self._check_rate_limit(user)

            payload = request.model_dump(exclude_none=True)
            if "model" not in payload and self.config.DEFAULT_COMPLETION_MODEL:
                payload["model"] = self.config.DEFAULT_COMPLETION_MODEL

            model_name = payload.get("model") or "gateway"
            prompt_tokens = self._estimate_chat_tokens(request.messages)
            endpoint = "/v1/chat/completions"
            request_start: Optional[float] = None

            try:
                if request.stream:
                    request_start = start_request_tracking(model_name)
                    backend_request = self.http_client.build_request(
                        "POST",
                        f"{self.config.VLLM_BACKEND_URL}{endpoint}",
                        json=payload,
                    )
                    backend_response = await self.http_client.send(
                        backend_request,
                        stream=True,
                    )

                    if backend_response.status_code != 200:
                        backend_error = (await backend_response.aread()).decode(
                            "utf-8",
                            errors="replace",
                        )
                        await backend_response.aclose()
                        finish_request_tracking(
                            model_name=model_name,
                            endpoint=endpoint,
                            start_time=request_start,
                            status="error",
                            prompt_tokens=prompt_tokens,
                            error_type="HTTPException",
                        )
                        raise HTTPException(
                            status_code=backend_response.status_code,
                            detail=f"Backend error: {backend_error}",
                        )

                    return StreamingResponse(
                        self._proxy_streaming_response(
                            backend_response=backend_response,
                            model_name=model_name,
                            endpoint=endpoint,
                            request_start=request_start,
                            prompt_tokens=prompt_tokens,
                        ),
                        media_type=backend_response.headers.get(
                            "content-type",
                            "text/event-stream",
                        ),
                    )

                request_start = start_request_tracking(model_name)
                response = await self.http_client.post(
                    f"{self.config.VLLM_BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=300.0,
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Backend error: {response.text}",
                    )

                result = response.json()
                usage = result.get("usage", {})
                finish_request_tracking(
                    model_name=model_name,
                    endpoint=endpoint,
                    start_time=request_start,
                    prompt_tokens=usage.get("prompt_tokens", prompt_tokens),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                return JSONResponse(content=result)

            except HTTPException as exc:
                if request_start is not None and not request.stream:
                    finish_request_tracking(
                        model_name=model_name,
                        endpoint=endpoint,
                        start_time=request_start,
                        status="error",
                        prompt_tokens=prompt_tokens,
                        error_type=type(exc).__name__,
                    )
                raise

            except httpx.RequestError as e:
                if request_start is not None:
                    finish_request_tracking(
                        model_name=model_name,
                        endpoint=endpoint,
                        start_time=request_start,
                        status="error",
                        prompt_tokens=prompt_tokens,
                        error_type=type(e).__name__,
                    )
                logger.error(f"Backend chat request failed: {e}")
                raise HTTPException(status_code=503, detail="Backend service unavailable")
        
        @self.app.get("/models")
        async def list_models(user: str = Depends(self._get_current_user)):
            """List available models"""
            return {
                "models": [
                    {"id": "llama2-7b", "type": "completion"},
                    {"id": "llama2-13b", "type": "completion"},
                ]
            }
        
        @self.app.get("/usage")
        async def get_usage(user: str = Depends(self._get_current_user)):
            """Get usage statistics for current user"""
            # TODO: Implement usage tracking
            return {
                "user": user,
                "total_requests": 0,
                "total_tokens": 0,
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the API gateway"""
        import uvicorn
        
        logger.info(f"Starting API Gateway on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    
    args = parser.parse_args()
    
    # Create and run gateway
    gateway = APIGateway()
    gateway.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
