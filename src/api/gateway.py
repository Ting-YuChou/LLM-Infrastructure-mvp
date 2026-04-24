"""
API Gateway for LLM Infrastructure
Provides authentication, rate limiting, and request routing
"""

import os
import time
import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import jwt
import redis
from collections import defaultdict
import httpx

try:
    from src.api.model_routing import ModelRouter, ResolvedRoute, UnknownModelRoute
    from src.monitoring.metrics import (
        add_metrics_endpoint,
        finish_request_tracking,
        record_inter_token_latency,
        record_time_to_first_token,
        start_request_tracking,
    )
except ModuleNotFoundError:
    from api.model_routing import (  # type: ignore[no-redef]
        ModelRouter,
        ResolvedRoute,
        UnknownModelRoute,
    )
    from monitoring.metrics import (  # type: ignore[no-redef]
        add_metrics_endpoint,
        finish_request_tracking,
        record_inter_token_latency,
        record_time_to_first_token,
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
    RATE_LIMIT_TOKENS_PER_MINUTE = _get_env_int("RATE_LIMIT_TOKENS_PER_MINUTE", 100000)
    RATE_LIMIT_TOKENS_PER_HOUR = _get_env_int("RATE_LIMIT_TOKENS_PER_HOUR", 1000000)
    
    # Redis for distributed rate limiting
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Backend services
    VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://localhost:8000")
    DEFAULT_COMPLETION_MODEL = os.getenv("DEFAULT_COMPLETION_MODEL")
    MODEL_ROUTING_CONFIG = os.getenv("MODEL_ROUTING_CONFIG")
    ROUTING_CANARY_SALT = os.getenv("ROUTING_CANARY_SALT", "default")
    
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
    
    Implements fixed-window RPM/RPH and TPM/TPH quota checks.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        tokens_per_minute: int = 100000,
        tokens_per_hour: int = 1000000,
    ):
        """
        Initialize rate limiter
        
        Args:
            redis_client: Redis client for distributed limiting
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            tokens_per_minute: Max estimated tokens per minute
            tokens_per_hour: Max estimated tokens per hour
        """
        self.redis = redis_client
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        self.tpm_limit = tokens_per_minute
        self.tph_limit = tokens_per_hour
        
        # Fallback to in-memory if Redis unavailable
        self.memory_store: Dict[str, List[tuple[float, int]]] = defaultdict(list)
    
    def _get_current_minute_key(self, user_id: str) -> str:
        """Get Redis key for current minute"""
        minute = int(time.time() / 60)
        return f"ratelimit:{user_id}:minute:{minute}"
    
    def _get_current_hour_key(self, user_id: str) -> str:
        """Get Redis key for current hour"""
        hour = int(time.time() / 3600)
        return f"ratelimit:{user_id}:hour:{hour}"
    
    def check_rate_limit(
        self,
        user_id: str,
        token_cost: int = 0,
    ) -> tuple[bool, Dict[str, int]]:
        """
        Check if user is within rate limits
        
        Args:
            user_id: User identifier
            token_cost: Estimated request token cost
            
        Returns:
            Tuple of (allowed: bool, remaining: dict)
        """
        current_time = time.time()
        
        if self.redis:
            # Use Redis for distributed rate limiting
            return self._check_redis_rate_limit(user_id, current_time, token_cost)
        else:
            # Use in-memory rate limiting
            return self._check_memory_rate_limit(user_id, current_time, token_cost)
    
    def _check_redis_rate_limit(
        self,
        user_id: str,
        current_time: float,
        token_cost: int,
    ) -> tuple[bool, Dict[str, int]]:
        """Check rate limit using Redis"""
        minute_key = self._get_current_minute_key(user_id)
        hour_key = self._get_current_hour_key(user_id)
        token_minute_key = f"{minute_key}:tokens"
        token_hour_key = f"{hour_key}:tokens"
        
        minute_count = self.redis.incr(minute_key)
        hour_count = self.redis.incr(hour_key)
        minute_tokens = self.redis.incrby(token_minute_key, token_cost)
        hour_tokens = self.redis.incrby(token_hour_key, token_cost)

        if minute_count == 1:
            self.redis.expire(minute_key, 60)
        if hour_count == 1:
            self.redis.expire(hour_key, 3600)
        if minute_tokens == token_cost:
            self.redis.expire(token_minute_key, 60)
        if hour_tokens == token_cost:
            self.redis.expire(token_hour_key, 3600)

        allowed = (
            self._under_limit(minute_count, self.rpm_limit)
            and self._under_limit(hour_count, self.rph_limit)
            and self._under_limit(minute_tokens, self.tpm_limit)
            and self._under_limit(hour_tokens, self.tph_limit)
        )
        
        remaining = {
            "requests_per_minute": max(0, self.rpm_limit - minute_count),
            "requests_per_hour": max(0, self.rph_limit - hour_count),
            "tokens_per_minute": max(0, self.tpm_limit - minute_tokens),
            "tokens_per_hour": max(0, self.tph_limit - hour_tokens),
        }
        
        return allowed, remaining
    
    def _check_memory_rate_limit(
        self,
        user_id: str,
        current_time: float,
        token_cost: int,
    ) -> tuple[bool, Dict[str, int]]:
        """Check rate limit using in-memory store"""
        entries = [
            entry for entry in self.memory_store[user_id]
            if current_time - entry[0] < 3600
        ]
        
        minute_entries = [entry for entry in entries if current_time - entry[0] < 60]
        minute_count = len(minute_entries)
        hour_count = len(entries)
        minute_tokens = sum(tokens for _, tokens in minute_entries)
        hour_tokens = sum(tokens for _, tokens in entries)

        next_minute_count = minute_count + 1
        next_hour_count = hour_count + 1
        next_minute_tokens = minute_tokens + token_cost
        next_hour_tokens = hour_tokens + token_cost

        allowed = (
            self._under_limit(next_minute_count, self.rpm_limit)
            and self._under_limit(next_hour_count, self.rph_limit)
            and self._under_limit(next_minute_tokens, self.tpm_limit)
            and self._under_limit(next_hour_tokens, self.tph_limit)
        )
        
        if allowed:
            entries.append((current_time, token_cost))
            self.memory_store[user_id] = entries
        
        remaining = {
            "requests_per_minute": max(0, self.rpm_limit - next_minute_count),
            "requests_per_hour": max(0, self.rph_limit - next_hour_count),
            "tokens_per_minute": max(0, self.tpm_limit - next_minute_tokens),
            "tokens_per_hour": max(0, self.tph_limit - next_hour_tokens),
        }
        
        return allowed, remaining

    def _under_limit(self, value: int, limit: int) -> bool:
        """Treat non-positive limits as disabled."""
        return limit <= 0 or value <= limit


class UsageTracker:
    """Track per-user serving usage in Redis with an in-memory fallback."""

    TOTAL_FIELDS = ("total_requests", "prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.memory_store: Dict[str, Dict[str, Any]] = defaultdict(self._empty_usage)

    def record_usage(
        self,
        user_id: str,
        model_name: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record one successful inference request."""
        total_tokens = prompt_tokens + completion_tokens

        if self.redis:
            self._record_usage_redis(
                user_id=user_id,
                model_name=model_name,
                endpoint=endpoint,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            return

        usage = self.memory_store[user_id]
        self._increment_usage_dict(usage, prompt_tokens, completion_tokens, total_tokens)
        self._increment_usage_dict(
            usage["models"].setdefault(model_name, self._empty_totals()),
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        self._increment_usage_dict(
            usage["endpoints"].setdefault(endpoint, self._empty_totals()),
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    def get_usage(self, user_id: str) -> Dict[str, Any]:
        """Return aggregate usage for one user."""
        if self.redis:
            return self._get_usage_redis(user_id)
        usage = self.memory_store[user_id]
        return {
            "user": user_id,
            **{field: usage[field] for field in self.TOTAL_FIELDS},
            "models": usage["models"],
            "endpoints": usage["endpoints"],
        }

    def _record_usage_redis(
        self,
        user_id: str,
        model_name: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        total_key = f"usage:{user_id}:total"
        model_key = f"usage:{user_id}:model:{model_name}"
        endpoint_key = f"usage:{user_id}:endpoint:{endpoint}"

        pipe = self.redis.pipeline()
        for key in (total_key, model_key, endpoint_key):
            pipe.hincrby(key, "total_requests", 1)
            pipe.hincrby(key, "prompt_tokens", prompt_tokens)
            pipe.hincrby(key, "completion_tokens", completion_tokens)
            pipe.hincrby(key, "total_tokens", total_tokens)
        pipe.sadd(f"usage:{user_id}:models", model_name)
        pipe.sadd(f"usage:{user_id}:endpoints", endpoint)
        pipe.execute()

    def _get_usage_redis(self, user_id: str) -> Dict[str, Any]:
        total = self._read_usage_hash(f"usage:{user_id}:total")
        model_names = sorted(self.redis.smembers(f"usage:{user_id}:models"))
        endpoint_names = sorted(self.redis.smembers(f"usage:{user_id}:endpoints"))

        return {
            "user": user_id,
            **total,
            "models": {
                model_name: self._read_usage_hash(f"usage:{user_id}:model:{model_name}")
                for model_name in model_names
            },
            "endpoints": {
                endpoint: self._read_usage_hash(f"usage:{user_id}:endpoint:{endpoint}")
                for endpoint in endpoint_names
            },
        }

    def _read_usage_hash(self, key: str) -> Dict[str, int]:
        values = self.redis.hgetall(key) if self.redis else {}
        return {
            field: int(values.get(field, 0))
            for field in self.TOTAL_FIELDS
        }

    def _empty_usage(self) -> Dict[str, Any]:
        return {
            **self._empty_totals(),
            "models": {},
            "endpoints": {},
        }

    def _empty_totals(self) -> Dict[str, int]:
        return {field: 0 for field in self.TOTAL_FIELDS}

    def _increment_usage_dict(
        self,
        usage: Dict[str, int],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        usage["total_requests"] += 1
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["total_tokens"] += total_tokens


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
            requests_per_hour=config.RATE_LIMIT_REQUESTS_PER_HOUR,
            tokens_per_minute=config.RATE_LIMIT_TOKENS_PER_MINUTE,
            tokens_per_hour=config.RATE_LIMIT_TOKENS_PER_HOUR,
        )
        self.usage_tracker = UsageTracker(redis_client=redis_client)
        self.model_router = self._build_model_router()
        
        # HTTP client for backend requests
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("API Gateway initialized")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token counts when the backend does not provide usage."""
        return max(1, len(text.split())) if text else 0

    def _build_model_router(self) -> ModelRouter:
        """Build the gateway model router from manifest or legacy backend config."""
        if self.config.MODEL_ROUTING_CONFIG:
            router = ModelRouter.from_file(
                self.config.MODEL_ROUTING_CONFIG,
                fallback_backend_url=self.config.VLLM_BACKEND_URL,
                canary_salt=self.config.ROUTING_CANARY_SALT,
            )
            logger.info("Loaded model routing manifest: %s", self.config.MODEL_ROUTING_CONFIG)
            return router

        return ModelRouter.from_static(
            backend_url=self.config.VLLM_BACKEND_URL,
            default_model=self.config.DEFAULT_COMPLETION_MODEL,
            canary_salt=self.config.ROUTING_CANARY_SALT,
        )

    def _estimate_chat_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate chat prompt tokens when backend usage is unavailable."""
        return sum(self._estimate_tokens(message.content) + 1 for message in messages)

    def _estimate_completion_tokens_from_result(self, result: Dict[str, Any]) -> int:
        """Estimate completion tokens from an OpenAI-compatible response."""
        texts = []
        for choice in result.get("choices") or []:
            if "text" in choice:
                texts.append(choice.get("text") or "")
                continue
            message = choice.get("message") or {}
            texts.append(message.get("content") or "")
        return self._estimate_tokens(" ".join(texts)) if texts else 0

    def _estimate_completion_request_tokens(
        self,
        prompt_tokens: int,
        max_tokens: int,
    ) -> int:
        """Estimate worst-case tokens before dispatching a request."""
        return prompt_tokens + max_tokens

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate token credentials against configured auth mode."""
        if not username or not password:
            return False

        if self.config.AUTH_USERS:
            return self.config.AUTH_USERS.get(username) == password

        return self.config.DEV_AUTH_ENABLED

    def _resolve_model_route(
        self,
        requested_model: Optional[str],
        user_id: str,
        endpoint: str,
    ) -> ResolvedRoute:
        """Resolve a requested logical model into a concrete backend target."""
        try:
            return self.model_router.resolve(
                requested_model=requested_model,
                user_id=user_id,
                endpoint=endpoint,
            )
        except UnknownModelRoute as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _proxy_streaming_response(
        self,
        backend_response: httpx.Response,
        model_name: str,
        endpoint: str,
        request_start: float,
        prompt_tokens: int,
        user_id: str,
    ):
        """Proxy backend SSE bytes while recording gateway metrics."""
        status = "success"
        error_type: Optional[str] = None
        buffer = ""
        completion_chunks: List[str] = []
        first_token_seen = False
        last_token_time: Optional[float] = None

        try:
            async for chunk in backend_response.aiter_raw():
                now = time.time()
                decoded_chunk = chunk.decode("utf-8", errors="ignore")
                buffer += decoded_chunk
                events = buffer.split("\n\n")
                buffer = events.pop()

                for event in events:
                    for text in self._extract_stream_texts(event):
                        if not text:
                            continue
                        if not first_token_seen:
                            record_time_to_first_token(model_name, now - request_start)
                            first_token_seen = True
                        elif last_token_time is not None:
                            record_inter_token_latency(model_name, now - last_token_time)
                        last_token_time = now
                        completion_chunks.append(text)
                yield chunk
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            await backend_response.aclose()
            completion_tokens = self._estimate_tokens(" ".join(completion_chunks))
            if not completion_chunks:
                completion_tokens = 0
            finish_request_tracking(
                model_name=model_name,
                endpoint=endpoint,
                start_time=request_start,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_type=error_type,
            )
            if status == "success":
                self.usage_tracker.record_usage(
                    user_id=user_id,
                    model_name=model_name,
                    endpoint=endpoint,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

    def _extract_stream_texts(self, sse_event: str) -> List[str]:
        """Extract completion text deltas from one SSE event."""
        texts: List[str] = []
        for line in sse_event.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = choice.get("text") or delta.get("content") or ""
                if text:
                    texts.append(text)
        return texts
    
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
    
    def _check_rate_limit(self, user_id: str, token_cost: int = 0) -> None:
        """
        Dependency to check rate limit
        
        Args:
            user_id: User identifier
            token_cost: Estimated token cost for TPM/TPH limits
            
        Raises:
            HTTPException if rate limit exceeded
        """
        if not self.config.RATE_LIMIT_ENABLED:
            return
        
        allowed, remaining = self.rate_limiter.check_rate_limit(user_id, token_cost)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Remaining-Minute": str(remaining["requests_per_minute"]),
                    "X-RateLimit-Remaining-Hour": str(remaining["requests_per_hour"]),
                    "X-RateLimit-Remaining-Tokens-Minute": str(remaining["tokens_per_minute"]),
                    "X-RateLimit-Remaining-Tokens-Hour": str(remaining["tokens_per_hour"]),
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
        
        @self.app.post("/v1/completions")
        async def create_completion(
            request: CompletionRequest,
            user: str = Depends(self._get_current_user)
        ):
            """
            Create text completion
            
            Requires authentication and respects rate limits
            """
            endpoint = "/v1/completions"
            payload = request.model_dump(exclude_none=True)
            route = self._resolve_model_route(request.model, user, endpoint)
            if route.target.model:
                payload["model"] = route.target.model
            else:
                payload.pop("model", None)

            model_name = route.target.model or "gateway"
            prompt_tokens = self._estimate_tokens(request.prompt)
            estimated_tokens = self._estimate_completion_request_tokens(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
            )
            self._check_rate_limit(user, estimated_tokens)
            request_start: Optional[float] = None

            try:
                if request.stream:
                    request_start = start_request_tracking(model_name)
                    backend_request = self.http_client.build_request(
                        "POST",
                        f"{route.target.backend_url}{endpoint}",
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
                            user_id=user,
                        ),
                        media_type=backend_response.headers.get(
                            "content-type",
                            "text/event-stream",
                        ),
                    )

                request_start = start_request_tracking(model_name)
                response = await self.http_client.post(
                    f"{route.target.backend_url}{endpoint}",
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
                actual_prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                actual_completion_tokens = usage.get(
                    "completion_tokens",
                    self._estimate_completion_tokens_from_result(result),
                )
                finish_request_tracking(
                    model_name=model_name,
                    endpoint=endpoint,
                    start_time=request_start,
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=actual_completion_tokens,
                )
                self.usage_tracker.record_usage(
                    user_id=user,
                    model_name=model_name,
                    endpoint=endpoint,
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=actual_completion_tokens,
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
            endpoint = "/v1/chat/completions"
            payload = request.model_dump(exclude_none=True)
            route = self._resolve_model_route(request.model, user, endpoint)
            if route.target.model:
                payload["model"] = route.target.model
            else:
                payload.pop("model", None)

            model_name = route.target.model or "gateway"
            prompt_tokens = self._estimate_chat_tokens(request.messages)
            estimated_tokens = self._estimate_completion_request_tokens(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
            )
            self._check_rate_limit(user, estimated_tokens)
            request_start: Optional[float] = None

            try:
                if request.stream:
                    request_start = start_request_tracking(model_name)
                    backend_request = self.http_client.build_request(
                        "POST",
                        f"{route.target.backend_url}{endpoint}",
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
                            user_id=user,
                        ),
                        media_type=backend_response.headers.get(
                            "content-type",
                            "text/event-stream",
                        ),
                    )

                request_start = start_request_tracking(model_name)
                response = await self.http_client.post(
                    f"{route.target.backend_url}{endpoint}",
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
                actual_prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                actual_completion_tokens = usage.get("completion_tokens", 0)
                finish_request_tracking(
                    model_name=model_name,
                    endpoint=endpoint,
                    start_time=request_start,
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=actual_completion_tokens,
                )
                self.usage_tracker.record_usage(
                    user_id=user,
                    model_name=model_name,
                    endpoint=endpoint,
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=actual_completion_tokens,
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
            return {"models": self.model_router.list_models()}
        
        @self.app.get("/usage")
        async def get_usage(user: str = Depends(self._get_current_user)):
            """Get usage statistics for current user"""
            return self.usage_tracker.get_usage(user)
    
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
