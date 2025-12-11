"""
Decorator Pattern for Cross-Cutting Concerns
Wraps use cases and services with logging, metrics, caching, etc.

This demonstrates the Decorator Pattern - one of the most useful GoF patterns
for adding behavior without modifying existing code.

Benefits:
- Separation of concerns: Business logic separated from logging/metrics
- Composable: Stack multiple decorators
- Open/Closed Principle: Add new decorators without changing existing code
- Testable: Test business logic without decorators

Time Complexity: O(1) overhead per decorator
"""

import time
import asyncio
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime
import hashlib

from utils.structured_logging import get_structured_logger
from utils.log_sanitizer import sanitize_log_data


logger = get_structured_logger("decorators")


class LoggingDecorator:
    """
    Decorator: Adds structured logging to any async function

    Logs:
    - Function entry (with sanitized params)
    - Function exit (with result summary)
    - Errors (with context)
    - Execution time

    Time Complexity: O(1) overhead
    """

    def __init__(self, logger_name: Optional[str] = None):
        self._logger = get_structured_logger(logger_name or "decorated_function")

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate request ID for tracing
            request_id = hashlib.sha256(
                f"{func.__name__}:{time.time()}".encode()
            ).hexdigest()[:16]

            # Log entry
            self._logger.info(
                f"[{request_id}] Entering {func.__name__}",
                extra=sanitize_log_data({
                    'function': func.__name__,
                    'request_id': request_id,
                    'kwargs': kwargs  # args are positional, kwargs are named
                })
            )

            start_time = time.time()
            error = None

            try:
                result = await func(*args, **kwargs)

                # Log successful exit
                duration_ms = (time.time() - start_time) * 1000
                self._logger.info(
                    f"[{request_id}] Exiting {func.__name__} (success)",
                    extra={
                        'function': func.__name__,
                        'request_id': request_id,
                        'duration_ms': duration_ms,
                        'status': 'success'
                    }
                )

                return result

            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000
                error = e

                self._logger.error(
                    f"[{request_id}] Exiting {func.__name__} (error)",
                    extra=sanitize_log_data({
                        'function': func.__name__,
                        'request_id': request_id,
                        'duration_ms': duration_ms,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    })
                )

                raise

        return wrapper


class MetricsDecorator:
    """
    Decorator: Adds metrics collection to any async function

    Collects:
    - Execution time (histogram)
    - Success/failure count (counter)
    - Active executions (gauge)

    Time Complexity: O(1) overhead
    """

    def __init__(self, metrics_collector, metric_prefix: Optional[str] = None):
        """
        Args:
            metrics_collector: Metrics collector instance (Dependency Injection)
            metric_prefix: Prefix for metric names (default: function name)
        """
        self._metrics = metrics_collector
        self._prefix = metric_prefix

    def __call__(self, func: Callable) -> Callable:
        metric_name = self._prefix or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Increment active executions
            self._metrics.record_gauge(
                f"{metric_name}_active",
                1.0,
                tags={'function': func.__name__}
            )

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Record success
                duration_ms = (time.time() - start_time) * 1000
                self._metrics.increment_counter(
                    f"{metric_name}_success",
                    tags={'function': func.__name__}
                )
                self._metrics.record_histogram(
                    f"{metric_name}_duration_ms",
                    duration_ms,
                    tags={'function': func.__name__, 'status': 'success'}
                )

                return result

            except Exception as e:
                # Record failure
                duration_ms = (time.time() - start_time) * 1000
                self._metrics.increment_counter(
                    f"{metric_name}_failure",
                    tags={'function': func.__name__, 'error_type': type(e).__name__}
                )
                self._metrics.record_histogram(
                    f"{metric_name}_duration_ms",
                    duration_ms,
                    tags={'function': func.__name__, 'status': 'error'}
                )

                raise

            finally:
                # Decrement active executions
                self._metrics.record_gauge(
                    f"{metric_name}_active",
                    -1.0,
                    tags={'function': func.__name__}
                )

        return wrapper


class CachingDecorator:
    """
    Decorator: Adds caching to any async function

    Uses cache-aside pattern:
    1. Check cache
    2. If miss, execute function
    3. Store result in cache
    4. Return result

    Time Complexity:
    - Cache hit: O(1)
    - Cache miss: O(f) where f is function complexity
    """

    def __init__(self, cache_service, ttl: int = 3600, key_prefix: Optional[str] = None):
        """
        Args:
            cache_service: Cache service instance (Dependency Injection)
            ttl: Time to live in seconds (default: 1 hour)
            key_prefix: Prefix for cache keys (default: function name)
        """
        self._cache = cache_service
        self._ttl = ttl
        self._prefix = key_prefix

    def __call__(self, func: Callable) -> Callable:
        cache_prefix = self._prefix or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from arguments
            cache_key = self._generate_cache_key(cache_prefix, args, kwargs)

            # Try cache first - O(1)
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(
                    f"Cache HIT for {func.__name__}",
                    extra={'cache_key': cache_key}
                )
                return cached_result

            # Cache miss - execute function
            logger.debug(
                f"Cache MISS for {func.__name__}",
                extra={'cache_key': cache_key}
            )
            result = await func(*args, **kwargs)

            # Store in cache - O(1)
            try:
                await self._cache.set(cache_key, result, ttl=self._ttl)
            except Exception as e:
                # Cache failure shouldn't fail the operation
                logger.warning(
                    f"Failed to cache result for {func.__name__}: {e}",
                    extra={'cache_key': cache_key}
                )

            return result

        return wrapper

    @staticmethod
    def _generate_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key from function arguments

        Time Complexity: O(n) where n is total arg length
        """
        # Convert args and kwargs to a deterministic string
        key_parts = [prefix]

        # Add positional args
        for arg in args:
            if hasattr(arg, '__dict__'):
                # For objects, use their dict representation
                key_parts.append(str(sorted(arg.__dict__.items())))
            else:
                key_parts.append(str(arg))

        # Add keyword args (sorted for determinism)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # Hash the key to keep it short
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]


class RetryDecorator:
    """
    Decorator: Adds retry logic with exponential backoff

    Useful for:
    - Transient network errors
    - Database deadlocks
    - Rate limiting

    Time Complexity: O(f * r) where:
    - f = function complexity
    - r = max retries
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple = (Exception,)
    ):
        """
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            retryable_exceptions: Tuple of exception types to retry
        """
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._exponential_base = exponential_base
        self._retryable_exceptions = retryable_exceptions

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(self._max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except self._retryable_exceptions as e:
                    last_exception = e

                    if attempt == self._max_retries:
                        # Last attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after {self._max_retries + 1} attempts",
                            extra={
                                'function': func.__name__,
                                'attempts': attempt + 1,
                                'error': str(e)
                            }
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        self._initial_delay * (self._exponential_base ** attempt),
                        self._max_delay
                    )

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {delay}s",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'delay': delay,
                            'error': str(e)
                        }
                    )

                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper


class CircuitBreakerDecorator:
    """
    Decorator: Implements Circuit Breaker pattern

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered

    Time Complexity: O(1) overhead
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting recovery (HALF_OPEN)
            expected_exception: Exception type that counts as failure
        """
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._expected_exception = expected_exception

        # State
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is OPEN
            if self._state == "OPEN":
                # Check if timeout passed (transition to HALF_OPEN)
                if self._last_failure_time and \
                   (time.time() - self._last_failure_time) > self._timeout:
                    self._state = "HALF_OPEN"
                    logger.info(
                        f"Circuit breaker for {func.__name__} transitioned to HALF_OPEN",
                        extra={'function': func.__name__}
                    )
                else:
                    # Still OPEN, reject request immediately
                    raise Exception(
                        f"Circuit breaker OPEN for {func.__name__}. "
                        f"Try again in {self._timeout - (time.time() - self._last_failure_time):.1f}s"
                    )

            try:
                result = await func(*args, **kwargs)

                # Success - reset if in HALF_OPEN or decrease failure count
                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker for {func.__name__} transitioned to CLOSED (recovered)",
                        extra={'function': func.__name__}
                    )
                elif self._failure_count > 0:
                    self._failure_count -= 1

                return result

            except self._expected_exception as e:
                self._failure_count += 1
                self._last_failure_time = time.time()

                if self._failure_count >= self._failure_threshold:
                    self._state = "OPEN"
                    logger.error(
                        f"Circuit breaker for {func.__name__} transitioned to OPEN",
                        extra={
                            'function': func.__name__,
                            'failure_count': self._failure_count,
                            'threshold': self._failure_threshold
                        }
                    )

                raise

        return wrapper


# Convenience function to stack multiple decorators
def apply_standard_decorators(
    func: Callable,
    logger_name: str,
    metrics_collector,
    cache_service = None,
    cache_ttl: int = 3600
) -> Callable:
    """
    Apply standard decorators to a function

    Order matters:
    1. Logging (outermost - logs everything)
    2. Metrics (second - metrics include cache time)
    3. Caching (innermost - avoids executing function if cached)

    Usage:
        @apply_standard_decorators(
            logger_name='my_service',
            metrics_collector=metrics,
            cache_service=cache,
            cache_ttl=1800
        )
        async def my_function():
            ...

    Time Complexity: O(1) to apply decorators
    """
    decorated = func

    # Apply in reverse order (innermost to outermost)
    if cache_service:
        decorated = CachingDecorator(cache_service, ttl=cache_ttl)(decorated)

    decorated = MetricsDecorator(metrics_collector)(decorated)
    decorated = LoggingDecorator(logger_name)(decorated)

    return decorated
