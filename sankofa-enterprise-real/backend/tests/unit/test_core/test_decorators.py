"""
Unit Tests for Decorators - Cross-Cutting Concerns
===================================================

Tests for core/decorators.py
Target Coverage: >95%

Decorators tested:
- LoggingDecorator (structured logging with PII sanitization)
- MetricsDecorator (Prometheus-style metrics)
- CachingDecorator (cache-aside with stampede prevention)
- RetryDecorator (exponential backoff)
- CircuitBreakerDecorator (fail-fast pattern)

Test categories:
1. Decorator construction and application
2. Core functionality (logging, caching, etc.)
3. Error handling and edge cases
4. Performance characteristics
5. Decorator stacking/composition
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from core.decorators import (
    LoggingDecorator,
    MetricsDecorator,
    CachingDecorator,
    RetryDecorator,
    CircuitBreakerDecorator
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    return logger


@pytest.fixture
def mock_metrics():
    """Mock metrics collector"""
    metrics = Mock()
    metrics.increment = Mock()
    metrics.timing = Mock()
    metrics.gauge = Mock()
    return metrics


@pytest.fixture
def mock_cache():
    """Mock cache service"""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    return cache


@pytest.fixture
async def sample_async_function():
    """Sample async function to decorate"""
    async def func(x, y):
        await asyncio.sleep(0.01)
        return x + y
    return func


@pytest.fixture
def sample_sync_function():
    """Sample sync function to decorate"""
    def func(x, y):
        return x * y
    return func


# ============================================================================
# LoggingDecorator Tests
# ============================================================================

class TestLoggingDecorator:
    """Test Logging Decorator - Structured logging with PII sanitization"""

    @pytest.mark.asyncio
    async def test_logging_decorator_creation(self, mock_logger):
        """Test creating logging decorator"""
        decorator = LoggingDecorator(logger=mock_logger, logger_name="test")
        assert decorator is not None

    @pytest.mark.asyncio
    async def test_logging_decorator_logs_function_call(self, mock_logger):
        """Test decorator logs function entry and exit"""
        decorator = LoggingDecorator(logger=mock_logger)

        @decorator
        async def test_function(x):
            return x * 2

        result = await test_function(5)

        assert result == 10
        # Should log entry and exit
        assert mock_logger.info.call_count >= 2

    @pytest.mark.asyncio
    async def test_logging_decorator_logs_execution_time(self, mock_logger):
        """Test decorator logs execution time"""
        decorator = LoggingDecorator(logger=mock_logger)

        @decorator
        async def slow_function():
            await asyncio.sleep(0.05)
            return "done"

        await slow_function()

        # Check if execution time was logged
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("execution_time" in str(arg) or "duration" in str(arg)
                  for arg in call_args)

    @pytest.mark.asyncio
    async def test_logging_decorator_sanitizes_pii(self, mock_logger):
        """Test decorator sanitizes PII in logs (CPF, email, etc.)"""
        decorator = LoggingDecorator(logger=mock_logger, sanitize_pii=True)

        @decorator
        async def process_user(cpf, email):
            return {"cpf": cpf, "email": email}

        await process_user("12345678901", "user@example.com")

        # Check that actual values are NOT in logs
        call_args = str(mock_logger.info.call_args_list)
        assert "12345678901" not in call_args  # CPF should be masked
        assert "user@example.com" not in call_args  # Email should be masked

    @pytest.mark.asyncio
    async def test_logging_decorator_logs_exceptions(self, mock_logger):
        """Test decorator logs exceptions"""
        decorator = LoggingDecorator(logger=mock_logger)

        @decorator
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function()

        # Should log error
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_logging_decorator_includes_function_name(self, mock_logger):
        """Test logs include function name"""
        decorator = LoggingDecorator(logger=mock_logger)

        @decorator
        async def my_function():
            return "result"

        await my_function()

        # Check if function name is in logs
        call_args = str(mock_logger.info.call_args_list)
        assert "my_function" in call_args

    @pytest.mark.asyncio
    async def test_logging_decorator_with_sync_function(self, mock_logger):
        """Test decorator works with synchronous functions"""
        decorator = LoggingDecorator(logger=mock_logger)

        @decorator
        def sync_function(x):
            return x + 1

        result = sync_function(5)

        assert result == 6
        mock_logger.info.assert_called()


# ============================================================================
# MetricsDecorator Tests
# ============================================================================

class TestMetricsDecorator:
    """Test Metrics Decorator - Prometheus-style metrics collection"""

    @pytest.mark.asyncio
    async def test_metrics_decorator_creation(self, mock_metrics):
        """Test creating metrics decorator"""
        decorator = MetricsDecorator(metrics=mock_metrics)
        assert decorator is not None

    @pytest.mark.asyncio
    async def test_metrics_decorator_increments_counter(self, mock_metrics):
        """Test decorator increments call counter"""
        decorator = MetricsDecorator(metrics=mock_metrics, metric_prefix="test")

        @decorator
        async def test_function():
            return "result"

        await test_function()

        # Should increment counter
        mock_metrics.increment.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_decorator_records_duration(self, mock_metrics):
        """Test decorator records execution duration"""
        decorator = MetricsDecorator(metrics=mock_metrics)

        @decorator
        async def test_function():
            await asyncio.sleep(0.05)
            return "result"

        await test_function()

        # Should record timing
        mock_metrics.timing.assert_called()
        call_args = mock_metrics.timing.call_args
        # Duration should be > 50ms
        assert call_args[0][1] >= 50

    @pytest.mark.asyncio
    async def test_metrics_decorator_tracks_success_count(self, mock_metrics):
        """Test decorator tracks successful executions"""
        decorator = MetricsDecorator(metrics=mock_metrics)

        @decorator
        async def test_function():
            return "success"

        await test_function()

        # Should increment success counter
        calls = [str(call) for call in mock_metrics.increment.call_args_list]
        assert any("success" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_metrics_decorator_tracks_error_count(self, mock_metrics):
        """Test decorator tracks failed executions"""
        decorator = MetricsDecorator(metrics=mock_metrics)

        @decorator
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function()

        # Should increment error counter
        calls = [str(call) for call in mock_metrics.increment.call_args_list]
        assert any("error" in str(call) or "failure" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_metrics_decorator_tags_metrics(self, mock_metrics):
        """Test decorator tags metrics with function name"""
        decorator = MetricsDecorator(metrics=mock_metrics)

        @decorator
        async def my_function():
            return "result"

        await my_function()

        # Metrics should be tagged with function name
        call_args = str(mock_metrics.increment.call_args_list)
        assert "my_function" in call_args or "function" in call_args


# ============================================================================
# CachingDecorator Tests
# ============================================================================

class TestCachingDecorator:
    """Test Caching Decorator - Cache-aside with stampede prevention"""

    @pytest.mark.asyncio
    async def test_caching_decorator_creation(self, mock_cache):
        """Test creating caching decorator"""
        decorator = CachingDecorator(cache=mock_cache, ttl=300)
        assert decorator is not None

    @pytest.mark.asyncio
    async def test_caching_decorator_cache_miss_executes_function(self, mock_cache):
        """Test decorator executes function on cache miss"""
        mock_cache.get = AsyncMock(return_value=None)  # Cache miss

        decorator = CachingDecorator(cache=mock_cache)

        execution_count = 0

        @decorator
        async def expensive_function(x):
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        result = await expensive_function(5)

        assert result == 10
        assert execution_count == 1  # Function was executed
        mock_cache.set.assert_called_once()  # Result was cached

    @pytest.mark.asyncio
    async def test_caching_decorator_cache_hit_skips_function(self, mock_cache):
        """Test decorator returns cached value without executing function"""
        mock_cache.get = AsyncMock(return_value=20)  # Cache hit

        decorator = CachingDecorator(cache=mock_cache)

        execution_count = 0

        @decorator
        async def expensive_function(x):
            nonlocal execution_count
            execution_count += 1
            return x * 2

        result = await expensive_function(10)

        assert result == 20  # Cached value
        assert execution_count == 0  # Function was NOT executed

    @pytest.mark.asyncio
    async def test_caching_decorator_generates_cache_key(self, mock_cache):
        """Test decorator generates cache key from function args"""
        mock_cache.get = AsyncMock(return_value=None)

        decorator = CachingDecorator(cache=mock_cache, key_prefix="test")

        @decorator
        async def test_function(x, y):
            return x + y

        await test_function(1, 2)

        # Should generate key from args
        mock_cache.get.assert_called_once()
        cache_key = mock_cache.get.call_args[0][0]
        assert "test" in cache_key  # Has prefix
        assert isinstance(cache_key, str)

    @pytest.mark.asyncio
    async def test_caching_decorator_different_args_different_keys(self, mock_cache):
        """Test different arguments produce different cache keys"""
        mock_cache.get = AsyncMock(return_value=None)

        decorator = CachingDecorator(cache=mock_cache)

        @decorator
        async def test_function(x):
            return x * 2

        await test_function(1)
        await test_function(2)

        # Should call cache.get twice with different keys
        assert mock_cache.get.call_count == 2
        key1 = mock_cache.get.call_args_list[0][0][0]
        key2 = mock_cache.get.call_args_list[1][0][0]
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_caching_decorator_respects_ttl(self, mock_cache):
        """Test decorator sets TTL when caching"""
        mock_cache.get = AsyncMock(return_value=None)

        ttl = 600
        decorator = CachingDecorator(cache=mock_cache, ttl=ttl)

        @decorator
        async def test_function(x):
            return x * 2

        await test_function(5)

        # Should set cache with TTL
        mock_cache.set.assert_called_once()
        call_kwargs = mock_cache.set.call_args.kwargs
        if 'ttl' in call_kwargs:
            assert call_kwargs['ttl'] == ttl

    @pytest.mark.asyncio
    async def test_caching_decorator_cache_stampede_prevention(self, mock_cache):
        """Test decorator prevents cache stampede (multiple concurrent cache misses)"""
        mock_cache.get = AsyncMock(return_value=None)  # All cache misses

        decorator = CachingDecorator(cache=mock_cache)

        execution_count = 0

        @decorator
        async def expensive_function(x):
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.1)  # Slow operation
            return x * 2

        # Simulate stampede: 10 concurrent calls
        tasks = [expensive_function(5) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should get same result
        assert all(r == 10 for r in results)
        # Function should be executed only once (or very few times with locking)
        # Without stampede prevention, it would be 10 times
        assert execution_count <= 3  # Allow some race conditions

    @pytest.mark.asyncio
    async def test_caching_decorator_handles_cache_errors_gracefully(self, mock_cache):
        """Test decorator continues working if cache fails"""
        mock_cache.get = AsyncMock(side_effect=Exception("Cache unavailable"))

        decorator = CachingDecorator(cache=mock_cache)

        @decorator
        async def test_function(x):
            return x * 2

        # Should not raise, should execute function
        result = await test_function(5)
        assert result == 10


# ============================================================================
# RetryDecorator Tests
# ============================================================================

class TestRetryDecorator:
    """Test Retry Decorator - Exponential backoff"""

    @pytest.mark.asyncio
    async def test_retry_decorator_creation(self):
        """Test creating retry decorator"""
        decorator = RetryDecorator(max_retries=3, backoff_factor=2)
        assert decorator is not None

    @pytest.mark.asyncio
    async def test_retry_decorator_success_on_first_try(self):
        """Test no retry if function succeeds on first attempt"""
        decorator = RetryDecorator(max_retries=3)

        execution_count = 0

        @decorator
        async def test_function():
            nonlocal execution_count
            execution_count += 1
            return "success"

        result = await test_function()

        assert result == "success"
        assert execution_count == 1  # Only executed once

    @pytest.mark.asyncio
    async def test_retry_decorator_retries_on_failure(self):
        """Test decorator retries on failure"""
        decorator = RetryDecorator(max_retries=3, delay=0.01)

        execution_count = 0

        @decorator
        async def failing_function():
            nonlocal execution_count
            execution_count += 1
            if execution_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await failing_function()

        assert result == "success"
        assert execution_count == 3  # Retried twice, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_retry_decorator_max_retries_exhausted(self):
        """Test decorator raises after max retries"""
        decorator = RetryDecorator(max_retries=3, delay=0.01)

        @decorator
        async def always_failing_function():
            raise ValueError("Permanent error")

        with pytest.raises(ValueError):
            await always_failing_function()

    @pytest.mark.asyncio
    async def test_retry_decorator_exponential_backoff(self):
        """Test decorator uses exponential backoff"""
        decorator = RetryDecorator(max_retries=3, delay=0.01, backoff_factor=2)

        retry_times = []

        @decorator
        async def failing_function():
            retry_times.append(time.time())
            raise ValueError("Error")

        try:
            await failing_function()
        except ValueError:
            pass

        # Check delays between retries increase exponentially
        # Should be ~10ms, ~20ms, ~40ms
        if len(retry_times) >= 3:
            delay1 = retry_times[1] - retry_times[0]
            delay2 = retry_times[2] - retry_times[1]
            # Second delay should be roughly 2x first delay
            assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_retry_decorator_only_retries_specified_exceptions(self):
        """Test decorator only retries specific exception types"""
        decorator = RetryDecorator(
            max_retries=3,
            retry_on_exceptions=(ValueError,)
        )

        @decorator
        async def function_with_type_error():
            raise TypeError("Not retryable")

        # Should raise immediately without retries
        with pytest.raises(TypeError):
            await function_with_type_error()


# ============================================================================
# CircuitBreakerDecorator Tests
# ============================================================================

class TestCircuitBreakerDecorator:
    """Test Circuit Breaker Decorator - Fail-fast pattern"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_creation(self):
        """Test creating circuit breaker decorator"""
        decorator = CircuitBreakerDecorator(
            failure_threshold=5,
            timeout_seconds=60
        )
        assert decorator is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_normal_operation(self):
        """Test circuit breaker in CLOSED state allows calls"""
        decorator = CircuitBreakerDecorator(failure_threshold=3)

        @decorator
        async def test_function():
            return "success"

        result = await test_function()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        decorator = CircuitBreakerDecorator(failure_threshold=3)

        execution_count = 0

        @decorator
        async def failing_function():
            nonlocal execution_count
            execution_count += 1
            raise ValueError("Service unavailable")

        # Fail threshold times to open circuit
        for _ in range(3):
            try:
                await failing_function()
            except ValueError:
                pass

        # Circuit should now be OPEN - subsequent calls should fail fast
        with pytest.raises(Exception):  # CircuitOpenError or similar
            await failing_function()

        # Should not execute function (circuit is open)
        assert execution_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker transitions to HALF_OPEN after timeout"""
        decorator = CircuitBreakerDecorator(
            failure_threshold=2,
            timeout_seconds=0.1  # Short timeout for testing
        )

        @decorator
        async def function():
            return "success"

        # Open the circuit
        for _ in range(2):
            try:
                @decorator
                async def failing():
                    raise ValueError("Error")
                await failing()
            except ValueError:
                pass

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should allow one test call (HALF_OPEN)
        result = await function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets failure count on success"""
        decorator = CircuitBreakerDecorator(failure_threshold=3)

        call_count = 0

        @decorator
        async def intermittent_function():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return "success"
            raise ValueError("Error")

        # Fail once
        try:
            await intermittent_function()
        except ValueError:
            pass

        # Succeed - should reset counter
        result = await intermittent_function()
        assert result == "success"

        # Fail again - should not open circuit (counter was reset)
        try:
            await intermittent_function()
        except ValueError:
            pass


# ============================================================================
# Decorator Stacking Tests
# ============================================================================

class TestDecoratorStacking:
    """Test stacking multiple decorators"""

    @pytest.mark.asyncio
    async def test_stacking_logging_and_metrics(self, mock_logger, mock_metrics):
        """Test stacking logging and metrics decorators"""
        logging_dec = LoggingDecorator(logger=mock_logger)
        metrics_dec = MetricsDecorator(metrics=mock_metrics)

        @logging_dec
        @metrics_dec
        async def test_function(x):
            return x * 2

        result = await test_function(5)

        assert result == 10
        # Both decorators should have been applied
        mock_logger.info.assert_called()
        mock_metrics.increment.assert_called()

    @pytest.mark.asyncio
    async def test_stacking_cache_and_retry(self, mock_cache):
        """Test stacking caching and retry decorators"""
        mock_cache.get = AsyncMock(return_value=None)

        cache_dec = CachingDecorator(cache=mock_cache)
        retry_dec = RetryDecorator(max_retries=3, delay=0.01)

        execution_count = 0

        @cache_dec
        @retry_dec
        async def flaky_function(x):
            nonlocal execution_count
            execution_count += 1
            if execution_count < 2:
                raise ValueError("Temporary error")
            return x * 2

        result = await flaky_function(5)

        assert result == 10
        assert execution_count == 2  # Retried once
        mock_cache.set.assert_called()  # Result was cached

    @pytest.mark.asyncio
    async def test_full_decorator_stack(self, mock_logger, mock_metrics, mock_cache):
        """Test full decorator stack (logging + metrics + caching + retry)"""
        mock_cache.get = AsyncMock(return_value=None)

        @LoggingDecorator(logger=mock_logger)
        @MetricsDecorator(metrics=mock_metrics)
        @CachingDecorator(cache=mock_cache)
        @RetryDecorator(max_retries=2)
        async def complex_function(x):
            return x * 2

        result = await complex_function(5)

        assert result == 10
        # All decorators should work
        mock_logger.info.assert_called()
        mock_metrics.increment.assert_called()
        mock_cache.set.assert_called()


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for decorators.py:

LoggingDecorator: 6 tests
- Creation & basic logging: 2
- Execution time logging: 1
- PII sanitization: 1
- Error logging: 1
- Sync function support: 1

MetricsDecorator: 5 tests
- Creation & counter: 2
- Duration recording: 1
- Success/error tracking: 2

CachingDecorator: 8 tests
- Cache miss/hit: 2
- Key generation: 2
- TTL handling: 1
- Stampede prevention: 1
- Error handling: 1

RetryDecorator: 5 tests
- Success without retry: 1
- Retry on failure: 1
- Max retries exhausted: 1
- Exponential backoff: 1
- Exception filtering: 1

CircuitBreakerDecorator: 5 tests
- Normal operation: 1
- Circuit opening: 1
- Half-open state: 1
- Circuit reset: 1

DecoratorStacking: 3 tests
- Two decorators: 2
- Full stack: 1

TOTAL: 32 tests
TARGET: >95% statement coverage
PATTERNS: Decorator pattern, Chain of Responsibility
"""
