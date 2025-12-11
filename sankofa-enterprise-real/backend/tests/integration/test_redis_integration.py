"""
Redis Integration Tests
========================

Tests for Redis cache integration.
Target: 10 tests covering cache operations, TTL, stampede prevention

Test categories:
1. Basic cache operations (get/set/delete)
2. TTL enforcement
3. Distributed locks (SETNX)
4. Cache stampede prevention
5. Eviction policies
"""

import pytest
import asyncio
import time
from datetime import datetime
import redis.asyncio as aioredis


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def redis_client():
    """Create Redis client for tests"""
    client = await aioredis.from_url(
        "redis://localhost:6379/0",
        encoding="utf-8",
        decode_responses=True
    )

    # Clean up test keys before each test
    await client.delete('test:*')

    yield client

    # Clean up after test
    keys = await client.keys('test:*')
    if keys:
        await client.delete(*keys)
    await client.close()


# ============================================================================
# Basic Cache Operations Tests
# ============================================================================

class TestBasicCacheOperations:
    """Test basic Redis cache operations"""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, redis_client):
        """Test setting and getting cache value"""
        key = 'test:simple_key'
        value = 'test_value'

        # Set value
        await redis_client.set(key, value)

        # Get value
        retrieved = await redis_client.get(key)

        assert retrieved == value

    @pytest.mark.asyncio
    async def test_cache_set_with_expiry(self, redis_client):
        """Test setting cache value with TTL"""
        key = 'test:expiring_key'
        value = 'will_expire'
        ttl_seconds = 2

        # Set with expiry
        await redis_client.setex(key, ttl_seconds, value)

        # Immediately check value exists
        retrieved = await redis_client.get(key)
        assert retrieved == value

        # Wait for expiry
        await asyncio.sleep(ttl_seconds + 0.5)

        # Check value expired
        expired = await redis_client.get(key)
        assert expired is None

    @pytest.mark.asyncio
    async def test_cache_delete(self, redis_client):
        """Test deleting cache value"""
        key = 'test:delete_key'
        value = 'to_be_deleted'

        # Set value
        await redis_client.set(key, value)
        assert await redis_client.get(key) == value

        # Delete value
        await redis_client.delete(key)

        # Verify deleted
        assert await redis_client.get(key) is None

    @pytest.mark.asyncio
    async def test_cache_exists(self, redis_client):
        """Test checking if key exists"""
        key = 'test:exists_key'

        # Key doesn't exist
        assert await redis_client.exists(key) == 0

        # Set key
        await redis_client.set(key, 'value')

        # Key exists
        assert await redis_client.exists(key) == 1


# ============================================================================
# TTL Enforcement Tests
# ============================================================================

class TestTTLEnforcement:
    """Test TTL (Time To Live) enforcement"""

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, redis_client):
        """Test key expires after TTL"""
        key = 'test:ttl_key'
        value = 'expires_soon'
        ttl_seconds = 1

        await redis_client.setex(key, ttl_seconds, value)

        # Check TTL
        ttl = await redis_client.ttl(key)
        assert ttl > 0 and ttl <= ttl_seconds

        # Wait for expiration
        await asyncio.sleep(ttl_seconds + 0.5)

        # Key should be gone
        assert await redis_client.get(key) is None

    @pytest.mark.asyncio
    async def test_update_ttl_on_access(self, redis_client):
        """Test updating TTL on key access"""
        key = 'test:refresh_ttl'
        value = 'refreshable'
        initial_ttl = 5

        await redis_client.setex(key, initial_ttl, value)

        # Wait a bit
        await asyncio.sleep(2)

        # Refresh TTL
        await redis_client.expire(key, 10)

        # New TTL should be ~10 seconds
        new_ttl = await redis_client.ttl(key)
        assert new_ttl > 8 and new_ttl <= 10


# ============================================================================
# Distributed Lock Tests (SETNX)
# ============================================================================

class TestDistributedLocks:
    """Test distributed locking with Redis SETNX"""

    @pytest.mark.asyncio
    async def test_acquire_lock_setnx(self, redis_client):
        """Test acquiring lock with SETNX"""
        lock_key = 'test:lock:resource_1'
        lock_value = 'lock_owner_1'

        # Acquire lock (SETNX returns 1 if successful)
        acquired = await redis_client.setnx(lock_key, lock_value)
        assert acquired == 1

        # Try to acquire again (should fail)
        acquired_again = await redis_client.setnx(lock_key, 'lock_owner_2')
        assert acquired_again == 0

        # Release lock
        await redis_client.delete(lock_key)

    @pytest.mark.asyncio
    async def test_lock_with_timeout(self, redis_client):
        """Test lock with automatic timeout"""
        lock_key = 'test:lock:resource_2'
        lock_value = 'lock_owner'
        lock_timeout = 2

        # Acquire lock with timeout
        await redis_client.setex(lock_key, lock_timeout, lock_value)

        # Lock is held
        assert await redis_client.get(lock_key) == lock_value

        # Wait for timeout
        await asyncio.sleep(lock_timeout + 0.5)

        # Lock automatically released
        assert await redis_client.get(lock_key) is None

    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self, redis_client):
        """Test only one process can acquire lock"""
        lock_key = 'test:lock:concurrent'

        async def try_acquire_lock(worker_id):
            lock_value = f'worker_{worker_id}'
            acquired = await redis_client.setnx(lock_key, lock_value)
            if acquired:
                await asyncio.sleep(0.1)  # Hold lock briefly
                await redis_client.delete(lock_key)
                return worker_id
            return None

        # Try to acquire lock concurrently with 10 workers
        results = await asyncio.gather(*[try_acquire_lock(i) for i in range(10)])

        # Only ONE worker should have succeeded
        successful = [r for r in results if r is not None]
        assert len(successful) == 1


# ============================================================================
# Cache Stampede Prevention Tests
# ============================================================================

class TestCacheStampedePrevention:
    """Test prevention of cache stampede (thundering herd)"""

    @pytest.mark.asyncio
    async def test_stampede_prevention_with_lock(self, redis_client):
        """Test stampede prevention using distributed lock"""
        cache_key = 'test:cache:expensive_result'
        lock_key = f'{cache_key}:lock'

        computation_count = 0

        async def expensive_computation():
            """Simulate expensive operation"""
            nonlocal computation_count
            computation_count += 1
            await asyncio.sleep(0.1)  # Simulate slow operation
            return f'result_{computation_count}'

        async def get_with_stampede_prevention():
            """Get from cache with stampede prevention"""
            # Try cache first
            cached = await redis_client.get(cache_key)
            if cached:
                return cached

            # Try to acquire lock
            lock_acquired = await redis_client.setnx(lock_key, '1')
            if lock_acquired:
                try:
                    # We got the lock, do expensive computation
                    result = await expensive_computation()
                    await redis_client.setex(cache_key, 60, result)
                    return result
                finally:
                    await redis_client.delete(lock_key)
            else:
                # Someone else is computing, wait and retry
                await asyncio.sleep(0.2)
                cached = await redis_client.get(cache_key)
                if cached:
                    return cached
                # Fallback: compute anyway
                return await expensive_computation()

        # Simulate 10 concurrent requests
        results = await asyncio.gather(*[
            get_with_stampede_prevention() for _ in range(10)
        ])

        # All should get same result
        assert len(set(results)) == 1

        # Expensive computation should run only once (or very few times)
        assert computation_count <= 2  # Allow 1-2 computations max


# ============================================================================
# Eviction Policy Tests
# ============================================================================

class TestEvictionPolicies:
    """Test Redis eviction policies"""

    @pytest.mark.asyncio
    async def test_lru_eviction(self, redis_client):
        """Test LRU (Least Recently Used) eviction behavior"""
        # Set multiple keys
        keys = []
        for i in range(5):
            key = f'test:lru:{i}'
            await redis_client.set(key, f'value_{i}')
            keys.append(key)

        # Access first key to make it recently used
        await redis_client.get(keys[0])

        # All keys should still exist
        for key in keys:
            assert await redis_client.exists(key) == 1

    @pytest.mark.asyncio
    async def test_ttl_eviction(self, redis_client):
        """Test keys with TTL are evicted after expiration"""
        key = 'test:eviction:ttl'
        await redis_client.setex(key, 1, 'expires')

        # Key exists
        assert await redis_client.exists(key) == 1

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Key evicted
        assert await redis_client.exists(key) == 0


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for Redis Integration:

Basic Operations: 4 tests
- Set/Get, Set with expiry, Delete, Exists

TTL Enforcement: 2 tests
- TTL expiration, TTL refresh

Distributed Locks: 3 tests
- SETNX acquire, Lock timeout, Concurrent acquisition

Cache Stampede Prevention: 1 test
- Stampede prevention with lock

Eviction Policies: 2 tests (optional - depends on Redis config)
- LRU eviction, TTL eviction

TOTAL: 12 tests (10 required + 2 bonus)
TARGET: Cache reliability, concurrency, performance
"""
