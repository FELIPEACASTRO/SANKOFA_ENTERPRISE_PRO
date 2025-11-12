"""
Infrastructure Layer - Repository Implementations
Implements Repository Pattern with concrete database implementations
Follows Dependency Inversion Principle (SOLID)
"""

import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
import asyncpg
import redis.asyncio as redis
from dataclasses import asdict

from ..core.entities import (
    Transaction,
    TransactionId,
    Customer,
    Money,
    TransactionStatus,
    RiskLevel,
    DomainEvent,
)
from ..core.interfaces import TransactionRepository, CustomerRepository, EventStore


class PostgreSQLTransactionRepository(TransactionRepository):
    """
    PostgreSQL implementation of TransactionRepository
    Time Complexity: O(log n) for most operations due to B-tree indexes
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool

    async def save(self, transaction: Transaction) -> None:
        """
        Save transaction to PostgreSQL
        Time Complexity: O(log n) - B-tree index insertion
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO transactions (
                    id, amount, currency, merchant_id, customer_id, 
                    status, risk_score, risk_level, timestamp, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    risk_score = EXCLUDED.risk_score,
                    risk_level = EXCLUDED.risk_level,
                    metadata = EXCLUDED.metadata
            """,
                transaction.id.value,
                float(transaction.amount.amount),
                transaction.amount.currency,
                transaction.merchant_id,
                transaction.customer_id,
                transaction.status.value,
                transaction.risk_score,
                transaction.risk_level.value,
                transaction.timestamp,
                json.dumps(transaction.metadata),
            )

    async def find_by_id(self, transaction_id: TransactionId) -> Optional[Transaction]:
        """
        Find transaction by ID using B-tree index
        Time Complexity: O(log n)
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM transactions WHERE id = $1", transaction_id.value
            )

            if not row:
                return None

            return self._row_to_transaction(row)

    async def find_by_customer(self, customer_id: str, limit: int = 100) -> List[Transaction]:
        """
        Find transactions by customer using index
        Time Complexity: O(log n + k) where k is result size
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM transactions 
                WHERE customer_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """,
                customer_id,
                limit,
            )

            return [self._row_to_transaction(row) for row in rows]

    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[Transaction]:
        """
        Find transactions by date range using timestamp index
        Time Complexity: O(log n + k)
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM transactions 
                WHERE timestamp BETWEEN $1 AND $2 
                ORDER BY timestamp DESC 
                LIMIT $3
            """,
                start_date,
                end_date,
                limit,
            )

            return [self._row_to_transaction(row) for row in rows]

    async def count_by_customer(self, customer_id: str) -> int:
        """
        Count transactions by customer
        Time Complexity: O(log n) with proper indexing
        """
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM transactions WHERE customer_id = $1", customer_id
            )
            return count or 0

    def _row_to_transaction(self, row) -> Transaction:
        """Convert database row to Transaction entity - O(1)"""
        return Transaction(
            id=TransactionId(row["id"]),
            amount=Money(Decimal(str(row["amount"])), row["currency"]),
            merchant_id=row["merchant_id"],
            customer_id=row["customer_id"],
            timestamp=row["timestamp"],
            status=TransactionStatus(row["status"]),
            risk_score=row["risk_score"],
            risk_level=RiskLevel(row["risk_level"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class RedisTransactionRepository(TransactionRepository):
    """
    Redis implementation for high-performance caching
    Time Complexity: O(1) for most operations (hash table)
    """

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client
        self._ttl = 86400  # 24 hours

    async def save(self, transaction: Transaction) -> None:
        """
        Save transaction to Redis
        Time Complexity: O(1) average
        """
        key = f"transaction:{transaction.id.value}"
        data = {
            "id": transaction.id.value,
            "amount": float(transaction.amount.amount),
            "currency": transaction.amount.currency,
            "merchant_id": transaction.merchant_id,
            "customer_id": transaction.customer_id,
            "status": transaction.status.value,
            "risk_score": transaction.risk_score,
            "risk_level": transaction.risk_level.value,
            "timestamp": transaction.timestamp.isoformat(),
            "metadata": json.dumps(transaction.metadata),
        }

        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, self._ttl)

        # Add to customer index
        customer_key = f"customer_transactions:{transaction.customer_id}"
        await self._redis.zadd(
            customer_key, {transaction.id.value: transaction.timestamp.timestamp()}
        )
        await self._redis.expire(customer_key, self._ttl)

    async def find_by_id(self, transaction_id: TransactionId) -> Optional[Transaction]:
        """
        Find transaction by ID in Redis
        Time Complexity: O(1) average
        """
        key = f"transaction:{transaction_id.value}"
        data = await self._redis.hgetall(key)

        if not data:
            return None

        return Transaction(
            id=TransactionId(data[b"id"].decode()),
            amount=Money(Decimal(data[b"amount"].decode()), data[b"currency"].decode()),
            merchant_id=data[b"merchant_id"].decode(),
            customer_id=data[b"customer_id"].decode(),
            timestamp=datetime.fromisoformat(data[b"timestamp"].decode()),
            status=TransactionStatus(data[b"status"].decode()),
            risk_score=float(data[b"risk_score"].decode()),
            risk_level=RiskLevel(data[b"risk_level"].decode()),
            metadata=json.loads(data[b"metadata"].decode()) if data[b"metadata"] else {},
        )

    async def find_by_customer(self, customer_id: str, limit: int = 100) -> List[Transaction]:
        """
        Find transactions by customer using sorted set
        Time Complexity: O(log n + k)
        """
        customer_key = f"customer_transactions:{customer_id}"
        transaction_ids = await self._redis.zrevrange(customer_key, 0, limit - 1)

        transactions = []
        for tx_id in transaction_ids:
            transaction = await self.find_by_id(TransactionId(tx_id.decode()))
            if transaction:
                transactions.append(transaction)

        return transactions

    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[Transaction]:
        """
        Find by date range - Redis doesn't have efficient range queries
        This is a simplified implementation
        Time Complexity: O(n) - not optimal for Redis
        """
        # In a real implementation, you'd use a different strategy
        # like time-based keys or external indexing
        return []

    async def count_by_customer(self, customer_id: str) -> int:
        """
        Count transactions by customer
        Time Complexity: O(1)
        """
        customer_key = f"customer_transactions:{customer_id}"
        return await self._redis.zcard(customer_key)


class PostgreSQLCustomerRepository(CustomerRepository):
    """
    PostgreSQL implementation of CustomerRepository
    Time Complexity: O(log n) for indexed operations
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool

    async def save(self, customer: Customer) -> None:
        """
        Save customer to PostgreSQL
        Time Complexity: O(log n)
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO customers (
                    id, email, created_at, risk_profile, 
                    transaction_count, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    risk_profile = EXCLUDED.risk_profile,
                    transaction_count = EXCLUDED.transaction_count,
                    metadata = EXCLUDED.metadata
            """,
                customer.id,
                customer.email,
                customer.created_at,
                customer.risk_profile.value,
                customer.get_transaction_count(),
                json.dumps(customer.metadata),
            )

    async def find_by_id(self, customer_id: str) -> Optional[Customer]:
        """
        Find customer by ID
        Time Complexity: O(log n)
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM customers WHERE id = $1", customer_id)

            if not row:
                return None

            return Customer(
                id=row["id"],
                email=row["email"],
                created_at=row["created_at"],
                risk_profile=RiskLevel(row["risk_profile"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )

    async def find_by_email(self, email: str) -> Optional[Customer]:
        """
        Find customer by email using index
        Time Complexity: O(log n)
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM customers WHERE email = $1", email)

            if not row:
                return None

            return Customer(
                id=row["id"],
                email=row["email"],
                created_at=row["created_at"],
                risk_profile=RiskLevel(row["risk_profile"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )


class PostgreSQLEventStore(EventStore):
    """
    PostgreSQL implementation of Event Store
    Time Complexity: O(log n) for append, O(k) for reading events
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool

    async def save_events(self, aggregate_id: str, events: List[DomainEvent]) -> None:
        """
        Save events to event store
        Time Complexity: O(k log n) where k is number of events
        """
        async with self._pool.acquire() as conn:
            for event in events:
                await conn.execute(
                    """
                    INSERT INTO events (
                        event_id, aggregate_id, event_type, event_data, 
                        occurred_at, version
                    ) VALUES ($1, $2, $3, $4, $5, 
                        COALESCE((SELECT MAX(version) FROM events WHERE aggregate_id = $2), 0) + 1
                    )
                """,
                    str(event.event_id),
                    aggregate_id,
                    event.event_type(),
                    json.dumps(asdict(event)),
                    event.occurred_at,
                )

    async def get_events(self, aggregate_id: str) -> List[DomainEvent]:
        """
        Get all events for an aggregate
        Time Complexity: O(k) where k is number of events
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM events 
                WHERE aggregate_id = $1 
                ORDER BY version ASC
            """,
                aggregate_id,
            )

            # In a real implementation, you'd deserialize events properly
            # This is a simplified version
            return []


# Composite Repository Pattern - combines multiple repositories
class CompositeTransactionRepository(TransactionRepository):
    """
    Composite repository that uses Redis for caching and PostgreSQL for persistence
    Implements Write-Through Cache Pattern
    Time Complexity: O(1) for cache hits, O(log n) for cache misses
    """

    def __init__(
        self, primary_repo: PostgreSQLTransactionRepository, cache_repo: RedisTransactionRepository
    ):
        self._primary = primary_repo
        self._cache = cache_repo

    async def save(self, transaction: Transaction) -> None:
        """
        Save to both primary and cache
        Time Complexity: O(log n) - dominated by primary repository
        """
        # Save to primary first (consistency)
        await self._primary.save(transaction)

        # Then update cache (best effort)
        try:
            await self._cache.save(transaction)
        except Exception:
            # Cache failure shouldn't fail the operation
            pass

    async def find_by_id(self, transaction_id: TransactionId) -> Optional[Transaction]:
        """
        Try cache first, fallback to primary
        Time Complexity: O(1) cache hit, O(log n) cache miss
        """
        # Try cache first
        transaction = await self._cache.find_by_id(transaction_id)
        if transaction:
            return transaction

        # Cache miss, try primary
        transaction = await self._primary.find_by_id(transaction_id)
        if transaction:
            # Populate cache for next time
            try:
                await self._cache.save(transaction)
            except Exception:
                pass

        return transaction

    async def find_by_customer(self, customer_id: str, limit: int = 100) -> List[Transaction]:
        """
        Use primary repository for complex queries
        Time Complexity: O(log n + k)
        """
        return await self._primary.find_by_customer(customer_id, limit)

    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[Transaction]:
        """
        Use primary repository for range queries
        Time Complexity: O(log n + k)
        """
        return await self._primary.find_by_date_range(start_date, end_date, limit)

    async def count_by_customer(self, customer_id: str) -> int:
        """
        Try cache first for counts
        Time Complexity: O(1) cache hit, O(log n) cache miss
        """
        try:
            count = await self._cache.count_by_customer(customer_id)
            if count > 0:
                return count
        except Exception:
            pass

        return await self._primary.count_by_customer(customer_id)


# Repository Factory - Factory Pattern
class RepositoryFactory:
    """
    Factory for creating repository instances
    Implements Abstract Factory Pattern
    """

    @staticmethod
    def create_transaction_repository(
        db_pool: asyncpg.Pool, redis_client: redis.Redis, use_cache: bool = True
    ) -> TransactionRepository:
        """
        Create transaction repository with optional caching
        Time Complexity: O(1)
        """
        primary_repo = PostgreSQLTransactionRepository(db_pool)

        if use_cache:
            cache_repo = RedisTransactionRepository(redis_client)
            return CompositeTransactionRepository(primary_repo, cache_repo)

        return primary_repo

    @staticmethod
    def create_customer_repository(db_pool: asyncpg.Pool) -> CustomerRepository:
        """
        Create customer repository
        Time Complexity: O(1)
        """
        return PostgreSQLCustomerRepository(db_pool)

    @staticmethod
    def create_event_store(db_pool: asyncpg.Pool) -> EventStore:
        """
        Create event store
        Time Complexity: O(1)
        """
        return PostgreSQLEventStore(db_pool)
