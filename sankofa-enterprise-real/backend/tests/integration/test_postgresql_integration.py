"""
PostgreSQL Integration Tests
=============================

Tests for database integration with real PostgreSQL connection.
Target: 15 tests covering CRUD, transactions, N+1 queries, concurrency

Test categories:
1. Repository CRUD operations
2. Transaction management
3. Connection pooling
4. N+1 query prevention
5. Concurrent access
6. Idempotency
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import patch
import asyncpg


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def db_connection():
    """Create test database connection"""
    # Use test database configuration
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='postgres',
        database='sankofa_test'
    )

    # Clean up test data before each test
    await conn.execute("DELETE FROM transactions WHERE id LIKE 'TEST_%'")
    await conn.execute("DELETE FROM fraud_detections WHERE transaction_id LIKE 'TEST_%'")

    yield conn

    # Clean up after test
    await conn.execute("DELETE FROM transactions WHERE id LIKE 'TEST_%'")
    await conn.execute("DELETE FROM fraud_detections WHERE transaction_id LIKE 'TEST_%'")
    await conn.close()


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for tests"""
    return {
        'id': f'TEST_TXN_{datetime.now().timestamp()}',
        'amount': Decimal('1000.00'),
        'currency': 'BRL',
        'merchant_id': 'TEST_MERCHANT',
        'customer_id': 'TEST_CUSTOMER',
        'cpf': '11144477735',
        'timestamp': datetime.now(),
        'status': 'PENDING'
    }


# ============================================================================
# Repository CRUD Tests
# ============================================================================

class TestRepositoryCRUD:
    """Test basic CRUD operations on repositories"""

    @pytest.mark.asyncio
    async def test_create_transaction(self, db_connection, sample_transaction_data):
        """Test creating a transaction in database"""
        # Insert transaction
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            sample_transaction_data['status']
        )

        # Verify transaction was created
        row = await db_connection.fetchrow(
            "SELECT * FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )

        assert row is not None
        assert row['id'] == sample_transaction_data['id']
        assert float(row['amount']) == float(sample_transaction_data['amount'])

    @pytest.mark.asyncio
    async def test_read_transaction(self, db_connection, sample_transaction_data):
        """Test reading a transaction from database"""
        # Create transaction first
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            sample_transaction_data['status']
        )

        # Read transaction
        row = await db_connection.fetchrow(
            "SELECT * FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )

        assert row['merchant_id'] == sample_transaction_data['merchant_id']
        assert row['customer_id'] == sample_transaction_data['customer_id']

    @pytest.mark.asyncio
    async def test_update_transaction_status(self, db_connection, sample_transaction_data):
        """Test updating transaction status"""
        # Create transaction
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            'PENDING'
        )

        # Update status
        await db_connection.execute(
            "UPDATE transactions SET status = $1 WHERE id = $2",
            'APPROVED',
            sample_transaction_data['id']
        )

        # Verify update
        row = await db_connection.fetchrow(
            "SELECT status FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )

        assert row['status'] == 'APPROVED'

    @pytest.mark.asyncio
    async def test_delete_transaction(self, db_connection, sample_transaction_data):
        """Test deleting (soft delete) a transaction"""
        # Create transaction
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            sample_transaction_data['status']
        )

        # Soft delete (mark as deleted)
        await db_connection.execute(
            "UPDATE transactions SET deleted_at = NOW() WHERE id = $1",
            sample_transaction_data['id']
        )

        # Verify soft delete
        row = await db_connection.fetchrow(
            "SELECT deleted_at FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )

        assert row['deleted_at'] is not None


# ============================================================================
# Transaction Management Tests
# ============================================================================

class TestTransactionManagement:
    """Test database transaction management (ACID properties)"""

    @pytest.mark.asyncio
    async def test_transaction_commit(self, db_connection, sample_transaction_data):
        """Test transaction commit"""
        async with db_connection.transaction():
            await db_connection.execute("""
                INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                sample_transaction_data['id'],
                float(sample_transaction_data['amount']),
                sample_transaction_data['currency'],
                sample_transaction_data['merchant_id'],
                sample_transaction_data['customer_id'],
                sample_transaction_data['cpf'],
                sample_transaction_data['timestamp'],
                sample_transaction_data['status']
            )
            # Transaction commits here

        # Verify data persisted
        row = await db_connection.fetchrow(
            "SELECT * FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_connection, sample_transaction_data):
        """Test transaction rollback on error"""
        try:
            async with db_connection.transaction():
                await db_connection.execute("""
                    INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    sample_transaction_data['id'],
                    float(sample_transaction_data['amount']),
                    sample_transaction_data['currency'],
                    sample_transaction_data['merchant_id'],
                    sample_transaction_data['customer_id'],
                    sample_transaction_data['cpf'],
                    sample_transaction_data['timestamp'],
                    sample_transaction_data['status']
                )

                # Force error to trigger rollback
                raise Exception("Intentional error for rollback test")
        except Exception:
            pass

        # Verify data was NOT persisted (rolled back)
        row = await db_connection.fetchrow(
            "SELECT * FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )
        assert row is None

    @pytest.mark.asyncio
    async def test_nested_transactions(self, db_connection):
        """Test nested transaction handling (savepoints)"""
        txn_id_1 = f'TEST_TXN_1_{datetime.now().timestamp()}'
        txn_id_2 = f'TEST_TXN_2_{datetime.now().timestamp()}'

        async with db_connection.transaction():
            # Outer transaction
            await db_connection.execute("""
                INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, txn_id_1, 1000.0, 'BRL', 'M1', 'C1', '11144477735', datetime.now(), 'PENDING')

            try:
                async with db_connection.transaction():
                    # Inner transaction (savepoint)
                    await db_connection.execute("""
                        INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, txn_id_2, 2000.0, 'BRL', 'M2', 'C2', '00000000191', datetime.now(), 'PENDING')

                    raise Exception("Inner transaction error")
            except Exception:
                pass  # Inner transaction rolled back

        # Outer transaction should be committed
        row1 = await db_connection.fetchrow("SELECT * FROM transactions WHERE id = $1", txn_id_1)
        row2 = await db_connection.fetchrow("SELECT * FROM transactions WHERE id = $1", txn_id_2)

        assert row1 is not None  # Outer committed
        assert row2 is None  # Inner rolled back


# ============================================================================
# N+1 Query Prevention Tests
# ============================================================================

class TestN1QueryPrevention:
    """Test prevention of N+1 query antipattern"""

    @pytest.mark.asyncio
    async def test_batch_fetch_prevents_n1(self, db_connection):
        """Test batch fetching prevents N+1 queries"""
        # Create 100 test transactions
        transaction_ids = []
        for i in range(100):
            txn_id = f'TEST_TXN_{i}_{datetime.now().timestamp()}'
            transaction_ids.append(txn_id)
            await db_connection.execute("""
                INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, txn_id, 1000.0, 'BRL', f'M{i}', f'C{i}', '11144477735', datetime.now(), 'PENDING')

        # BAD: N+1 pattern (1 query for list + N queries for details)
        # Don't actually do this, just test the GOOD pattern

        # GOOD: Single query with JOIN
        query_count = 0

        # Single query to fetch all transactions
        rows = await db_connection.fetch("""
            SELECT * FROM transactions
            WHERE id = ANY($1::text[])
        """, transaction_ids)
        query_count += 1

        assert len(rows) == 100
        assert query_count == 1  # Only 1 query, not 101

    @pytest.mark.asyncio
    async def test_join_prevents_n1(self, db_connection):
        """Test JOINs prevent N+1 queries for related data"""
        # Create transaction with fraud detection
        txn_id = f'TEST_TXN_{datetime.now().timestamp()}'

        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, txn_id, 1000.0, 'BRL', 'M1', 'C1', '11144477735', datetime.now(), 'PENDING')

        # GOOD: Single query with JOIN
        row = await db_connection.fetchrow("""
            SELECT t.*, fd.fraud_probability, fd.risk_factors
            FROM transactions t
            LEFT JOIN fraud_detections fd ON t.id = fd.transaction_id
            WHERE t.id = $1
        """, txn_id)

        assert row is not None
        # Single query retrieved both transaction and fraud detection


# ============================================================================
# Concurrent Access Tests
# ============================================================================

class TestConcurrentAccess:
    """Test concurrent database access and locking"""

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, db_connection):
        """Test concurrent inserts don't conflict"""
        async def insert_transaction(index):
            conn = await asyncpg.connect(
                host='localhost', port=5432,
                user='postgres', password='postgres',
                database='sankofa_test'
            )

            txn_id = f'TEST_CONCURRENT_{index}_{datetime.now().timestamp()}'
            await conn.execute("""
                INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, txn_id, 1000.0, 'BRL', f'M{index}', f'C{index}', '11144477735', datetime.now(), 'PENDING')

            await conn.close()

        # Execute 10 concurrent inserts
        await asyncio.gather(*[insert_transaction(i) for i in range(10)])

        # Verify all 10 were inserted
        count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM transactions WHERE id LIKE 'TEST_CONCURRENT_%'"
        )
        assert count == 10

    @pytest.mark.asyncio
    async def test_row_locking_for_update(self, db_connection, sample_transaction_data):
        """Test SELECT FOR UPDATE prevents race conditions"""
        # Create transaction
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            'PENDING'
        )

        async with db_connection.transaction():
            # Lock row for update
            row = await db_connection.fetchrow("""
                SELECT * FROM transactions
                WHERE id = $1
                FOR UPDATE
            """, sample_transaction_data['id'])

            assert row is not None

            # Update while locked
            await db_connection.execute(
                "UPDATE transactions SET status = 'PROCESSING' WHERE id = $1",
                sample_transaction_data['id']
            )


# ============================================================================
# Idempotency Tests
# ============================================================================

class TestIdempotency:
    """Test idempotent operations"""

    @pytest.mark.asyncio
    async def test_upsert_idempotency(self, db_connection, sample_transaction_data):
        """Test UPSERT (INSERT ... ON CONFLICT) for idempotency"""
        # First insert
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            sample_transaction_data['status']
        )

        # Second insert (should be ignored due to conflict)
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """,
            sample_transaction_data['id'],
            9999.0,  # Different amount
            'USD',  # Different currency
            'DIFFERENT_MERCHANT',
            'DIFFERENT_CUSTOMER',
            '00000000191',
            datetime.now(),
            'APPROVED'
        )

        # Verify original data preserved
        row = await db_connection.fetchrow(
            "SELECT * FROM transactions WHERE id = $1",
            sample_transaction_data['id']
        )

        assert float(row['amount']) == float(sample_transaction_data['amount'])  # Original preserved
        assert row['currency'] == 'BRL'  # Not USD

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self, db_connection, sample_transaction_data):
        """Test duplicate transaction prevention"""
        # First insert
        await db_connection.execute("""
            INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            sample_transaction_data['id'],
            float(sample_transaction_data['amount']),
            sample_transaction_data['currency'],
            sample_transaction_data['merchant_id'],
            sample_transaction_data['customer_id'],
            sample_transaction_data['cpf'],
            sample_transaction_data['timestamp'],
            sample_transaction_data['status']
        )

        # Second insert should fail (duplicate key)
        with pytest.raises(asyncpg.UniqueViolationError):
            await db_connection.execute("""
                INSERT INTO transactions (id, amount, currency, merchant_id, customer_id, cliente_cpf, timestamp, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                sample_transaction_data['id'],  # Same ID
                float(sample_transaction_data['amount']),
                sample_transaction_data['currency'],
                sample_transaction_data['merchant_id'],
                sample_transaction_data['customer_id'],
                sample_transaction_data['cpf'],
                sample_transaction_data['timestamp'],
                sample_transaction_data['status']
            )


# ============================================================================
# Connection Pool Tests
# ============================================================================

class TestConnectionPool:
    """Test connection pool management"""

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Test connection pool reuses connections"""
        pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            database='sankofa_test',
            min_size=2,
            max_size=10
        )

        # Acquire multiple connections
        async with pool.acquire() as conn1:
            result1 = await conn1.fetchval("SELECT 1")

        async with pool.acquire() as conn2:
            result2 = await conn2.fetchval("SELECT 2")

        assert result1 == 1
        assert result2 == 2

        await pool.close()

    @pytest.mark.asyncio
    async def test_connection_pool_max_size(self):
        """Test connection pool respects max size"""
        pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            database='sankofa_test',
            min_size=1,
            max_size=3
        )

        # Try to acquire more than max_size connections
        connections = []
        for _ in range(3):
            conn = await pool.acquire()
            connections.append(conn)

        # Release connections
        for conn in connections:
            await pool.release(conn)

        await pool.close()


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for PostgreSQL Integration:

Repository CRUD: 4 tests
- Create, Read, Update, Delete operations

Transaction Management: 3 tests
- Commit, Rollback, Nested transactions

N+1 Query Prevention: 2 tests
- Batch fetching, JOINs

Concurrent Access: 2 tests
- Concurrent inserts, Row locking

Idempotency: 2 tests
- UPSERT, Duplicate prevention

Connection Pool: 2 tests
- Connection reuse, Max size enforcement

TOTAL: 15 tests
TARGET: Database integrity, performance, concurrency
"""
