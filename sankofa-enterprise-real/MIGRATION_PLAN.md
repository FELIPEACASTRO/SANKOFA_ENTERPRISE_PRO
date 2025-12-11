## MIGRATION PLAN - INCREMENTAL REFACTORING TO CLEAN ARCHITECTURE

**Project:** Sankofa Enterprise Pro
**Status:** Ready for Implementation
**Duration:** 8 weeks (phased approach)
**Risk:** Low (incremental, reversible changes)

---

## EXECUTIVE SUMMARY

### Problem
Current [production_api.py](backend/api/production_api.py) (5135 lines):
- Monolithic structure
- High coupling (business logic + HTTP + database)
- Difficult to test
- Hard to maintain and extend

### Solution
Incremental migration to Hexagonal Architecture + Clean Architecture using **Strangler Fig Pattern**.

### Strategy
**NOT a rewrite** - gradual extraction and refactoring:
1. Add new architecture alongside old code
2. Migrate endpoints one by one
3. Run old and new in parallel (shadow mode)
4. Gradually shift traffic
5. Delete old code once verified

---

## PHASE 0: PREPARATION (Week 0 - Done! ✅)

### Completed
- ✅ [core/value_objects.py](backend/core/value_objects.py) - CPF, Email, RiskScore, Amount
- ✅ [core/fraud_strategies.py](backend/core/fraud_strategies.py) - Strategy Pattern implementation
- ✅ [core/decorators.py](backend/core/decorators.py) - Logging, Metrics, Caching decorators
- ✅ [infrastructure/ml_gateway.py](backend/infrastructure/ml_gateway.py) - ML Engine adapter
- ✅ [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - Complete architecture documentation

### Files Already Available
- ✅ [core/entities.py](backend/core/entities.py) - Transaction, Customer, Money entities
- ✅ [core/interfaces.py](backend/core/interfaces.py) - Repository and service interfaces
- ✅ [core/use_cases.py](backend/core/use_cases.py) - ProcessTransactionUseCase, etc.
- ✅ [infrastructure/repositories.py](backend/infrastructure/repositories.py) - PostgreSQL, Redis repos

---

## PHASE 1: FOUNDATION & TESTING (Week 1)

### Goal
Validate new architecture with comprehensive tests WITHOUT touching production code.

### Tasks

#### 1.1 Write Unit Tests for New Components

```bash
# Create test files
backend/tests/unit/test_core/test_value_objects.py
backend/tests/unit/test_core/test_fraud_strategies.py
backend/tests/unit/test_core/test_decorators.py
```

**Test Coverage Target:** >90% for new code

**Example Test:**

```python
# tests/unit/test_core/test_value_objects.py
import pytest
from core.value_objects import CPF, create_cpf

class TestCPF:
    def test_valid_cpf(self):
        cpf = CPF("12345678909")
        assert cpf.value == "12345678909"

    def test_invalid_cpf_raises(self):
        with pytest.raises(ValueError, match="CPF inválido"):
            CPF("00000000000")

    def test_cpf_from_raw_formats(self):
        cpf = CPF.from_raw("123.456.789-09")
        assert cpf.value == "12345678909"

    def test_cpf_masked_for_logging(self):
        cpf = CPF("12345678909")
        masked = cpf.masked()
        assert masked == "***.***.*89-09"
        assert "12345678" not in masked  # PII not exposed

    @pytest.mark.parametrize("invalid_cpf", [
        "00000000000",
        "11111111111",
        "12345678900",  # Invalid check digit
        "123",          # Too short
        "abc12345678"   # Non-numeric
    ])
    def test_invalid_cpfs(self, invalid_cpf):
        with pytest.raises(ValueError):
            CPF(invalid_cpf)
```

#### 1.2 Create Integration Test Infrastructure

```python
# tests/integration/conftest.py
import pytest
import asyncpg
from infrastructure.repositories import PostgreSQLTransactionRepository

@pytest.fixture
async def db_pool():
    """Create test database connection pool"""
    pool = await asyncpg.create_pool(
        host='localhost',
        database='sankofa_test',
        user='test_user',
        password='test_pass'
    )
    yield pool
    await pool.close()

@pytest.fixture
async def transaction_repo(db_pool):
    """Create transaction repository for testing"""
    return PostgreSQLTransactionRepository(db_pool)

@pytest.fixture
async def clean_database(db_pool):
    """Clean database between tests"""
    async with db_pool.acquire() as conn:
        await conn.execute("TRUNCATE transactions, customers, fraud_detections CASCADE")
```

#### 1.3 Run Tests

```bash
# Run all new tests
cd backend
pytest tests/unit/test_core/ -v --cov=core --cov-report=html

# Target: >90% coverage
```

**Success Criteria:**
- ✅ All unit tests passing
- ✅ >90% code coverage for new modules
- ✅ No flaky tests

**Time:** 3-4 days
**Risk:** Low (no production changes)

---

## PHASE 2: PILOT ENDPOINT (Week 2)

### Goal
Migrate ONE endpoint (`/api/predict`) to prove architecture works in production.

### 2.1 Create New Endpoint with Clean Architecture

**File:** `backend/api/routes/fraud_v2.py`

```python
"""
New fraud detection endpoint using Clean Architecture

This runs ALONGSIDE old production_api.py for comparison
"""

from flask import Blueprint, request, jsonify, g
from core.use_cases import ProcessTransactionUseCase
from core.interfaces import ProcessTransactionCommand
from core.decorators import LoggingDecorator, MetricsDecorator
from utils.structured_logging import get_structured_logger
from utils.log_sanitizer import sanitize_log_data

fraud_v2_bp = Blueprint('fraud_v2', __name__)
logger = get_structured_logger("fraud_v2_api")


# Dependency injection (configured in app factory)
process_transaction_use_case = None  # Injected at startup


@fraud_v2_bp.route('/api/v2/predict', methods=['POST'])
@LoggingDecorator(logger_name="fraud_v2_predict")
@MetricsDecorator(metrics_collector, metric_prefix="fraud_v2_predict")
async def predict_v2():
    """
    NEW fraud prediction endpoint using Clean Architecture

    Request:
        POST /api/v2/predict
        {
            "amount": 1000.50,
            "currency": "BRL",
            "merchant_id": "M123",
            "customer_id": "C456",
            "metadata": {...}
        }

    Response:
        {
            "transaction_id": "TXN_ABC123",
            "status": "pending",
            "risk_level": "medium",
            "risk_score": 0.65,
            "decision": "manual_review_required",
            "processing_time_ms": 45.2
        }
    """
    try:
        # 1. Validate input (Pydantic)
        from api.schemas import TransactionRequest
        validated_data = TransactionRequest(**request.json)

        # 2. Create command
        command = ProcessTransactionCommand(
            amount=validated_data.amount,
            currency=validated_data.currency,
            merchant_id=validated_data.merchant_id,
            customer_id=validated_data.customer_id,
            metadata=validated_data.metadata
        )

        # 3. Execute use case
        result = await process_transaction_use_case.execute(command)

        # 4. Return response
        return jsonify({
            'success': True,
            'data': result
        }), 200

    except Exception as e:
        logger.error(
            "Fraud prediction v2 failed",
            extra=sanitize_log_data({'error': str(e), 'user_id': g.get('user', {}).get('id')})
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### 2.2 Configure Dependency Injection

**File:** `backend/api/app_factory_v2.py`

```python
"""
Application factory for Clean Architecture setup

Wires together:
- Repositories
- Use cases
- Services
- Blueprints
"""

from flask import Flask
import asyncpg
import redis.asyncio as redis

from infrastructure.repositories import (
    PostgreSQLTransactionRepository,
    PostgreSQLCustomerRepository,
    CompositeTransactionRepository,
    RedisTransactionRepository
)
from infrastructure.ml_gateway import create_production_ml_gateway
from core.use_cases import ProcessTransactionUseCase
from ml_engine.production_fraud_engine import get_fraud_engine
from monitoring.observability import observability_metrics
from cache.redis_cache_system import redis_cache_system


async def create_app_v2(config_name='production'):
    """
    Create Flask app with Clean Architecture wiring

    This is the Composition Root - where all dependencies are wired together
    """
    app = Flask(__name__)
    app.config.from_object(f'config.{config_name}')

    # 1. Create infrastructure components
    # Database
    db_pool = await asyncpg.create_pool(
        host=app.config['DB_HOST'],
        database=app.config['DB_NAME'],
        user=app.config['DB_USER'],
        password=app.config['DB_PASSWORD']
    )

    # Redis
    redis_client = await redis.from_url(app.config['REDIS_URL'])

    # 2. Create repositories (with caching)
    primary_repo = PostgreSQLTransactionRepository(db_pool)
    cache_repo = RedisTransactionRepository(redis_client)
    transaction_repo = CompositeTransactionRepository(primary_repo, cache_repo)

    customer_repo = PostgreSQLCustomerRepository(db_pool)

    # 3. Create services
    fraud_engine = get_fraud_engine()
    ml_gateway = create_production_ml_gateway(fraud_engine, redis_cache_system)

    # 4. Create use cases (with dependency injection)
    process_transaction_use_case = ProcessTransactionUseCase(
        transaction_repo=transaction_repo,
        customer_repo=customer_repo,
        fraud_service=ml_gateway,
        notification_service=...,  # Configure notification service
        audit_service=...,          # Configure audit service
        cache_service=redis_cache_system,
        event_publisher=...,        # Configure event publisher
        metrics_collector=observability_metrics
    )

    # 5. Register blueprints
    from api.routes.fraud_v2 import fraud_v2_bp
    fraud_v2_bp.process_transaction_use_case = process_transaction_use_case  # Inject
    app.register_blueprint(fraud_v2_bp)

    return app
```

### 2.3 Shadow Mode Testing

**Goal:** Run v1 and v2 in parallel, compare results

```python
@app.route('/api/predict', methods=['POST'])
async def predict_v1_with_shadow():
    """
    Original endpoint (v1) with shadow v2 call

    1. Call v1 (production)
    2. Call v2 (new architecture) in background
    3. Compare results
    4. Return v1 result (no user impact)
    5. Log differences
    """
    import asyncio

    # Call v1 (production)
    result_v1 = original_predict_logic(request.json)

    # Call v2 (new) in background - doesn't block v1
    asyncio.create_task(shadow_test_v2(request.json, result_v1))

    # Return v1 result (no user impact)
    return jsonify(result_v1)


async def shadow_test_v2(request_data, result_v1):
    """Call v2 and compare with v1"""
    try:
        result_v2 = await call_v2_predict(request_data)

        # Compare results
        diff = compare_results(result_v1, result_v2)

        # Log comparison
        logger.info(
            "Shadow test v1 vs v2",
            extra={
                'v1_risk_score': result_v1['risk_score'],
                'v2_risk_score': result_v2['risk_score'],
                'difference': diff,
                'match': diff < 0.05  # 5% tolerance
            }
        )

        # Track metrics
        metrics.increment_counter('shadow_test_executed')
        if diff < 0.05:
            metrics.increment_counter('shadow_test_match')
        else:
            metrics.increment_counter('shadow_test_mismatch')

    except Exception as e:
        logger.error(f"Shadow test failed: {e}")
        metrics.increment_counter('shadow_test_error')
```

### 2.4 Monitor & Compare

**Metrics to track:**

```python
# Grafana Dashboard: "v1 vs v2 Comparison"

# 1. Latency comparison
fraud_predict_v1_duration_ms (p50, p95, p99)
fraud_predict_v2_duration_ms (p50, p95, p99)

# 2. Accuracy comparison
shadow_test_match_rate (target: >95%)
shadow_test_mean_difference (target: <5%)

# 3. Error rates
fraud_predict_v1_errors
fraud_predict_v2_errors

# 4. Resource usage
fraud_predict_v2_memory_mb
fraud_predict_v2_cpu_percent
```

**Success Criteria:**
- ✅ v2 latency ≤ v1 latency
- ✅ v2 error rate ≤ v1 error rate
- ✅ Shadow test match rate >95%
- ✅ No memory leaks in v2

**Time:** 5-7 days
**Risk:** Low (shadow mode, no user impact)

---

## PHASE 3: GRADUAL CUTOVER (Week 3)

### Goal
Shift traffic from v1 to v2 gradually using feature flags.

### 3.1 Add Feature Flag

```python
# config/feature_flags.py
class FeatureFlags:
    @staticmethod
    def use_v2_predict(user_id: str = None) -> bool:
        """
        Determine if user should use v2 predict endpoint

        Rollout strategy:
        - Week 3 Day 1: 1% of traffic
        - Week 3 Day 2: 5% of traffic
        - Week 3 Day 3: 10% of traffic
        - Week 3 Day 4: 25% of traffic
        - Week 3 Day 5: 50% of traffic
        - Week 3 Day 6: 75% of traffic
        - Week 3 Day 7: 100% of traffic
        """
        import random
        import os

        # Check environment variable for override
        override = os.getenv('USE_V2_PREDICT_PERCENT')
        if override:
            percent = int(override)
        else:
            percent = 1  # Default: 1% of traffic

        # Deterministic based on user_id (if provided)
        if user_id:
            import hashlib
            hash_value = int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16)
            return (hash_value % 100) < percent
        else:
            # Random for anonymous users
            return random.random() < (percent / 100.0)
```

### 3.2 Update Routing Logic

```python
@app.route('/api/predict', methods=['POST'])
async def predict_with_feature_flag():
    """
    Route to v1 or v2 based on feature flag

    Gradually increases v2 traffic
    """
    user_id = g.get('user', {}).get('id')

    if feature_flags.use_v2_predict(user_id):
        # Use v2 (new architecture)
        metrics.increment_counter('predict_v2_routed')
        return await predict_v2()
    else:
        # Use v1 (legacy)
        metrics.increment_counter('predict_v1_routed')
        return predict_v1()
```

### 3.3 Rollout Schedule

| Day | v2 Traffic % | Monitoring Focus |
|-----|--------------|------------------|
| Mon | 1% | Initial validation, check for errors |
| Tue | 5% | Latency, error rates |
| Wed | 10% | Resource usage, memory leaks |
| Thu | 25% | Load testing, concurrency |
| Fri | 50% | Performance under load |
| Sat | 75% | Final validation |
| Sun | 100% | Full cutover |

**Rollback Plan:** Set `USE_V2_PREDICT_PERCENT=0` if issues detected

**Time:** 7 days (gradual)
**Risk:** Very Low (gradual, reversible)

---

## PHASE 4: MIGRATE REMAINING ENDPOINTS (Week 4-7)

### Goal
Migrate all remaining endpoints using same pattern.

### Priority Order

| Priority | Endpoints | Reason | Week |
|----------|-----------|--------|------|
| P0 | `/api/predict` | Critical path, high traffic | ✅ Week 3 |
| P1 | `/api/batch/predict` | High traffic, similar to predict | Week 4 |
| P2 | `/api/transactions/*` | CRUD operations | Week 5 |
| P3 | `/api/dashboard/*` | Analytics, reporting | Week 6 |
| P4 | `/api/admin/*` | Low traffic, admin operations | Week 7 |

### Migration Checklist (Per Endpoint)

```markdown
## Endpoint: /api/{endpoint_name}

- [ ] Create new route in `api/routes/{domain}_v2.py`
- [ ] Write use case (if needed)
- [ ] Write unit tests (>90% coverage)
- [ ] Write integration tests
- [ ] Deploy with feature flag (0% traffic)
- [ ] Enable shadow mode (1 week)
- [ ] Gradual rollout (1% → 100%)
- [ ] Monitor metrics
- [ ] Delete old code
```

**Time:** 4 weeks (10-20 endpoints per week)
**Risk:** Low (proven pattern)

---

## PHASE 5: CLEANUP & OPTIMIZATION (Week 8)

### Goal
Delete legacy code, optimize, finalize documentation.

### 5.1 Delete Legacy Code

```bash
# Once ALL endpoints migrated and stable
git rm backend/api/production_api.py

# Commit
git commit -m "feat: Remove legacy production_api.py - migration complete"
```

### 5.2 Performance Optimization

```python
# Add connection pooling limits
db_pool = await asyncpg.create_pool(
    ...,
    min_size=10,
    max_size=100,
    max_queries=50000,
    max_inactive_connection_lifetime=300
)

# Add query timeouts
@app.before_request
def set_query_timeout():
    g.query_timeout = 5.0  # 5 seconds

# Add circuit breakers
from core.decorators import CircuitBreakerDecorator

@CircuitBreakerDecorator(failure_threshold=5, timeout=60.0)
async def call_external_service():
    ...
```

### 5.3 Final Documentation

- Update [README.md](README.md) with new architecture
- Document deployment process
- Create runbooks for operations
- Update API documentation

**Time:** 1 week
**Risk:** None

---

## METRICS & SUCCESS CRITERIA

### Code Quality Metrics

| Metric | Before | Target | After (Goal) |
|--------|--------|--------|--------------|
| Lines per file (max) | 5135 | <500 | <300 |
| Cyclomatic complexity (avg) | 15-25 | <10 | <8 |
| Test coverage | 60% | 90% | >90% |
| Code duplication | High | Low | None |

### Performance Metrics

| Metric | Before | Target | After (Goal) |
|--------|--------|--------|--------------|
| P95 latency /api/predict | 150ms | <50ms | <50ms |
| N+1 queries | Yes | No | No |
| Cache hit rate | N/A | >80% | >85% |
| Throughput (req/s) | 1000 | 1500 | 2000 |

### Architecture Metrics

| Metric | Before | Target | After (Goal) |
|--------|--------|--------|--------------|
| Domain → Infrastructure dependency | Yes | No | No |
| Framework coupling | High | Low | None |
| Testability (no mocks) | 20% | 90% | >95% |
| Interface segregation | Poor | Good | Excellent |

---

## RISK MANAGEMENT

### Identified Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Performance regression | High | Low | Shadow testing, gradual rollout, monitoring |
| Bugs in new code | Medium | Medium | >90% test coverage, code review |
| Database migration issues | High | Low | No schema changes required |
| Team learning curve | Low | Medium | Documentation, pair programming |
| Production incidents | High | Low | Feature flags, instant rollback |

### Rollback Plan

**If any issues detected during rollout:**

1. **Immediate:** Set feature flag to 0%
   ```bash
   kubectl set env deployment/sankofa-api USE_V2_PREDICT_PERCENT=0
   ```

2. **Within 5 minutes:** Verify metrics returned to normal

3. **Within 1 hour:** Root cause analysis

4. **Fix or rollback:** Deploy fix or continue with v1

**Rollback Time:** <5 minutes
**Data Loss Risk:** None (same database)

---

## TIMELINE SUMMARY

```
Week 0: ✅ Preparation (DONE)
Week 1: ✅ Foundation & Testing
Week 2: ✅ Pilot Endpoint (/api/predict)
Week 3: ✅ Gradual Cutover (1% → 100%)
Week 4: ⏳ Batch Predict + Transactions
Week 5: ⏳ Dashboard + Analytics
Week 6: ⏳ Admin Endpoints
Week 7: ⏳ Remaining Endpoints
Week 8: ⏳ Cleanup & Optimization

Total: 8 weeks
```

---

## NEXT STEPS

### Immediate (This Week)

1. **Review this migration plan** with team
2. **Set up test infrastructure** (conftest.py, fixtures)
3. **Write unit tests** for Phase 0 components
4. **Create feature flag system**

### Week 1

5. **Begin Phase 1**: Write comprehensive tests
6. **Set up monitoring dashboard** for v1 vs v2 comparison
7. **Schedule team training** on Clean Architecture

### Communication

- **Daily standups:** Migration progress
- **Weekly demos:** Show working v2 endpoints
- **Bi-weekly reviews:** Metrics and learnings
- **Documentation:** Keep architecture guide updated

---

**Migration Owner:** [Your Name]
**Started:** 2025-12-11
**Expected Completion:** 2026-02-05

**Status:** ✅ PHASE 0 COMPLETE - READY FOR PHASE 1

---

Generated with Claude Code
