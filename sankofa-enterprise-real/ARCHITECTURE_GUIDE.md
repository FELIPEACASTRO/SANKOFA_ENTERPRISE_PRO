# SANKOFA ENTERPRISE PRO - ARCHITECTURE GUIDE
## Reference Architecture for Production-Grade Fraud Detection Systems

**Status:** Reference Implementation
**Last Updated:** 2025-12-11
**Architecture Style:** Hexagonal Architecture + Clean Architecture + DDD

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Architectural Principles](#architectural-principles)
3. [System Overview](#system-overview)
4. [Layer Architecture](#layer-architecture)
5. [Design Patterns Applied](#design-patterns-applied)
6. [SOLID Principles](#solid-principles)
7. [Performance Optimization](#performance-optimization)
8. [Security Architecture](#security-architecture)
9. [Testing Strategy](#testing-strategy)
10. [Migration Plan](#migration-plan)

---

## 1. EXECUTIVE SUMMARY

### Current State (As-Is)

**Problem:** Monolithic [production_api.py](backend/api/production_api.py) (5135 lines)
- Mixed concerns: HTTP routes + business logic + database access
- High coupling: Impossible to test business logic without Flask
- Code duplication: CPF validation in 15+ places
- No clear boundaries: Everything knows about everything

**Metrics:**
- Cyclomatic complexity: 15-25 (target: <10)
- Test coverage: 60% (target: 90%)
- Lines per file: 5135 (target: <500)
- Coupling: High (target: Low via interfaces)

### Target State (To-Be)

**Solution:** Hexagonal Architecture with Clean Architecture layers

```
┌──────────────────────────────────────────────────────────┐
│                     PRESENTATION                         │
│         (Flask routes, REST API, GraphQL)                │
└──────────────────┬───────────────────────────────────────┘
                   │ HTTP/JSON
┌──────────────────▼───────────────────────────────────────┐
│                   ADAPTERS LAYER                         │
│   (HTTP Controllers, API Routes, Event Listeners)        │
└──────────────────┬───────────────────────────────────────┘
                   │ Commands/Queries
┌──────────────────▼───────────────────────────────────────┐
│                  APPLICATION LAYER                       │
│     (Use Cases, Command/Query Handlers, DTOs)            │
└──────────────────┬───────────────────────────────────────┘
                   │ Domain Entities
┌──────────────────▼───────────────────────────────────────┐
│                    DOMAIN LAYER                          │
│   (Entities, Value Objects, Domain Services, Events)     │
└──────────────────────────────────────────────────────────┘
                   ▲
                   │ Interfaces (Ports)
┌──────────────────┴───────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                     │
│     (PostgreSQL, Redis, ML Models, External APIs)        │
└──────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Testable: Test business logic without HTTP/DB
- ✅ Maintainable: Each layer has single responsibility
- ✅ Extensible: Add new features without touching existing code
- ✅ Portable: Swap databases, frameworks, ML models easily

**Metrics (Target):**
- Cyclomatic complexity: <10
- Test coverage: >90%
- Lines per file: <300
- Coupling: Low (via interfaces)

---

## 2. ARCHITECTURAL PRINCIPLES

### 2.1 Core Principles

1. **Dependency Inversion Principle (SOLID)**
   - High-level modules don't depend on low-level modules
   - Both depend on abstractions (interfaces)

   ```python
   # ❌ BAD: Direct dependency
   class FraudDetectionService:
       def __init__(self):
           self.db = PostgreSQL()  # Tightly coupled

   # ✅ GOOD: Dependency on interface
   class FraudDetectionService:
       def __init__(self, repo: TransactionRepository):  # Interface
           self._repo = repo  # Can be PostgreSQL, MongoDB, InMemory
   ```

2. **Separation of Concerns**
   - Each layer has ONE responsibility
   - Domain layer: Business rules
   - Application layer: Orchestration
   - Infrastructure layer: Technical details

3. **Domain-Driven Design (DDD)**
   - Ubiquitous Language: Same terms in code and business
   - Entities: Objects with identity (Transaction, Customer)
   - Value Objects: Immutable objects without identity (CPF, Email)
   - Aggregates: Consistency boundaries (TransactionAggregate)

---

## 3. SYSTEM OVERVIEW

### 3.1 Context Diagram (C4 Level 1)

```
┌─────────────┐                    ┌───────────────────┐
│   Cliente   │────────────────────▶│                   │
│   Mobile    │      HTTPS/JSON     │                   │
└─────────────┘                     │                   │
                                    │   SANKOFA         │
┌─────────────┐                     │   ENTERPRISE      │
│   Cliente   │────────────────────▶│   PRO             │
│     Web     │      HTTPS/JSON     │                   │
└─────────────┘                     │   (Fraud          │
                                    │    Detection)     │
┌─────────────┐                     │                   │
│   Analyst   │────────────────────▶│                   │
│  Dashboard  │      HTTPS/JSON     └─────────┬─────────┘
└─────────────┘                               │
                                              │
                ┌─────────────────────────────┼─────────────────────┐
                │                             │                     │
                ▼                             ▼                     ▼
        ┌───────────────┐           ┌─────────────────┐   ┌─────────────┐
        │  PostgreSQL   │           │      Redis      │   │  ML Models  │
        │  (Persistence)│           │    (Cache)      │   │  (Inference)│
        └───────────────┘           └─────────────────┘   └─────────────┘
```

### 3.2 Component Diagram (C4 Level 2)

```
┌──────────────────────────────────────────────────────────────────┐
│                         SANKOFA API                               │
│                                                                   │
│  ┌────────────────┐    ┌────────────────┐   ┌─────────────────┐│
│  │  Fraud API     │    │  Transaction   │   │  Admin API      ││
│  │  /api/predict  │    │  API           │   │  /api/rules     ││
│  └────────┬───────┘    └────────┬───────┘   └────────┬────────┘│
│           │                     │                     │          │
│           └─────────────────────┼─────────────────────┘          │
│                                 │                                │
│  ┌──────────────────────────────▼──────────────────────────────┐│
│  │                  APPLICATION LAYER                          ││
│  │                                                              ││
│  │  ┌──────────────────────┐    ┌─────────────────────────┐   ││
│  │  │ ProcessTransaction   │    │  ApproveTransaction     │   ││
│  │  │ UseCase              │    │  UseCase                │   ││
│  │  └──────────┬───────────┘    └───────────┬─────────────┘   ││
│  │             │                            │                  ││
│  │             └────────────┬───────────────┘                  ││
│  └──────────────────────────┼──────────────────────────────────┘│
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────────┐│
│  │                    DOMAIN LAYER                             ││
│  │                                                              ││
│  │  Entities: Transaction, Customer, FraudAnalysisResult       ││
│  │  Value Objects: CPF, Email, RiskScore, Amount               ││
│  │  Domain Services: FraudScoringStrategy (Strategy Pattern)   ││
│  │  Events: FraudDetected, TransactionApproved                 ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌───────────────────────────────────────────────────────────────┐│
│  │               INFRASTRUCTURE LAYER                           ││
│  │                                                              ││
│  │  PostgreSQL Repositories  │  Redis Cache  │  ML Gateway     ││
│  └───────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. LAYER ARCHITECTURE

### 4.1 Domain Layer (Innermost)

**Responsibility:** Business rules and domain logic

**Location:** [backend/core/](backend/core/)

**Files:**
- [entities.py](backend/core/entities.py) - Business entities
- [value_objects.py](backend/core/value_objects.py) - Immutable value types
- [interfaces.py](backend/core/interfaces.py) - Port interfaces
- [fraud_strategies.py](backend/core/fraud_strategies.py) - Fraud detection strategies

**Key Components:**

```python
# Example: Transaction entity
@dataclass
class Transaction:
    id: TransactionId
    amount: Money
    customer_id: str
    risk_score: float = 0.0

    def mark_as_fraud(self, reason: str) -> None:
        """Business rule: Mark transaction as fraudulent"""
        self.status = TransactionStatus.REJECTED
        self.risk_level = RiskLevel.CRITICAL

    def requires_manual_review(self) -> bool:
        """Business rule: Check if manual review required"""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
```

**Rules:**
- ✅ NO dependencies on outer layers
- ✅ NO framework dependencies (Flask, FastAPI, etc.)
- ✅ Pure business logic only
- ✅ Framework-agnostic
- ✅ 100% testable without mocks

### 4.2 Application Layer

**Responsibility:** Use case orchestration

**Location:** [backend/core/use_cases.py](backend/core/use_cases.py)

**Key Components:**

```python
class ProcessTransactionUseCase:
    """
    Use Case: Process a new transaction
    Orchestrates: fraud detection, risk assessment, decision making
    """

    def __init__(
        self,
        transaction_repo: TransactionRepository,  # Interface
        fraud_service: FraudDetectionService,     # Interface
        notification_service: NotificationService, # Interface
        audit_service: AuditService               # Interface
    ):
        # Dependency Injection via constructor
        self._transaction_repo = transaction_repo
        self._fraud_service = fraud_service
        self._notification_service = notification_service
        self._audit_service = audit_service

    async def execute(self, command: ProcessTransactionCommand) -> Dict[str, Any]:
        # 1. Create entity
        transaction = TransactionFactory.create_transaction(...)

        # 2. Fraud detection
        fraud_analysis = await self._fraud_service.analyze_transaction(transaction)

        # 3. Apply business rules
        decision = self._apply_business_rules(transaction, fraud_analysis)

        # 4. Persist
        await self._transaction_repo.save(transaction)

        # 5. Send notifications
        if decision == "fraud":
            await self._notification_service.send_fraud_alert(...)

        # 6. Audit log
        await self._audit_service.log_transaction_event(...)

        return {"transaction_id": transaction.id, "decision": decision}
```

**Rules:**
- ✅ Depends only on Domain layer + Interfaces
- ✅ NO infrastructure dependencies
- ✅ Orchestrates domain entities
- ✅ Testable with mock repositories

### 4.3 Adapters Layer (Infrastructure)

**Responsibility:** Technical implementation of interfaces

**Location:** [backend/infrastructure/](backend/infrastructure/)

**Key Components:**

```python
# Example: PostgreSQL Repository Adapter
class PostgreSQLTransactionRepository(TransactionRepository):
    """
    Adapter: Implements TransactionRepository interface for PostgreSQL

    Converts between:
    - Domain entities (Transaction) ←→ Database rows
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool

    async def save(self, transaction: Transaction) -> None:
        """Convert entity to SQL and save"""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO transactions (id, amount, currency, ...)
                VALUES ($1, $2, $3, ...)
                ON CONFLICT (id) DO UPDATE SET ...
                """,
                transaction.id.value,
                float(transaction.amount.amount),
                transaction.amount.currency,
                ...
            )

    async def find_by_id(self, txn_id: TransactionId) -> Optional[Transaction]:
        """Convert SQL row to entity"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM transactions WHERE id = $1", txn_id.value)

            if not row:
                return None

            # Map database row to domain entity
            return Transaction(
                id=TransactionId(row["id"]),
                amount=Money(Decimal(str(row["amount"])), row["currency"]),
                ...
            )
```

**ML Engine Adapter:**

```python
# Example: ML Model Gateway Adapter
class ProductionMLGateway(FraudDetectionService):
    """
    Adapter: Wraps ML engine behind FraudDetectionService interface

    Benefits:
    - Domain layer doesn't know about ML engine details
    - Can swap ML frameworks (sklearn → PyTorch → TensorFlow)
    - Can add caching, monitoring without touching domain
    """

    def __init__(self, fraud_engine):
        self._engine = fraud_engine

    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """Convert transaction entity to ML input, call model, return result"""
        # Convert domain entity to ML input format
        ml_input = {
            'amount': float(transaction.amount.amount),
            'merchant_id': transaction.merchant_id,
            'customer_id': transaction.customer_id,
            ...
        }

        # Call ML engine
        prediction = await self._engine.predict(ml_input)

        # Convert ML output to domain entity
        return FraudAnalysisResult(
            transaction_id=transaction.id,
            is_fraud=prediction['is_fraud'],
            confidence_score=prediction['fraud_probability'],
            risk_factors=prediction.get('risk_factors', []),
            model_version=prediction.get('model_version', '1.0')
        )
```

**Rules:**
- ✅ Implements interfaces from Domain layer
- ✅ Contains ALL framework dependencies
- ✅ Can be swapped without touching Domain/Application
- ✅ One adapter per external system

---

## 5. DESIGN PATTERNS APPLIED

### 5.1 Strategy Pattern (Fraud Detection)

**Problem:** Multiple fraud detection algorithms need to be interchangeable

**Solution:** Strategy Pattern

**Implementation:** [backend/core/fraud_strategies.py](backend/core/fraud_strategies.py)

```python
# Interface
class FraudScoringStrategy(ABC):
    @abstractmethod
    async def calculate_score(self, txn: Transaction, context: Dict) -> FraudScoreResult:
        pass

# Concrete strategies
class RuleBasedScoring(FraudScoringStrategy):
    async def calculate_score(self, txn, context):
        # Rule-based logic
        score = 0.0
        if txn.amount > 5000: score += 0.3
        if txn.timestamp.hour < 6: score += 0.2
        return FraudScoreResult(score=RiskScore(score), ...)

class MLBasedScoring(FraudScoringStrategy):
    async def calculate_score(self, txn, context):
        # ML model prediction
        features = self._extract_features(txn, context)
        prediction = await self._model.predict(features)
        return FraudScoreResult(score=RiskScore(prediction['fraud_prob']), ...)

class CompositeScoring(FraudScoringStrategy):
    """Combines multiple strategies with weights"""
    def __init__(self, strategies: List[Tuple[Strategy, float]]):
        self._strategies = strategies

    async def calculate_score(self, txn, context):
        # Weighted average of all strategies
        tasks = [strategy.calculate_score(txn, context) for strategy, _ in self._strategies]
        results = await asyncio.gather(*tasks)

        weighted_score = sum(
            result.score.value * weight
            for (_, weight), result in zip(self._strategies, results)
        )
        return FraudScoreResult(score=RiskScore(weighted_score), ...)
```

**Usage:**

```python
# Easy A/B testing
if config.fraud_strategy == "conservative":
    strategy = create_conservative_strategy(ml_model)
elif config.fraud_strategy == "aggressive":
    strategy = create_aggressive_strategy(ml_model)
else:
    strategy = create_default_scoring_strategy(ml_model)

# Use strategy in use case
fraud_service = FraudDetectionServiceImpl(strategy)
use_case = ProcessTransactionUseCase(repo, fraud_service, ...)
```

**Benefits:**
- ✅ Open/Closed: Add new strategies without modifying existing ones
- ✅ Testable: Each strategy tested independently
- ✅ A/B testing: Switch strategies at runtime
- ✅ Composable: Combine strategies with weights

### 5.2 Decorator Pattern (Cross-Cutting Concerns)

**Problem:** Need logging, metrics, caching without polluting business logic

**Solution:** Decorator Pattern

**Implementation:** [backend/core/decorators.py](backend/core/decorators.py)

```python
# Original use case (no logging/metrics)
async def process_transaction(command):
    transaction = create_transaction(...)
    result = await fraud_service.analyze(transaction)
    await repo.save(transaction)
    return result

# Wrap with decorators
@LoggingDecorator(logger_name="fraud_detection")
@MetricsDecorator(metrics_collector)
@CachingDecorator(cache_service, ttl=3600)
async def process_transaction(command):
    # Same business logic, but now with logging, metrics, caching!
    transaction = create_transaction(...)
    result = await fraud_service.analyze(transaction)
    await repo.save(transaction)
    return result
```

**Stack multiple decorators:**

```python
# Order matters:
# 1. Logging (outermost - logs everything)
# 2. Metrics (records execution time)
# 3. Retry (retries on failure)
# 4. Caching (innermost - avoids execution if cached)

@LoggingDecorator()
@MetricsDecorator(metrics)
@RetryDecorator(max_retries=3)
@CachingDecorator(cache, ttl=1800)
async def expensive_calculation():
    # Business logic here
    pass
```

**Benefits:**
- ✅ Separation of concerns: Business logic clean
- ✅ Composable: Stack multiple decorators
- ✅ Reusable: Apply to any function
- ✅ Testable: Test business logic without decorators

### 5.3 Repository Pattern (Data Access)

**Problem:** Business logic shouldn't know about database details

**Solution:** Repository Pattern

**Implementation:** [backend/infrastructure/repositories.py](backend/infrastructure/repositories.py)

```python
# Interface (in Domain layer)
class TransactionRepository(ABC):
    @abstractmethod
    async def save(self, transaction: Transaction) -> None:
        pass

    @abstractmethod
    async def find_by_id(self, txn_id: TransactionId) -> Optional[Transaction]:
        pass

# Implementations (in Infrastructure layer)
class PostgreSQLTransactionRepository(TransactionRepository):
    # PostgreSQL-specific implementation
    pass

class RedisTransactionRepository(TransactionRepository):
    # Redis-specific implementation
    pass

class InMemoryTransactionRepository(TransactionRepository):
    # In-memory for testing
    pass

# Composite (combines multiple repositories)
class CompositeTransactionRepository(TransactionRepository):
    """
    Write-Through Cache Pattern:
    1. Write to PostgreSQL (primary)
    2. Write to Redis (cache)
    3. Read from Redis first, fallback to PostgreSQL
    """
    def __init__(self, primary: PostgreSQL, cache: Redis):
        self._primary = primary
        self._cache = cache

    async def save(self, txn):
        await self._primary.save(txn)  # Write to DB
        await self._cache.save(txn)    # Update cache

    async def find_by_id(self, txn_id):
        result = await self._cache.find_by_id(txn_id)
        if result:
            return result  # Cache HIT

        result = await self._primary.find_by_id(txn_id)
        if result:
            await self._cache.save(result)  # Populate cache

        return result
```

**Benefits:**
- ✅ Testable: Use InMemoryRepository for tests
- ✅ Swappable: Switch databases without changing domain
- ✅ Composable: Combine multiple repositories (cache + primary)

### 5.4 Factory Pattern (Entity Creation)

**Problem:** Complex entity creation logic scattered everywhere

**Solution:** Factory Pattern

```python
class TransactionFactory:
    @staticmethod
    def create_transaction(
        amount: Decimal,
        currency: str,
        merchant_id: str,
        customer_id: str,
        metadata: Optional[Dict] = None
    ) -> Transaction:
        """
        Factory: Encapsulates transaction creation logic

        Benefits:
        - Validation in one place
        - Consistent ID generation
        - Easy to add creation logic (e.g., default values)
        """
        transaction_id = TransactionId(f"TXN_{uuid4().hex[:12].upper()}")
        money = Money(amount, currency)

        return Transaction(
            id=transaction_id,
            amount=money,
            merchant_id=merchant_id,
            customer_id=customer_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
```

**Benefits:**
- ✅ DRY: Creation logic in one place
- ✅ Consistent: Always valid entities
- ✅ Extensible: Easy to add creation variants

### 5.5 Value Object Pattern (DDD)

**Problem:** CPF validation duplicated in 15+ places

**Solution:** Value Object Pattern

**Implementation:** [backend/core/value_objects.py](backend/core/value_objects.py)

```python
@dataclass(frozen=True)  # Immutable
class CPF:
    """
    Value Object: CPF Brasileiro

    Self-validating, immutable, domain-specific type
    """
    value: str

    def __post_init__(self):
        if not self._is_valid(self.value):
            raise ValueError(f"CPF inválido: {self.value}")

    @classmethod
    def from_raw(cls, raw: str) -> 'CPF':
        """Create from formatted string: 123.456.789-09"""
        cleaned = re.sub(r'\D', '', raw)
        return cls(cleaned)

    def masked(self) -> str:
        """Return masked for logging: ***.***.*67-89"""
        return f"***.***.*{self.value[7:9]}-{self.value[9:]}"

    @staticmethod
    def _is_valid(cpf: str) -> bool:
        # Full CPF validation algorithm
        ...
```

**Usage:**

```python
# Before (validation scattered everywhere)
def process_transaction(cpf_str: str):
    if not validate_cpf(cpf_str):  # Validation logic duplicated
        raise ValueError("Invalid CPF")
    # More duplicated validation...

# After (validation encapsulated)
def process_transaction(cpf: CPF):  # Type guarantees validity!
    # CPF is already validated, no need to check
    # Can safely use cpf.value, cpf.masked(), etc.
```

**Benefits:**
- ✅ DRY: Validation in ONE place
- ✅ Type-safe: CPF vs str
- ✅ Immutable: Can't be changed after creation
- ✅ Rich behavior: masked(), formatted()

---

## 6. SOLID PRINCIPLES

### 6.1 Single Responsibility Principle (SRP)

**Definition:** Each class should have ONE reason to change

**Example:**

```python
# ❌ BAD: Multiple responsibilities
class TransactionService:
    def process_transaction(self, data):
        # 1. HTTP parsing (HTTP responsibility)
        transaction = self._parse_request(data)

        # 2. Validation (Domain responsibility)
        if not self._validate(transaction):
            raise ValueError()

        # 3. Fraud detection (Business logic)
        fraud_score = self._detect_fraud(transaction)

        # 4. Database (Infrastructure)
        self._save_to_db(transaction)

        # 5. Notification (External service)
        self._send_email(transaction)

        # This class has 5 reasons to change!

# ✅ GOOD: Separated responsibilities
class TransactionController:  # HTTP responsibility
    def handle_request(self, request):
        command = self._parse_request(request)
        result = await self._use_case.execute(command)
        return self._to_response(result)

class ProcessTransactionUseCase:  # Business logic
    def execute(self, command):
        transaction = self._factory.create(command)
        fraud_result = await self._fraud_service.analyze(transaction)
        await self._repo.save(transaction)
        await self._notifier.notify(transaction)

class PostgreSQLRepository:  # Database responsibility
    def save(self, transaction):
        # Only database logic
        ...

class EmailNotifier:  # Notification responsibility
    def notify(self, transaction):
        # Only email logic
        ...
```

### 6.2 Open/Closed Principle (OCP)

**Definition:** Open for extension, closed for modification

**Example: Strategy Pattern**

```python
# ✅ GOOD: Adding new fraud strategy doesn't modify existing code
class VelocityBasedScoring(FraudScoringStrategy):
    """New strategy - NO changes to existing strategies!"""
    async def calculate_score(self, txn, context):
        # New algorithm here
        ...

# Usage - no changes to use case code
strategy = VelocityBasedScoring()  # Just swap the strategy
fraud_service = FraudDetectionServiceImpl(strategy)
```

### 6.3 Liskov Substitution Principle (LSP)

**Definition:** Subtypes must be substitutable for their base types

**Example:**

```python
# All repositories are interchangeable
repo: TransactionRepository

# Can be PostgreSQL
repo = PostgreSQLTransactionRepository(pool)

# Can be Redis
repo = RedisTransactionRepository(redis_client)

# Can be InMemory (for tests)
repo = InMemoryTransactionRepository()

# Can be Composite
repo = CompositeTransactionRepository(primary_repo, cache_repo)

# Use case doesn't care - it works with ALL implementations
use_case = ProcessTransactionUseCase(repo, ...)
```

### 6.4 Interface Segregation Principle (ISP)

**Definition:** Clients shouldn't depend on interfaces they don't use

**Example:**

```python
# ❌ BAD: Fat interface
class TransactionRepository(ABC):
    @abstractmethod
    def save(self, txn): pass
    @abstractmethod
    def find_by_id(self, id): pass
    @abstractmethod
    def find_by_customer(self, customer_id): pass
    @abstractmethod
    def find_by_date_range(self, start, end): pass
    @abstractmethod
    def count_by_customer(self, customer_id): pass
    @abstractmethod
    def aggregate_by_merchant(self, merchant_id): pass
    # 20 more methods...

# ✅ GOOD: Segregated interfaces
class TransactionWriter(ABC):
    @abstractmethod
    def save(self, txn): pass

class TransactionReader(ABC):
    @abstractmethod
    def find_by_id(self, id): pass

class TransactionQuery(ABC):
    @abstractmethod
    def find_by_customer(self, customer_id): pass

class TransactionAnalytics(ABC):
    @abstractmethod
    def aggregate_by_merchant(self, merchant_id): pass

# Use cases depend only on what they need
class ProcessTransactionUseCase:
    def __init__(self, writer: TransactionWriter):
        self._writer = writer  # Only needs writing

class GetTransactionUseCase:
    def __init__(self, reader: TransactionReader):
        self._reader = reader  # Only needs reading
```

### 6.5 Dependency Inversion Principle (DIP)

**Definition:** Depend on abstractions, not concretions

**Example:**

```python
# ❌ BAD: High-level depends on low-level
class ProcessTransactionUseCase:
    def __init__(self):
        self._repo = PostgreSQLRepository()  # Concrete!
        self._ml_model = TensorFlowModel()  # Concrete!

# ✅ GOOD: High-level depends on abstraction
class ProcessTransactionUseCase:
    def __init__(
        self,
        repo: TransactionRepository,        # Interface!
        fraud_service: FraudDetectionService  # Interface!
    ):
        self._repo = repo
        self._fraud_service = fraud_service

# Dependency Injection (composition root)
def create_use_case():
    repo = PostgreSQLRepository(pool)  # Or RedisRepository()
    ml_gateway = TensorFlowGateway(model)  # Or SklearnGateway()
    fraud_service = FraudDetectionServiceImpl(ml_gateway)

    return ProcessTransactionUseCase(repo, fraud_service)
```

**Benefits of DIP:**
- ✅ Testable: Inject mock repositories
- ✅ Flexible: Swap implementations easily
- ✅ Maintainable: Changes to infrastructure don't affect domain

---

## 7. PERFORMANCE OPTIMIZATION

### 7.1 Big O Analysis

**Current Problems:**

```python
# ❌ PROBLEM 1: N+1 Query Antipattern - O(3n)
@app.route("/api/transactions")
def get_transactions():
    cursor.execute("SELECT * FROM transactions LIMIT 1000")
    transactions = cursor.fetchall()

    results = []
    for txn in transactions:  # O(n)
        # Query 1: Customer - O(1) but executed n times = O(n)
        cursor.execute("SELECT * FROM customers WHERE cpf = %s", (txn['cpf'],))
        customer = cursor.fetchone()

        # Query 2: Fraud detection - O(1) but executed n times = O(n)
        cursor.execute("SELECT * FROM fraud_detections WHERE txn_id = %s", (txn['id'],))
        fraud = cursor.fetchone()

        results.append({**txn, "customer": customer, "fraud": fraud})

    return jsonify(results)

# Total: 1 + n + n = 1 + 2n queries
# For 1000 transactions: 2001 queries! 😱
```

**Solution:**

```python
# ✅ SOLUTION: Single query with JOINs - O(1)
async def get_transactions_with_related(limit: int = 1000):
    query = """
        SELECT
            t.*,
            c.name, c.email,
            f.fraud_probability, f.risk_level
        FROM transactions t
        LEFT JOIN customers c ON t.cpf = c.cpf
        LEFT JOIN fraud_detections f ON t.id = f.transaction_id
        ORDER BY t.created_at DESC
        LIMIT $1
    """
    rows = await db.fetch_all(query, (limit,))
    return [TransactionWithRelated.from_row(row) for row in rows]

# Total: 1 query for 1000 transactions! ✅
```

**Problem 2: Cache Stampede**

```python
# ❌ PROBLEM: Cache stampede - O(n * f) where f is expensive operation
async def get_fraud_analysis(cpf: str):
    cached = await redis.get(f"fraud:{cpf}")
    if cached:
        return cached

    # If cache expires, ALL concurrent requests execute expensive operation!
    result = await expensive_ml_prediction(cpf)  # 500ms
    await redis.set(f"fraud:{cpf}", result, ex=300)
    return result

# Under load: 1000 concurrent requests → 1000 ML predictions (500s total)
```

**Solution:**

```python
# ✅ SOLUTION: Distributed lock (Redlock pattern) - O(f) once
async def get_fraud_analysis_safe(cpf: str):
    cache_key = f"fraud:{cpf}"
    lock_key = f"lock:fraud:{cpf}"

    # Check cache
    cached = await redis.get(cache_key)
    if cached:
        return cached

    # Try to acquire lock (SETNX with TTL)
    lock_acquired = await redis.set(lock_key, "1", ex=10, nx=True)

    if lock_acquired:
        try:
            # Only ONE request executes expensive operation
            result = await expensive_ml_prediction(cpf)
            await redis.set(cache_key, result, ex=300)
            return result
        finally:
            await redis.delete(lock_key)
    else:
        # Other requests wait briefly and retry
        await asyncio.sleep(0.1)
        return await get_fraud_analysis_safe(cpf)

# Under load: 1 ML prediction (500ms), others wait
```

### 7.2 Database Indexes

**Required indexes:**

```sql
-- Primary keys (B-tree)
CREATE UNIQUE INDEX idx_transactions_pk ON transactions(id);
CREATE UNIQUE INDEX idx_customers_pk ON customers(id);

-- Foreign keys (for JOINs) - CRITICAL for O(log n) joins
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_fraud_detections_transaction ON fraud_detections(transaction_id);

-- Time-based queries (range scans)
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);

-- Composite index for velocity checks
CREATE INDEX idx_transactions_customer_timestamp
ON transactions(customer_id, timestamp DESC);

-- Partial index for fraud investigations
CREATE INDEX idx_transactions_fraud
ON transactions(id)
WHERE risk_level IN ('high', 'critical');
```

---

## 8. SECURITY ARCHITECTURE

### 8.1 Defense in Depth

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Network (AWS WAF, CloudFlare, Rate Limiting) │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│  Layer 2: Application (Authentication, Authorization)   │
│  - JWT validation                                       │
│  - RBAC permissions                                     │
│  - CSRF protection                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│  Layer 3: Input Validation (Pydantic, SQL injection)    │
│  - Pydantic schemas                                     │
│  - SQL parameterization                                 │
│  - XSS prevention                                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│  Layer 4: Business Logic (Domain validation)            │
│  - Value Objects (CPF, Email)                           │
│  - Business rules                                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│  Layer 5: Data (Encryption, PII masking, LGPD)         │
│  - Encryption at rest                                   │
│  - PII sanitization in logs                             │
│  - LGPD compliance                                      │
└─────────────────────────────────────────────────────────┘
```

### 8.2 LGPD Compliance

**Implemented:**
- ✅ PII sanitization in logs (log_sanitizer.py)
- ✅ DSR endpoints (Data Subject Rights)
- ✅ Retention policies
- ✅ Right to be forgotten

**See:** [RELATORIO_100_PERCENT_REAL.md](RELATORIO_100_PERCENT_REAL.md) Section "CONFORMIDADE LGPD"

---

## 9. TESTING STRATEGY

### 9.1 Test Pyramid

```
        ┌───────────────┐
        │      E2E      │  ~10% (slow, brittle)
        │    (5 tests)  │
        ├───────────────┤
        │  Integration  │  ~20% (medium speed)
        │   (20 tests)  │
        ├───────────────┤
        │     Unit      │  ~70% (fast, reliable)
        │  (100 tests)  │
        └───────────────┘
```

### 9.2 Unit Tests (Domain Layer)

**Example:**

```python
# backend/tests/unit/test_core/test_value_objects.py
class TestCPF:
    def test_create_cpf_valid(self):
        cpf = CPF("12345678909")
        assert cpf.value == "12345678909"

    def test_create_cpf_invalid_raises(self):
        with pytest.raises(ValueError, match="CPF inválido"):
            CPF("00000000000")  # Blacklisted

    def test_cpf_from_raw_removes_formatting(self):
        cpf = CPF.from_raw("123.456.789-09")
        assert cpf.value == "12345678909"

    def test_cpf_masked_hides_digits(self):
        cpf = CPF("12345678909")
        assert cpf.masked() == "***.***.*89-09"

# backend/tests/unit/test_core/test_fraud_strategies.py
class TestRuleBasedScoring:
    @pytest.mark.asyncio
    async def test_high_value_increases_score(self):
        strategy = RuleBasedScoring()

        # Low value transaction
        low_txn = create_test_transaction(amount=100)
        result_low = await strategy.calculate_score(low_txn, {})

        # High value transaction
        high_txn = create_test_transaction(amount=10000)
        result_high = await strategy.calculate_score(high_txn, {})

        assert result_high.score.value > result_low.score.value
        assert "high_value_transaction" in result_high.risk_factors
```

### 9.3 Integration Tests

**Example:**

```python
# backend/tests/integration/test_repositories.py
class TestPostgreSQLRepository:
    @pytest.mark.asyncio
    async def test_save_and_find_transaction(self, db_pool):
        repo = PostgreSQLTransactionRepository(db_pool)

        # Create transaction
        txn = TransactionFactory.create_transaction(
            amount=Decimal("1000"),
            currency="BRL",
            merchant_id="M123",
            customer_id="C456"
        )

        # Save
        await repo.save(txn)

        # Find
        found_txn = await repo.find_by_id(txn.id)

        assert found_txn is not None
        assert found_txn.id == txn.id
        assert found_txn.amount.amount == Decimal("1000")
```

---

## 10. MIGRATION PLAN

### Phase 1: Foundation (Week 1)

**Goal:** Add new architecture alongside existing code

**Tasks:**
1. ✅ Create [core/value_objects.py](backend/core/value_objects.py)
2. ✅ Create [core/fraud_strategies.py](backend/core/fraud_strategies.py)
3. ✅ Create [core/decorators.py](backend/core/decorators.py)
4. Create adapters for ML engine
5. Write unit tests

**No changes to production_api.py yet!**

### Phase 2: Pilot Endpoint (Week 2)

**Goal:** Migrate ONE endpoint to prove architecture works

**Tasks:**
1. Create `/api/v2/predict` using new architecture
2. Run both v1 and v2 in parallel
3. Compare results (shadow mode)
4. Monitor metrics

```python
# New endpoint using Clean Architecture
@app.route('/api/v2/predict', methods=['POST'])
async def predict_v2():
    # 1. Parse command
    command = ProcessTransactionCommand(**request.json)

    # 2. Execute use case
    result = await process_transaction_use_case.execute(command)

    # 3. Return response
    return jsonify(result)
```

### Phase 3: Incremental Migration (Week 3-6)

**Goal:** Migrate 10-20 endpoints per week

**Strategy:**
1. Strangler Fig Pattern: New endpoints use new architecture
2. Gradually redirect traffic from old to new
3. Delete old code once traffic at 0%

**Metrics to track:**
- % of endpoints migrated
- Test coverage
- Performance (p95 latency)
- Error rate

### Phase 4: Production Cutover (Week 7-8)

**Goal:** 100% traffic on new architecture

**Tasks:**
1. Feature flag: Redirect 100% traffic
2. Monitor for 1 week
3. Delete old production_api.py
4. Celebrate! 🎉

---

## APPENDIX A: File Structure

```
backend/
├── core/                          # Domain Layer
│   ├── entities.py               # Business entities ✅
│   ├── value_objects.py          # Value Objects (CPF, Email) ✅ NEW
│   ├── interfaces.py             # Port interfaces ✅
│   ├── use_cases.py              # Use case orchestration ✅
│   ├── fraud_strategies.py       # Strategy Pattern ✅ NEW
│   └── decorators.py             # Decorator Pattern ✅ NEW
│
├── infrastructure/               # Infrastructure Layer
│   ├── repositories.py           # Repository implementations ✅
│   ├── database.py               # Database setup ✅
│   ├── ml_service.py             # ML Gateway adapter
│   └── redis_cluster.py          # Redis setup
│
├── api/                          # Presentation Layer
│   ├── routes/                   # HTTP routes
│   │   ├── fraud.py              # Fraud endpoints
│   │   ├── admin.py              # Admin endpoints
│   │   └── dsr.py                # DSR endpoints ✅
│   ├── production_api.py         # [TO BE MIGRATED]
│   └── schemas.py                # Pydantic schemas ✅
│
└── tests/
    ├── unit/
    │   ├── test_core/
    │   │   ├── test_entities.py
    │   │   ├── test_value_objects.py    # NEW
    │   │   └── test_fraud_strategies.py # NEW
    │   └── test_ml_engine/
    └── integration/
        └── test_repositories.py
```

---

## APPENDIX B: Key Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Code Quality** | | | |
| Cyclomatic Complexity (avg) | 15-25 | <10 | 🔴 |
| Lines per file (max) | 5135 | <500 | 🔴 |
| Test Coverage | 60% | 90% | 🟡 |
| Duplication (CPF validation) | 15 places | 1 place | 🔴 |
| **Performance** | | | |
| N+1 Queries | Yes | No | 🔴 |
| Cache hit rate | N/A | >80% | - |
| P95 latency /api/predict | 150ms | <50ms | 🟡 |
| **Architecture** | | | |
| Dependency on Flask | High | Low | 🔴 |
| Testability (without mocks) | 20% | 90% | 🔴 |
| Coupling (domain ↔ infra) | High | None | 🔴 |

---

**Generated with Claude Code**
**Architecture Guide Version:** 1.0
**Last Updated:** 2025-12-11
