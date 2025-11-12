"""
Clean Architecture API Layer
Implements Clean Architecture principles with proper dependency injection
Follows SOLID principles and uses design patterns
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import structlog
import asyncpg
import redis.asyncio as redis

# Core domain imports
from ..core.entities import TransactionId
from ..core.interfaces import (
    ProcessTransactionCommand,
    ApproveTransactionCommand,
    RejectTransactionCommand,
    GetTransactionQuery,
    GetFraudStatisticsQuery,
)
from ..core.use_cases import (
    ProcessTransactionUseCase,
    ApproveTransactionUseCase,
    GetTransactionUseCase,
    GetFraudStatisticsUseCase,
    TransactionCommandHandler,
    TransactionQueryHandler,
)

# Infrastructure imports
from ..infrastructure.repositories import RepositoryFactory
from ..infrastructure.ml_service import MLServiceFactory

# Configure structured logging
logger = structlog.get_logger(__name__)


class CleanArchitectureAPI:
    """
    Main API class implementing Clean Architecture
    Follows Dependency Injection and Inversion of Control patterns
    Time Complexity: O(1) for initialization, varies by endpoint
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = Flask(__name__)
        self._setup_cors()
        self._setup_logging()

        # Infrastructure dependencies (will be injected)
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None

        # Use case dependencies (will be injected)
        self.command_handler: Optional[TransactionCommandHandler] = None
        self.query_handler: Optional[TransactionQueryHandler] = None

        # Register routes
        self._register_routes()

    def _setup_cors(self) -> None:
        """Setup CORS configuration - O(1)"""
        allowed_origins = self.config.get("cors_origins", ["http://localhost:3000"])
        CORS(self.app, origins=allowed_origins)

    def _setup_logging(self) -> None:
        """Setup structured logging - O(1)"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    async def initialize_dependencies(self) -> None:
        """
        Initialize all dependencies using Dependency Injection
        Time Complexity: O(1) for setup, O(log n) for database connections
        """
        logger.info("Initializing Clean Architecture dependencies")

        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            host=self.config.get("db_host", "localhost"),
            port=self.config.get("db_port", 5432),
            database=self.config.get("db_name", "sankofa"),
            user=self.config.get("db_user", "postgres"),
            password=self.config.get("db_password", ""),
            min_size=5,
            max_size=20,
        )

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
            password=self.config.get("redis_password"),
            decode_responses=True,
        )

        # Create repositories using Factory Pattern
        transaction_repo = RepositoryFactory.create_transaction_repository(
            self.db_pool, self.redis_client, use_cache=True
        )
        customer_repo = RepositoryFactory.create_customer_repository(self.db_pool)
        event_store = RepositoryFactory.create_event_store(self.db_pool)

        # Create ML service using Factory Pattern
        fraud_service = MLServiceFactory.create_fraud_service("ensemble")

        # Create mock services (in real implementation, these would be proper services)
        notification_service = MockNotificationService()
        audit_service = MockAuditService()
        cache_service = MockCacheService(self.redis_client)
        event_publisher = MockEventPublisher()
        metrics_collector = MockMetricsCollector()

        # Create use cases with dependency injection
        process_use_case = ProcessTransactionUseCase(
            transaction_repo=transaction_repo,
            customer_repo=customer_repo,
            fraud_service=fraud_service,
            notification_service=notification_service,
            audit_service=audit_service,
            cache_service=cache_service,
            event_publisher=event_publisher,
            metrics_collector=metrics_collector,
        )

        approve_use_case = ApproveTransactionUseCase(
            transaction_repo=transaction_repo,
            audit_service=audit_service,
            notification_service=notification_service,
            event_publisher=event_publisher,
            metrics_collector=metrics_collector,
        )

        get_transaction_use_case = GetTransactionUseCase(
            transaction_repo=transaction_repo, cache_service=cache_service
        )

        get_fraud_stats_use_case = GetFraudStatisticsUseCase(
            transaction_repo=transaction_repo, cache_service=cache_service
        )

        # Create command and query handlers (CQRS pattern)
        self.command_handler = TransactionCommandHandler(
            process_use_case=process_use_case, approve_use_case=approve_use_case
        )

        self.query_handler = TransactionQueryHandler(
            get_transaction_use_case=get_transaction_use_case,
            get_fraud_stats_use_case=get_fraud_stats_use_case,
        )

        logger.info("Clean Architecture dependencies initialized successfully")

    def _register_routes(self) -> None:
        """Register API routes - O(1)"""

        # Middleware for request timing
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = f"req_{int(time.time() * 1000)}"
            logger.info("Request started", request_id=g.request_id, path=request.path)

        @self.app.after_request
        def after_request(response):
            duration = time.time() - g.start_time
            logger.info(
                "Request completed",
                request_id=g.request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )
            response.headers["X-Request-ID"] = g.request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            return response

        # Health check endpoint
        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """Health check endpoint - O(1)"""
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "architecture": "clean_architecture",
                }
            )

        # Command endpoints (CQRS Write side)
        @self.app.route("/api/v1/transactions", methods=["POST"])
        def process_transaction():
            """
            Process new transaction - Command
            Time Complexity: O(f + log n) where f is ML inference time
            """
            try:
                data = request.get_json()

                # Input validation
                if not data:
                    return jsonify({"error": "Request body is required"}), 400

                required_fields = ["amount", "merchant_id", "customer_id"]
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Field '{field}' is required"}), 400

                # Create command
                command = ProcessTransactionCommand(
                    amount=float(data["amount"]),
                    currency=data.get("currency", "BRL"),
                    merchant_id=data["merchant_id"],
                    customer_id=data["customer_id"],
                    metadata=data.get("metadata", {}),
                )

                # Execute command asynchronously
                result = asyncio.run(self.command_handler.handle(command))

                return jsonify({"success": True, "data": result, "request_id": g.request_id})

            except ValueError as e:
                logger.warning("Validation error", error=str(e), request_id=g.request_id)
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error("Internal error", error=str(e), request_id=g.request_id)
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/v1/transactions/<transaction_id>/approve", methods=["POST"])
        def approve_transaction(transaction_id: str):
            """
            Approve transaction - Command
            Time Complexity: O(log n)
            """
            try:
                data = request.get_json() or {}
                approved_by = data.get("approved_by", "system")

                # Create command
                command = ApproveTransactionCommand(
                    transaction_id=transaction_id, approved_by=approved_by
                )

                # Execute command
                result = asyncio.run(self.command_handler.handle(command))

                return jsonify({"success": True, "data": result, "request_id": g.request_id})

            except ValueError as e:
                logger.warning("Validation error", error=str(e), request_id=g.request_id)
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error("Internal error", error=str(e), request_id=g.request_id)
                return jsonify({"error": "Internal server error"}), 500

        # Query endpoints (CQRS Read side)
        @self.app.route("/api/v1/transactions/<transaction_id>", methods=["GET"])
        def get_transaction(transaction_id: str):
            """
            Get transaction by ID - Query
            Time Complexity: O(1) cache hit, O(log n) cache miss
            """
            try:
                # Create query
                query = GetTransactionQuery(transaction_id)

                # Execute query
                result = asyncio.run(self.query_handler.handle(query))

                if result is None:
                    return jsonify({"error": "Transaction not found"}), 404

                return jsonify({"success": True, "data": result, "request_id": g.request_id})

            except Exception as e:
                logger.error("Internal error", error=str(e), request_id=g.request_id)
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/v1/fraud-statistics", methods=["GET"])
        def get_fraud_statistics():
            """
            Get fraud statistics - Query
            Time Complexity: O(log n + k) where k is result size
            """
            try:
                # Parse query parameters
                start_date_str = request.args.get("start_date")
                end_date_str = request.args.get("end_date")

                if not start_date_str or not end_date_str:
                    return (
                        jsonify({"error": "start_date and end_date parameters are required"}),
                        400,
                    )

                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)

                # Create query
                query = GetFraudStatisticsQuery(start_date, end_date)

                # Execute query
                result = asyncio.run(self.query_handler.handle(query))

                return jsonify({"success": True, "data": result, "request_id": g.request_id})

            except ValueError as e:
                logger.warning("Validation error", error=str(e), request_id=g.request_id)
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error("Internal error", error=str(e), request_id=g.request_id)
                return jsonify({"error": "Internal server error"}), 500

        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Endpoint not found"}), 404

        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({"error": "Method not allowed"}), 405

        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error("Unhandled internal error", error=str(error))
            return jsonify({"error": "Internal server error"}), 500

    async def shutdown(self) -> None:
        """Cleanup resources - O(1)"""
        logger.info("Shutting down Clean Architecture API")

        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            await self.redis_client.close()


# Mock implementations for demonstration
# In a real application, these would be proper implementations


class MockNotificationService:
    """Mock notification service"""

    async def send_fraud_alert(self, transaction, analysis):
        logger.info("Fraud alert sent", transaction_id=transaction.id.value)

    async def send_approval_notification(self, transaction):
        logger.info("Approval notification sent", transaction_id=transaction.id.value)


class MockAuditService:
    """Mock audit service"""

    async def log_transaction_event(self, transaction_id, event_type, details):
        logger.info(
            "Audit event logged",
            transaction_id=(
                transaction_id.value if hasattr(transaction_id, "value") else str(transaction_id)
            ),
            event_type=event_type,
            details=details,
        )

    async def log_fraud_detection(self, transaction, analysis):
        logger.info(
            "Fraud detection logged",
            transaction_id=transaction.id.value,
            confidence_score=analysis.confidence_score,
        )


class MockCacheService:
    """Mock cache service using Redis"""

    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def get(self, key: str):
        try:
            return await self.redis_client.get(key)
        except:
            return None

    async def set(self, key: str, value, ttl: int = 3600):
        try:
            await self.redis_client.setex(key, ttl, str(value))
        except:
            pass

    async def delete(self, key: str):
        try:
            await self.redis_client.delete(key)
        except:
            pass

    async def exists(self, key: str) -> bool:
        try:
            return await self.redis_client.exists(key)
        except:
            return False


class MockEventPublisher:
    """Mock event publisher"""

    async def publish(self, event):
        logger.info("Event published", event_type=event.event_type(), event_id=str(event.event_id))

    async def publish_batch(self, events):
        for event in events:
            await self.publish(event)


class MockMetricsCollector:
    """Mock metrics collector"""

    def increment_counter(self, metric_name: str, tags=None):
        logger.info("Counter incremented", metric=metric_name, tags=tags)

    def record_histogram(self, metric_name: str, value: float, tags=None):
        logger.info("Histogram recorded", metric=metric_name, value=value, tags=tags)

    def record_gauge(self, metric_name: str, value: float, tags=None):
        logger.info("Gauge recorded", metric=metric_name, value=value, tags=tags)


# Factory for creating the API
class APIFactory:
    """
    Factory for creating API instances
    Implements Factory Pattern
    """

    @staticmethod
    def create_api(config: Dict[str, Any]) -> CleanArchitectureAPI:
        """
        Create Clean Architecture API
        Time Complexity: O(1)
        """
        return CleanArchitectureAPI(config)

    @staticmethod
    def create_production_config() -> Dict[str, Any]:
        """
        Create production configuration
        Time Complexity: O(1)
        """
        return {
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "sankofa_production",
            "db_user": "sankofa_user",
            "db_password": "secure_password",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_password": "redis_password",
            "cors_origins": ["https://sankofa.example.com"],
            "log_level": "INFO",
        }

    @staticmethod
    def create_development_config() -> Dict[str, Any]:
        """
        Create development configuration
        Time Complexity: O(1)
        """
        return {
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "sankofa_dev",
            "db_user": "postgres",
            "db_password": "postgres",
            "redis_host": "localhost",
            "redis_port": 6379,
            "cors_origins": ["http://localhost:3000", "http://localhost:5173"],
            "log_level": "DEBUG",
        }


# Application entry point
async def create_app(config: Dict[str, Any]) -> CleanArchitectureAPI:
    """
    Application factory function
    Time Complexity: O(log n) due to database connections
    """
    api = APIFactory.create_api(config)
    await api.initialize_dependencies()
    return api


if __name__ == "__main__":
    # Development server
    config = APIFactory.create_development_config()

    async def main():
        api = await create_app(config)
        api.app.run(
            host="localhost",
            port=8445,
            debug=False,  # Never use debug=True in production
            threaded=True,
        )

    asyncio.run(main())
