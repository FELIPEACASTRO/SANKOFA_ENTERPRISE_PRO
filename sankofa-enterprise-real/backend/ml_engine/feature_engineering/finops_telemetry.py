"""
P0-002: FinOps Telemetry Module
Telemetria granular para otimizacao de custos.

Inclui:
- Cost-per-prediction tracking
- Feature compute cost attribution
- Model inference cost
- Cache hit/miss economics

Author: Sankofa Enterprise Pro
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
import time
import threading


class ServiceType(Enum):
    """Tipos de servicos rastreados"""
    ML_INFERENCE = "ml_inference"
    FEATURE_ENGINEERING = "feature_engineering"
    CACHE_OPERATION = "cache_operation"
    DATABASE_QUERY = "database_query"
    EXTERNAL_API = "external_api"
    RULES_ENGINE = "rules_engine"


class OperationType(Enum):
    """Tipos de operacoes"""
    PREDICT = "predict"
    PREDICT_PROBA = "predict_proba"
    BATCH_PREDICT = "batch_predict"
    FEATURE_COMPUTE = "feature_compute"
    CACHE_GET = "cache_get"
    CACHE_SET = "cache_set"
    DB_READ = "db_read"
    DB_WRITE = "db_write"
    API_CALL = "api_call"
    RULE_EVAL = "rule_eval"


@dataclass
class CostMetric:
    """Metrica de custo individual"""
    service: ServiceType
    operation: OperationType
    timestamp: datetime
    duration_ms: float
    cost_usd: float
    cache_hit: bool = False
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service.value,
            "operation": self.operation.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "cache_hit": self.cache_hit,
            "batch_size": self.batch_size,
            "metadata": self.metadata,
        }


@dataclass
class CostSummary:
    """Resumo de custos por periodo"""
    start_time: datetime
    end_time: datetime
    total_cost_usd: float
    total_operations: int
    avg_cost_per_operation: float
    cost_by_service: Dict[str, float]
    cost_by_operation: Dict[str, float]
    cache_savings_usd: float
    total_duration_ms: float
    avg_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_operations": self.total_operations,
            "avg_cost_per_operation": round(self.avg_cost_per_operation, 8),
            "cost_by_service": {k: round(v, 6) for k, v in self.cost_by_service.items()},
            "cost_by_operation": {k: round(v, 6) for k, v in self.cost_by_operation.items()},
            "cache_savings_usd": round(self.cache_savings_usd, 6),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class CostModel:
    """
    Modelo de custos AWS para operacoes de ML.
    Baseado em precos reais da AWS (us-east-1).
    """

    # Custos base por tipo de servico (USD por operacao)
    # Estes sao estimativas baseadas em:
    # - Lambda: $0.0000166667 por GB-segundo
    # - SageMaker Inference: ~$0.0001 por inference (ml.t2.medium)
    # - ElastiCache Redis: $0.000001 por operacao
    # - RDS PostgreSQL: $0.00001 por query

    BASE_COSTS = {
        ServiceType.ML_INFERENCE: {
            OperationType.PREDICT: 0.00015,
            OperationType.PREDICT_PROBA: 0.00015,
            OperationType.BATCH_PREDICT: 0.0001,  # Por item no batch
        },
        ServiceType.FEATURE_ENGINEERING: {
            OperationType.FEATURE_COMPUTE: 0.00005,
        },
        ServiceType.CACHE_OPERATION: {
            OperationType.CACHE_GET: 0.000001,
            OperationType.CACHE_SET: 0.000002,
        },
        ServiceType.DATABASE_QUERY: {
            OperationType.DB_READ: 0.00001,
            OperationType.DB_WRITE: 0.00002,
        },
        ServiceType.EXTERNAL_API: {
            OperationType.API_CALL: 0.0001,
        },
        ServiceType.RULES_ENGINE: {
            OperationType.RULE_EVAL: 0.00001,
        },
    }

    # Custo por milissegundo de compute (Lambda-based)
    COMPUTE_COST_PER_MS = 0.0000000167  # $0.0000166667 per GB-second / 1000

    # Economia estimada por cache hit
    CACHE_SAVINGS_MULTIPLIER = 0.9  # Cache hit economiza 90% do custo

    @classmethod
    def calculate_cost(
        cls,
        service: ServiceType,
        operation: OperationType,
        duration_ms: float,
        cache_hit: bool = False,
        batch_size: int = 1,
    ) -> float:
        """Calcula custo de uma operacao"""

        # Custo base
        base_cost = cls.BASE_COSTS.get(service, {}).get(operation, 0.0001)

        # Custo de compute
        compute_cost = duration_ms * cls.COMPUTE_COST_PER_MS

        # Custo total
        total_cost = (base_cost * batch_size) + compute_cost

        # Desconto por cache hit
        if cache_hit:
            total_cost *= (1 - cls.CACHE_SAVINGS_MULTIPLIER)

        return total_cost

    @classmethod
    def estimate_cache_savings(
        cls,
        service: ServiceType,
        operation: OperationType,
        cache_hits: int,
    ) -> float:
        """Estima economia por cache hits"""
        base_cost = cls.BASE_COSTS.get(service, {}).get(operation, 0.0001)
        return base_cost * cache_hits * cls.CACHE_SAVINGS_MULTIPLIER


class FinOpsTelemetry:
    """
    Coletor de telemetria de custos.
    Thread-safe para uso em producao.
    """

    VERSION = "1.0.0"

    def __init__(self, max_metrics: int = 100000):
        self._metrics: List[CostMetric] = []
        self._lock = threading.Lock()
        self._max_metrics = max_metrics

        # Contadores agregados (para resumos rapidos)
        self._total_cost = 0.0
        self._total_operations = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cost_by_service: Dict[str, float] = defaultdict(float)
        self._cost_by_operation: Dict[str, float] = defaultdict(float)
        self._total_duration_ms = 0.0

    def record(
        self,
        service: ServiceType,
        operation: OperationType,
        duration_ms: float,
        cache_hit: bool = False,
        batch_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostMetric:
        """Registra uma metrica de custo"""

        cost = CostModel.calculate_cost(
            service, operation, duration_ms, cache_hit, batch_size
        )

        metric = CostMetric(
            service=service,
            operation=operation,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            cost_usd=cost,
            cache_hit=cache_hit,
            batch_size=batch_size,
            metadata=metadata or {},
        )

        with self._lock:
            # Evitar crescimento infinito
            if len(self._metrics) >= self._max_metrics:
                # Remover metricas mais antigas (manter 50%)
                self._metrics = self._metrics[self._max_metrics // 2:]

            self._metrics.append(metric)

            # Atualizar agregados
            self._total_cost += cost
            self._total_operations += 1
            self._total_duration_ms += duration_ms

            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            self._cost_by_service[service.value] += cost
            self._cost_by_operation[operation.value] += cost

        return metric

    def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> CostSummary:
        """Retorna resumo de custos para um periodo"""

        with self._lock:
            if start_time is None:
                start_time = datetime.utcnow() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.utcnow()

            # Filtrar metricas pelo periodo
            filtered = [
                m for m in self._metrics
                if start_time <= m.timestamp <= end_time
            ]

            if not filtered:
                return CostSummary(
                    start_time=start_time,
                    end_time=end_time,
                    total_cost_usd=0.0,
                    total_operations=0,
                    avg_cost_per_operation=0.0,
                    cost_by_service={},
                    cost_by_operation={},
                    cache_savings_usd=0.0,
                    total_duration_ms=0.0,
                    avg_latency_ms=0.0,
                )

            total_cost = sum(m.cost_usd for m in filtered)
            total_duration = sum(m.duration_ms for m in filtered)
            total_ops = len(filtered)

            # Calcular por servico/operacao
            cost_by_service: Dict[str, float] = defaultdict(float)
            cost_by_operation: Dict[str, float] = defaultdict(float)
            cache_hits = 0

            for m in filtered:
                cost_by_service[m.service.value] += m.cost_usd
                cost_by_operation[m.operation.value] += m.cost_usd
                if m.cache_hit:
                    cache_hits += 1

            # Estimar economia por cache
            cache_savings = CostModel.estimate_cache_savings(
                ServiceType.ML_INFERENCE,
                OperationType.PREDICT,
                cache_hits,
            )

            return CostSummary(
                start_time=start_time,
                end_time=end_time,
                total_cost_usd=total_cost,
                total_operations=total_ops,
                avg_cost_per_operation=total_cost / total_ops if total_ops > 0 else 0,
                cost_by_service=dict(cost_by_service),
                cost_by_operation=dict(cost_by_operation),
                cache_savings_usd=cache_savings,
                total_duration_ms=total_duration,
                avg_latency_ms=total_duration / total_ops if total_ops > 0 else 0,
            )

    def get_cost_per_prediction(self) -> float:
        """Retorna custo medio por predicao"""
        with self._lock:
            ml_ops = [
                m for m in self._metrics
                if m.service == ServiceType.ML_INFERENCE
            ]
            if not ml_ops:
                return 0.0
            return sum(m.cost_usd for m in ml_ops) / len(ml_ops)

    def get_cache_hit_rate(self) -> float:
        """Retorna taxa de cache hit"""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            if total == 0:
                return 0.0
            return self._cache_hits / total

    def get_total_cost(self) -> float:
        """Retorna custo total acumulado"""
        with self._lock:
            return self._total_cost

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Retorna breakdown completo de custos"""
        with self._lock:
            # Calculate cache hit rate inline to avoid deadlock
            total_cache_ops = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

            # Calculate cost per prediction inline to avoid deadlock
            ml_ops = [
                m for m in self._metrics
                if m.service == ServiceType.ML_INFERENCE
            ]
            avg_cost_per_pred = sum(m.cost_usd for m in ml_ops) / len(ml_ops) if ml_ops else 0.0

            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_operations": self._total_operations,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "avg_cost_per_prediction": round(avg_cost_per_pred, 8),
                "cost_by_service": {k: round(v, 6) for k, v in self._cost_by_service.items()},
                "cost_by_operation": {k: round(v, 6) for k, v in self._cost_by_operation.items()},
                "avg_latency_ms": round(
                    self._total_duration_ms / self._total_operations
                    if self._total_operations > 0 else 0, 2
                ),
            }

    def reset(self) -> None:
        """Reseta todas as metricas"""
        with self._lock:
            self._metrics.clear()
            self._total_cost = 0.0
            self._total_operations = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._cost_by_service.clear()
            self._cost_by_operation.clear()
            self._total_duration_ms = 0.0


class FinOpsTracker:
    """Context manager para tracking de custos"""

    def __init__(
        self,
        telemetry: FinOpsTelemetry,
        service: ServiceType,
        operation: OperationType,
        batch_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.telemetry = telemetry
        self.service = service
        self.operation = operation
        self.batch_size = batch_size
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None
        self._cache_hit = False

    def set_cache_hit(self, cache_hit: bool) -> None:
        """Define se houve cache hit"""
        self._cache_hit = cache_hit

    def __enter__(self) -> "FinOpsTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self.telemetry.record(
                service=self.service,
                operation=self.operation,
                duration_ms=duration_ms,
                cache_hit=self._cache_hit,
                batch_size=self.batch_size,
                metadata=self.metadata,
            )


# Singleton global
_finops_telemetry: Optional[FinOpsTelemetry] = None
_telemetry_lock = threading.Lock()


def get_finops_telemetry() -> FinOpsTelemetry:
    """Retorna instancia singleton da telemetria"""
    global _finops_telemetry
    if _finops_telemetry is None:
        with _telemetry_lock:
            if _finops_telemetry is None:
                _finops_telemetry = FinOpsTelemetry()
    return _finops_telemetry


def track_cost(
    service: ServiceType,
    operation: OperationType,
    batch_size: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> FinOpsTracker:
    """Decorator/context manager para tracking de custos"""
    return FinOpsTracker(
        get_finops_telemetry(),
        service,
        operation,
        batch_size,
        metadata,
    )
