"""
Sankofa Enterprise Pro - Observability Module
Sistema de métricas e monitoramento para compliance BACEN

Funcionalidades:
- Métricas Prometheus-style (TPS, latência, erros)
- Alertas automáticos baseados em SLAs
- Health checks
- Dashboard de status em tempo real
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import statistics
import json
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class SLAConfig:
    """Configuração de SLAs para compliance BACEN"""
    max_latency_p95_ms: float = 100.0
    max_latency_p99_ms: float = 200.0
    min_availability_percent: float = 99.9
    max_error_rate_percent: float = 0.1
    min_tps: float = 100.0


@dataclass
class Alert:
    """Alerta do sistema"""
    id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: str
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


@dataclass
class HealthStatus:
    """Status de saúde do sistema"""
    status: str
    uptime_seconds: float
    last_check: str
    components: Dict[str, Dict[str, Any]]
    sla_compliance: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    Coletor de métricas Prometheus-style para observabilidade
    
    Funcionalidades:
    - Contadores (requests, erros, fraudes)
    - Gauges (conexões ativas, fila)
    - Histogramas (latência, valores)
    - Summaries (percentis)
    """
    
    def __init__(self, window_seconds: int = 300, max_samples: int = 10000):
        self._lock = threading.RLock()
        self._window_seconds = window_seconds
        self._max_samples = max_samples
        
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = {}
        
        self._start_time = time.time()
        self._last_reset = time.time()
        
        self._initialize_default_metrics()
        
        logger.info("MetricsCollector initialized", extra={
            "window_seconds": window_seconds,
            "max_samples": max_samples
        })
    
    def _initialize_default_metrics(self):
        """Inicializa métricas padrão"""
        self._counters = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "predictions_total": 0,
            "predictions_fraud": 0,
            "predictions_legitimate": 0,
            "explanations_generated": 0,
            "alerts_triggered": 0,
            "db_queries_total": 0,
            "db_queries_error": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self._gauges = {
            "active_connections": 0,
            "db_pool_size": 0,
            "db_pool_available": 0,
            "model_threshold": 0.5,
            "queue_size": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0
        }
        
        self._histograms = {
            "request_latency_ms": deque(maxlen=self._max_samples),
            "prediction_latency_ms": deque(maxlen=self._max_samples),
            "db_query_latency_ms": deque(maxlen=self._max_samples),
            "transaction_amount": deque(maxlen=self._max_samples),
            "risk_score": deque(maxlen=self._max_samples)
        }
    
    def increment(self, name: str, value: int = 1):
        """Incrementa um contador"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Define um gauge"""
        with self._lock:
            self._gauges[name] = value
    
    def observe(self, name: str, value: float):
        """Adiciona observação a um histograma"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = deque(maxlen=self._max_samples)
            self._histograms[name].append((time.time(), value))
    
    def get_counter(self, name: str) -> int:
        """Retorna valor de um contador"""
        with self._lock:
            return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Retorna valor de um gauge"""
        with self._lock:
            return self._gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str, window_seconds: Optional[int] = None) -> Dict[str, float]:
        """Retorna estatísticas de um histograma"""
        with self._lock:
            if name not in self._histograms:
                return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
            
            cutoff = time.time() - (window_seconds or self._window_seconds)
            values = [v for t, v in self._histograms[name] if t >= cutoff]
            
            if not values:
                return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            return {
                "count": n,
                "sum": sum(values),
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(n * 0.50)] if n > 0 else 0,
                "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
            }
    
    def get_tps(self, window_seconds: int = 60) -> float:
        """Calcula transações por segundo"""
        with self._lock:
            if "request_latency_ms" not in self._histograms:
                return 0.0
            
            cutoff = time.time() - window_seconds
            count = sum(1 for t, _ in self._histograms["request_latency_ms"] if t >= cutoff)
            return count / window_seconds if window_seconds > 0 else 0.0
    
    def get_error_rate(self, window_seconds: int = 300) -> float:
        """Calcula taxa de erro"""
        with self._lock:
            total = self._counters.get("requests_total", 0)
            errors = self._counters.get("requests_error", 0)
            
            if total == 0:
                return 0.0
            return (errors / total) * 100
    
    def get_fraud_rate(self) -> float:
        """Calcula taxa de fraude detectada"""
        with self._lock:
            total = self._counters.get("predictions_total", 0)
            frauds = self._counters.get("predictions_fraud", 0)
            
            if total == 0:
                return 0.0
            return (frauds / total) * 100
    
    def get_uptime_seconds(self) -> float:
        """Retorna uptime em segundos"""
        return time.time() - self._start_time
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Retorna todas as métricas"""
        with self._lock:
            latency_stats = self.get_histogram_stats("request_latency_ms", 60)
            prediction_stats = self.get_histogram_stats("prediction_latency_ms", 60)
            
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latency": latency_stats,
                "prediction_latency": prediction_stats,
                "tps": round(self.get_tps(), 2),
                "error_rate_percent": round(self.get_error_rate(), 4),
                "fraud_rate_percent": round(self.get_fraud_rate(), 4),
                "uptime_seconds": round(self.get_uptime_seconds(), 2),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            }
    
    def export_prometheus(self) -> str:
        """Exporta métricas em formato Prometheus"""
        lines = []
        
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"sankofa_{name} {value}")
            
            for name, value in self._gauges.items():
                lines.append(f"sankofa_{name} {value}")
            
            latency_stats = self.get_histogram_stats("request_latency_ms", 60)
            lines.append(f"sankofa_request_latency_ms_p50 {latency_stats['p50']}")
            lines.append(f"sankofa_request_latency_ms_p95 {latency_stats['p95']}")
            lines.append(f"sankofa_request_latency_ms_p99 {latency_stats['p99']}")
            
            lines.append(f"sankofa_tps {self.get_tps()}")
            lines.append(f"sankofa_error_rate {self.get_error_rate()}")
            lines.append(f"sankofa_uptime_seconds {self.get_uptime_seconds()}")
        
        return "\n".join(lines)


class AlertManager:
    """
    Gerenciador de alertas baseado em SLAs
    
    Funcionalidades:
    - Verificação automática de SLAs
    - Alertas por threshold
    - Histórico de alertas
    - Acknowledgment e resolução
    """
    
    def __init__(self, metrics: MetricsCollector, sla_config: Optional[SLAConfig] = None):
        self._metrics = metrics
        self._sla = sla_config or SLAConfig()
        self._alerts: List[Alert] = []
        self._lock = threading.RLock()
        self._alert_counter = 0
        self._check_interval = 30
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        logger.info("AlertManager initialized", extra={
            "sla_config": asdict(self._sla)
        })
    
    def start(self):
        """Inicia verificação automática de SLAs"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._check_thread.start()
        logger.info("AlertManager started")
    
    def stop(self):
        """Para verificação automática"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("AlertManager stopped")
    
    def _check_loop(self):
        """Loop de verificação de SLAs"""
        while self._running:
            try:
                self._check_slas()
            except Exception as e:
                logger.error(f"SLA check failed: {e}")
            
            time.sleep(self._check_interval)
    
    def _check_slas(self):
        """Verifica todos os SLAs"""
        latency_stats = self._metrics.get_histogram_stats("request_latency_ms", 60)
        
        if latency_stats["p95"] > self._sla.max_latency_p95_ms and latency_stats["count"] > 10:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                message=f"Latência P95 acima do SLA: {latency_stats['p95']:.1f}ms > {self._sla.max_latency_p95_ms}ms",
                metric_name="latency_p95_ms",
                current_value=latency_stats["p95"],
                threshold_value=self._sla.max_latency_p95_ms
            )
        
        if latency_stats["p99"] > self._sla.max_latency_p99_ms and latency_stats["count"] > 10:
            self._create_alert(
                severity=AlertSeverity.ERROR,
                message=f"Latência P99 acima do SLA: {latency_stats['p99']:.1f}ms > {self._sla.max_latency_p99_ms}ms",
                metric_name="latency_p99_ms",
                current_value=latency_stats["p99"],
                threshold_value=self._sla.max_latency_p99_ms
            )
        
        error_rate = self._metrics.get_error_rate()
        if error_rate > self._sla.max_error_rate_percent:
            self._create_alert(
                severity=AlertSeverity.ERROR,
                message=f"Taxa de erro acima do SLA: {error_rate:.2f}% > {self._sla.max_error_rate_percent}%",
                metric_name="error_rate_percent",
                current_value=error_rate,
                threshold_value=self._sla.max_error_rate_percent
            )
        
        tps = self._metrics.get_tps()
        if tps < self._sla.min_tps and self._metrics.get_counter("requests_total") > 100:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                message=f"TPS abaixo do esperado: {tps:.1f} < {self._sla.min_tps}",
                metric_name="tps",
                current_value=tps,
                threshold_value=self._sla.min_tps
            )
    
    def _create_alert(self, severity: AlertSeverity, message: str, metric_name: str,
                      current_value: float, threshold_value: float):
        """Cria novo alerta"""
        with self._lock:
            recent_similar = [
                a for a in self._alerts[-10:]
                if a.metric_name == metric_name and not a.resolved
                and (datetime.now(timezone.utc) - datetime.fromisoformat(a.timestamp.replace("Z", ""))) < timedelta(minutes=5)
            ]
            if recent_similar:
                return
            
            self._alert_counter += 1
            alert = Alert(
                id=f"ALERT_{self._alert_counter:06d}",
                severity=severity,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                timestamp=datetime.now(timezone.utc).isoformat() + "Z"
            )
            
            self._alerts.append(alert)
            self._metrics.increment("alerts_triggered")
            
            logger.warning(f"Alert created: {alert.id} - {message}", extra={
                "alert_id": alert.id,
                "severity": severity.value,
                "metric": metric_name
            })
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retorna alertas ativos"""
        with self._lock:
            return [a.to_dict() for a in self._alerts if not a.resolved]
    
    def get_all_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de alertas"""
        with self._lock:
            return [a.to_dict() for a in self._alerts[-limit:]]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconhece um alerta"""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve um alerta"""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    return True
            return False
    
    def get_sla_compliance(self) -> Dict[str, Any]:
        """Retorna status de compliance dos SLAs"""
        latency_stats = self._metrics.get_histogram_stats("request_latency_ms", 60)
        error_rate = self._metrics.get_error_rate()
        tps = self._metrics.get_tps()
        
        return {
            "latency_p95": {
                "compliant": latency_stats["p95"] <= self._sla.max_latency_p95_ms,
                "current": round(latency_stats["p95"], 2),
                "threshold": self._sla.max_latency_p95_ms,
                "unit": "ms"
            },
            "latency_p99": {
                "compliant": latency_stats["p99"] <= self._sla.max_latency_p99_ms,
                "current": round(latency_stats["p99"], 2),
                "threshold": self._sla.max_latency_p99_ms,
                "unit": "ms"
            },
            "error_rate": {
                "compliant": error_rate <= self._sla.max_error_rate_percent,
                "current": round(error_rate, 4),
                "threshold": self._sla.max_error_rate_percent,
                "unit": "%"
            },
            "tps": {
                "compliant": tps >= self._sla.min_tps or self._metrics.get_counter("requests_total") < 100,
                "current": round(tps, 2),
                "threshold": self._sla.min_tps,
                "unit": "req/s"
            }
        }


class HealthChecker:
    """
    Verificador de saúde do sistema
    
    Componentes monitorados:
    - API (response time)
    - Modelo ML (trained status)
    - Banco de dados (connection)
    - Cache (availability)
    """
    
    def __init__(self, metrics: MetricsCollector):
        self._metrics = metrics
        self._start_time = time.time()
        self._component_checks: Dict[str, Callable[[], bool]] = {}
        
        logger.info("HealthChecker initialized")
    
    def register_component(self, name: str, check_fn: Callable[[], bool]):
        """Registra componente para verificação"""
        self._component_checks[name] = check_fn
        logger.info(f"Component registered for health check: {name}")
    
    def check_all(self) -> HealthStatus:
        """Verifica todos os componentes"""
        components = {}
        all_healthy = True
        
        for name, check_fn in self._component_checks.items():
            try:
                start = time.time()
                is_healthy = check_fn()
                latency = (time.time() - start) * 1000
                
                components[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "latency_ms": round(latency, 2),
                    "last_check": datetime.now(timezone.utc).isoformat() + "Z"
                }
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                components[name] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": datetime.now(timezone.utc).isoformat() + "Z"
                }
                all_healthy = False
        
        latency_stats = self._metrics.get_histogram_stats("request_latency_ms", 60)
        sla_compliance = {
            "latency_p95_ok": latency_stats["p95"] <= 100,
            "latency_p99_ok": latency_stats["p99"] <= 200,
            "error_rate_ok": self._metrics.get_error_rate() <= 0.1
        }
        
        status = "healthy" if all_healthy else "degraded"
        if any(c["status"] == "error" for c in components.values()):
            status = "unhealthy"
        
        return HealthStatus(
            status=status,
            uptime_seconds=time.time() - self._start_time,
            last_check=datetime.now(timezone.utc).isoformat() + "Z",
            components=components,
            sla_compliance=sla_compliance
        )
    
    def get_readiness(self) -> Dict[str, Any]:
        """Verifica se sistema está pronto para receber requisições"""
        health = self.check_all()
        return {
            "ready": health.status != "unhealthy",
            "status": health.status,
            "components_ok": sum(1 for c in health.components.values() if c["status"] == "healthy"),
            "components_total": len(health.components)
        }
    
    def get_liveness(self) -> Dict[str, Any]:
        """Verifica se sistema está vivo (para Kubernetes)"""
        return {
            "alive": True,
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }


observability_metrics = MetricsCollector()
alert_manager = AlertManager(observability_metrics)
health_checker = HealthChecker(observability_metrics)


def start_observability():
    """Inicia sistema de observabilidade"""
    alert_manager.start()
    logger.info("Observability system started")


def stop_observability():
    """Para sistema de observabilidade"""
    alert_manager.stop()
    logger.info("Observability system stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Observability Module")
    
    for i in range(100):
        observability_metrics.increment("requests_total")
        observability_metrics.observe("request_latency_ms", 10 + (i % 50))
        if i % 10 == 0:
            observability_metrics.increment("predictions_fraud")
        else:
            observability_metrics.increment("predictions_legitimate")
        observability_metrics.increment("predictions_total")
    
    print("\n=== All Metrics ===")
    metrics = observability_metrics.get_all_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n=== Prometheus Export ===")
    print(observability_metrics.export_prometheus())
    
    print("\n=== SLA Compliance ===")
    sla = alert_manager.get_sla_compliance()
    print(json.dumps(sla, indent=2))
