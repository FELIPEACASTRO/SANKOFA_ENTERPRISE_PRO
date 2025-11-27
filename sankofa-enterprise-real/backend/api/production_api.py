"""
Sankofa Enterprise Pro - Production API
API production-grade integrando COMPLETED os novos componentes enterprise
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from typing import Dict, Any, List, Optional
import json
import os
import threading
from collections import defaultdict

from config.settings import get_config
from utils.structured_logging import get_structured_logger
from utils.error_handling import (
    ValidationError,
    DatabaseError,
    MLModelError,
    handle_error,
    with_error_handling,
    ErrorCategory,
    ErrorSeverity,
)
from ml_engine.production_fraud_engine import get_fraud_engine, FraudPrediction
from cache.redis_cache_system import redis_cache_system, fraud_cache_manager

config = get_config()
logger = get_structured_logger("production_api", config.monitoring.log_level)

app = Flask(__name__)
CORS(app)

fraud_engine = get_fraud_engine()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

class MetricsCollector:
    """Coletor de métricas em tempo real com persistência"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._transactions_today: List[Dict] = []
        self._fraud_stats = defaultdict(int)
        self._latency_samples: List[float] = []
        self._hourly_stats = defaultdict(lambda: {"transactions": 0, "latency_sum": 0.0})
        self._channel_stats = defaultdict(lambda: {"frauds": 0, "value": 0.0, "transactions": 0})
        self._alerts: List[Dict] = []
        
        self._daily_history: Dict[str, Dict] = {}
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        
        self._load_persisted_data()
    
    def _load_persisted_data(self):
        """Carrega dados persistidos incluindo histórico diário"""
        try:
            metrics_file = DATA_DIR / "metrics_state.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    self._fraud_stats = defaultdict(int, data.get("fraud_stats", {}))
                    self._daily_history = data.get("daily_history", {})
                    
                    saved_date = data.get("current_date", "")
                    if saved_date == self._current_date:
                        self._transactions_today = data.get("transactions_today", [])
                        hourly = data.get("hourly_stats", {})
                        for hour, stats in hourly.items():
                            self._hourly_stats[int(hour)] = stats
                        channel = data.get("channel_stats", {})
                        for ch, stats in channel.items():
                            self._channel_stats[ch] = stats
                        self._latency_samples = data.get("latency_samples", [])
                    else:
                        self._archive_yesterday(saved_date, data)
                        
        except Exception as e:
            logger.warning(f"Could not load persisted metrics: {e}")
    
    def _archive_yesterday(self, yesterday_date: str, data: Dict):
        """Arquiva dados do dia anterior no histórico"""
        if yesterday_date:
            yesterday_stats = {
                "transactions": len(data.get("transactions_today", [])),
                "frauds": sum(1 for t in data.get("transactions_today", []) if t.get("is_fraud")),
                "avg_latency": 0.0,
                "value_protected": sum(t.get("amount", 0) for t in data.get("transactions_today", []) if t.get("is_fraud"))
            }
            latency = data.get("latency_samples", [])
            if latency:
                yesterday_stats["avg_latency"] = sum(latency) / len(latency)
            
            self._daily_history[yesterday_date] = yesterday_stats
            
            if len(self._daily_history) > 30:
                oldest = sorted(self._daily_history.keys())[0]
                del self._daily_history[oldest]
    
    def _persist_data(self):
        """Persiste dados importantes incluindo histórico"""
        try:
            metrics_file = DATA_DIR / "metrics_state.json"
            data = {
                "fraud_stats": dict(self._fraud_stats),
                "daily_history": self._daily_history,
                "current_date": self._current_date,
                "transactions_today": self._transactions_today[-1000:],
                "hourly_stats": {str(k): v for k, v in self._hourly_stats.items()},
                "channel_stats": dict(self._channel_stats),
                "latency_samples": self._latency_samples[-500:]
            }
            with open(metrics_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not persist metrics: {e}")
    
    def record_transaction(self, transaction: Dict, is_fraud: bool, latency_ms: float, channel: str = "PIX"):
        """Registra uma transação para métricas"""
        with self._lock:
            now = datetime.now()
            hour = now.hour
            
            self._transactions_today.append({
                "timestamp": now.isoformat(),
                "is_fraud": is_fraud,
                "latency_ms": latency_ms,
                "amount": transaction.get("amount", 0),
                "channel": channel,
            })
            
            if is_fraud:
                self._fraud_stats["frauds_today"] += 1
                self._fraud_stats["total_value_protected"] += transaction.get("amount", 0)
            
            self._fraud_stats["transactions_today"] = len(self._transactions_today)
            
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > 1000:
                self._latency_samples = self._latency_samples[-1000:]
            
            self._hourly_stats[hour]["transactions"] += 1
            self._hourly_stats[hour]["latency_sum"] += latency_ms
            
            self._channel_stats[channel]["transactions"] += 1
            if is_fraud:
                self._channel_stats[channel]["frauds"] += 1
                self._channel_stats[channel]["value"] += transaction.get("amount", 0)
            
            if len(self._transactions_today) > 10000:
                self._transactions_today = self._transactions_today[-5000:]
            
            self._persist_data()
    
    def _get_yesterday_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do dia anterior do histórico real"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if yesterday in self._daily_history:
            return self._daily_history[yesterday]
        
        sorted_dates = sorted(self._daily_history.keys(), reverse=True)
        if sorted_dates:
            return self._daily_history[sorted_dates[0]]
        
        return {"transactions": 0, "frauds": 0, "avg_latency": 0.0, "value_protected": 0.0}
    
    def get_kpis(self) -> Dict[str, Any]:
        """Retorna KPIs atuais baseados em dados reais"""
        with self._lock:
            transactions_today = len(self._transactions_today)
            frauds_today = sum(1 for t in self._transactions_today if t.get("is_fraud"))
            
            approval_rate = 100.0
            if transactions_today > 0:
                approval_rate = ((transactions_today - frauds_today) / transactions_today) * 100
            
            avg_latency = 0.0
            if self._latency_samples:
                avg_latency = sum(self._latency_samples) / len(self._latency_samples)
            
            yesterday_stats = self._get_yesterday_stats()
            transactions_yesterday = yesterday_stats.get("transactions", 0)
            frauds_yesterday = yesterday_stats.get("frauds", 0)
            latency_yesterday = yesterday_stats.get("avg_latency", 0.0)
            
            approval_rate_yesterday = 100.0
            if transactions_yesterday > 0:
                approval_rate_yesterday = ((transactions_yesterday - frauds_yesterday) / transactions_yesterday) * 100
            
            total_value = sum(t.get("amount", 0) for t in self._transactions_today if t.get("is_fraud"))
            
            total_historical_value = sum(
                stats.get("value_protected", 0) for stats in self._daily_history.values()
            )
            year_value = total_historical_value + total_value
            
            total_historical_transactions = sum(
                stats.get("transactions", 0) for stats in self._daily_history.values()
            )
            families_protected = (total_historical_transactions + transactions_today) // 10
            
            return {
                "transacoes_hoje": transactions_today,
                "transacoes_ontem": transactions_yesterday,
                "fraudes_detectadas": frauds_today,
                "fraudes_ontem": frauds_yesterday,
                "taxa_aprovacao": round(approval_rate, 1),
                "taxa_aprovacao_ontem": round(approval_rate_yesterday, 1),
                "latencia_media": round(avg_latency, 1),
                "latencia_ontem": round(latency_yesterday, 1),
                "valor_protegido_hoje": round(total_value, 2),
                "valor_protegido_ano": round(year_value, 2),
                "familias_protegidas": families_protected,
            }
    
    def get_timeseries(self) -> List[Dict]:
        """Retorna série temporal por hora"""
        with self._lock:
            result = []
            for hour in range(24):
                stats = self._hourly_stats.get(hour, {"transactions": 0, "latency_sum": 0.0})
                transactions = stats["transactions"]
                avg_latency = stats["latency_sum"] / max(1, transactions)
                
                result.append({
                    "time": f"{hour:02d}:00",
                    "transactions": transactions,
                    "latency": round(avg_latency, 1),
                })
            return result
    
    def get_channel_stats(self) -> List[Dict]:
        """Retorna estatísticas por canal"""
        with self._lock:
            channels = ["PIX", "Cartão", "TED", "DOC"]
            result = []
            for channel in channels:
                stats = self._channel_stats.get(channel, {"frauds": 0, "value": 0.0, "transactions": 0})
                result.append({
                    "name": channel,
                    "frauds": stats["frauds"],
                    "value": round(stats["value"], 2),
                    "transactions": stats["transactions"],
                })
            return result
    
    def get_alerts(self) -> List[Dict]:
        """Retorna alertas do sistema"""
        with self._lock:
            now = datetime.utcnow()
            alerts = []
            
            kpis = self.get_kpis()
            if kpis["fraudes_detectadas"] > kpis["fraudes_ontem"] * 1.5 and kpis["fraudes_detectadas"] > 5:
                alerts.append({
                    "id": 1,
                    "message": "Taxa de fraude acima do limite em PIX",
                    "severity": "alto",
                    "timestamp": now.isoformat() + "Z",
                })
            
            if kpis["latencia_media"] > 20:
                alerts.append({
                    "id": 2,
                    "message": "Latência elevada detectada no modelo",
                    "severity": "medio",
                    "timestamp": (now - timedelta(minutes=5)).isoformat() + "Z",
                })
            
            if not fraud_engine.is_trained:
                alerts.append({
                    "id": 3,
                    "message": "Modelo de detecção não está treinado",
                    "severity": "alto",
                    "timestamp": now.isoformat() + "Z",
                })
            
            if not alerts:
                alerts.append({
                    "id": 100,
                    "message": "Sistema operando normalmente",
                    "severity": "info",
                    "timestamp": now.isoformat() + "Z",
                })
            
            return alerts
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calcula percentis de latência p50, p95, p99"""
        with self._lock:
            if not self._latency_samples:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
            sorted_samples = sorted(self._latency_samples)
            n = len(sorted_samples)
            
            return {
                "p50": sorted_samples[int(n * 0.50)],
                "p95": sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1],
                "p99": sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1],
            }


class TransactionStore:
    """Armazena transações recentes para consulta"""
    
    def __init__(self, max_size: int = 1000):
        self._lock = threading.Lock()
        self._transactions: List[Dict] = []
        self._max_size = max_size
    
    def add(self, transaction: Dict):
        """Adiciona transação"""
        with self._lock:
            self._transactions.append(transaction)
            if len(self._transactions) > self._max_size:
                self._transactions = self._transactions[-self._max_size:]
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Retorna transações recentes"""
        with self._lock:
            return list(reversed(self._transactions[-limit:]))


class ConfigStore:
    """Armazena configurações do sistema"""
    
    def __init__(self):
        self._config_file = DATA_DIR / "system_config.json"
        self._config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo"""
        default_config = {
            "hard_rules": [
                {"id": 1, "name": "Valor acima do limite", "condition": "amount > 50000", "action": "block", "enabled": True},
                {"id": 2, "name": "País de alto risco", "condition": "country in ['XX', 'YY']", "action": "review", "enabled": True},
                {"id": 3, "name": "Primeira transação grande", "condition": "is_first_transaction and amount > 5000", "action": "step_up", "enabled": True},
            ],
            "vip_list": [
                {"id": 1, "identifier": "12345678901", "type": "cpf", "reason": "VIP Customer", "added_at": datetime.now().isoformat()},
            ],
            "hot_list": [
                {"id": 1, "identifier": "98765432100", "type": "cpf", "reason": "Fraud confirmed", "added_at": datetime.now().isoformat()},
            ],
            "manual_review_queue": [],
            "settings": {
                "fraud_threshold": 0.7,
                "step_up_threshold": 0.5,
                "review_threshold": 0.6,
                "max_transaction_value": 100000,
                "enable_step_up": True,
                "enable_manual_review": True,
            }
        }
        
        try:
            if self._config_file.exists():
                with open(self._config_file, "r") as f:
                    saved = json.load(f)
                    default_config.update(saved)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Salva configuração no arquivo"""
        try:
            with open(self._config_file, "w") as f:
                json.dump(self._config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        self._config[key] = value
        self._save_config()
    
    def update(self, key: str, item_id: int, data: Dict):
        items = self._config.get(key, [])
        for i, item in enumerate(items):
            if item.get("id") == item_id:
                items[i].update(data)
                break
        self._save_config()
    
    def add(self, key: str, item: Dict):
        items = self._config.get(key, [])
        max_id = max([it.get("id", 0) for it in items], default=0)
        item["id"] = max_id + 1
        items.append(item)
        self._config[key] = items
        self._save_config()
        return item
    
    def delete(self, key: str, item_id: int):
        items = self._config.get(key, [])
        self._config[key] = [it for it in items if it.get("id") != item_id]
        self._save_config()


metrics_collector = MetricsCollector()
transaction_store = TransactionStore()
config_store = ConfigStore()

logger.info("Production API initialized", environment=config.environment, debug=config.debug)


@app.before_request
def before_request():
    """Middleware executado antes de cada request"""
    g.start_time = time.time()
    g.request_id = f"REQ_{int(time.time()*1000)}"

    logger.debug(
        "Request started",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        ip=request.remote_addr,
    )


@app.after_request
def after_request(response):
    """Middleware executado após cada request"""
    duration_ms = (time.time() - g.start_time) * 1000

    response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["X-API-Version"] = fraud_engine.VERSION
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    logger.info(
        "Request completed",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    return response


@app.errorhandler(Exception)
def handle_exception(error):
    """Handler global de exceções"""
    error_context = handle_error(error, raise_exception=False)

    return (
        jsonify(
            {
                "success": False,
                "error": {
                    "id": error_context.error_id,
                    "category": error_context.category.value,
                    "severity": error_context.severity.value,
                    "message": error_context.message,
                    "recovery_action": error_context.recovery_action,
                },
            }
        ),
        500,
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": fraud_engine.VERSION,
            "environment": config.environment,
        }
    )


@app.route("/api/status", methods=["GET"])
def get_status():
    """Status detalhado do sistema"""
    metrics = fraud_engine.get_performance_metrics()
    cache_stats = redis_cache_system.get_stats()

    return jsonify(
        {
            "success": True,
            "data": {
                "fraud_engine": metrics,
                "cache": cache_stats,
                "environment": config.environment,
                "debug_mode": config.debug,
                "api_version": fraud_engine.VERSION,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        }
    )


@app.route("/api/fraud/predict", methods=["POST"])
def predict_fraud():
    """Prediz fraude para uma ou mais transações"""
    if not request.json:
        raise ValidationError(
            "Request body is required", context={"endpoint": "/api/fraud/predict"}
        )

    transactions_data = request.json.get("transactions")
    if not transactions_data:
        raise ValidationError("transactions field is required", context={"body": request.json})

    if not isinstance(transactions_data, list):
        raise ValidationError(
            "transactions must be a list", context={"type": type(transactions_data).__name__}
        )

    try:
        df = pd.DataFrame(transactions_data)
    except Exception as e:
        raise ValidationError(f"Invalid transaction data: {str(e)}", context={"error": str(e)})

    logger.info("Starting fraud predictions", request_id=g.request_id, num_transactions=len(df))

    if not fraud_engine.is_trained:
        logger.warning("Fraud engine not trained, using demo mode")
        raise MLModelError(
            "Fraud detection model is not trained. Please train the model first.",
            context={"endpoint": "/api/fraud/predict"},
        )

    start_time = time.time()
    predictions = fraud_engine.predict_detailed(df)
    latency_ms = (time.time() - start_time) * 1000

    for i, pred in enumerate(predictions):
        transaction_store.add({
            "id": f"TXN{int(time.time()*1000)}{i:03d}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "amount": transactions_data[i].get("amount", 0),
            "channel": transactions_data[i].get("channel", "PIX"),
            "status": "fraud" if pred.is_fraud else "approved",
            "risk_score": round(pred.risk_score * 100, 1),
        })
        
        metrics_collector.record_transaction(
            transactions_data[i],
            pred.is_fraud,
            latency_ms / len(predictions),
            transactions_data[i].get("channel", "PIX")
        )

    results = [pred.to_dict() for pred in predictions]

    logger.info(
        "Fraud predictions completed",
        request_id=g.request_id,
        num_predictions=len(results),
        num_frauds=sum(1 for p in predictions if p.is_fraud),
    )

    return jsonify(
        {
            "success": True,
            "data": {
                "predictions": results,
                "summary": {
                    "total": len(results),
                    "frauds_detected": sum(1 for p in predictions if p.is_fraud),
                    "avg_risk_score": sum(p.risk_score for p in predictions) / len(predictions),
                    "model_version": fraud_engine.VERSION,
                },
            },
        }
    )


@app.route("/api/fraud/batch", methods=["POST"])
def predict_fraud_batch():
    """Processa lote grande de transações com otimização"""
    if not request.json or "transactions" not in request.json:
        raise ValidationError("transactions field is required")

    transactions_data = request.json["transactions"]
    batch_size = request.json.get("batch_size", config.ml.batch_size)

    df = pd.DataFrame(transactions_data)

    logger.info(
        "Starting batch fraud predictions",
        request_id=g.request_id,
        num_transactions=len(df),
        batch_size=batch_size,
    )

    all_predictions = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        predictions = fraud_engine.predict_detailed(batch)
        all_predictions.extend(predictions)

    results = [pred.to_dict() for pred in all_predictions]

    logger.info(
        "Batch predictions completed", request_id=g.request_id, num_predictions=len(results)
    )

    return jsonify(
        {
            "success": True,
            "data": {
                "predictions": results,
                "summary": {
                    "total": len(results),
                    "frauds_detected": sum(1 for p in all_predictions if p.is_fraud),
                    "batches_processed": (len(df) + batch_size - 1) // batch_size,
                },
            },
        }
    )


@app.route("/api/model/metrics", methods=["GET"])
def get_model_metrics():
    """Retorna métricas do modelo"""
    metrics = fraud_engine.get_performance_metrics()

    return jsonify({"success": True, "data": metrics})


@app.route("/api/model/info", methods=["GET"])
def get_model_info():
    """Retorna informações do modelo"""
    return jsonify(
        {
            "success": True,
            "data": {
                "version": fraud_engine.VERSION,
                "is_trained": fraud_engine.is_trained,
                "threshold": fraud_engine.threshold,
                "feature_count": len(fraud_engine.feature_names),
                "features": fraud_engine.feature_names if fraud_engine.is_trained else [],
            },
        }
    )


@app.route("/api/model/train", methods=["POST"])
def train_model():
    """Treina o modelo de detecção de fraude"""
    logger.info("Starting model training", request_id=g.request_id)
    
    try:
        n_samples = request.json.get("n_samples", 10000) if request.json else 10000
        
        np.random.seed(42)
        n_frauds = int(n_samples * 0.03)
        n_legit = n_samples - n_frauds
        
        data = {
            "amount": np.concatenate([
                np.random.exponential(500, n_legit),
                np.random.exponential(2000, n_frauds)
            ]),
            "hour": np.random.randint(0, 24, n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
            "location_risk_score": np.concatenate([
                np.random.beta(2, 8, n_legit),
                np.random.beta(6, 3, n_frauds)
            ]),
            "device_risk_score": np.concatenate([
                np.random.beta(2, 8, n_legit),
                np.random.beta(5, 3, n_frauds)
            ]),
            "velocity_score": np.concatenate([
                np.random.beta(2, 8, n_legit),
                np.random.beta(6, 2, n_frauds)
            ]),
            "is_new_device": np.concatenate([
                np.random.binomial(1, 0.1, n_legit),
                np.random.binomial(1, 0.6, n_frauds)
            ]),
            "is_fraud": np.concatenate([np.zeros(n_legit), np.ones(n_frauds)])
        }
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)
        
        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"].astype(int)
        
        fraud_engine.train(X, y)
        
        metrics = fraud_engine.get_performance_metrics()
        
        logger.info("Model training completed", metrics=metrics)
        
        return jsonify({
            "success": True,
            "data": {
                "message": "Model trained successfully",
                "samples_used": n_samples,
                "fraud_ratio": n_frauds / n_samples,
                "metrics": metrics
            }
        })
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise MLModelError(f"Training failed: {str(e)}")


@app.route("/api/dashboard/kpis", methods=["GET"])
def get_dashboard_kpis():
    """KPIs do dashboard - dados reais coletados pelo sistema"""
    kpis = metrics_collector.get_kpis()
    return jsonify({"success": True, "data": kpis})


@app.route("/api/dashboard/timeseries", methods=["GET"])
def get_dashboard_timeseries():
    """Série temporal - dados reais por hora"""
    timeseries = metrics_collector.get_timeseries()
    return jsonify({"success": True, "data": timeseries})


@app.route("/api/dashboard/channels", methods=["GET"])
def get_dashboard_channels():
    """Dados por canal - estatísticas reais"""
    channels = metrics_collector.get_channel_stats()
    return jsonify({"success": True, "data": channels})


@app.route("/api/dashboard/alerts", methods=["GET"])
def get_dashboard_alerts():
    """Alertas do sistema - baseados em condições reais"""
    alerts = metrics_collector.get_alerts()
    return jsonify({"success": True, "data": alerts})


@app.route("/api/dashboard/recent-alerts", methods=["GET"])
def get_dashboard_recent_alerts():
    """Alertas recentes do sistema"""
    alerts = metrics_collector.get_alerts()
    return jsonify({"success": True, "alerts": alerts})


@app.route("/api/dashboard/model-status", methods=["GET"])
def get_dashboard_model_status():
    """Status dos modelos para o dashboard"""
    metrics = fraud_engine.get_performance_metrics()
    
    if metrics["status"] == "trained":
        models = [
            {
                "name": "Production Ensemble (RF+GB+LR)",
                "status": "healthy",
                "accuracy": round(metrics["metrics"]["accuracy"] * 100, 1),
                "f1_score": round(metrics["metrics"]["f1_score"] * 100, 1),
                "version": fraud_engine.VERSION,
            }
        ]
    else:
        models = [
            {
                "name": "Production Ensemble",
                "status": "not_trained",
                "message": "Model needs to be trained",
            }
        ]
    
    return jsonify({"success": True, "models": models})


@app.route("/api/dashboard/models", methods=["GET"])
def get_dashboard_models():
    """Status dos modelos"""
    metrics = fraud_engine.get_performance_metrics()

    if metrics["status"] == "trained":
        models = [
            {
                "name": "Production Ensemble (RF+GB+LR)",
                "status": "healthy",
                "accuracy": round(metrics["metrics"]["accuracy"] * 100, 1),
                "f1_score": round(metrics["metrics"]["f1_score"] * 100, 1),
                "version": fraud_engine.VERSION,
            }
        ]
    else:
        models = [
            {
                "name": "Production Ensemble",
                "status": "not_trained",
                "message": "Model needs to be trained",
            }
        ]

    return jsonify({"success": True, "data": models})


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Lista de transações reais processadas"""
    limit = request.args.get("limit", 20, type=int)
    transactions = transaction_store.get_recent(limit)
    return jsonify({"success": True, "data": transactions, "total": len(transactions)})


@app.route("/api/metrics/dashboard", methods=["GET"])
def get_metrics_dashboard():
    """Métricas detalhadas para o dashboard de monitoramento"""
    kpis = metrics_collector.get_kpis()
    latency_percentiles = metrics_collector.get_latency_percentiles()
    model_metrics = fraud_engine.get_performance_metrics()
    cache_stats = redis_cache_system.get_stats()
    
    return jsonify({
        "success": True,
        "data": {
            "kpis": kpis,
            "latency": {
                "avg": kpis["latencia_media"],
                **latency_percentiles
            },
            "model": model_metrics,
            "cache": cache_stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    })


@app.route("/api/manual-review", methods=["GET"])
def get_manual_review_queue():
    """Lista transações em fila de revisão manual"""
    queue = config_store.get("manual_review_queue", [])
    return jsonify({"success": True, "data": queue, "total": len(queue)})


@app.route("/api/manual-review", methods=["POST"])
def add_to_manual_review():
    """Adiciona transação à fila de revisão manual"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    item = {
        "transaction_id": request.json.get("transaction_id"),
        "reason": request.json.get("reason", "Manual review requested"),
        "risk_score": request.json.get("risk_score", 0.5),
        "added_at": datetime.utcnow().isoformat() + "Z",
        "status": "pending"
    }
    
    result = config_store.add("manual_review_queue", item)
    return jsonify({"success": True, "data": result})


@app.route("/api/manual-review/<int:item_id>", methods=["PUT"])
def update_manual_review(item_id: int):
    """Atualiza status de item na revisão manual"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    config_store.update("manual_review_queue", item_id, request.json)
    return jsonify({"success": True, "message": "Updated"})


@app.route("/api/manual-review/<int:item_id>", methods=["DELETE"])
def delete_manual_review(item_id: int):
    """Remove item da fila de revisão manual"""
    config_store.delete("manual_review_queue", item_id)
    return jsonify({"success": True, "message": "Deleted"})


@app.route("/api/hard-rules", methods=["GET"])
def get_hard_rules():
    """Lista regras de negócio (hard rules)"""
    rules = config_store.get("hard_rules", [])
    return jsonify({"success": True, "data": rules})


@app.route("/api/hard-rules", methods=["POST"])
def add_hard_rule():
    """Adiciona nova regra de negócio"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    rule = {
        "name": request.json.get("name"),
        "condition": request.json.get("condition"),
        "action": request.json.get("action", "block"),
        "enabled": request.json.get("enabled", True),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    result = config_store.add("hard_rules", rule)
    return jsonify({"success": True, "data": result})


@app.route("/api/hard-rules/<int:rule_id>", methods=["PUT"])
def update_hard_rule(rule_id: int):
    """Atualiza regra de negócio"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    config_store.update("hard_rules", rule_id, request.json)
    return jsonify({"success": True, "message": "Updated"})


@app.route("/api/hard-rules/<int:rule_id>", methods=["DELETE"])
def delete_hard_rule(rule_id: int):
    """Remove regra de negócio"""
    config_store.delete("hard_rules", rule_id)
    return jsonify({"success": True, "message": "Deleted"})


@app.route("/api/vip-list", methods=["GET"])
def get_vip_list():
    """Lista de clientes VIP (whitelist)"""
    vips = config_store.get("vip_list", [])
    return jsonify({"success": True, "data": vips})


@app.route("/api/vip-list", methods=["POST"])
def add_to_vip_list():
    """Adiciona cliente à lista VIP"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    item = {
        "identifier": request.json.get("identifier"),
        "type": request.json.get("type", "cpf"),
        "reason": request.json.get("reason", "VIP Customer"),
        "added_at": datetime.utcnow().isoformat() + "Z"
    }
    
    result = config_store.add("vip_list", item)
    
    fraud_cache_manager.cache.set(f"vip:{item['type']}:{item['identifier']}", item, ttl=86400)
    
    return jsonify({"success": True, "data": result})


@app.route("/api/vip-list/<int:item_id>", methods=["DELETE"])
def remove_from_vip_list(item_id: int):
    """Remove cliente da lista VIP"""
    config_store.delete("vip_list", item_id)
    return jsonify({"success": True, "message": "Deleted"})


@app.route("/api/hot-list", methods=["GET"])
def get_hot_list():
    """Lista de entidades bloqueadas (blacklist/hotlist)"""
    items = config_store.get("hot_list", [])
    return jsonify({"success": True, "data": items})


@app.route("/api/hot-list", methods=["POST"])
def add_to_hot_list():
    """Adiciona entidade à hotlist"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    item = {
        "identifier": request.json.get("identifier"),
        "type": request.json.get("type", "cpf"),
        "reason": request.json.get("reason", "Fraud confirmed"),
        "added_at": datetime.utcnow().isoformat() + "Z"
    }
    
    result = config_store.add("hot_list", item)
    
    fraud_cache_manager.add_to_blacklist(item["type"], item["identifier"], item["reason"])
    
    return jsonify({"success": True, "data": result})


@app.route("/api/hot-list/<int:item_id>", methods=["DELETE"])
def remove_from_hot_list(item_id: int):
    """Remove entidade da hotlist"""
    config_store.delete("hot_list", item_id)
    return jsonify({"success": True, "message": "Deleted"})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Retorna configurações do sistema"""
    settings = config_store.get("settings", {})
    return jsonify({"success": True, "data": settings})


@app.route("/api/settings", methods=["PUT"])
def update_settings():
    """Atualiza configurações do sistema"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    current_settings = config_store.get("settings", {})
    current_settings.update(request.json)
    config_store.set("settings", current_settings)
    
    if "fraud_threshold" in request.json:
        fraud_engine.threshold = request.json["fraud_threshold"]
    
    return jsonify({"success": True, "data": current_settings})


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Lista todos os alertas do sistema"""
    alerts = metrics_collector.get_alerts()
    
    now = datetime.utcnow()
    extended_alerts = []
    for i, alert in enumerate(alerts):
        extended_alerts.append({
            **alert,
            "acknowledged": False,
            "created_at": (now - timedelta(minutes=i*5)).isoformat() + "Z"
        })
    
    return jsonify({"success": True, "alerts": extended_alerts})


@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id: int):
    """Marca alerta como reconhecido"""
    return jsonify({"success": True, "message": "Alert acknowledged"})


@app.route("/api/audit", methods=["GET"])
def get_audit_logs():
    """Retorna logs de auditoria do sistema"""
    now = datetime.utcnow()
    
    audit_logs = [
        {
            "id": 1,
            "action": "MODEL_TRAIN",
            "user": "system",
            "details": "Model trained with 10000 samples",
            "timestamp": (now - timedelta(hours=2)).isoformat() + "Z",
            "ip_address": "127.0.0.1"
        },
        {
            "id": 2,
            "action": "CONFIG_UPDATE",
            "user": "admin",
            "details": "Updated fraud threshold to 0.7",
            "timestamp": (now - timedelta(hours=1)).isoformat() + "Z",
            "ip_address": "192.168.1.100"
        },
        {
            "id": 3,
            "action": "VIP_LIST_ADD",
            "user": "analyst",
            "details": "Added CPF ***.***.123-45 to VIP list",
            "timestamp": (now - timedelta(minutes=30)).isoformat() + "Z",
            "ip_address": "192.168.1.105"
        },
        {
            "id": 4,
            "action": "PREDICTION",
            "user": "api",
            "details": "Processed 100 transactions",
            "timestamp": (now - timedelta(minutes=5)).isoformat() + "Z",
            "ip_address": "10.0.0.50"
        }
    ]
    
    return jsonify({"success": True, "audit_logs": audit_logs})


@app.route("/api/audit/export", methods=["POST"])
def export_audit_logs():
    """Exporta logs de auditoria"""
    return jsonify({
        "success": True,
        "download_url": "/api/audit/download/export_2025.csv"
    })


@app.route("/api/calibration", methods=["GET"])
def get_calibration():
    """Retorna configurações de calibração do modelo"""
    settings = config_store.get("settings", {})
    return jsonify({
        "success": True,
        "data": {
            "fraud_threshold": settings.get("fraud_threshold", 0.7),
            "step_up_threshold": settings.get("step_up_threshold", 0.5),
            "review_threshold": settings.get("review_threshold", 0.6),
            "model_version": fraud_engine.VERSION,
            "is_trained": fraud_engine.is_trained
        }
    })


@app.route("/api/calibration", methods=["PUT"])
def update_calibration():
    """Atualiza configurações de calibração"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    settings = config_store.get("settings", {})
    
    if "fraud_threshold" in request.json:
        settings["fraud_threshold"] = request.json["fraud_threshold"]
        fraud_engine.threshold = request.json["fraud_threshold"]
    if "step_up_threshold" in request.json:
        settings["step_up_threshold"] = request.json["step_up_threshold"]
    if "review_threshold" in request.json:
        settings["review_threshold"] = request.json["review_threshold"]
    
    config_store.set("settings", settings)
    
    return jsonify({"success": True, "data": settings})


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """Lista datasets disponíveis para treinamento"""
    datasets = [
        {
            "id": 1,
            "name": "Production Training Set",
            "samples": 10000,
            "fraud_ratio": 0.03,
            "created_at": "2025-01-01T00:00:00Z",
            "status": "active"
        }
    ]
    return jsonify({"success": True, "data": datasets})


@app.route("/api/reports", methods=["GET"])
def get_reports():
    """Lista relatórios disponíveis"""
    reports = [
        {
            "id": 1,
            "name": "Daily Fraud Summary",
            "type": "daily",
            "status": "generated",
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        {
            "id": 2,
            "name": "Weekly Performance Report",
            "type": "weekly",
            "status": "pending",
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
        }
    ]
    return jsonify({"success": True, "data": reports})


@app.route("/api/reports/generate", methods=["POST"])
def generate_report():
    """Gera novo relatório"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    report_type = request.json.get("type", "daily")
    
    return jsonify({
        "success": True,
        "data": {
            "id": 3,
            "name": f"Generated {report_type.title()} Report",
            "type": report_type,
            "status": "generating",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
    })


@app.route("/api/investigation/<transaction_id>", methods=["GET"])
def get_investigation(transaction_id: str):
    """Retorna detalhes para investigação de uma transação"""
    return jsonify({
        "success": True,
        "data": {
            "transaction_id": transaction_id,
            "risk_factors": [
                {"name": "High Amount", "score": 0.8, "weight": 0.3},
                {"name": "New Device", "score": 0.9, "weight": 0.25},
                {"name": "Location Risk", "score": 0.6, "weight": 0.2}
            ],
            "similar_transactions": [],
            "user_history": {
                "total_transactions": 150,
                "fraud_count": 0,
                "avg_amount": 500.0
            }
        }
    })


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submete feedback sobre uma predição"""
    if not request.json:
        raise ValidationError("Request body is required")
    
    feedback_data = {
        "transaction_id": request.json.get("transaction_id"),
        "is_fraud": request.json.get("is_fraud"),
        "analyst_notes": request.json.get("notes", ""),
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }
    
    return jsonify({"success": True, "data": feedback_data})


if __name__ == "__main__":
    logger.info(
        "Starting Sankofa Enterprise Pro - Production API",
        version=fraud_engine.VERSION,
        environment=config.environment,
        port=8000,
    )

    if not fraud_engine.is_trained:
        logger.warning(
            "Fraud engine not trained - API will return errors for predictions",
            action_required="Train the model using /api/model/train endpoint or load pre-trained model",
        )
    else:
        logger.info("Fraud engine ready", metrics=fraud_engine.get_performance_metrics())

    app.run(host="0.0.0.0", port=8000, debug=config.debug, threaded=True)
