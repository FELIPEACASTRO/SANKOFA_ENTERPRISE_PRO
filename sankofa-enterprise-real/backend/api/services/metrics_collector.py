"""
Sankofa Enterprise Pro - Metrics Collector Service
Coletor de métricas em tempo real com persistência
"""

import json
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class MetricsCollector:
    """Coletor de métricas em tempo real com persistência"""
    
    def __init__(self, fraud_engine_ref=None):
        self._lock = threading.RLock()
        self._transactions_today: List[Dict] = []
        self._fraud_stats: Dict[str, Any] = defaultdict(int)
        self._latency_samples: List[float] = []
        self._hourly_stats: Dict[int, Dict] = defaultdict(lambda: {"transactions": 0, "latency_sum": 0.0})
        self._channel_stats: Dict[str, Dict] = defaultdict(lambda: {"frauds": 0, "value": 0.0, "transactions": 0})
        self._alerts: List[Dict] = []
        self._fraud_engine_ref = fraud_engine_ref
        
        self._daily_history: Dict[str, Dict] = {}
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        
        self._load_persisted_data()
    
    def set_fraud_engine(self, engine):
        """Define referência ao fraud engine para alertas"""
        self._fraud_engine_ref = engine
    
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
                        
        except Exception:
            pass
    
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
        except Exception:
            pass
    
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
            
            if self._fraud_engine_ref and not self._fraud_engine_ref.is_trained:
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


metrics_collector = MetricsCollector()
