"""
Sankofa Enterprise Pro - Shadow Mode
Sistema de deploy gradual com comparação de modelos
Permite testar novos modelos sem impactar produção
"""

import json
import time
import threading
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionComparison:
    """Comparação entre predições de modelos"""
    transaction_id: str
    timestamp: str
    
    primary_prediction: bool
    primary_probability: float
    primary_risk_level: str
    primary_latency_ms: float
    
    shadow_prediction: bool
    shadow_probability: float
    shadow_risk_level: str
    shadow_latency_ms: float
    
    prediction_match: bool
    probability_diff: float
    ground_truth: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowModeStats:
    """Estatísticas do Shadow Mode"""
    total_comparisons: int = 0
    prediction_matches: int = 0
    prediction_mismatches: int = 0
    
    primary_fraud_count: int = 0
    shadow_fraud_count: int = 0
    
    avg_probability_diff: float = 0.0
    max_probability_diff: float = 0.0
    
    primary_avg_latency_ms: float = 0.0
    shadow_avg_latency_ms: float = 0.0
    
    true_positives_primary: int = 0
    true_positives_shadow: int = 0
    false_positives_primary: int = 0
    false_positives_shadow: int = 0
    
    start_time: str = ""
    last_update: str = ""
    
    def match_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.prediction_matches / self.total_comparisons * 100
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['match_rate_percent'] = round(self.match_rate(), 2)
        return data


class ShadowModeManager:
    """
    Gerenciador de Shadow Mode para Deploy Gradual
    
    Features:
    - Executa modelo de produção e shadow em paralelo
    - Compara predições e coleta métricas
    - Suporta traffic splitting configurável
    - Permite rollback automático se shadow falhar
    - Armazena histórico de comparações
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        primary_model: Optional[Callable] = None,
        shadow_model: Optional[Callable] = None,
        shadow_traffic_percent: float = 100.0,
        comparison_log_dir: str = "mlops/shadow_logs"
    ):
        self.primary_model = primary_model
        self.shadow_model = shadow_model
        self.shadow_traffic_percent = shadow_traffic_percent
        self.comparison_log_dir = Path(comparison_log_dir)
        self.comparison_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_active = False
        self.stats = ShadowModeStats()
        self.comparisons: List[PredictionComparison] = []
        self._lock = threading.RLock()
        
        self._probability_diffs: List[float] = []
        self._primary_latencies: List[float] = []
        self._shadow_latencies: List[float] = []
        
        self._max_comparisons = 10000
        
        logger.info(f"ShadowModeManager initialized v{self.VERSION}")
    
    def start(
        self,
        primary_model: Optional[Callable] = None,
        shadow_model: Optional[Callable] = None,
        shadow_traffic_percent: float = 100.0
    ):
        """Inicia Shadow Mode"""
        if primary_model:
            self.primary_model = primary_model
        if shadow_model:
            self.shadow_model = shadow_model
        if shadow_traffic_percent:
            self.shadow_traffic_percent = shadow_traffic_percent
        
        if not self.primary_model or not self.shadow_model:
            raise ValueError("Both primary and shadow models must be set")
        
        self.is_active = True
        self.stats = ShadowModeStats(
            start_time=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(
            f"Shadow Mode started with {self.shadow_traffic_percent}% traffic"
        )
    
    def stop(self) -> Dict[str, Any]:
        """Para Shadow Mode e retorna relatório final"""
        self.is_active = False
        
        report = self.get_report()
        
        self._save_report(report)
        
        logger.info("Shadow Mode stopped")
        return report
    
    def _should_run_shadow(self) -> bool:
        """Determina se deve executar shadow (baseado em traffic split)"""
        return random.random() * 100 < self.shadow_traffic_percent
    
    def predict_with_shadow(
        self,
        transaction: Dict[str, Any],
        force_shadow: bool = False
    ) -> Tuple[Dict[str, Any], Optional[PredictionComparison]]:
        """
        Executa predição com comparação shadow
        
        Args:
            transaction: Dados da transação
            force_shadow: Força execução do shadow (ignora traffic split)
            
        Returns:
            Tuple com (predição_primária, comparação_shadow ou None)
        """
        
        if self.primary_model is None:
            raise ValueError("Primary model not set")
        
        primary_start = time.time()
        primary_result = self.primary_model(transaction)
        primary_latency = (time.time() - primary_start) * 1000
        
        comparison = None
        
        if self.is_active and self.shadow_model and (force_shadow or self._should_run_shadow()):
            try:
                shadow_start = time.time()
                shadow_result = self.shadow_model(transaction)
                shadow_latency = (time.time() - shadow_start) * 1000
                
                comparison = self._create_comparison(
                    transaction=transaction,
                    primary_result=primary_result,
                    primary_latency=primary_latency,
                    shadow_result=shadow_result,
                    shadow_latency=shadow_latency
                )
                
                self._record_comparison(comparison)
                
            except Exception as e:
                logger.warning(f"Shadow prediction failed: {e}")
        
        return primary_result, comparison
    
    def _create_comparison(
        self,
        transaction: Dict[str, Any],
        primary_result: Dict[str, Any],
        primary_latency: float,
        shadow_result: Dict[str, Any],
        shadow_latency: float
    ) -> PredictionComparison:
        """Cria objeto de comparação"""
        
        primary_is_fraud = primary_result.get('is_fraud', False)
        shadow_is_fraud = shadow_result.get('is_fraud', False)
        
        primary_prob = primary_result.get('fraud_probability', 0.0)
        shadow_prob = shadow_result.get('fraud_probability', 0.0)
        
        return PredictionComparison(
            transaction_id=transaction.get('transaction_id', str(time.time())),
            timestamp=datetime.utcnow().isoformat() + "Z",
            primary_prediction=primary_is_fraud,
            primary_probability=primary_prob,
            primary_risk_level=primary_result.get('risk_level', 'UNKNOWN'),
            primary_latency_ms=round(primary_latency, 2),
            shadow_prediction=shadow_is_fraud,
            shadow_probability=shadow_prob,
            shadow_risk_level=shadow_result.get('risk_level', 'UNKNOWN'),
            shadow_latency_ms=round(shadow_latency, 2),
            prediction_match=primary_is_fraud == shadow_is_fraud,
            probability_diff=abs(primary_prob - shadow_prob)
        )
    
    def _record_comparison(self, comparison: PredictionComparison):
        """Registra comparação nas estatísticas"""
        with self._lock:
            if len(self.comparisons) >= self._max_comparisons:
                self.comparisons.pop(0)
            
            self.comparisons.append(comparison)
            
            self.stats.total_comparisons += 1
            
            if comparison.prediction_match:
                self.stats.prediction_matches += 1
            else:
                self.stats.prediction_mismatches += 1
            
            if comparison.primary_prediction:
                self.stats.primary_fraud_count += 1
            if comparison.shadow_prediction:
                self.stats.shadow_fraud_count += 1
            
            self._probability_diffs.append(comparison.probability_diff)
            self._primary_latencies.append(comparison.primary_latency_ms)
            self._shadow_latencies.append(comparison.shadow_latency_ms)
            
            if len(self._probability_diffs) > 1000:
                self._probability_diffs = self._probability_diffs[-1000:]
                self._primary_latencies = self._primary_latencies[-1000:]
                self._shadow_latencies = self._shadow_latencies[-1000:]
            
            self._update_aggregates()
            
            self.stats.last_update = datetime.utcnow().isoformat() + "Z"
    
    def _update_aggregates(self):
        """Atualiza métricas agregadas"""
        if self._probability_diffs:
            self.stats.avg_probability_diff = sum(self._probability_diffs) / len(self._probability_diffs)
            self.stats.max_probability_diff = max(self._probability_diffs)
        
        if self._primary_latencies:
            self.stats.primary_avg_latency_ms = sum(self._primary_latencies) / len(self._primary_latencies)
        
        if self._shadow_latencies:
            self.stats.shadow_avg_latency_ms = sum(self._shadow_latencies) / len(self._shadow_latencies)
    
    def record_ground_truth(self, transaction_id: str, is_fraud: bool):
        """Registra ground truth para uma transação"""
        with self._lock:
            for comparison in reversed(self.comparisons):
                if comparison.transaction_id == transaction_id:
                    comparison.ground_truth = is_fraud
                    
                    if is_fraud:
                        if comparison.primary_prediction:
                            self.stats.true_positives_primary += 1
                        if comparison.shadow_prediction:
                            self.stats.true_positives_shadow += 1
                    else:
                        if comparison.primary_prediction:
                            self.stats.false_positives_primary += 1
                        if comparison.shadow_prediction:
                            self.stats.false_positives_shadow += 1
                    
                    return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        return self.stats.to_dict()
    
    def get_report(self) -> Dict[str, Any]:
        """Gera relatório completo do Shadow Mode"""
        with self._lock:
            recommendation = "CONTINUE_SHADOW"
            
            if self.stats.total_comparisons >= 100:
                match_rate = self.stats.match_rate()
                
                if match_rate >= 95 and self.stats.avg_probability_diff < 0.05:
                    recommendation = "PROMOTE_SHADOW"
                elif match_rate < 80 or self.stats.avg_probability_diff > 0.15:
                    recommendation = "ROLLBACK_SHADOW"
            
            mismatch_details = []
            for comp in self.comparisons:
                if not comp.prediction_match:
                    mismatch_details.append({
                        'transaction_id': comp.transaction_id,
                        'primary': comp.primary_prediction,
                        'shadow': comp.shadow_prediction,
                        'prob_diff': round(comp.probability_diff, 4)
                    })
            
            return {
                'version': self.VERSION,
                'is_active': self.is_active,
                'shadow_traffic_percent': self.shadow_traffic_percent,
                'statistics': self.stats.to_dict(),
                'recommendation': recommendation,
                'mismatch_sample': mismatch_details[:20],
                'generated_at': datetime.utcnow().isoformat() + "Z"
            }
    
    def _save_report(self, report: Dict[str, Any]):
        """Salva relatório em arquivo"""
        filename = f"shadow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.comparison_log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Shadow report saved to {filepath}")
    
    def get_divergent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna transações com maior divergência entre modelos"""
        with self._lock:
            sorted_comparisons = sorted(
                self.comparisons,
                key=lambda x: x.probability_diff,
                reverse=True
            )
            
            return [c.to_dict() for c in sorted_comparisons[:limit]]


_shadow_manager: Optional[ShadowModeManager] = None


def get_shadow_manager() -> ShadowModeManager:
    """Singleton para ShadowModeManager"""
    global _shadow_manager
    if _shadow_manager is None:
        _shadow_manager = ShadowModeManager()
    return _shadow_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def mock_primary_model(txn):
        return {
            'is_fraud': txn.get('amount', 0) > 10000,
            'fraud_probability': min(txn.get('amount', 0) / 20000, 1.0),
            'risk_level': 'HIGH' if txn.get('amount', 0) > 10000 else 'LOW'
        }
    
    def mock_shadow_model(txn):
        return {
            'is_fraud': txn.get('amount', 0) > 8000,
            'fraud_probability': min(txn.get('amount', 0) / 15000, 1.0),
            'risk_level': 'HIGH' if txn.get('amount', 0) > 8000 else 'LOW'
        }
    
    manager = ShadowModeManager()
    manager.start(
        primary_model=mock_primary_model,
        shadow_model=mock_shadow_model,
        shadow_traffic_percent=100
    )
    
    test_transactions = [
        {'transaction_id': 'T1', 'amount': 500},
        {'transaction_id': 'T2', 'amount': 5000},
        {'transaction_id': 'T3', 'amount': 9000},
        {'transaction_id': 'T4', 'amount': 15000},
        {'transaction_id': 'T5', 'amount': 25000},
    ]
    
    for txn in test_transactions:
        result, comparison = manager.predict_with_shadow(txn)
        if comparison:
            print(f"Transaction {txn['transaction_id']}: Match={comparison.prediction_match}")
    
    report = manager.stop()
    print(f"\nFinal Report:")
    print(json.dumps(report, indent=2))
