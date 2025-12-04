"""
Sankofa Enterprise Pro - Advanced Modules Orchestrator
Orquestração de módulos avançados de ML seguindo staged enrichment pipeline
Mantém SLA sub-50ms com processamento condicional/assíncrono
"""

import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum

logger = logging.getLogger(__name__)


class EnrichmentLevel(Enum):
    """Níveis de enriquecimento da predição"""
    BASIC = "basic"
    STANDARD = "standard"
    FULL = "full"
    ASYNC_PENDING = "async_pending"


@dataclass
class EnrichedPrediction:
    """Predição enriquecida com todos os módulos avançados"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    model_version: str
    detection_reasons: List[str]
    enrichment_level: str
    autoencoder_anomaly_score: Optional[float] = None
    sequence_risk_score: Optional[float] = None
    moe_consensus: Optional[float] = None
    gnn_community_risk: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None
    lgpd_audit: Optional[Dict[str, Any]] = None
    advanced_insights: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorConfig:
    """Configuração do orquestrador"""
    enable_autoencoder: bool = True
    enable_sequence_analysis: bool = True
    enable_moe: bool = True
    enable_gnn: bool = True
    enable_explainability: bool = True
    high_risk_threshold: float = 0.6
    async_timeout_ms: int = 100
    parallel_modules: bool = True
    max_workers: int = 4


class AdvancedModulesOrchestrator:
    """
    Orquestrador de Módulos Avançados de ML
    
    Arquitetura Staged Enrichment:
    1. Modelo Base: Gatekeeper em tempo real (<50ms)
    2. Módulos Avançados: Enriquecimento condicional
       - Autoencoder: Detecção de anomalias não supervisionada
       - Sequence Analyzer: Padrões temporais
       - Mixture of Experts: Especialização por tipo
       - GNN: Análise de grafo de transações
       - Self-Explainable: Explicações LGPD
    
    Estratégia:
    - Baixo risco: Apenas modelo base
    - Médio risco: + Autoencoder + MoE
    - Alto risco: Todos os módulos
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        self._autoencoder = None
        self._sequence_analyzer = None
        self._moe = None
        self._gnn = None
        self._explainer = None
        
        self._modules_loaded = False
        self._load_modules()
        
        logger.info(f"AdvancedModulesOrchestrator initialized (v{self.VERSION})")
    
    def _load_modules(self) -> None:
        """Carrega módulos avançados de forma lazy"""
        try:
            if self.config.enable_autoencoder:
                from ml_engine.autoencoder_anomaly_detector import get_autoencoder_detector
                self._autoencoder = get_autoencoder_detector()
                logger.info("Autoencoder module loaded")
        except ImportError as e:
            logger.warning(f"Autoencoder not available: {e}")
        
        try:
            if self.config.enable_sequence_analysis:
                from ml_engine.bilstm_sequence_analyzer import get_bilstm_analyzer
                self._sequence_analyzer = get_bilstm_analyzer()
                logger.info("Sequence analyzer module loaded")
        except ImportError as e:
            logger.warning(f"Sequence analyzer not available: {e}")
        
        try:
            if self.config.enable_moe:
                from ml_engine.mixture_of_experts import get_mixture_of_experts
                self._moe = get_mixture_of_experts()
                logger.info("Mixture of Experts module loaded")
        except ImportError as e:
            logger.warning(f"MoE not available: {e}")
        
        try:
            if self.config.enable_gnn:
                from ml_engine.gnn_fraud_detector import GNNFraudDetector
                self._gnn = GNNFraudDetector()
                logger.info("GNN module loaded")
        except ImportError as e:
            logger.warning(f"GNN not available: {e}")
        
        try:
            if self.config.enable_explainability:
                from ml_engine.self_explainable_module import get_self_explainer
                self._explainer = get_self_explainer()
                logger.info("Self-explainable module loaded")
        except ImportError as e:
            logger.warning(f"Explainer not available: {e}")
        
        self._modules_loaded = True
    
    def enrich_prediction(
        self,
        transaction: Dict[str, Any],
        base_prediction: Dict[str, Any],
        user_id: Optional[str] = None,
        force_full_enrichment: bool = False
    ) -> EnrichedPrediction:
        """
        Enriquece predição base com módulos avançados
        
        Args:
            transaction: Dados da transação
            base_prediction: Predição do modelo base
            user_id: ID do usuário (para sequência)
            force_full_enrichment: Força enriquecimento completo
            
        Returns:
            Predição enriquecida
        """
        start_time = time.time()
        
        transaction_id = transaction.get('transaction_id', 
            f"TXN_{hashlib.md5(str(transaction).encode()).hexdigest()[:8]}")
        
        base_prob = base_prediction.get('fraud_probability', 0.0)
        
        enrichment_level = self._determine_enrichment_level(base_prob, force_full_enrichment)
        
        enriched = EnrichedPrediction(
            transaction_id=transaction_id,
            is_fraud=base_prediction.get('is_fraud', False),
            fraud_probability=base_prob,
            risk_score=base_prediction.get('risk_score', 0.0),
            risk_level=base_prediction.get('risk_level', 'low'),
            confidence=base_prediction.get('confidence', 0.0),
            processing_time_ms=0.0,
            model_version=f"orchestrator-{self.VERSION}",
            detection_reasons=base_prediction.get('detection_reason', []),
            enrichment_level=enrichment_level.value,
            timestamp=datetime.now().isoformat()
        )
        
        if enrichment_level == EnrichmentLevel.BASIC:
            enriched.processing_time_ms = (time.time() - start_time) * 1000
            return enriched
        
        if self.config.parallel_modules:
            self._enrich_parallel(enriched, transaction, user_id, enrichment_level)
        else:
            self._enrich_sequential(enriched, transaction, user_id, enrichment_level)
        
        self._adjust_final_prediction(enriched)
        
        enriched.processing_time_ms = (time.time() - start_time) * 1000
        
        return enriched
    
    def _determine_enrichment_level(
        self,
        base_probability: float,
        force_full: bool
    ) -> EnrichmentLevel:
        """Determina nível de enriquecimento baseado no risco"""
        if force_full:
            return EnrichmentLevel.FULL
        
        if base_probability < 0.3:
            return EnrichmentLevel.BASIC
        elif base_probability < self.config.high_risk_threshold:
            return EnrichmentLevel.STANDARD
        else:
            return EnrichmentLevel.FULL
    
    def _enrich_parallel(
        self,
        enriched: EnrichedPrediction,
        transaction: Dict[str, Any],
        user_id: Optional[str],
        level: EnrichmentLevel
    ) -> None:
        """Enriquece em paralelo usando ThreadPoolExecutor"""
        futures = {}
        
        if level in [EnrichmentLevel.STANDARD, EnrichmentLevel.FULL]:
            if self._autoencoder:
                futures['autoencoder'] = self.executor.submit(
                    self._run_autoencoder, transaction
                )
            
            if self._moe:
                futures['moe'] = self.executor.submit(
                    self._run_moe, transaction
                )
        
        if level == EnrichmentLevel.FULL:
            if self._sequence_analyzer and user_id:
                futures['sequence'] = self.executor.submit(
                    self._run_sequence_analysis, user_id, transaction
                )
            
            if self._gnn:
                futures['gnn'] = self.executor.submit(
                    self._run_gnn, transaction
                )
            
            if self._explainer:
                futures['explainer'] = self.executor.submit(
                    self._run_explainer, transaction, {
                        'is_fraud': enriched.is_fraud,
                        'fraud_probability': enriched.fraud_probability,
                        'risk_score': enriched.risk_score
                    }
                )
        
        timeout_sec = self.config.async_timeout_ms / 1000.0
        
        for name, future in futures.items():
            try:
                result = future.result(timeout=timeout_sec)
                self._apply_result(enriched, name, result)
            except FuturesTimeoutError:
                logger.warning(f"Module {name} timed out")
                enriched.advanced_insights[f"{name}_status"] = "timeout"
            except Exception as e:
                logger.warning(f"Module {name} failed: {e}")
                enriched.advanced_insights[f"{name}_status"] = f"error: {str(e)}"
    
    def _enrich_sequential(
        self,
        enriched: EnrichedPrediction,
        transaction: Dict[str, Any],
        user_id: Optional[str],
        level: EnrichmentLevel
    ) -> None:
        """Enriquece sequencialmente"""
        if level in [EnrichmentLevel.STANDARD, EnrichmentLevel.FULL]:
            if self._autoencoder:
                result = self._run_autoencoder(transaction)
                self._apply_result(enriched, 'autoencoder', result)
            
            if self._moe:
                result = self._run_moe(transaction)
                self._apply_result(enriched, 'moe', result)
        
        if level == EnrichmentLevel.FULL:
            if self._sequence_analyzer and user_id:
                result = self._run_sequence_analysis(user_id, transaction)
                self._apply_result(enriched, 'sequence', result)
            
            if self._gnn:
                result = self._run_gnn(transaction)
                self._apply_result(enriched, 'gnn', result)
            
            if self._explainer:
                result = self._run_explainer(transaction, {
                    'is_fraud': enriched.is_fraud,
                    'fraud_probability': enriched.fraud_probability
                })
                self._apply_result(enriched, 'explainer', result)
    
    def _run_autoencoder(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Executa detector de anomalias"""
        if not self._autoencoder:
            return {}
        
        try:
            result = self._autoencoder.detect_anomaly(transaction)
            return {
                'anomaly_score': result.anomaly_score,
                'is_anomaly': result.is_anomaly,
                'anomaly_type': result.anomaly_type,
                'percentile': result.percentile_rank
            }
        except Exception as e:
            logger.warning(f"Autoencoder error: {e}")
            return {'error': str(e)}
    
    def _run_sequence_analysis(
        self,
        user_id: str,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executa análise de sequência"""
        if not self._sequence_analyzer:
            return {}
        
        try:
            result = self._sequence_analyzer.analyze_sequence(user_id, transaction)
            return {
                'sequence_risk': result.sequence_risk_score,
                'temporal_anomaly': result.temporal_anomaly_score,
                'velocity_anomaly': result.velocity_anomaly_score,
                'patterns': [p.pattern_type for p in result.detected_patterns],
                'recommendation': result.recommendation
            }
        except Exception as e:
            logger.warning(f"Sequence analyzer error: {e}")
            return {'error': str(e)}
    
    def _run_moe(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Executa Mixture of Experts"""
        if not self._moe:
            return {}
        
        try:
            result = self._moe.predict(transaction)
            return {
                'moe_probability': result.final_probability,
                'consensus': result.consensus_level,
                'routing_decision': result.routing_decision,
                'expert_weights': result.expert_weights
            }
        except Exception as e:
            logger.warning(f"MoE error: {e}")
            return {'error': str(e)}
    
    def _run_gnn(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Executa análise GNN"""
        if not self._gnn:
            return {}
        
        try:
            result = self._gnn.analyze_transaction(transaction)
            return {
                'graph_risk': result.graph_risk_score if hasattr(result, 'graph_risk_score') else 0.0,
                'community_risk': result.community_risk if hasattr(result, 'community_risk') else 0.0,
                'suspicious_patterns': result.suspicious_patterns if hasattr(result, 'suspicious_patterns') else []
            }
        except Exception as e:
            logger.warning(f"GNN error: {e}")
            return {'error': str(e)}
    
    def _run_explainer(
        self,
        transaction: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executa explicador auto-interpretável"""
        if not self._explainer:
            return {}
        
        try:
            result = self._explainer.generate_explanation(transaction, prediction)
            return {
                'explanation_text': result.natural_language_explanation,
                'feature_importance': result.feature_importance,
                'rule_triggers': result.rule_triggers,
                'lgpd_audit': result.lgpd_audit_trail
            }
        except Exception as e:
            logger.warning(f"Explainer error: {e}")
            return {'error': str(e)}
    
    def _apply_result(
        self,
        enriched: EnrichedPrediction,
        module_name: str,
        result: Dict[str, Any]
    ) -> None:
        """Aplica resultado do módulo à predição enriquecida"""
        if 'error' in result:
            enriched.advanced_insights[f"{module_name}_error"] = result['error']
            return
        
        if module_name == 'autoencoder':
            enriched.autoencoder_anomaly_score = result.get('anomaly_score')
            enriched.advanced_insights['autoencoder'] = result
        
        elif module_name == 'sequence':
            enriched.sequence_risk_score = result.get('sequence_risk')
            enriched.advanced_insights['sequence'] = result
        
        elif module_name == 'moe':
            enriched.moe_consensus = result.get('consensus')
            enriched.advanced_insights['moe'] = result
        
        elif module_name == 'gnn':
            enriched.gnn_community_risk = result.get('community_risk')
            enriched.advanced_insights['gnn'] = result
        
        elif module_name == 'explainer':
            enriched.explanation = {
                'text': result.get('explanation_text'),
                'features': result.get('feature_importance'),
                'rules': result.get('rule_triggers')
            }
            enriched.lgpd_audit = result.get('lgpd_audit')
    
    def _adjust_final_prediction(self, enriched: EnrichedPrediction) -> None:
        """Ajusta predição final baseado nos módulos avançados"""
        adjustments = []
        combined_score = enriched.fraud_probability
        weight_sum = 1.0
        
        if enriched.autoencoder_anomaly_score is not None:
            if enriched.autoencoder_anomaly_score > 0.8:
                combined_score += enriched.autoencoder_anomaly_score * 0.2
                weight_sum += 0.2
                adjustments.append("Anomalia detectada pelo autoencoder")
        
        if enriched.sequence_risk_score is not None:
            if enriched.sequence_risk_score > 0.7:
                combined_score += enriched.sequence_risk_score * 0.15
                weight_sum += 0.15
                adjustments.append("Padrão suspeito em sequência")
        
        if enriched.moe_consensus is not None:
            moe_prob = enriched.advanced_insights.get('moe', {}).get('moe_probability', 0.5)
            if enriched.moe_consensus > 0.7 and moe_prob > 0.6:
                combined_score += moe_prob * 0.15
                weight_sum += 0.15
                adjustments.append(f"Consenso MoE alto ({enriched.moe_consensus:.1%})")
        
        if enriched.gnn_community_risk is not None:
            if enriched.gnn_community_risk > 0.7:
                combined_score += enriched.gnn_community_risk * 0.1
                weight_sum += 0.1
                adjustments.append("Comunidade de risco detectada")
        
        if weight_sum > 1.0:
            enriched.fraud_probability = min(1.0, combined_score / weight_sum)
            enriched.is_fraud = enriched.fraud_probability > 0.5
            enriched.risk_score = enriched.fraud_probability * 100
            
            if enriched.fraud_probability > 0.8:
                enriched.risk_level = 'critical'
            elif enriched.fraud_probability > 0.6:
                enriched.risk_level = 'high'
            elif enriched.fraud_probability > 0.4:
                enriched.risk_level = 'medium'
            else:
                enriched.risk_level = 'low'
        
        enriched.detection_reasons.extend(adjustments)
        
        if adjustments:
            enriched.advanced_insights['adjustments'] = adjustments
    
    def get_module_status(self) -> Dict[str, Any]:
        """Retorna status de todos os módulos"""
        status = {
            'version': self.VERSION,
            'modules_loaded': self._modules_loaded,
            'config': {
                'enable_autoencoder': self.config.enable_autoencoder,
                'enable_sequence_analysis': self.config.enable_sequence_analysis,
                'enable_moe': self.config.enable_moe,
                'enable_gnn': self.config.enable_gnn,
                'enable_explainability': self.config.enable_explainability,
                'high_risk_threshold': self.config.high_risk_threshold,
                'parallel_modules': self.config.parallel_modules
            },
            'modules': {}
        }
        
        if self._autoencoder:
            status['modules']['autoencoder'] = self._autoencoder.get_status()
        
        if self._sequence_analyzer:
            status['modules']['sequence_analyzer'] = self._sequence_analyzer.get_status()
        
        if self._moe:
            status['modules']['mixture_of_experts'] = self._moe.get_expert_status()
        
        if self._gnn:
            status['modules']['gnn'] = {'available': True}
        
        if self._explainer:
            status['modules']['explainer'] = self._explainer.get_status()
        
        return status
    
    def train_modules(self, training_data: Any) -> Dict[str, Any]:
        """Treina todos os módulos com dados fornecidos"""
        results = {}
        
        if isinstance(training_data, dict):
            X = training_data.get('X')
            y = training_data.get('y')
        else:
            X = training_data
            y = None
        
        if self._autoencoder and X is not None:
            try:
                import pandas as pd
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                results['autoencoder'] = self._autoencoder.fit(X)
            except Exception as e:
                results['autoencoder'] = {'error': str(e)}
        
        if self._moe and X is not None and y is not None:
            try:
                import pandas as pd
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)
                results['moe'] = self._moe.fit(X, y)
            except Exception as e:
                results['moe'] = {'error': str(e)}
        
        if self._explainer and X is not None and y is not None:
            try:
                import pandas as pd
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)
                results['explainer'] = self._explainer.fit(X, y)
            except Exception as e:
                results['explainer'] = {'error': str(e)}
        
        return results


_orchestrator_instance = None

def get_orchestrator() -> AdvancedModulesOrchestrator:
    """Obtém instância singleton do orquestrador"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AdvancedModulesOrchestrator()
    return _orchestrator_instance
