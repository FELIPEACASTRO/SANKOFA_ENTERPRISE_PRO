"""
Sankofa Enterprise Pro - Fairness Analyzer
Análise de viés e fairness em modelos ML
Conformidade com práticas éticas de ML e regulações
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import logging
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    """Métricas para um subgrupo específico"""
    subgroup_name: str
    subgroup_value: str
    sample_size: int
    fraud_rate: float
    prediction_rate: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    selection_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class FairnessMetrics:
    """Métricas de fairness do modelo"""
    demographic_parity_ratio: float
    equalized_odds_ratio: float
    predictive_parity_ratio: float
    calibration_ratio: float
    
    disparate_impact: float
    
    is_fair_demographic_parity: bool
    is_fair_equalized_odds: bool
    is_fair_predictive_parity: bool
    
    overall_fairness_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiasReport:
    """Relatório completo de viés"""
    model_version: str
    analysis_timestamp: str
    
    protected_attributes: List[str]
    reference_group: str
    
    subgroup_metrics: List[SubgroupMetrics]
    fairness_metrics: FairnessMetrics
    
    recommendations: List[str]
    risk_level: str
    compliance_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['subgroup_metrics'] = [s.to_dict() for s in self.subgroup_metrics]
        data['fairness_metrics'] = self.fairness_metrics.to_dict()
        return data


class FairnessAnalyzer:
    """
    Analisador de Fairness e Viés em Modelos ML
    
    Features:
    - Análise de disparidade demográfica
    - Equalized odds e predictive parity
    - Detecção de viés por subgrupo
    - Recomendações de mitigação
    - Conformidade com regulações (LGPD, etc)
    
    Métricas Implementadas:
    - Demographic Parity Ratio (DPR)
    - Equalized Odds Ratio (EOR)
    - Predictive Parity Ratio (PPR)
    - Disparate Impact (DI)
    - Calibration Ratio (CR)
    """
    
    VERSION = "1.0.0"
    
    FAIRNESS_THRESHOLD = 0.8
    DISPARATE_IMPACT_THRESHOLD = 0.8
    
    PROTECTED_ATTRIBUTES_BRAZIL = [
        'regiao',
        'estado',
        'faixa_etaria',
        'tipo_conta',
        'canal_origem'
    ]
    
    def __init__(
        self,
        fairness_threshold: float = 0.8,
        protected_attributes: Optional[List[str]] = None
    ):
        self.fairness_threshold = fairness_threshold
        self.protected_attributes = protected_attributes or self.PROTECTED_ATTRIBUTES_BRAZIL
        
        logger.info(f"FairnessAnalyzer initialized v{self.VERSION}")
    
    def analyze(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: Optional[List[bool]] = None,
        reference_group: Optional[str] = None
    ) -> BiasReport:
        """
        Analisa fairness das predições
        
        Args:
            predictions: Lista de predições com atributos protegidos
            ground_truth: Labels reais (opcional, para métricas completas)
            reference_group: Grupo de referência para comparação
            
        Returns:
            BiasReport com métricas e recomendações
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty")
        
        subgroup_data = self._segment_by_protected_attributes(predictions, ground_truth)
        
        subgroup_metrics = []
        for (attr, value), data in subgroup_data.items():
            metrics = self._calculate_subgroup_metrics(attr, value, data)
            subgroup_metrics.append(metrics)
        
        if reference_group is None:
            reference_group = self._determine_reference_group(subgroup_metrics)
        
        fairness_metrics = self._calculate_fairness_metrics(
            subgroup_metrics,
            reference_group
        )
        
        recommendations = self._generate_recommendations(
            subgroup_metrics,
            fairness_metrics
        )
        
        risk_level = self._determine_risk_level(fairness_metrics)
        compliance_status = self._check_compliance(fairness_metrics)
        
        return BiasReport(
            model_version=self.VERSION,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            protected_attributes=self.protected_attributes,
            reference_group=reference_group,
            subgroup_metrics=subgroup_metrics,
            fairness_metrics=fairness_metrics,
            recommendations=recommendations,
            risk_level=risk_level,
            compliance_status=compliance_status
        )
    
    def _segment_by_protected_attributes(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: Optional[List[bool]]
    ) -> Dict[Tuple[str, str], Dict[str, List]]:
        """Segmenta dados por atributos protegidos"""
        segments: Dict[Tuple[str, str], Dict[str, List]] = defaultdict(
            lambda: {'predictions': [], 'ground_truth': [], 'probabilities': []}
        )
        
        for i, pred in enumerate(predictions):
            for attr in self.protected_attributes:
                if attr in pred:
                    value = str(pred[attr])
                    key = (attr, value)
                    
                    segments[key]['predictions'].append(
                        pred.get('is_fraud', pred.get('prediction', False))
                    )
                    segments[key]['probabilities'].append(
                        pred.get('fraud_probability', pred.get('probability', 0.0))
                    )
                    
                    if ground_truth and i < len(ground_truth):
                        segments[key]['ground_truth'].append(ground_truth[i])
        
        return dict(segments)
    
    def _calculate_subgroup_metrics(
        self,
        attr: str,
        value: str,
        data: Dict[str, List]
    ) -> SubgroupMetrics:
        """Calcula métricas para um subgrupo"""
        predictions = data['predictions']
        ground_truth = data['ground_truth']
        probabilities = data['probabilities']
        
        n = len(predictions)
        if n == 0:
            return SubgroupMetrics(
                subgroup_name=attr,
                subgroup_value=value,
                sample_size=0,
                fraud_rate=0.0,
                prediction_rate=0.0,
                true_positive_rate=0.0,
                false_positive_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                selection_rate=0.0
            )
        
        prediction_rate = sum(predictions) / n if n > 0 else 0
        
        fraud_rate = 0.0
        tpr = 0.0
        fpr = 0.0
        precision = 0.0
        recall = 0.0
        
        if ground_truth:
            fraud_rate = sum(ground_truth) / len(ground_truth) if ground_truth else 0
            
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
            tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        selection_rate = prediction_rate
        
        return SubgroupMetrics(
            subgroup_name=attr,
            subgroup_value=value,
            sample_size=n,
            fraud_rate=round(fraud_rate, 4),
            prediction_rate=round(prediction_rate, 4),
            true_positive_rate=round(tpr, 4),
            false_positive_rate=round(fpr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            selection_rate=round(selection_rate, 4)
        )
    
    def _determine_reference_group(self, subgroup_metrics: List[SubgroupMetrics]) -> str:
        """Determina grupo de referência (maior amostra)"""
        if not subgroup_metrics:
            return "unknown"
        
        largest = max(subgroup_metrics, key=lambda x: x.sample_size)
        return f"{largest.subgroup_name}:{largest.subgroup_value}"
    
    def _calculate_fairness_metrics(
        self,
        subgroup_metrics: List[SubgroupMetrics],
        reference_group: str
    ) -> FairnessMetrics:
        """Calcula métricas de fairness"""
        
        ref_attr, ref_value = reference_group.split(':') if ':' in reference_group else ('', '')
        ref_metrics = None
        
        for sm in subgroup_metrics:
            if sm.subgroup_name == ref_attr and sm.subgroup_value == ref_value:
                ref_metrics = sm
                break
        
        if ref_metrics is None and subgroup_metrics:
            ref_metrics = max(subgroup_metrics, key=lambda x: x.sample_size)
        
        if ref_metrics is None:
            return FairnessMetrics(
                demographic_parity_ratio=1.0,
                equalized_odds_ratio=1.0,
                predictive_parity_ratio=1.0,
                calibration_ratio=1.0,
                disparate_impact=1.0,
                is_fair_demographic_parity=True,
                is_fair_equalized_odds=True,
                is_fair_predictive_parity=True,
                overall_fairness_score=1.0
            )
        
        dpr_values = []
        eor_values = []
        ppr_values = []
        di_values = []
        
        for sm in subgroup_metrics:
            if sm.sample_size < 30:
                continue
            
            if ref_metrics.selection_rate > 0:
                dpr = sm.selection_rate / ref_metrics.selection_rate
                dpr_values.append(min(dpr, 1/dpr) if dpr > 0 else 0)
            
            if ref_metrics.true_positive_rate > 0:
                eor = sm.true_positive_rate / ref_metrics.true_positive_rate
                eor_values.append(min(eor, 1/eor) if eor > 0 else 0)
            
            if ref_metrics.precision > 0:
                ppr = sm.precision / ref_metrics.precision
                ppr_values.append(min(ppr, 1/ppr) if ppr > 0 else 0)
            
            if ref_metrics.prediction_rate > 0:
                di = sm.prediction_rate / ref_metrics.prediction_rate
                di_values.append(di)
        
        avg_dpr = sum(dpr_values) / len(dpr_values) if dpr_values else 1.0
        avg_eor = sum(eor_values) / len(eor_values) if eor_values else 1.0
        avg_ppr = sum(ppr_values) / len(ppr_values) if ppr_values else 1.0
        min_di = min(di_values) if di_values else 1.0
        
        is_fair_dp = avg_dpr >= self.fairness_threshold
        is_fair_eo = avg_eor >= self.fairness_threshold
        is_fair_pp = avg_ppr >= self.fairness_threshold
        
        overall_score = (avg_dpr + avg_eor + avg_ppr) / 3
        
        return FairnessMetrics(
            demographic_parity_ratio=round(avg_dpr, 4),
            equalized_odds_ratio=round(avg_eor, 4),
            predictive_parity_ratio=round(avg_ppr, 4),
            calibration_ratio=round(avg_ppr, 4),
            disparate_impact=round(min_di, 4),
            is_fair_demographic_parity=is_fair_dp,
            is_fair_equalized_odds=is_fair_eo,
            is_fair_predictive_parity=is_fair_pp,
            overall_fairness_score=round(overall_score, 4)
        )
    
    def _generate_recommendations(
        self,
        subgroup_metrics: List[SubgroupMetrics],
        fairness_metrics: FairnessMetrics
    ) -> List[str]:
        """Gera recomendações de mitigação de viés"""
        recommendations = []
        
        if not fairness_metrics.is_fair_demographic_parity:
            recommendations.append(
                "ALERTA: Disparidade demográfica detectada. "
                "Considerar re-balanceamento de dados ou threshold diferenciado por grupo."
            )
        
        if not fairness_metrics.is_fair_equalized_odds:
            recommendations.append(
                "ALERTA: Taxas de erro desiguais entre grupos. "
                "Verificar features correlacionadas com atributos protegidos."
            )
        
        if not fairness_metrics.is_fair_predictive_parity:
            recommendations.append(
                "ALERTA: Precisão varia significativamente entre grupos. "
                "Considerar calibração de probabilidades por subgrupo."
            )
        
        if fairness_metrics.disparate_impact < self.DISPARATE_IMPACT_THRESHOLD:
            recommendations.append(
                "CRÍTICO: Impacto disparado abaixo do threshold legal (80%). "
                "Revisar modelo para conformidade regulatória."
            )
        
        small_groups = [sm for sm in subgroup_metrics if sm.sample_size < 100]
        if small_groups:
            groups = ", ".join([f"{sm.subgroup_name}:{sm.subgroup_value}" for sm in small_groups[:3]])
            recommendations.append(
                f"AVISO: Grupos com amostra pequena detectados ({groups}). "
                "Métricas podem ser instáveis. Coletar mais dados."
            )
        
        high_fpr_groups = [sm for sm in subgroup_metrics if sm.false_positive_rate > 0.3]
        if high_fpr_groups:
            groups = ", ".join([f"{sm.subgroup_name}:{sm.subgroup_value}" for sm in high_fpr_groups[:3]])
            recommendations.append(
                f"ALERTA: Taxa de falso positivo elevada em grupos: {groups}. "
                "Pode indicar viés sistemático."
            )
        
        if not recommendations:
            recommendations.append(
                "OK: Nenhum viés significativo detectado nas métricas analisadas."
            )
        
        return recommendations
    
    def _determine_risk_level(self, fairness_metrics: FairnessMetrics) -> str:
        """Determina nível de risco de viés"""
        if fairness_metrics.disparate_impact < 0.6:
            return "CRITICAL"
        elif fairness_metrics.overall_fairness_score < 0.7:
            return "HIGH"
        elif fairness_metrics.overall_fairness_score < 0.85:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_compliance(self, fairness_metrics: FairnessMetrics) -> str:
        """Verifica conformidade regulatória"""
        if fairness_metrics.disparate_impact >= 0.8 and fairness_metrics.overall_fairness_score >= 0.8:
            return "COMPLIANT"
        elif fairness_metrics.disparate_impact >= 0.6:
            return "PARTIALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"
    
    def compare_models(
        self,
        model_a_predictions: List[Dict[str, Any]],
        model_b_predictions: List[Dict[str, Any]],
        ground_truth: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """Compara fairness entre dois modelos"""
        report_a = self.analyze(model_a_predictions, ground_truth)
        report_b = self.analyze(model_b_predictions, ground_truth)
        
        comparison = {
            'model_a': report_a.to_dict(),
            'model_b': report_b.to_dict(),
            'comparison': {
                'demographic_parity_diff': abs(
                    report_a.fairness_metrics.demographic_parity_ratio -
                    report_b.fairness_metrics.demographic_parity_ratio
                ),
                'equalized_odds_diff': abs(
                    report_a.fairness_metrics.equalized_odds_ratio -
                    report_b.fairness_metrics.equalized_odds_ratio
                ),
                'overall_fairness_diff': abs(
                    report_a.fairness_metrics.overall_fairness_score -
                    report_b.fairness_metrics.overall_fairness_score
                ),
                'fairer_model': 'model_a' if 
                    report_a.fairness_metrics.overall_fairness_score > 
                    report_b.fairness_metrics.overall_fairness_score 
                    else 'model_b'
            }
        }
        
        return comparison


_analyzer_instance: Optional[FairnessAnalyzer] = None


def get_fairness_analyzer() -> FairnessAnalyzer:
    """Singleton para FairnessAnalyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FairnessAnalyzer()
    return _analyzer_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_predictions = [
        {'is_fraud': True, 'fraud_probability': 0.85, 'regiao': 'Sudeste', 'faixa_etaria': 'adulto'},
        {'is_fraud': False, 'fraud_probability': 0.15, 'regiao': 'Sudeste', 'faixa_etaria': 'adulto'},
        {'is_fraud': True, 'fraud_probability': 0.92, 'regiao': 'Nordeste', 'faixa_etaria': 'jovem'},
        {'is_fraud': False, 'fraud_probability': 0.25, 'regiao': 'Nordeste', 'faixa_etaria': 'jovem'},
        {'is_fraud': True, 'fraud_probability': 0.78, 'regiao': 'Sul', 'faixa_etaria': 'adulto'},
        {'is_fraud': False, 'fraud_probability': 0.12, 'regiao': 'Sul', 'faixa_etaria': 'idoso'},
    ]
    
    test_ground_truth = [True, False, True, False, True, False]
    
    analyzer = FairnessAnalyzer()
    report = analyzer.analyze(test_predictions, test_ground_truth)
    
    print("Fairness Report:")
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
