"""
Sankofa Enterprise Pro - Explanation Generator
Geração de explicações em linguagem natural para decisões de fraude

Baseado em:
- XAI best practices
- Financial services regulatory requirements
- User-centric explanation design
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExplanationAudience(Enum):
    """Público-alvo da explicação"""
    CUSTOMER = "customer"  # Usuário final
    ANALYST = "analyst"  # Analista de fraude
    REGULATOR = "regulator"  # Regulador/compliance
    TECHNICAL = "technical"  # Técnico/desenvolvedor


@dataclass
class FraudExplanation:
    """Explicação de decisão de fraude"""
    decision: str  # APPROVED, BLOCKED, REVIEW
    risk_score: float
    summary: str
    key_factors: List[Dict[str, Any]]
    detailed_explanation: str
    counterfactual: str  # O que mudaria a decisão
    confidence: float
    audience: ExplanationAudience
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FactorExplanation:
    """Explicação de um fator de risco"""
    factor_name: str
    factor_value: Any
    contribution: float  # Contribuição para o score
    direction: str  # "increases_risk" ou "decreases_risk"
    explanation: str
    typical_value: Optional[Any] = None
    percentile: Optional[float] = None


class ExplanationGenerator:
    """
    Gerador de Explicações para Decisões de Fraude

    Gera explicações adaptadas para diferentes públicos:
    - Clientes: Explicações simples e acionáveis
    - Analistas: Detalhes técnicos e evidências
    - Reguladores: Compliance e auditoria
    """

    VERSION = "1.0.0"

    # Templates de explicação por audiência
    TEMPLATES = {
        ExplanationAudience.CUSTOMER: {
            "blocked": (
                "Por motivos de segurança, esta transação foi bloqueada. "
                "Isso pode acontecer quando detectamos atividade incomum em sua conta. "
                "Se você não reconhece esta transação, entre em contato conosco. "
                "Se foi você, pode tentar novamente ou usar outro método de pagamento."
            ),
            "review": (
                "Esta transação está em análise por motivos de segurança. "
                "Isso pode levar alguns minutos. Se a análise demorar mais, "
                "entraremos em contato."
            ),
            "approved_high_risk": (
                "Transação aprovada. Por segurança, fique atento a movimentações "
                "não reconhecidas em sua conta."
            )
        },
        ExplanationAudience.ANALYST: {
            "header": "ANÁLISE DE FRAUDE - CASO #{case_id}",
            "sections": [
                "RESUMO DA DECISÃO",
                "FATORES DE RISCO PRINCIPAIS",
                "ANÁLISE DETALHADA",
                "COMPARAÇÃO COM BASELINE",
                "RECOMENDAÇÕES",
                "EVIDÊNCIAS"
            ]
        },
        ExplanationAudience.REGULATOR: {
            "header": "RELATÓRIO DE COMPLIANCE - DECISÃO DE FRAUDE",
            "required_fields": [
                "decision_timestamp",
                "model_version",
                "features_used",
                "risk_score",
                "threshold_applied",
                "human_review",
                "audit_trail"
            ]
        }
    }

    # Mapeamento de fatores para explicações
    FACTOR_EXPLANATIONS = {
        "amount_deviation": {
            "high": "Valor significativamente acima do padrão do cliente",
            "low": "Valor dentro do padrão habitual"
        },
        "velocity_24h": {
            "high": "Muitas transações em curto período",
            "low": "Frequência normal de transações"
        },
        "new_device": {
            "true": "Transação de dispositivo nunca usado antes",
            "false": "Dispositivo já conhecido"
        },
        "unusual_hour": {
            "true": "Horário incomum para este cliente",
            "false": "Horário habitual"
        },
        "new_receiver": {
            "true": "Primeiro pagamento para este destinatário",
            "false": "Destinatário já conhecido"
        },
        "location_anomaly": {
            "true": "Localização diferente do padrão",
            "false": "Localização habitual"
        },
        "behavioral_anomaly": {
            "true": "Comportamento de navegação atípico",
            "false": "Comportamento normal"
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        logger.info(f"ExplanationGenerator v{self.VERSION} initialized")

    def generate(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        audience: ExplanationAudience = ExplanationAudience.ANALYST,
        case_id: Optional[str] = None
    ) -> FraudExplanation:
        """
        Gerar explicação para decisão de fraude

        Args:
            decision: Decisão tomada (APPROVED, BLOCKED, REVIEW)
            risk_score: Score de risco
            factors: Fatores que contribuíram para a decisão
            transaction: Dados da transação
            audience: Público-alvo da explicação
            case_id: ID do caso (opcional)

        Returns:
            FraudExplanation com explicação completa
        """
        # 1. Gerar resumo
        summary = self._generate_summary(decision, risk_score, audience)

        # 2. Processar fatores principais
        key_factors = self._process_factors(factors, audience)

        # 3. Gerar explicação detalhada
        detailed = self._generate_detailed_explanation(
            decision, risk_score, factors, transaction, audience, case_id
        )

        # 4. Gerar counterfactual
        counterfactual = self._generate_counterfactual(
            decision, risk_score, factors, audience
        )

        # 5. Gerar recomendações
        recommendations = self._generate_recommendations(
            decision, risk_score, factors, audience
        )

        # 6. Calcular confiança
        confidence = self._calculate_confidence(factors)

        return FraudExplanation(
            decision=decision,
            risk_score=risk_score,
            summary=summary,
            key_factors=key_factors,
            detailed_explanation=detailed,
            counterfactual=counterfactual,
            confidence=confidence,
            audience=audience,
            recommendations=recommendations
        )

    def _generate_summary(
        self,
        decision: str,
        risk_score: float,
        audience: ExplanationAudience
    ) -> str:
        """Gerar resumo da decisão"""

        if audience == ExplanationAudience.CUSTOMER:
            templates = self.TEMPLATES[ExplanationAudience.CUSTOMER]
            if decision == "BLOCKED":
                return templates["blocked"]
            elif decision == "REVIEW":
                return templates["review"]
            else:
                if risk_score > 0.5:
                    return templates["approved_high_risk"]
                return "Transação aprovada com sucesso."

        elif audience == ExplanationAudience.ANALYST:
            risk_level = self._get_risk_level(risk_score)
            return (
                f"Decisão: {decision} | Risk Score: {risk_score:.2f} ({risk_level}) | "
                f"Requer revisão: {'Sim' if decision == 'REVIEW' else 'Não'}"
            )

        elif audience == ExplanationAudience.REGULATOR:
            return (
                f"Transação avaliada pelo sistema antifraude. "
                f"Decisão automatizada: {decision}. "
                f"Score de risco: {risk_score:.4f}. "
                f"Threshold aplicado: {self._get_threshold(decision)}."
            )

        else:  # TECHNICAL
            return (
                f"Decision={decision}, RiskScore={risk_score:.4f}, "
                f"ModelVersion={self.VERSION}"
            )

    def _process_factors(
        self,
        factors: List[Dict[str, Any]],
        audience: ExplanationAudience
    ) -> List[Dict[str, Any]]:
        """Processar e explicar fatores"""
        processed = []

        for factor in factors:
            factor_name = factor.get("name", "unknown")
            factor_value = factor.get("value")
            contribution = factor.get("contribution", 0)

            # Gerar explicação do fator
            explanation = self._explain_factor(
                factor_name, factor_value, contribution, audience
            )

            processed.append({
                "name": factor_name,
                "value": factor_value,
                "contribution": contribution,
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                "explanation": explanation,
                "importance_rank": abs(contribution)
            })

        # Ordenar por importância
        processed.sort(key=lambda x: x["importance_rank"], reverse=True)

        return processed[:10]  # Top 10 fatores

    def _explain_factor(
        self,
        factor_name: str,
        factor_value: Any,
        contribution: float,
        audience: ExplanationAudience
    ) -> str:
        """Explicar um fator específico"""

        # Obter template de explicação
        factor_templates = self.FACTOR_EXPLANATIONS.get(factor_name, {})

        if factor_templates:
            # Determinar chave baseado no valor
            if isinstance(factor_value, bool):
                key = "true" if factor_value else "false"
            elif isinstance(factor_value, (int, float)):
                key = "high" if contribution > 0 else "low"
            else:
                key = "high"

            explanation = factor_templates.get(key, f"Fator: {factor_name}")
        else:
            # Explicação genérica
            direction = "aumenta" if contribution > 0 else "diminui"
            explanation = f"{factor_name} {direction} o risco"

        # Adaptar para audiência
        if audience == ExplanationAudience.CUSTOMER:
            explanation = self._simplify_explanation(explanation)
        elif audience == ExplanationAudience.TECHNICAL:
            explanation = f"{factor_name}={factor_value}, contrib={contribution:.4f}"

        return explanation

    def _simplify_explanation(self, explanation: str) -> str:
        """Simplificar explicação para cliente"""
        # Substituir termos técnicos
        replacements = {
            "velocity": "frequência de transações",
            "deviation": "variação",
            "anomaly": "comportamento diferente",
            "baseline": "padrão habitual",
            "threshold": "limite"
        }

        result = explanation
        for technical, simple in replacements.items():
            result = result.replace(technical, simple)

        return result

    def _generate_detailed_explanation(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        audience: ExplanationAudience,
        case_id: Optional[str]
    ) -> str:
        """Gerar explicação detalhada"""

        if audience == ExplanationAudience.CUSTOMER:
            return self._customer_detailed_explanation(decision, risk_score)

        elif audience == ExplanationAudience.ANALYST:
            return self._analyst_detailed_explanation(
                decision, risk_score, factors, transaction, case_id
            )

        elif audience == ExplanationAudience.REGULATOR:
            return self._regulator_detailed_explanation(
                decision, risk_score, factors, transaction, case_id
            )

        else:  # TECHNICAL
            return self._technical_detailed_explanation(
                decision, risk_score, factors, transaction
            )

    def _customer_detailed_explanation(
        self,
        decision: str,
        risk_score: float
    ) -> str:
        """Explicação detalhada para cliente"""
        if decision == "BLOCKED":
            return (
                "Para sua proteção, bloqueamos esta transação porque detectamos "
                "algumas características incomuns. Isso não significa que há algo "
                "errado com sua conta - é apenas uma medida de segurança. "
                "Se foi você quem tentou fazer esta transação, pode tentar novamente "
                "ou entrar em contato conosco para liberar."
            )
        elif decision == "REVIEW":
            return (
                "Estamos verificando esta transação para garantir sua segurança. "
                "Este processo é rápido e automático na maioria dos casos."
            )
        else:
            return "Transação processada normalmente."

    def _analyst_detailed_explanation(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        case_id: Optional[str]
    ) -> str:
        """Explicação detalhada para analista"""
        lines = []

        lines.append(f"{'='*60}")
        lines.append(f"ANÁLISE DE FRAUDE - CASO #{case_id or 'N/A'}")
        lines.append(f"{'='*60}")

        lines.append(f"\n** DECISÃO: {decision} **")
        lines.append(f"Risk Score: {risk_score:.4f}")
        lines.append(f"Risk Level: {self._get_risk_level(risk_score)}")

        lines.append(f"\n** TRANSAÇÃO **")
        lines.append(f"Valor: R$ {transaction.get('amount', 0):,.2f}")
        lines.append(f"Canal: {transaction.get('channel', 'N/A')}")
        lines.append(f"Tipo: {transaction.get('type', 'N/A')}")

        lines.append(f"\n** FATORES PRINCIPAIS **")
        for i, factor in enumerate(factors[:5], 1):
            contrib = factor.get("contribution", 0)
            direction = "↑" if contrib > 0 else "↓"
            lines.append(
                f"{i}. {factor.get('name')}: {factor.get('value')} "
                f"[{direction} {abs(contrib):.3f}]"
            )

        lines.append(f"\n** INDICADORES DE RISCO **")
        high_risk = [f for f in factors if f.get("contribution", 0) > 0.1]
        for factor in high_risk:
            lines.append(f"  - {factor.get('name')}: {factor.get('value')}")

        return "\n".join(lines)

    def _regulator_detailed_explanation(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        case_id: Optional[str]
    ) -> str:
        """Explicação detalhada para regulador"""
        lines = []

        lines.append("RELATÓRIO DE COMPLIANCE - DECISÃO AUTOMATIZADA DE FRAUDE")
        lines.append(f"Data/Hora: {datetime.now().isoformat()}")
        lines.append(f"ID do Caso: {case_id or 'N/A'}")
        lines.append(f"Versão do Modelo: {self.VERSION}")

        lines.append(f"\n1. DECISÃO")
        lines.append(f"   Resultado: {decision}")
        lines.append(f"   Score de Risco: {risk_score:.6f}")
        lines.append(f"   Threshold Aplicado: {self._get_threshold(decision)}")

        lines.append(f"\n2. METODOLOGIA")
        lines.append(f"   Modelo: Ensemble (RF + GBM + GNN)")
        lines.append(f"   Features: {len(factors)} variáveis analisadas")
        lines.append(f"   Explicabilidade: SHAP values")

        lines.append(f"\n3. FATORES DETERMINANTES")
        for factor in factors[:10]:
            lines.append(
                f"   - {factor.get('name')}: "
                f"valor={factor.get('value')}, "
                f"contribuição={factor.get('contribution', 0):.4f}"
            )

        lines.append(f"\n4. CONFORMIDADE")
        lines.append(f"   LGPD: Dados processados conforme Art. 7º, II")
        lines.append(f"   BACEN: Res. 4.658/2018 - Política de segurança")
        lines.append(f"   PCI-DSS: Compliant")

        return "\n".join(lines)

    def _technical_detailed_explanation(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        transaction: Dict[str, Any]
    ) -> str:
        """Explicação técnica"""
        import json
        return json.dumps({
            "decision": decision,
            "risk_score": risk_score,
            "factors": factors,
            "transaction_id": transaction.get("id"),
            "model_version": self.VERSION,
            "timestamp": datetime.now().isoformat()
        }, indent=2)

    def _generate_counterfactual(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        audience: ExplanationAudience
    ) -> str:
        """Gerar explicação contrafactual"""

        if decision == "APPROVED":
            # O que faria bloquear
            high_impact = [
                f for f in factors
                if f.get("contribution", 0) < 0
            ][:3]

            if high_impact:
                changes = [f.get("name") for f in high_impact]
                return (
                    f"A transação seria bloqueada se os seguintes fatores "
                    f"fossem diferentes: {', '.join(changes)}."
                )
        else:
            # O que faria aprovar
            high_impact = [
                f for f in factors
                if f.get("contribution", 0) > 0.1
            ][:3]

            if high_impact:
                if audience == ExplanationAudience.CUSTOMER:
                    return (
                        "Para que a transação seja aprovada, recomendamos "
                        "usar um dispositivo já cadastrado ou entrar em contato "
                        "com o suporte."
                    )
                else:
                    changes = [f.get("name") for f in high_impact]
                    return (
                        f"A transação seria aprovada se os seguintes fatores "
                        f"não estivessem presentes: {', '.join(changes)}."
                    )

        return "Não há mudanças simples que alterariam a decisão."

    def _generate_recommendations(
        self,
        decision: str,
        risk_score: float,
        factors: List[Dict[str, Any]],
        audience: ExplanationAudience
    ) -> List[str]:
        """Gerar recomendações"""
        recommendations = []

        if audience == ExplanationAudience.CUSTOMER:
            if decision == "BLOCKED":
                recommendations.extend([
                    "Verifique se o dispositivo usado está cadastrado",
                    "Tente usar um método de pagamento alternativo",
                    "Entre em contato com o suporte se precisar de ajuda"
                ])
            elif decision == "REVIEW":
                recommendations.append(
                    "Aguarde a conclusão da análise (geralmente menos de 5 minutos)"
                )

        elif audience == ExplanationAudience.ANALYST:
            if risk_score > 0.8:
                recommendations.extend([
                    "Contatar cliente para verificação",
                    "Verificar transações relacionadas",
                    "Considerar bloqueio preventivo da conta"
                ])
            elif risk_score > 0.5:
                recommendations.extend([
                    "Monitorar atividade subsequente",
                    "Adicionar nota ao perfil do cliente"
                ])

        return recommendations

    def _calculate_confidence(self, factors: List[Dict[str, Any]]) -> float:
        """Calcular confiança na explicação"""
        if not factors:
            return 0.5

        # Baseado na magnitude das contribuições
        contributions = [abs(f.get("contribution", 0)) for f in factors]
        avg_contribution = sum(contributions) / len(contributions)

        # Mais fatores com alta contribuição = mais confiança
        return min(avg_contribution * 2 + 0.5, 1.0)

    def _get_risk_level(self, risk_score: float) -> str:
        """Obter nível de risco textual"""
        if risk_score >= 0.8:
            return "CRÍTICO"
        elif risk_score >= 0.6:
            return "ALTO"
        elif risk_score >= 0.4:
            return "MÉDIO"
        elif risk_score >= 0.2:
            return "BAIXO"
        else:
            return "MUITO BAIXO"

    def _get_threshold(self, decision: str) -> str:
        """Obter threshold aplicado"""
        thresholds = {
            "BLOCKED": ">= 0.80",
            "REVIEW": ">= 0.50 e < 0.80",
            "APPROVED": "< 0.50"
        }
        return thresholds.get(decision, "N/A")

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas"""
        return {
            "version": self.VERSION,
            "supported_audiences": [a.value for a in ExplanationAudience],
            "factor_explanations_count": len(self.FACTOR_EXPLANATIONS)
        }


def create_explanation_generator(
    config: Optional[Dict[str, Any]] = None
) -> ExplanationGenerator:
    """Factory function para criar ExplanationGenerator"""
    return ExplanationGenerator(config)
