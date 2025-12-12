"""
Sankofa Enterprise Pro - Scam Pattern Detector
Detecção de padrões de golpes e social engineering

Baseado em:
- FICO Scam Detection Score
- UK Finance scam taxonomy
- BioCatch social engineering detection
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScamType(Enum):
    """Tipos de scam conhecidos"""
    # Impersonation
    BANK_IMPERSONATION = "bank_impersonation"
    POLICE_IMPERSONATION = "police_impersonation"
    GOVERNMENT_IMPERSONATION = "government_impersonation"
    TECH_SUPPORT = "tech_support_scam"

    # Financial
    INVESTMENT_SCAM = "investment_scam"
    ROMANCE_SCAM = "romance_scam"
    LOTTERY_SCAM = "lottery_scam"
    INHERITANCE_SCAM = "inheritance_scam"
    CRYPTO_SCAM = "crypto_scam"

    # Social
    FAMILY_EMERGENCY = "family_emergency"
    KIDNAPPING_SCAM = "kidnapping_scam"
    JOB_OFFER_SCAM = "job_offer_scam"

    # Digital
    PHISHING = "phishing"
    SMISHING = "smishing"
    VISHING = "vishing"

    # Other
    ADVANCE_FEE = "advance_fee_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    UNKNOWN = "unknown"


@dataclass
class ScamIndicator:
    """Indicador de scam detectado"""
    indicator_type: str
    description: str
    severity: float  # 0-1
    evidence: str
    confidence: float


@dataclass
class ScamDetectionResult:
    """Resultado da detecção de scam"""
    is_scam: bool
    scam_type: ScamType
    confidence: float
    risk_score: float
    indicators: List[ScamIndicator]
    manipulation_techniques: List[str]
    urgency_score: float
    coercion_score: float
    recommended_action: str
    intervention_message: str
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class ScamPatternDetector:
    """
    Detector de Padrões de Scam

    Detecta técnicas de:
    - Social engineering
    - Urgência artificial
    - Impersonação de autoridade
    - Manipulação emocional
    - Coerção
    """

    VERSION = "1.0.0"

    # Padrões por tipo de scam (português e inglês)
    SCAM_PATTERNS = {
        ScamType.BANK_IMPERSONATION: {
            "keywords": [
                r"central\s+de\s+atendimento", r"gerente", r"banco",
                r"conta\s+bloqueada", r"atividade\s+suspeita",
                r"bank\s+manager", r"customer\s+service", r"account\s+locked"
            ],
            "weight": 0.9
        },
        ScamType.POLICE_IMPERSONATION: {
            "keywords": [
                r"polícia", r"delegado", r"mandado", r"prisão",
                r"police", r"officer", r"warrant", r"arrest"
            ],
            "weight": 0.95
        },
        ScamType.GOVERNMENT_IMPERSONATION: {
            "keywords": [
                r"receita\s+federal", r"inss", r"governo", r"imposto",
                r"tax\s+authority", r"government", r"irs", r"social\s+security"
            ],
            "weight": 0.9
        },
        ScamType.INVESTMENT_SCAM: {
            "keywords": [
                r"investimento\s+garantido", r"retorno\s+certo", r"dobre\s+seu",
                r"guaranteed\s+return", r"double\s+your", r"risk\s+free"
            ],
            "weight": 0.85
        },
        ScamType.ROMANCE_SCAM: {
            "keywords": [
                r"te\s+amo", r"me\s+ajuda", r"preciso\s+de\s+dinheiro",
                r"love\s+you", r"help\s+me", r"need\s+money", r"send\s+money"
            ],
            "weight": 0.8
        },
        ScamType.FAMILY_EMERGENCY: {
            "keywords": [
                r"seu\s+filho", r"sua\s+filha", r"acidente", r"hospital",
                r"your\s+son", r"your\s+daughter", r"accident", r"hospital"
            ],
            "weight": 0.9
        },
        ScamType.LOTTERY_SCAM: {
            "keywords": [
                r"prêmio", r"sorteio", r"loteria", r"ganhou",
                r"prize", r"lottery", r"winner", r"congratulations"
            ],
            "weight": 0.85
        },
        ScamType.CRYPTO_SCAM: {
            "keywords": [
                r"bitcoin", r"ethereum", r"criptomoeda", r"nft",
                r"crypto", r"blockchain", r"token", r"airdrop"
            ],
            "weight": 0.75
        },
        ScamType.TECH_SUPPORT: {
            "keywords": [
                r"suporte\s+técnico", r"vírus", r"computador\s+infectado",
                r"tech\s+support", r"virus", r"computer\s+infected", r"microsoft"
            ],
            "weight": 0.85
        },
        ScamType.PHISHING: {
            "keywords": [
                r"clique\s+aqui", r"confirme\s+seus\s+dados", r"atualize",
                r"click\s+here", r"verify\s+your", r"update\s+your"
            ],
            "weight": 0.8
        }
    }

    # Técnicas de manipulação
    MANIPULATION_TECHNIQUES = {
        "urgency": {
            "patterns": [
                r"agora", r"imediatamente", r"urgente", r"prazo",
                r"now", r"immediately", r"urgent", r"deadline"
            ],
            "description": "Criação de urgência artificial"
        },
        "authority": {
            "patterns": [
                r"autoridade", r"oficial", r"legal", r"ordem",
                r"authority", r"official", r"legal", r"order"
            ],
            "description": "Apelo à autoridade"
        },
        "fear": {
            "patterns": [
                r"prisão", r"multa", r"bloqueio", r"perder",
                r"arrest", r"fine", r"block", r"lose"
            ],
            "description": "Indução de medo"
        },
        "greed": {
            "patterns": [
                r"ganhar", r"lucro", r"prêmio", r"grátis",
                r"win", r"profit", r"prize", r"free"
            ],
            "description": "Exploração da ganância"
        },
        "trust": {
            "patterns": [
                r"confie", r"seguro", r"garantido", r"oficial",
                r"trust", r"secure", r"guaranteed", r"official"
            ],
            "description": "Construção de falsa confiança"
        },
        "isolation": {
            "patterns": [
                r"não\s+conte", r"segredo", r"entre\s+nós",
                r"don't\s+tell", r"secret", r"between\s+us"
            ],
            "description": "Isolamento da vítima"
        },
        "reciprocity": {
            "patterns": [
                r"te\s+ajudei", r"favor", r"devolva",
                r"helped\s+you", r"favor", r"return"
            ],
            "description": "Exploração de reciprocidade"
        },
        "scarcity": {
            "patterns": [
                r"última\s+chance", r"restam\s+poucos", r"limitado",
                r"last\s+chance", r"only\s+few", r"limited"
            ],
            "description": "Criação de escassez"
        }
    }

    # Mensagens de intervenção
    INTERVENTION_MESSAGES = {
        ScamType.BANK_IMPERSONATION: (
            "ATENÇÃO: Bancos NUNCA pedem senhas, códigos ou transferências "
            "por telefone/mensagem. Em caso de dúvida, ligue para o número "
            "oficial no verso do seu cartão."
        ),
        ScamType.POLICE_IMPERSONATION: (
            "CUIDADO: Policiais NÃO pedem dinheiro para evitar prisão ou "
            "resolver problemas. Isso é golpe! Desligue e procure uma "
            "delegacia se necessário."
        ),
        ScamType.FAMILY_EMERGENCY: (
            "PARE: Antes de transferir, confirme diretamente com seu "
            "familiar pelo número que você já conhece. Golpistas se "
            "passam por parentes em emergências falsas."
        ),
        ScamType.INVESTMENT_SCAM: (
            "ALERTA: Investimentos com 'retorno garantido' não existem. "
            "Se parece bom demais para ser verdade, provavelmente é golpe."
        ),
        ScamType.ROMANCE_SCAM: (
            "ATENÇÃO: Pessoas que pedem dinheiro online, mesmo após "
            "longo relacionamento virtual, frequentemente são golpistas. "
            "Nunca envie dinheiro para quem você não conhece pessoalmente."
        ),
        ScamType.UNKNOWN: (
            "CUIDADO: Esta comunicação apresenta características de golpe. "
            "Não forneça dados pessoais, senhas ou faça transferências."
        )
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Compilar padrões regex
        self._compile_patterns()

        logger.info(f"ScamPatternDetector v{self.VERSION} initialized")

    def _compile_patterns(self):
        """Compilar padrões regex para eficiência"""
        self.compiled_scam_patterns = {}
        for scam_type, data in self.SCAM_PATTERNS.items():
            self.compiled_scam_patterns[scam_type] = {
                "patterns": [re.compile(p, re.IGNORECASE) for p in data["keywords"]],
                "weight": data["weight"]
            }

        self.compiled_manipulation_patterns = {}
        for technique, data in self.MANIPULATION_TECHNIQUES.items():
            self.compiled_manipulation_patterns[technique] = {
                "patterns": [re.compile(p, re.IGNORECASE) for p in data["patterns"]],
                "description": data["description"]
            }

    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ScamDetectionResult:
        """
        Detectar scam em texto

        Args:
            text: Texto para análise (mensagem, conversa, etc.)
            context: Contexto adicional

        Returns:
            ScamDetectionResult com análise completa
        """
        if not text:
            return self._empty_result()

        indicators = []
        manipulation_techniques = []

        # 1. Detectar tipos de scam
        scam_scores = self._detect_scam_types(text)

        # 2. Detectar técnicas de manipulação
        manipulation_techniques = self._detect_manipulation_techniques(text)

        # 3. Calcular scores
        urgency_score = self._calculate_urgency_score(text)
        coercion_score = self._calculate_coercion_score(text, manipulation_techniques)

        # 4. Determinar tipo de scam mais provável
        scam_type, type_confidence = self._determine_scam_type(scam_scores)

        # 5. Criar indicadores
        indicators = self._create_indicators(
            scam_scores, manipulation_techniques, text
        )

        # 6. Calcular risk score final
        risk_score = self._calculate_risk_score(
            type_confidence, urgency_score, coercion_score,
            len(indicators), len(manipulation_techniques)
        )

        # 7. Determinar se é scam
        is_scam = risk_score > 0.6

        # 8. Calcular confiança geral
        confidence = self._calculate_confidence(
            indicators, manipulation_techniques, context
        )

        # 9. Determinar ação recomendada
        recommended_action = self._recommend_action(risk_score, scam_type)

        # 10. Obter mensagem de intervenção
        intervention_message = self._get_intervention_message(scam_type, risk_score)

        # 11. Gerar explicação
        explanation = self._generate_explanation(
            scam_type, indicators, manipulation_techniques, risk_score
        )

        return ScamDetectionResult(
            is_scam=is_scam,
            scam_type=scam_type,
            confidence=confidence,
            risk_score=risk_score,
            indicators=indicators,
            manipulation_techniques=manipulation_techniques,
            urgency_score=urgency_score,
            coercion_score=coercion_score,
            recommended_action=recommended_action,
            intervention_message=intervention_message,
            explanation=explanation
        )

    def _detect_scam_types(self, text: str) -> Dict[ScamType, float]:
        """Detectar tipos de scam e seus scores"""
        scores = {}

        for scam_type, data in self.compiled_scam_patterns.items():
            match_count = 0
            for pattern in data["patterns"]:
                if pattern.search(text):
                    match_count += 1

            if match_count > 0:
                # Score baseado em matches e peso do tipo
                base_score = min(match_count / 3, 1.0)
                scores[scam_type] = base_score * data["weight"]

        return scores

    def _detect_manipulation_techniques(self, text: str) -> List[str]:
        """Detectar técnicas de manipulação usadas"""
        detected = []

        for technique, data in self.compiled_manipulation_patterns.items():
            for pattern in data["patterns"]:
                if pattern.search(text):
                    detected.append(technique)
                    break

        return detected

    def _calculate_urgency_score(self, text: str) -> float:
        """Calcular score de urgência"""
        urgency_patterns = self.compiled_manipulation_patterns.get("urgency", {})
        patterns = urgency_patterns.get("patterns", [])

        matches = sum(1 for p in patterns if p.search(text))
        return min(matches / 3, 1.0)

    def _calculate_coercion_score(
        self,
        text: str,
        techniques: List[str]
    ) -> float:
        """Calcular score de coerção"""
        coercion_techniques = ["fear", "authority", "isolation"]
        coercion_count = sum(1 for t in techniques if t in coercion_techniques)

        base_score = coercion_count / len(coercion_techniques)

        # Ajustar por intensidade
        if "fear" in techniques and "authority" in techniques:
            base_score += 0.2

        return min(base_score, 1.0)

    def _determine_scam_type(
        self,
        scores: Dict[ScamType, float]
    ) -> Tuple[ScamType, float]:
        """Determinar tipo de scam mais provável"""
        if not scores:
            return ScamType.UNKNOWN, 0.0

        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]

    def _create_indicators(
        self,
        scam_scores: Dict[ScamType, float],
        techniques: List[str],
        text: str
    ) -> List[ScamIndicator]:
        """Criar lista de indicadores detectados"""
        indicators = []

        # Indicadores de tipo de scam
        for scam_type, score in scam_scores.items():
            if score > 0.3:
                indicators.append(ScamIndicator(
                    indicator_type="scam_type",
                    description=f"Padrão de {scam_type.value} detectado",
                    severity=score,
                    evidence=f"Score: {score:.2f}",
                    confidence=score
                ))

        # Indicadores de técnicas de manipulação
        for technique in techniques:
            tech_data = self.MANIPULATION_TECHNIQUES.get(technique, {})
            indicators.append(ScamIndicator(
                indicator_type="manipulation",
                description=tech_data.get("description", technique),
                severity=0.7,
                evidence=f"Técnica: {technique}",
                confidence=0.8
            ))

        return indicators

    def _calculate_risk_score(
        self,
        type_confidence: float,
        urgency: float,
        coercion: float,
        indicators_count: int,
        techniques_count: int
    ) -> float:
        """Calcular score de risco final"""
        # Pesos
        weights = {
            "type": 0.35,
            "urgency": 0.20,
            "coercion": 0.25,
            "indicators": 0.10,
            "techniques": 0.10
        }

        score = (
            type_confidence * weights["type"] +
            urgency * weights["urgency"] +
            coercion * weights["coercion"] +
            min(indicators_count / 5, 1.0) * weights["indicators"] +
            min(techniques_count / 4, 1.0) * weights["techniques"]
        )

        return min(score, 1.0)

    def _calculate_confidence(
        self,
        indicators: List[ScamIndicator],
        techniques: List[str],
        context: Optional[Dict]
    ) -> float:
        """Calcular confiança na detecção"""
        # Base na quantidade de evidências
        evidence_score = min(len(indicators) / 3 + len(techniques) / 4, 1.0)

        # Ajustar com contexto
        if context:
            if context.get("known_scammer"):
                evidence_score = min(evidence_score + 0.3, 1.0)
            if context.get("verified_source"):
                evidence_score = max(evidence_score - 0.2, 0.0)

        return evidence_score * 0.8 + 0.2  # Mínimo de 20% de confiança

    def _recommend_action(
        self,
        risk_score: float,
        scam_type: ScamType
    ) -> str:
        """Recomendar ação baseada no risco"""
        if risk_score > 0.85:
            return "BLOCK_AND_ALERT"
        elif risk_score > 0.7:
            return "REQUIRE_CONFIRMATION"
        elif risk_score > 0.5:
            return "SHOW_WARNING"
        elif risk_score > 0.3:
            return "SOFT_WARNING"
        else:
            return "MONITOR"

    def _get_intervention_message(
        self,
        scam_type: ScamType,
        risk_score: float
    ) -> str:
        """Obter mensagem de intervenção apropriada"""
        if risk_score < 0.4:
            return ""

        return self.INTERVENTION_MESSAGES.get(
            scam_type,
            self.INTERVENTION_MESSAGES[ScamType.UNKNOWN]
        )

    def _generate_explanation(
        self,
        scam_type: ScamType,
        indicators: List[ScamIndicator],
        techniques: List[str],
        risk_score: float
    ) -> str:
        """Gerar explicação da detecção"""
        parts = []

        if risk_score > 0.6:
            parts.append(f"Alta probabilidade de golpe tipo '{scam_type.value}'.")
        elif risk_score > 0.3:
            parts.append(f"Possíveis sinais de golpe detectados.")

        if indicators:
            parts.append(f"Indicadores identificados: {len(indicators)}.")

        if techniques:
            parts.append(f"Técnicas de manipulação: {', '.join(techniques)}.")

        return " ".join(parts) if parts else "Análise inconclusiva."

    def _empty_result(self) -> ScamDetectionResult:
        """Resultado vazio"""
        return ScamDetectionResult(
            is_scam=False,
            scam_type=ScamType.UNKNOWN,
            confidence=0.0,
            risk_score=0.0,
            indicators=[],
            manipulation_techniques=[],
            urgency_score=0.0,
            coercion_score=0.0,
            recommended_action="ALLOW",
            intervention_message="",
            explanation="Sem dados para análise."
        )

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas"""
        return {
            "version": self.VERSION,
            "scam_types_count": len(self.SCAM_PATTERNS),
            "manipulation_techniques_count": len(self.MANIPULATION_TECHNIQUES)
        }


def create_scam_detector(
    config: Optional[Dict[str, Any]] = None
) -> ScamPatternDetector:
    """Factory function para criar ScamPatternDetector"""
    return ScamPatternDetector(config)
