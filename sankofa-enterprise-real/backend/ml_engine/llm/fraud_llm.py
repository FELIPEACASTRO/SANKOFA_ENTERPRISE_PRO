"""
Sankofa Enterprise Pro - Fraud LLM Analyzer
Análise de fraude usando Large Language Models

Baseado em:
- Mistral-7B-LLM-Fraud-Detection (Hugging Face)
- FinBERT for financial analysis
- Prompt engineering for fraud detection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

# Tentar importar transformers (opcional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. LLM features will be limited.")


class RiskLevel(Enum):
    """Níveis de risco"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConversationRisk:
    """Resultado da análise de risco de conversa"""
    risk_level: RiskLevel
    risk_score: float  # 0-1
    scam_probability: float
    urgency_detected: bool
    authority_impersonation: bool
    emotional_manipulation: bool
    sensitive_info_request: bool
    suspicious_links: List[str]
    key_phrases: List[str]
    explanation: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScamAnalysis:
    """Análise detalhada de scam"""
    is_scam: bool
    scam_type: str
    confidence: float
    indicators: List[str]
    victim_profile: str
    estimated_loss: Optional[float]
    urgency_level: str
    manipulation_techniques: List[str]
    recommended_action: str
    evidence: List[Dict[str, Any]]


@dataclass
class TransactionNarrative:
    """Narrativa gerada para transação"""
    summary: str
    risk_factors: List[str]
    context: str
    recommendation: str


class FraudLLMAnalyzer:
    """
    Analisador de Fraude baseado em LLM

    Casos de uso:
    1. Análise de conversas suspeitas (chat/call transcripts)
    2. Detecção de social engineering patterns
    3. Explicações em linguagem natural para analistas
    4. Geração de relatórios automatizados
    5. Análise de narrativas em disputas
    """

    VERSION = "1.0.0"

    # Padrões de scam conhecidos
    SCAM_PATTERNS = {
        "urgency": [
            r"urgent", r"immediately", r"right now", r"don't wait",
            r"expires today", r"limited time", r"act fast", r"hurry",
            r"urgente", r"agora", r"imediatamente", r"prazo"
        ],
        "authority": [
            r"bank manager", r"police", r"government", r"tax authority",
            r"court order", r"legal action", r"arrest warrant",
            r"gerente", r"polícia", r"governo", r"receita federal"
        ],
        "emotional": [
            r"help me", r"emergency", r"accident", r"hospital",
            r"your account", r"suspended", r"locked", r"compromised",
            r"ajuda", r"emergência", r"acidente", r"sua conta"
        ],
        "financial": [
            r"prize", r"lottery", r"inheritance", r"investment",
            r"guaranteed return", r"double your money", r"bitcoin",
            r"prêmio", r"loteria", r"herança", r"investimento"
        ],
        "info_request": [
            r"password", r"pin", r"otp", r"verification code",
            r"credit card", r"bank account", r"social security",
            r"senha", r"código", r"cartão", r"conta bancária"
        ]
    }

    # Frases suspeitas
    SUSPICIOUS_PHRASES = [
        "don't tell anyone",
        "keep this secret",
        "transfer now",
        "send money",
        "gift card",
        "wire transfer",
        "não conte para ninguém",
        "mantenha em segredo",
        "transfira agora",
        "envie dinheiro",
        "cartão presente",
        "pix agora"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.sentiment_analyzer = None

        # Carregar modelo se disponível
        if HAS_TRANSFORMERS and self.config.get("load_model", False):
            self._load_model()

        logger.info(f"FraudLLMAnalyzer v{self.VERSION} initialized")
        logger.info(f"Transformers available: {HAS_TRANSFORMERS}")

    def _load_model(self):
        """Carregar modelo LLM"""
        try:
            model_name = self.config.get(
                "model_name",
                "Bilic/Mistral-7B-LLM-Fraud-Detection"
            )

            logger.info(f"Loading model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True  # Quantização para economia de memória
            )

            # Sentiment analyzer para análise rápida
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def analyze_conversation(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationRisk:
        """
        Analisar conversa para detectar social engineering

        Args:
            messages: Lista de mensagens [{"role": "user/agent", "content": "..."}]
            context: Contexto adicional (usuário, transação, etc.)

        Returns:
            ConversationRisk com análise completa
        """
        if not messages:
            return self._empty_risk()

        # Concatenar mensagens
        full_text = self._format_conversation(messages)

        # 1. Detectar padrões de urgência
        urgency_detected = self._detect_patterns(full_text, "urgency")

        # 2. Detectar impersonação de autoridade
        authority_impersonation = self._detect_patterns(full_text, "authority")

        # 3. Detectar manipulação emocional
        emotional_manipulation = self._detect_patterns(full_text, "emotional")

        # 4. Detectar solicitação de informações sensíveis
        sensitive_info_request = self._detect_patterns(full_text, "info_request")

        # 5. Detectar links suspeitos
        suspicious_links = self._extract_suspicious_links(full_text)

        # 6. Extrair frases-chave suspeitas
        key_phrases = self._extract_key_phrases(full_text)

        # 7. Calcular score de risco
        risk_score = self._calculate_risk_score(
            urgency_detected,
            authority_impersonation,
            emotional_manipulation,
            sensitive_info_request,
            len(suspicious_links),
            len(key_phrases)
        )

        # 8. Calcular probabilidade de scam
        scam_probability = self._calculate_scam_probability(
            full_text, risk_score, context
        )

        # 9. Determinar nível de risco
        risk_level = self._determine_risk_level(risk_score, scam_probability)

        # 10. Gerar explicação
        explanation = self._generate_explanation(
            risk_level, urgency_detected, authority_impersonation,
            emotional_manipulation, sensitive_info_request
        )

        # 11. Gerar recomendações
        recommendations = self._generate_recommendations(
            risk_level, scam_probability
        )

        return ConversationRisk(
            risk_level=risk_level,
            risk_score=risk_score,
            scam_probability=scam_probability,
            urgency_detected=urgency_detected,
            authority_impersonation=authority_impersonation,
            emotional_manipulation=emotional_manipulation,
            sensitive_info_request=sensitive_info_request,
            suspicious_links=suspicious_links,
            key_phrases=key_phrases,
            explanation=explanation,
            recommendations=recommendations
        )

    def analyze_scam(
        self,
        text: str,
        transaction: Optional[Dict[str, Any]] = None
    ) -> ScamAnalysis:
        """
        Análise detalhada de possível scam

        Args:
            text: Texto da comunicação
            transaction: Dados da transação (opcional)

        Returns:
            ScamAnalysis com análise detalhada
        """
        # Detectar indicadores
        indicators = []
        manipulation_techniques = []

        # Verificar padrões
        for pattern_type, patterns in self.SCAM_PATTERNS.items():
            if self._detect_patterns(text, pattern_type):
                indicators.append(f"{pattern_type}_detected")
                manipulation_techniques.append(pattern_type)

        # Verificar frases suspeitas
        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase.lower() in text.lower():
                indicators.append(f"suspicious_phrase: {phrase}")

        # Calcular confiança
        confidence = min(len(indicators) / 5, 1.0)

        # Determinar tipo de scam
        scam_type = self._determine_scam_type(indicators, text)

        # Determinar se é scam
        is_scam = confidence > 0.6

        # Determinar urgência
        urgency_level = "high" if "urgency_detected" in indicators else "low"

        # Perfil da vítima
        victim_profile = self._analyze_victim_profile(text, transaction)

        # Ação recomendada
        recommended_action = self._recommend_action(confidence, scam_type)

        return ScamAnalysis(
            is_scam=is_scam,
            scam_type=scam_type,
            confidence=confidence,
            indicators=indicators,
            victim_profile=victim_profile,
            estimated_loss=transaction.get("amount") if transaction else None,
            urgency_level=urgency_level,
            manipulation_techniques=manipulation_techniques,
            recommended_action=recommended_action,
            evidence=[{"text_snippet": text[:200], "indicators": indicators}]
        )

    def generate_transaction_narrative(
        self,
        transaction: Dict[str, Any],
        risk_factors: List[str]
    ) -> TransactionNarrative:
        """
        Gerar narrativa explicativa para transação

        Args:
            transaction: Dados da transação
            risk_factors: Fatores de risco identificados

        Returns:
            TransactionNarrative com explicação
        """
        # Gerar summary
        summary = self._generate_transaction_summary(transaction)

        # Contexto
        context = self._generate_context(transaction)

        # Recomendação
        recommendation = self._generate_transaction_recommendation(
            transaction, risk_factors
        )

        return TransactionNarrative(
            summary=summary,
            risk_factors=risk_factors,
            context=context,
            recommendation=recommendation
        )

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Formatar conversa para análise"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _detect_patterns(self, text: str, pattern_type: str) -> bool:
        """Detectar padrões de um tipo específico"""
        patterns = self.SCAM_PATTERNS.get(pattern_type, [])
        text_lower = text.lower()

        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_suspicious_links(self, text: str) -> List[str]:
        """Extrair links suspeitos"""
        # Padrão de URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)

        suspicious = []
        for url in urls:
            url_lower = url.lower()
            # Verificar indicadores de phishing
            if any(indicator in url_lower for indicator in [
                "bit.ly", "tinyurl", "goo.gl",  # Encurtadores
                "verify", "secure", "update", "confirm",  # Palavras suspeitas
                "login", "signin", "account", "password"
            ]):
                suspicious.append(url)

        return suspicious

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extrair frases-chave suspeitas"""
        found = []
        text_lower = text.lower()

        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase.lower() in text_lower:
                found.append(phrase)

        return found

    def _calculate_risk_score(
        self,
        urgency: bool,
        authority: bool,
        emotional: bool,
        info_request: bool,
        links_count: int,
        phrases_count: int
    ) -> float:
        """Calcular score de risco agregado"""
        score = 0.0

        if urgency:
            score += 0.2
        if authority:
            score += 0.25
        if emotional:
            score += 0.15
        if info_request:
            score += 0.25

        score += min(links_count * 0.1, 0.3)
        score += min(phrases_count * 0.05, 0.15)

        return min(score, 1.0)

    def _calculate_scam_probability(
        self,
        text: str,
        risk_score: float,
        context: Optional[Dict]
    ) -> float:
        """Calcular probabilidade de ser scam"""
        # Base no risk score
        probability = risk_score * 0.7

        # Ajustar com contexto
        if context:
            # Transação de alto valor
            amount = context.get("transaction", {}).get("amount", 0)
            if amount > 10000:
                probability += 0.1

            # Primeiro contato com destinatário
            if context.get("first_contact", False):
                probability += 0.1

        # Se tiver modelo carregado, usar para análise adicional
        if self.model and self.tokenizer:
            model_prob = self._get_model_prediction(text)
            probability = (probability + model_prob) / 2

        return min(probability, 1.0)

    def _get_model_prediction(self, text: str) -> float:
        """Obter predição do modelo LLM"""
        try:
            prompt = f"""Analyze the following text for fraud indicators:

Text: {text[:500]}

Is this text likely to be part of a scam or fraud attempt?
Answer with a probability score from 0 to 1:"""

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extrair probabilidade da resposta
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
            if numbers:
                return float(numbers[0])

        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")

        return 0.5  # Default

    def _determine_risk_level(
        self,
        risk_score: float,
        scam_probability: float
    ) -> RiskLevel:
        """Determinar nível de risco"""
        combined = (risk_score + scam_probability) / 2

        if combined >= 0.8:
            return RiskLevel.CRITICAL
        elif combined >= 0.6:
            return RiskLevel.HIGH
        elif combined >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_explanation(
        self,
        risk_level: RiskLevel,
        urgency: bool,
        authority: bool,
        emotional: bool,
        info_request: bool
    ) -> str:
        """Gerar explicação da análise"""
        parts = [f"Nível de risco: {risk_level.value.upper()}."]

        if urgency:
            parts.append("Táticas de urgência detectadas.")
        if authority:
            parts.append("Possível impersonação de autoridade.")
        if emotional:
            parts.append("Manipulação emocional identificada.")
        if info_request:
            parts.append("Solicitação de informações sensíveis detectada.")

        return " ".join(parts)

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        scam_probability: float
    ) -> List[str]:
        """Gerar recomendações"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "BLOQUEAR transação imediatamente",
                "Contatar cliente por canal seguro",
                "Reportar para equipe de fraude"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Adicionar delay na transação",
                "Solicitar confirmação adicional",
                "Exibir alerta de segurança ao usuário"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Monitorar transação",
                "Registrar para análise posterior"
            ])
        else:
            recommendations.append("Continuar monitoramento padrão")

        return recommendations

    def _determine_scam_type(
        self,
        indicators: List[str],
        text: str
    ) -> str:
        """Determinar tipo de scam"""
        text_lower = text.lower()

        if "authority" in str(indicators):
            if "polícia" in text_lower or "police" in text_lower:
                return "police_impersonation"
            if "banco" in text_lower or "bank" in text_lower:
                return "bank_impersonation"
            return "authority_impersonation"

        if "financial" in str(indicators):
            if "investimento" in text_lower or "investment" in text_lower:
                return "investment_scam"
            if "prêmio" in text_lower or "prize" in text_lower:
                return "prize_scam"
            return "financial_scam"

        if "emotional" in str(indicators):
            if "família" in text_lower or "family" in text_lower:
                return "family_emergency"
            return "emotional_manipulation"

        return "unknown"

    def _analyze_victim_profile(
        self,
        text: str,
        transaction: Optional[Dict]
    ) -> str:
        """Analisar perfil provável da vítima"""
        # Análise simplificada
        if transaction:
            amount = transaction.get("amount", 0)
            if amount > 50000:
                return "high_value_target"
            if transaction.get("customer_age", 0) > 60:
                return "elderly_target"

        return "general_population"

    def _recommend_action(self, confidence: float, scam_type: str) -> str:
        """Recomendar ação baseada na análise"""
        if confidence > 0.8:
            return "BLOCK_TRANSACTION"
        elif confidence > 0.6:
            return "REQUIRE_VERIFICATION"
        elif confidence > 0.4:
            return "DISPLAY_WARNING"
        else:
            return "MONITOR"

    def _generate_transaction_summary(self, transaction: Dict) -> str:
        """Gerar resumo da transação"""
        amount = transaction.get("amount", 0)
        channel = transaction.get("channel", "unknown")
        tx_type = transaction.get("type", "payment")

        return (
            f"Transação de {tx_type} no valor de R$ {amount:,.2f} "
            f"via canal {channel}."
        )

    def _generate_context(self, transaction: Dict) -> str:
        """Gerar contexto da transação"""
        parts = []

        if transaction.get("is_first_to_receiver"):
            parts.append("Primeira transação para este destinatário.")

        if transaction.get("is_high_value"):
            parts.append("Valor acima da média do cliente.")

        if transaction.get("unusual_hour"):
            parts.append("Horário incomum para o cliente.")

        return " ".join(parts) if parts else "Transação dentro do padrão."

    def _generate_transaction_recommendation(
        self,
        transaction: Dict,
        risk_factors: List[str]
    ) -> str:
        """Gerar recomendação para transação"""
        if len(risk_factors) >= 3:
            return "Recomenda-se revisão manual antes de aprovar."
        elif len(risk_factors) >= 1:
            return "Monitorar para atividade subsequente."
        else:
            return "Aprovar transação."

    def _empty_risk(self) -> ConversationRisk:
        """Retornar risco vazio"""
        return ConversationRisk(
            risk_level=RiskLevel.LOW,
            risk_score=0.0,
            scam_probability=0.0,
            urgency_detected=False,
            authority_impersonation=False,
            emotional_manipulation=False,
            sensitive_info_request=False,
            suspicious_links=[],
            key_phrases=[],
            explanation="Sem dados para análise.",
            recommendations=[]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas"""
        return {
            "version": self.VERSION,
            "model_loaded": self.model is not None,
            "patterns_count": sum(len(p) for p in self.SCAM_PATTERNS.values()),
            "suspicious_phrases_count": len(self.SUSPICIOUS_PHRASES)
        }


def create_fraud_llm(
    config: Optional[Dict[str, Any]] = None
) -> FraudLLMAnalyzer:
    """Factory function para criar FraudLLMAnalyzer"""
    return FraudLLMAnalyzer(config)


# Aliases para compatibilidade
FraudLLM = FraudLLMAnalyzer
LLMFraudAnalyzer = FraudLLMAnalyzer
