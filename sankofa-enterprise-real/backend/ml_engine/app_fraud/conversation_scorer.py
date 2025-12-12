"""
Sankofa Enterprise Pro - Conversation Scorer
Análise de conversas para detectar social engineering

Baseado em:
- NLP Social Engineering research
- FICO conversation analysis
- Academic papers on scam detection in text
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationRisk:
    """Resultado da análise de risco de conversa"""
    conversation_id: str
    risk_score: float
    risk_level: str
    scam_type: str
    manipulation_tactics: List[str]
    suspicious_phrases: List[str]
    urgency_level: float
    emotional_manipulation: float
    recommended_action: str
    explanation: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationScorer:
    """
    Analisador de conversas para detectar social engineering

    Analisa mensagens em busca de:
    - Táticas de manipulação
    - Urgência artificial
    - Manipulação emocional
    - Padrões de golpes conhecidos
    - Solicitações suspeitas
    """

    VERSION = "1.0.0"

    # Padrões de urgência
    URGENCY_PATTERNS = [
        r"\b(urgent[e]?|imediato|agora|já|rápido|pressa|emergência|atenção)\b",
        r"\b(último[a]? chance|última oportunidade|tempo limitado|expira hoje)\b",
        r"\b(bloqueado|suspens[oa]|cancel|encerr)\b",
        r"\b(2[4]?\s*h(ora)?s?|48\s*h|prazo)\b",
        r"\b(aviso importante|comunicado urgente|ação necessária)\b",
        r"\b(senão|caso contrário|do contrário)\b",
        r"\b(somente hoje|só hoje|válido até)\b"
    ]

    # Padrões de manipulação emocional
    EMOTIONAL_PATTERNS = [
        r"\b(parabéns|ganhou|premiado|sortead[oa]|contemplad[oa])\b",
        r"\b(grátis|gratuito|sem custo|bonificação|bônus)\b",
        r"\b(medo|perigo|risco|ameaça|problema grave)\b",
        r"\b(família|filho|mãe|pai|familiar|parente)\b",
        r"\b(ajud[ae]|socorro|precisando|necessidade)\b",
        r"\b(confi[ae]|acredit[ae]|verdade|juro)\b",
        r"\b(oportunidade única|chance única|imperdível)\b"
    ]

    # Padrões de autoridade/impersonação
    AUTHORITY_PATTERNS = [
        r"\b(banco|itaú|bradesco|santander|caixa|bb|nubank|inter|c6)\b",
        r"\b(central de atendimento|suporte|sac|ouvidoria)\b",
        r"\b(gerente|diretor|supervisor|atendente)\b",
        r"\b(policia|delegacia|ministerio|governo|receita)\b",
        r"\b(oficial|autorizado|certificado|verificado)\b",
        r"\b(funcionário|colaborador|representante)\b"
    ]

    # Padrões de solicitação de dados
    DATA_REQUEST_PATTERNS = [
        r"\b(senha|token|código|pin|cvv)\b",
        r"\b(cpf|rg|documento|identidade)\b",
        r"\b(confirme?|envie?|informe?|digite?)\s+(seus?|o|a)\s+dados",
        r"\b(dados cadastrais|informações pessoais)\b",
        r"\b(cartão|conta|agência|número)\b",
        r"\b(chave\s+pix|qr\s*code)\b"
    ]

    # Padrões de golpes específicos PIX
    PIX_SCAM_PATTERNS = [
        r"\b(pix\s+errado|pix\s+por\s+engano|devolv[ae])\b",
        r"\b(bug\s+do\s+pix|pix\s+em\s+dobro|retorno\s+automático)\b",
        r"\b(teste\s+de\s+pix|envie\s+para\s+testar)\b",
        r"\b(promoção\s+pix|ganhe\s+pix|pix\s+grátis)\b",
        r"\b(pix\s+premiado|sorteio\s+pix)\b"
    ]

    # Padrões de golpe WhatsApp
    WHATSAPP_PATTERNS = [
        r"\b(troquei\s+de\s+número|número\s+novo|meu\s+novo\s+número)\b",
        r"\b(me\s+empresta|pode\s+me\s+ajudar|preciso\s+de\s+dinheiro)\b",
        r"\b(paga\s+pra\s+mim|faz\s+um\s+pix|transfere\s+pra\s+mim)\b",
        r"\b(depois\s+te\s+devolvo|amanhã\s+te\s+pago)\b",
        r"\b(salva\s+esse\s+número|esse\s+é\s+meu\s+novo)\b"
    ]

    # Padrões de links suspeitos
    SUSPICIOUS_LINK_PATTERNS = [
        r"https?://[^\s]+\.(tk|ml|ga|cf|gq|xyz|top|click|link)",
        r"https?://[^\s]*bit\.ly",
        r"https?://[^\s]*tinyurl",
        r"https?://[^\s]+/[a-zA-Z0-9]{4,8}$",
        r"\b(clique aqui|acesse agora|link abaixo)\b"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)

        # Compilar padrões regex
        self._compile_patterns()

        logger.info(f"ConversationScorer v{self.VERSION} initialized")

    def _compile_patterns(self):
        """Compila padrões regex para eficiência"""
        self.compiled_urgency = [
            re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS
        ]
        self.compiled_emotional = [
            re.compile(p, re.IGNORECASE) for p in self.EMOTIONAL_PATTERNS
        ]
        self.compiled_authority = [
            re.compile(p, re.IGNORECASE) for p in self.AUTHORITY_PATTERNS
        ]
        self.compiled_data_request = [
            re.compile(p, re.IGNORECASE) for p in self.DATA_REQUEST_PATTERNS
        ]
        self.compiled_pix_scam = [
            re.compile(p, re.IGNORECASE) for p in self.PIX_SCAM_PATTERNS
        ]
        self.compiled_whatsapp = [
            re.compile(p, re.IGNORECASE) for p in self.WHATSAPP_PATTERNS
        ]
        self.compiled_links = [
            re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_LINK_PATTERNS
        ]

    def analyze_conversation(
        self,
        messages: List[str],
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationRisk:
        """
        Analisa uma conversa para detectar social engineering

        Args:
            messages: Lista de mensagens da conversa
            conversation_id: ID da conversa
            context: Contexto adicional (fonte, remetente, etc.)

        Returns:
            ConversationRisk com resultado da análise
        """
        conversation_id = conversation_id or f"CONV_{hash(str(messages)) % 10000:04d}"

        # Combinar mensagens para análise
        full_text = " ".join(messages)
        normalized_text = self._normalize_text(full_text)

        # 1. Analisar urgência
        urgency_score, urgency_matches = self._analyze_urgency(normalized_text)

        # 2. Analisar manipulação emocional
        emotional_score, emotional_matches = self._analyze_emotional(normalized_text)

        # 3. Analisar uso de autoridade
        authority_score, authority_matches = self._analyze_authority(normalized_text)

        # 4. Analisar solicitação de dados
        data_score, data_matches = self._analyze_data_requests(normalized_text)

        # 5. Analisar padrões de golpes específicos
        scam_score, scam_type, scam_matches = self._analyze_scam_patterns(normalized_text)

        # 6. Analisar links suspeitos
        link_score, link_matches = self._analyze_links(normalized_text)

        # Coletar todas as frases suspeitas
        suspicious_phrases = list(set(
            urgency_matches + emotional_matches + authority_matches +
            data_matches + scam_matches + link_matches
        ))

        # Identificar táticas de manipulação
        manipulation_tactics = []
        if urgency_score > 0.3:
            manipulation_tactics.append("urgency_pressure")
        if emotional_score > 0.3:
            manipulation_tactics.append("emotional_manipulation")
        if authority_score > 0.3:
            manipulation_tactics.append("authority_impersonation")
        if data_score > 0.3:
            manipulation_tactics.append("data_harvesting")
        if link_score > 0.3:
            manipulation_tactics.append("suspicious_links")

        # Calcular score de risco final
        risk_score = self._calculate_risk_score(
            urgency_score, emotional_score, authority_score,
            data_score, scam_score, link_score
        )

        # Determinar nível de risco
        risk_level = self._get_risk_level(risk_score)

        # Determinar ação recomendada
        recommended_action = self._get_recommended_action(risk_score, scam_type)

        # Gerar explicação
        explanation = self._generate_explanation(
            risk_score, scam_type, manipulation_tactics, suspicious_phrases
        )

        # Calcular confiança
        confidence = self._calculate_confidence(
            len(messages), len(suspicious_phrases), len(manipulation_tactics)
        )

        return ConversationRisk(
            conversation_id=conversation_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            scam_type=scam_type,
            manipulation_tactics=manipulation_tactics,
            suspicious_phrases=suspicious_phrases[:10],  # Limitar para não sobrecarregar
            urgency_level=urgency_score,
            emotional_manipulation=emotional_score,
            recommended_action=recommended_action,
            explanation=explanation,
            confidence=confidence
        )

    def analyze_single_message(
        self,
        message: str,
        message_id: Optional[str] = None
    ) -> ConversationRisk:
        """Analisa uma única mensagem"""
        return self.analyze_conversation([message], message_id)

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para análise"""
        text = text.lower()
        text = re.sub(r"[^\w\s\-@:/.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _analyze_urgency(self, text: str) -> Tuple[float, List[str]]:
        """Analisa nível de urgência"""
        matches = []
        for pattern in self.compiled_urgency:
            found = pattern.findall(text)
            matches.extend(found)

        score = min(len(matches) / 3.0, 1.0)
        return score, list(set(matches))

    def _analyze_emotional(self, text: str) -> Tuple[float, List[str]]:
        """Analisa manipulação emocional"""
        matches = []
        for pattern in self.compiled_emotional:
            found = pattern.findall(text)
            matches.extend(found)

        score = min(len(matches) / 3.0, 1.0)
        return score, list(set(matches))

    def _analyze_authority(self, text: str) -> Tuple[float, List[str]]:
        """Analisa uso de autoridade/impersonação"""
        matches = []
        for pattern in self.compiled_authority:
            found = pattern.findall(text)
            matches.extend(found)

        score = min(len(matches) / 4.0, 1.0)
        return score, list(set(matches))

    def _analyze_data_requests(self, text: str) -> Tuple[float, List[str]]:
        """Analisa solicitações de dados sensíveis"""
        matches = []
        for pattern in self.compiled_data_request:
            found = pattern.findall(text)
            matches.extend(found if isinstance(found[0], str) else [f[0] for f in found] if found else [])

        score = min(len(matches) / 2.0, 1.0)
        return score, list(set(matches))

    def _analyze_scam_patterns(self, text: str) -> Tuple[float, str, List[str]]:
        """Analisa padrões de golpes específicos"""
        pix_matches = []
        for pattern in self.compiled_pix_scam:
            found = pattern.findall(text)
            pix_matches.extend(found)

        whatsapp_matches = []
        for pattern in self.compiled_whatsapp:
            found = pattern.findall(text)
            whatsapp_matches.extend(found)

        # Determinar tipo de golpe mais provável
        if len(pix_matches) > len(whatsapp_matches):
            return min(len(pix_matches) / 2.0, 1.0), "PIX_SCAM", list(set(pix_matches))
        elif len(whatsapp_matches) > 0:
            return min(len(whatsapp_matches) / 2.0, 1.0), "WHATSAPP_CLONE", list(set(whatsapp_matches))

        return 0.0, "UNKNOWN", []

    def _analyze_links(self, text: str) -> Tuple[float, List[str]]:
        """Analisa links suspeitos"""
        matches = []
        for pattern in self.compiled_links:
            found = pattern.findall(text)
            matches.extend(found)

        score = min(len(matches) / 2.0, 1.0)
        return score, list(set(matches))

    def _calculate_risk_score(
        self,
        urgency: float,
        emotional: float,
        authority: float,
        data: float,
        scam: float,
        links: float
    ) -> float:
        """Calcula score de risco final"""
        # Pesos dos componentes
        weights = {
            "urgency": 0.15,
            "emotional": 0.15,
            "authority": 0.20,
            "data": 0.25,
            "scam": 0.15,
            "links": 0.10
        }

        weighted_score = (
            urgency * weights["urgency"] +
            emotional * weights["emotional"] +
            authority * weights["authority"] +
            data * weights["data"] +
            scam * weights["scam"] +
            links * weights["links"]
        )

        # Boost se múltiplos indicadores altos
        high_indicators = sum(1 for s in [urgency, emotional, authority, data, scam, links] if s > 0.5)
        if high_indicators >= 3:
            weighted_score *= 1.3

        # Boost especial para combinação perigosa
        if authority > 0.5 and data > 0.5:
            weighted_score *= 1.2

        return min(weighted_score, 1.0)

    def _get_risk_level(self, score: float) -> str:
        """Determina nível de risco"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        return "VERY_LOW"

    def _get_recommended_action(self, score: float, scam_type: str) -> str:
        """Determina ação recomendada"""
        if score >= 0.8:
            return "BLOCK_MESSAGE"
        elif score >= 0.6:
            return "WARN_USER_STRONGLY"
        elif score >= 0.4:
            return "WARN_USER"
        elif score >= 0.2:
            return "FLAG_FOR_REVIEW"
        return "ALLOW"

    def _generate_explanation(
        self,
        score: float,
        scam_type: str,
        tactics: List[str],
        phrases: List[str]
    ) -> str:
        """Gera explicação da análise"""
        if score < 0.2:
            return "Mensagem não apresenta sinais significativos de golpe."

        explanation_parts = []

        if score >= 0.6:
            explanation_parts.append(
                f"ALERTA: Esta mensagem apresenta ALTO risco de ser um golpe "
                f"(score: {score*100:.0f}%)."
            )
        else:
            explanation_parts.append(
                f"Esta mensagem apresenta alguns sinais de risco "
                f"(score: {score*100:.0f}%)."
            )

        # Tipo de golpe
        scam_descriptions = {
            "PIX_SCAM": "Possível golpe relacionado ao PIX",
            "WHATSAPP_CLONE": "Possível golpe de clonagem de WhatsApp",
            "INVESTMENT": "Possível golpe de investimento",
            "PRIZE": "Possível golpe de falso prêmio"
        }
        if scam_type in scam_descriptions:
            explanation_parts.append(scam_descriptions[scam_type] + ".")

        # Táticas identificadas
        tactic_descriptions = {
            "urgency_pressure": "Pressão por urgência",
            "emotional_manipulation": "Manipulação emocional",
            "authority_impersonation": "Falsa autoridade/impersonação",
            "data_harvesting": "Solicitação de dados sensíveis",
            "suspicious_links": "Links suspeitos"
        }

        identified_tactics = [
            tactic_descriptions.get(t, t)
            for t in tactics[:3]
        ]
        if identified_tactics:
            explanation_parts.append(
                "Táticas identificadas: " + "; ".join(identified_tactics) + "."
            )

        return " ".join(explanation_parts)

    def _calculate_confidence(
        self,
        num_messages: int,
        num_suspicious: int,
        num_tactics: int
    ) -> float:
        """Calcula confiança na análise"""
        confidence = 0.5

        # Mais mensagens = mais contexto
        confidence += min(num_messages * 0.05, 0.15)

        # Mais evidências = mais confiança
        confidence += min(num_suspicious * 0.02, 0.15)
        confidence += min(num_tactics * 0.05, 0.15)

        return min(confidence, 0.95)


def create_conversation_scorer(config: Optional[Dict[str, Any]] = None) -> ConversationScorer:
    """Factory function para criar ConversationScorer"""
    return ConversationScorer(config)
