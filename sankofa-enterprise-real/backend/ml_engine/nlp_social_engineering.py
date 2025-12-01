"""
Sankofa Enterprise Pro - NLP Social Engineering Detection
Detecção de engenharia social usando processamento de linguagem natural

Baseado em: DIFrauD Dataset (HuggingFace) - 95K amostras multi-domínio
Foco: SMS phishing, mensagens fraudulentas, golpes via texto

Contexto: ~70% das fraudes PIX no Brasil envolvem engenharia social
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SocialEngineeringAnalysis:
    """Resultado da análise de engenharia social"""
    text_id: str
    fraud_probability: float
    fraud_type: str
    indicators: List[str]
    matched_patterns: List[str]
    urgency_score: float
    emotional_score: float
    suspicious_elements: List[str]
    recommendation: str
    confidence: float


class NLPSocialEngineeringDetector:
    """
    Detector de engenharia social baseado em NLP
    
    Detecta padrões de:
    - SMS Phishing (Smishing)
    - Golpes de WhatsApp
    - Emails fraudulentos
    - Mensagens de urgência
    - Impersonação
    """
    
    VERSION = "1.0.0"
    
    URGENCY_PATTERNS = [
        r'\b(urgent[e]?|imediato|agora|já|rápido|pressa|emergência|atenção)\b',
        r'\b(último[a]? chance|última oportunidade|tempo limitado|expira hoje)\b',
        r'\b(bloqueado|suspens[oa]|cancel|encerr)\b',
        r'\b(2[4]?\s*h(ora)?s?|48\s*h|prazo)\b',
        r'\b(aviso importante|comunicado urgente|ação necessária)\b',
    ]
    
    EMOTIONAL_PATTERNS = [
        r'\b(parabéns|ganhou|premiado|sortead[oa]|contemplad[oa])\b',
        r'\b(grátis|gratuito|sem custo|bonificação|bônus)\b',
        r'\b(medo|perigo|risco|ameaça|problema grave)\b',
        r'\b(família|filho|mãe|pai|familiar|parente)\b',
        r'\b(ajud[ae]|socorro|precisando|necessidade)\b',
        r'\b(amor|querido|amigo|conhecido)\b',
    ]
    
    BANK_IMPERSONATION_PATTERNS = [
        r'\b(banco|itaú|bradesco|santander|caixa|bb|nubank|inter|c6)\b',
        r'\b(central de atendimento|suporte|sac|ouvidoria)\b',
        r'\b(token|senha|código|verificação|autenticação)\b',
        r'\b(cartão|conta|pix|ted|doc|transferência)\b',
        r'\b(atualiz[ae]|confirm[ae]|valid[ae]|verifiqu[e])\b',
        r'\b(dados cadastrais|informações pessoais|cpf|rg)\b',
    ]
    
    PHISHING_PATTERNS = [
        r'https?://[^\s]+\.(tk|ml|ga|cf|gq|xyz|top|click|link)',
        r'https?://[^\s]*bit\.ly',
        r'https?://[^\s]*tinyurl',
        r'https?://[^\s]+/[a-zA-Z0-9]{4,8}$',
        r'\b(clique aqui|acesse agora|link abaixo)\b',
        r'\b(copie e cole|acesse pelo link)\b',
    ]
    
    PIX_FRAUD_PATTERNS = [
        r'\b(pix\s+errado|pix\s+por\s+engano|devolv[ae])\b',
        r'\b(bug\s+do\s+pix|pix\s+em\s+dobro|retorno\s+automático)\b',
        r'\b(chave\s+pix|qr\s*code|código\s+pix)\b',
        r'\b(teste\s+de\s+pix|envie\s+para\s+testar)\b',
        r'\b(promoção\s+pix|ganhe\s+pix|pix\s+grátis)\b',
    ]
    
    WHATSAPP_CLONE_PATTERNS = [
        r'\b(troquei\s+de\s+número|número\s+novo|meu\s+novo\s+número)\b',
        r'\b(me\s+empresta|pode\s+me\s+ajudar|preciso\s+de\s+dinheiro)\b',
        r'\b(paga\s+pra\s+mim|faz\s+um\s+pix|transfere\s+pra\s+mim)\b',
        r'\b(depois\s+te\s+devolvo|amanhã\s+te\s+pago|te\s+dou\s+o\s+dinheiro)\b',
        r'\b(é\s+o\s*[\w]+|aqui\s+é|sou\s+eu|oi\s+[\w]+)\b',
    ]
    
    BLACKLIST_KEYWORDS = {
        'high_risk': [
            'senha', 'token', 'código', 'verificação', 
            'bloqueado', 'suspenso', 'urgente', 'imediato',
            'ganhou', 'premiado', 'sorteado', 'contemplado',
        ],
        'medium_risk': [
            'atualizar', 'confirmar', 'validar', 'cadastro',
            'promoção', 'desconto', 'oferta', 'grátis',
            'clique', 'acesse', 'link',
        ],
        'low_risk': [
            'banco', 'conta', 'cartão', 'pix',
            'transferência', 'pagamento',
        ],
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        
        self._compile_patterns()
        
        logger.info(f"NLP Social Engineering Detector v{self.VERSION} initialized")
    
    def _compile_patterns(self):
        """Compila padrões regex para eficiência"""
        self.compiled_urgency = [
            re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS
        ]
        self.compiled_emotional = [
            re.compile(p, re.IGNORECASE) for p in self.EMOTIONAL_PATTERNS
        ]
        self.compiled_bank = [
            re.compile(p, re.IGNORECASE) for p in self.BANK_IMPERSONATION_PATTERNS
        ]
        self.compiled_phishing = [
            re.compile(p, re.IGNORECASE) for p in self.PHISHING_PATTERNS
        ]
        self.compiled_pix = [
            re.compile(p, re.IGNORECASE) for p in self.PIX_FRAUD_PATTERNS
        ]
        self.compiled_whatsapp = [
            re.compile(p, re.IGNORECASE) for p in self.WHATSAPP_CLONE_PATTERNS
        ]
    
    def analyze_text(
        self,
        text: str,
        text_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> SocialEngineeringAnalysis:
        """
        Analisa texto para detectar engenharia social
        
        Args:
            text: Texto a analisar (SMS, email, mensagem)
            text_id: ID do texto
            source: Fonte (sms, whatsapp, email)
            metadata: Metadados adicionais
            
        Returns:
            Análise de engenharia social
        """
        text_id = text_id or f"TXT_{hash(text) % 10000:04d}"
        text_normalized = self._normalize_text(text)
        
        indicators = []
        matched_patterns = []
        suspicious_elements = []
        
        urgency_score, urgency_matches = self._calculate_urgency_score(text_normalized)
        matched_patterns.extend(urgency_matches)
        if urgency_score > 0.3:
            indicators.append(f"URGENCY: score={urgency_score:.2f}")
        
        emotional_score, emotional_matches = self._calculate_emotional_score(text_normalized)
        matched_patterns.extend(emotional_matches)
        if emotional_score > 0.3:
            indicators.append(f"EMOTIONAL: score={emotional_score:.2f}")
        
        bank_score, bank_matches = self._check_bank_impersonation(text_normalized)
        matched_patterns.extend(bank_matches)
        if bank_score > 0.3:
            indicators.append(f"BANK_IMPERSONATION: score={bank_score:.2f}")
        
        phishing_score, phishing_elements = self._check_phishing_links(text_normalized)
        suspicious_elements.extend(phishing_elements)
        if phishing_score > 0.3:
            indicators.append(f"PHISHING_LINK: score={phishing_score:.2f}")
        
        pix_score, pix_matches = self._check_pix_fraud(text_normalized)
        matched_patterns.extend(pix_matches)
        if pix_score > 0.3:
            indicators.append(f"PIX_FRAUD: score={pix_score:.2f}")
        
        whatsapp_score, whatsapp_matches = self._check_whatsapp_clone(text_normalized)
        matched_patterns.extend(whatsapp_matches)
        if whatsapp_score > 0.3:
            indicators.append(f"WHATSAPP_CLONE: score={whatsapp_score:.2f}")
        
        keyword_score, risky_keywords = self._check_blacklist_keywords(text_normalized)
        suspicious_elements.extend(risky_keywords)
        if keyword_score > 0.2:
            indicators.append(f"RISKY_KEYWORDS: {len(risky_keywords)} found")
        
        base_scores = [
            urgency_score * 0.20,
            emotional_score * 0.15,
            bank_score * 0.30,
            phishing_score * 0.25,
            pix_score * 0.20,
            whatsapp_score * 0.15,
            keyword_score * 0.10,
        ]
        
        base_probability = min(sum(base_scores), 1.0)
        
        high_risk_count = sum(1 for s in [urgency_score, bank_score, phishing_score, pix_score, whatsapp_score] if s > 0.3)
        combination_bonus = high_risk_count * 0.15
        
        fraud_probability = min(base_probability + combination_bonus, 1.0)
        
        score_map = {
            'BANK_IMPERSONATION': bank_score,
            'PHISHING': phishing_score,
            'PIX_FRAUD': pix_score,
            'WHATSAPP_CLONE': whatsapp_score,
            'URGENCY_SCAM': urgency_score,
            'EMOTIONAL_MANIPULATION': emotional_score,
        }
        fraud_type = max(score_map, key=score_map.get)
        
        if max(score_map.values()) < 0.3:
            fraud_type = "UNKNOWN"
        
        if fraud_probability > 0.7:
            recommendation = "BLOCK"
        elif fraud_probability > 0.5:
            recommendation = "REVIEW"
        elif fraud_probability > 0.3:
            recommendation = "WARN_USER"
        else:
            recommendation = "ALLOW"
        
        pattern_count = len(matched_patterns)
        confidence = min(0.5 + (pattern_count * 0.1), 0.95)
        
        return SocialEngineeringAnalysis(
            text_id=text_id,
            fraud_probability=fraud_probability,
            fraud_type=fraud_type,
            indicators=indicators,
            matched_patterns=list(set(matched_patterns)),
            urgency_score=urgency_score,
            emotional_score=emotional_score,
            suspicious_elements=list(set(suspicious_elements)),
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para análise"""
        text = text.lower()
        text = re.sub(r'[^\w\s\-@:/.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _calculate_urgency_score(self, text: str) -> Tuple[float, List[str]]:
        """Calcula score de urgência"""
        matches = []
        for pattern in self.compiled_urgency:
            found = pattern.findall(text)
            matches.extend(found)
        
        score = min(len(matches) / 3.0, 1.0)
        return score, matches
    
    def _calculate_emotional_score(self, text: str) -> Tuple[float, List[str]]:
        """Calcula score emocional"""
        matches = []
        for pattern in self.compiled_emotional:
            found = pattern.findall(text)
            matches.extend(found)
        
        score = min(len(matches) / 3.0, 1.0)
        return score, matches
    
    def _check_bank_impersonation(self, text: str) -> Tuple[float, List[str]]:
        """Verifica impersonação de banco"""
        matches = []
        for pattern in self.compiled_bank:
            found = pattern.findall(text)
            matches.extend(found)
        
        score = min(len(matches) / 4.0, 1.0)
        return score, matches
    
    def _check_phishing_links(self, text: str) -> Tuple[float, List[str]]:
        """Verifica links de phishing"""
        suspicious = []
        for pattern in self.compiled_phishing:
            found = pattern.findall(text)
            suspicious.extend(found)
        
        score = min(len(suspicious) / 2.0, 1.0)
        return score, suspicious
    
    def _check_pix_fraud(self, text: str) -> Tuple[float, List[str]]:
        """Verifica padrões de fraude PIX"""
        matches = []
        for pattern in self.compiled_pix:
            found = pattern.findall(text)
            matches.extend(found)
        
        score = min(len(matches) / 2.0, 1.0)
        return score, matches
    
    def _check_whatsapp_clone(self, text: str) -> Tuple[float, List[str]]:
        """Verifica padrões de clone de WhatsApp"""
        matches = []
        for pattern in self.compiled_whatsapp:
            found = pattern.findall(text)
            matches.extend(found)
        
        score = min(len(matches) / 2.0, 1.0)
        return score, matches
    
    def _check_blacklist_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Verifica palavras-chave de risco"""
        words = text.split()
        risky = []
        
        for word in words:
            if word in self.BLACKLIST_KEYWORDS['high_risk']:
                risky.append(f"HIGH:{word}")
            elif word in self.BLACKLIST_KEYWORDS['medium_risk']:
                risky.append(f"MED:{word}")
        
        score = min(len(risky) / 5.0, 1.0)
        return score, risky
    
    def batch_analyze(
        self,
        texts: List[str],
        source: Optional[str] = None
    ) -> List[SocialEngineeringAnalysis]:
        """Analisa múltiplos textos"""
        results = []
        for i, text in enumerate(texts):
            result = self.analyze_text(
                text=text,
                text_id=f"BATCH_{i:04d}",
                source=source
            )
            results.append(result)
        return results
    
    def get_pattern_summary(self) -> Dict[str, int]:
        """Retorna resumo dos padrões configurados"""
        return {
            'urgency_patterns': len(self.URGENCY_PATTERNS),
            'emotional_patterns': len(self.EMOTIONAL_PATTERNS),
            'bank_patterns': len(self.BANK_IMPERSONATION_PATTERNS),
            'phishing_patterns': len(self.PHISHING_PATTERNS),
            'pix_patterns': len(self.PIX_FRAUD_PATTERNS),
            'whatsapp_patterns': len(self.WHATSAPP_CLONE_PATTERNS),
            'blacklist_keywords': sum(len(v) for v in self.BLACKLIST_KEYWORDS.values()),
        }


def create_nlp_detector(config: Optional[Dict[str, Any]] = None) -> NLPSocialEngineeringDetector:
    """Factory function para criar detector"""
    return NLPSocialEngineeringDetector(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = NLPSocialEngineeringDetector()
    
    test_messages = [
        "URGENTE: Seu cartão foi bloqueado! Clique aqui para desbloquear: bit.ly/xyz123",
        "Oi mãe, troquei de número. Pode me fazer um PIX de R$500? Te pago amanhã.",
        "Parabéns! Você foi sorteado para ganhar R$10.000. Acesse agora: premio.tk",
        "Seu pedido foi enviado. Rastreie em: correios.com.br/rastreio/ABC123",
        "Bug do PIX! Envie R$100 e receba R$200 de volta automaticamente!",
    ]
    
    for msg in test_messages:
        result = detector.analyze_text(msg)
        print(f"\n{'='*50}")
        print(f"Message: {msg[:50]}...")
        print(f"Fraud Probability: {result.fraud_probability:.1%}")
        print(f"Type: {result.fraud_type}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Indicators: {result.indicators}")
