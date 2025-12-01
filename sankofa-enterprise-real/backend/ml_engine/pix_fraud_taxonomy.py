"""
Sankofa Enterprise Pro - PIX Fraud Taxonomy Module
Baseado em: arXiv:2511.20902 - Taxonomia de Fraudes PIX Brasil (2025)

Tipos de fraude implementados:
1. QR-Code Adulterado (QR_TAMPERED)
2. Mão Fantasma (GHOST_HAND)
3. Central Falsa (FAKE_BANK_CENTER)
4. Clone WhatsApp (WHATSAPP_CLONE)
5. PIX Errado (WRONG_PIX)
6. Comprovante Falso (FAKE_RECEIPT)
7. Sequestro Relâmpago (KIDNAPPING)
8. Golpe do Falso Funcionário (FAKE_EMPLOYEE)
9. Leilão/Marketplace Falso (FAKE_MARKETPLACE)
10. Bug do PIX (PIX_BUG)

Contexto: ~70% das perdas de fraude no Brasil são de engenharia social
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class PIXFraudType(Enum):
    """Tipos de fraude PIX baseados em taxonomia arXiv:2511.20902"""
    QR_TAMPERED = "qr_code_adulterado"
    GHOST_HAND = "mao_fantasma"
    FAKE_BANK_CENTER = "central_falsa"
    WHATSAPP_CLONE = "clone_whatsapp"
    WRONG_PIX = "pix_errado"
    FAKE_RECEIPT = "comprovante_falso"
    KIDNAPPING = "sequestro_relampago"
    FAKE_EMPLOYEE = "falso_funcionario"
    FAKE_MARKETPLACE = "leilao_falso"
    PIX_BUG = "bug_do_pix"
    SOCIAL_ENGINEERING = "engenharia_social_geral"
    ACCOUNT_TAKEOVER = "tomada_de_conta"
    UNKNOWN = "desconhecido"


@dataclass
class PIXFraudIndicator:
    """Indicador de fraude PIX"""
    indicator_type: str
    weight: float
    description: str
    detected: bool = False
    confidence: float = 0.0


@dataclass
class PIXFraudAnalysis:
    """Resultado da análise de fraude PIX"""
    transaction_id: str
    fraud_probability: float
    predicted_fraud_type: PIXFraudType
    indicators_detected: List[PIXFraudIndicator]
    risk_factors: Dict[str, float]
    recommended_action: str
    explanation: str
    compliance_flags: List[str]


class PIXFraudTaxonomy:
    """
    Analisador de fraudes PIX baseado em taxonomia acadêmica
    
    Implementa detecção de 10+ tipos de fraude PIX com
    regras específicas para o mercado brasileiro.
    """
    
    VERSION = "1.0.0"
    
    FRAUD_RULES = {
        PIXFraudType.QR_TAMPERED: {
            'indicators': [
                'broadcast_donation',
                'ngo_stream',
                'qr_mismatch',
                'recipient_first_transaction',
                'urgent_context',
            ],
            'risk_weight': 0.9,
            'min_indicators': 2,
            'description': 'QR Code adulterado em lives/doações',
        },
        PIXFraudType.GHOST_HAND: {
            'indicators': [
                'remote_access_detected',
                'fear_inducing_context',
                'bank_impersonation',
                'unusual_session_behavior',
                'device_anomaly',
            ],
            'risk_weight': 0.95,
            'min_indicators': 2,
            'description': 'Acesso remoto não autorizado ao dispositivo',
        },
        PIXFraudType.FAKE_BANK_CENTER: {
            'indicators': [
                'phone_call_context',
                'bank_impersonation',
                'urgency_pressure',
                'personal_data_request',
                'transfer_to_new_account',
            ],
            'risk_weight': 0.85,
            'min_indicators': 2,
            'description': 'Golpista se passa por central do banco',
        },
        PIXFraudType.WHATSAPP_CLONE: {
            'indicators': [
                'profile_change_recent',
                'urgent_money_request',
                'family_contact_impersonation',
                'new_phone_number',
                'emotional_appeal',
            ],
            'risk_weight': 0.88,
            'min_indicators': 2,
            'description': 'Golpista se passa por familiar/amigo no WhatsApp',
        },
        PIXFraudType.WRONG_PIX: {
            'indicators': [
                'request_return_transfer',
                'first_contact_recipient',
                'round_amount',
                'immediate_return_request',
                'emotional_story',
            ],
            'risk_weight': 0.75,
            'min_indicators': 2,
            'description': 'Golpe do PIX enviado por engano',
        },
        PIXFraudType.FAKE_RECEIPT: {
            'indicators': [
                'screenshot_shared',
                'no_notification_received',
                'pressure_for_delivery',
                'marketplace_context',
                'rush_to_complete',
            ],
            'risk_weight': 0.80,
            'min_indicators': 2,
            'description': 'Comprovante de PIX falsificado',
        },
        PIXFraudType.KIDNAPPING: {
            'indicators': [
                'night_transaction',
                'unusual_location',
                'multiple_rapid_transfers',
                'max_limit_transactions',
                'device_movement_anomaly',
            ],
            'risk_weight': 0.98,
            'min_indicators': 3,
            'description': 'Transação sob coação física',
        },
        PIXFraudType.FAKE_EMPLOYEE: {
            'indicators': [
                'employee_impersonation',
                'company_context',
                'invoice_fraud',
                'new_bank_details',
                'urgency_pressure',
            ],
            'risk_weight': 0.82,
            'min_indicators': 2,
            'description': 'Golpista se passa por funcionário de empresa',
        },
        PIXFraudType.FAKE_MARKETPLACE: {
            'indicators': [
                'too_good_price',
                'upfront_payment_request',
                'no_product_verification',
                'pressure_to_pay_now',
                'new_seller_account',
            ],
            'risk_weight': 0.78,
            'min_indicators': 2,
            'description': 'Venda falsa em marketplace/leilão',
        },
        PIXFraudType.PIX_BUG: {
            'indicators': [
                'promise_of_return',
                'test_transaction_request',
                'social_media_promotion',
                'guaranteed_profit',
                'pyramid_scheme_pattern',
            ],
            'risk_weight': 0.85,
            'min_indicators': 2,
            'description': 'Golpe prometendo retorno automático de PIX',
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        
        logger.info(f"PIX Fraud Taxonomy v{self.VERSION} initialized")
    
    def analyze_transaction(
        self,
        transaction_id: str,
        amount: float,
        timestamp: datetime,
        sender_id: str,
        receiver_id: str,
        pix_key_type: Optional[str] = None,
        channel: Optional[str] = None,
        device_info: Optional[Dict] = None,
        context_indicators: Optional[List[str]] = None,
        historical_data: Optional[Dict] = None,
        **kwargs
    ) -> PIXFraudAnalysis:
        """
        Analisa transação PIX para fraude
        
        Args:
            transaction_id: ID da transação
            amount: Valor
            timestamp: Data/hora
            sender_id: ID do remetente
            receiver_id: ID do destinatário
            pix_key_type: Tipo de chave PIX (CPF, CNPJ, email, telefone, aleatoria)
            channel: Canal (app, web, etc)
            device_info: Informações do dispositivo
            context_indicators: Indicadores de contexto
            historical_data: Dados históricos
            
        Returns:
            Análise de fraude PIX
        """
        context_indicators = context_indicators or []
        device_info = device_info or {}
        historical_data = historical_data or {}
        
        detected_indicators = []
        risk_factors = {}
        
        temporal_indicators = self._analyze_temporal_patterns(timestamp, amount)
        detected_indicators.extend(temporal_indicators)
        
        amount_indicators = self._analyze_amount_patterns(amount, historical_data)
        detected_indicators.extend(amount_indicators)
        
        receiver_indicators = self._analyze_receiver_patterns(
            receiver_id, historical_data
        )
        detected_indicators.extend(receiver_indicators)
        
        device_indicators = self._analyze_device_patterns(device_info)
        detected_indicators.extend(device_indicators)
        
        for indicator in context_indicators:
            detected_indicators.append(PIXFraudIndicator(
                indicator_type=indicator,
                weight=0.3,
                description=f"Contexto: {indicator}",
                detected=True,
                confidence=0.7
            ))
        
        fraud_scores = {}
        for fraud_type, rules in self.FRAUD_RULES.items():
            score = self._calculate_fraud_type_score(
                fraud_type, rules, detected_indicators
            )
            fraud_scores[fraud_type] = score
            risk_factors[fraud_type.value] = score
        
        predicted_type = max(fraud_scores, key=fraud_scores.get)
        max_score = fraud_scores[predicted_type]
        
        base_risk = sum(ind.weight * ind.confidence for ind in detected_indicators if ind.detected)
        high_weight_indicators = [ind for ind in detected_indicators if ind.weight >= 0.5 and ind.detected]
        critical_boost = len(high_weight_indicators) * 0.2
        
        combined_probability = max(max_score, min(base_risk * 0.5 + critical_boost, 1.0))
        
        if max_score < self.threshold and combined_probability >= self.threshold:
            predicted_type = PIXFraudType.SOCIAL_ENGINEERING
        elif max_score < self.threshold:
            predicted_type = PIXFraudType.UNKNOWN
        
        fraud_probability = combined_probability
        
        if fraud_probability > 0.8:
            recommended_action = "BLOCK"
        elif fraud_probability > 0.6:
            recommended_action = "REVIEW"
        elif fraud_probability > 0.4:
            recommended_action = "MONITOR"
        else:
            recommended_action = "APPROVE"
        
        explanation = self._generate_explanation(
            predicted_type, detected_indicators, fraud_probability
        )
        
        compliance_flags = self._generate_compliance_flags(
            amount, timestamp, predicted_type
        )
        
        return PIXFraudAnalysis(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            predicted_fraud_type=predicted_type,
            indicators_detected=detected_indicators,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            explanation=explanation,
            compliance_flags=compliance_flags
        )
    
    def _analyze_temporal_patterns(
        self,
        timestamp: datetime,
        amount: float
    ) -> List[PIXFraudIndicator]:
        """Analisa padrões temporais"""
        indicators = []
        
        hour = timestamp.hour
        if hour >= 22 or hour < 6:
            indicators.append(PIXFraudIndicator(
                indicator_type='night_transaction',
                weight=0.4,
                description='Transação em horário noturno (22h-6h)',
                detected=True,
                confidence=0.9
            ))
        
        if timestamp.weekday() >= 5:
            indicators.append(PIXFraudIndicator(
                indicator_type='weekend_transaction',
                weight=0.2,
                description='Transação em final de semana',
                detected=True,
                confidence=0.9
            ))
        
        if hour >= 22 or hour < 6:
            if amount > 1000:
                indicators.append(PIXFraudIndicator(
                    indicator_type='high_value_night',
                    weight=0.5,
                    description='Alto valor em horário noturno (limite PIX reduzido)',
                    detected=True,
                    confidence=0.85
                ))
        
        return indicators
    
    def _analyze_amount_patterns(
        self,
        amount: float,
        historical_data: Dict
    ) -> List[PIXFraudIndicator]:
        """Analisa padrões de valor"""
        indicators = []
        
        if amount % 100 == 0:
            indicators.append(PIXFraudIndicator(
                indicator_type='round_amount',
                weight=0.15,
                description='Valor redondo (múltiplo de 100)',
                detected=True,
                confidence=0.8
            ))
        
        user_avg = historical_data.get('user_avg_amount', amount)
        if amount > user_avg * 3:
            indicators.append(PIXFraudIndicator(
                indicator_type='unusual_amount',
                weight=0.4,
                description=f'Valor {amount/user_avg:.1f}x maior que média do usuário',
                detected=True,
                confidence=0.85
            ))
        
        if amount >= 5000:
            indicators.append(PIXFraudIndicator(
                indicator_type='high_value_transaction',
                weight=0.3,
                description='Transação de alto valor (>R$5000)',
                detected=True,
                confidence=0.9
            ))
        
        user_max = historical_data.get('user_max_amount', amount)
        if amount > user_max:
            indicators.append(PIXFraudIndicator(
                indicator_type='max_limit_transactions',
                weight=0.35,
                description='Maior transação já realizada pelo usuário',
                detected=True,
                confidence=0.8
            ))
        
        return indicators
    
    def _analyze_receiver_patterns(
        self,
        receiver_id: str,
        historical_data: Dict
    ) -> List[PIXFraudIndicator]:
        """Analisa padrões do destinatário"""
        indicators = []
        
        known_receivers = historical_data.get('known_receivers', [])
        if receiver_id not in known_receivers:
            indicators.append(PIXFraudIndicator(
                indicator_type='first_contact_recipient',
                weight=0.35,
                description='Primeira transação para este destinatário',
                detected=True,
                confidence=0.9
            ))
            
            indicators.append(PIXFraudIndicator(
                indicator_type='recipient_first_transaction',
                weight=0.3,
                description='Destinatário nunca recebeu PIX deste usuário',
                detected=True,
                confidence=0.9
            ))
        
        receiver_risk = historical_data.get('receiver_fraud_rate', 0)
        if receiver_risk > 0.1:
            indicators.append(PIXFraudIndicator(
                indicator_type='high_risk_receiver',
                weight=0.5,
                description=f'Destinatário com histórico de fraude ({receiver_risk:.1%})',
                detected=True,
                confidence=0.95
            ))
        
        receiver_age_days = historical_data.get('receiver_account_age_days', 365)
        if receiver_age_days < 30:
            indicators.append(PIXFraudIndicator(
                indicator_type='new_receiver_account',
                weight=0.4,
                description=f'Conta destinatária muito recente ({receiver_age_days} dias)',
                detected=True,
                confidence=0.85
            ))
        
        return indicators
    
    def _analyze_device_patterns(
        self,
        device_info: Dict
    ) -> List[PIXFraudIndicator]:
        """Analisa padrões do dispositivo"""
        indicators = []
        
        if device_info.get('remote_access_detected'):
            indicators.append(PIXFraudIndicator(
                indicator_type='remote_access_detected',
                weight=0.8,
                description='Software de acesso remoto detectado (AnyDesk, TeamViewer)',
                detected=True,
                confidence=0.95
            ))
        
        if device_info.get('is_new_device'):
            indicators.append(PIXFraudIndicator(
                indicator_type='device_anomaly',
                weight=0.35,
                description='Dispositivo não reconhecido',
                detected=True,
                confidence=0.85
            ))
        
        if device_info.get('location_anomaly'):
            indicators.append(PIXFraudIndicator(
                indicator_type='unusual_location',
                weight=0.4,
                description='Localização incomum para este usuário',
                detected=True,
                confidence=0.8
            ))
        
        if device_info.get('unusual_session_behavior'):
            indicators.append(PIXFraudIndicator(
                indicator_type='unusual_session_behavior',
                weight=0.45,
                description='Comportamento de sessão anormal',
                detected=True,
                confidence=0.85
            ))
        
        return indicators
    
    def _calculate_fraud_type_score(
        self,
        fraud_type: PIXFraudType,
        rules: Dict,
        detected_indicators: List[PIXFraudIndicator]
    ) -> float:
        """Calcula score para tipo específico de fraude"""
        required_indicators = rules['indicators']
        min_indicators = rules['min_indicators']
        risk_weight = rules['risk_weight']
        
        detected_types = {ind.indicator_type for ind in detected_indicators if ind.detected}
        matched = detected_types.intersection(set(required_indicators))
        
        if len(matched) < min_indicators:
            return 0.0
        
        match_ratio = len(matched) / len(required_indicators)
        
        matched_indicators = [
            ind for ind in detected_indicators 
            if ind.indicator_type in matched
        ]
        avg_confidence = np.mean([ind.confidence for ind in matched_indicators]) if matched_indicators else 0.5
        
        score = match_ratio * risk_weight * avg_confidence
        
        return min(score, 1.0)
    
    def _generate_explanation(
        self,
        fraud_type: PIXFraudType,
        indicators: List[PIXFraudIndicator],
        probability: float
    ) -> str:
        """Gera explicação da análise para compliance LGPD"""
        if fraud_type == PIXFraudType.UNKNOWN:
            return "Nenhum padrão de fraude significativo detectado."
        
        rules = self.FRAUD_RULES.get(fraud_type, {})
        fraud_description = rules.get('description', 'Tipo de fraude não identificado')
        
        detected = [ind for ind in indicators if ind.detected]
        top_indicators = sorted(detected, key=lambda x: x.weight, reverse=True)[:3]
        
        indicator_texts = [f"- {ind.description}" for ind in top_indicators]
        
        explanation = f"""
Tipo de fraude detectado: {fraud_description}
Probabilidade: {probability:.1%}

Principais indicadores:
{chr(10).join(indicator_texts)}

Esta análise foi gerada automaticamente pelo sistema de detecção de fraudes
Sankofa Enterprise Pro, em conformidade com as regulamentações BACEN e LGPD.
        """.strip()
        
        return explanation
    
    def _generate_compliance_flags(
        self,
        amount: float,
        timestamp: datetime,
        fraud_type: PIXFraudType
    ) -> List[str]:
        """Gera flags de compliance BACEN/LGPD"""
        flags = []
        
        hour = timestamp.hour
        if (hour >= 20 or hour < 6) and amount > 1000:
            flags.append("BACEN_LIMITE_NOTURNO")
        
        if fraud_type in [PIXFraudType.KIDNAPPING, PIXFraudType.GHOST_HAND]:
            flags.append("MED_ELEGIVEL")
        
        if amount > 10000:
            flags.append("BACEN_ALTO_VALOR")
        
        if fraud_type != PIXFraudType.UNKNOWN:
            flags.append("LGPD_EXPLICACAO_REQUERIDA")
        
        return flags
    
    def get_fraud_types_summary(self) -> Dict[str, Dict]:
        """Retorna resumo dos tipos de fraude"""
        summary = {}
        for fraud_type, rules in self.FRAUD_RULES.items():
            summary[fraud_type.value] = {
                'name': fraud_type.name,
                'description': rules['description'],
                'risk_weight': rules['risk_weight'],
                'indicators_count': len(rules['indicators']),
            }
        return summary


def create_pix_taxonomy(config: Optional[Dict[str, Any]] = None) -> PIXFraudTaxonomy:
    """Factory function para criar analisador"""
    return PIXFraudTaxonomy(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    analyzer = PIXFraudTaxonomy()
    
    result = analyzer.analyze_transaction(
        transaction_id="TXN001",
        amount=5000,
        timestamp=datetime.now().replace(hour=23),
        sender_id="USER001",
        receiver_id="NEW_RECEIVER",
        device_info={'remote_access_detected': True},
        context_indicators=['fear_inducing_context', 'bank_impersonation'],
        historical_data={'known_receivers': [], 'user_avg_amount': 500}
    )
    
    print(f"\nFraud Probability: {result.fraud_probability:.1%}")
    print(f"Predicted Type: {result.predicted_fraud_type.value}")
    print(f"Recommended Action: {result.recommended_action}")
    print(f"\nExplanation:\n{result.explanation}")
