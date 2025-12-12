"""
Sankofa Enterprise Pro - Intervention Engine
Sistema de intervenção em tempo real para prevenção de APP Fraud

Baseado em:
- FICO Scam Intervention
- UK PSR cooling-off regulations
- BioCatch real-time alerts
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Tipos de intervenção disponíveis"""
    ALLOW = "allow"
    SOFT_WARN = "soft_warn"
    WARN = "warn"
    CHALLENGE = "challenge"
    DELAY = "delay"
    CALL_VERIFICATION = "call_verification"
    BLOCK = "block"


class VerificationMethod(Enum):
    """Métodos de verificação adicional"""
    NONE = "none"
    SMS_OTP = "sms_otp"
    EMAIL_OTP = "email_otp"
    PUSH_NOTIFICATION = "push_notification"
    BIOMETRIC = "biometric"
    PHONE_CALL = "phone_call"
    VIDEO_CALL = "video_call"


@dataclass
class InterventionDecision:
    """Decisão de intervenção"""
    transaction_id: str
    intervention_type: InterventionType
    verification_method: VerificationMethod
    message: str
    cooling_off_minutes: int
    requires_acknowledgment: bool
    can_override: bool
    escalation_contact: Optional[str]
    risk_factors: List[str]
    next_steps: List[str]
    expires_at: Optional[datetime]
    created_at: datetime = field(default_factory=datetime.now)


class InterventionEngine:
    """
    Motor de Intervenção em Tempo Real

    Decide quando e como intervir em transações suspeitas:
    - Avisos informativos
    - Avisos fortes com confirmação
    - Desafios de autenticação
    - Períodos de cooling-off
    - Verificação por telefone
    - Bloqueio imediato
    """

    VERSION = "1.0.0"

    # Configuração de intervenção por nível de risco
    INTERVENTION_CONFIG = {
        "VERY_LOW": {
            "type": InterventionType.ALLOW,
            "verification": VerificationMethod.NONE,
            "cooling_off": 0,
            "requires_ack": False,
            "can_override": True
        },
        "LOW": {
            "type": InterventionType.SOFT_WARN,
            "verification": VerificationMethod.NONE,
            "cooling_off": 0,
            "requires_ack": False,
            "can_override": True
        },
        "MEDIUM": {
            "type": InterventionType.WARN,
            "verification": VerificationMethod.PUSH_NOTIFICATION,
            "cooling_off": 0,
            "requires_ack": True,
            "can_override": True
        },
        "HIGH": {
            "type": InterventionType.DELAY,
            "verification": VerificationMethod.SMS_OTP,
            "cooling_off": 15,
            "requires_ack": True,
            "can_override": False
        },
        "CRITICAL": {
            "type": InterventionType.CALL_VERIFICATION,
            "verification": VerificationMethod.PHONE_CALL,
            "cooling_off": 30,
            "requires_ack": True,
            "can_override": False
        }
    }

    # Templates de mensagem
    MESSAGE_TEMPLATES = {
        InterventionType.SOFT_WARN: {
            "title": "Atenção",
            "body": """
Detectamos que esta transação é diferente do seu padrão habitual.

Valor: R$ {amount:.2f}
Destinatário: {receiver}

Se você não reconhece esta operação, cancele imediatamente e entre em contato com seu banco.
            """.strip()
        },
        InterventionType.WARN: {
            "title": "⚠️ ATENÇÃO: Transação Suspeita",
            "body": """
Esta transação apresenta características que podem indicar um golpe.

Antes de confirmar, verifique:
✓ Você conhece pessoalmente o destinatário?
✓ Alguém está te pedindo para fazer isso com urgência?
✓ Prometeram retorno financeiro ou prêmio?
✓ Você está sendo pressionado a fazer essa transferência?

Se respondeu "sim" a qualquer pergunta, PARE e ligue para seu banco.

Valor: R$ {amount:.2f}
Destinatário: {receiver}
            """.strip()
        },
        InterventionType.CHALLENGE: {
            "title": "🔐 Verificação de Segurança Necessária",
            "body": """
Por sua segurança, precisamos verificar sua identidade antes de prosseguir.

Esta transação apresentou alertas em nosso sistema de proteção contra fraudes.

Valor: R$ {amount:.2f}
Destinatário: {receiver}

Por favor, complete a verificação para continuar.
            """.strip()
        },
        InterventionType.DELAY: {
            "title": "🕐 Período de Segurança",
            "body": """
Por sua segurança, esta transação será processada em {delay_minutes} minutos.

Este tempo permite que você:
- Confirme a identidade do destinatário
- Verifique se não está sendo vítima de golpe
- Entre em contato com seu banco se necessário

Você pode cancelar a qualquer momento durante este período.

Valor: R$ {amount:.2f}
Destinatário: {receiver}
Processamento às: {process_time}
            """.strip()
        },
        InterventionType.CALL_VERIFICATION: {
            "title": "📞 Verificação por Telefone Necessária",
            "body": """
Detectamos sinais de que você pode estar em uma situação de risco.

Um atendente entrará em contato para verificar esta transação.

IMPORTANTE:
- NÃO confirme a transação até falar conosco
- Se alguém estiver te pressionando, você está seguro
- Se você está em uma situação de emergência, disque 190

Valor: R$ {amount:.2f}

Você receberá uma ligação em breve do número {bank_phone}.
            """.strip()
        },
        InterventionType.BLOCK: {
            "title": "🚫 Transação Bloqueada",
            "body": """
Esta transação foi bloqueada pelo nosso sistema de proteção contra fraudes.

Motivo: Alto risco de ser um golpe

Valor: R$ {amount:.2f}
Destinatário: {receiver}

Se você acredita que isto é um erro, entre em contato com nosso suporte:
Telefone: {bank_phone}

Lembre-se: Nosso banco NUNCA solicita senhas ou transferências por telefone.
            """.strip()
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.bank_phone = self.config.get("bank_phone", "0800-123-4567")
        self.support_email = self.config.get("support_email", "suporte@banco.com.br")

        logger.info(f"InterventionEngine v{self.VERSION} initialized")

    def decide_intervention(
        self,
        transaction: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> InterventionDecision:
        """
        Decide qual intervenção aplicar baseado na avaliação de risco

        Args:
            transaction: Dados da transação
            risk_assessment: Resultado da avaliação de risco (do ScamDetector ou outro)

        Returns:
            InterventionDecision com a decisão de intervenção
        """
        transaction_id = transaction.get("id", "UNKNOWN")
        risk_level = risk_assessment.get("risk_level", "MEDIUM")
        risk_score = risk_assessment.get("scam_probability", 0.5)
        indicators = risk_assessment.get("indicators", [])
        duress_detected = risk_assessment.get("duress_detected", False)

        # Override para duress detectado
        if duress_detected:
            risk_level = "CRITICAL"

        # Obter configuração base
        config = self.INTERVENTION_CONFIG.get(risk_level, self.INTERVENTION_CONFIG["MEDIUM"])

        # Ajustar baseado em fatores específicos
        intervention_type = config["type"]
        verification_method = config["verification"]
        cooling_off = config["cooling_off"]

        # Ajustes baseados em valor
        amount = float(transaction.get("amount") or transaction.get("value") or 0)
        if amount > 10000 and intervention_type.value in ["soft_warn", "warn"]:
            intervention_type = InterventionType.CHALLENGE
            verification_method = VerificationMethod.SMS_OTP

        if amount > 50000 and intervention_type != InterventionType.BLOCK:
            intervention_type = InterventionType.DELAY
            cooling_off = max(cooling_off, 30)

        # Ajustar baseado em indicadores específicos
        if "possible_duress" in indicators or duress_detected:
            intervention_type = InterventionType.CALL_VERIFICATION
            verification_method = VerificationMethod.PHONE_CALL
            cooling_off = 30

        if "first_time_receiver" in indicators and amount > 5000:
            cooling_off = max(cooling_off, 10)

        # Gerar mensagem
        message = self._generate_message(
            intervention_type, transaction, cooling_off
        )

        # Definir próximos passos
        next_steps = self._get_next_steps(intervention_type, verification_method)

        # Determinar se pode fazer override
        can_override = risk_score < 0.9 and not duress_detected

        # Calcular expiração (para delays)
        expires_at = None
        if cooling_off > 0:
            expires_at = datetime.now() + timedelta(minutes=cooling_off)

        # Determinar contato de escalação
        escalation_contact = None
        if intervention_type in [InterventionType.CALL_VERIFICATION, InterventionType.BLOCK]:
            escalation_contact = self.bank_phone

        return InterventionDecision(
            transaction_id=transaction_id,
            intervention_type=intervention_type,
            verification_method=verification_method,
            message=message,
            cooling_off_minutes=cooling_off,
            requires_acknowledgment=config["requires_ack"],
            can_override=can_override,
            escalation_contact=escalation_contact,
            risk_factors=indicators[:5],
            next_steps=next_steps,
            expires_at=expires_at
        )

    def _generate_message(
        self,
        intervention_type: InterventionType,
        transaction: Dict[str, Any],
        cooling_off: int
    ) -> str:
        """Gera mensagem de intervenção personalizada"""
        template = self.MESSAGE_TEMPLATES.get(intervention_type)
        if not template:
            return "Por favor, verifique esta transação com cuidado."

        amount = float(transaction.get("amount") or transaction.get("value") or 0)
        receiver = transaction.get("receiver_name") or transaction.get("conta_recebedor") or "Não identificado"

        process_time = ""
        if cooling_off > 0:
            process_time = (datetime.now() + timedelta(minutes=cooling_off)).strftime("%H:%M")

        message = template["body"].format(
            amount=amount,
            receiver=receiver,
            delay_minutes=cooling_off,
            process_time=process_time,
            bank_phone=self.bank_phone
        )

        return message

    def _get_next_steps(
        self,
        intervention_type: InterventionType,
        verification_method: VerificationMethod
    ) -> List[str]:
        """Determina próximos passos para o usuário"""
        steps = []

        if intervention_type == InterventionType.SOFT_WARN:
            steps = [
                "Revise os detalhes da transação",
                "Confirme se conhece o destinatário",
                "Prossiga se tiver certeza"
            ]
        elif intervention_type == InterventionType.WARN:
            steps = [
                "Leia o aviso de segurança com atenção",
                "Responda às perguntas de verificação",
                "Marque que entendeu os riscos para continuar"
            ]
        elif intervention_type == InterventionType.CHALLENGE:
            steps = [
                "Complete a verificação de segurança",
                f"Confirme sua identidade via {verification_method.value}",
                "Aguarde a confirmação para prosseguir"
            ]
        elif intervention_type == InterventionType.DELAY:
            steps = [
                "Aguarde o período de segurança",
                "Use este tempo para verificar a transação",
                "Cancele a qualquer momento se tiver dúvidas"
            ]
        elif intervention_type == InterventionType.CALL_VERIFICATION:
            steps = [
                "Aguarde nossa ligação de verificação",
                "NÃO confirme a transação até falar conosco",
                "Se precisar, ligue para " + self.bank_phone
            ]
        elif intervention_type == InterventionType.BLOCK:
            steps = [
                "Esta transação foi bloqueada por segurança",
                "Entre em contato com o suporte se precisar",
                f"Ligue para {self.bank_phone}"
            ]

        return steps

    def process_user_response(
        self,
        intervention_decision: InterventionDecision,
        user_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processa a resposta do usuário à intervenção

        Args:
            intervention_decision: Decisão de intervenção original
            user_response: Resposta do usuário

        Returns:
            Dict com resultado do processamento
        """
        response_type = user_response.get("type")  # "acknowledge", "cancel", "override", "verify"
        verification_code = user_response.get("verification_code")

        result = {
            "transaction_id": intervention_decision.transaction_id,
            "original_intervention": intervention_decision.intervention_type.value,
            "user_response": response_type,
            "timestamp": datetime.now().isoformat()
        }

        # Cancelamento
        if response_type == "cancel":
            result["outcome"] = "CANCELLED"
            result["can_proceed"] = False
            result["message"] = "Transação cancelada pelo usuário."
            return result

        # Acknowledgemnt (para soft_warn e warn)
        if response_type == "acknowledge":
            if intervention_decision.intervention_type in [
                InterventionType.SOFT_WARN,
                InterventionType.WARN
            ]:
                result["outcome"] = "ACKNOWLEDGED"
                result["can_proceed"] = True
                result["message"] = "Usuário reconheceu o aviso."
            else:
                result["outcome"] = "REQUIRES_VERIFICATION"
                result["can_proceed"] = False
                result["message"] = "Verificação adicional necessária."
            return result

        # Verificação de código
        if response_type == "verify" and verification_code:
            # Aqui seria feita a validação real do código
            is_valid = self._validate_verification_code(
                intervention_decision, verification_code
            )

            if is_valid:
                result["outcome"] = "VERIFIED"
                result["can_proceed"] = True
                result["message"] = "Verificação concluída com sucesso."
            else:
                result["outcome"] = "VERIFICATION_FAILED"
                result["can_proceed"] = False
                result["message"] = "Código de verificação inválido."
            return result

        # Override (apenas se permitido)
        if response_type == "override":
            if intervention_decision.can_override:
                result["outcome"] = "OVERRIDDEN"
                result["can_proceed"] = True
                result["message"] = "Transação autorizada pelo usuário."
                result["requires_audit"] = True
            else:
                result["outcome"] = "OVERRIDE_DENIED"
                result["can_proceed"] = False
                result["message"] = "Esta transação não pode ser sobrescrita."
            return result

        # Resposta não reconhecida
        result["outcome"] = "INVALID_RESPONSE"
        result["can_proceed"] = False
        result["message"] = "Resposta não reconhecida."
        return result

    def _validate_verification_code(
        self,
        intervention: InterventionDecision,
        code: str
    ) -> bool:
        """
        Valida código de verificação

        Em produção, isso verificaria contra um sistema real de OTP
        """
        # Placeholder - em produção, verificar contra sistema real
        return len(code) == 6 and code.isdigit()

    def get_intervention_stats(
        self,
        interventions: List[InterventionDecision]
    ) -> Dict[str, Any]:
        """Retorna estatísticas das intervenções"""
        if not interventions:
            return {}

        type_counts = {}
        for intervention in interventions:
            t = intervention.intervention_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        avg_cooling_off = np.mean([
            i.cooling_off_minutes for i in interventions
            if i.cooling_off_minutes > 0
        ]) if any(i.cooling_off_minutes > 0 for i in interventions) else 0

        return {
            "total_interventions": len(interventions),
            "by_type": type_counts,
            "avg_cooling_off_minutes": round(avg_cooling_off, 1),
            "requires_call_verification": sum(
                1 for i in interventions
                if i.verification_method == VerificationMethod.PHONE_CALL
            ),
            "blocked": sum(
                1 for i in interventions
                if i.intervention_type == InterventionType.BLOCK
            )
        }


def create_intervention_engine(config: Optional[Dict[str, Any]] = None) -> InterventionEngine:
    """Factory function para criar InterventionEngine"""
    return InterventionEngine(config)
