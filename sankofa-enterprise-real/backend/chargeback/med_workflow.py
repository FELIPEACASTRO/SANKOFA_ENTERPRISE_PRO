"""
MED Workflow - Mecanismo Especial de Devolução (Brasil)
Automated workflow for Brazilian instant refund system
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MEDType(str, Enum):
    """Tipos de MED conforme BACEN"""
    FRAUD = "fraud"  # Fraude confirmada
    UNRECOGNIZED = "unrecognized"  # Transação não reconhecida
    DUPLICATE = "duplicate"  # Duplicidade
    VALUE_DIVERGENCE = "value_divergence"  # Divergência de valor
    SERVICE_NOT_PROVIDED = "service_not_provided"  # Serviço não prestado


class MEDStatus(str, Enum):
    """Status do MED"""
    RECEIVED = "received"
    ANALYZING = "analyzing"
    APPROVED = "approved"
    REJECTED = "rejected"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


@dataclass
class MEDRequest:
    """Solicitação de MED"""
    request_id: str
    transaction_id: str
    pix_key: str
    amount: float
    med_type: MEDType
    requester_institution: str  # Instituição solicitante
    requested_institution: str  # Instituição solicitada (merchant)
    request_date: datetime
    deadline: datetime  # 7 dias úteis
    customer_cpf: str
    justification: str
    evidence_urls: List[str]


class MEDWorkflow:
    """
    Workflow automatizado para MED (BACEN)

    Regras BACEN:
    - Prazo de resposta: 7 dias úteis
    - Contestação: até 7 dias úteis após análise
    - Devolução: imediata se aprovado
    - Taxa: sem custo para o cliente

    Fluxo:
    1. Recebe solicitação MED
    2. Valida elegibilidade
    3. Análise automatizada (ML)
    4. Decisão: aceitar ou contestar
    5. Execução (refund ou defesa)
    6. Acompanhamento e feedback
    """

    # Deadlines BACEN (dias úteis)
    RESPONSE_DEADLINE_DAYS = 7
    DISPUTE_DEADLINE_DAYS = 7

    def __init__(self):
        """Initialize MED workflow"""
        self.requests_processed = 0
        self.auto_approved = 0
        self.auto_rejected = 0
        self.manual_review = 0

        logger.info("MED Workflow initialized")

    async def process_med_request(
        self,
        request: MEDRequest
    ) -> Dict[str, Any]:
        """
        Processa solicitação de MED

        Args:
            request: MED request

        Returns:
            Decision and actions
        """
        try:
            logger.info(f"Processing MED request: {request.request_id}")

            # Step 1: Validate request
            if not self._validate_med_request(request):
                return self._create_response(
                    status=MEDStatus.REJECTED,
                    reason="Invalid MED request"
                )

            # Step 2: Check deadline
            if datetime.utcnow() > request.deadline:
                logger.warning(f"MED past deadline: {request.request_id}")
                # Auto-approve se passou prazo
                return await self._auto_approve_med(request, "Past deadline")

            # Step 3: Analyze eligibility
            eligibility = await self._analyze_eligibility(request)

            if not eligibility['is_eligible']:
                return self._create_response(
                    status=MEDStatus.REJECTED,
                    reason=eligibility['reason']
                )

            # Step 4: Fraud analysis
            fraud_analysis = await self._analyze_fraud_claim(request)

            # Step 5: Make decision
            decision = await self._make_med_decision(
                request=request,
                fraud_analysis=fraud_analysis,
                eligibility=eligibility
            )

            # Step 6: Execute decision
            result = await self._execute_med_decision(request, decision)

            self.requests_processed += 1

            if decision['action'] == 'approve':
                self.auto_approved += 1
            elif decision['action'] == 'reject':
                self.auto_rejected += 1
            else:
                self.manual_review += 1

            return result

        except Exception as e:
            logger.error(f"Error processing MED: {e}")
            return self._create_response(
                status=MEDStatus.DISPUTED,
                reason=f"Error: {e}",
                requires_manual_review=True
            )

    async def _analyze_eligibility(self, request: MEDRequest) -> Dict[str, Any]:
        """
        Analisa elegibilidade da solicitação MED

        Critérios BACEN:
        - Transação PIX
        - Dentro de 80 dias
        - Evidências mínimas
        - Tipo de MED válido

        Args:
            request: MED request

        Returns:
            Eligibility analysis
        """
        # Check transaction age (80 dias BACEN)
        # Em produção, buscar transaction_date do banco
        transaction_age_days = 15  # Placeholder

        if transaction_age_days > 80:
            return {
                'is_eligible': False,
                'reason': 'Transaction older than 80 days (BACEN limit)'
            }

        # Check if PIX transaction
        # Em produção, verificar no banco
        is_pix = True  # Placeholder

        if not is_pix:
            return {
                'is_eligible': False,
                'reason': 'MED only applies to PIX transactions'
            }

        # Check evidence provided
        if not request.evidence_urls:
            return {
                'is_eligible': False,
                'reason': 'No evidence provided'
            }

        # Check MED type validity
        if request.med_type not in MEDType:
            return {
                'is_eligible': False,
                'reason': f'Invalid MED type: {request.med_type}'
            }

        return {
            'is_eligible': True,
            'reason': 'All eligibility criteria met',
            'confidence': 1.0
        }

    async def _analyze_fraud_claim(self, request: MEDRequest) -> Dict[str, Any]:
        """
        Analisa claim de fraude

        Args:
            request: MED request

        Returns:
            Fraud analysis
        """
        # Em produção, integrar com fraud detection engine
        from ml_engine.production_fraud_engine import get_fraud_engine

        fraud_engine = get_fraud_engine()

        # Get transaction data
        # Em produção, buscar do banco
        transaction_data = {
            'id': request.transaction_id,
            'amount': request.amount,
            'customer_id': request.customer_cpf,
            'channel': 'PIX'
        }

        # Run fraud detection
        try:
            fraud_result = fraud_engine.predict(transaction_data)

            is_likely_fraud = (
                hasattr(fraud_result, 'is_fraud') and fraud_result.is_fraud
            ) or (
                hasattr(fraud_result, 'risk_score') and fraud_result.risk_score > 0.7
            )

            return {
                'is_likely_fraud': is_likely_fraud,
                'fraud_score': fraud_result.risk_score if hasattr(fraud_result, 'risk_score') else 0.0,
                'fraud_indicators': fraud_result.matched_rules if hasattr(fraud_result, 'matched_rules') else [],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")
            return {
                'is_likely_fraud': False,
                'fraud_score': 0.0,
                'error': str(e)
            }

    async def _make_med_decision(
        self,
        request: MEDRequest,
        fraud_analysis: Dict[str, Any],
        eligibility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decide ação para MED

        Decision tree:
        - Fraude confirmada + evidências -> APPROVE
        - Sem fraude + transação legítima -> REJECT
        - Ambíguo -> MANUAL REVIEW

        Args:
            request: MED request
            fraud_analysis: Fraud analysis result
            eligibility: Eligibility analysis

        Returns:
            Decision
        """
        # High confidence fraud -> Auto-approve refund
        if fraud_analysis.get('is_likely_fraud') and fraud_analysis.get('fraud_score', 0) > 0.8:
            return {
                'action': 'approve',
                'reason': 'Confirmed fraud with high confidence',
                'auto_approved': True,
                'confidence': fraud_analysis.get('fraud_score', 0)
            }

        # MED Type: Fraud but low fraud score -> Review
        if request.med_type == MEDType.FRAUD:
            if not fraud_analysis.get('is_likely_fraud'):
                # Customer claims fraud but our detection says no
                return {
                    'action': 'manual_review',
                    'reason': 'Fraud claim but low fraud indicators',
                    'requires_investigation': True
                }

        # MED Type: Duplicate
        if request.med_type == MEDType.DUPLICATE:
            # Em produção, verificar duplicatas no banco
            has_duplicate = False  # Placeholder

            if has_duplicate:
                return {
                    'action': 'approve',
                    'reason': 'Duplicate transaction confirmed',
                    'auto_approved': True
                }
            else:
                return {
                    'action': 'reject',
                    'reason': 'No duplicate found',
                    'auto_rejected': True
                }

        # MED Type: Unrecognized
        if request.med_type == MEDType.UNRECOGNIZED:
            # Se não temos evidências fortes de legitimidade -> approve
            # (benefício ao consumidor)
            return {
                'action': 'approve',
                'reason': 'Unrecognized transaction - consumer protection',
                'auto_approved': True
            }

        # High value -> Always manual review
        if request.amount > 10000:
            return {
                'action': 'manual_review',
                'reason': 'High-value transaction requires manual review',
                'priority': 'HIGH'
            }

        # Default: Manual review
        return {
            'action': 'manual_review',
            'reason': 'Requires detailed investigation'
        }

    async def _execute_med_decision(
        self,
        request: MEDRequest,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa decisão do MED

        Args:
            request: MED request
            decision: Decision made

        Returns:
            Execution result
        """
        if decision['action'] == 'approve':
            # Execute refund
            refund_result = await self._execute_refund(request)

            return {
                'request_id': request.request_id,
                'status': MEDStatus.APPROVED.value,
                'decision': decision,
                'refund': refund_result,
                'processed_at': datetime.utcnow().isoformat(),
                'auto_processed': decision.get('auto_approved', False)
            }

        elif decision['action'] == 'reject':
            # Send rejection notification
            return {
                'request_id': request.request_id,
                'status': MEDStatus.REJECTED.value,
                'decision': decision,
                'processed_at': datetime.utcnow().isoformat(),
                'auto_processed': decision.get('auto_rejected', False)
            }

        else:  # manual_review
            # Create task for analyst
            return {
                'request_id': request.request_id,
                'status': MEDStatus.ANALYZING.value,
                'decision': decision,
                'assigned_to': 'fraud_analysis_team',
                'priority': decision.get('priority', 'MEDIUM'),
                'deadline': request.deadline.isoformat(),
                'requires_action': True
            }

    async def _execute_refund(self, request: MEDRequest) -> Dict[str, Any]:
        """
        Executa devolução via PIX

        Args:
            request: MED request

        Returns:
            Refund result
        """
        # Em produção, integrar com sistema PIX/BACEN
        try:
            refund_id = f"REF_{request.request_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Simulate PIX refund
            # Em produção: chamar API do banco/PSP

            logger.info(
                f"Executing MED refund: {refund_id}, "
                f"amount={request.amount}, pix_key={request.pix_key}"
            )

            return {
                'refund_id': refund_id,
                'amount': request.amount,
                'pix_key': request.pix_key,
                'status': 'completed',
                'executed_at': datetime.utcnow().isoformat(),
                'estimated_credit': (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }

        except Exception as e:
            logger.error(f"Error executing refund: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def _auto_approve_med(self, request: MEDRequest, reason: str) -> Dict[str, Any]:
        """Auto-approve MED (ex: past deadline)"""
        refund_result = await self._execute_refund(request)

        self.auto_approved += 1

        return {
            'request_id': request.request_id,
            'status': MEDStatus.APPROVED.value,
            'reason': reason,
            'auto_approved': True,
            'refund': refund_result,
            'processed_at': datetime.utcnow().isoformat()
        }

    def _validate_med_request(self, request: MEDRequest) -> bool:
        """Valida dados do request MED"""
        if not request.request_id or not request.transaction_id:
            return False

        if request.amount <= 0:
            return False

        if not request.pix_key or not request.customer_cpf:
            return False

        return True

    def _create_response(
        self,
        status: MEDStatus,
        reason: str,
        requires_manual_review: bool = False
    ) -> Dict[str, Any]:
        """Helper to create MED response"""
        return {
            'status': status.value,
            'reason': reason,
            'requires_manual_review': requires_manual_review,
            'processed_at': datetime.utcnow().isoformat()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do MED workflow"""
        total = self.requests_processed

        return {
            'requests_processed': total,
            'auto_approved': self.auto_approved,
            'auto_approved_rate': self.auto_approved / total if total > 0 else 0,
            'auto_rejected': self.auto_rejected,
            'auto_rejected_rate': self.auto_rejected / total if total > 0 else 0,
            'manual_review': self.manual_review,
            'manual_review_rate': self.manual_review / total if total > 0 else 0,
            'automation_rate': (self.auto_approved + self.auto_rejected) / total if total > 0 else 0
        }


# Singleton
_workflow_instance: Optional[MEDWorkflow] = None


def get_med_workflow() -> MEDWorkflow:
    """Returns singleton MED workflow"""
    global _workflow_instance

    if _workflow_instance is None:
        _workflow_instance = MEDWorkflow()

    return _workflow_instance
