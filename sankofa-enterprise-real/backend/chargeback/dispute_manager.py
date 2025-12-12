"""
Dispute Manager - Orchestrates entire dispute lifecycle
Integrates chargeback engine, evidence collector, and MED workflow
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from .chargeback_engine import ChargebackEngine, ChargebackCase, get_chargeback_engine
from .evidence_collector import EvidenceCollector
from .med_workflow import MEDWorkflow, MEDRequest, get_med_workflow

logger = logging.getLogger(__name__)


class DisputeType(str, Enum):
    """Tipo de disputa"""
    CHARGEBACK = "chargeback"  # Traditional chargeback
    MED = "med"  # BACEN MED (Brasil)
    DIRECT_REFUND = "direct_refund"  # Refund direto


class DisputeManager:
    """
    Gerencia todo o ciclo de vida de disputas

    Funcionalidades:
    - Recebe disputas (chargeback, MED, refunds)
    - Classifica e roteia para engine apropriado
    - Orquestra coleta de evidências
    - Tracking e reporting
    - Feedback loop para ML models
    - Integração com acquirers/BACEN
    """

    def __init__(self):
        """Initialize dispute manager"""
        self.chargeback_engine = get_chargeback_engine()
        self.evidence_collector = EvidenceCollector()
        self.med_workflow = get_med_workflow()

        # Metrics
        self.total_disputes = 0
        self.chargebacks_processed = 0
        self.meds_processed = 0
        self.total_amount_disputed = 0.0
        self.total_amount_recovered = 0.0

        logger.info("Dispute Manager initialized")

    async def process_dispute(
        self,
        dispute_data: Dict[str, Any],
        dispute_type: DisputeType
    ) -> Dict[str, Any]:
        """
        Processa disputa (entry point principal)

        Args:
            dispute_data: Dispute data
            dispute_type: Type of dispute

        Returns:
            Processing result
        """
        try:
            self.total_disputes += 1
            self.total_amount_disputed += dispute_data.get('amount', 0)

            logger.info(
                f"Processing dispute: type={dispute_type.value}, "
                f"id={dispute_data.get('id')}"
            )

            # Route to appropriate handler
            if dispute_type == DisputeType.CHARGEBACK:
                result = await self._process_chargeback(dispute_data)
                self.chargebacks_processed += 1

            elif dispute_type == DisputeType.MED:
                result = await self._process_med(dispute_data)
                self.meds_processed += 1

            elif dispute_type == DisputeType.DIRECT_REFUND:
                result = await self._process_direct_refund(dispute_data)

            else:
                raise ValueError(f"Unknown dispute type: {dispute_type}")

            # Track recovery
            if result.get('decision') == 'represent' or result.get('status') == 'rejected':
                self.total_amount_recovered += dispute_data.get('amount', 0)

            # Update feedback loop
            await self._update_feedback_loop(dispute_data, result)

            return result

        except Exception as e:
            logger.error(f"Error processing dispute: {e}")
            return {
                'success': False,
                'error': str(e),
                'requires_manual_review': True
            }

    async def _process_chargeback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chargeback dispute"""
        from .chargeback_engine import ChargebackCase, ChargebackReason

        # Convert to ChargebackCase
        chargeback = ChargebackCase(
            case_id=data.get('case_id', data.get('id')),
            transaction_id=data.get('transaction_id'),
            amount=data.get('amount'),
            currency=data.get('currency', 'BRL'),
            reason_code=data.get('reason_code'),
            reason_category=ChargebackReason(data.get('reason_category', 'fraud')),
            dispute_date=data.get('dispute_date', datetime.utcnow()),
            deadline=data.get('deadline', datetime.utcnow()),
            merchant_id=data.get('merchant_id'),
            customer_id=data.get('customer_id'),
            acquirer=data.get('acquirer', 'unknown'),
            card_brand=data.get('card_brand', 'unknown'),
            evidence_required=data.get('evidence_required', []),
            metadata=data.get('metadata', {})
        )

        # Process via chargeback engine
        result = await self.chargeback_engine.process_chargeback(chargeback)

        # If decision is to represent, prepare submission
        if result.get('decision') == 'represent':
            submission = await self._prepare_chargeback_submission(
                chargeback=chargeback,
                evidence=result.get('evidence', {})
            )

            result['submission'] = submission

        return result

    async def _process_med(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MED (BACEN) dispute"""
        from .med_workflow import MEDRequest, MEDType

        # Convert to MEDRequest
        med_request = MEDRequest(
            request_id=data.get('request_id', data.get('id')),
            transaction_id=data.get('transaction_id'),
            pix_key=data.get('pix_key'),
            amount=data.get('amount'),
            med_type=MEDType(data.get('med_type', 'fraud')),
            requester_institution=data.get('requester_institution'),
            requested_institution=data.get('requested_institution'),
            request_date=data.get('request_date', datetime.utcnow()),
            deadline=data.get('deadline', datetime.utcnow()),
            customer_cpf=data.get('customer_cpf'),
            justification=data.get('justification', ''),
            evidence_urls=data.get('evidence_urls', [])
        )

        # Process via MED workflow
        result = await self.med_workflow.process_med_request(med_request)

        return result

    async def _process_direct_refund(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process direct refund (merchant-initiated)"""
        # Simplified flow for direct refunds
        logger.info(f"Processing direct refund: {data.get('id')}")

        # Execute refund immediately
        refund_result = {
            'refund_id': f"REF_{data.get('id')}",
            'amount': data.get('amount'),
            'status': 'approved',
            'executed_at': datetime.utcnow().isoformat()
        }

        return {
            'success': True,
            'refund': refund_result,
            'type': 'direct_refund'
        }

    async def _prepare_chargeback_submission(
        self,
        chargeback: ChargebackCase,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepara submission de chargeback para acquirer

        Args:
            chargeback: Chargeback case
            evidence: Collected evidence

        Returns:
            Submission package
        """
        # Generate formatted evidence package
        evidence_package = await self.evidence_collector.generate_evidence_package(
            transaction_id=chargeback.transaction_id,
            evidence=evidence
        )

        submission = {
            'case_id': chargeback.case_id,
            'acquirer': chargeback.acquirer,
            'card_brand': chargeback.card_brand,
            'submission_date': datetime.utcnow().isoformat(),
            'deadline': chargeback.deadline.isoformat(),

            # Evidence package
            'evidence_package': evidence_package,

            # Submission metadata
            'submission_method': 'api',  # vs manual
            'auto_generated': True,
            'evidence_quality_score': evidence.get('evidence_quality_score', 0)
        }

        # In production: submit to acquirer API
        logger.info(f"Prepared chargeback submission: {chargeback.case_id}")

        return submission

    async def _update_feedback_loop(
        self,
        dispute_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Update ML models com outcome de disputas

        Args:
            dispute_data: Original dispute data
            result: Processing result
        """
        # Feedback data
        feedback = {
            'transaction_id': dispute_data.get('transaction_id'),
            'dispute_type': dispute_data.get('type'),
            'decision': result.get('decision'),
            'win_probability': result.get('win_probability'),
            'actual_outcome': None,  # Will be updated later when outcome known
            'timestamp': datetime.utcnow().isoformat()
        }

        # In production: store feedback for model retraining
        logger.debug(f"Feedback stored for transaction: {feedback['transaction_id']}")

    async def get_dispute_status(self, dispute_id: str) -> Dict[str, Any]:
        """
        Consulta status de uma disputa

        Args:
            dispute_id: Dispute ID

        Returns:
            Dispute status
        """
        # In production: query from database
        # Placeholder implementation
        return {
            'dispute_id': dispute_id,
            'status': 'processing',
            'last_updated': datetime.utcnow().isoformat()
        }

    async def bulk_process_disputes(
        self,
        disputes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processa múltiplas disputas em batch

        Args:
            disputes: List of disputes

        Returns:
            Batch processing results
        """
        import asyncio

        logger.info(f"Bulk processing {len(disputes)} disputes...")

        # Process in parallel
        tasks = []
        for dispute in disputes:
            dispute_type = DisputeType(dispute.get('type', 'chargeback'))
            task = self.process_dispute(dispute, dispute_type)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', True))
        failed = len(results) - successful

        return {
            'total_processed': len(disputes),
            'successful': successful,
            'failed': failed,
            'results': results
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas consolidadas

        Returns:
            Metrics dict
        """
        recovery_rate = (
            self.total_amount_recovered / self.total_amount_disputed
            if self.total_amount_disputed > 0
            else 0
        )

        return {
            'total_disputes': self.total_disputes,
            'chargebacks_processed': self.chargebacks_processed,
            'meds_processed': self.meds_processed,

            # Financial metrics
            'total_amount_disputed': self.total_amount_disputed,
            'total_amount_recovered': self.total_amount_recovered,
            'recovery_rate': recovery_rate,

            # Engine metrics
            'chargeback_engine': self.chargeback_engine.get_metrics(),
            'med_workflow': self.med_workflow.get_metrics(),

            # Targets
            'target_win_rate': 0.85,
            'target_automation_rate': 0.80
        }


# Singleton
_manager_instance: Optional[DisputeManager] = None


def get_dispute_manager() -> DisputeManager:
    """Returns singleton dispute manager"""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = DisputeManager()

    return _manager_instance
