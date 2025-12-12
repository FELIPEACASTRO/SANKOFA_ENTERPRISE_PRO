"""
Evidence Collector - Automated Evidence Gathering
Collects transaction proof, delivery confirmation, customer data
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import asyncio

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """
    Coleta evidências automaticamente para disputas de chargeback

    Evidence types:
    - Transaction details (timestamp, amount, method)
    - Customer information (history, risk profile)
    - Device/IP fingerprinting
    - Delivery confirmation (tracking, signature)
    - Communication logs (emails, SMS)
    - Prior successful transactions
    - Merchant documentation
    """

    def __init__(self):
        """Initialize evidence collector"""
        logger.info("Evidence Collector initialized")

    async def collect_evidence(
        self,
        transaction_id: str,
        reason_category: str
    ) -> Dict[str, Any]:
        """
        Coleta todas evidências relevantes para uma transação

        Args:
            transaction_id: Transaction ID
            reason_category: Chargeback reason category

        Returns:
            Dict com todas evidências
        """
        try:
            logger.info(f"Collecting evidence for transaction: {transaction_id}")

            # Collect in parallel for speed
            evidence_tasks = [
                self._collect_transaction_data(transaction_id),
                self._collect_customer_data(transaction_id),
                self._collect_device_data(transaction_id),
                self._collect_delivery_proof(transaction_id),
                self._collect_communication_logs(transaction_id),
                self._collect_merchant_data(transaction_id),
            ]

            results = await asyncio.gather(*evidence_tasks, return_exceptions=True)

            # Merge all evidence
            evidence = {}
            for result in results:
                if isinstance(result, dict):
                    evidence.update(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error collecting evidence: {result}")

            # Add metadata
            evidence['collected_at'] = datetime.now(timezone.utc).isoformat()
            evidence['evidence_count'] = len([v for v in evidence.values() if v])
            evidence['evidence_quality_score'] = self._calculate_evidence_quality(evidence)

            logger.info(
                f"Evidence collected: {transaction_id}, "
                f"count={evidence['evidence_count']}, "
                f"quality={evidence['evidence_quality_score']:.2f}"
            )

            return evidence

        except Exception as e:
            logger.error(f"Error collecting evidence: {e}")
            return {'error': str(e)}

    async def _collect_transaction_data(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta dados da transação"""
        # Em produção, buscar do banco de dados
        # Placeholder com dados simulados

        return {
            'transaction_id': transaction_id,
            'transaction_date': datetime.now(timezone.utc) - timedelta(days=15),
            'amount': 1500.0,
            'currency': 'BRL',
            'payment_method': 'credit_card',
            'card_last4': '1234',
            'card_brand': 'visa',
            'authorization_code': 'AUTH123456',
            'avs_result': 'match',
            'cvv_result': 'match',
            'transaction_status': 'approved',
            'merchant_id': 'MERCHANT_123',
            'merchant_name': 'Loja Virtual XYZ',
            'merchant_category': 'electronics',
        }

    async def _collect_customer_data(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta dados do cliente"""
        return {
            'customer_id': 'CUST_456',
            'customer_email': 'customer@example.com',
            'customer_name': 'João Silva',
            'customer_phone': '+5511999999999',
            'customer_since': datetime.now(timezone.utc) - timedelta(days=365),
            'account_age_days': 365,

            # Historical data
            'prior_transactions_count': 12,
            'prior_successful_transactions': 11,
            'prior_chargebacks': 0,
            'total_lifetime_value': 15000.0,

            # Risk indicators
            'customer_risk_score': 0.15,  # Low risk
            'is_repeat_customer': True,
            'has_verified_email': True,
            'has_verified_phone': True,
        }

    async def _collect_device_data(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta dados de device/IP fingerprinting"""
        return {
            'device_id': 'DEV_789',
            'device_fingerprint': 'fp_abc123xyz',
            'device_type': 'mobile',
            'device_os': 'iOS 15.0',
            'device_browser': 'Safari',

            # IP data
            'ip_address': '192.168.1.100',
            'ip_country': 'BR',
            'ip_city': 'São Paulo',
            'ip_location_match': True,  # Matches billing address

            # Device history
            'device_first_seen': datetime.now(timezone.utc) - timedelta(days=180),
            'device_transaction_count': 5,
            'is_known_device': True,
            'device_risk_score': 0.10,  # Low risk

            # Fraud signals
            'is_proxy': False,
            'is_vpn': False,
            'is_tor': False,
            'device_fingerprint_match': True,
        }

    async def _collect_delivery_proof(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta comprovantes de entrega"""
        return {
            'has_delivery': True,
            'delivery_method': 'courier',
            'tracking_number': 'TRACK123456789',
            'delivery_date': datetime.now(timezone.utc) - timedelta(days=10),
            'delivery_status': 'delivered',

            # Proof of delivery
            'has_signature': True,
            'signature_name': 'J. Silva',
            'signature_match': True,  # Matches customer name

            # Delivery address
            'delivery_address': 'Rua Example, 123, São Paulo, SP',
            'delivery_address_match': True,  # Matches billing

            # Photos/confirmation
            'has_delivery_photo': True,
            'has_gps_confirmation': True,
            'delivery_confirmation': True,

            # Carrier info
            'carrier': 'Correios',
            'carrier_proof_url': 'https://tracking.example.com/TRACK123456789'
        }

    async def _collect_communication_logs(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta logs de comunicação com cliente"""
        return {
            'has_order_confirmation_email': True,
            'order_confirmation_sent_at': datetime.now(timezone.utc) - timedelta(days=15),
            'order_confirmation_opened': True,

            'has_shipping_notification': True,
            'shipping_notification_sent_at': datetime.now(timezone.utc) - timedelta(days=12),

            'has_delivery_confirmation_email': True,
            'delivery_confirmation_sent_at': datetime.now(timezone.utc) - timedelta(days=10),

            # Customer service
            'customer_service_contacts': 0,
            'has_complaints': False,
            'has_refund_requests': False,

            # Marketing consent
            'has_marketing_consent': True,
            'email_list_subscribed': True,
        }

    async def _collect_merchant_data(self, transaction_id: str) -> Dict[str, Any]:
        """Coleta dados do merchant"""
        return {
            'merchant_id': 'MERCHANT_123',
            'merchant_name': 'Loja Virtual XYZ',
            'merchant_category': 'electronics',

            # Merchant history
            'merchant_account_age_days': 730,
            'merchant_transaction_volume_30d': 1500000.0,
            'merchant_chargeback_rate': 0.008,  # 0.8%
            'merchant_win_rate': 0.75,  # 75% win rate histórico

            # Compliance
            'has_terms_of_service': True,
            'has_return_policy': True,
            'has_privacy_policy': True,
            'is_pci_compliant': True,

            # Product info
            'product_id': 'PROD_456',
            'product_name': 'Smartphone XYZ',
            'product_category': 'electronics',
            'product_price': 1500.0,
            'is_digital_good': False,
            'is_physical_good': True,
        }

    def _calculate_evidence_quality(self, evidence: Dict[str, Any]) -> float:
        """
        Calcula quality score das evidências (0.0 to 1.0)

        Args:
            evidence: Evidence dict

        Returns:
            Quality score
        """
        score = 0.0
        max_score = 0.0

        # Transaction proof (30%)
        max_score += 0.30
        if evidence.get('authorization_code'):
            score += 0.15
        if evidence.get('avs_result') == 'match':
            score += 0.10
        if evidence.get('cvv_result') == 'match':
            score += 0.05

        # Delivery proof (40%)
        max_score += 0.40
        if evidence.get('delivery_confirmation'):
            score += 0.20
        if evidence.get('has_signature'):
            score += 0.10
        if evidence.get('has_delivery_photo'):
            score += 0.05
        if evidence.get('delivery_address_match'):
            score += 0.05

        # Customer history (20%)
        max_score += 0.20
        if evidence.get('is_repeat_customer'):
            score += 0.10
        if evidence.get('prior_chargebacks', 0) == 0:
            score += 0.05
        if evidence.get('has_verified_email'):
            score += 0.05

        # Device/IP (10%)
        max_score += 0.10
        if evidence.get('device_fingerprint_match'):
            score += 0.05
        if evidence.get('ip_location_match'):
            score += 0.05

        return score

    async def generate_evidence_package(
        self,
        transaction_id: str,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera pacote formatado de evidências para submission

        Args:
            transaction_id: Transaction ID
            evidence: Collected evidence

        Returns:
            Formatted evidence package
        """
        package = {
            'transaction_id': transaction_id,
            'generated_at': datetime.now(timezone.utc).isoformat(),

            # Section 1: Transaction Details
            'transaction_details': {
                'date': evidence.get('transaction_date', '').isoformat() if evidence.get('transaction_date') else '',
                'amount': evidence.get('amount'),
                'currency': evidence.get('currency'),
                'authorization_code': evidence.get('authorization_code'),
                'card_verification': {
                    'avs': evidence.get('avs_result'),
                    'cvv': evidence.get('cvv_result')
                }
            },

            # Section 2: Customer Information
            'customer_information': {
                'customer_id': evidence.get('customer_id'),
                'email': evidence.get('customer_email'),
                'account_age_days': evidence.get('account_age_days'),
                'transaction_history': {
                    'total_transactions': evidence.get('prior_transactions_count'),
                    'lifetime_value': evidence.get('total_lifetime_value'),
                    'prior_chargebacks': evidence.get('prior_chargebacks')
                }
            },

            # Section 3: Delivery Proof
            'delivery_proof': {
                'tracking_number': evidence.get('tracking_number'),
                'delivery_date': evidence.get('delivery_date', '').isoformat() if evidence.get('delivery_date') else '',
                'signature': evidence.get('has_signature'),
                'signature_name': evidence.get('signature_name'),
                'carrier': evidence.get('carrier'),
                'proof_url': evidence.get('carrier_proof_url')
            },

            # Section 4: Device/IP Evidence
            'device_evidence': {
                'device_fingerprint': evidence.get('device_fingerprint'),
                'ip_address': evidence.get('ip_address'),
                'ip_location': f"{evidence.get('ip_city')}, {evidence.get('ip_country')}",
                'location_match': evidence.get('ip_location_match'),
                'fraud_indicators': {
                    'is_proxy': evidence.get('is_proxy'),
                    'is_vpn': evidence.get('is_vpn')
                }
            },

            # Section 5: Communication Logs
            'communication_logs': {
                'order_confirmation_sent': evidence.get('has_order_confirmation_email'),
                'shipping_notification_sent': evidence.get('has_shipping_notification'),
                'customer_service_contacts': evidence.get('customer_service_contacts'),
                'complaints': evidence.get('has_complaints')
            },

            # Metadata
            'evidence_quality_score': evidence.get('evidence_quality_score'),
            'evidence_count': evidence.get('evidence_count')
        }

        return package
