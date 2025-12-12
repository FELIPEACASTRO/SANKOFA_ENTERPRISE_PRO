"""
PIX Fraud Typologies Engine
50+ fraud patterns específicos para PIX (Brasil)
Benchmark: Nubank (<200ms), Stripe Radar (150+ typologies)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class FraudTypology:
    """Definição de uma typology de fraude"""
    typology_id: str
    name: str
    description: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    check_function: callable
    remediation: str


class PIXFraudTypologies:
    """
    50+ Fraud Typologies específicas para PIX

    Categorias:
    1. Golpe do Motoboy
    2. Phishing/Engenharia Social
    3. Account Takeover (ATO)
    4. Mulas/Laranjas
    5. Fraude de Primeiro Depósito
    6. Velocity Abuse
    7. Round-tripping
    8. Merchant Fraud
    """

    def __init__(self):
        self.typologies = self._init_typologies()

    def _init_typologies(self) -> List[FraudTypology]:
        """Inicializa todas as typologies"""
        return [
            # ===== GOLPE DO MOTOBOY =====
            FraudTypology(
                typology_id="PIX-001",
                name="Golpe do Motoboy - PIX Inverso",
                description="Fraudador finge ser motoboy e pede PIX 'para confirmar identidade'",
                risk_level="CRITICAL",
                check_function=self._check_reverse_pix_scam,
                remediation="Bloquear PIX inverso de delivery apps + step-up MFA"
            ),

            # ===== PHISHING =====
            FraudTypology(
                typology_id="PIX-002",
                name="Phishing - Link Falso PIX",
                description="Usuário clica em link falso e autoriza PIX fraudulento",
                risk_level="HIGH",
                check_function=self._check_phishing_pix,
                remediation="Validar origem da transação + tempo desde login"
            ),

            FraudTypology(
                typology_id="PIX-003",
                name="Engenharia Social - Falso Suporte",
                description="Fraudador se passa por suporte bancário e solicita PIX teste",
                risk_level="HIGH",
                check_function=self._check_fake_support,
                remediation="Alertar usuário + step-up biométrico"
            ),

            # ===== ACCOUNT TAKEOVER =====
            FraudTypology(
                typology_id="PIX-004",
                name="ATO - Troca de Device Suspeita",
                description="PIX realizado após troca de device/IP em <24h",
                risk_level="CRITICAL",
                check_function=self._check_ato_device_change,
                remediation="Bloquear + contato cliente + reset senha"
            ),

            FraudTypology(
                typology_id="PIX-005",
                name="ATO - Horário Anormal",
                description="PIX em horário incompatível com padrão do usuário",
                risk_level="HIGH",
                check_function=self._check_ato_unusual_time,
                remediation="Step-up OTP + validação adicional"
            ),

            FraudTypology(
                typology_id="PIX-006",
                name="ATO - Localização Impossível",
                description="PIX de localização distante da última transação (velocidade impossível)",
                risk_level="CRITICAL",
                check_function=self._check_ato_impossible_travel,
                remediation="Bloquear imediatamente"
            ),

            # ===== MULAS/LARANJAS =====
            FraudTypology(
                typology_id="PIX-007",
                name="Mula - Múltiplos Recebimentos Rápidos",
                description="Conta recebe PIX de múltiplas origens e saca/transfere rapidamente",
                risk_level="HIGH",
                check_function=self._check_mule_rapid_turnover,
                remediation="Marcar como mula + investigação"
            ),

            FraudTypology(
                typology_id="PIX-008",
                name="Mula - Padrão de Distribuição",
                description="Conta recebe PIX grande e distribui em múltiplos PIX menores",
                risk_level="HIGH",
                check_function=self._check_mule_distribution,
                remediation="Freezar fundos + investigação AML"
            ),

            FraudTypology(
                typology_id="PIX-009",
                name="Laranja - Conta Nova com Alto Volume",
                description="Conta aberta recentemente com volume PIX incompatível",
                risk_level="MEDIUM",
                check_function=self._check_laranja_new_account,
                remediation="Limitar PIX + KYC adicional"
            ),

            # ===== PRIMEIRO DEPÓSITO =====
            FraudTypology(
                typology_id="PIX-010",
                name="First Deposit Fraud - PIX Imediato",
                description="PIX realizado imediatamente após primeiro depósito",
                risk_level="HIGH",
                check_function=self._check_first_deposit_fraud,
                remediation="Cooling period de 24h"
            ),

            # ===== VELOCITY ABUSE =====
            FraudTypology(
                typology_id="PIX-011",
                name="Velocity - Múltiplos PIX em Curto Intervalo",
                description=">5 PIX em <10 minutos (potencial ATO ou teste de limites)",
                risk_level="HIGH",
                check_function=self._check_velocity_high_frequency,
                remediation="Rate limit + step-up"
            ),

            FraudTypology(
                typology_id="PIX-012",
                name="Velocity - Esgotamento de Limite",
                description="Múltiplas transações até esgotar limite diário",
                risk_level="MEDIUM",
                check_function=self._check_velocity_limit_exhaustion,
                remediation="Alertar + monitorar próximas 48h"
            ),

            FraudTypology(
                typology_id="PIX-013",
                name="Velocity - Round Robin Merchants",
                description="PIX para múltiplos merchants em sequência rápida",
                risk_level="MEDIUM",
                check_function=self._check_velocity_round_robin,
                remediation="Verificar merchants + step-up"
            ),

            # ===== VALORES SUSPEITOS =====
            FraudTypology(
                typology_id="PIX-014",
                name="Valor - Centavos (Teste)",
                description="PIX de centavos (R$ 0.01-0.99) - potencial teste de conta válida",
                risk_level="LOW",
                check_function=self._check_amount_test,
                remediation="Monitorar para PIX subsequente grande"
            ),

            FraudTypology(
                typology_id="PIX-015",
                name="Valor - Limite Exato",
                description="PIX no valor exato do limite (R$ 1000.00) - possível fraude",
                risk_level="MEDIUM",
                check_function=self._check_amount_exact_limit,
                remediation="Step-up MFA"
            ),

            FraudTypology(
                typology_id="PIX-016",
                name="Valor - Acima da Média 10x",
                description="PIX >10x a média histórica do usuário",
                risk_level="HIGH",
                check_function=self._check_amount_above_average,
                remediation="Step-up + ligação cliente"
            ),

            # ===== HORÁRIO SUSPEITO =====
            FraudTypology(
                typology_id="PIX-017",
                name="Horário - Madrugada (2-5 AM)",
                description="PIX entre 2h-5h AM (horário de menor vigilância)",
                risk_level="MEDIUM",
                check_function=self._check_time_early_morning,
                remediation="Step-up obrigatório"
            ),

            FraudTypology(
                typology_id="PIX-018",
                name="Horário - Final de Semana Tarde da Noite",
                description="PIX após 22h em fim de semana",
                risk_level="LOW",
                check_function=self._check_time_weekend_late,
                remediation="Monitorar padrão"
            ),

            # ===== CHAVES PIX SUSPEITAS =====
            FraudTypology(
                typology_id="PIX-019",
                name="Chave PIX - CPF Diferente do Titular",
                description="PIX para CPF diferente do cadastro da conta destino",
                risk_level="HIGH",
                check_function=self._check_pix_key_cpf_mismatch,
                remediation="Validar relação + step-up"
            ),

            FraudTypology(
                typology_id="PIX-020",
                name="Chave PIX - Chave Aleatória Nova",
                description="PIX para chave aleatória criada há <24h",
                risk_level="MEDIUM",
                check_function=self._check_pix_key_new_random,
                remediation="Cooling period 24h"
            ),

            # ===== DEVICE/IP =====
            FraudTypology(
                typology_id="PIX-021",
                name="Device - Primeiro PIX em Device Novo",
                description="Primeiro PIX realizado em device nunca usado antes",
                risk_level="MEDIUM",
                check_function=self._check_device_first_pix,
                remediation="Step-up OTP"
            ),

            FraudTypology(
                typology_id="PIX-022",
                name="Device - Múltiplas Contas no Mesmo Device",
                description="Device usado por >3 contas diferentes em <30 dias",
                risk_level="HIGH",
                check_function=self._check_device_multiple_accounts,
                remediation="Bloquear device + investigação"
            ),

            FraudTypology(
                typology_id="PIX-023",
                name="IP - VPN/Proxy Detectado",
                description="PIX originado de VPN/proxy/Tor",
                risk_level="MEDIUM",
                check_function=self._check_ip_vpn_proxy,
                remediation="Step-up + validar cliente"
            ),

            FraudTypology(
                typology_id="PIX-024",
                name="IP - País Incompatível",
                description="IP de país diferente do Brasil",
                risk_level="HIGH",
                check_function=self._check_ip_foreign_country,
                remediation="Bloquear se não travel mode"
            ),

            # ===== MERCHANT FRAUD =====
            FraudTypology(
                typology_id="PIX-025",
                name="Merchant - Merchant Novo sem Histórico",
                description="PIX para merchant criado há <7 dias",
                risk_level="LOW",
                check_function=self._check_merchant_new,
                remediation="Monitorar chargebacks"
            ),

            FraudTypology(
                typology_id="PIX-026",
                name="Merchant - Alta Taxa de Chargeback",
                description="Merchant com >5% chargeback rate",
                risk_level="HIGH",
                check_function=self._check_merchant_high_chargeback,
                remediation="Bloquear merchant"
            ),

            # ===== PADRÕES COMPLEXOS =====
            FraudTypology(
                typology_id="PIX-027",
                name="Round-Tripping - PIX Circular",
                description="PIX A→B→C→A (potencial lavagem)",
                risk_level="HIGH",
                check_function=self._check_round_tripping,
                remediation="Reportar COAF"
            ),

            FraudTypology(
                typology_id="PIX-028",
                name="Smurfing - Múltiplos PIX Abaixo Limite Reporte",
                description="Múltiplos PIX <R$ 10.000 para evitar reporte",
                risk_level="MEDIUM",
                check_function=self._check_smurfing,
                remediation="Agregar volume + reportar se >R$ 50k"
            ),

            FraudTypology(
                typology_id="PIX-029",
                name="Structuring - PIX Fracionado Suspeito",
                description="PIX grande fracionado em múltiplos menores",
                risk_level="MEDIUM",
                check_function=self._check_structuring,
                remediation="Investigação AML"
            ),

            FraudTypology(
                typology_id="PIX-030",
                name="Behavioral Change - Mudança Abrupta de Padrão",
                description="Mudança súbita no padrão de uso PIX (volume, frequência, horário)",
                risk_level="MEDIUM",
                check_function=self._check_behavioral_change,
                remediation="Modelo de anomaly detection"
            ),

            # ===== ADVANCED PATTERNS (PIX-031 to PIX-050) =====
            FraudTypology(
                typology_id="PIX-031",
                name="Synthetic Identity Fraud",
                description="CPF real com dados falsos/inconsistentes",
                risk_level="CRITICAL",
                check_function=self._check_synthetic_identity,
                remediation="Bloquear + validação de documentos"
            ),

            FraudTypology(
                typology_id="PIX-032",
                name="Cross-Border Fraud Indicators",
                description="IP/VPN estrangeiro em PIX doméstico",
                risk_level="HIGH",
                check_function=self._check_cross_border_fraud,
                remediation="Step-up MFA + validação origem"
            ),

            FraudTypology(
                typology_id="PIX-033",
                name="Dormant Account Reactivation",
                description="Conta inativa >180 dias repentinamente ativa",
                risk_level="HIGH",
                check_function=self._check_dormant_account_reactivation,
                remediation="Contato com titular + step-up"
            ),

            FraudTypology(
                typology_id="PIX-034",
                name="Bust-Out Fraud Pattern",
                description="Histórico bom seguido de maxout fraudulento",
                risk_level="CRITICAL",
                check_function=self._check_bust_out_pattern,
                remediation="Bloquear + investigação imediata"
            ),

            FraudTypology(
                typology_id="PIX-035",
                name="Piggyback Fraud",
                description="PIX logo após transação legítima (device comprometido)",
                risk_level="HIGH",
                check_function=self._check_piggyback_fraud,
                remediation="Bloquear device + reset credenciais"
            ),

            FraudTypology(
                typology_id="PIX-036",
                name="Social Engineering Indicators",
                description="Padrões típicos de vítimas de engenharia social",
                risk_level="MEDIUM",
                check_function=self._check_social_engineering_indicators,
                remediation="Alertar cliente + step-up"
            ),

            FraudTypology(
                typology_id="PIX-037",
                name="Triangle Fraud",
                description="Lavagem via 3+ contas em círculo (A→B→C→A)",
                risk_level="CRITICAL",
                check_function=self._check_triangle_fraud,
                remediation="Bloquear rede + reportar COAF"
            ),

            FraudTypology(
                typology_id="PIX-038",
                name="Refund/Chargeback Fraud",
                description="Taxa excessiva de refunds/chargebacks",
                risk_level="HIGH",
                check_function=self._check_refund_fraud,
                remediation="Limitar refunds + investigação"
            ),

            FraudTypology(
                typology_id="PIX-039",
                name="Account Testing",
                description="Múltiplas transações pequenas (teste de conta ativa)",
                risk_level="MEDIUM",
                check_function=self._check_account_testing,
                remediation="Rate limiting + step-up"
            ),

            FraudTypology(
                typology_id="PIX-040",
                name="Authorized Push Payment (APP) Fraud",
                description="Vítima autoriza pagamento fraudulento (invoice/romance scam)",
                risk_level="HIGH",
                check_function=self._check_authorized_push_payment_fraud,
                remediation="Validação beneficiário + step-up"
            ),

            FraudTypology(
                typology_id="PIX-041",
                name="Money Mule Recruitment",
                description="Conta nova recebendo PIX de múltiplas fontes",
                risk_level="CRITICAL",
                check_function=self._check_money_mule_recruitment,
                remediation="Bloquear conta + investigação AML"
            ),

            FraudTypology(
                typology_id="PIX-042",
                name="Invoice Manipulation Fraud",
                description="QR code/invoice alterado para conta do fraudador",
                risk_level="MEDIUM",
                check_function=self._check_invoice_manipulation,
                remediation="Validar QR code + step-up"
            ),

            FraudTypology(
                typology_id="PIX-043",
                name="Collusion Fraud",
                description="Merchant e cliente em conluio",
                risk_level="HIGH",
                check_function=self._check_collusion_fraud,
                remediation="Bloquear merchant + investigação"
            ),

            FraudTypology(
                typology_id="PIX-044",
                name="SIM Swap Fraud",
                description="Troca de SIM card seguida de PIX",
                risk_level="CRITICAL",
                check_function=self._check_sim_swap_fraud,
                remediation="Bloquear + reset credenciais + contato"
            ),

            FraudTypology(
                typology_id="PIX-045",
                name="Credential Stuffing",
                description="Múltiplos logins falhados antes de sucesso",
                risk_level="CRITICAL",
                check_function=self._check_credential_stuffing,
                remediation="Bloquear IP + reset senha obrigatório"
            ),

            FraudTypology(
                typology_id="PIX-046",
                name="Bot/Automated Fraud",
                description="Padrões de automação (timing, user agent)",
                risk_level="HIGH",
                check_function=self._check_bot_automated_fraud,
                remediation="CAPTCHA + device fingerprinting"
            ),

            FraudTypology(
                typology_id="PIX-047",
                name="Merchant Category Code Mismatch",
                description="MCC não corresponde ao padrão esperado",
                risk_level="MEDIUM",
                check_function=self._check_merchant_category_mismatch,
                remediation="Validar merchant + step-up"
            ),

            FraudTypology(
                typology_id="PIX-048",
                name="Geofencing Violation",
                description="Transação fora da área permitida pelo cliente",
                risk_level="HIGH",
                check_function=self._check_geofencing_violation,
                remediation="Bloquear + contato cliente"
            ),

            FraudTypology(
                typology_id="PIX-049",
                name="High-Risk Beneficiary",
                description="Beneficiário em lista de alto risco/fraudes",
                risk_level="CRITICAL",
                check_function=self._check_high_risk_beneficiary,
                remediation="Bloquear + investigação"
            ),

            FraudTypology(
                typology_id="PIX-050",
                name="Rapid Account Changes",
                description="Múltiplas mudanças cadastrais em curto período",
                risk_level="HIGH",
                check_function=self._check_rapid_account_changes,
                remediation="Validação identidade + step-up"
            ),
        ]

    # =========================================================================
    # CHECK FUNCTIONS - Implementação das validações
    # =========================================================================

    def _check_reverse_pix_scam(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-001: Golpe do Motoboy"""
        # Verifica se é PIX para merchant de delivery + valor baixo
        merchant_id = txn.get('merchant_id', '')
        amount = float(txn.get('amount', 0))

        is_delivery = any(word in merchant_id.lower() for word in ['ifood', 'rappi', 'uber', 'delivery'])
        is_small_amount = 0.01 <= amount <= 10.0

        if is_delivery and is_small_amount:
            return True, 0.85, "Possível golpe do motoboy (PIX inverso pequeno para delivery)"

        return False, 0.0, ""

    def _check_phishing_pix(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-002: Phishing"""
        # Verifica tempo desde último login
        last_login = context.get('last_login_time')
        if last_login:
            minutes_since_login = (datetime.utcnow() - last_login).total_seconds() / 60
            if minutes_since_login < 5:  # PIX em <5min após login
                return True, 0.70, "PIX muito rápido após login (possível phishing)"

        return False, 0.0, ""

    def _check_fake_support(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-003: Falso Suporte"""
        # PIX pequeno após contato com "suporte"
        amount = float(txn.get('amount', 0))
        recent_support_contact = context.get('recent_support_contact', False)

        if recent_support_contact and amount < 1.0:
            return True, 0.80, "PIX teste após contato com suporte (possível golpe)"

        return False, 0.0, ""

    def _check_ato_device_change(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-004: ATO - Device Change"""
        device_change_hours = context.get('hours_since_device_change', 999)

        if device_change_hours < 24:
            return True, 0.95, f"PIX em device novo (trocado há {device_change_hours}h)"

        return False, 0.0, ""

    def _check_ato_unusual_time(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-005: Horário Anormal"""
        txn_hour = datetime.utcnow().hour
        user_avg_hour = context.get('avg_transaction_hour', 14)

        hour_diff = abs(txn_hour - user_avg_hour)
        if hour_diff > 8:  # >8h diferença do padrão
            return True, 0.65, f"PIX em horário anormal ({txn_hour}h vs média {user_avg_hour}h)"

        return False, 0.0, ""

    def _check_ato_impossible_travel(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-006: Impossible Travel"""
        last_location = context.get('last_transaction_location')
        current_location = txn.get('location')

        if last_location and current_location:
            distance_km = self._calculate_distance(last_location, current_location)
            time_diff_hours = context.get('hours_since_last_txn', 1)

            speed_kmh = distance_km / time_diff_hours if time_diff_hours > 0 else 0

            if speed_kmh > 800:  # Velocidade impossível (mais rápido que avião)
                return True, 0.98, f"Viagem impossível ({speed_kmh:.0f} km/h)"

        return False, 0.0, ""

    def _check_mule_rapid_turnover(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-007: Mula - Turnover Rápido"""
        received_last_24h = context.get('pix_received_count_24h', 0)
        sent_last_24h = context.get('pix_sent_count_24h', 0)

        if received_last_24h >= 5 and sent_last_24h >= 5:
            turnover_ratio = sent_last_24h / received_last_24h if received_last_24h > 0 else 0
            if 0.7 <= turnover_ratio <= 1.0:  # Recebe e envia quase tudo
                return True, 0.85, f"Padrão de mula (recebeu {received_last_24h}, enviou {sent_last_24h})"

        return False, 0.0, ""

    def _check_mule_distribution(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-008: Mula - Distribuição"""
        # Recebeu PIX grande e está enviando múltiplos menores
        amount = float(txn.get('amount', 0))
        last_received_amount = context.get('last_received_pix_amount', 0)
        pix_sent_count_1h = context.get('pix_sent_count_1h', 0)

        if last_received_amount > amount * 3 and pix_sent_count_1h >= 3:
            return True, 0.80, f"Distribuição suspeita (recebeu R$ {last_received_amount}, enviando múltiplos)"

        return False, 0.0, ""

    def _check_laranja_new_account(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-009: Laranja - Conta Nova"""
        account_age_days = context.get('account_age_days', 999)
        total_pix_volume_7d = context.get('total_pix_volume_7d', 0)

        if account_age_days < 30 and total_pix_volume_7d > 10000:
            return True, 0.70, f"Conta nova ({account_age_days}d) com alto volume (R$ {total_pix_volume_7d})"

        return False, 0.0, ""

    def _check_first_deposit_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-010: First Deposit Fraud"""
        minutes_since_first_deposit = context.get('minutes_since_first_deposit', 999)

        if minutes_since_first_deposit < 30:
            return True, 0.75, f"PIX {minutes_since_first_deposit}min após primeiro depósito"

        return False, 0.0, ""

    def _check_velocity_high_frequency(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-011: Alta Frequência"""
        pix_count_10min = context.get('pix_count_10min', 0)

        if pix_count_10min >= 5:
            return True, 0.80, f"{pix_count_10min} PIX em <10min (velocity abuse)"

        return False, 0.0, ""

    def _check_velocity_limit_exhaustion(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-012: Esgotamento de Limite"""
        daily_limit = context.get('daily_pix_limit', 1000)
        used_limit = context.get('daily_pix_used', 0)
        amount = float(txn.get('amount', 0))

        if (used_limit + amount) / daily_limit > 0.95:
            return True, 0.65, f"Esgotando limite diário ({((used_limit + amount) / daily_limit * 100):.1f}%)"

        return False, 0.0, ""

    def _check_velocity_round_robin(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-013: Round Robin Merchants"""
        unique_merchants_1h = context.get('unique_merchants_1h', 0)

        if unique_merchants_1h >= 5:
            return True, 0.60, f"{unique_merchants_1h} merchants diferentes em 1h"

        return False, 0.0, ""

    def _check_amount_test(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-014: Valor de Teste"""
        amount = float(txn.get('amount', 0))

        if 0.01 <= amount <= 0.99:
            return True, 0.50, f"PIX de centavos (R$ {amount:.2f}) - possível teste"

        return False, 0.0, ""

    def _check_amount_exact_limit(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-015: Valor Exato do Limite"""
        amount = float(txn.get('amount', 0))
        daily_limit = context.get('daily_pix_limit', 1000)

        if abs(amount - daily_limit) < 0.01:
            return True, 0.60, f"PIX no valor exato do limite (R$ {amount:.2f})"

        return False, 0.0, ""

    def _check_amount_above_average(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-016: Acima da Média"""
        amount = float(txn.get('amount', 0))
        avg_pix_amount = context.get('avg_pix_amount_30d', 100)

        if avg_pix_amount > 0 and amount > avg_pix_amount * 10:
            return True, 0.75, f"PIX {(amount / avg_pix_amount):.1f}x acima da média"

        return False, 0.0, ""

    def _check_time_early_morning(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-017: Madrugada"""
        hour = datetime.utcnow().hour

        if 2 <= hour <= 5:
            return True, 0.55, f"PIX em horário de madrugada ({hour}h)"

        return False, 0.0, ""

    def _check_time_weekend_late(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-018: Fim de Semana Tarde"""
        now = datetime.utcnow()
        is_weekend = now.weekday() in [5, 6]  # Sábado/Domingo
        is_late = now.hour >= 22

        if is_weekend and is_late:
            return True, 0.45, "PIX em fim de semana após 22h"

        return False, 0.0, ""

    def _check_pix_key_cpf_mismatch(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-019: CPF Diferente"""
        pix_key_cpf = txn.get('pix_key_cpf')
        account_cpf = context.get('account_cpf')

        if pix_key_cpf and account_cpf and pix_key_cpf != account_cpf:
            return True, 0.70, "Chave PIX com CPF diferente do titular"

        return False, 0.0, ""

    def _check_pix_key_new_random(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-020: Chave Aleatória Nova"""
        pix_key_type = txn.get('pix_key_type')
        pix_key_age_hours = context.get('pix_key_age_hours', 999)

        if pix_key_type == 'random' and pix_key_age_hours < 24:
            return True, 0.60, f"Chave aleatória criada há {pix_key_age_hours}h"

        return False, 0.0, ""

    def _check_device_first_pix(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-021: Primeiro PIX em Device Novo"""
        device_pix_count = context.get('device_pix_count', 0)

        if device_pix_count == 0:
            return True, 0.55, "Primeiro PIX neste device"

        return False, 0.0, ""

    def _check_device_multiple_accounts(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-022: Múltiplas Contas no Device"""
        device_account_count_30d = context.get('device_account_count_30d', 1)

        if device_account_count_30d >= 3:
            return True, 0.75, f"Device usado por {device_account_count_30d} contas em 30d"

        return False, 0.0, ""

    def _check_ip_vpn_proxy(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-023: VPN/Proxy"""
        is_vpn = context.get('is_vpn', False)
        is_proxy = context.get('is_proxy', False)

        if is_vpn or is_proxy:
            return True, 0.65, "IP de VPN/Proxy detectado"

        return False, 0.0, ""

    def _check_ip_foreign_country(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-024: IP Estrangeiro"""
        ip_country = context.get('ip_country', 'BR')

        if ip_country != 'BR':
            return True, 0.80, f"IP de país estrangeiro ({ip_country})"

        return False, 0.0, ""

    def _check_merchant_new(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-025: Merchant Novo"""
        merchant_age_days = context.get('merchant_age_days', 999)

        if merchant_age_days < 7:
            return True, 0.45, f"Merchant criado há {merchant_age_days}d"

        return False, 0.0, ""

    def _check_merchant_high_chargeback(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-026: Merchant com Alto Chargeback"""
        merchant_chargeback_rate = context.get('merchant_chargeback_rate', 0)

        if merchant_chargeback_rate > 0.05:  # >5%
            return True, 0.85, f"Merchant com {(merchant_chargeback_rate * 100):.1f}% chargeback"

        return False, 0.0, ""

    def _check_round_tripping(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-027: Round-Tripping"""
        circular_path_detected = context.get('circular_path_detected', False)

        if circular_path_detected:
            return True, 0.90, "PIX circular detectado (A→B→C→A)"

        return False, 0.0, ""

    def _check_smurfing(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-028: Smurfing"""
        pix_count_24h = context.get('pix_count_24h', 0)
        total_volume_24h = context.get('total_pix_volume_24h', 0)
        amount = float(txn.get('amount', 0))

        # Múltiplos PIX <R$ 10k, total >R$ 50k
        if pix_count_24h >= 6 and amount < 10000 and total_volume_24h > 50000:
            return True, 0.75, f"{pix_count_24h} PIX totalizando R$ {total_volume_24h} (smurfing)"

        return False, 0.0, ""

    def _check_structuring(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-029: Structuring"""
        # Similar ao smurfing, mas mais específico para fracionamento
        recent_similar_amounts = context.get('recent_similar_amounts_count', 0)

        if recent_similar_amounts >= 3:
            return True, 0.70, f"{recent_similar_amounts} PIX de valores similares (structuring)"

        return False, 0.0, ""

    def _check_behavioral_change(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-030: Mudança de Comportamento"""
        behavioral_score = context.get('behavioral_anomaly_score', 0)

        if behavioral_score > 0.8:
            return True, 0.65, f"Mudança abrupta de comportamento (score: {behavioral_score:.2f})"

        return False, 0.0, ""

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_distance(self, loc1: str, loc2: str) -> float:
        """Calcula distância entre localizações (simplificado)"""
        # Em produção, usar Haversine formula com lat/lon
        return 100.0  # Placeholder

    # =========================================================================
    # ADVANCED TYPOLOGIES (PIX-031 to PIX-050) - 20 PATTERNS
    # =========================================================================

    def _check_synthetic_identity(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-031: Synthetic Identity Fraud

        Detecta identidades sintéticas (CPF real + dados falsos)
        """
        # CPF existe mas outros dados inconsistentes
        cpf_age_days = context.get('cpf_age_days', 999)
        phone_age_days = context.get('phone_age_days', 999)
        email_age_days = context.get('email_age_days', 999)

        # CPF antigo mas outros dados muito novos = synthetic
        if cpf_age_days > 365 and phone_age_days < 30 and email_age_days < 30:
            return True, 0.90, f"Synthetic identity (CPF {cpf_age_days}d, phone {phone_age_days}d)"

        return False, 0.0, ""

    def _check_cross_border_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-032: Cross-Border Fraud Indicators

        PIX é doméstico, mas IP/VPN estrangeiro é suspeito
        """
        ip_country = context.get('ip_country', 'BR')
        is_vpn = context.get('is_vpn', False)
        is_proxy = context.get('is_proxy', False)

        if ip_country != 'BR' or is_vpn or is_proxy:
            return True, 0.85, f"Foreign IP/VPN (country: {ip_country}, VPN: {is_vpn})"

        return False, 0.0, ""

    def _check_dormant_account_reactivation(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-033: Dormant Account Sudden Reactivation

        Conta inativa por meses repentinamente ativa com PIX
        """
        days_since_last_txn = context.get('days_since_last_transaction', 0)
        txn_count_30d = context.get('transaction_count_30d', 0)

        # Conta inativa >180 dias com atividade súbita
        if days_since_last_txn > 180 and txn_count_30d >= 5:
            return True, 0.80, f"Dormant account reactivated ({days_since_last_txn}d inactive)"

        return False, 0.0, ""

    def _check_bust_out_pattern(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-034: Bust-Out Fraud Pattern

        Cliente estabelece histórico bom, depois maxout em fraude
        """
        account_age_days = context.get('account_age_days', 0)
        avg_amount_90d = context.get('avg_transaction_amount_90d', 0)
        current_amount = txn.get('amount', 0)

        # Conta relativamente nova com transação muito acima da média
        if 30 < account_age_days < 180 and avg_amount_90d > 0:
            if current_amount > avg_amount_90d * 10:
                return True, 0.88, f"Bust-out pattern (amount {current_amount/avg_amount_90d:.1f}x average)"

        return False, 0.0, ""

    def _check_piggyback_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-035: Piggyback Fraud (adiciona PIX após compra legítima)

        Fraudador adiciona PIX logo após transação legítima do titular
        """
        seconds_since_last_txn = context.get('seconds_since_last_transaction', 999)
        last_txn_type = context.get('last_transaction_type', '')

        # PIX logo após compra (possível device comprometido)
        if seconds_since_last_txn < 60 and last_txn_type in ['purchase', 'payment']:
            return True, 0.75, f"PIX {seconds_since_last_txn}s after legitimate purchase"

        return False, 0.0, ""

    def _check_social_engineering_indicators(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-036: Social Engineering Indicators

        Padrões típicos de vítimas de engenharia social
        """
        is_elderly = context.get('customer_age', 0) >= 65
        pix_key_type = txn.get('pix_key_type', '')
        beneficiary_name_similarity = context.get('beneficiary_name_similarity', 1.0)

        # Idoso enviando para chave aleatória com nome diferente
        if is_elderly and pix_key_type == 'RANDOM' and beneficiary_name_similarity < 0.3:
            return True, 0.70, "Elderly customer, random key, different beneficiary name"

        return False, 0.0, ""

    def _check_triangle_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-037: Triangle Fraud (3+ contas em círculo)

        A -> B -> C -> A (lavagem via múltiplas contas)
        """
        circular_transaction_detected = context.get('circular_transaction_detected', False)
        circle_size = context.get('circle_size', 0)

        if circular_transaction_detected and circle_size >= 3:
            return True, 0.92, f"Triangle fraud detected (circle size: {circle_size})"

        return False, 0.0, ""

    def _check_refund_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-038: Refund/Chargeback Fraud

        Padrão de solicitar refunds/chargebacks em excesso
        """
        chargeback_count_90d = context.get('chargeback_count_90d', 0)
        refund_count_90d = context.get('refund_count_90d', 0)

        if chargeback_count_90d >= 3 or refund_count_90d >= 5:
            return True, 0.75, f"High refund/chargeback rate ({chargeback_count_90d} CB, {refund_count_90d} refunds)"

        return False, 0.0, ""

    def _check_account_testing(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-039: Account Testing (multiple small txns)

        Fraudador testa se conta está ativa com múltiplas transações pequenas
        """
        txn_count_1h = context.get('transaction_count_1h', 0)
        avg_amount_1h = context.get('avg_transaction_amount_1h', 0)

        # Múltiplas transações pequenas em 1 hora
        if txn_count_1h >= 5 and avg_amount_1h < 10:
            return True, 0.68, f"Account testing ({txn_count_1h} txns, avg R${avg_amount_1h:.2f})"

        return False, 0.0, ""

    def _check_authorized_push_payment_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-040: Authorized Push Payment (APP) Fraud

        Vítima é enganada a autorizar pagamento (invoice fraud, romance scam)
        """
        is_new_beneficiary = context.get('is_new_beneficiary', False)
        amount = txn.get('amount', 0)
        beneficiary_risk_score = context.get('beneficiary_risk_score', 0)

        # Novo beneficiário + alto valor + beneficiário suspeito
        if is_new_beneficiary and amount > 5000 and beneficiary_risk_score > 0.7:
            return True, 0.82, f"APP fraud risk (new beneficiary, R${amount}, risk {beneficiary_risk_score})"

        return False, 0.0, ""

    def _check_money_mule_recruitment(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-041: Money Mule Recruitment Pattern

        Conta recém aberta recebendo múltiplos PIX de diferentes fontes
        """
        account_age_days = context.get('account_age_days', 999)
        unique_senders_7d = context.get('unique_pix_senders_7d', 0)
        pix_received_count_7d = context.get('pix_received_count_7d', 0)

        # Conta nova com muitos senders diferentes
        if account_age_days < 30 and unique_senders_7d >= 10 and pix_received_count_7d >= 15:
            return True, 0.90, f"Mule recruitment ({account_age_days}d old, {unique_senders_7d} senders)"

        return False, 0.0, ""

    def _check_invoice_manipulation(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-042: Invoice Manipulation Fraud

        Fraudador altera invoice/boleto para conta própria
        """
        # Verifica se QR code PIX foi modificado recentemente
        qr_code_age_seconds = context.get('qr_code_age_seconds', 999)
        is_qr_code_payment = txn.get('is_qr_code', False)

        if is_qr_code_payment and qr_code_age_seconds < 300:  # QR code muito recente
            return True, 0.65, f"QR code very recent ({qr_code_age_seconds}s), possible manipulation"

        return False, 0.0, ""

    def _check_collusion_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-043: Collusion Fraud (merchant + customer)

        Merchant e customer em conluio para fraude
        """
        merchant_customer_relationship = context.get('merchant_customer_relationship_score', 0)
        refund_rate = context.get('merchant_refund_rate', 0)

        # Alto relacionamento + alta taxa de refund = possível conluio
        if merchant_customer_relationship > 0.8 and refund_rate > 0.3:
            return True, 0.85, f"Collusion indicators (relationship {merchant_customer_relationship}, refund {refund_rate})"

        return False, 0.0, ""

    def _check_sim_swap_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-044: SIM Swap Fraud Detection

        Troca de chip + PIX = possível fraude
        """
        hours_since_sim_change = context.get('hours_since_sim_change', 999)
        is_2fa_sms = context.get('is_2fa_sms', False)

        if hours_since_sim_change < 48 and is_2fa_sms:
            return True, 0.95, f"SIM swap {hours_since_sim_change}h ago + SMS 2FA"

        return False, 0.0, ""

    def _check_credential_stuffing(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-045: Credential Stuffing Detection

        Múltiplas tentativas de login antes de PIX
        """
        failed_login_attempts_1h = context.get('failed_login_attempts_1h', 0)
        successful_login_from_new_ip = context.get('successful_login_from_new_ip', False)

        if failed_login_attempts_1h >= 5 and successful_login_from_new_ip:
            return True, 0.88, f"{failed_login_attempts_1h} failed logins before success from new IP"

        return False, 0.0, ""

    def _check_bot_automated_fraud(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-046: Bot/Automated Fraud Detection

        Padrões de automação (timing perfeito, user agent suspeito)
        """
        is_bot_user_agent = context.get('is_bot_user_agent', False)
        transaction_timing_variance = context.get('transaction_timing_variance', 1.0)

        # User agent de bot + timing muito regular
        if is_bot_user_agent or transaction_timing_variance < 0.1:
            return True, 0.80, "Bot/automated behavior detected"

        return False, 0.0, ""

    def _check_merchant_category_mismatch(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-047: Merchant Category Code Mismatch

        Merchant MCC não bate com padrão de transações
        """
        merchant_mcc = txn.get('merchant_mcc', '')
        expected_mcc = context.get('expected_mcc_for_merchant', '')

        if merchant_mcc and expected_mcc and merchant_mcc != expected_mcc:
            return True, 0.60, f"MCC mismatch (got {merchant_mcc}, expected {expected_mcc})"

        return False, 0.0, ""

    def _check_geofencing_violation(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-048: Geofencing Violation

        Transação fora da área permitida pelo cliente
        """
        is_geofence_enabled = context.get('customer_geofence_enabled', False)
        is_within_geofence = context.get('is_within_geofence', True)

        if is_geofence_enabled and not is_within_geofence:
            current_location = context.get('current_location', 'unknown')
            return True, 0.75, f"Transaction outside geofence (location: {current_location})"

        return False, 0.0, ""

    def _check_high_risk_beneficiary(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-049: High-Risk Beneficiary

        Beneficiário está em lista de alto risco (CPF irregular, lavagem)
        """
        beneficiary_cpf = txn.get('beneficiary_cpf', '')
        beneficiary_risk_level = context.get('beneficiary_risk_level', 'LOW')
        beneficiary_fraud_reports = context.get('beneficiary_fraud_reports_count', 0)

        if beneficiary_risk_level in ['HIGH', 'CRITICAL'] or beneficiary_fraud_reports >= 3:
            return True, 0.90, f"High-risk beneficiary (level: {beneficiary_risk_level}, {beneficiary_fraud_reports} reports)"

        return False, 0.0, ""

    def _check_rapid_account_changes(self, txn: Dict, context: Dict) -> Tuple[bool, float, str]:
        """PIX-050: Rapid Account Changes

        Múltiplas mudanças de dados cadastrais em curto período
        """
        profile_changes_7d = context.get('profile_changes_count_7d', 0)

        # Mudanças excessivas de perfil
        if profile_changes_7d >= 4:
            return True, 0.78, f"Rapid account changes ({profile_changes_7d} changes in 7d)"

        return False, 0.0, ""

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def analyze_transaction(
        self,
        transaction: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa transação PIX contra todas as typologies

        Returns:
            matched_typologies: lista de typologies detectadas
            max_risk_score: score de risco máximo
            risk_level: LOW/MEDIUM/HIGH/CRITICAL
            recommendations: ações recomendadas
        """
        matched_typologies = []
        max_risk_score = 0.0

        for typology in self.typologies:
            try:
                matched, risk_score, reason = typology.check_function(transaction, context)

                if matched:
                    matched_typologies.append({
                        'typology_id': typology.typology_id,
                        'name': typology.name,
                        'risk_score': risk_score,
                        'risk_level': typology.risk_level,
                        'reason': reason,
                        'remediation': typology.remediation
                    })

                    max_risk_score = max(max_risk_score, risk_score)

            except Exception as e:
                logger.error(f"Error checking typology {typology.typology_id}: {e}")

        # Determine overall risk level
        if max_risk_score >= 0.8:
            risk_level = "CRITICAL"
        elif max_risk_score >= 0.6:
            risk_level = "HIGH"
        elif max_risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Aggregate recommendations
        recommendations = list(set(t['remediation'] for t in matched_typologies))

        return {
            'matched_typologies': matched_typologies,
            'num_matches': len(matched_typologies),
            'max_risk_score': max_risk_score,
            'risk_level': risk_level,
            'recommendations': recommendations
        }


# Singleton
_pix_typologies_engine = None


def get_pix_typologies_engine() -> PIXFraudTypologies:
    """Get singleton PIX typologies engine"""
    global _pix_typologies_engine
    if _pix_typologies_engine is None:
        _pix_typologies_engine = PIXFraudTypologies()
    return _pix_typologies_engine
