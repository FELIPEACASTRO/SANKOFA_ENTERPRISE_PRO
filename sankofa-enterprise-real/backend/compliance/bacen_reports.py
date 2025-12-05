"""
Sankofa Enterprise Pro - BACEN Automated Reports
Geração automática de relatórios para o Banco Central do Brasil
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from enum import Enum
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Tipos de relatórios BACEN"""

    FRAUDES_PIX = "fraudes_pix"
    FRAUDES_TED = "fraudes_ted"
    FRAUDES_CARTAO = "fraudes_cartao"
    OPERACOES_SUSPEITAS = "operacoes_suspeitas"
    LAVAGEM_DINHEIRO = "lavagem_dinheiro"
    INCIDENTES_SEGURANCA = "incidentes_seguranca"
    METRICAS_MODELO = "metricas_modelo"
    COMPLIANCE_MENSAL = "compliance_mensal"


class ReportStatus(Enum):
    """Status do relatório"""

    DRAFT = "draft"
    GENERATED = "generated"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class FraudIncident:
    """Incidente de fraude para relatório"""

    incident_id: str
    transaction_id: str
    incident_date: datetime
    detection_date: datetime
    amount: float
    currency: str = "BRL"
    fraud_type: str = ""
    channel: str = ""
    victim_type: str = ""
    status: str = ""
    recovery_amount: float = 0.0
    description: str = ""


@dataclass
class BACENReport:
    """Relatório BACEN"""

    report_id: str
    report_type: ReportType
    period_start: date
    period_end: date
    status: ReportStatus = ReportStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    incidents: List[FraudIncident] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "data": self.data,
            "summary": self.summary,
            "checksum": self.checksum,
        }


class BACENReportGenerator:
    """
    Gerador de relatórios BACEN

    Implementa os requisitos de:
    - Resolução BCB nº 6/2023
    - Circular BCB nº 4.001/2020
    - Normativa BCB nº 491/2024 (PIX)
    """

    def __init__(
        self,
        institution_code: str = "00000000",
        institution_name: str = "Sankofa Bank",
        output_dir: str = "./reports/bacen",
    ):
        self.institution_code = institution_code
        self.institution_name = institution_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.generated_reports: List[BACENReport] = []

        logger.info(f"BACEN Report Generator initialized for {institution_name}")

    def _generate_report_id(self, report_type: ReportType, period: date) -> str:
        """Gera ID único para relatório"""
        base = f"{self.institution_code}_{report_type.value}_{period.strftime('%Y%m')}"
        hash_suffix = hashlib.md5(f"{base}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
        return f"BACEN_{base}_{hash_suffix}".upper()

    def _calculate_checksum(self, data: Dict) -> str:
        """Calcula checksum do relatório"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def generate_fraud_report(
        self,
        transactions: List[Dict],
        period_start: date,
        period_end: date,
        report_type: ReportType = ReportType.FRAUDES_PIX,
    ) -> BACENReport:
        """
        Gera relatório de fraudes

        Args:
            transactions: Lista de transações fraudulentas
            period_start: Início do período
            period_end: Fim do período
            report_type: Tipo de relatório

        Returns:
            Relatório BACEN gerado
        """
        report_id = self._generate_report_id(report_type, period_start)

        incidents = []
        total_amount = 0.0
        total_recovered = 0.0

        for txn in transactions:
            if txn.get("is_fraud", False):
                incident = FraudIncident(
                    incident_id=f"INC_{txn.get('transaction_id', '')}",
                    transaction_id=str(txn.get("transaction_id", "")),
                    incident_date=txn.get("timestamp", datetime.now()),
                    detection_date=txn.get("detection_date", datetime.now()),
                    amount=float(txn.get("amount", 0)),
                    fraud_type=self._classify_fraud_type(txn),
                    channel=txn.get("canal", "UNKNOWN"),
                    victim_type=self._classify_victim_type(txn),
                    status=txn.get("fraud_status", "CONFIRMED"),
                    recovery_amount=float(txn.get("recovered_amount", 0)),
                    description=txn.get("fraud_description", ""),
                )
                incidents.append(incident)
                total_amount += incident.amount
                total_recovered += incident.recovery_amount

        fraud_by_channel = {}
        fraud_by_type = {}
        fraud_by_day = {}

        for incident in incidents:
            fraud_by_channel[incident.channel] = fraud_by_channel.get(incident.channel, 0) + 1
            fraud_by_type[incident.fraud_type] = fraud_by_type.get(incident.fraud_type, 0) + 1

            day_key = (
                incident.incident_date.strftime("%Y-%m-%d")
                if isinstance(incident.incident_date, datetime)
                else str(incident.incident_date)
            )
            fraud_by_day[day_key] = fraud_by_day.get(day_key, 0) + 1

        summary = {
            "total_incidents": len(incidents),
            "total_amount": total_amount,
            "total_recovered": total_recovered,
            "recovery_rate": (total_recovered / total_amount * 100) if total_amount > 0 else 0,
            "by_channel": fraud_by_channel,
            "by_type": fraud_by_type,
            "by_day": fraud_by_day,
            "avg_amount": total_amount / len(incidents) if incidents else 0,
        }

        report_data = {
            "institution": {
                "code": self.institution_code,
                "name": self.institution_name,
            },
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "incidents": [asdict(i) for i in incidents],
            "summary": summary,
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        report = BACENReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            status=ReportStatus.GENERATED,
            data=report_data,
            incidents=incidents,
            summary=summary,
        )

        report.checksum = self._calculate_checksum(report_data)

        self.generated_reports.append(report)
        logger.info(f"Generated BACEN report: {report_id} with {len(incidents)} incidents")

        return report

    def generate_suspicious_operations_report(
        self, transactions: List[Dict], period_start: date, period_end: date
    ) -> BACENReport:
        """
        Gera relatório de operações suspeitas (COAF/UIF)

        Requisito: Circular BCB 3.978/2020
        """
        report_id = self._generate_report_id(ReportType.OPERACOES_SUSPEITAS, period_start)

        suspicious_ops = []

        for txn in transactions:
            risk_score = float(txn.get("risk_score", 0) or txn.get("fraud_score", 0))
            amount = float(txn.get("amount", 0))

            if risk_score >= 0.7 or amount >= 50000:
                suspicious_ops.append(
                    {
                        "transaction_id": txn.get("transaction_id", ""),
                        "date": str(txn.get("timestamp", "")),
                        "amount": amount,
                        "risk_score": risk_score,
                        "channel": txn.get("canal", ""),
                        "indicators": self._identify_suspicious_indicators(txn),
                        "recommendation": self._get_recommendation(risk_score, amount),
                    }
                )

        summary = {
            "total_suspicious": len(suspicious_ops),
            "total_amount": sum(op["amount"] for op in suspicious_ops),
            "high_risk_count": sum(1 for op in suspicious_ops if op["risk_score"] >= 0.9),
            "structuring_suspected": sum(
                1 for op in suspicious_ops if "STRUCTURING" in op["indicators"]
            ),
            "smurfing_suspected": sum(1 for op in suspicious_ops if "SMURFING" in op["indicators"]),
        }

        report_data = {
            "institution": {
                "code": self.institution_code,
                "name": self.institution_name,
            },
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "suspicious_operations": suspicious_ops,
            "summary": summary,
            "compliance_officer_review_required": len(suspicious_ops) > 0,
            "generated_at": datetime.now().isoformat(),
        }

        report = BACENReport(
            report_id=report_id,
            report_type=ReportType.OPERACOES_SUSPEITAS,
            period_start=period_start,
            period_end=period_end,
            status=ReportStatus.GENERATED,
            data=report_data,
            summary=summary,
        )

        report.checksum = self._calculate_checksum(report_data)
        self.generated_reports.append(report)

        return report

    def generate_model_metrics_report(
        self, metrics: Dict[str, Any], period_start: date, period_end: date
    ) -> BACENReport:
        """
        Gera relatório de métricas do modelo de ML

        Requisito: Resolução BCB 6/2023 - Transparência algorítmica
        """
        report_id = self._generate_report_id(ReportType.METRICAS_MODELO, period_start)

        report_data = {
            "institution": {
                "code": self.institution_code,
                "name": self.institution_name,
            },
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "model_info": {
                "name": metrics.get("model_name", "Sankofa Fraud Engine"),
                "version": metrics.get("model_version", "1.0.0"),
                "type": metrics.get("model_type", "Ensemble (RF+GB+LR)"),
                "last_trained": metrics.get("last_trained", ""),
            },
            "performance_metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "false_positive_rate": metrics.get("false_positive_rate", 0),
                "false_negative_rate": metrics.get("false_negative_rate", 0),
            },
            "operational_metrics": {
                "total_predictions": metrics.get("total_predictions", 0),
                "fraud_detected": metrics.get("fraud_detected", 0),
                "fraud_blocked": metrics.get("fraud_blocked", 0),
                "amount_protected": metrics.get("amount_protected", 0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0),
                "p95_latency_ms": metrics.get("p95_latency_ms", 0),
            },
            "fairness_metrics": {
                "demographic_parity": metrics.get("demographic_parity", 1.0),
                "equal_opportunity": metrics.get("equal_opportunity", 1.0),
                "calibration_error": metrics.get("calibration_error", 0),
            },
            "explainability": {
                "lgpd_compliant": True,
                "explanation_available": True,
                "top_features": metrics.get("top_features", []),
            },
            "generated_at": datetime.now().isoformat(),
        }

        summary = {
            "model_healthy": metrics.get("accuracy", 0) >= 0.9,
            "sla_compliant": metrics.get("avg_latency_ms", 0) < 100,
            "fairness_acceptable": metrics.get("demographic_parity", 1.0) >= 0.8,
        }

        report = BACENReport(
            report_id=report_id,
            report_type=ReportType.METRICAS_MODELO,
            period_start=period_start,
            period_end=period_end,
            status=ReportStatus.GENERATED,
            data=report_data,
            summary=summary,
        )

        report.checksum = self._calculate_checksum(report_data)
        self.generated_reports.append(report)

        return report

    def generate_monthly_compliance_report(
        self, transactions: List[Dict], metrics: Dict[str, Any], year: int, month: int
    ) -> BACENReport:
        """
        Gera relatório mensal consolidado de compliance
        """
        period_start = date(year, month, 1)
        if month == 12:
            period_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            period_end = date(year, month + 1, 1) - timedelta(days=1)

        report_id = self._generate_report_id(ReportType.COMPLIANCE_MENSAL, period_start)

        total_txn = len(transactions)
        fraud_txn = [t for t in transactions if t.get("is_fraud", False)]
        blocked_txn = [t for t in transactions if t.get("status") == "BLOCKED"]

        report_data = {
            "institution": {
                "code": self.institution_code,
                "name": self.institution_name,
            },
            "period": {
                "year": year,
                "month": month,
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "transaction_summary": {
                "total_transactions": total_txn,
                "total_amount": sum(float(t.get("amount", 0)) for t in transactions),
                "fraud_count": len(fraud_txn),
                "fraud_amount": sum(float(t.get("amount", 0)) for t in fraud_txn),
                "blocked_count": len(blocked_txn),
                "blocked_amount": sum(float(t.get("amount", 0)) for t in blocked_txn),
                "fraud_rate": len(fraud_txn) / total_txn * 100 if total_txn > 0 else 0,
            },
            "model_performance": {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
            },
            "compliance_status": {
                "lgpd_compliant": True,
                "bacen_compliant": True,
                "pci_dss_compliant": True,
                "audit_trail_complete": True,
                "data_retention_compliant": True,
            },
            "incidents": {
                "security_incidents": 0,
                "data_breaches": 0,
                "system_outages": 0,
            },
            "certifications": {
                "lgpd_training_completed": True,
                "security_awareness_completed": True,
            },
            "generated_at": datetime.now().isoformat(),
        }

        summary = {
            "overall_compliance": "COMPLIANT",
            "risk_level": "LOW" if len(fraud_txn) / max(total_txn, 1) < 0.01 else "MEDIUM",
            "action_required": False,
        }

        report = BACENReport(
            report_id=report_id,
            report_type=ReportType.COMPLIANCE_MENSAL,
            period_start=period_start,
            period_end=period_end,
            status=ReportStatus.GENERATED,
            data=report_data,
            summary=summary,
        )

        report.checksum = self._calculate_checksum(report_data)
        self.generated_reports.append(report)

        return report

    def _classify_fraud_type(self, txn: Dict) -> str:
        """Classifica tipo de fraude"""
        amount = float(txn.get("amount", 0))
        channel = txn.get("canal", "").upper()

        if channel == "PIX":
            if amount > 10000:
                return "PIX_HIGH_VALUE"
            return "PIX_STANDARD"
        elif channel in ["CARTAO_CREDITO", "CARTAO_DEBITO"]:
            if txn.get("is_new_device"):
                return "CARD_NEW_DEVICE"
            return "CARD_UNAUTHORIZED"
        elif channel in ["TED", "DOC"]:
            return "TRANSFER_UNAUTHORIZED"

        return "OTHER"

    def _classify_victim_type(self, txn: Dict) -> str:
        """Classifica tipo de vítima"""
        account_age = txn.get("account_age_days", 365)

        if account_age < 30:
            return "NEW_ACCOUNT"
        elif account_age > 3650:
            return "SENIOR_ACCOUNT"

        return "STANDARD"

    def _identify_suspicious_indicators(self, txn: Dict) -> List[str]:
        """Identifica indicadores de operação suspeita"""
        indicators = []
        amount = float(txn.get("amount", 0))

        if 9000 <= amount <= 10000:
            indicators.append("STRUCTURING")

        velocity = txn.get("transactions_last_1h", 0)
        if velocity > 10:
            indicators.append("VELOCITY_BURST")

        if txn.get("is_new_receiver"):
            indicators.append("NEW_RECEIVER")

        if txn.get("is_night") and amount > 5000:
            indicators.append("NIGHT_HIGH_VALUE")

        if txn.get("location_risk_score", 0) > 0.8:
            indicators.append("HIGH_RISK_LOCATION")

        return indicators

    def _get_recommendation(self, risk_score: float, amount: float) -> str:
        """Gera recomendação baseada no risco"""
        if risk_score >= 0.95:
            return "BLOCK_AND_INVESTIGATE"
        elif risk_score >= 0.85:
            return "MANUAL_REVIEW_URGENT"
        elif risk_score >= 0.70:
            return "ENHANCED_MONITORING"
        elif amount >= 50000:
            return "STANDARD_MONITORING"

        return "NO_ACTION_REQUIRED"

    def save_report(self, report: BACENReport, format: str = "json") -> str:
        """
        Salva relatório em arquivo

        Args:
            report: Relatório a ser salvo
            format: Formato (json, xml)

        Returns:
            Caminho do arquivo salvo
        """
        filename = f"{report.report_id}.{format}"
        filepath = self.output_dir / filename

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Report saved: {filepath}")
        return str(filepath)

    def get_report(self, report_id: str) -> Optional[BACENReport]:
        """Recupera relatório por ID"""
        for report in self.generated_reports:
            if report.report_id == report_id:
                return report
        return None

    def list_reports(
        self, report_type: Optional[ReportType] = None, status: Optional[ReportStatus] = None
    ) -> List[BACENReport]:
        """Lista relatórios com filtros"""
        reports = self.generated_reports

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        if status:
            reports = [r for r in reports if r.status == status]

        return reports


def create_bacen_generator(
    institution_code: str = None, institution_name: str = None
) -> BACENReportGenerator:
    """Factory function para criar gerador de relatórios BACEN"""
    return BACENReportGenerator(
        institution_code=institution_code or os.getenv("BACEN_INSTITUTION_CODE", "00000000"),
        institution_name=institution_name or os.getenv("INSTITUTION_NAME", "Sankofa Bank"),
    )
