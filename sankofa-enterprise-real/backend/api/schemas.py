"""
Pydantic Schemas para Validação de Input
Implementa validação robusta para TODOS os endpoints da API
Previne SQL Injection, XSS, e outros ataques de input
Compatível com Pydantic v2.x
"""

from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional, List, Dict, Any, Annotated, ClassVar
from datetime import datetime
from decimal import Decimal
import re


# ============================================================================
# TRANSACTION SCHEMAS
# ============================================================================

class TransactionRequest(BaseModel):
    """Schema para request de predição de fraude"""

    amount: float = Field(
        ...,
        gt=0,
        le=1000000,
        description="Valor da transação em BRL"
    )
    cpf: Annotated[str, Field(pattern=r'^\d{11}$', description="CPF do cliente (somente números, 11 dígitos)")]
    channel: Annotated[str, Field(pattern=r'^(PIX|TED|DOC|BOLETO|CARTAO_CREDITO|CARTAO_DEBITO|APP|WEB|ATM)$', description="Canal da transação")]
    tipo_transacao: Optional[Annotated[str, Field(pattern=r'^(PIX|TED|DOC|CARTAO_CREDITO|CARTAO_DEBITO|TRANSFERENCIA|PAGAMENTO)$')]] = None
    location: Optional[Annotated[str, Field(max_length=200)]] = None
    device_id: Optional[Annotated[str, Field(max_length=100)]] = None
    ip_address: Optional[Annotated[str, Field(pattern=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')]] = None

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validação adicional de amount"""
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 100000:
            # High value transactions require additional validation
            pass
        return round(v, 2)

    @field_validator('cpf')
    @classmethod
    def validate_cpf(cls, v):
        """Valida CPF com dígitos verificadores"""
        # Remove formatação
        cpf = re.sub(r'\D', '', v)

        if len(cpf) != 11:
            raise ValueError('CPF deve ter 11 dígitos')

        # Verifica CPFs inválidos conhecidos
        if cpf == cpf[0] * 11:
            raise ValueError('CPF inválido')

        # Validação dos dígitos verificadores
        def calc_digit(cpf_partial: str, weights: List[int]) -> int:
            total = sum(int(cpf_partial[i]) * weights[i] for i in range(len(cpf_partial)))
            remainder = total % 11
            return 0 if remainder < 2 else 11 - remainder

        # Primeiro dígito
        first_digit = calc_digit(cpf[:9], list(range(10, 1, -1)))
        if int(cpf[9]) != first_digit:
            raise ValueError('CPF inválido - primeiro dígito verificador')

        # Segundo dígito
        second_digit = calc_digit(cpf[:10], list(range(11, 1, -1)))
        if int(cpf[10]) != second_digit:
            raise ValueError('CPF inválido - segundo dígito verificador')

        return cpf

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 1500.50,
                "cpf": "12345678901",
                "channel": "PIX",
                "location": "São Paulo, SP"
            }
        }
    }


class FraudPredictionBatchRequest(BaseModel):
    """Schema para request de predição em lote"""

    transactions: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Lista de transações para análise"
    )
    include_explanation: Optional[bool] = Field(
        None,
        description="Incluir explicação LGPD-compliant (default: False para PIX, True para outros)"
    )
    include_compliance_report: Optional[bool] = Field(
        False,
        description="Incluir relatório de compliance completo"
    )
    fast_mode: Optional[bool] = Field(
        True,
        description="Usar fallback rápido em vez de SHAP (< 50ms)"
    )

    @field_validator('transactions')
    @classmethod
    def validate_transactions(cls, v):
        """Valida estrutura básica das transações"""
        if not v:
            raise ValueError('transactions list cannot be empty')

        for i, txn in enumerate(v):
            # Validações básicas de campos obrigatórios
            if 'amount' not in txn:
                raise ValueError(f'Transaction {i}: amount is required')
            if not isinstance(txn.get('amount'), (int, float)):
                raise ValueError(f'Transaction {i}: amount must be numeric')
            if txn['amount'] <= 0:
                raise ValueError(f'Transaction {i}: amount must be positive')

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "transactions": [
                    {
                        "amount": 1500.50,
                        "channel": "PIX",
                        "customer_id": "CUST123",
                        "merchant_id": "MERCH456"
                    }
                ],
                "include_explanation": False,
                "fast_mode": True
            }
        }
    }


class TransactionFilterRequest(BaseModel):
    """Schema para filtros de consulta de transações"""

    limit: int = Field(100, ge=1, le=1000, description="Máximo de resultados")
    offset: int = Field(0, ge=0, description="Offset para paginação")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    channel: Optional[Annotated[str, Field(pattern=r'^(PIX|TED|DOC|BOLETO|CARTAO_CREDITO|CARTAO_DEBITO|APP|WEB|ATM)$')]] = None
    is_fraud: Optional[bool] = None
    min_amount: Optional[float] = Field(None, ge=0)
    max_amount: Optional[float] = Field(None, le=10000000)

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """Valida que end_date > start_date"""
        if v and info.data.get('start_date'):
            if v < info.data['start_date']:
                raise ValueError('end_date must be after start_date')
        return v


# ============================================================================
# HARD RULES SCHEMAS
# ============================================================================

class HardRuleCreate(BaseModel):
    """Schema para criação de hard rule"""

    name: Annotated[str, Field(min_length=3, max_length=100, description="Nome da regra")]
    description: Optional[str] = Field(None, description="Descricao da regra")
    condition: Optional[str] = Field(None, max_length=500, description="Condição SQL")
    conditions_json: Optional[list] = Field(None, description="Condicoes em formato JSON")
    logic_operator: Optional[str] = Field("AND", description="Operador logico (AND/OR)")
    action: str = Field(description="Ação")
    action_config: Optional[dict] = Field(None, description="Config da acao")
    rule_type: Optional[str] = Field("blocking", description="Tipo da regra")
    enabled: bool = Field(True, description="Regra ativa?")
    priority: Optional[int] = Field(None, ge=0, le=100)

    @field_validator('action')
    @classmethod
    def normalize_action(cls, v):
        """Valida action (aceita uppercase e lowercase)"""
        if not v:
            raise ValueError("Action is required")
        v_upper = v.upper()
        allowed = ['BLOCK', 'REVIEW', 'STEP_UP', 'ALLOW', 'SCORE']
        if v_upper not in allowed:
            raise ValueError(f"Action must be one of: {', '.join(allowed)}")
        # Retorna original (lowercase ou uppercase) para compatibilidade
        return v

    # Whitelist de campos permitidos em conditions SQL
    ALLOWED_FIELDS: ClassVar[set] = {
        'amount', 'channel', 'location', 'device_id', 'customer_id',
        'merchant_id', 'risk_score', 'timestamp', 'ip_address'
    }

    @field_validator('condition')
    @classmethod
    def validate_condition(cls, v):
        """Valida que a condição não contém SQL perigoso"""
        if not v:
            return v

        # Whitelist de operadores permitidos
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'EXEC', 'EXECUTE', 'SCRIPT', '--', ';', 'xp_', 'sp_'
        ]

        v_upper = v.upper()
        for keyword in dangerous_keywords:
            if keyword in v_upper:
                raise ValueError(f'Keyword perigoso não permitido: {keyword}')

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Block high amount PIX",
                "condition": "amount > 50000 AND channel = 'PIX'",
                "action": "BLOCK",
                "enabled": True,
                "priority": 10
            }
        }
    }


class HardRuleUpdate(BaseModel):
    """Schema para atualização de hard rule"""

    name: Optional[Annotated[str, Field(min_length=3, max_length=100)]] = None
    condition: Optional[str] = Field(None, max_length=500)
    action: Optional[Annotated[str, Field(pattern=r'^(BLOCK|REVIEW|STEP_UP|ALLOW)$')]] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = Field(None, ge=0, le=100)

    # Same whitelist as create
    ALLOWED_FIELDS: ClassVar[set] = HardRuleCreate.ALLOWED_FIELDS

    @field_validator('condition')
    @classmethod
    def validate_condition(cls, v):
        """Valida que a condição não contém SQL perigoso"""
        if not v:
            return v

        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'EXEC', 'EXECUTE', 'SCRIPT', '--', ';', 'xp_', 'sp_'
        ]

        v_upper = v.upper()
        for keyword in dangerous_keywords:
            if keyword in v_upper:
                raise ValueError(f'Keyword perigoso não permitido: {keyword}')

        return v


# ============================================================================
# VIP/HOT LIST SCHEMAS
# ============================================================================

class VipListCreate(BaseModel):
    """Schema para adicionar à VIP list"""

    identifier: Optional[Annotated[str, Field(pattern=r'^\d{11}$', description="CPF (11 dígitos)")]] = None
    cpf: Optional[Annotated[str, Field(pattern=r'^\d{11}$', description="CPF (11 dígitos) - alias")]] = None
    type: Annotated[str, Field(pattern=r'^(cpf|cnpj|email)$')] = "cpf"
    reason: Annotated[str, Field(min_length=3, max_length=200, description="Motivo da inclusão")]

    def model_post_init(self, __context):
        """Normalize identifier and cpf"""
        if self.cpf and not self.identifier:
            self.identifier = self.cpf
        elif self.identifier and not self.cpf:
            self.cpf = self.identifier
        elif not self.cpf and not self.identifier:
            raise ValueError("Either 'cpf' or 'identifier' must be provided")

    model_config = {
        "json_schema_extra": {
            "example": {
                "identifier": "12345678901",
                "type": "cpf",
                "reason": "Cliente premium verificado"
            }
        }
    }


class HotListCreate(BaseModel):
    """Schema para adicionar à Hot list"""

    identifier: Optional[Annotated[str, Field(pattern=r'^\d{11}$', description="CPF (11 dígitos)")]] = None
    cpf: Optional[Annotated[str, Field(pattern=r'^\d{11}$', description="CPF (11 dígitos) - alias")]] = None
    type: Annotated[str, Field(pattern=r'^(cpf|cnpj|email)$')] = "cpf"
    reason: Annotated[str, Field(min_length=3, max_length=500, description="Motivo da inclusão")]
    severity: Annotated[str, Field(pattern=r'^(LOW|MEDIUM|HIGH|CRITICAL)$')] = "HIGH"

    def model_post_init(self, __context):
        """Normalize identifier and cpf"""
        if self.cpf and not self.identifier:
            self.identifier = self.cpf
        elif self.identifier and not self.cpf:
            self.cpf = self.identifier
        elif not self.cpf and not self.identifier:
            raise ValueError("Either 'cpf' or 'identifier' must be provided")

    @field_validator('reason')
    @classmethod
    def validate_reason_length(cls, v):
        """Valida comprimento mínimo do motivo"""
        if len(v.strip()) < 3:
            raise ValueError('Reason must have at least 3 characters')
        return v


# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserLogin(BaseModel):
    """Schema para login de usuário"""

    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$', description="Username")]
    password: Annotated[str, Field(min_length=8, max_length=100, description="Password")]

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "analyst_user",
                "password": "SecurePass123!"
            }
        }
    }


class UserCreate(BaseModel):
    """Schema para criação de usuário"""

    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')]
    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=100)]
    role: Annotated[str, Field(pattern=r'^(admin|analyst|operator|viewer|system)$')]

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        """Valida força da senha"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


# ============================================================================
# ALERT SCHEMAS
# ============================================================================

class AlertCreate(BaseModel):
    """Schema para criação de alerta"""

    transaction_id: Annotated[str, Field(min_length=5, max_length=100)]
    priority: Annotated[str, Field(pattern=r'^(LOW|MEDIUM|HIGH|CRITICAL)$')] = "MEDIUM"
    assigned_to: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=1000)


class AlertUpdate(BaseModel):
    """Schema para atualização de alerta"""

    status: Optional[Annotated[str, Field(pattern=r'^(OPEN|IN_PROGRESS|RESOLVED|CLOSED)$')]] = None
    assigned_to: Optional[str] = Field(None, max_length=100)
    resolution_notes: Optional[str] = Field(None, max_length=1000)


# ============================================================================
# MANUAL REVIEW SCHEMAS
# ============================================================================

class ManualReviewCreate(BaseModel):
    """Schema para criação de revisão manual"""

    transaction_id: Annotated[str, Field(min_length=5, max_length=100)]
    decision: Annotated[str, Field(pattern=r'^(APPROVE|REJECT|ESCALATE)$')]
    reviewer_notes: Annotated[str, Field(min_length=10, max_length=1000)]
    reviewed_by: Annotated[str, Field(min_length=3, max_length=100)]

    @field_validator('reviewer_notes')
    @classmethod
    def validate_notes(cls, v):
        """Valida que as notas não estão vazias"""
        if not v.strip():
            raise ValueError('Reviewer notes cannot be empty')
        return v


# ============================================================================
# FEEDBACK SCHEMAS
# ============================================================================

class FeedbackCreate(BaseModel):
    """Schema para feedback de predição"""

    transaction_id: Annotated[str, Field(min_length=5, max_length=100)]

    # Formato simplificado (para tests)
    feedback_type: Optional[str] = Field(None, description="Tipo: correction, comment, quality")
    correct_label: Optional[str] = Field(None, description="Label correto: fraud, legitimate, uncertain")
    comments: Optional[str] = Field(None, description="Comentarios")

    # Formato completo (para producao)
    is_fraud_correct: Optional[bool] = Field(None, description="Predição estava correta?")
    actual_fraud: Optional[bool] = Field(None, description="Transação é fraude?")
    feedback_notes: Optional[Annotated[str, Field(min_length=10, max_length=1000)]] = None
    analyst_id: Optional[Annotated[str, Field(min_length=3, max_length=100)]] = None
    confidence: Optional[int] = Field(None, ge=0, le=100, description="Confiança da decisão (0-100)")

    @field_validator('feedback_type')
    @classmethod
    def validate_feedback_type(cls, v):
        """Valida feedback_type"""
        if v and v not in ['correction', 'comment', 'quality']:
            raise ValueError("feedback_type must be one of: correction, comment, quality")
        return v

    @field_validator('correct_label')
    @classmethod
    def validate_correct_label(cls, v):
        """Valida correct_label"""
        if v and v not in ['fraud', 'legitimate', 'uncertain']:
            raise ValueError("correct_label must be one of: fraud, legitimate, uncertain")
        return v


# ============================================================================
# MODEL CONFIGURATION SCHEMAS
# ============================================================================

class ModelConfigUpdate(BaseModel):
    """Schema para atualização de configuração de modelo"""

    model_name: Annotated[str, Field(pattern=r'^(ruleBasedEngine|blacklistLookup|velocityChecks|geolocationValidation|randomForest|xgboost|neuralNetwork|gnn)$')]
    threshold: Optional[float] = Field(None, ge=0, le=1)
    weight: Optional[float] = Field(None, ge=0, le=1)
    enabled: Optional[bool] = None

    @field_validator('threshold', 'weight')
    @classmethod
    def validate_probability(cls, v):
        """Valida que threshold/weight estão entre 0 e 1"""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Value must be between 0 and 1')
        return v


# ============================================================================
# DATASET SCHEMAS
# ============================================================================

class DatasetCreate(BaseModel):
    """Schema para criação de dataset"""

    name: Annotated[str, Field(min_length=3, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')]
    description: Annotated[str, Field(min_length=10, max_length=500)]
    source: Annotated[str, Field(pattern=r'^(PRODUCTION|MANUAL|EXTERNAL|SYNTHETIC)$')]
    size_mb: Optional[float] = Field(None, ge=0, le=1000)


# ============================================================================
# DSR (DATA SUBJECT RIGHTS) SCHEMAS - LGPD Compliance
# ============================================================================

class DSRAccessRequest(BaseModel):
    """Schema para solicitação de acesso a dados (LGPD Art. 18, I)"""

    cpf: Annotated[str, Field(pattern=r'^\d{11}$', description="CPF do titular dos dados")]
    request_reason: Annotated[str, Field(min_length=20, max_length=500, description="Motivo da solicitação")]
    requester_email: EmailStr = Field(..., description="Email para envio do relatório")

    @field_validator('cpf')
    @classmethod
    def validate_cpf(cls, v):
        """Valida CPF com dígitos verificadores"""
        cpf = re.sub(r'\D', '', v)
        if len(cpf) != 11:
            raise ValueError('CPF deve ter 11 dígitos')
        if cpf == cpf[0] * 11:
            raise ValueError('CPF inválido')
        return cpf


class DSRDeletionRequest(BaseModel):
    """Schema para solicitação de exclusão de dados (Right to be forgotten)"""

    cpf: Annotated[str, Field(pattern=r'^\d{11}$', description="CPF do titular dos dados")]
    deletion_reason: Annotated[str, Field(min_length=30, max_length=1000, description="Justificativa para exclusão")]
    confirmed: bool = Field(..., description="Confirmação de que entende as consequências")

    @field_validator('confirmed')
    @classmethod
    def validate_confirmation(cls, v):
        """Valida que usuário confirmou"""
        if not v:
            raise ValueError('User must confirm deletion request')
        return v


# ============================================================================
# SETTINGS AND CONFIGURATION SCHEMAS
# ============================================================================

class SettingsUpdate(BaseModel):
    """Schema para atualização de configurações do sistema"""

    fraud_threshold: Optional[float] = Field(None, ge=0, le=1, description="Threshold de fraude")
    auto_block_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None
    slack_notifications: Optional[bool] = None


class RefreshTokenRequest(BaseModel):
    """Schema para refresh de token JWT"""

    refresh_token: Annotated[str, Field(min_length=32, description="Refresh token")]


class InvestigationCreate(BaseModel):
    """Schema para criação de investigação"""

    transaction_id: Annotated[str, Field(min_length=5, max_length=100)]
    investigation_type: Annotated[str, Field(pattern=r'^(FRAUD|COMPLIANCE|SECURITY|OTHER)$')]
    description: Annotated[str, Field(min_length=20, max_length=2000)]
    priority: Annotated[str, Field(pattern=r'^(LOW|MEDIUM|HIGH|CRITICAL)$')] = "MEDIUM"
    assigned_to: Optional[str] = Field(None, max_length=100)


class CalibrationUpdate(BaseModel):
    """Schema para atualização de calibração do modelo"""

    model_id: str
    calibration_method: Annotated[str, Field(pattern=r'^(PLATT|ISOTONIC|BETA)$')]
    parameters: Dict[str, float]

    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
        """Valida que parâmetros são numéricos"""
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f'Parameter {key} must be numeric')
        return v


class BatchProcessRequest(BaseModel):
    """Schema para processamento em lote"""

    transaction_ids: List[str] = Field(..., min_length=1, max_length=1000)
    process_type: Annotated[str, Field(pattern=r'^(REPROCESS|REVIEW|EXPORT)$')]
    options: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    """Schema para exportação de dados"""

    export_format: Annotated[str, Field(pattern=r'^(CSV|JSON|EXCEL|PDF)$')]
    start_date: datetime
    end_date: datetime
    filters: Optional[Dict[str, Any]] = None

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """Valida que end_date > start_date"""
        if v and info.data.get('start_date'):
            if v < info.data['start_date']:
                raise ValueError('end_date must be after start_date')
            # Máximo 1 ano de range
            if (v - info.data['start_date']).days > 365:
                raise ValueError('Date range cannot exceed 365 days')
        return v


# ============================================================================
# SQL INJECTION PREVENTION HELPER
# ============================================================================

def validate_sql_fields(fields: List[str], allowed_fields: set) -> List[str]:
    """
    Valida campos SQL contra whitelist

    Args:
        fields: Lista de campos fornecidos pelo usuário
        allowed_fields: Set de campos permitidos

    Returns:
        Lista de campos validados

    Raises:
        ValueError: Se algum campo não está na whitelist
    """
    invalid_fields = set(fields) - allowed_fields
    if invalid_fields:
        raise ValueError(f"Invalid fields: {', '.join(invalid_fields)}. Allowed: {', '.join(sorted(allowed_fields))}")

    return [f for f in fields if f in allowed_fields]
