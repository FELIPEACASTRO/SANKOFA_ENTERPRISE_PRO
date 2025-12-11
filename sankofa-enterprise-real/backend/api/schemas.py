"""
Pydantic Schemas para Validação de Input
Implementa validação robusta para TODOS os endpoints da API
Previne SQL Injection, XSS, e outros ataques de input
"""

from pydantic import BaseModel, Field, validator, EmailStr, constr
from typing import Optional, List, Dict, Any
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
    cpf: constr(regex=r'^\d{11}$') = Field(
        ...,
        description="CPF do cliente (somente números, 11 dígitos)"
    )
    channel: constr(regex=r'^(PIX|TED|DOC|BOLETO|CARTAO_CREDITO|CARTAO_DEBITO|APP|WEB|ATM)$') = Field(
        ...,
        description="Canal da transação"
    )
    tipo_transacao: Optional[str] = Field(
        None,
        regex=r'^(PIX|TED|DOC|CARTAO_CREDITO|CARTAO_DEBITO|TRANSFERENCIA|PAGAMENTO)$'
    )
    location: Optional[str] = Field(None, max_length=200)
    device_id: Optional[str] = Field(None, max_length=100)
    ip_address: Optional[str] = Field(None, regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')

    @validator('amount')
    def validate_amount(cls, v):
        """Validação adicional de amount"""
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 100000:
            # High value transactions require additional validation
            pass
        return round(v, 2)

    @validator('cpf')
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

    class Config:
        schema_extra = {
            "example": {
                "amount": 1500.50,
                "cpf": "12345678901",
                "channel": "PIX",
                "location": "São Paulo, SP"
            }
        }


class FraudPredictionBatchRequest(BaseModel):
    """Schema para request de predição em lote"""

    transactions: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=1000,
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

    @validator('transactions')
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

    class Config:
        schema_extra = {
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


class TransactionFilterRequest(BaseModel):
    """Schema para filtros de consulta de transações"""

    limit: int = Field(100, ge=1, le=1000, description="Máximo de resultados")
    offset: int = Field(0, ge=0, description="Offset para paginação")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    channel: Optional[str] = Field(None, regex=r'^(PIX|TED|DOC|BOLETO|CARTAO_CREDITO|CARTAO_DEBITO|APP|WEB|ATM)$')
    is_fraud: Optional[bool] = None
    min_amount: Optional[float] = Field(None, ge=0)
    max_amount: Optional[float] = Field(None, le=10000000)

    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Valida que end_date > start_date"""
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v


# ============================================================================
# HARD RULES SCHEMAS
# ============================================================================

class HardRuleCreate(BaseModel):
    """Schema para criação de hard rule"""

    name: constr(min_length=3, max_length=100) = Field(..., description="Nome da regra")
    condition: Optional[str] = Field(None, max_length=500, description="Condição SQL")
    action: constr(regex=r'^(BLOCK|REVIEW|STEP_UP|ALLOW)$') = Field(..., description="Ação")
    enabled: bool = Field(True, description="Regra ativa?")
    priority: Optional[int] = Field(None, ge=0, le=100)

    @validator('condition')
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

    class Config:
        schema_extra = {
            "example": {
                "name": "Bloqueio Alto Valor",
                "condition": "amount > 50000",
                "action": "BLOCK",
                "enabled": True
            }
        }


class HardRuleUpdate(BaseModel):
    """Schema para atualização de hard rule - apenas campos permitidos"""

    # WHITELIST DE CAMPOS PERMITIDOS - Previne SQL injection
    ALLOWED_FIELDS = {'name', 'condition', 'action', 'enabled', 'priority'}

    name: Optional[constr(min_length=3, max_length=100)] = None
    condition: Optional[str] = Field(None, max_length=500)
    action: Optional[constr(regex=r'^(BLOCK|REVIEW|STEP_UP|ALLOW)$')] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = Field(None, ge=0, le=100)

    @validator('condition')
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

    def get_safe_fields(self) -> Dict[str, Any]:
        """Retorna apenas campos permitidos que foram definidos"""
        data = self.dict(exclude_unset=True)
        return {k: v for k, v in data.items() if k in self.ALLOWED_FIELDS}


# ============================================================================
# VIP/HOT LIST SCHEMAS
# ============================================================================

class VipListCreate(BaseModel):
    """Schema para adicionar CPF à lista VIP"""

    identifier: constr(regex=r'^\d{11}$') = Field(..., description="CPF (11 dígitos)")
    type: constr(regex=r'^(cpf|cnpj|email)$') = Field("cpf", description="Tipo de identificador")
    reason: constr(min_length=5, max_length=200) = Field(..., description="Motivo da inclusão")

    class Config:
        schema_extra = {
            "example": {
                "identifier": "12345678901",
                "type": "cpf",
                "reason": "Cliente premium com histórico limpo"
            }
        }


class HotListCreate(BaseModel):
    """Schema para adicionar CPF à lista negra"""

    identifier: constr(regex=r'^\d{11}$') = Field(..., description="CPF (11 dígitos)")
    type: constr(regex=r'^(cpf|cnpj|email)$') = Field("cpf", description="Tipo de identificador")
    reason: constr(min_length=10, max_length=500) = Field(..., description="Motivo da inclusão")
    severity: constr(regex=r'^(LOW|MEDIUM|HIGH|CRITICAL)$') = Field("HIGH", description="Severidade")

    @validator('reason')
    def validate_reason(cls, v):
        """Motivo deve ser detalhado para hot list"""
        if len(v) < 10:
            raise ValueError('Motivo deve ter pelo menos 10 caracteres para hot list')
        return v


# ============================================================================
# USER & AUTH SCHEMAS
# ============================================================================

class UserLogin(BaseModel):
    """Schema para login"""

    username: constr(min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$') = Field(
        ...,
        description="Username (apenas letras, números, _ e -)"
    )
    password: constr(min_length=8, max_length=100) = Field(..., description="Password")

    class Config:
        schema_extra = {
            "example": {
                "username": "analyst_user",
                "password": "SecureP@ssw0rd"
            }
        }


class UserCreate(BaseModel):
    """Schema para criação de usuário"""

    username: constr(min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr
    password: constr(min_length=8, max_length=100)
    role: constr(regex=r'^(admin|analyst|operator|viewer|system)$')

    @validator('password')
    def validate_password_strength(cls, v):
        """Valida força da senha"""
        if len(v) < 8:
            raise ValueError('Senha deve ter no mínimo 8 caracteres')

        if not re.search(r'[A-Z]', v):
            raise ValueError('Senha deve conter pelo menos uma letra maiúscula')

        if not re.search(r'[a-z]', v):
            raise ValueError('Senha deve conter pelo menos uma letra minúscula')

        if not re.search(r'\d', v):
            raise ValueError('Senha deve conter pelo menos um número')

        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', v):
            raise ValueError('Senha deve conter pelo menos um caractere especial')

        return v


# ============================================================================
# INVESTIGATION SCHEMAS
# ============================================================================

class InvestigationCreate(BaseModel):
    """Schema para criar investigação"""

    transaction_id: constr(min_length=5, max_length=100)
    priority: constr(regex=r'^(LOW|MEDIUM|HIGH|CRITICAL)$') = Field("MEDIUM")
    assigned_to: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)


class InvestigationUpdate(BaseModel):
    """Schema para atualizar investigação"""

    status: Optional[constr(regex=r'^(OPEN|IN_PROGRESS|RESOLVED|CLOSED)$')] = None
    assigned_to: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)
    resolution: Optional[str] = Field(None, max_length=1000)


# ============================================================================
# MANUAL REVIEW SCHEMAS
# ============================================================================

class ManualReviewDecision(BaseModel):
    """Schema para decisão de revisão manual"""

    transaction_id: constr(min_length=5, max_length=100)
    decision: constr(regex=r'^(APPROVE|REJECT|ESCALATE)$')
    reviewer_notes: constr(min_length=10, max_length=1000)
    reviewed_by: constr(min_length=3, max_length=100)

    @validator('reviewer_notes')
    def validate_notes(cls, v, values):
        """Notas são obrigatórias e detalhadas para REJECT"""
        if 'decision' in values and values['decision'] == 'REJECT':
            if len(v) < 20:
                raise ValueError('Notas devem ser detalhadas para rejeição (mín. 20 caracteres)')
        return v


# ============================================================================
# FEEDBACK SCHEMAS
# ============================================================================

class FeedbackSubmit(BaseModel):
    """Schema para feedback de analista"""

    transaction_id: constr(min_length=5, max_length=100)
    is_fraud_correct: bool = Field(..., description="Predição estava correta?")
    actual_fraud: bool = Field(..., description="Transação é fraude?")
    feedback_notes: constr(min_length=10, max_length=1000)
    analyst_id: constr(min_length=3, max_length=100)
    confidence: int = Field(..., ge=0, le=100, description="Confiança da decisão (0-100)")


# ============================================================================
# CALIBRATION SCHEMAS
# ============================================================================

class CalibrationUpdate(BaseModel):
    """Schema para atualizar calibração de modelos"""

    model_name: constr(
        regex=r'^(ruleBasedEngine|blacklistLookup|velocityChecks|geolocationValidation|randomForest|xgboost|neuralNetwork|gnn)$'
    )
    enabled: Optional[bool] = None
    threshold: Optional[float] = Field(None, ge=0, le=1)
    weight: Optional[float] = Field(None, ge=0, le=1)

    @validator('threshold', 'weight')
    def validate_probability(cls, v):
        """Valida que valores são probabilidades válidas"""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Valor deve estar entre 0 e 1')
        return v


# ============================================================================
# DATASET SCHEMAS
# ============================================================================

class DatasetUpload(BaseModel):
    """Schema para upload de dataset"""

    name: constr(min_length=3, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    description: constr(min_length=10, max_length=500)
    source: constr(regex=r'^(PRODUCTION|MANUAL|EXTERNAL|SYNTHETIC)$')
    size_mb: Optional[float] = Field(None, ge=0, le=1000)


# ============================================================================
# QUERY PARAMETER SCHEMAS
# ============================================================================

class PaginationParams(BaseModel):
    """Schema para parâmetros de paginação"""

    page: int = Field(1, ge=1, le=10000, description="Número da página")
    per_page: int = Field(100, ge=1, le=1000, description="Itens por página")

    @property
    def offset(self) -> int:
        """Calcula offset para query"""
        return (self.page - 1) * self.per_page

    @property
    def limit(self) -> int:
        """Retorna limit para query"""
        return self.per_page


class DateRangeParams(BaseModel):
    """Schema para parâmetros de range de datas"""

    start_date: datetime
    end_date: datetime

    @validator('end_date')
    def validate_range(cls, v, values):
        """Valida range de datas"""
        if 'start_date' in values:
            if v < values['start_date']:
                raise ValueError('end_date deve ser após start_date')

            # Máximo de 1 ano de range
            delta = v - values['start_date']
            if delta.days > 365:
                raise ValueError('Range máximo de 1 ano')

        return v


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class FraudPredictionResponse(BaseModel):
    """Schema para resposta de predição de fraude"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    model_version: str
    detection_reason: List[str]
    timestamp: str
    lgpd_explanation: Optional[str] = None


class SuccessResponse(BaseModel):
    """Schema para resposta de sucesso genérica"""

    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Schema para resposta de erro"""

    success: bool = False
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_sql_fields(fields: List[str], allowed_fields: set) -> List[str]:
    """
    Valida que apenas campos permitidos estão sendo usados em queries SQL
    Previne SQL injection através de field names maliciosos

    Args:
        fields: Lista de campos a validar
        allowed_fields: Set de campos permitidos

    Returns:
        Lista de campos validados

    Raises:
        ValueError: Se algum campo não está na whitelist
    """
    safe_fields = []

    for field in fields:
        # Remove espaços
        field = field.strip()

        # Verifica se está na whitelist
        if field not in allowed_fields:
            raise ValueError(f'Campo não permitido: {field}')

        # Verifica caracteres perigosos
        if not re.match(r'^[a-zA-Z0-9_]+$', field):
            raise ValueError(f'Campo contém caracteres inválidos: {field}')

        safe_fields.append(field)

    return safe_fields


def sanitize_sql_value(value: Any) -> Any:
    """
    Sanitiza valores para uso em queries SQL

    Args:
        value: Valor a sanitizar

    Returns:
        Valor sanitizado
    """
    if isinstance(value, str):
        # Remove caracteres perigosos
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
        for char in dangerous_chars:
            if char in value:
                raise ValueError(f'Caractere perigoso detectado: {char}')

    return value
