"""
P0-003: MCC (Merchant Category Code) Hierarchy Module
Hierarquia de categorias de comerciantes para enriquecimento de features.

Inclui:
- Codigo MCC completo
- Categorias e subcategorias
- Risk tiers por categoria
- Features derivadas

Author: Sankofa Enterprise Pro
Version: 1.0.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import numpy as np


class MCCRiskTier(Enum):
    """Nivel de risco por categoria MCC"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MCCCategory(Enum):
    """Categorias principais de MCC"""
    AGRICULTURAL = "agricultural"
    CONTRACTED_SERVICES = "contracted_services"
    AIRLINES = "airlines"
    CAR_RENTAL = "car_rental"
    LODGING = "lodging"
    TRANSPORTATION = "transportation"
    UTILITIES = "utilities"
    RETAIL_STORES = "retail_stores"
    CLOTHING = "clothing"
    MISC_STORES = "misc_stores"
    BUSINESS_SERVICES = "business_services"
    PROFESSIONAL_SERVICES = "professional_services"
    GOVERNMENT = "government"
    MONEY_TRANSFER = "money_transfer"
    GAMBLING = "gambling"
    CRYPTO = "crypto"
    FOOD_BEVERAGE = "food_beverage"
    DIGITAL_GOODS = "digital_goods"
    HEALTH = "health"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MCCCode:
    """Representacao de um codigo MCC"""
    code: str
    description: str
    category: MCCCategory
    subcategory: str
    risk_tier: MCCRiskTier
    fraud_rate_baseline: float  # Taxa de fraude historica (0-1)
    chargeback_rate: float  # Taxa de chargeback (0-1)
    is_high_ticket: bool  # Ticket medio alto
    is_recurring: bool  # Pagamentos recorrentes comuns
    is_digital: bool  # Produto/servico digital

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "risk_tier": self.risk_tier.value,
            "fraud_rate_baseline": self.fraud_rate_baseline,
            "chargeback_rate": self.chargeback_rate,
            "is_high_ticket": self.is_high_ticket,
            "is_recurring": self.is_recurring,
            "is_digital": self.is_digital,
        }


@dataclass
class MCCFeatures:
    """Features derivadas do MCC"""
    mcc_code: str
    category: str
    subcategory: str
    risk_tier: str
    risk_score: float  # 0-1
    fraud_rate_baseline: float
    chargeback_rate: float
    is_high_ticket: bool
    is_recurring: bool
    is_digital: bool
    is_high_risk_category: bool
    is_money_transfer: bool
    is_gambling: bool
    is_crypto: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mcc_code": self.mcc_code,
            "mcc_category": self.category,
            "mcc_subcategory": self.subcategory,
            "mcc_risk_tier": self.risk_tier,
            "mcc_risk_score": self.risk_score,
            "mcc_fraud_rate_baseline": self.fraud_rate_baseline,
            "mcc_chargeback_rate": self.chargeback_rate,
            "mcc_is_high_ticket": self.is_high_ticket,
            "mcc_is_recurring": self.is_recurring,
            "mcc_is_digital": self.is_digital,
            "mcc_is_high_risk_category": self.is_high_risk_category,
            "mcc_is_money_transfer": self.is_money_transfer,
            "mcc_is_gambling": self.is_gambling,
            "mcc_is_crypto": self.is_crypto,
        }

    def to_feature_array(self) -> np.ndarray:
        """Converte para array de features numericas"""
        risk_tier_map = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9,
        }
        return np.array([
            risk_tier_map.get(self.risk_tier, 0.5),
            self.risk_score,
            self.fraud_rate_baseline,
            self.chargeback_rate,
            float(self.is_high_ticket),
            float(self.is_recurring),
            float(self.is_digital),
            float(self.is_high_risk_category),
            float(self.is_money_transfer),
            float(self.is_gambling),
            float(self.is_crypto),
        ], dtype=np.float32)


class MCCHierarchy:
    """
    Hierarquia completa de codigos MCC.
    Baseado nos padroes ISO 18245.
    """

    VERSION = "1.0.0"

    # Mapeamento de MCCs conhecidos (subset representativo)
    # Em producao, isso seria uma tabela completa com ~400+ MCCs
    MCC_DATABASE: Dict[str, MCCCode] = {
        # Servicos financeiros - ALTO RISCO
        "6012": MCCCode("6012", "Financial Institutions", MCCCategory.MONEY_TRANSFER, "Banks", MCCRiskTier.HIGH, 0.015, 0.008, True, False, False),
        "6051": MCCCode("6051", "Non-Financial Institutions - Foreign Currency", MCCCategory.MONEY_TRANSFER, "Currency Exchange", MCCRiskTier.VERY_HIGH, 0.025, 0.012, True, False, False),
        "6211": MCCCode("6211", "Security Brokers/Dealers", MCCCategory.MONEY_TRANSFER, "Securities", MCCRiskTier.HIGH, 0.012, 0.006, True, False, False),
        "6300": MCCCode("6300", "Insurance Services", MCCCategory.PROFESSIONAL_SERVICES, "Insurance", MCCRiskTier.LOW, 0.003, 0.002, True, True, False),
        "6540": MCCCode("6540", "POI Funding Transactions", MCCCategory.MONEY_TRANSFER, "Payment Services", MCCRiskTier.VERY_HIGH, 0.035, 0.015, True, False, True),

        # Criptomoedas - MUITO ALTO RISCO
        "6051": MCCCode("6051", "Crypto Currency", MCCCategory.CRYPTO, "Cryptocurrency", MCCRiskTier.VERY_HIGH, 0.045, 0.025, True, False, True),

        # Jogos e apostas - MUITO ALTO RISCO
        "7801": MCCCode("7801", "Government Licensed Online Casinos", MCCCategory.GAMBLING, "Online Casino", MCCRiskTier.VERY_HIGH, 0.055, 0.030, True, False, True),
        "7802": MCCCode("7802", "Government Licensed Horse Racing", MCCCategory.GAMBLING, "Horse Racing", MCCRiskTier.VERY_HIGH, 0.045, 0.025, True, False, False),
        "7995": MCCCode("7995", "Betting", MCCCategory.GAMBLING, "Sports Betting", MCCRiskTier.VERY_HIGH, 0.050, 0.028, True, False, True),

        # Viagem e hospedagem - MEDIO RISCO
        "3000": MCCCode("3000", "Airlines", MCCCategory.AIRLINES, "Airlines", MCCRiskTier.MEDIUM, 0.008, 0.010, True, False, False),
        "3501": MCCCode("3501", "Hilton Hotels", MCCCategory.LODGING, "Hotels", MCCRiskTier.MEDIUM, 0.006, 0.008, True, False, False),
        "7011": MCCCode("7011", "Lodging - Hotels", MCCCategory.LODGING, "Hotels", MCCRiskTier.MEDIUM, 0.007, 0.009, True, False, False),
        "7512": MCCCode("7512", "Car Rental", MCCCategory.CAR_RENTAL, "Car Rental", MCCRiskTier.MEDIUM, 0.006, 0.007, True, False, False),

        # Varejo - BAIXO/MEDIO RISCO
        "5411": MCCCode("5411", "Grocery Stores/Supermarkets", MCCCategory.RETAIL_STORES, "Grocery", MCCRiskTier.VERY_LOW, 0.001, 0.001, False, False, False),
        "5311": MCCCode("5311", "Department Stores", MCCCategory.RETAIL_STORES, "Department", MCCRiskTier.LOW, 0.003, 0.004, False, False, False),
        "5812": MCCCode("5812", "Eating Places/Restaurants", MCCCategory.FOOD_BEVERAGE, "Restaurants", MCCRiskTier.VERY_LOW, 0.002, 0.002, False, False, False),
        "5814": MCCCode("5814", "Fast Food Restaurants", MCCCategory.FOOD_BEVERAGE, "Fast Food", MCCRiskTier.VERY_LOW, 0.001, 0.001, False, False, False),
        "5912": MCCCode("5912", "Drug Stores/Pharmacies", MCCCategory.HEALTH, "Pharmacy", MCCRiskTier.VERY_LOW, 0.002, 0.002, False, True, False),
        "5541": MCCCode("5541", "Service Stations", MCCCategory.TRANSPORTATION, "Gas Stations", MCCRiskTier.LOW, 0.003, 0.003, False, False, False),

        # Eletronicos e digital - ALTO RISCO (fraude comum)
        "5732": MCCCode("5732", "Electronics Stores", MCCCategory.RETAIL_STORES, "Electronics", MCCRiskTier.HIGH, 0.015, 0.012, True, False, False),
        "5735": MCCCode("5735", "Record Stores", MCCCategory.ENTERTAINMENT, "Music", MCCRiskTier.MEDIUM, 0.008, 0.006, False, False, True),
        "5816": MCCCode("5816", "Digital Goods - Games", MCCCategory.DIGITAL_GOODS, "Games", MCCRiskTier.HIGH, 0.020, 0.015, False, True, True),
        "5817": MCCCode("5817", "Digital Goods - Apps", MCCCategory.DIGITAL_GOODS, "Apps", MCCRiskTier.MEDIUM, 0.012, 0.008, False, True, True),
        "5818": MCCCode("5818", "Digital Goods - Large Seller", MCCCategory.DIGITAL_GOODS, "Digital Marketplace", MCCRiskTier.HIGH, 0.018, 0.012, False, True, True),

        # Servicos profissionais - BAIXO RISCO
        "8011": MCCCode("8011", "Doctors", MCCCategory.HEALTH, "Medical", MCCRiskTier.VERY_LOW, 0.002, 0.002, True, False, False),
        "8021": MCCCode("8021", "Dentists", MCCCategory.HEALTH, "Dental", MCCRiskTier.VERY_LOW, 0.002, 0.002, True, False, False),
        "8111": MCCCode("8111", "Legal Services", MCCCategory.PROFESSIONAL_SERVICES, "Legal", MCCRiskTier.LOW, 0.003, 0.003, True, False, False),
        "8299": MCCCode("8299", "Schools", MCCCategory.EDUCATION, "Education", MCCRiskTier.VERY_LOW, 0.002, 0.002, True, True, False),

        # Transferencia de dinheiro - MUITO ALTO RISCO
        "4829": MCCCode("4829", "Money Transfer", MCCCategory.MONEY_TRANSFER, "Wire Transfer", MCCRiskTier.VERY_HIGH, 0.040, 0.020, True, False, True),
        "6010": MCCCode("6010", "Manual Cash Disbursement", MCCCategory.MONEY_TRANSFER, "Cash", MCCRiskTier.VERY_HIGH, 0.035, 0.018, True, False, False),
        "6011": MCCCode("6011", "ATM", MCCCategory.MONEY_TRANSFER, "ATM", MCCRiskTier.HIGH, 0.020, 0.010, True, False, False),

        # Governo - MUITO BAIXO RISCO
        "9211": MCCCode("9211", "Government Services", MCCCategory.GOVERNMENT, "Court Costs", MCCRiskTier.VERY_LOW, 0.001, 0.001, True, False, False),
        "9311": MCCCode("9311", "Tax Payments", MCCCategory.GOVERNMENT, "Taxes", MCCRiskTier.VERY_LOW, 0.001, 0.001, True, False, False),
    }

    # Categorias de alto risco
    HIGH_RISK_CATEGORIES: Set[MCCCategory] = {
        MCCCategory.GAMBLING,
        MCCCategory.CRYPTO,
        MCCCategory.MONEY_TRANSFER,
    }

    def __init__(self):
        self._code_index: Dict[str, MCCCode] = self.MCC_DATABASE.copy()

    def get_mcc(self, code: str) -> Optional[MCCCode]:
        """Retorna informacoes de um MCC"""
        return self._code_index.get(code)

    def get_mcc_or_default(self, code: str) -> MCCCode:
        """Retorna MCC ou um default para codigos desconhecidos"""
        mcc = self._code_index.get(code)
        if mcc:
            return mcc

        # MCC desconhecido - usar valores medios
        return MCCCode(
            code=code,
            description=f"Unknown MCC {code}",
            category=MCCCategory.UNKNOWN,
            subcategory="Unknown",
            risk_tier=MCCRiskTier.MEDIUM,
            fraud_rate_baseline=0.01,
            chargeback_rate=0.005,
            is_high_ticket=False,
            is_recurring=False,
            is_digital=False,
        )

    def get_category_risk_score(self, category: MCCCategory) -> float:
        """Retorna score de risco para uma categoria"""
        risk_scores = {
            MCCCategory.AGRICULTURAL: 0.2,
            MCCCategory.CONTRACTED_SERVICES: 0.3,
            MCCCategory.AIRLINES: 0.5,
            MCCCategory.CAR_RENTAL: 0.4,
            MCCCategory.LODGING: 0.4,
            MCCCategory.TRANSPORTATION: 0.3,
            MCCCategory.UTILITIES: 0.2,
            MCCCategory.RETAIL_STORES: 0.3,
            MCCCategory.CLOTHING: 0.3,
            MCCCategory.MISC_STORES: 0.4,
            MCCCategory.BUSINESS_SERVICES: 0.3,
            MCCCategory.PROFESSIONAL_SERVICES: 0.2,
            MCCCategory.GOVERNMENT: 0.1,
            MCCCategory.MONEY_TRANSFER: 0.9,
            MCCCategory.GAMBLING: 0.95,
            MCCCategory.CRYPTO: 0.9,
            MCCCategory.FOOD_BEVERAGE: 0.2,
            MCCCategory.DIGITAL_GOODS: 0.6,
            MCCCategory.HEALTH: 0.2,
            MCCCategory.EDUCATION: 0.2,
            MCCCategory.ENTERTAINMENT: 0.4,
            MCCCategory.UNKNOWN: 0.5,
        }
        return risk_scores.get(category, 0.5)


class MCCFeatureGenerator:
    """Gerador de features baseadas em MCC"""

    VERSION = "1.0.0"

    def __init__(self):
        self.hierarchy = MCCHierarchy()

    def generate(self, mcc_code: str) -> MCCFeatures:
        """Gera features para um MCC"""

        mcc = self.hierarchy.get_mcc_or_default(mcc_code)

        # Calcular risk score combinado
        tier_scores = {
            MCCRiskTier.VERY_LOW: 0.1,
            MCCRiskTier.LOW: 0.3,
            MCCRiskTier.MEDIUM: 0.5,
            MCCRiskTier.HIGH: 0.7,
            MCCRiskTier.VERY_HIGH: 0.9,
        }
        base_risk = tier_scores[mcc.risk_tier]

        # Ajustar por taxas historicas
        risk_score = (
            base_risk * 0.5 +
            mcc.fraud_rate_baseline * 10 * 0.3 +  # Normalizar fraud rate
            mcc.chargeback_rate * 10 * 0.2  # Normalizar chargeback rate
        )
        risk_score = min(1.0, max(0.0, risk_score))

        return MCCFeatures(
            mcc_code=mcc.code,
            category=mcc.category.value,
            subcategory=mcc.subcategory,
            risk_tier=mcc.risk_tier.value,
            risk_score=round(risk_score, 4),
            fraud_rate_baseline=mcc.fraud_rate_baseline,
            chargeback_rate=mcc.chargeback_rate,
            is_high_ticket=mcc.is_high_ticket,
            is_recurring=mcc.is_recurring,
            is_digital=mcc.is_digital,
            is_high_risk_category=mcc.category in self.hierarchy.HIGH_RISK_CATEGORIES,
            is_money_transfer=mcc.category == MCCCategory.MONEY_TRANSFER,
            is_gambling=mcc.category == MCCCategory.GAMBLING,
            is_crypto=mcc.category == MCCCategory.CRYPTO,
        )

    def generate_batch(self, mcc_codes: List[str]) -> List[MCCFeatures]:
        """Gera features para um batch de MCCs"""
        return [self.generate(code) for code in mcc_codes]


# Singleton global
_mcc_generator: Optional[MCCFeatureGenerator] = None


def get_mcc_generator() -> MCCFeatureGenerator:
    """Retorna instancia singleton do gerador MCC"""
    global _mcc_generator
    if _mcc_generator is None:
        _mcc_generator = MCCFeatureGenerator()
    return _mcc_generator
