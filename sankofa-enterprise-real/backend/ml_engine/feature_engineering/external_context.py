"""
P0-001: External Context Feature Engineering
Contexto externo para enriquecimento de features de fraude.

Inclui:
- Feriados brasileiros (nacionais e estaduais)
- Eventos comerciais (Black Friday, Cyber Monday, etc.)
- Indicadores economicos
- Contexto temporal (dia de pagamento, fim de mes)

Author: Sankofa Enterprise Pro
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import numpy as np


class HolidayType(Enum):
    """Tipos de feriados"""
    NATIONAL = "national"
    STATE = "state"
    MUNICIPAL = "municipal"
    COMMERCIAL = "commercial"
    RELIGIOUS = "religious"


class EventRiskLevel(Enum):
    """Nivel de risco associado a eventos"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Holiday:
    """Representacao de um feriado"""
    name: str
    date: date
    holiday_type: HolidayType
    risk_multiplier: float = 1.0
    states: Optional[Set[str]] = None  # None = nacional

    def applies_to_state(self, state: str) -> bool:
        """Verifica se o feriado se aplica a um estado"""
        if self.states is None:
            return True  # Nacional
        return state.upper() in self.states


@dataclass(frozen=True)
class CommercialEvent:
    """Evento comercial que afeta comportamento de transacoes"""
    name: str
    start_date: date
    end_date: date
    risk_level: EventRiskLevel
    fraud_multiplier: float
    volume_multiplier: float
    description: str = ""


@dataclass
class ExternalContextFeatures:
    """Features extraidas do contexto externo"""
    # Feriados
    is_holiday: bool = False
    is_holiday_eve: bool = False
    is_bridge_day: bool = False  # Dia entre feriado e fim de semana
    holiday_name: Optional[str] = None
    holiday_type: Optional[str] = None
    days_to_next_holiday: int = 999
    days_from_last_holiday: int = 999

    # Eventos comerciais
    is_commercial_event: bool = False
    commercial_event_name: Optional[str] = None
    event_risk_level: Optional[str] = None
    event_fraud_multiplier: float = 1.0
    event_volume_multiplier: float = 1.0

    # Contexto temporal de pagamento
    is_salary_period: bool = False  # Dias 1-5 do mes
    is_end_of_month: bool = False  # Ultimos 3 dias
    is_first_business_day: bool = False
    days_to_month_end: int = 0

    # Indicadores economicos (placeholder para integracao futura)
    economic_stress_index: float = 0.5  # 0-1, maior = mais estresse

    # Risk score combinado
    context_risk_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario"""
        return {
            "is_holiday": self.is_holiday,
            "is_holiday_eve": self.is_holiday_eve,
            "is_bridge_day": self.is_bridge_day,
            "holiday_name": self.holiday_name,
            "holiday_type": self.holiday_type,
            "days_to_next_holiday": self.days_to_next_holiday,
            "days_from_last_holiday": self.days_from_last_holiday,
            "is_commercial_event": self.is_commercial_event,
            "commercial_event_name": self.commercial_event_name,
            "event_risk_level": self.event_risk_level,
            "event_fraud_multiplier": self.event_fraud_multiplier,
            "event_volume_multiplier": self.event_volume_multiplier,
            "is_salary_period": self.is_salary_period,
            "is_end_of_month": self.is_end_of_month,
            "is_first_business_day": self.is_first_business_day,
            "days_to_month_end": self.days_to_month_end,
            "economic_stress_index": self.economic_stress_index,
            "context_risk_multiplier": self.context_risk_multiplier,
        }

    def to_feature_array(self) -> np.ndarray:
        """Converte para array de features numericas"""
        return np.array([
            float(self.is_holiday),
            float(self.is_holiday_eve),
            float(self.is_bridge_day),
            min(self.days_to_next_holiday, 30) / 30.0,
            min(self.days_from_last_holiday, 30) / 30.0,
            float(self.is_commercial_event),
            self.event_fraud_multiplier,
            self.event_volume_multiplier,
            float(self.is_salary_period),
            float(self.is_end_of_month),
            float(self.is_first_business_day),
            self.days_to_month_end / 31.0,
            self.economic_stress_index,
            self.context_risk_multiplier,
        ], dtype=np.float32)


class BrazilianHolidayCalendar:
    """Calendario de feriados brasileiros"""

    # Feriados nacionais fixos
    FIXED_NATIONAL_HOLIDAYS = [
        ("Confraternizacao Universal", 1, 1, 1.2),
        ("Tiradentes", 4, 21, 1.1),
        ("Dia do Trabalho", 5, 1, 1.3),
        ("Independencia do Brasil", 9, 7, 1.1),
        ("Nossa Senhora Aparecida", 10, 12, 1.2),
        ("Finados", 11, 2, 1.0),
        ("Proclamacao da Republica", 11, 15, 1.1),
        ("Natal", 12, 25, 1.5),
    ]

    # Feriados estaduais (principais)
    STATE_HOLIDAYS = {
        "SP": [("Revolucao Constitucionalista", 7, 9)],
        "RJ": [("Dia de Sao Jorge", 4, 23), ("Consciencia Negra", 11, 20)],
        "MG": [("Dia de Minas Gerais", 7, 16)],
        "BA": [("Independencia da Bahia", 7, 2)],
        "RS": [("Dia do Gaucho", 9, 20)],
    }

    def __init__(self):
        self._holiday_cache: Dict[int, List[Holiday]] = {}

    def _calculate_easter(self, year: int) -> date:
        """Calcula a data da Pascoa usando o algoritmo de Meeus/Jones/Butcher"""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def get_holidays_for_year(self, year: int) -> List[Holiday]:
        """Retorna todos os feriados de um ano"""
        if year in self._holiday_cache:
            return self._holiday_cache[year]

        holidays = []

        # Feriados fixos nacionais
        for name, month, day, risk_mult in self.FIXED_NATIONAL_HOLIDAYS:
            holidays.append(Holiday(
                name=name,
                date=date(year, month, day),
                holiday_type=HolidayType.NATIONAL,
                risk_multiplier=risk_mult,
            ))

        # Feriados moveis baseados na Pascoa
        easter = self._calculate_easter(year)

        # Carnaval (47 dias antes da Pascoa)
        carnaval = easter - timedelta(days=47)
        holidays.append(Holiday(
            name="Carnaval",
            date=carnaval,
            holiday_type=HolidayType.NATIONAL,
            risk_multiplier=1.8,  # Alto risco - muito volume
        ))

        # Sexta-feira Santa (2 dias antes da Pascoa)
        sexta_santa = easter - timedelta(days=2)
        holidays.append(Holiday(
            name="Sexta-feira Santa",
            date=sexta_santa,
            holiday_type=HolidayType.NATIONAL,
            risk_multiplier=1.3,
        ))

        # Corpus Christi (60 dias apos Pascoa)
        corpus_christi = easter + timedelta(days=60)
        holidays.append(Holiday(
            name="Corpus Christi",
            date=corpus_christi,
            holiday_type=HolidayType.NATIONAL,
            risk_multiplier=1.2,
        ))

        # Feriados estaduais
        for state, state_holidays in self.STATE_HOLIDAYS.items():
            for name, month, day in state_holidays:
                holidays.append(Holiday(
                    name=name,
                    date=date(year, month, day),
                    holiday_type=HolidayType.STATE,
                    risk_multiplier=1.1,
                    states={state},
                ))

        self._holiday_cache[year] = holidays
        return holidays

    def is_holiday(self, check_date: date, state: Optional[str] = None) -> Optional[Holiday]:
        """Verifica se uma data e feriado"""
        holidays = self.get_holidays_for_year(check_date.year)
        for holiday in holidays:
            if holiday.date == check_date:
                if state is None or holiday.applies_to_state(state):
                    return holiday
        return None

    def days_to_next_holiday(self, check_date: date, state: Optional[str] = None) -> int:
        """Calcula dias ate o proximo feriado"""
        holidays = self.get_holidays_for_year(check_date.year)
        holidays.extend(self.get_holidays_for_year(check_date.year + 1))

        min_days = 999
        for holiday in holidays:
            if holiday.date > check_date:
                if state is None or holiday.applies_to_state(state):
                    days = (holiday.date - check_date).days
                    min_days = min(min_days, days)

        return min_days

    def days_from_last_holiday(self, check_date: date, state: Optional[str] = None) -> int:
        """Calcula dias desde o ultimo feriado"""
        holidays = self.get_holidays_for_year(check_date.year)
        holidays.extend(self.get_holidays_for_year(check_date.year - 1))

        min_days = 999
        for holiday in holidays:
            if holiday.date < check_date:
                if state is None or holiday.applies_to_state(state):
                    days = (check_date - holiday.date).days
                    min_days = min(min_days, days)

        return min_days


class CommercialEventCalendar:
    """Calendario de eventos comerciais"""

    def __init__(self):
        self._events_cache: Dict[int, List[CommercialEvent]] = {}

    def _get_black_friday(self, year: int) -> date:
        """Black Friday = 4a sexta-feira de novembro"""
        nov_first = date(year, 11, 1)
        # Encontrar primeira sexta
        days_until_friday = (4 - nov_first.weekday()) % 7
        first_friday = nov_first + timedelta(days=days_until_friday)
        # 4a sexta = 3 semanas depois
        return first_friday + timedelta(weeks=3)

    def get_events_for_year(self, year: int) -> List[CommercialEvent]:
        """Retorna eventos comerciais de um ano"""
        if year in self._events_cache:
            return self._events_cache[year]

        events = []

        # Black Friday (maior risco de fraude)
        bf_date = self._get_black_friday(year)
        events.append(CommercialEvent(
            name="Black Friday",
            start_date=bf_date,
            end_date=bf_date,
            risk_level=EventRiskLevel.CRITICAL,
            fraud_multiplier=3.0,
            volume_multiplier=5.0,
            description="Maior evento de vendas do ano - alto risco de fraude",
        ))

        # Cyber Monday (segunda apos Black Friday)
        cm_date = bf_date + timedelta(days=3)
        events.append(CommercialEvent(
            name="Cyber Monday",
            start_date=cm_date,
            end_date=cm_date,
            risk_level=EventRiskLevel.HIGH,
            fraud_multiplier=2.5,
            volume_multiplier=3.0,
            description="Vendas online pos-Black Friday",
        ))

        # Semana do Consumidor (15 de marco)
        events.append(CommercialEvent(
            name="Semana do Consumidor",
            start_date=date(year, 3, 10),
            end_date=date(year, 3, 15),
            risk_level=EventRiskLevel.MEDIUM,
            fraud_multiplier=1.5,
            volume_multiplier=2.0,
            description="Semana do Consumidor",
        ))

        # Dia das Maes (2o domingo de maio)
        may_first = date(year, 5, 1)
        days_until_sunday = (6 - may_first.weekday()) % 7
        first_sunday = may_first + timedelta(days=days_until_sunday)
        mothers_day = first_sunday + timedelta(weeks=1)
        events.append(CommercialEvent(
            name="Dia das Maes",
            start_date=mothers_day - timedelta(days=7),
            end_date=mothers_day,
            risk_level=EventRiskLevel.MEDIUM,
            fraud_multiplier=1.8,
            volume_multiplier=2.5,
            description="Semana do Dia das Maes",
        ))

        # Dia dos Namorados (12 de junho no Brasil)
        events.append(CommercialEvent(
            name="Dia dos Namorados",
            start_date=date(year, 6, 5),
            end_date=date(year, 6, 12),
            risk_level=EventRiskLevel.MEDIUM,
            fraud_multiplier=1.6,
            volume_multiplier=2.0,
            description="Semana do Dia dos Namorados",
        ))

        # Dia dos Pais (2o domingo de agosto)
        aug_first = date(year, 8, 1)
        days_until_sunday = (6 - aug_first.weekday()) % 7
        first_sunday = aug_first + timedelta(days=days_until_sunday)
        fathers_day = first_sunday + timedelta(weeks=1)
        events.append(CommercialEvent(
            name="Dia dos Pais",
            start_date=fathers_day - timedelta(days=7),
            end_date=fathers_day,
            risk_level=EventRiskLevel.MEDIUM,
            fraud_multiplier=1.5,
            volume_multiplier=1.8,
            description="Semana do Dia dos Pais",
        ))

        # Dia das Criancas (12 de outubro)
        events.append(CommercialEvent(
            name="Dia das Criancas",
            start_date=date(year, 10, 5),
            end_date=date(year, 10, 12),
            risk_level=EventRiskLevel.MEDIUM,
            fraud_multiplier=1.6,
            volume_multiplier=2.2,
            description="Semana do Dia das Criancas",
        ))

        # Natal (dezembro)
        events.append(CommercialEvent(
            name="Compras de Natal",
            start_date=date(year, 12, 1),
            end_date=date(year, 12, 24),
            risk_level=EventRiskLevel.HIGH,
            fraud_multiplier=2.0,
            volume_multiplier=3.5,
            description="Periodo de compras natalinas",
        ))

        self._events_cache[year] = events
        return events

    def get_active_event(self, check_date: date) -> Optional[CommercialEvent]:
        """Retorna evento ativo em uma data"""
        events = self.get_events_for_year(check_date.year)
        for event in events:
            if event.start_date <= check_date <= event.end_date:
                return event
        return None


class ExternalContextGenerator:
    """Gerador de features de contexto externo"""

    VERSION = "1.0.0"

    def __init__(self):
        self.holiday_calendar = BrazilianHolidayCalendar()
        self.event_calendar = CommercialEventCalendar()

    def _is_bridge_day(self, check_date: date, state: Optional[str] = None) -> bool:
        """Verifica se e dia de ponte (entre feriado e fim de semana)"""
        weekday = check_date.weekday()

        # Sexta ou segunda
        if weekday == 4:  # Sexta
            # Verifica se quinta foi feriado
            thursday = check_date - timedelta(days=1)
            if self.holiday_calendar.is_holiday(thursday, state):
                return True
        elif weekday == 0:  # Segunda
            # Verifica se terca e feriado
            tuesday = check_date + timedelta(days=1)
            if self.holiday_calendar.is_holiday(tuesday, state):
                return True

        return False

    def _is_first_business_day(self, check_date: date) -> bool:
        """Verifica se e o primeiro dia util do mes"""
        if check_date.day > 5:
            return False

        # Contar dias uteis desde o inicio do mes
        business_days = 0
        current = date(check_date.year, check_date.month, 1)
        while current <= check_date:
            if current.weekday() < 5:  # Nao e fim de semana
                if not self.holiday_calendar.is_holiday(current):
                    business_days += 1
            current += timedelta(days=1)

        return business_days == 1

    def generate(
        self,
        transaction_date: datetime,
        state: Optional[str] = None,
    ) -> ExternalContextFeatures:
        """Gera features de contexto externo para uma transacao"""

        check_date = transaction_date.date() if isinstance(transaction_date, datetime) else transaction_date

        # Inicializar features
        features = ExternalContextFeatures()

        # Verificar feriados
        holiday = self.holiday_calendar.is_holiday(check_date, state)
        if holiday:
            features = ExternalContextFeatures(
                is_holiday=True,
                holiday_name=holiday.name,
                holiday_type=holiday.holiday_type.value,
                days_to_next_holiday=0,
                days_from_last_holiday=0,
                context_risk_multiplier=holiday.risk_multiplier,
            )
        else:
            # Verificar vespera de feriado
            tomorrow = check_date + timedelta(days=1)
            tomorrow_holiday = self.holiday_calendar.is_holiday(tomorrow, state)

            features = ExternalContextFeatures(
                is_holiday=False,
                is_holiday_eve=tomorrow_holiday is not None,
                holiday_name=tomorrow_holiday.name if tomorrow_holiday else None,
                days_to_next_holiday=self.holiday_calendar.days_to_next_holiday(check_date, state),
                days_from_last_holiday=self.holiday_calendar.days_from_last_holiday(check_date, state),
            )

        # Verificar ponte
        features = ExternalContextFeatures(
            **{**features.to_dict(), "is_bridge_day": self._is_bridge_day(check_date, state)}
        )

        # Verificar eventos comerciais
        event = self.event_calendar.get_active_event(check_date)
        if event:
            features = ExternalContextFeatures(
                **{
                    **features.to_dict(),
                    "is_commercial_event": True,
                    "commercial_event_name": event.name,
                    "event_risk_level": event.risk_level.value,
                    "event_fraud_multiplier": event.fraud_multiplier,
                    "event_volume_multiplier": event.volume_multiplier,
                }
            )

        # Contexto de pagamento
        day_of_month = check_date.day
        is_salary_period = day_of_month <= 5

        # Dias ate fim do mes
        if check_date.month == 12:
            next_month = date(check_date.year + 1, 1, 1)
        else:
            next_month = date(check_date.year, check_date.month + 1, 1)
        days_to_month_end = (next_month - check_date).days - 1

        is_end_of_month = days_to_month_end <= 3
        is_first_business_day = self._is_first_business_day(check_date)

        features = ExternalContextFeatures(
            **{
                **features.to_dict(),
                "is_salary_period": is_salary_period,
                "is_end_of_month": is_end_of_month,
                "is_first_business_day": is_first_business_day,
                "days_to_month_end": days_to_month_end,
            }
        )

        # Calcular risk multiplier combinado
        risk_mult = 1.0
        if features.is_holiday:
            # Pegar do holiday original
            holiday = self.holiday_calendar.is_holiday(check_date, state)
            if holiday:
                risk_mult *= holiday.risk_multiplier
        if features.is_holiday_eve:
            risk_mult *= 1.1
        if features.is_bridge_day:
            risk_mult *= 1.15
        if features.is_commercial_event:
            risk_mult *= features.event_fraud_multiplier
        if features.is_salary_period:
            risk_mult *= 1.2  # Mais transacoes legitimas, mas tambem mais fraude
        if features.is_end_of_month:
            risk_mult *= 1.1

        features = ExternalContextFeatures(
            **{**features.to_dict(), "context_risk_multiplier": round(risk_mult, 3)}
        )

        return features

    def generate_batch(
        self,
        transaction_dates: List[datetime],
        states: Optional[List[str]] = None,
    ) -> List[ExternalContextFeatures]:
        """Gera features para um batch de transacoes"""
        if states is None:
            states = [None] * len(transaction_dates)

        return [
            self.generate(dt, state)
            for dt, state in zip(transaction_dates, states)
        ]


# Singleton para uso global
_context_generator: Optional[ExternalContextGenerator] = None


def get_context_generator() -> ExternalContextGenerator:
    """Retorna instancia singleton do gerador de contexto"""
    global _context_generator
    if _context_generator is None:
        _context_generator = ExternalContextGenerator()
    return _context_generator
