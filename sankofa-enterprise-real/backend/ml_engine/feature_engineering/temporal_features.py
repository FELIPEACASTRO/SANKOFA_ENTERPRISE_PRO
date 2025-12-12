"""
Sankofa Enterprise Pro - Temporal Feature Generator
Features baseadas em tempo e sazonalidade

Baseado em:
- IEEE-CIS winning solutions
- Time series feature engineering best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureGenerator:
    """
    Gerador de Features Temporais

    Gera 60+ features baseadas em:
    - Hora, dia, semana, mês
    - Ciclicidade (sin/cos)
    - Períodos específicos
    - Holidays e eventos
    """

    VERSION = "1.0.0"

    # Feriados brasileiros (simplificado)
    BRAZILIAN_HOLIDAYS = [
        (1, 1),   # Ano Novo
        (4, 21),  # Tiradentes
        (5, 1),   # Dia do Trabalho
        (9, 7),   # Independência
        (10, 12), # Nossa Senhora
        (11, 2),  # Finados
        (11, 15), # Proclamação da República
        (12, 25), # Natal
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"TemporalFeatureGenerator v{self.VERSION} initialized")

    def generate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Gerar todas as features temporais

        Args:
            df: DataFrame com coluna de timestamp
            timestamp_col: Nome da coluna de timestamp

        Returns:
            DataFrame com features temporais adicionadas
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Column {timestamp_col} not found")
            return df

        df = df.copy()
        ts = pd.to_datetime(df[timestamp_col])

        # 1. Features básicas
        df = self._generate_basic_features(df, ts)

        # 2. Features cíclicas
        df = self._generate_cyclic_features(df, ts)

        # 3. Features de período
        df = self._generate_period_features(df, ts)

        # 4. Features de holidays
        df = self._generate_holiday_features(df, ts)

        # 5. Features de sazonalidade
        df = self._generate_seasonality_features(df, ts)

        # 6. Features de padrão temporal
        df = self._generate_pattern_features(df, ts)

        return df

    def _generate_basic_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features básicas de tempo"""

        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["second"] = ts.dt.second
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["day_of_year"] = ts.dt.dayofyear
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year

        # Minutos desde meia-noite
        df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]

        # Segundos desde meia-noite
        df["seconds_since_midnight"] = df["hour"] * 3600 + df["minute"] * 60 + df["second"]

        # Fração do dia
        df["day_fraction"] = df["seconds_since_midnight"] / 86400

        return df

    def _generate_cyclic_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features cíclicas usando sin/cos"""

        # Hora (ciclo de 24h)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Minuto (ciclo de 60)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        # Dia da semana (ciclo de 7)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Dia do mês (ciclo de ~30)
        df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        # Semana do ano (ciclo de 52)
        df["week_of_year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_of_year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

        # Mês (ciclo de 12)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Dia do ano (ciclo de 365)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        return df

    def _generate_period_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features de períodos específicos"""

        # Período do dia
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df["is_early_morning"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
        df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
        df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
        df["is_late_night"] = ((df["hour"] >= 22) | (df["hour"] < 2)).astype(int)

        # Horário comercial
        df["is_business_hours"] = (
            (df["hour"] >= 9) & (df["hour"] <= 18) & (df["day_of_week"] < 5)
        ).astype(int)

        # Horário de almoço
        df["is_lunch_time"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)

        # Dias da semana
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_weekday"] = (df["day_of_week"] < 5).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_sunday"] = (df["day_of_week"] == 6).astype(int)

        # Início/fim de período
        df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)
        df["is_quarter_start"] = (df["month"].isin([1, 4, 7, 10]) & (df["day_of_month"] <= 5)).astype(int)
        df["is_quarter_end"] = (df["month"].isin([3, 6, 9, 12]) & (df["day_of_month"] >= 25)).astype(int)
        df["is_year_start"] = ((df["month"] == 1) & (df["day_of_month"] <= 5)).astype(int)
        df["is_year_end"] = ((df["month"] == 12) & (df["day_of_month"] >= 25)).astype(int)

        # Dia do pagamento típico (dia 5 ou último dia útil)
        df["is_payday"] = ((df["day_of_month"] >= 1) & (df["day_of_month"] <= 7)).astype(int)

        return df

    def _generate_holiday_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features relacionadas a feriados"""

        # Criar coluna de data
        dates = ts.dt.date

        # Verificar feriados
        def is_holiday(date):
            return (date.month, date.day) in self.BRAZILIAN_HOLIDAYS

        df["is_holiday"] = dates.apply(is_holiday).astype(int)

        # Véspera de feriado
        def is_holiday_eve(date):
            next_day = date + timedelta(days=1)
            return (next_day.month, next_day.day) in self.BRAZILIAN_HOLIDAYS

        df["is_holiday_eve"] = dates.apply(is_holiday_eve).astype(int)

        # Pós-feriado
        def is_post_holiday(date):
            prev_day = date - timedelta(days=1)
            return (prev_day.month, prev_day.day) in self.BRAZILIAN_HOLIDAYS

        df["is_post_holiday"] = dates.apply(is_post_holiday).astype(int)

        # Período de férias escolares (simplificado)
        df["is_school_vacation"] = (
            (df["month"].isin([1, 7, 12])) |
            ((df["month"] == 2) & (df["day_of_month"] <= 28))
        ).astype(int)

        return df

    def _generate_seasonality_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features de sazonalidade"""

        # Estações do ano (hemisfério sul)
        df["is_summer"] = df["month"].isin([12, 1, 2]).astype(int)
        df["is_autumn"] = df["month"].isin([3, 4, 5]).astype(int)
        df["is_winter"] = df["month"].isin([6, 7, 8]).astype(int)
        df["is_spring"] = df["month"].isin([9, 10, 11]).astype(int)

        # Primeiro/segundo semestre
        df["is_first_semester"] = (df["month"] <= 6).astype(int)
        df["is_second_semester"] = (df["month"] > 6).astype(int)

        # Black Friday period (última semana de novembro)
        df["is_black_friday_week"] = (
            (df["month"] == 11) & (df["day_of_month"] >= 20)
        ).astype(int)

        # Período de Natal
        df["is_christmas_period"] = (
            (df["month"] == 12) & (df["day_of_month"] >= 15)
        ).astype(int)

        return df

    def _generate_pattern_features(
        self,
        df: pd.DataFrame,
        ts: pd.Series
    ) -> pd.DataFrame:
        """Features de padrões temporais"""

        # Combinações de período
        df["weekend_night"] = (df["is_weekend"] & df["is_night"]).astype(int)
        df["weekday_business"] = (df["is_weekday"] & df["is_business_hours"]).astype(int)
        df["weekday_night"] = (df["is_weekday"] & df["is_night"]).astype(int)

        # Horários suspeitos
        df["is_suspicious_time"] = (
            ((df["hour"] >= 2) & (df["hour"] <= 5)) |  # Madrugada
            ((df["hour"] >= 23) & (df["day_of_week"] < 5))  # Tarde da noite em dia útil
        ).astype(int)

        # Timestamp normalizado
        min_ts = ts.min()
        max_ts = ts.max()
        if max_ts > min_ts:
            df["timestamp_normalized"] = (ts - min_ts).dt.total_seconds() / (max_ts - min_ts).total_seconds()
        else:
            df["timestamp_normalized"] = 0.5

        # Unix timestamp (para modelos que preferem)
        df["unix_timestamp"] = ts.astype(np.int64) // 10**9

        return df

    def get_feature_names(self) -> List[str]:
        """Retornar lista de features temporais"""
        return [
            # Básicas
            "hour", "minute", "second", "day_of_week", "day_of_month",
            "day_of_year", "week_of_year", "month", "quarter", "year",
            "minutes_since_midnight", "seconds_since_midnight", "day_fraction",

            # Cíclicas
            "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "day_of_week_sin", "day_of_week_cos",
            "day_of_month_sin", "day_of_month_cos",
            "week_of_year_sin", "week_of_year_cos",
            "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos",

            # Períodos
            "is_night", "is_early_morning", "is_morning", "is_afternoon",
            "is_evening", "is_late_night", "is_business_hours", "is_lunch_time",
            "is_weekend", "is_weekday", "is_monday", "is_friday", "is_sunday",
            "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
            "is_year_start", "is_year_end", "is_payday",

            # Holidays
            "is_holiday", "is_holiday_eve", "is_post_holiday", "is_school_vacation",

            # Sazonalidade
            "is_summer", "is_autumn", "is_winter", "is_spring",
            "is_first_semester", "is_second_semester",
            "is_black_friday_week", "is_christmas_period",

            # Padrões
            "weekend_night", "weekday_business", "weekday_night",
            "is_suspicious_time", "timestamp_normalized", "unix_timestamp"
        ]


def create_temporal_generator(config: Optional[Dict[str, Any]] = None) -> TemporalFeatureGenerator:
    """Factory function para criar TemporalFeatureGenerator"""
    return TemporalFeatureGenerator(config)
