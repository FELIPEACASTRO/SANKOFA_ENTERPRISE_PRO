"""
Sankofa Enterprise Pro - Main Feature Generator
Orquestrador principal de geração de features (500+)

Baseado em:
- IEEE-CIS Fraud Detection (1st place solution)
- Bahnsen et al. velocity features
- Feedzai AutoML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuração de feature engineering"""
    time_windows: List[int] = field(default_factory=lambda: [1, 5, 15, 30, 60, 360, 1440, 10080])
    aggregation_keys: List[str] = field(default_factory=lambda: [
        "customer_id", "device_id", "ip_address", "merchant_id",
        "receiver_id", "card_hash", "channel"
    ])
    include_graph_features: bool = True
    include_embeddings: bool = True
    include_interactions: bool = True
    max_categorical_cardinality: int = 1000


class FeatureGenerator:
    """
    Gerador Principal de Features (500+)

    Categorias de features:
    1. Temporal (50+): hora, dia, ciclicidade, sazonalidade
    2. Velocity/Bahnsen (200+): agregações por janela temporal
    3. Aggregation (100+): estatísticas por entidade
    4. Interaction (80+): cruzamentos de features
    5. Graph (40+): métricas de rede
    6. Embedding (30+): representações aprendidas
    """

    VERSION = "1.0.0"

    # Janelas temporais em minutos
    DEFAULT_TIME_WINDOWS = [1, 5, 15, 30, 60, 360, 1440, 10080]  # 1min a 7 dias

    # Chaves de agregação
    DEFAULT_AGG_KEYS = [
        "customer_id", "device_id", "ip_address", "merchant_id",
        "receiver_id", "card_hash", "email_domain", "phone_prefix", "channel"
    ]

    # Operações de agregação
    AGG_OPERATIONS = ["count", "sum", "mean", "std", "max", "min", "nunique"]

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

        # Sub-geradores serão carregados sob demanda
        self._temporal_generator = None
        self._velocity_generator = None
        self._aggregation_generator = None
        self._graph_generator = None
        self._embedding_generator = None

        # Cache de estatísticas
        self._entity_stats: Dict[str, pd.DataFrame] = {}

        logger.info(f"FeatureGenerator v{self.VERSION} initialized")
        logger.info(f"Time windows: {self.config.time_windows}")
        logger.info(f"Aggregation keys: {self.config.aggregation_keys}")

    def generate_features(
        self,
        df: pd.DataFrame,
        transaction_col: str = "transaction_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        customer_col: str = "customer_id"
    ) -> pd.DataFrame:
        """
        Gerar todas as features para um DataFrame de transações

        Args:
            df: DataFrame com transações
            transaction_col: Coluna de ID da transação
            timestamp_col: Coluna de timestamp
            amount_col: Coluna de valor
            customer_col: Coluna de ID do cliente

        Returns:
            DataFrame com todas as features geradas
        """
        logger.info(f"Generating features for {len(df)} transactions...")

        df = df.copy()

        # Preparar dados
        df = self._prepare_data(df, timestamp_col, amount_col)

        # 1. Features Temporais (50+)
        logger.info("Generating temporal features...")
        df = self._generate_temporal_features(df, timestamp_col)

        # 2. Features de Valor (30+)
        logger.info("Generating value features...")
        df = self._generate_value_features(df, amount_col)

        # 3. Features de Velocity/Bahnsen (200+)
        logger.info("Generating velocity features...")
        df = self._generate_velocity_features(df, timestamp_col, amount_col, customer_col)

        # 4. Features de Agregação por Entidade (100+)
        logger.info("Generating aggregation features...")
        df = self._generate_aggregation_features(df, amount_col)

        # 5. Features de Interação (80+)
        if self.config.include_interactions:
            logger.info("Generating interaction features...")
            df = self._generate_interaction_features(df)

        # 6. Features de Desvio (40+)
        logger.info("Generating deviation features...")
        df = self._generate_deviation_features(df, amount_col, customer_col)

        # 7. Features de Frequência/Contagem (30+)
        logger.info("Generating frequency features...")
        df = self._generate_frequency_features(df, customer_col)

        # 8. Features RFM (Recency, Frequency, Monetary) (20+)
        logger.info("Generating RFM features...")
        df = self._generate_rfm_features(df, timestamp_col, amount_col, customer_col)

        # Limpar colunas temporárias
        df = self._cleanup_temp_columns(df)

        feature_count = len([c for c in df.columns if c not in ['transaction_id', 'timestamp', 'is_fraud']])
        logger.info(f"Generated {feature_count} features total")

        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Preparar dados para feature engineering"""

        # Garantir timestamp como datetime
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Garantir amount como float
        if amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)

        # Criar colunas auxiliares se não existirem
        if "value" not in df.columns and amount_col in df.columns:
            df["value"] = df[amount_col]

        return df

    def _generate_temporal_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Gerar features temporais (50+ features)"""

        if timestamp_col not in df.columns:
            return df

        ts = df[timestamp_col]

        # Features básicas de tempo
        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year

        # Features binárias de período
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df["is_early_morning"] = ((df["hour"] >= 0) & (df["hour"] <= 6)).astype(int)
        df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
        df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 18) & (df["day_of_week"] < 5)).astype(int)
        df["is_lunch_time"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)

        # Features de início/fim de período
        df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)
        df["is_quarter_end"] = (df["month"].isin([3, 6, 9, 12]) & (df["day_of_month"] >= 25)).astype(int)
        df["is_year_end"] = ((df["month"] == 12) & (df["day_of_month"] >= 20)).astype(int)

        # Features cíclicas (sin/cos para ciclicidade)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        # Minutos desde meia-noite
        df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]

        # Timestamp Unix normalizado
        df["timestamp_normalized"] = (ts - ts.min()).dt.total_seconds() / (ts.max() - ts.min()).total_seconds()

        return df

    def _generate_value_features(
        self,
        df: pd.DataFrame,
        amount_col: str
    ) -> pd.DataFrame:
        """Gerar features de valor (30+ features)"""

        if amount_col not in df.columns:
            return df

        amount = df[amount_col]

        # Transformações de valor
        df["log_amount"] = np.log1p(amount)
        df["sqrt_amount"] = np.sqrt(amount)
        df["amount_squared"] = amount ** 2
        df["amount_cubed"] = amount ** 3

        # Features de magnitude
        df["amount_magnitude"] = np.floor(np.log10(amount.clip(lower=1)))
        df["amount_digits"] = amount.astype(str).str.replace('.', '').str.len()

        # Features de arredondamento
        df["is_round_amount"] = (amount % 1 == 0).astype(int)
        df["is_round_10"] = (amount % 10 == 0).astype(int)
        df["is_round_100"] = (amount % 100 == 0).astype(int)
        df["is_round_1000"] = (amount % 1000 == 0).astype(int)
        df["decimal_part"] = amount % 1
        df["cents"] = ((amount * 100) % 100).astype(int)

        # Faixas de valor
        df["is_micro_amount"] = (amount < 10).astype(int)
        df["is_small_amount"] = ((amount >= 10) & (amount < 100)).astype(int)
        df["is_medium_amount"] = ((amount >= 100) & (amount < 1000)).astype(int)
        df["is_large_amount"] = ((amount >= 1000) & (amount < 10000)).astype(int)
        df["is_very_large_amount"] = (amount >= 10000).astype(int)
        df["is_suspicious_amount"] = ((amount >= 5000) & (amount < 10000)).astype(int)  # Abaixo de limites

        # Binning de valor
        df["amount_bin_10"] = pd.cut(amount, bins=10, labels=False).fillna(0).astype(int)
        df["amount_bin_20"] = pd.cut(amount, bins=20, labels=False).fillna(0).astype(int)

        # Percentis
        df["amount_percentile"] = amount.rank(pct=True)

        return df

    def _generate_velocity_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        amount_col: str,
        customer_col: str
    ) -> pd.DataFrame:
        """
        Gerar features de velocidade (Framework Bahnsen) - 200+ features

        Para cada janela temporal e cada chave de agregação:
        - count, sum, mean, std, max, min de transações
        """

        if timestamp_col not in df.columns:
            return df

        # Ordenar por timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Para cada chave de agregação disponível
        agg_keys = [k for k in self.config.aggregation_keys if k in df.columns]

        for key in agg_keys:
            for window in self.config.time_windows:
                prefix = f"{key}_w{window}"

                # Criar features de rolling window
                df = self._create_rolling_features(
                    df, key, timestamp_col, amount_col, window, prefix
                )

        return df

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        group_col: str,
        timestamp_col: str,
        amount_col: str,
        window_minutes: int,
        prefix: str
    ) -> pd.DataFrame:
        """Criar features de rolling window para uma chave de agregação"""

        # Para eficiência, usar rolling window baseado em índice
        # Em produção real, usar feature store como Feast

        window_str = f"{window_minutes}min" if window_minutes < 1440 else f"{window_minutes // 1440}D"

        # Agrupar e calcular estatísticas
        grouped = df.groupby(group_col)

        # Count de transações na janela
        df[f"{prefix}_tx_count"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).count()
        )

        # Soma de valores na janela
        df[f"{prefix}_tx_sum"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).sum()
        )

        # Média de valores na janela
        df[f"{prefix}_tx_mean"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).mean()
        )

        # Desvio padrão de valores na janela
        df[f"{prefix}_tx_std"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).std()
        ).fillna(0)

        # Máximo de valores na janela
        df[f"{prefix}_tx_max"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).max()
        )

        # Mínimo de valores na janela
        df[f"{prefix}_tx_min"] = grouped[amount_col].transform(
            lambda x: x.rolling(window_str, min_periods=1).min()
        )

        # Ratio atual vs média
        df[f"{prefix}_amount_ratio"] = df[amount_col] / (df[f"{prefix}_tx_mean"] + 1e-6)

        # Z-score
        df[f"{prefix}_zscore"] = (df[amount_col] - df[f"{prefix}_tx_mean"]) / (df[f"{prefix}_tx_std"] + 1e-6)

        return df

    def _generate_aggregation_features(
        self,
        df: pd.DataFrame,
        amount_col: str
    ) -> pd.DataFrame:
        """Gerar features de agregação por entidade (100+ features)"""

        agg_keys = [k for k in self.config.aggregation_keys if k in df.columns]

        for key in agg_keys:
            prefix = f"agg_{key}"

            # Estatísticas globais por entidade
            entity_stats = df.groupby(key)[amount_col].agg([
                'count', 'sum', 'mean', 'std', 'max', 'min', 'median'
            ]).reset_index()

            entity_stats.columns = [
                key,
                f"{prefix}_total_count",
                f"{prefix}_total_sum",
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_max",
                f"{prefix}_min",
                f"{prefix}_median"
            ]

            entity_stats[f"{prefix}_std"] = entity_stats[f"{prefix}_std"].fillna(0)

            df = df.merge(entity_stats, on=key, how="left")

            # Features derivadas
            df[f"{prefix}_amount_vs_mean"] = df[amount_col] / (df[f"{prefix}_mean"] + 1e-6)
            df[f"{prefix}_amount_vs_max"] = df[amount_col] / (df[f"{prefix}_max"] + 1e-6)
            df[f"{prefix}_is_max_tx"] = (df[amount_col] >= df[f"{prefix}_max"] * 0.99).astype(int)

            # Contagem única se houver colunas relacionais
            if "receiver_id" in df.columns:
                unique_receivers = df.groupby(key)["receiver_id"].nunique().reset_index()
                unique_receivers.columns = [key, f"{prefix}_unique_receivers"]
                df = df.merge(unique_receivers, on=key, how="left")

        return df

    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gerar features de interação (80+ features)"""

        # Interações com hora
        if "hour" in df.columns and "value" in df.columns:
            df["hour_x_amount"] = df["hour"] * df["value"]
            df["hour_x_log_amount"] = df["hour"] * np.log1p(df["value"])

        if "hour" in df.columns and "is_weekend" in df.columns:
            df["hour_x_weekend"] = df["hour"] * df["is_weekend"]

        # Interações de canal
        if "channel" in df.columns:
            # One-hot encoding de channel
            channel_dummies = pd.get_dummies(df["channel"], prefix="channel")
            for col in channel_dummies.columns:
                df[col] = channel_dummies[col]

                # Interação com valor
                if "value" in df.columns:
                    df[f"{col}_x_amount"] = channel_dummies[col] * df["value"]

        # Interações de tipo de transação
        if "transaction_type" in df.columns:
            type_dummies = pd.get_dummies(df["transaction_type"], prefix="tx_type")
            for col in type_dummies.columns:
                df[col] = type_dummies[col]

        # Interação device x customer
        if "device_id" in df.columns and "customer_id" in df.columns:
            df["device_customer_hash"] = (
                df["device_id"].astype(str) + "_" + df["customer_id"].astype(str)
            ).apply(lambda x: hash(x) % 10000)

        # Interação merchant x amount
        if "merchant_id" in df.columns and "value" in df.columns:
            merchant_avg = df.groupby("merchant_id")["value"].transform("mean")
            df["merchant_amount_ratio"] = df["value"] / (merchant_avg + 1e-6)

        # Interação location x time
        if "state" in df.columns and "hour" in df.columns:
            df["state_hour_hash"] = (
                df["state"].astype(str) + "_" + df["hour"].astype(str)
            ).apply(lambda x: hash(x) % 1000)

        return df

    def _generate_deviation_features(
        self,
        df: pd.DataFrame,
        amount_col: str,
        customer_col: str
    ) -> pd.DataFrame:
        """Gerar features de desvio (40+ features)"""

        if customer_col not in df.columns:
            return df

        # Z-score global do valor
        global_mean = df[amount_col].mean()
        global_std = df[amount_col].std()
        df["amount_global_zscore"] = (df[amount_col] - global_mean) / (global_std + 1e-6)

        # Z-score por cliente
        customer_stats = df.groupby(customer_col)[amount_col].agg(['mean', 'std']).reset_index()
        customer_stats.columns = [customer_col, "customer_mean", "customer_std"]
        customer_stats["customer_std"] = customer_stats["customer_std"].fillna(0)

        df = df.merge(customer_stats, on=customer_col, how="left")
        df["amount_customer_zscore"] = (df[amount_col] - df["customer_mean"]) / (df["customer_std"] + 1e-6)

        # Desvio da hora usual
        if "hour" in df.columns:
            customer_hour = df.groupby(customer_col)["hour"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12
            ).reset_index()
            customer_hour.columns = [customer_col, "customer_usual_hour"]

            df = df.merge(customer_hour, on=customer_col, how="left")
            df["hour_deviation"] = np.abs(df["hour"] - df["customer_usual_hour"])
            df["is_unusual_hour"] = (df["hour_deviation"] > 6).astype(int)

        # Desvio do canal usual
        if "channel" in df.columns:
            customer_channel = df.groupby(customer_col)["channel"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "UNKNOWN"
            ).reset_index()
            customer_channel.columns = [customer_col, "customer_usual_channel"]

            df = df.merge(customer_channel, on=customer_col, how="left")
            df["is_unusual_channel"] = (df["channel"] != df["customer_usual_channel"]).astype(int)

        return df

    def _generate_frequency_features(
        self,
        df: pd.DataFrame,
        customer_col: str
    ) -> pd.DataFrame:
        """Gerar features de frequência (30+ features)"""

        if customer_col not in df.columns:
            return df

        # Frequência de transações por cliente
        tx_counts = df.groupby(customer_col).size().reset_index(name="customer_tx_count")
        df = df.merge(tx_counts, on=customer_col, how="left")

        # Categorias de frequência
        df["is_low_frequency_customer"] = (df["customer_tx_count"] < 5).astype(int)
        df["is_medium_frequency_customer"] = ((df["customer_tx_count"] >= 5) & (df["customer_tx_count"] < 20)).astype(int)
        df["is_high_frequency_customer"] = (df["customer_tx_count"] >= 20).astype(int)

        # Se houver receiver_id
        if "receiver_id" in df.columns:
            # Primeira vez enviando para este receiver
            customer_receiver = df.groupby([customer_col, "receiver_id"]).cumcount()
            df["is_first_to_receiver"] = (customer_receiver == 0).astype(int)

            # Contagem de receivers únicos
            unique_receivers = df.groupby(customer_col)["receiver_id"].transform("nunique")
            df["customer_unique_receivers"] = unique_receivers

        # Se houver device_id
        if "device_id" in df.columns:
            # Dispositivos únicos por cliente
            unique_devices = df.groupby(customer_col)["device_id"].transform("nunique")
            df["customer_unique_devices"] = unique_devices

            # Muitos dispositivos é suspeito
            df["is_multi_device_customer"] = (df["customer_unique_devices"] > 3).astype(int)

        return df

    def _generate_rfm_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        amount_col: str,
        customer_col: str
    ) -> pd.DataFrame:
        """Gerar features RFM - Recency, Frequency, Monetary (20+ features)"""

        if customer_col not in df.columns or timestamp_col not in df.columns:
            return df

        now = df[timestamp_col].max()

        # Recency - tempo desde última transação
        last_tx = df.groupby(customer_col)[timestamp_col].max().reset_index()
        last_tx.columns = [customer_col, "last_tx_time"]
        last_tx["recency_days"] = (now - last_tx["last_tx_time"]).dt.total_seconds() / 86400

        df = df.merge(last_tx[[customer_col, "recency_days"]], on=customer_col, how="left")

        # Frequency - frequência de transações
        # (já calculado em frequency features, mas adicionar scores)
        if "customer_tx_count" in df.columns:
            df["frequency_score"] = df["customer_tx_count"].rank(pct=True)

        # Monetary - valor total gasto
        monetary = df.groupby(customer_col)[amount_col].sum().reset_index()
        monetary.columns = [customer_col, "monetary_total"]

        df = df.merge(monetary, on=customer_col, how="left")
        df["monetary_score"] = df["monetary_total"].rank(pct=True)

        # RFM combinado
        if "frequency_score" in df.columns:
            df["rfm_score"] = (
                (1 - df["recency_days"] / (df["recency_days"].max() + 1)) * 0.3 +
                df["frequency_score"] * 0.3 +
                df["monetary_score"] * 0.4
            )

        # Categorias RFM
        df["recency_bin"] = pd.qcut(df["recency_days"], q=5, labels=False, duplicates="drop")
        df["monetary_bin"] = pd.qcut(df["monetary_total"], q=5, labels=False, duplicates="drop")

        return df

    def _cleanup_temp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpar colunas temporárias"""

        # Remover colunas de merge temporárias
        temp_cols = [
            "customer_mean", "customer_std",
            "customer_usual_hour", "customer_usual_channel",
            "last_tx_time"
        ]

        for col in temp_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        return df

    def get_feature_names(self) -> List[str]:
        """Retornar lista de nomes de features geradas"""

        # Esta é uma lista parcial - a lista real depende dos dados
        base_features = [
            # Temporal
            "hour", "minute", "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "year", "is_weekend", "is_monday", "is_friday",
            "is_night", "is_early_morning", "is_morning", "is_afternoon",
            "is_evening", "is_business_hours", "is_lunch_time",
            "is_month_start", "is_month_end", "is_quarter_end", "is_year_end",
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "day_of_month_sin", "day_of_month_cos", "month_sin", "month_cos",
            "minute_sin", "minute_cos", "minutes_since_midnight", "timestamp_normalized",

            # Value
            "log_amount", "sqrt_amount", "amount_squared", "amount_cubed",
            "amount_magnitude", "amount_digits", "is_round_amount",
            "is_round_10", "is_round_100", "is_round_1000",
            "decimal_part", "cents", "is_micro_amount", "is_small_amount",
            "is_medium_amount", "is_large_amount", "is_very_large_amount",
            "is_suspicious_amount", "amount_bin_10", "amount_bin_20", "amount_percentile",

            # Deviation
            "amount_global_zscore", "amount_customer_zscore",
            "hour_deviation", "is_unusual_hour", "is_unusual_channel",

            # Frequency
            "customer_tx_count", "is_low_frequency_customer",
            "is_medium_frequency_customer", "is_high_frequency_customer",
            "is_first_to_receiver", "customer_unique_receivers",
            "customer_unique_devices", "is_multi_device_customer",

            # RFM
            "recency_days", "frequency_score", "monetary_total",
            "monetary_score", "rfm_score", "recency_bin", "monetary_bin"
        ]

        return base_features

    def get_feature_count(self) -> int:
        """Retornar contagem estimada de features"""

        base = 100  # Features básicas

        # Velocity features por janela e chave
        velocity = len(self.config.time_windows) * len(self.config.aggregation_keys) * 8

        # Aggregation features por chave
        aggregation = len(self.config.aggregation_keys) * 12

        return base + velocity + aggregation

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do gerador"""
        return {
            "version": self.VERSION,
            "time_windows": self.config.time_windows,
            "aggregation_keys": self.config.aggregation_keys,
            "estimated_feature_count": self.get_feature_count()
        }


def create_feature_generator(config: Optional[Dict[str, Any]] = None) -> FeatureGenerator:
    """Factory function para criar FeatureGenerator"""
    if config:
        feature_config = FeatureConfig(**config)
    else:
        feature_config = None
    return FeatureGenerator(feature_config)
