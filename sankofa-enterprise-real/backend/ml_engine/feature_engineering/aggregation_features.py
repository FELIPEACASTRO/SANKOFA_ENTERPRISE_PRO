"""
Sankofa Enterprise Pro - Aggregation Feature Generator
Features de agregação por entidade

Baseado em:
- IEEE-CIS winning solutions
- Feedzai AutoML aggregations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy.stats import entropy, skew, kurtosis
import logging

logger = logging.getLogger(__name__)


class AggregationFeatureGenerator:
    """
    Gerador de Features de Agregação por Entidade

    Para cada entidade (customer, device, merchant, etc):
    - Estatísticas de valor
    - Contagens e frequências
    - Distribuições e entropia
    - Métricas RFM

    Gera 100+ features
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.entity_stats_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"AggregationFeatureGenerator v{self.VERSION} initialized")

    def generate(
        self,
        df: pd.DataFrame,
        entity_cols: Optional[List[str]] = None,
        amount_col: str = "amount",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Gerar features de agregação

        Args:
            df: DataFrame com transações
            entity_cols: Colunas de entidade para agregar
            amount_col: Coluna de valor
            timestamp_col: Coluna de timestamp

        Returns:
            DataFrame com features de agregação
        """
        df = df.copy()

        # Entidades padrão
        default_entities = [
            "customer_id", "device_id", "ip_address",
            "merchant_id", "receiver_id", "channel",
            "card_hash", "email_domain"
        ]

        entities = entity_cols or default_entities
        available_entities = [e for e in entities if e in df.columns]

        logger.info(f"Generating aggregation features for {len(available_entities)} entities...")

        for entity in available_entities:
            logger.info(f"Processing entity: {entity}")
            df = self._generate_entity_features(
                df, entity, amount_col, timestamp_col
            )

        # Features cross-entity
        df = self._generate_cross_entity_features(df, available_entities, amount_col)

        return df

    def _generate_entity_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        amount_col: str,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Gerar features para uma entidade específica"""

        prefix = f"agg_{entity_col}"

        # 1. Estatísticas básicas de valor
        df = self._add_value_stats(df, entity_col, amount_col, prefix)

        # 2. Contagens e frequências
        df = self._add_count_features(df, entity_col, amount_col, prefix)

        # 3. Features de distribuição
        df = self._add_distribution_features(df, entity_col, amount_col, prefix)

        # 4. Features de recência
        if timestamp_col in df.columns:
            df = self._add_recency_features(df, entity_col, timestamp_col, prefix)

        # 5. Features de relação com outras entidades
        df = self._add_relationship_features(df, entity_col, prefix)

        return df

    def _add_value_stats(
        self,
        df: pd.DataFrame,
        entity_col: str,
        amount_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Adicionar estatísticas de valor por entidade"""

        if amount_col not in df.columns:
            return df

        # Calcular estatísticas
        stats = df.groupby(entity_col)[amount_col].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max', 'median'
        ]).reset_index()

        stats.columns = [
            entity_col,
            f"{prefix}_count",
            f"{prefix}_sum",
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_min",
            f"{prefix}_max",
            f"{prefix}_median"
        ]

        stats[f"{prefix}_std"] = stats[f"{prefix}_std"].fillna(0)

        # Merge
        df = df.merge(stats, on=entity_col, how="left")

        # Features derivadas
        df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
        df[f"{prefix}_cv"] = df[f"{prefix}_std"] / (df[f"{prefix}_mean"] + 1e-6)
        df[f"{prefix}_amount_vs_mean"] = df[amount_col] / (df[f"{prefix}_mean"] + 1e-6)
        df[f"{prefix}_amount_vs_max"] = df[amount_col] / (df[f"{prefix}_max"] + 1e-6)
        df[f"{prefix}_amount_vs_median"] = df[amount_col] / (df[f"{prefix}_median"] + 1e-6)

        # Z-score
        df[f"{prefix}_zscore"] = (
            df[amount_col] - df[f"{prefix}_mean"]
        ) / (df[f"{prefix}_std"] + 1e-6)

        # É o máximo?
        df[f"{prefix}_is_max_tx"] = (
            df[amount_col] >= df[f"{prefix}_max"] * 0.99
        ).astype(int)

        # É outlier?
        df[f"{prefix}_is_outlier"] = (
            np.abs(df[f"{prefix}_zscore"]) > 2
        ).astype(int)

        return df

    def _add_count_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        amount_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Adicionar features de contagem"""

        # Percentil do count
        if f"{prefix}_count" in df.columns:
            df[f"{prefix}_count_percentile"] = df[f"{prefix}_count"].rank(pct=True)

            # Categorias de frequência
            df[f"{prefix}_is_low_freq"] = (df[f"{prefix}_count"] < 5).astype(int)
            df[f"{prefix}_is_high_freq"] = (df[f"{prefix}_count"] > 50).astype(int)

        # Contagens únicas de relacionamentos
        for related_col in ["customer_id", "device_id", "merchant_id", "receiver_id"]:
            if related_col in df.columns and related_col != entity_col:
                unique_counts = df.groupby(entity_col)[related_col].nunique().reset_index()
                unique_counts.columns = [entity_col, f"{prefix}_unique_{related_col}s"]
                df = df.merge(unique_counts, on=entity_col, how="left")

        return df

    def _add_distribution_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        amount_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Adicionar features de distribuição"""

        if amount_col not in df.columns:
            return df

        # Quartis e percentis
        quartiles = df.groupby(entity_col)[amount_col].quantile([0.25, 0.75]).unstack()
        quartiles.columns = [f"{prefix}_q25", f"{prefix}_q75"]
        quartiles = quartiles.reset_index()

        df = df.merge(quartiles, on=entity_col, how="left")

        # IQR
        df[f"{prefix}_iqr"] = df[f"{prefix}_q75"] - df[f"{prefix}_q25"]

        # Skewness e Kurtosis por grupo
        def safe_skew(x):
            if len(x) < 3:
                return 0
            return skew(x, nan_policy='omit')

        def safe_kurtosis(x):
            if len(x) < 4:
                return 0
            return kurtosis(x, nan_policy='omit')

        skew_kurt = df.groupby(entity_col)[amount_col].agg([
            ('skew', safe_skew),
            ('kurtosis', safe_kurtosis)
        ]).reset_index()

        skew_kurt.columns = [entity_col, f"{prefix}_skew", f"{prefix}_kurtosis"]

        df = df.merge(skew_kurt, on=entity_col, how="left")

        return df

    def _add_recency_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Adicionar features de recência"""

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Primeira e última transação
        time_stats = df.groupby(entity_col)[timestamp_col].agg(['min', 'max']).reset_index()
        time_stats.columns = [entity_col, f"{prefix}_first_tx", f"{prefix}_last_tx"]

        df = df.merge(time_stats, on=entity_col, how="left")

        # Tempo desde primeira transação (em dias)
        df[f"{prefix}_account_age_days"] = (
            df[timestamp_col] - df[f"{prefix}_first_tx"]
        ).dt.total_seconds() / 86400

        # Tempo desde última transação anterior (em minutos)
        df = df.sort_values([entity_col, timestamp_col])
        df[f"{prefix}_time_since_last"] = df.groupby(entity_col)[timestamp_col].diff().dt.total_seconds() / 60
        df[f"{prefix}_time_since_last"] = df[f"{prefix}_time_since_last"].fillna(999999)

        # Média de tempo entre transações
        avg_time_between = df.groupby(entity_col)[f"{prefix}_time_since_last"].mean().reset_index()
        avg_time_between.columns = [entity_col, f"{prefix}_avg_time_between"]

        df = df.merge(avg_time_between, on=entity_col, how="left")

        # Transação rápida comparada ao normal
        df[f"{prefix}_is_rapid_tx"] = (
            df[f"{prefix}_time_since_last"] < df[f"{prefix}_avg_time_between"] * 0.5
        ).astype(int)

        # Limpar colunas temporárias
        df = df.drop(columns=[f"{prefix}_first_tx", f"{prefix}_last_tx"], errors='ignore')

        return df

    def _add_relationship_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """Adicionar features de relacionamento entre entidades"""

        # Diversidade de relacionamentos
        relationship_cols = ["customer_id", "device_id", "merchant_id", "receiver_id", "channel"]

        for rel_col in relationship_cols:
            if rel_col in df.columns and rel_col != entity_col:
                # Entropia de distribuição
                def calc_entropy(x):
                    value_counts = x.value_counts(normalize=True)
                    return entropy(value_counts) if len(value_counts) > 1 else 0

                ent = df.groupby(entity_col)[rel_col].agg(calc_entropy).reset_index()
                ent.columns = [entity_col, f"{prefix}_{rel_col}_entropy"]

                df = df.merge(ent, on=entity_col, how="left")

        return df

    def _generate_cross_entity_features(
        self,
        df: pd.DataFrame,
        entities: List[str],
        amount_col: str
    ) -> pd.DataFrame:
        """Gerar features cruzadas entre entidades"""

        # Combinações de entidades
        if "customer_id" in entities and "device_id" in entities:
            # Clientes por dispositivo
            customers_per_device = df.groupby("device_id")["customer_id"].nunique().reset_index()
            customers_per_device.columns = ["device_id", "customers_per_device"]

            df = df.merge(customers_per_device, on="device_id", how="left")

            # Dispositivo compartilhado
            df["is_shared_device"] = (df["customers_per_device"] > 1).astype(int)

        if "customer_id" in entities and "ip_address" in entities and "ip_address" in df.columns:
            # Clientes por IP
            customers_per_ip = df.groupby("ip_address")["customer_id"].nunique().reset_index()
            customers_per_ip.columns = ["ip_address", "customers_per_ip"]

            df = df.merge(customers_per_ip, on="ip_address", how="left")

            # IP compartilhado
            df["is_shared_ip"] = (df["customers_per_ip"] > 1).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Retornar lista de nomes de features"""
        # Lista base - a real depende das entidades disponíveis
        base_suffixes = [
            "_count", "_sum", "_mean", "_std", "_min", "_max", "_median",
            "_range", "_cv", "_amount_vs_mean", "_amount_vs_max",
            "_amount_vs_median", "_zscore", "_is_max_tx", "_is_outlier",
            "_count_percentile", "_is_low_freq", "_is_high_freq",
            "_q25", "_q75", "_iqr", "_skew", "_kurtosis",
            "_account_age_days", "_time_since_last", "_avg_time_between",
            "_is_rapid_tx"
        ]

        return [f"agg_entity{suffix}" for suffix in base_suffixes]

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do gerador"""
        return {
            "version": self.VERSION,
            "cached_entities": list(self.entity_stats_cache.keys())
        }


def create_aggregation_generator(config: Optional[Dict[str, Any]] = None) -> AggregationFeatureGenerator:
    """Factory function para criar AggregationFeatureGenerator"""
    return AggregationFeatureGenerator(config)
