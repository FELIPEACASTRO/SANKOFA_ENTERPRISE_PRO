"""
Sankofa Enterprise Pro - Velocity Feature Generator
Features de velocidade baseadas no Framework Bahnsen

Baseado em:
- Bahnsen et al. "Feature Engineering for Fraud Detection" (2016)
- IEEE-CIS winning solutions
- Feedzai velocity features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class VelocityFeatureGenerator:
    """
    Gerador de Features de Velocidade (Framework Bahnsen)

    Para cada combinação de (chave de agregação, janela temporal):
    - Contagem de transações
    - Soma de valores
    - Média, std, max, min
    - Valores únicos (nunique)
    - Ratios e z-scores

    Gera 200+ features
    """

    VERSION = "1.0.0"

    # Janelas temporais padrão em minutos
    DEFAULT_WINDOWS = [1, 5, 15, 30, 60, 120, 360, 720, 1440, 4320, 10080]
    # 1min, 5min, 15min, 30min, 1h, 2h, 6h, 12h, 1d, 3d, 7d

    # Operações de agregação
    AGG_OPERATIONS = {
        "count": lambda x: x.count(),
        "sum": lambda x: x.sum(),
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
        "median": lambda x: x.median(),
        "nunique": lambda x: x.nunique()
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.windows = self.config.get("windows", self.DEFAULT_WINDOWS)
        self.agg_keys = self.config.get("agg_keys", [
            "customer_id", "device_id", "ip_address",
            "merchant_id", "receiver_id", "channel"
        ])

        logger.info(f"VelocityFeatureGenerator v{self.VERSION} initialized")
        logger.info(f"Windows: {self.windows}")
        logger.info(f"Aggregation keys: {self.agg_keys}")

    def generate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        agg_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Gerar features de velocidade

        Args:
            df: DataFrame com transações
            timestamp_col: Coluna de timestamp
            amount_col: Coluna de valor
            agg_keys: Chaves de agregação (None = usar default)

        Returns:
            DataFrame com features de velocidade
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Column {timestamp_col} not found")
            return df

        df = df.copy()

        # Garantir timestamp como datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Ordenar por timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Usar chaves disponíveis
        keys = agg_keys or self.agg_keys
        available_keys = [k for k in keys if k in df.columns]

        logger.info(f"Generating velocity features for {len(available_keys)} keys...")

        # Para cada chave de agregação
        for key in available_keys:
            logger.info(f"Processing key: {key}")
            df = self._generate_velocity_for_key(
                df, key, timestamp_col, amount_col
            )

        # Gerar features de velocity cruzadas
        df = self._generate_cross_velocity_features(df, available_keys, amount_col)

        return df

    def _generate_velocity_for_key(
        self,
        df: pd.DataFrame,
        key: str,
        timestamp_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Gerar features de velocidade para uma chave específica"""

        for window in self.windows:
            prefix = f"vel_{key}_w{window}"

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
        """Criar features de rolling window"""

        # Definir janela de tempo
        window_td = timedelta(minutes=window_minutes)

        # Para performance, agrupar e calcular features
        # Nota: em produção real, usar feature store

        # Método otimizado usando rolling com index de tempo
        df = df.set_index(timestamp_col, drop=False)

        # Agrupar por chave
        grouped = df.groupby(group_col)

        # Features de contagem
        df[f"{prefix}_count"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).count()
        )

        # Features de soma
        df[f"{prefix}_sum"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).sum()
        )

        # Features de média
        df[f"{prefix}_mean"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).mean()
        )

        # Features de desvio padrão
        df[f"{prefix}_std"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).std()
        ).fillna(0)

        # Features de máximo
        df[f"{prefix}_max"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).max()
        )

        # Features de mínimo
        df[f"{prefix}_min"] = grouped[amount_col].transform(
            lambda x: x.rolling(f"{window_minutes}min", min_periods=1).min()
        )

        # Resetar index
        df = df.reset_index(drop=True)

        # Features derivadas
        df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
        df[f"{prefix}_amount_ratio"] = df[amount_col] / (df[f"{prefix}_mean"] + 1e-6)
        df[f"{prefix}_zscore"] = (df[amount_col] - df[f"{prefix}_mean"]) / (df[f"{prefix}_std"] + 1e-6)
        df[f"{prefix}_is_max"] = (df[amount_col] >= df[f"{prefix}_max"] * 0.99).astype(int)

        # Coeficiente de variação
        df[f"{prefix}_cv"] = df[f"{prefix}_std"] / (df[f"{prefix}_mean"] + 1e-6)

        return df

    def _generate_cross_velocity_features(
        self,
        df: pd.DataFrame,
        keys: List[str],
        amount_col: str
    ) -> pd.DataFrame:
        """Gerar features de velocidade cruzadas entre diferentes janelas"""

        # Ratios entre janelas para cada chave
        for key in keys:
            # Ratio janela curta vs longa
            short_window = self.windows[0] if self.windows else 1
            long_window = self.windows[-1] if self.windows else 10080

            short_col = f"vel_{key}_w{short_window}_count"
            long_col = f"vel_{key}_w{long_window}_count"

            if short_col in df.columns and long_col in df.columns:
                df[f"vel_{key}_short_long_ratio"] = df[short_col] / (df[long_col] + 1e-6)

            # Velocidade média (transações/hora)
            if f"vel_{key}_w60_count" in df.columns:
                df[f"vel_{key}_tx_per_hour"] = df[f"vel_{key}_w60_count"]

            # Aceleração (mudança na velocidade)
            if f"vel_{key}_w5_count" in df.columns and f"vel_{key}_w30_count" in df.columns:
                df[f"vel_{key}_acceleration"] = (
                    df[f"vel_{key}_w5_count"] * 6  # Normalizar para 30min
                ) / (df[f"vel_{key}_w30_count"] + 1e-6)

        return df

    def generate_historical_velocity(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        key: str,
        timestamp_col: str = "timestamp",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Gerar features de velocidade usando histórico separado

        Útil para scoring em tempo real onde o histórico
        está em outra fonte
        """
        # Implementação para feature store em produção
        # Por enquanto, concatenar e calcular
        combined = pd.concat([history_df, df], ignore_index=True)
        combined = combined.sort_values(timestamp_col).reset_index(drop=True)

        # Gerar features
        combined = self._generate_velocity_for_key(
            combined, key, timestamp_col, amount_col
        )

        # Retornar apenas as novas transações
        return combined.tail(len(df)).reset_index(drop=True)

    def get_feature_names(self) -> List[str]:
        """Retornar lista de nomes de features de velocidade"""
        features = []

        for key in self.agg_keys:
            for window in self.windows:
                prefix = f"vel_{key}_w{window}"
                features.extend([
                    f"{prefix}_count",
                    f"{prefix}_sum",
                    f"{prefix}_mean",
                    f"{prefix}_std",
                    f"{prefix}_max",
                    f"{prefix}_min",
                    f"{prefix}_range",
                    f"{prefix}_amount_ratio",
                    f"{prefix}_zscore",
                    f"{prefix}_is_max",
                    f"{prefix}_cv"
                ])

            # Features cruzadas
            features.extend([
                f"vel_{key}_short_long_ratio",
                f"vel_{key}_tx_per_hour",
                f"vel_{key}_acceleration"
            ])

        return features

    def get_feature_count(self) -> int:
        """Retornar contagem de features"""
        # 11 features por janela + 3 features cruzadas por chave
        return len(self.agg_keys) * (len(self.windows) * 11 + 3)

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do gerador"""
        return {
            "version": self.VERSION,
            "windows": self.windows,
            "agg_keys": self.agg_keys,
            "feature_count": self.get_feature_count()
        }


def create_velocity_generator(config: Optional[Dict[str, Any]] = None) -> VelocityFeatureGenerator:
    """Factory function para criar VelocityFeatureGenerator"""
    return VelocityFeatureGenerator(config)
