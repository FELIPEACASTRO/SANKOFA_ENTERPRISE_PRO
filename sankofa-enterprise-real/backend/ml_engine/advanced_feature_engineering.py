"""
Advanced Feature Engineering - Cria features avançadas para detecção de fraude
Inclui: Entropia de localização, padrões comportamentais, análise de velocidade
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.stats import entropy as scipy_entropy
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineering:
    """
    Engenharia de features avançada para detecção de fraude.
    """

    def __init__(self):
        self.client_stats = None
        self.device_stats = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features avançadas a partir dos dados brutos.

        Args:
            df: DataFrame com dados de transações

        Returns:
            DataFrame com features avançadas
        """
        df = df.copy()

        # 1. Features Temporais
        df = self._create_temporal_features(df)

        # 2. Features de Valor
        df = self._create_value_features(df)

        # 3. Features de Comportamento do Cliente
        df = self._create_client_behavior_features(df)

        # 4. Features de Dispositivo
        df = self._create_device_features(df)

        # 5. Features de Localização
        df = self._create_location_features(df)

        # 6. Features de Canal e Tipo
        df = self._create_channel_type_features(df)

        # 7. Features de Velocidade
        df = self._create_velocity_features(df)

        # 8. Features de Entropia de Localização (NOVA)
        df = self._create_location_entropy_features(df)

        # 9. Features de Padrão de Transação (NOVA)
        df = self._create_transaction_pattern_features(df)

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
            df["is_business_hours"] = df["hour"].between(9, 18).astype(int)
            df["is_early_morning"] = (df["hour"] <= 6).astype(int)
        elif "hour" in df.columns:
            df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
            df["is_business_hours"] = df["hour"].between(9, 18).astype(int)
            df["is_early_morning"] = (df["hour"] <= 6).astype(int)
            df["day_of_week"] = 2
            df["is_weekend"] = 0
        else:
            df["hour"] = 12
            df["day_of_week"] = 2
            df["is_weekend"] = 0
            df["is_night"] = 0
            df["is_business_hours"] = 1
            df["is_early_morning"] = 0

        return df

    def _create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de valor."""
        if "value" in df.columns:
            df["log_value"] = np.log1p(df["value"])
            df["value_rounded"] = (df["value"] % 1 == 0).astype(int)
            df["is_high_value"] = (df["value"] > 5000).astype(int)
            df["is_very_high_value"] = (df["value"] > 10000).astype(int)
        elif "amount" in df.columns:
            df["value"] = df["amount"]
            df["log_value"] = np.log1p(df["amount"])
            df["value_rounded"] = (df["amount"] % 1 == 0).astype(int)
            df["is_high_value"] = (df["amount"] > 5000).astype(int)
            df["is_very_high_value"] = (df["amount"] > 10000).astype(int)

        return df

    def _create_client_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de comportamento do cliente."""
        if "client_cpf" in df.columns and "value" in df.columns:
            # Calcular estatísticas por cliente
            client_stats = (
                df.groupby("client_cpf")
                .agg({"value": ["mean", "std", "count", "max"]})
                .reset_index()
            )

            client_stats.columns = [
                "client_cpf",
                "avg_value",
                "std_value",
                "num_transactions",
                "max_value",
            ]
            client_stats["std_value"] = client_stats["std_value"].fillna(0)

            df = df.merge(client_stats, on="client_cpf", how="left")

            # Desvio do comportamento normal
            df["value_deviation"] = (df["value"] - df["avg_value"]) / (df["std_value"] + 1e-10)
            df["is_new_client"] = (df["num_transactions"] < 5).astype(int)
            df["is_max_value"] = (df["value"] >= df["max_value"] * 0.9).astype(int)

            self.client_stats = client_stats

        return df

    def _create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de dispositivo."""
        if "device_id" in df.columns:
            # Estatísticas por dispositivo
            device_stats = (
                df.groupby("device_id")
                .agg(
                    {
                        "client_cpf": "nunique" if "client_cpf" in df.columns else "count",
                        "value": "sum" if "value" in df.columns else "count",
                    }
                )
                .reset_index()
            )

            device_stats.columns = ["device_id", "num_clients_per_device", "total_value_device"]

            df = df.merge(device_stats, on="device_id", how="left")

            # Dispositivo compartilhado é suspeito
            df["is_shared_device"] = (df["num_clients_per_device"] > 1).astype(int)
            df["is_new_device"] = (
                df["device_id"].str.contains("new", case=False, na=False).astype(int)
            )

            self.device_stats = device_stats

        return df

    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de localização."""
        if "state" in df.columns:
            high_risk_states = ["SP", "RJ", "MG"]
            df["is_high_risk_state"] = df["state"].isin(high_risk_states).astype(int)

        if "country" in df.columns:
            df["is_brazil"] = (df["country"] == "BR").astype(int)

        return df

    def _create_channel_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de canal e tipo de transação."""
        if "channel" in df.columns:
            df["is_mobile"] = (df["channel"] == "MOBILE").astype(int)
            df["is_web"] = (df["channel"] == "WEB").astype(int)
            df["is_atm"] = (df["channel"] == "ATM").astype(int)

        if "transaction_type" in df.columns:
            df["is_pix"] = (df["transaction_type"] == "PIX").astype(int)
            df["is_boleto"] = (df["transaction_type"] == "BOLETO").astype(int)
            df["is_credit"] = (df["transaction_type"] == "CREDITO").astype(int)

        return df

    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de velocidade (tempo entre transações)."""
        if "client_cpf" in df.columns and "timestamp" in df.columns:
            # Ordenar por cliente e timestamp
            df = df.sort_values(["client_cpf", "timestamp"])

            # Tempo desde última transação
            df["time_since_last_transaction"] = (
                df.groupby("client_cpf")["timestamp"].diff().dt.total_seconds().fillna(999999)
            )

            # Transações muito rápidas são suspeitas
            df["is_rapid_transaction"] = (df["time_since_last_transaction"] < 60).astype(int)
            df["is_very_rapid_transaction"] = (df["time_since_last_transaction"] < 10).astype(int)

        return df

    def _create_location_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de entropia de localização.

        Entropia alta = cliente transaciona em muitos locais diferentes (suspeito)
        Entropia baixa = cliente transaciona em poucos locais (normal)

        Baseado em: AIForge - Feature Engineering Best Practices
        """
        location_col = None
        for col in ["state", "city", "merchant_id", "location_id"]:
            if col in df.columns:
                location_col = col
                break

        client_col = None
        for col in ["client_cpf", "customer_id", "user_id"]:
            if col in df.columns:
                client_col = col
                break

        if location_col and client_col:

            def calculate_entropy(locations):
                """Calcula entropia de Shannon para distribuição de locais"""
                if len(locations) <= 1:
                    return 0.0
                value_counts = locations.value_counts(normalize=True)
                return float(scipy_entropy(value_counts))

            location_entropy = df.groupby(client_col)[location_col].transform(calculate_entropy)
            df["location_entropy"] = location_entropy.fillna(0)

            unique_locations = df.groupby(client_col)[location_col].transform("nunique")
            df["unique_locations_count"] = unique_locations.fillna(1)

            df["is_diverse_locations"] = (df["location_entropy"] > 1.5).astype(int)

            df["location_entropy_normalized"] = df["location_entropy"] / (
                np.log(df["unique_locations_count"] + 1) + 1e-6
            )
        else:
            df["location_entropy"] = 0.0
            df["unique_locations_count"] = 1
            df["is_diverse_locations"] = 0
            df["location_entropy_normalized"] = 0.0

        return df

    def _create_transaction_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de padrões de transação.

        Detecta anomalias em padrões de hora, valor e frequência.

        Baseado em: AIForge - Behavioral Analytics
        """
        client_col = None
        for col in ["client_cpf", "customer_id", "user_id"]:
            if col in df.columns:
                client_col = col
                break

        value_col = (
            "value" if "value" in df.columns else "amount" if "amount" in df.columns else None
        )

        if client_col and value_col:
            client_value_stats = df.groupby(client_col)[value_col].agg(["mean", "std", "median"])
            client_value_stats.columns = [
                "client_mean_value",
                "client_std_value",
                "client_median_value",
            ]
            client_value_stats = client_value_stats.reset_index()

            df = df.merge(client_value_stats, on=client_col, how="left", suffixes=("", "_pattern"))

            df["client_std_value"] = df["client_std_value"].fillna(0)
            df["value_zscore"] = (df[value_col] - df["client_mean_value"]) / (
                df["client_std_value"] + 1e-6
            )

            df["is_value_outlier"] = (np.abs(df["value_zscore"]) > 2).astype(int)

            df["value_to_median_ratio"] = df[value_col] / (df["client_median_value"] + 1e-6)
            df["is_above_median"] = (df["value_to_median_ratio"] > 2).astype(int)

        if client_col and "hour" in df.columns:

            def hour_entropy(hours):
                if len(hours) <= 1:
                    return 0.0
                value_counts = hours.value_counts(normalize=True)
                return float(scipy_entropy(value_counts))

            hour_pattern = df.groupby(client_col)["hour"].transform(hour_entropy)
            df["hour_entropy"] = hour_pattern.fillna(0)

            client_hour_mode = df.groupby(client_col)["hour"].transform(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12
            )
            df["hour_deviation"] = np.abs(df["hour"] - client_hour_mode)
            df["is_unusual_hour"] = (df["hour_deviation"] > 6).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de features criadas."""
        return [
            # Temporais
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "is_business_hours",
            "is_early_morning",
            # Valor
            "log_value",
            "value_rounded",
            "is_high_value",
            "is_very_high_value",
            # Comportamento
            "avg_value",
            "std_value",
            "num_transactions",
            "value_deviation",
            "is_new_client",
            "is_max_value",
            # Dispositivo
            "num_clients_per_device",
            "is_shared_device",
            "is_new_device",
            # Localização
            "is_high_risk_state",
            "is_brazil",
            # Canal/Tipo
            "is_mobile",
            "is_web",
            "is_atm",
            "is_pix",
            "is_boleto",
            "is_credit",
            # Velocidade
            "time_since_last_transaction",
            "is_rapid_transaction",
            "is_very_rapid_transaction",
            # Entropia de Localização (NOVAS)
            "location_entropy",
            "unique_locations_count",
            "is_diverse_locations",
            "location_entropy_normalized",
            # Padrões de Transação (NOVAS)
            "value_zscore",
            "is_value_outlier",
            "value_to_median_ratio",
            "is_above_median",
            "hour_entropy",
            "hour_deviation",
            "is_unusual_hour",
        ]
