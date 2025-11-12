import logging

logger = logging.getLogger(__name__)
"""
Advanced Feature Engineering - Cria features avançadas para detecção de fraude
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")


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

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_night"] = df["hour"].between(22, 6).astype(int)
            df["is_business_hours"] = df["hour"].between(9, 18).astype(int)
            df["is_early_morning"] = df["hour"].between(0, 6).astype(int)
        elif "hour" not in df.columns:
            # Se não tem timestamp nem hour, criar features padrão
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
        ]
