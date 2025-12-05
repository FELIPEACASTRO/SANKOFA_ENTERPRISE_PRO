"""
Dataset Loaders - Integração com Datasets Gratuitos
Sankofa Enterprise Pro - 100% Open Source

Datasets suportados:
- CiferAI (21M transações) - Hugging Face
- IEEE-CIS (590K transações) - Kaggle/Local
- Elliptic++ (822K wallets) - GitHub
- PaySim (6.3M transações) - GitHub
- Credit Card ULB (284K) - OpenML
"""

import os
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    name: str
    records: int
    features: int
    fraud_rate: float
    source: str
    local_path: Optional[str] = None
    loaded: bool = False


class DatasetLoader:
    """Carregador unificado de datasets gratuitos para detecção de fraude"""

    DATASETS = {
        "ciferai": DatasetInfo(
            name="CiferAI Mobile Money",
            records=21_000_000,
            features=14,
            fraud_rate=0.001,
            source="huggingface",
        ),
        "ieee_cis": DatasetInfo(
            name="IEEE-CIS Fraud Detection",
            records=590_000,
            features=394,
            fraud_rate=0.035,
            source="kaggle",
        ),
        "elliptic": DatasetInfo(
            name="Elliptic++ Bitcoin",
            records=822_000,
            features=56,
            fraud_rate=0.02,
            source="github",
        ),
        "paysim": DatasetInfo(
            name="PaySim Mobile Money",
            records=6_300_000,
            features=10,
            fraud_rate=0.0013,
            source="github",
        ),
        "credit_card_ulb": DatasetInfo(
            name="Credit Card ULB", records=284_000, features=31, fraud_rate=0.0017, source="openml"
        ),
    }

    def __init__(self, cache_dir: str = "./data/datasets"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def list_available(self) -> Dict[str, DatasetInfo]:
        """Lista todos os datasets disponíveis"""
        return self.DATASETS

    def load_ciferai(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Carrega dataset CiferAI do Hugging Face
        21M transações de mobile money (similar a PIX)

        Args:
            sample_size: Número de registros para carregar (None = todos)

        Returns:
            DataFrame com transações
        """
        try:
            from datasets import load_dataset

            logger.info("Carregando CiferAI do Hugging Face...")

            dataset = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF", split="train")

            df = dataset.to_pandas()

            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)

            df = self._add_pix_features(df)

            logger.info(f"CiferAI carregado: {len(df)} registros")
            return df

        except ImportError:
            logger.error("Instale: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar CiferAI: {e}")
            raise

    def load_ieee_cis(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega dataset IEEE-CIS (requer download manual do Kaggle)
        590K transações com 394 features

        Args:
            data_path: Caminho para os arquivos CSV

        Returns:
            Tuple (train_df, test_df)
        """
        if data_path is None:
            data_path = os.path.join(self.cache_dir, "ieee-cis")

        train_transaction = os.path.join(data_path, "train_transaction.csv")
        train_identity = os.path.join(data_path, "train_identity.csv")

        if not os.path.exists(train_transaction):
            raise FileNotFoundError(
                f"Dataset IEEE-CIS não encontrado em {data_path}. "
                "Baixe de: https://www.kaggle.com/c/ieee-fraud-detection"
            )

        logger.info("Carregando IEEE-CIS...")

        train_tx = pd.read_csv(train_transaction)
        train_id = pd.read_csv(train_identity)

        train = train_tx.merge(train_id, on="TransactionID", how="left")

        logger.info(f"IEEE-CIS carregado: {len(train)} registros, {train.shape[1]} features")
        return train

    def load_elliptic(self) -> Dict[str, Any]:
        """
        Carrega dataset Elliptic++ do GitHub
        822K wallets Bitcoin para detecção de redes de fraude

        Returns:
            Dict com nodes, edges, features e labels
        """
        try:
            import torch
            from torch_geometric.datasets import EllipticBitcoinDataset

            logger.info("Carregando Elliptic++ do PyTorch Geometric...")

            dataset = EllipticBitcoinDataset(root=self.cache_dir)
            data = dataset[0]

            result = {
                "num_nodes": data.num_nodes,
                "num_edges": data.num_edges,
                "features": data.x,
                "edge_index": data.edge_index,
                "labels": data.y,
                "dataset": dataset,
            }

            logger.info(f"Elliptic++ carregado: {data.num_nodes} nós, {data.num_edges} arestas")
            return result

        except ImportError:
            logger.error("Instale: pip install torch torch-geometric")
            raise

    def load_paysim(
        self, sample_size: Optional[int] = None, full_dataset: bool = False, encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """
        Carrega dataset PaySim.
        6.3M transações de mobile money (proxy para PIX)

        NOTA: Por padrão baixa apenas o sample (~600K registros).
        Para o dataset completo (6.3M), defina full_dataset=True e
        baixe manualmente de: https://www.kaggle.com/datasets/ealaxi/paysim1

        Args:
            sample_size: Número de registros para sample
            full_dataset: Se True, espera o dataset completo no cache_dir

        Returns:
            DataFrame com transações
        """
        if full_dataset:
            paysim_path = os.path.join(self.cache_dir, "PS_20174392719_1491204439457_log.csv")
            if not os.path.exists(paysim_path):
                raise FileNotFoundError(
                    f"Dataset completo não encontrado em {paysim_path}. "
                    "Baixe de: https://www.kaggle.com/datasets/ealaxi/paysim1"
                )
        else:
            paysim_path = os.path.join(self.cache_dir, "paysim_sample.csv")

            if not os.path.exists(paysim_path):
                logger.info("Baixando PaySim SAMPLE do GitHub (~600K registros)...")
                logger.warning("Para o dataset completo (6.3M), baixe manualmente do Kaggle")
                import urllib.request

                url = "https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/PS_sample.csv"
                urllib.request.urlretrieve(url, paysim_path)

        logger.info("Carregando PaySim...")
        try:
            df = pd.read_csv(paysim_path, encoding=encoding, on_bad_lines="skip")
        except (UnicodeDecodeError, LookupError):
            logger.warning(f"Erro com encoding {encoding}, tentando latin-1")
            df = pd.read_csv(paysim_path, encoding="latin-1", on_bad_lines="skip")

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        df = self._add_pix_features(df)

        logger.info(f"PaySim carregado: {len(df)} registros")
        if not full_dataset:
            logger.warning("Usando SAMPLE do PaySim. Para produção, use full_dataset=True")

        return df

    def load_credit_card_ulb(self, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Carrega dataset Credit Card ULB do OpenML
        284K transações de cartão de crédito europeu

        Returns:
            DataFrame com transações
        """
        try:
            from sklearn.datasets import fetch_openml

            logger.info("Carregando Credit Card ULB do OpenML...")

            data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
            df = (
                data["frame"]
                if isinstance(data, dict)
                else pd.DataFrame(data.data, columns=data.feature_names)
            )

            df["Class"] = df["Class"].astype(int)

            logger.info(f"Credit Card ULB carregado: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao carregar Credit Card ULB: {e}")
            raise

    def _add_pix_features(self, df: pd.DataFrame, synthetic: bool = True) -> pd.DataFrame:
        """
        Adiciona features específicas de PIX ao dataset.

        IMPORTANTE: Features geradas são SINTÉTICAS para demonstração/teste.
        Em produção, use dados reais do sistema de transações.

        Args:
            df: DataFrame com transações
            synthetic: Se True, marca features como sintéticas (default: True)

        Returns:
            DataFrame com features PIX adicionadas
        """

        if "amount" in df.columns:
            amount_col = "amount"
        elif "Amount" in df.columns:
            amount_col = "Amount"
        elif "TransactionAmt" in df.columns:
            amount_col = "TransactionAmt"
        else:
            logger.warning("Coluna de valor não encontrada - features PIX não adicionadas")
            return df

        np.random.seed(42)

        df["_synthetic_pix_features"] = synthetic

        df["device_registered"] = np.random.choice([True, False], size=len(df), p=[0.85, 0.15])

        if "step" in df.columns:
            df["hour"] = df["step"] % 24
        elif "TransactionDT" in df.columns:
            df["hour"] = (df["TransactionDT"] // 3600) % 24
        else:
            df["hour"] = (df.index % 24).astype(int)

        df["is_nocturnal"] = df["hour"].apply(lambda h: h >= 23 or h < 5)

        df["nocturnal_limit_exceeded"] = df["is_nocturnal"] & (df[amount_col] > 1000)

        df["unregistered_device_exceeded"] = ~df["device_registered"] & (df[amount_col] > 200)

        df["recipient_is_pj"] = np.random.choice([True, False], size=len(df), p=[0.33, 0.67])

        df["pix_key_type"] = np.random.choice(
            ["cpf", "cnpj", "email", "telefone", "aleatoria"],
            size=len(df),
            p=[0.35, 0.15, 0.20, 0.20, 0.10],
        )

        logger.warning(
            "Features PIX SINTÉTICAS adicionadas ao dataset (use dados reais em produção)"
        )
        return df

    def prepare_for_training(
        self, df: pd.DataFrame, target_col: str = "isFraud", test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dataset para treinamento de modelo

        Args:
            df: DataFrame com dados
            target_col: Nome da coluna target
            test_size: Proporção de teste

        Returns:
            X_train, X_test, y_train, y_test

        Raises:
            ValueError: Se dataset for muito pequeno ou não tiver coluna target
        """
        from sklearn.model_selection import train_test_split

        # VALIDAÇÃO 1: Verificar tamanho mínimo
        if len(df) < 10:
            raise ValueError(f"Dataset muito pequeno: {len(df)} registros (mínimo 10)")

        # VALIDAÇÃO 2: Encontrar coluna target
        if target_col not in df.columns:
            possible_targets = ["isFraud", "is_fraud", "fraud", "Class", "label"]
            found = False
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    found = True
                    break
            if not found:
                raise ValueError(f"Coluna target não encontrada. Esperado: {target_col}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

        if len(numeric_cols) == 0:
            raise ValueError("Nenhuma coluna numérica encontrada para features")

        X = df[numeric_cols].fillna(0)
        y = df[target_col]

        # VALIDAÇÃO 3: Verificar classes
        unique_classes = y.nunique()
        if unique_classes < 2:
            raise ValueError(f"Target precisa de pelo menos 2 classes, encontrou {unique_classes}")

        # VALIDAÇÃO 4: Garantir suficientes amostras por classe
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 2:
            logger.warning(
                f"Classe minoritária tem apenas {min_class_count} amostra(s), desabilitando stratify"
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        logger.info(f"Dataset preparado: {len(X_train)} treino, {len(X_test)} teste")
        logger.info(f"Taxa de fraude: {y.mean():.4%}")
        logger.info(f"Features: {len(numeric_cols)} colunas numéricas")

        return X_train, X_test, y_train, y_test


class PIXFeatureEngineer:
    """Engenharia de features específicas para PIX"""

    BACEN_FEATURES = [
        "device_registered",
        "nocturnal_limit_exceeded",
        "unregistered_device_exceeded",
        "recipient_is_pj",
        "pix_key_type",
        "first_pix_to_recipient",
        "mule_account_indicator",
    ]

    VELOCITY_FEATURES = [
        "pix_count_1h",
        "pix_count_24h",
        "pix_count_7d",
        "distinct_recipients_1h",
        "distinct_recipients_24h",
        "total_amount_1h",
        "total_amount_24h",
        "avg_amount_7d",
    ]

    BEHAVIORAL_FEATURES = [
        "amount_deviation_zscore",
        "hour_deviation",
        "typical_day_of_week",
        "session_duration",
        "time_since_last_pix",
    ]

    def __init__(self):
        self.user_history = {}

    def extract_features(self, transaction: Dict) -> Dict:
        """
        Extrai todas as features para uma transação PIX

        Args:
            transaction: Dados da transação

        Returns:
            Dict com features extraídas
        """
        features = {}

        features.update(self._extract_bacen_features(transaction))
        features.update(self._extract_velocity_features(transaction))
        features.update(self._extract_behavioral_features(transaction))

        return features

    def _extract_bacen_features(self, tx: Dict) -> Dict:
        """Features de compliance BACEN"""

        amount = tx.get("amount", 0)
        hour = tx.get("hour", 12)

        return {
            "device_registered": tx.get("device_registered", True),
            "nocturnal_limit_exceeded": (hour >= 23 or hour < 5) and amount > 1000,
            "unregistered_device_exceeded": not tx.get("device_registered", True) and amount > 200,
            "recipient_is_pj": tx.get("recipient_type") == "PJ",
            "pix_key_type": tx.get("pix_key_type", "cpf"),
            "first_pix_to_recipient": tx.get("first_to_recipient", False),
        }

    def _extract_velocity_features(self, tx: Dict) -> Dict:
        """Features de velocidade"""

        user_id = tx.get("user_id")

        if user_id not in self.user_history:
            return {feat: 0 for feat in self.VELOCITY_FEATURES}

        history = self.user_history[user_id]

        return {
            "pix_count_1h": history.get("count_1h", 0),
            "pix_count_24h": history.get("count_24h", 0),
            "pix_count_7d": history.get("count_7d", 0),
            "distinct_recipients_1h": history.get("recipients_1h", 0),
            "distinct_recipients_24h": history.get("recipients_24h", 0),
            "total_amount_1h": history.get("amount_1h", 0),
            "total_amount_24h": history.get("amount_24h", 0),
            "avg_amount_7d": history.get("avg_amount_7d", 0),
        }

    def _extract_behavioral_features(self, tx: Dict) -> Dict:
        """Features comportamentais"""

        user_id = tx.get("user_id")
        amount = tx.get("amount", 0)

        if user_id not in self.user_history:
            return {
                "amount_deviation_zscore": 0,
                "hour_deviation": 0,
                "typical_day_of_week": True,
                "session_duration": tx.get("session_duration", 0),
                "time_since_last_pix": 999999,
            }

        history = self.user_history[user_id]
        avg_amount = history.get("avg_amount", amount)
        std_amount = history.get("std_amount", 1)

        zscore = (amount - avg_amount) / std_amount if std_amount > 0 else 0

        return {
            "amount_deviation_zscore": zscore,
            "hour_deviation": abs(tx.get("hour", 12) - history.get("typical_hour", 12)),
            "typical_day_of_week": tx.get("day_of_week") in history.get("typical_days", []),
            "session_duration": tx.get("session_duration", 0),
            "time_since_last_pix": tx.get("time_since_last", 0),
        }

    def update_user_history(self, user_id: str, transaction: Dict):
        """Atualiza histórico do usuário"""

        if user_id not in self.user_history:
            self.user_history[user_id] = {
                "transactions": [],
                "avg_amount": 0,
                "std_amount": 0,
                "typical_hour": 12,
                "typical_days": [],
            }

        history = self.user_history[user_id]
        history["transactions"].append(transaction)

        amounts = [t["amount"] for t in history["transactions"]]
        history["avg_amount"] = np.mean(amounts)
        history["std_amount"] = np.std(amounts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = DatasetLoader()

    print("\n=== Datasets Disponíveis ===")
    for key, info in loader.list_available().items():
        print(f"\n{key}:")
        print(f"  Nome: {info.name}")
        print(f"  Registros: {info.records:,}")
        print(f"  Features: {info.features}")
        print(f"  Taxa de fraude: {info.fraud_rate:.2%}")
        print(f"  Fonte: {info.source}")

    print("\n=== Testando Credit Card ULB (mais rápido) ===")
    try:
        df = loader.load_credit_card_ulb()
        print(f"Carregado: {len(df)} registros")
        print(f"Colunas: {list(df.columns)[:10]}...")
    except Exception as e:
        print(f"Erro: {e}")
