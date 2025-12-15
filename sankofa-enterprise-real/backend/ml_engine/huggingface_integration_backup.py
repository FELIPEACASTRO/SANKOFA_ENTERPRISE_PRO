"""
Integração com Hugging Face - Modelos e Datasets de Detecção de Fraude
Sankofa Enterprise Pro - Novembro 2025
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class HuggingFaceFraudDetector:
    """
    Integração com modelos e datasets do Hugging Face para detecção de fraude
    """

    # Modelos pré-treinados disponíveis
    AVAILABLE_MODELS = {
        "cifer_fraud_detection": "CiferAI/cifer-fraud-detection-k1-a",
        "keras_imbalanced": "keras-io/imbalanced_classification",
        "kmasiak_fraud": "kmasiak/FraudDetection",
        "mistral_fraud": "Bilic/Mistral-7B-LLM-Fraud-Detection",
    }

    # Datasets disponíveis
    AVAILABLE_DATASETS = {
        "cifer_af": "CiferAI/Cifer-Fraud-Detection-Dataset-AF",
        "credit_card": "liberatoratif/Credit-card-fraud-detection",
        "financial_fraud": "amitkedia/Financial-Fraud-Dataset",
        "kmasiak_fraud": "kmasiak/FraudDetection",
    }

    def __init__(self, model_name: str = "cifer_fraud_detection", use_gpu: bool = True):
        """
        Inicializa detector com modelo pré-treinado

        Args:
            model_name: Nome do modelo (chave em AVAILABLE_MODELS)
            use_gpu: Usar GPU se disponível
        """
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Carrega modelo pré-treinado do Hugging Face"""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo desconhecido: {self.model_name}")

        model_path = self.AVAILABLE_MODELS[self.model_name]
        logger.info(f"Carregando modelo: {model_path}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Modelo carregado com sucesso em {self.device}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def predict_fraud(self, features: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Prediz se uma transação é fraude

        Args:
            features: Dicionário com features da transação

        Returns:
            (is_fraud: bool, fraud_probability: float)
        """
        # Converter features para string para tokenização
        feature_str = self._features_to_string(features)

        # Tokenizar
        inputs = self.tokenizer(
            feature_str, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        # Predição
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extrair probabilidade
        probabilities = outputs.logits.softmax(dim=1)
        fraud_probability = probabilities[0, 1].item()

        return fraud_probability > 0.5, fraud_probability

    def predict_batch(self, features_batch: list) -> np.ndarray:
        """
        Predição em lote (mais eficiente)

        Args:
            features_batch: Lista de dicionários com features

        Returns:
            Array com probabilidades de fraude
        """
        probabilities = []

        for features in features_batch:
            _, prob = self.predict_fraud(features)
            probabilities.append(prob)

        return np.array(probabilities)

    @staticmethod
    def _features_to_string(features: Dict) -> str:
        """Converte features dict para string para tokenização"""
        parts = []
        for key, value in features.items():
            parts.append(f"{key} is {value}")
        return ", ".join(parts)


class HuggingFaceDatasetLoader:
    """
    Carrega e prepara datasets do Hugging Face para Sankofa
    """

    DATASETS = HuggingFaceFraudDetector.AVAILABLE_DATASETS

    @staticmethod
    def load_cifer_dataset(partition: int = 1) -> pd.DataFrame:
        """
        Carrega dataset CiferAI (21M transações, mobile money)

        Args:
            partition: Qual partição (1-14)

        Returns:
            DataFrame com transações
        """
        logger.info(f"Carregando CiferAI partition {partition}...")

        dataset = load_dataset(
            "CiferAI/Cifer-Fraud-Detection-Dataset-AF",
            data_files=f"Cifer-Fraud-Detection-Dataset-AF-part-{partition}-14.csv",
        )

        df = dataset["train"].to_pandas()
        logger.info(f"Dataset carregado: {len(df)} registros")

        return df

    @staticmethod
    def load_credit_card_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega dataset de cartão de crédito (284K, ULB)

        Returns:
            (X: features, y: labels)
        """
        logger.info("Carregando Credit Card Dataset...")

        dataset = load_dataset("liberatoratif/Credit-card-fraud-detection")
        df = dataset["train"].to_pandas()

        X = df.drop(columns=["fraud"])
        y = df["fraud"]

        logger.info(f"Dataset: {len(df)} registros, {(y.sum()/len(y)*100):.2f}% fraude")

        return X, y

    @staticmethod
    def prepare_for_sankofa(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transforma dataset Hugging Face para schema Sankofa

        Args:
            df: DataFrame original

        Returns:
            Dicionário com dados mapeados para Sankofa
        """
        # Mapear colunas para schema Sankofa
        mapping = {
            "amount": "transaction_amount",
            "time": "timestamp",
            "merchant": "merchant_id",
            "customer": "customer_id",
            "fraud": "is_fraud",
            "category": "merchant_category",
            "type": "transaction_type",
        }

        df_mapped = df.rename(columns=mapping)

        # Pré-processamento
        df_mapped = HuggingFaceDatasetLoader._preprocess(df_mapped)

        return {
            "data": df_mapped,
            "features": list(df_mapped.columns),
            "target": "is_fraud",
            "metrics": {
                "total_records": len(df_mapped),
                "fraud_rate": (
                    (df_mapped["is_fraud"].sum() / len(df_mapped))
                    if "is_fraud" in df_mapped
                    else None
                ),
            },
        }

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Pré-processamento básico"""
        # Remover nulls críticos
        df = df.dropna(
            subset=(
                ["transaction_amount", "is_fraud"]
                if "is_fraud" in df.columns
                else ["transaction_amount"]
            )
        )

        # Converter tipos
        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")

        return df

    @staticmethod
    def get_feature_statistics(df: pd.DataFrame) -> Dict:
        """Retorna estatísticas do dataset"""
        return {
            "total_records": len(df),
            "features": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "fraud_rate": (df["is_fraud"].sum() / len(df)) if "is_fraud" in df else None,
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": df.describe().to_dict(),
        }


class HuggingFaceTransferLearning:
    """
    Transfer learning com modelos Hugging Face para domínios específicos
    """

    def __init__(self, base_model_name: str = "CiferAI/cifer-fraud-detection-k1-a"):
        """
        Inicializa para fine-tuning

        Args:
            base_model_name: Modelo base para transfer learning
        """
        self.base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model.to(self.device)

    def fine_tune(
        self,
        train_data: pd.DataFrame,
        target_col: str = "is_fraud",
        epochs: int = 3,
        batch_size: int = 32,
    ) -> Dict:
        """
        Fine-tune modelo para domínio específico (ex: PIX, Crédito)

        Args:
            train_data: DataFrame com dados de treinamento
            target_col: Coluna alvo (fraud)
            epochs: Épocas de treinamento
            batch_size: Tamanho do lote

        Returns:
            Métricas de treinamento
        """
        logger.info(f"Iniciando fine-tuning por {epochs} épocas...")

        # Preparar data loader (simplificado)
        features = train_data.drop(columns=[target_col]).to_dict("records")
        labels = train_data[target_col].values

        # Fine-tuning (implementação simplificada)
        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(features), batch_size):
                batch_features = features[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                # Tokenizar batch
                batch_str = [str(f) for f in batch_features]
                inputs = self.tokenizer(
                    batch_str, return_tensors="pt", max_length=512, truncation=True, padding=True
                ).to(self.device)

                targets = torch.LongTensor(batch_labels).to(self.device)

                # Forward pass
                outputs = self.base_model(**inputs, labels=targets)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / (len(features) // batch_size)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return {"final_loss": avg_loss, "epochs": epochs}

    def save_model(self, path: str):
        """Salva modelo fine-tuned"""
        self.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Modelo salvo em: {path}")


# Exemplo de uso
if __name__ == "__main__":
    # 1. Carregar dataset CiferAI
    print("=== Carregando Dataset ===")
    loader = HuggingFaceDatasetLoader()
    df_cifer = loader.load_cifer_dataset(partition=1)
    df_prepared = loader.prepare_for_sankofa(df_cifer)
    print(f"Dataset preparado: {df_prepared['metrics']}")

    # 2. Carregar modelo pré-treinado
    print("\n=== Carregando Modelo ===")
    detector = HuggingFaceFraudDetector(model_name="cifer_fraud_detection")

    # 3. Fazer predição
    print("\n=== Fazendo Predição ===")
    test_transaction = {
        "amount": 1500.00,
        "merchant": "Coffee Shop",
        "time": "14:30",
        "category": "Food & Beverage",
    }

    is_fraud, probability = detector.predict_fraud(test_transaction)
    print(f"Fraude: {is_fraud}, Probabilidade: {probability:.2%}")

    print("\n Integracao Hugging Face funcionando!")


# Aliases para compatibilidade
HuggingFaceIntegration = HuggingFaceFraudDetector
HFIntegration = HuggingFaceFraudDetector
