"""
Sankofa Enterprise Pro - Embedding Feature Generator
Features baseadas em embeddings aprendidos

Baseado em:
- Entity embeddings for categorical variables
- Word2Vec for transaction sequences
- Autoencoders for dimensionality reduction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingFeatureGenerator:
    """
    Gerador de Features de Embedding

    Features baseadas em:
    - Entity embeddings (customer, device, merchant)
    - Transaction sequence embeddings
    - Co-occurrence embeddings

    Gera 30+ features
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", 16)

        # Embeddings armazenados
        self.entity_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.co_occurrence_matrix: Dict[str, Dict[Tuple, int]] = {}

        logger.info(f"EmbeddingFeatureGenerator v{self.VERSION} initialized")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def generate(
        self,
        df: pd.DataFrame,
        entity_cols: Optional[List[str]] = None,
        amount_col: str = "amount",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Gerar features de embedding

        Args:
            df: DataFrame com transações
            entity_cols: Colunas de entidade para embeddings
            amount_col: Coluna de valor
            timestamp_col: Coluna de timestamp

        Returns:
            DataFrame com features de embedding
        """
        df = df.copy()

        # Entidades padrão
        default_entities = ["customer_id", "device_id", "merchant_id", "receiver_id"]
        entities = entity_cols or default_entities
        available_entities = [e for e in entities if e in df.columns]

        logger.info(f"Generating embedding features for {len(available_entities)} entities...")

        # 1. Entity embeddings (hash-based para eficiência)
        for entity in available_entities:
            logger.info(f"Generating embeddings for: {entity}")
            df = self._add_hash_embeddings(df, entity)

        # 2. Co-occurrence features
        if len(available_entities) >= 2:
            logger.info("Generating co-occurrence features...")
            df = self._add_co_occurrence_features(df, available_entities)

        # 3. Sequence features (se houver timestamp)
        if timestamp_col in df.columns and "customer_id" in df.columns:
            logger.info("Generating sequence features...")
            df = self._add_sequence_features(df, timestamp_col)

        # 4. Target encoding features (se houver target)
        if "is_fraud" in df.columns:
            logger.info("Generating target encoding features...")
            df = self._add_target_encoding(df, available_entities)

        # 5. Similarity features
        logger.info("Generating similarity features...")
        df = self._add_similarity_features(df, available_entities)

        return df

    def _add_hash_embeddings(
        self,
        df: pd.DataFrame,
        entity_col: str
    ) -> pd.DataFrame:
        """
        Adicionar embeddings baseados em hash

        Técnica: Random projection usando hash determinístico
        """
        prefix = f"emb_{entity_col}"

        # Função para gerar embedding determinístico
        def hash_to_embedding(value, dim):
            if pd.isna(value):
                return np.zeros(dim)

            # Hash determinístico
            hash_bytes = hashlib.md5(str(value).encode()).digest()

            # Converter para floats em [-1, 1]
            embedding = np.array([
                (hash_bytes[i % len(hash_bytes)] / 128.0) - 1.0
                for i in range(dim)
            ])

            # Normalizar
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        # Gerar embeddings para cada valor único
        unique_values = df[entity_col].dropna().unique()
        embeddings = {
            val: hash_to_embedding(val, self.embedding_dim)
            for val in unique_values
        }

        # Adicionar ao DataFrame
        for i in range(self.embedding_dim):
            df[f"{prefix}_{i}"] = df[entity_col].map(
                lambda x: embeddings.get(x, np.zeros(self.embedding_dim))[i]
            )

        # Estatísticas do embedding
        emb_cols = [f"{prefix}_{i}" for i in range(self.embedding_dim)]
        df[f"{prefix}_mean"] = df[emb_cols].mean(axis=1)
        df[f"{prefix}_std"] = df[emb_cols].std(axis=1)
        df[f"{prefix}_max"] = df[emb_cols].max(axis=1)
        df[f"{prefix}_min"] = df[emb_cols].min(axis=1)
        df[f"{prefix}_norm"] = np.linalg.norm(df[emb_cols].values, axis=1)

        return df

    def _add_co_occurrence_features(
        self,
        df: pd.DataFrame,
        entity_cols: List[str]
    ) -> pd.DataFrame:
        """Adicionar features de co-ocorrência entre entidades"""

        # Calcular co-ocorrência entre pares de entidades
        for i, col1 in enumerate(entity_cols):
            for col2 in entity_cols[i + 1:]:
                # Contar co-ocorrências
                co_occurrence = df.groupby([col1, col2]).size().reset_index(name="co_count")

                # Normalizar
                total = co_occurrence["co_count"].sum()
                co_occurrence["co_pmi"] = np.log(
                    (co_occurrence["co_count"] * total) /
                    (df.groupby(col1).size().reset_index(name="c1").merge(
                        df.groupby(col2).size().reset_index(name="c2"), how="cross"
                    )[["c1", "c2"]].prod(axis=1).values[:len(co_occurrence)] + 1)
                )

                # Merge
                feature_name = f"cooccur_{col1}_{col2}"
                df = df.merge(
                    co_occurrence[[col1, col2, "co_count"]].rename(
                        columns={"co_count": feature_name}
                    ),
                    on=[col1, col2],
                    how="left"
                )

                df[feature_name] = df[feature_name].fillna(0)

                # Normalizar
                df[f"{feature_name}_normalized"] = df[feature_name] / (df[feature_name].max() + 1)

        return df

    def _add_sequence_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Adicionar features de sequência de transações"""

        # Ordenar por timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Para cada cliente, analisar sequência
        if "customer_id" in df.columns:
            # Posição na sequência do cliente
            df["tx_sequence_position"] = df.groupby("customer_id").cumcount()

            # Posição normalizada
            customer_counts = df.groupby("customer_id")["tx_sequence_position"].transform("max")
            df["tx_sequence_position_normalized"] = df["tx_sequence_position"] / (customer_counts + 1)

            # É primeira transação?
            df["is_first_tx"] = (df["tx_sequence_position"] == 0).astype(int)

            # É última transação conhecida?
            df["tx_remaining"] = customer_counts - df["tx_sequence_position"]
            df["is_recent_tx"] = (df["tx_remaining"] <= 1).astype(int)

        # Variação de padrão na sequência
        if "amount" in df.columns and "customer_id" in df.columns:
            # Diferença do valor anterior
            df["amount_diff_prev"] = df.groupby("customer_id")["amount"].diff().fillna(0)

            # Ratio com valor anterior
            df["amount_ratio_prev"] = df["amount"] / (
                df.groupby("customer_id")["amount"].shift(1).fillna(df["amount"]) + 1
            )

        return df

    def _add_target_encoding(
        self,
        df: pd.DataFrame,
        entity_cols: List[str]
    ) -> pd.DataFrame:
        """Adicionar target encoding para entidades (se target disponível)"""

        if "is_fraud" not in df.columns:
            return df

        for entity_col in entity_cols:
            # Calcular taxa de fraude por entidade
            fraud_rate = df.groupby(entity_col)["is_fraud"].mean().reset_index()
            fraud_rate.columns = [entity_col, f"target_enc_{entity_col}"]

            # Suavização com prior global
            global_fraud_rate = df["is_fraud"].mean()
            entity_counts = df.groupby(entity_col).size().reset_index(name="count")

            fraud_rate = fraud_rate.merge(entity_counts, on=entity_col)

            # Smoothed target encoding
            smoothing_factor = 10
            fraud_rate[f"target_enc_{entity_col}_smoothed"] = (
                (fraud_rate[f"target_enc_{entity_col}"] * fraud_rate["count"] +
                 global_fraud_rate * smoothing_factor) /
                (fraud_rate["count"] + smoothing_factor)
            )

            df = df.merge(
                fraud_rate[[entity_col, f"target_enc_{entity_col}_smoothed"]],
                on=entity_col,
                how="left"
            )

        return df

    def _add_similarity_features(
        self,
        df: pd.DataFrame,
        entity_cols: List[str]
    ) -> pd.DataFrame:
        """Adicionar features de similaridade entre embeddings"""

        # Similaridade entre sender e receiver (se disponíveis)
        if "customer_id" in entity_cols and "receiver_id" in entity_cols:
            sender_emb_cols = [f"emb_customer_id_{i}" for i in range(self.embedding_dim)]
            receiver_emb_cols = [f"emb_receiver_id_{i}" for i in range(self.embedding_dim)]

            if all(c in df.columns for c in sender_emb_cols + receiver_emb_cols):
                # Similaridade de cosseno
                sender_emb = df[sender_emb_cols].values
                receiver_emb = df[receiver_emb_cols].values

                dot_product = np.sum(sender_emb * receiver_emb, axis=1)
                sender_norm = np.linalg.norm(sender_emb, axis=1)
                receiver_norm = np.linalg.norm(receiver_emb, axis=1)

                df["sender_receiver_cosine_sim"] = dot_product / (sender_norm * receiver_norm + 1e-6)

                # Distância euclidiana
                df["sender_receiver_euclidean_dist"] = np.linalg.norm(
                    sender_emb - receiver_emb, axis=1
                )

        # Similaridade com centróide do cluster (se houver comunidade)
        if "sender_community" in df.columns:
            # Calcular centróide por comunidade
            emb_cols = [f"emb_customer_id_{i}" for i in range(self.embedding_dim)]
            if all(c in df.columns for c in emb_cols):
                community_centroids = df.groupby("sender_community")[emb_cols].mean()

                # Distância ao centróide
                def dist_to_centroid(row):
                    comm = row.get("sender_community")
                    if pd.isna(comm) or comm not in community_centroids.index:
                        return 0

                    centroid = community_centroids.loc[comm].values
                    point = row[emb_cols].values
                    return np.linalg.norm(point - centroid)

                df["dist_to_community_centroid"] = df.apply(dist_to_centroid, axis=1)

        return df

    def fit_entity_embeddings(
        self,
        df: pd.DataFrame,
        entity_col: str,
        amount_col: str = "amount"
    ):
        """
        Treinar embeddings para uma entidade usando estatísticas

        Em produção, usar técnicas como:
        - Word2Vec em sequências de transações
        - Autoencoders
        - Graph Neural Networks
        """
        # Calcular estatísticas por entidade
        stats = df.groupby(entity_col).agg({
            amount_col: ["count", "mean", "std", "min", "max"]
        })

        stats.columns = ["count", "mean", "std", "min", "max"]
        stats["std"] = stats["std"].fillna(0)

        # Normalizar
        for col in stats.columns:
            stats[col] = (stats[col] - stats[col].min()) / (stats[col].max() - stats[col].min() + 1e-6)

        # Armazenar como embeddings
        self.entity_embeddings[entity_col] = {
            entity: stats.loc[entity].values
            for entity in stats.index
        }

    def get_feature_names(self) -> List[str]:
        """Retornar lista de nomes de features de embedding"""
        features = []

        # Features de embedding por entidade (exemplo para customer_id)
        for i in range(self.embedding_dim):
            features.append(f"emb_entity_{i}")

        features.extend([
            "emb_entity_mean", "emb_entity_std", "emb_entity_max",
            "emb_entity_min", "emb_entity_norm",

            # Co-occurrence
            "cooccur_col1_col2", "cooccur_col1_col2_normalized",

            # Sequence
            "tx_sequence_position", "tx_sequence_position_normalized",
            "is_first_tx", "tx_remaining", "is_recent_tx",
            "amount_diff_prev", "amount_ratio_prev",

            # Target encoding
            "target_enc_entity_smoothed",

            # Similarity
            "sender_receiver_cosine_sim", "sender_receiver_euclidean_dist",
            "dist_to_community_centroid"
        ])

        return features

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do gerador"""
        return {
            "version": self.VERSION,
            "embedding_dim": self.embedding_dim,
            "trained_entities": list(self.entity_embeddings.keys())
        }


def create_embedding_generator(config: Optional[Dict[str, Any]] = None) -> EmbeddingFeatureGenerator:
    """Factory function para criar EmbeddingFeatureGenerator"""
    return EmbeddingFeatureGenerator(config)
