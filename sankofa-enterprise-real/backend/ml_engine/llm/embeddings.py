"""
Sankofa Enterprise Pro - Embeddings Module
Gerador de embeddings para transacoes e textos

Baseado em:
- Sentence-BERT for text embeddings
- Transaction2Vec for transaction embeddings
- Word2Vec/FastText for financial vocabulary
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tentar importar bibliotecas opcionais
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Embeddings will use fallback.")


@dataclass
class EmbeddingResult:
    """Resultado de embedding"""
    embedding: np.ndarray
    dimension: int
    model_name: str


class TransactionEmbedder:
    """
    Gerador de embeddings para transacoes

    Gera embeddings densos a partir de features de transacao para uso
    em modelos de similaridade, clustering, e deteccao de anomalias.
    """

    VERSION = "1.0.0"
    DEFAULT_DIMENSION = 128

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o embedder de transacoes.

        Args:
            config: Configuracao opcional
        """
        self.config = config or {}
        self.dimension = self.config.get("dimension", self.DEFAULT_DIMENSION)
        self._initialized = True

        logger.info(f"TransactionEmbedder initialized: dimension={self.dimension}")

    def embed(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Gera embedding para uma transacao.

        Args:
            transaction: Dados da transacao

        Returns:
            Embedding numpy array
        """
        # Feature extraction
        features = []

        # Amount features
        amount = transaction.get("amount", 0)
        features.extend([
            np.log1p(amount),
            amount / 10000,  # Normalized
        ])

        # Temporal features
        hour = transaction.get("hour", 12)
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ])

        # Boolean features
        features.extend([
            float(transaction.get("is_new_device", False)),
            float(transaction.get("is_new_recipient", False)),
            float(transaction.get("is_high_amount", False)),
        ])

        # Velocity
        features.append(transaction.get("velocity_score", 0.0))

        # Pad to dimension
        feature_array = np.array(features)
        if len(feature_array) < self.dimension:
            padding = np.zeros(self.dimension - len(feature_array))
            feature_array = np.concatenate([feature_array, padding])

        return feature_array[:self.dimension]

    def embed_batch(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Gera embeddings para batch de transacoes.

        Args:
            transactions: Lista de transacoes

        Returns:
            Array de embeddings [n_transactions, dimension]
        """
        embeddings = [self.embed(t) for t in transactions]
        return np.array(embeddings)

    def get_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calcula similaridade de cosseno entre dois embeddings.

        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding

        Returns:
            Similaridade (0-1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class TextEmbedder:
    """
    Gerador de embeddings para textos

    Usa Sentence-BERT ou fallback simples para gerar embeddings
    de textos relacionados a fraude (mensagens, descricoes, etc).
    """

    VERSION = "1.0.0"
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o embedder de texto.

        Args:
            config: Configuracao opcional
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", self.DEFAULT_MODEL)
        self.model = None
        self.dimension = self.DEFAULT_DIMENSION

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"TextEmbedder initialized with {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {self.model_name}: {e}")
        else:
            logger.warning("Using fallback text embedder (sentence-transformers not available)")

    def embed(self, text: str) -> np.ndarray:
        """
        Gera embedding para texto.

        Args:
            text: Texto de entrada

        Returns:
            Embedding numpy array
        """
        if self.model is not None:
            return self.model.encode(text)

        # Fallback: simple bag-of-words style embedding
        return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para batch de textos.

        Args:
            texts: Lista de textos

        Returns:
            Array de embeddings [n_texts, dimension]
        """
        if self.model is not None:
            return self.model.encode(texts)

        return np.array([self._fallback_embed(t) for t in texts])

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Embedding fallback simples baseado em caracteres e palavras.

        Args:
            text: Texto de entrada

        Returns:
            Embedding array
        """
        embedding = np.zeros(self.dimension)

        if not text:
            return embedding

        # Character-level features
        text_lower = text.lower()
        embedding[0] = len(text) / 1000  # Length normalized
        embedding[1] = text_lower.count(' ') / 100  # Word count proxy
        embedding[2] = sum(c.isdigit() for c in text) / len(text)  # Digit ratio
        embedding[3] = sum(c.isupper() for c in text) / len(text)  # Uppercase ratio

        # Simple hash-based features
        for i, char in enumerate(text[:100]):
            idx = 4 + (hash(char) % (self.dimension - 4))
            embedding[idx] += 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calcula similaridade de cosseno entre dois embeddings.

        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding

        Returns:
            Similaridade (0-1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def create_transaction_embedder(
    config: Optional[Dict[str, Any]] = None
) -> TransactionEmbedder:
    """Factory function para criar TransactionEmbedder"""
    return TransactionEmbedder(config)


def create_text_embedder(
    config: Optional[Dict[str, Any]] = None
) -> TextEmbedder:
    """Factory function para criar TextEmbedder"""
    return TextEmbedder(config)


# Aliases para compatibilidade
TxnEmbedder = TransactionEmbedder
MessageEmbedder = TextEmbedder
