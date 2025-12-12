"""
Sankofa Enterprise Pro - LLM Integration Module
Integração com Large Language Models para análise de fraude

Baseado em:
- Mistral-7B-LLM-Fraud-Detection (Hugging Face)
- FinBERT for financial text analysis
- GPT-4 for explanations
"""

from .fraud_llm import (
    FraudLLMAnalyzer,
    ConversationRisk,
    ScamAnalysis,
    create_fraud_llm
)

from .scam_detector import (
    ScamPatternDetector,
    ScamType,
    ScamIndicator,
    create_scam_detector
)

from .explanation_generator import (
    ExplanationGenerator,
    FraudExplanation,
    create_explanation_generator
)

from .embeddings import (
    TransactionEmbedder,
    TextEmbedder,
    create_transaction_embedder,
    create_text_embedder
)

__all__ = [
    "FraudLLMAnalyzer",
    "ConversationRisk",
    "ScamAnalysis",
    "create_fraud_llm",
    "ScamPatternDetector",
    "ScamType",
    "ScamIndicator",
    "create_scam_detector",
    "ExplanationGenerator",
    "FraudExplanation",
    "create_explanation_generator",
    "TransactionEmbedder",
    "TextEmbedder",
    "create_transaction_embedder",
    "create_text_embedder",
]

__version__ = "1.0.0"
