"""
Sankofa Enterprise Pro - Advanced Feature Engineering Module
Sistema de Feature Engineering com 500+ features para detecção de fraude

Baseado em:
- IEEE-CIS Fraud Detection winning solution
- Framework Bahnsen (velocity features)
- Feedzai AutoML feature engineering
- Academic research on financial fraud detection
"""

from .feature_generator import (
    FeatureGenerator,
    create_feature_generator
)

from .temporal_features import (
    TemporalFeatureGenerator,
    create_temporal_generator
)

from .velocity_features import (
    VelocityFeatureGenerator,
    create_velocity_generator
)

from .aggregation_features import (
    AggregationFeatureGenerator,
    create_aggregation_generator
)

from .graph_features import (
    GraphFeatureGenerator,
    create_graph_feature_generator
)

from .embedding_features import (
    EmbeddingFeatureGenerator,
    create_embedding_generator
)

__all__ = [
    # Main generator
    "FeatureGenerator",
    "create_feature_generator",

    # Temporal
    "TemporalFeatureGenerator",
    "create_temporal_generator",

    # Velocity (Bahnsen)
    "VelocityFeatureGenerator",
    "create_velocity_generator",

    # Aggregation
    "AggregationFeatureGenerator",
    "create_aggregation_generator",

    # Graph
    "GraphFeatureGenerator",
    "create_graph_feature_generator",

    # Embeddings
    "EmbeddingFeatureGenerator",
    "create_embedding_generator",
]

__version__ = "1.0.0"
