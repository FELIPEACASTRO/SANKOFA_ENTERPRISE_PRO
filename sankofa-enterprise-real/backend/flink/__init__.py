"""
Apache Flink Integration - Real-time Feature Engineering
Window-based aggregations, stateful processing
"""

from .feature_store import FlinkFeatureStore
from .window_aggregator import WindowAggregator
from .feature_materializer import FeatureMaterializer

__all__ = [
    'FlinkFeatureStore',
    'WindowAggregator',
    'FeatureMaterializer'
]
