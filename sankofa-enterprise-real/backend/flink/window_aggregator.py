"""
Window Aggregator - Computes time-windowed features
Simulates Flink window operations for feature computation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a time window"""
    name: str
    duration_seconds: int
    slide_seconds: Optional[int] = None  # For sliding windows


class WindowAggregator:
    """
    Computes aggregations over time windows

    Simula Flink window operations:
    - Tumbling windows (non-overlapping)
    - Sliding windows (overlapping)
    - Session windows (event-based)

    Windows suportados:
    - 5 minutos (300s)
    - 1 hora (3600s)
    - 24 horas (86400s)
    - 7 dias (604800s)
    - 30 dias (2592000s)
    """

    # Window configurations
    WINDOWS = [
        WindowConfig(name='5m', duration_seconds=300),
        WindowConfig(name='1h', duration_seconds=3600),
        WindowConfig(name='24h', duration_seconds=86400),
        WindowConfig(name='7d', duration_seconds=604800),
        WindowConfig(name='30d', duration_seconds=2592000),
    ]

    def __init__(self, max_events_per_window: int = 10000):
        """
        Args:
            max_events_per_window: Max events to keep in memory per window
        """
        # In-memory storage (em produção seria Flink state)
        self.events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_events_per_window))
        self.max_events = max_events_per_window

        logger.info(f"Window Aggregator initialized: max_events={max_events_per_window}")

    async def add_event(
        self,
        entity_id: str,
        event: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Adiciona evento para agregação

        Args:
            entity_id: Entity ID (customer_id, device_id, etc.)
            event: Event data
            timestamp: Event timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Add timestamp to event
        event['_timestamp'] = timestamp

        # Store event
        self.events[entity_id].append(event)

    async def compute_features(
        self,
        entity_id: str,
        windows: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computa features para todas as windows

        Args:
            entity_id: Entity ID
            windows: Windows to compute (default: all)

        Returns:
            Dict com features agregadas
        """
        if windows is None:
            windows = [w.name for w in self.WINDOWS]

        features = {}

        for window_name in windows:
            window_features = await self._compute_window_features(entity_id, window_name)
            features.update(window_features)

        return features

    async def _compute_window_features(
        self,
        entity_id: str,
        window_name: str
    ) -> Dict[str, Any]:
        """
        Computa features para uma window específica

        Args:
            entity_id: Entity ID
            window_name: Window name (5m, 1h, etc.)

        Returns:
            Features da window
        """
        # Get window config
        window_config = next((w for w in self.WINDOWS if w.name == window_name), None)
        if not window_config:
            return {}

        # Get events within window
        events = self._get_events_in_window(entity_id, window_config.duration_seconds)

        if not events:
            return self._get_default_window_features(window_name)

        # Compute aggregations
        features = {}

        # Count
        features[f'txn_count_{window_name}'] = len(events)

        # Amount aggregations
        amounts = [e.get('amount', 0) for e in events]
        if amounts:
            features[f'amount_sum_{window_name}'] = sum(amounts)
            features[f'amount_avg_{window_name}'] = statistics.mean(amounts)
            features[f'amount_max_{window_name}'] = max(amounts)
            features[f'amount_min_{window_name}'] = min(amounts)
            features[f'amount_std_{window_name}'] = statistics.stdev(amounts) if len(amounts) > 1 else 0.0

        # Velocity
        if window_name in ['5m', '1h']:
            features[f'velocity_{window_name}'] = len(events)

        # Unique counts
        features[f'unique_merchants_{window_name}'] = len(set(e.get('merchant_id', '') for e in events))
        features[f'unique_devices_{window_name}'] = len(set(e.get('device_id', '') for e in events))
        features[f'unique_locations_{window_name}'] = len(set(e.get('location', '') for e in events))

        # Failed transactions
        failed_count = sum(1 for e in events if e.get('status') == 'failed')
        features[f'failed_txn_count_{window_name}'] = failed_count
        features[f'failed_rate_{window_name}'] = failed_count / len(events) if events else 0.0

        # Cross-border
        cross_border = sum(1 for e in events if e.get('is_cross_border', False))
        features[f'cross_border_count_{window_name}'] = cross_border

        # Device changes
        if window_name in ['24h', '7d']:
            devices = [e.get('device_id', '') for e in events]
            features[f'device_changes_{window_name}'] = len(set(devices)) - 1 if len(devices) > 1 else 0

        # Channel distribution
        channels = [e.get('channel', '') for e in events]
        channel_dist = defaultdict(int)
        for ch in channels:
            channel_dist[ch] += 1

        for channel, count in channel_dist.items():
            features[f'channel_{channel.lower()}_{window_name}'] = count

        return features

    def _get_events_in_window(
        self,
        entity_id: str,
        duration_seconds: int
    ) -> List[Dict[str, Any]]:
        """
        Retorna eventos dentro da window

        Args:
            entity_id: Entity ID
            duration_seconds: Window duration

        Returns:
            List of events in window
        """
        if entity_id not in self.events:
            return []

        cutoff_time = datetime.utcnow() - timedelta(seconds=duration_seconds)

        events_in_window = [
            e for e in self.events[entity_id]
            if e.get('_timestamp', datetime.min) >= cutoff_time
        ]

        return events_in_window

    def _get_default_window_features(self, window_name: str) -> Dict[str, Any]:
        """
        Retorna features default (zeros) para window vazia

        Args:
            window_name: Window name

        Returns:
            Default features
        """
        return {
            f'txn_count_{window_name}': 0,
            f'amount_sum_{window_name}': 0.0,
            f'amount_avg_{window_name}': 0.0,
            f'velocity_{window_name}': 0,
            f'unique_merchants_{window_name}': 0,
            f'failed_txn_count_{window_name}': 0,
        }

    async def compute_session_features(
        self,
        entity_id: str,
        session_gap_seconds: int = 1800  # 30 minutes
    ) -> Dict[str, Any]:
        """
        Computa features baseadas em sessions (event-time gaps)

        Args:
            entity_id: Entity ID
            session_gap_seconds: Gap para considerar nova session

        Returns:
            Session features
        """
        if entity_id not in self.events:
            return {'session_count': 0, 'avg_session_duration': 0.0}

        events = sorted(self.events[entity_id], key=lambda e: e.get('_timestamp', datetime.min))

        if not events:
            return {'session_count': 0, 'avg_session_duration': 0.0}

        # Detect sessions
        sessions = []
        current_session = [events[0]]

        for i in range(1, len(events)):
            time_gap = (events[i]['_timestamp'] - events[i-1]['_timestamp']).total_seconds()

            if time_gap > session_gap_seconds:
                # New session
                sessions.append(current_session)
                current_session = [events[i]]
            else:
                current_session.append(events[i])

        sessions.append(current_session)

        # Compute session features
        session_durations = []
        for session in sessions:
            if len(session) > 1:
                duration = (session[-1]['_timestamp'] - session[0]['_timestamp']).total_seconds()
                session_durations.append(duration)

        features = {
            'session_count': len(sessions),
            'avg_session_duration': statistics.mean(session_durations) if session_durations else 0.0,
            'max_session_duration': max(session_durations) if session_durations else 0.0,
            'avg_events_per_session': statistics.mean([len(s) for s in sessions]),
        }

        return features

    def cleanup_old_events(self, retention_seconds: int = 2592000) -> int:
        """
        Remove eventos antigos (garbage collection)

        Args:
            retention_seconds: Retention period (default: 30 days)

        Returns:
            Number of events removed
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=retention_seconds)
        removed_count = 0

        for entity_id in list(self.events.keys()):
            original_count = len(self.events[entity_id])

            # Filter events
            self.events[entity_id] = deque(
                [e for e in self.events[entity_id] if e.get('_timestamp', datetime.min) >= cutoff_time],
                maxlen=self.max_events
            )

            removed = original_count - len(self.events[entity_id])
            removed_count += removed

            # Remove empty entities
            if not self.events[entity_id]:
                del self.events[entity_id]

        logger.info(f"Cleanup: removed {removed_count} old events")
        return removed_count


# Singleton instance
_aggregator_instance: Optional[WindowAggregator] = None


def get_window_aggregator() -> WindowAggregator:
    """
    Retorna singleton do Window Aggregator

    Returns:
        WindowAggregator instance
    """
    global _aggregator_instance

    if _aggregator_instance is None:
        _aggregator_instance = WindowAggregator(max_events_per_window=10000)

    return _aggregator_instance
