"""
Sankofa Enterprise Pro - Keystroke Analyzer
Análise de dinâmica de digitação

Baseado em:
- Keystroke dynamics research (CMU)
- BioCatch typing patterns
- IBM Keystroke Authentication
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeystrokePattern:
    """Padrão de digitação extraído"""
    risk_score: float
    typing_speed: float  # chars/segundo
    hold_time_mean: float  # ms
    hold_time_std: float
    flight_time_mean: float  # ms
    flight_time_std: float
    error_rate: float
    pause_count: int
    rhythm_consistency: float  # 0-1
    digraph_timings: Dict[str, float]
    anomalies: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KeyEvent:
    """Evento de tecla"""
    key: str
    timestamp: float  # ms
    event_type: str  # "keydown" or "keyup"


class KeystrokeAnalyzer:
    """
    Analisador de Dinâmica de Digitação

    Métricas analisadas:
    - Hold time (tempo de pressão da tecla)
    - Flight time (tempo entre teclas)
    - Digraph timing (pares de teclas)
    - Typing rhythm (consistência do ritmo)
    - Error patterns (correções e backspaces)
    """

    VERSION = "1.0.0"

    # Thresholds de anomalia
    SPEED_ANOMALY_THRESHOLD = 3.0  # Z-score
    HOLD_TIME_ANOMALY_THRESHOLD = 2.5
    FLIGHT_TIME_ANOMALY_THRESHOLD = 2.5
    BOT_SPEED_THRESHOLD = 15.0  # chars/segundo (sobre-humano)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Valores típicos de referência
        self.reference_hold_time = self.config.get("ref_hold_time", 100)  # ms
        self.reference_flight_time = self.config.get("ref_flight_time", 150)  # ms
        self.reference_speed = self.config.get("ref_speed", 5.0)  # chars/s

        logger.info(f"KeystrokeAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        events: List[Dict[str, Any]],
        baseline: Optional[Dict[str, float]] = None
    ) -> KeystrokePattern:
        """
        Analisar eventos de digitação

        Args:
            events: Lista de eventos de tecla
            baseline: Baseline do usuário (opcional)

        Returns:
            KeystrokePattern com análise completa
        """
        if not events:
            return self._empty_pattern()

        # Converter para KeyEvent
        key_events = self._parse_events(events)

        if len(key_events) < 3:
            return self._empty_pattern()

        anomalies = []
        risk_score = 0.0

        # 1. Calcular hold times
        hold_times = self._calculate_hold_times(key_events)
        hold_mean = np.mean(hold_times) if hold_times else self.reference_hold_time
        hold_std = np.std(hold_times) if len(hold_times) > 1 else 0

        # 2. Calcular flight times
        flight_times = self._calculate_flight_times(key_events)
        flight_mean = np.mean(flight_times) if flight_times else self.reference_flight_time
        flight_std = np.std(flight_times) if len(flight_times) > 1 else 0

        # 3. Calcular velocidade de digitação
        typing_speed = self._calculate_typing_speed(key_events)

        # 4. Calcular taxa de erros
        error_rate = self._calculate_error_rate(key_events)

        # 5. Contar pausas longas
        pause_count = self._count_pauses(key_events)

        # 6. Calcular consistência de ritmo
        rhythm_consistency = self._calculate_rhythm_consistency(flight_times)

        # 7. Calcular digraph timings
        digraph_timings = self._calculate_digraph_timings(key_events)

        # 8. Detectar anomalias
        if baseline:
            anomalies, risk_score = self._detect_anomalies_with_baseline(
                typing_speed, hold_mean, hold_std, flight_mean, flight_std,
                error_rate, rhythm_consistency, digraph_timings, baseline
            )
        else:
            anomalies, risk_score = self._detect_anomalies_reference(
                typing_speed, hold_mean, hold_std, flight_mean, flight_std,
                error_rate, rhythm_consistency
            )

        # 9. Calcular confiança
        confidence = self._calculate_confidence(key_events, baseline)

        return KeystrokePattern(
            risk_score=risk_score,
            typing_speed=typing_speed,
            hold_time_mean=hold_mean,
            hold_time_std=hold_std,
            flight_time_mean=flight_mean,
            flight_time_std=flight_std,
            error_rate=error_rate,
            pause_count=pause_count,
            rhythm_consistency=rhythm_consistency,
            digraph_timings=digraph_timings,
            anomalies=anomalies,
            confidence=confidence
        )

    def _parse_events(self, events: List[Dict[str, Any]]) -> List[KeyEvent]:
        """Converter eventos raw para KeyEvent"""
        key_events = []

        for event in events:
            key = event.get("key", "")
            timestamp = event.get("timestamp", 0)
            event_type = event.get("type", event.get("event_type", "keydown"))

            key_events.append(KeyEvent(
                key=key,
                timestamp=float(timestamp),
                event_type=event_type
            ))

        # Ordenar por timestamp
        key_events.sort(key=lambda x: x.timestamp)

        return key_events

    def _calculate_hold_times(self, events: List[KeyEvent]) -> List[float]:
        """
        Calcular hold times (tempo de pressão de tecla)

        Hold time = tempo entre keydown e keyup da mesma tecla
        """
        hold_times = []
        pending_keydowns: Dict[str, float] = {}

        for event in events:
            if event.event_type == "keydown":
                pending_keydowns[event.key] = event.timestamp
            elif event.event_type == "keyup" and event.key in pending_keydowns:
                hold_time = event.timestamp - pending_keydowns[event.key]
                if 0 < hold_time < 1000:  # Filtrar valores absurdos
                    hold_times.append(hold_time)
                del pending_keydowns[event.key]

        return hold_times

    def _calculate_flight_times(self, events: List[KeyEvent]) -> List[float]:
        """
        Calcular flight times (tempo entre teclas)

        Flight time = tempo entre keyup de uma tecla e keydown da próxima
        """
        flight_times = []
        last_keyup_time = None

        for event in events:
            if event.event_type == "keyup":
                last_keyup_time = event.timestamp
            elif event.event_type == "keydown" and last_keyup_time is not None:
                flight_time = event.timestamp - last_keyup_time
                if -100 < flight_time < 2000:  # Permitir pequena sobreposição
                    flight_times.append(max(0, flight_time))

        return flight_times

    def _calculate_typing_speed(self, events: List[KeyEvent]) -> float:
        """Calcular velocidade de digitação em caracteres/segundo"""
        keydowns = [e for e in events if e.event_type == "keydown"]

        if len(keydowns) < 2:
            return 0

        first_ts = keydowns[0].timestamp
        last_ts = keydowns[-1].timestamp
        duration_seconds = (last_ts - first_ts) / 1000

        if duration_seconds > 0:
            return len(keydowns) / duration_seconds

        return 0

    def _calculate_error_rate(self, events: List[KeyEvent]) -> float:
        """Calcular taxa de erros (backspaces e deletes)"""
        keydowns = [e for e in events if e.event_type == "keydown"]
        total_keys = len(keydowns)

        if total_keys == 0:
            return 0

        error_keys = sum(1 for e in keydowns if e.key in ["Backspace", "Delete"])

        return error_keys / total_keys

    def _count_pauses(
        self,
        events: List[KeyEvent],
        threshold_ms: float = 1500
    ) -> int:
        """Contar pausas longas durante digitação"""
        flight_times = self._calculate_flight_times(events)
        return sum(1 for ft in flight_times if ft > threshold_ms)

    def _calculate_rhythm_consistency(self, flight_times: List[float]) -> float:
        """
        Calcular consistência do ritmo de digitação

        Quanto menor o coeficiente de variação, mais consistente
        Retorna 0-1, onde 1 é muito consistente
        """
        if len(flight_times) < 3:
            return 0.5

        mean_ft = np.mean(flight_times)
        std_ft = np.std(flight_times)

        if mean_ft > 0:
            cv = std_ft / mean_ft  # Coeficiente de variação

            # Converter para score 0-1 (inversamente proporcional)
            # CV típico humano: 0.3-0.6
            # CV de bot: < 0.1
            consistency = max(0, min(1, 1 - cv))
            return consistency

        return 0.5

    def _calculate_digraph_timings(
        self,
        events: List[KeyEvent]
    ) -> Dict[str, float]:
        """
        Calcular timings de digraphs (pares de teclas)

        Muito característico de cada pessoa
        """
        digraph_timings = defaultdict(list)
        keydowns = [e for e in events if e.event_type == "keydown"]

        for i in range(len(keydowns) - 1):
            key1 = keydowns[i].key.lower()
            key2 = keydowns[i + 1].key.lower()

            # Apenas caracteres alfanuméricos
            if key1.isalnum() and key2.isalnum():
                digraph = f"{key1}{key2}"
                timing = keydowns[i + 1].timestamp - keydowns[i].timestamp

                if 0 < timing < 1000:
                    digraph_timings[digraph].append(timing)

        # Calcular médias
        return {
            digraph: np.mean(timings)
            for digraph, timings in digraph_timings.items()
            if len(timings) >= 2
        }

    def _detect_anomalies_with_baseline(
        self,
        speed: float,
        hold_mean: float,
        hold_std: float,
        flight_mean: float,
        flight_std: float,
        error_rate: float,
        rhythm: float,
        digraphs: Dict[str, float],
        baseline: Dict[str, float]
    ) -> Tuple[List[str], float]:
        """Detectar anomalias comparando com baseline do usuário"""
        anomalies = []
        risk_score = 0.0

        # Comparar velocidade
        baseline_speed = baseline.get("avg_speed", speed)
        baseline_speed_std = baseline.get("std_speed", baseline_speed * 0.2)
        if baseline_speed_std > 0:
            speed_zscore = abs(speed - baseline_speed) / baseline_speed_std
            if speed_zscore > self.SPEED_ANOMALY_THRESHOLD:
                anomalies.append(f"Velocidade anormal (z={speed_zscore:.1f})")
                risk_score += 0.25

        # Comparar hold time
        baseline_hold = baseline.get("avg_hold_time", hold_mean)
        baseline_hold_std = baseline.get("std_hold_time", baseline_hold * 0.3)
        if baseline_hold_std > 0:
            hold_zscore = abs(hold_mean - baseline_hold) / baseline_hold_std
            if hold_zscore > self.HOLD_TIME_ANOMALY_THRESHOLD:
                anomalies.append(f"Hold time anormal (z={hold_zscore:.1f})")
                risk_score += 0.2

        # Comparar flight time
        baseline_flight = baseline.get("avg_flight_time", flight_mean)
        baseline_flight_std = baseline.get("std_flight_time", baseline_flight * 0.3)
        if baseline_flight_std > 0:
            flight_zscore = abs(flight_mean - baseline_flight) / baseline_flight_std
            if flight_zscore > self.FLIGHT_TIME_ANOMALY_THRESHOLD:
                anomalies.append(f"Flight time anormal (z={flight_zscore:.1f})")
                risk_score += 0.2

        # Comparar taxa de erro
        baseline_error = baseline.get("avg_error_rate", error_rate)
        if baseline_error > 0:
            error_ratio = error_rate / baseline_error
            if error_ratio > 2.5:
                anomalies.append(f"Taxa de erro muito alta ({error_ratio:.1f}x)")
                risk_score += 0.15

        # Comparar ritmo
        baseline_rhythm = baseline.get("rhythm_consistency", rhythm)
        if abs(rhythm - baseline_rhythm) > 0.3:
            anomalies.append("Ritmo de digitação inconsistente")
            risk_score += 0.1

        # Comparar digraphs
        baseline_digraphs = baseline.get("digraph_timings", {})
        if baseline_digraphs and digraphs:
            digraph_deviation = self._compare_digraphs(digraphs, baseline_digraphs)
            if digraph_deviation > 0.5:
                anomalies.append(f"Padrão de digraphs alterado ({digraph_deviation:.1%})")
                risk_score += 0.1

        # Detectar bot
        if speed > self.BOT_SPEED_THRESHOLD:
            anomalies.append(f"Velocidade sobre-humana ({speed:.1f} chars/s)")
            risk_score += 0.3

        if rhythm > 0.95 and speed > 3:
            anomalies.append("Ritmo muito uniforme (possível bot)")
            risk_score += 0.2

        return anomalies, min(risk_score, 1.0)

    def _detect_anomalies_reference(
        self,
        speed: float,
        hold_mean: float,
        hold_std: float,
        flight_mean: float,
        flight_std: float,
        error_rate: float,
        rhythm: float
    ) -> Tuple[List[str], float]:
        """Detectar anomalias usando valores de referência"""
        anomalies = []
        risk_score = 0.0

        # Verificar velocidade sobre-humana
        if speed > self.BOT_SPEED_THRESHOLD:
            anomalies.append(f"Velocidade sobre-humana ({speed:.1f} chars/s)")
            risk_score += 0.35

        # Verificar hold time muito baixo (bot)
        if hold_mean < 30:  # < 30ms é muito rápido
            anomalies.append(f"Hold time muito baixo ({hold_mean:.0f}ms)")
            risk_score += 0.25

        # Verificar ritmo muito uniforme (bot)
        if rhythm > 0.95 and speed > 3:
            anomalies.append("Ritmo muito uniforme (possível automação)")
            risk_score += 0.25

        # Verificar variação muito baixa (bot)
        if hold_std < 5 and len(str(hold_std)) > 0:
            anomalies.append("Variação de hold time muito baixa")
            risk_score += 0.15

        if flight_std < 10 and flight_mean > 0:
            anomalies.append("Variação de flight time muito baixa")
            risk_score += 0.15

        # Verificar muitos erros (possível nervosismo/fraude)
        if error_rate > 0.3:
            anomalies.append(f"Taxa de erro muito alta ({error_rate:.1%})")
            risk_score += 0.1

        return anomalies, min(risk_score, 1.0)

    def _compare_digraphs(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float]
    ) -> float:
        """Comparar digraphs atuais com baseline"""
        common_digraphs = set(current.keys()) & set(baseline.keys())

        if not common_digraphs:
            return 0.5  # Não há dados suficientes

        deviations = []
        for digraph in common_digraphs:
            curr_timing = current[digraph]
            base_timing = baseline[digraph]

            if base_timing > 0:
                deviation = abs(curr_timing - base_timing) / base_timing
                deviations.append(deviation)

        return np.mean(deviations) if deviations else 0

    def _calculate_confidence(
        self,
        events: List[KeyEvent],
        baseline: Optional[Dict]
    ) -> float:
        """Calcular confiança na análise"""
        confidence = 0.0

        # Quantidade de eventos
        keydowns = [e for e in events if e.event_type == "keydown"]
        if len(keydowns) >= 50:
            confidence += 0.4
        elif len(keydowns) >= 20:
            confidence += 0.25
        elif len(keydowns) >= 10:
            confidence += 0.15

        # Presença de baseline
        if baseline:
            confidence += 0.3
            if baseline.get("samples_count", 0) > 10:
                confidence += 0.15

        # Qualidade dos dados
        keyups = [e for e in events if e.event_type == "keyup"]
        if len(keyups) > len(keydowns) * 0.8:  # Dados pareados
            confidence += 0.15

        return min(confidence, 1.0)

    def _empty_pattern(self) -> KeystrokePattern:
        """Retornar padrão vazio quando não há dados"""
        return KeystrokePattern(
            risk_score=0.5,
            typing_speed=0,
            hold_time_mean=0,
            hold_time_std=0,
            flight_time_mean=0,
            flight_time_std=0,
            error_rate=0,
            pause_count=0,
            rhythm_consistency=0.5,
            digraph_timings={},
            anomalies=[],
            confidence=0
        )

    def extract_baseline_features(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extrair features para criar baseline"""
        pattern = self.analyze(events, None)

        return {
            "avg_speed": pattern.typing_speed,
            "std_speed": pattern.typing_speed * 0.2,  # Estimativa inicial
            "avg_hold_time": pattern.hold_time_mean,
            "std_hold_time": pattern.hold_time_std,
            "avg_flight_time": pattern.flight_time_mean,
            "std_flight_time": pattern.flight_time_std,
            "avg_error_rate": pattern.error_rate,
            "rhythm_consistency": pattern.rhythm_consistency,
            "digraph_timings": pattern.digraph_timings
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "reference_values": {
                "hold_time": self.reference_hold_time,
                "flight_time": self.reference_flight_time,
                "speed": self.reference_speed
            },
            "thresholds": {
                "speed_anomaly": self.SPEED_ANOMALY_THRESHOLD,
                "hold_time_anomaly": self.HOLD_TIME_ANOMALY_THRESHOLD,
                "flight_time_anomaly": self.FLIGHT_TIME_ANOMALY_THRESHOLD,
                "bot_speed": self.BOT_SPEED_THRESHOLD
            }
        }


def create_keystroke_analyzer(
    config: Optional[Dict[str, Any]] = None
) -> KeystrokeAnalyzer:
    """Factory function para criar KeystrokeAnalyzer"""
    return KeystrokeAnalyzer(config)
