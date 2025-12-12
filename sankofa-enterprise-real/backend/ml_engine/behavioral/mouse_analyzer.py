"""
Sankofa Enterprise Pro - Mouse Analyzer
Análise de dinâmica de movimentos de mouse/touch

Baseado em:
- BioCatch mouse dynamics
- Academic research on mouse biometrics
- Touch dynamics for mobile
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MousePattern:
    """Padrão de movimento de mouse extraído"""
    risk_score: float
    avg_velocity: float  # pixels/ms
    max_velocity: float
    velocity_std: float
    avg_acceleration: float
    curvature_score: float  # 0-1, 1=muito reto
    jitter_score: float  # 0-1, 1=muito tremido
    click_accuracy: float  # 0-1
    pause_count: int
    movement_efficiency: float  # 0-1
    direction_changes: int
    anomalies: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MouseEvent:
    """Evento de mouse/touch"""
    x: float
    y: float
    timestamp: float  # ms
    event_type: str  # "move", "click", "scroll"
    pressure: Optional[float] = None  # Para touch
    button: Optional[str] = None  # Para cliques


class MouseAnalyzer:
    """
    Analisador de Dinâmica de Mouse/Touch

    Métricas analisadas:
    - Velocidade e aceleração de movimento
    - Curvatura das trajetórias
    - Jitter (micro-movimentos/tremores)
    - Eficiência de movimento (linha reta vs caminho real)
    - Padrões de clique
    - Mudanças de direção
    """

    VERSION = "1.0.0"

    # Thresholds
    BOT_VELOCITY_THRESHOLD = 5.0  # pixels/ms (muito rápido)
    BOT_LINEARITY_THRESHOLD = 0.98  # Movimentos muito retos
    HUMAN_JITTER_MIN = 0.02  # Humanos têm algum tremor
    HUMAN_JITTER_MAX = 0.5  # Mas não muito

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        logger.info(f"MouseAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        events: List[Dict[str, Any]],
        baseline: Optional[Dict[str, float]] = None
    ) -> MousePattern:
        """
        Analisar eventos de mouse/touch

        Args:
            events: Lista de eventos de mouse
            baseline: Baseline do usuário (opcional)

        Returns:
            MousePattern com análise completa
        """
        if not events or len(events) < 3:
            return self._empty_pattern()

        # Converter para MouseEvent
        mouse_events = self._parse_events(events)

        if len(mouse_events) < 3:
            return self._empty_pattern()

        anomalies = []
        risk_score = 0.0

        # 1. Calcular velocidades
        velocities = self._calculate_velocities(mouse_events)
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        velocity_std = np.std(velocities) if len(velocities) > 1 else 0

        # 2. Calcular acelerações
        accelerations = self._calculate_accelerations(velocities)
        avg_acceleration = np.mean(np.abs(accelerations)) if accelerations else 0

        # 3. Calcular curvatura (linearidade)
        curvature_score = self._calculate_curvature(mouse_events)

        # 4. Calcular jitter
        jitter_score = self._calculate_jitter(mouse_events)

        # 5. Calcular eficiência de movimento
        movement_efficiency = self._calculate_efficiency(mouse_events)

        # 6. Contar mudanças de direção
        direction_changes = self._count_direction_changes(mouse_events)

        # 7. Calcular precisão de cliques
        click_accuracy = self._calculate_click_accuracy(mouse_events)

        # 8. Contar pausas
        pause_count = self._count_pauses(mouse_events)

        # 9. Detectar anomalias
        if baseline:
            anomalies, risk_score = self._detect_anomalies_with_baseline(
                avg_velocity, velocity_std, curvature_score, jitter_score,
                movement_efficiency, direction_changes, baseline
            )
        else:
            anomalies, risk_score = self._detect_anomalies_reference(
                avg_velocity, max_velocity, velocity_std, curvature_score,
                jitter_score, movement_efficiency, direction_changes
            )

        # 10. Calcular confiança
        confidence = self._calculate_confidence(mouse_events, baseline)

        return MousePattern(
            risk_score=risk_score,
            avg_velocity=avg_velocity,
            max_velocity=max_velocity,
            velocity_std=velocity_std,
            avg_acceleration=avg_acceleration,
            curvature_score=curvature_score,
            jitter_score=jitter_score,
            click_accuracy=click_accuracy,
            pause_count=pause_count,
            movement_efficiency=movement_efficiency,
            direction_changes=direction_changes,
            anomalies=anomalies,
            confidence=confidence
        )

    def _parse_events(self, events: List[Dict[str, Any]]) -> List[MouseEvent]:
        """Converter eventos raw para MouseEvent"""
        mouse_events = []

        for event in events:
            x = event.get("x", event.get("clientX", 0))
            y = event.get("y", event.get("clientY", 0))
            timestamp = event.get("timestamp", 0)
            event_type = event.get("type", event.get("event_type", "move"))
            pressure = event.get("pressure", event.get("force"))
            button = event.get("button")

            mouse_events.append(MouseEvent(
                x=float(x),
                y=float(y),
                timestamp=float(timestamp),
                event_type=event_type,
                pressure=pressure,
                button=button
            ))

        # Ordenar por timestamp
        mouse_events.sort(key=lambda x: x.timestamp)

        return mouse_events

    def _calculate_velocities(self, events: List[MouseEvent]) -> List[float]:
        """Calcular velocidades entre pontos consecutivos"""
        velocities = []

        for i in range(1, len(events)):
            dx = events[i].x - events[i-1].x
            dy = events[i].y - events[i-1].y
            dt = events[i].timestamp - events[i-1].timestamp

            distance = np.sqrt(dx**2 + dy**2)

            if dt > 0:
                velocity = distance / dt
                if velocity < 100:  # Filtrar valores absurdos
                    velocities.append(velocity)

        return velocities

    def _calculate_accelerations(self, velocities: List[float]) -> List[float]:
        """Calcular acelerações a partir das velocidades"""
        if len(velocities) < 2:
            return []

        return [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]

    def _calculate_curvature(self, events: List[MouseEvent]) -> float:
        """
        Calcular score de curvatura (linearidade)

        Retorna 0-1, onde:
        - 1 = movimentos muito retos (suspeito de bot)
        - 0 = movimentos muito curvos
        """
        if len(events) < 3:
            return 0.5

        # Calcular ângulos entre segmentos consecutivos
        angles = []

        for i in range(1, len(events) - 1):
            # Vetores
            v1 = (events[i].x - events[i-1].x, events[i].y - events[i-1].y)
            v2 = (events[i+1].x - events[i].x, events[i+1].y - events[i].y)

            # Magnitudes
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 0.1 and mag2 > 0.1:  # Filtrar movimentos muito pequenos
                # Produto escalar
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp

                angles.append(cos_angle)

        if not angles:
            return 0.5

        # Média dos cossenos (1 = reto, -1 = volta)
        avg_cos = np.mean(angles)

        # Normalizar para 0-1
        return (avg_cos + 1) / 2

    def _calculate_jitter(self, events: List[MouseEvent]) -> float:
        """
        Calcular jitter (micro-movimentos/tremores)

        Retorna 0-1, onde:
        - 0 = sem tremor (possível bot)
        - 1 = muito tremor (possível nervosismo ou problema)
        """
        if len(events) < 5:
            return 0.5

        # Contar mudanças de direção rápidas
        direction_reversals = 0

        for i in range(2, len(events)):
            dx1 = events[i-1].x - events[i-2].x
            dy1 = events[i-1].y - events[i-2].y
            dx2 = events[i].x - events[i-1].x
            dy2 = events[i].y - events[i-1].y

            # Reversão de direção
            if (dx1 * dx2 < 0) or (dy1 * dy2 < 0):
                direction_reversals += 1

        # Normalizar pelo número de movimentos
        jitter = direction_reversals / (len(events) - 2)

        return min(jitter, 1.0)

    def _calculate_efficiency(self, events: List[MouseEvent]) -> float:
        """
        Calcular eficiência de movimento

        Eficiência = distância direta / distância percorrida
        Retorna 0-1, onde 1 = perfeitamente eficiente
        """
        if len(events) < 2:
            return 0.5

        # Distância direta (linha reta)
        direct_distance = np.sqrt(
            (events[-1].x - events[0].x)**2 +
            (events[-1].y - events[0].y)**2
        )

        # Distância percorrida
        path_distance = 0
        for i in range(1, len(events)):
            dx = events[i].x - events[i-1].x
            dy = events[i].y - events[i-1].y
            path_distance += np.sqrt(dx**2 + dy**2)

        if path_distance > 0:
            efficiency = direct_distance / path_distance
            return min(efficiency, 1.0)

        return 0.5

    def _count_direction_changes(self, events: List[MouseEvent]) -> int:
        """Contar mudanças significativas de direção"""
        if len(events) < 3:
            return 0

        changes = 0
        threshold_angle = 0.5  # ~60 graus

        for i in range(1, len(events) - 1):
            v1 = (events[i].x - events[i-1].x, events[i].y - events[i-1].y)
            v2 = (events[i+1].x - events[i].x, events[i+1].y - events[i].y)

            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 5 and mag2 > 5:  # Movimentos significativos
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))

                if cos_angle < threshold_angle:
                    changes += 1

        return changes

    def _calculate_click_accuracy(self, events: List[MouseEvent]) -> float:
        """
        Calcular precisão de cliques

        Mede a estabilidade do mouse no momento do clique
        """
        click_events = [e for e in events if e.event_type == "click"]

        if not click_events:
            return 0.5

        # Para cada clique, verificar estabilidade antes do clique
        accuracies = []

        for click in click_events:
            # Encontrar eventos próximos antes do clique
            pre_click_events = [
                e for e in events
                if e.event_type == "move" and
                click.timestamp - 200 < e.timestamp < click.timestamp
            ]

            if len(pre_click_events) >= 2:
                # Calcular dispersão
                xs = [e.x for e in pre_click_events]
                ys = [e.y for e in pre_click_events]

                dispersion = np.std(xs) + np.std(ys)

                # Converter para accuracy (menos dispersão = mais preciso)
                accuracy = 1.0 / (1.0 + dispersion / 10)
                accuracies.append(accuracy)

        return np.mean(accuracies) if accuracies else 0.5

    def _count_pauses(
        self,
        events: List[MouseEvent],
        threshold_ms: float = 1000
    ) -> int:
        """Contar pausas no movimento"""
        pauses = 0

        for i in range(1, len(events)):
            dt = events[i].timestamp - events[i-1].timestamp
            if dt > threshold_ms:
                pauses += 1

        return pauses

    def _detect_anomalies_with_baseline(
        self,
        avg_velocity: float,
        velocity_std: float,
        curvature: float,
        jitter: float,
        efficiency: float,
        direction_changes: int,
        baseline: Dict[str, float]
    ) -> Tuple[List[str], float]:
        """Detectar anomalias comparando com baseline"""
        anomalies = []
        risk_score = 0.0

        # Comparar velocidade
        baseline_velocity = baseline.get("avg_velocity", avg_velocity)
        baseline_velocity_std = baseline.get("velocity_std", baseline_velocity * 0.3)

        if baseline_velocity_std > 0:
            velocity_zscore = abs(avg_velocity - baseline_velocity) / baseline_velocity_std
            if velocity_zscore > 2.5:
                anomalies.append(f"Velocidade anormal (z={velocity_zscore:.1f})")
                risk_score += 0.2

        # Comparar jitter
        baseline_jitter = baseline.get("avg_jitter", jitter)
        jitter_diff = abs(jitter - baseline_jitter)
        if jitter_diff > 0.2:
            if jitter < baseline_jitter:
                anomalies.append("Jitter muito baixo (possível automação)")
                risk_score += 0.25
            else:
                anomalies.append("Jitter muito alto (possível nervosismo)")
                risk_score += 0.1

        # Comparar curvatura
        baseline_curvature = baseline.get("avg_curvature", curvature)
        if abs(curvature - baseline_curvature) > 0.2:
            anomalies.append("Padrão de curvatura alterado")
            risk_score += 0.15

        # Comparar eficiência
        baseline_efficiency = baseline.get("avg_efficiency", efficiency)
        if abs(efficiency - baseline_efficiency) > 0.25:
            anomalies.append("Eficiência de movimento alterada")
            risk_score += 0.1

        # Detectar bot
        if curvature > self.BOT_LINEARITY_THRESHOLD:
            anomalies.append("Movimentos muito retos (possível bot)")
            risk_score += 0.3

        if jitter < self.HUMAN_JITTER_MIN and avg_velocity > 0.5:
            anomalies.append("Sem tremor humano detectado")
            risk_score += 0.25

        return anomalies, min(risk_score, 1.0)

    def _detect_anomalies_reference(
        self,
        avg_velocity: float,
        max_velocity: float,
        velocity_std: float,
        curvature: float,
        jitter: float,
        efficiency: float,
        direction_changes: int
    ) -> Tuple[List[str], float]:
        """Detectar anomalias usando valores de referência"""
        anomalies = []
        risk_score = 0.0

        # Velocidade sobre-humana
        if max_velocity > self.BOT_VELOCITY_THRESHOLD:
            anomalies.append(f"Velocidade sobre-humana ({max_velocity:.2f} px/ms)")
            risk_score += 0.3

        # Movimentos muito retos
        if curvature > self.BOT_LINEARITY_THRESHOLD:
            anomalies.append("Movimentos excessivamente retos")
            risk_score += 0.3

        # Sem jitter (bot)
        if jitter < self.HUMAN_JITTER_MIN and avg_velocity > 0.5:
            anomalies.append("Ausência de tremor humano natural")
            risk_score += 0.25

        # Jitter excessivo (nervosismo/problema)
        if jitter > self.HUMAN_JITTER_MAX:
            anomalies.append(f"Tremor excessivo ({jitter:.2f})")
            risk_score += 0.15

        # Eficiência perfeita (bot)
        if efficiency > 0.98 and direction_changes == 0:
            anomalies.append("Eficiência perfeita (possível automação)")
            risk_score += 0.2

        # Variação de velocidade muito baixa (bot)
        if velocity_std < 0.05 and avg_velocity > 0.3:
            anomalies.append("Velocidade muito constante")
            risk_score += 0.15

        return anomalies, min(risk_score, 1.0)

    def _calculate_confidence(
        self,
        events: List[MouseEvent],
        baseline: Optional[Dict]
    ) -> float:
        """Calcular confiança na análise"""
        confidence = 0.0

        # Quantidade de eventos
        move_events = [e for e in events if e.event_type == "move"]
        if len(move_events) >= 100:
            confidence += 0.4
        elif len(move_events) >= 50:
            confidence += 0.25
        elif len(move_events) >= 20:
            confidence += 0.15

        # Presença de baseline
        if baseline:
            confidence += 0.3
            if baseline.get("samples_count", 0) > 10:
                confidence += 0.15

        # Duração da sessão
        if events:
            duration = events[-1].timestamp - events[0].timestamp
            if duration > 30000:  # 30 segundos
                confidence += 0.15

        return min(confidence, 1.0)

    def _empty_pattern(self) -> MousePattern:
        """Retornar padrão vazio quando não há dados"""
        return MousePattern(
            risk_score=0.5,
            avg_velocity=0,
            max_velocity=0,
            velocity_std=0,
            avg_acceleration=0,
            curvature_score=0.5,
            jitter_score=0.5,
            click_accuracy=0.5,
            pause_count=0,
            movement_efficiency=0.5,
            direction_changes=0,
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
            "avg_velocity": pattern.avg_velocity,
            "velocity_std": pattern.velocity_std,
            "avg_curvature": pattern.curvature_score,
            "avg_jitter": pattern.jitter_score,
            "avg_efficiency": pattern.movement_efficiency,
            "avg_direction_changes": pattern.direction_changes
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "thresholds": {
                "bot_velocity": self.BOT_VELOCITY_THRESHOLD,
                "bot_linearity": self.BOT_LINEARITY_THRESHOLD,
                "human_jitter_min": self.HUMAN_JITTER_MIN,
                "human_jitter_max": self.HUMAN_JITTER_MAX
            }
        }


def create_mouse_analyzer(
    config: Optional[Dict[str, Any]] = None
) -> MouseAnalyzer:
    """Factory function para criar MouseAnalyzer"""
    return MouseAnalyzer(config)
