"""
Sankofa Enterprise Pro - Behavioral Analyzer
Analisador principal de biometria comportamental

Baseado em:
- BioCatch behavioral biometrics
- IBM Behavioral Authentication
- Academic research on behavioral biometrics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BehavioralScore:
    """Score de análise comportamental"""
    overall_score: float  # 0-1, quanto maior mais suspeito
    confidence: float  # Confiança na análise
    keystroke_score: float
    mouse_score: float
    device_score: float
    session_score: float
    duress_detected: bool
    bot_detected: bool
    anomalies: List[str]
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionProfile:
    """Perfil de sessão do usuário"""
    session_id: str
    user_id: str
    start_time: datetime
    duration_seconds: float
    page_views: int
    interactions: int
    idle_time_seconds: float
    device_changes: int
    ip_changes: int


@dataclass
class UserBaseline:
    """Baseline comportamental do usuário"""
    user_id: str
    keystroke_baseline: Dict[str, float]
    mouse_baseline: Dict[str, float]
    session_baseline: Dict[str, float]
    device_baseline: Dict[str, Any]
    samples_count: int
    last_updated: datetime
    confidence: float


class BehavioralAnalyzer:
    """
    Analisador Principal de Biometria Comportamental

    Combina análises de:
    - Dinâmica de digitação (keystroke)
    - Movimentos de mouse/touch
    - Padrões de sessão
    - Fingerprint de dispositivo

    Detecta:
    - Bots e automação
    - Usuários sob coerção (duress)
    - Takeover de conta
    - Fraude em tempo real
    """

    VERSION = "1.0.0"

    # Pesos dos componentes
    COMPONENT_WEIGHTS = {
        "keystroke": 0.30,
        "mouse": 0.25,
        "device": 0.20,
        "session": 0.25
    }

    # Thresholds
    ANOMALY_THRESHOLD = 0.7
    BOT_THRESHOLD = 0.8
    DURESS_THRESHOLD = 0.75

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Importar analisadores especializados
        from .keystroke_analyzer import KeystrokeAnalyzer
        from .mouse_analyzer import MouseAnalyzer
        from .device_analyzer import DeviceAnalyzer

        self.keystroke_analyzer = KeystrokeAnalyzer(config)
        self.mouse_analyzer = MouseAnalyzer(config)
        self.device_analyzer = DeviceAnalyzer(config)

        # Cache de baselines
        self.user_baselines: Dict[str, UserBaseline] = {}

        logger.info(f"BehavioralAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        user_id: str,
        session_data: Dict[str, Any],
        keystroke_events: Optional[List[Dict[str, Any]]] = None,
        mouse_events: Optional[List[Dict[str, Any]]] = None,
        device_info: Optional[Dict[str, Any]] = None
    ) -> BehavioralScore:
        """
        Análise comportamental completa

        Args:
            user_id: ID do usuário
            session_data: Dados da sessão atual
            keystroke_events: Eventos de teclado
            mouse_events: Eventos de mouse/touch
            device_info: Informações do dispositivo

        Returns:
            BehavioralScore com análise completa
        """
        logger.info(f"Analyzing behavior for user: {user_id}")

        anomalies = []
        scores = {}

        # Obter ou criar baseline do usuário
        baseline = self._get_or_create_baseline(user_id)

        # 1. Análise de Keystroke
        if keystroke_events:
            keystroke_result = self.keystroke_analyzer.analyze(
                keystroke_events,
                baseline.keystroke_baseline if baseline else None
            )
            scores["keystroke"] = keystroke_result.risk_score
            if keystroke_result.anomalies:
                anomalies.extend([f"keystroke: {a}" for a in keystroke_result.anomalies])
        else:
            scores["keystroke"] = 0.5  # Neutral se não houver dados

        # 2. Análise de Mouse
        if mouse_events:
            mouse_result = self.mouse_analyzer.analyze(
                mouse_events,
                baseline.mouse_baseline if baseline else None
            )
            scores["mouse"] = mouse_result.risk_score
            if mouse_result.anomalies:
                anomalies.extend([f"mouse: {a}" for a in mouse_result.anomalies])
        else:
            scores["mouse"] = 0.5

        # 3. Análise de Dispositivo
        if device_info:
            device_result = self.device_analyzer.analyze(
                device_info,
                baseline.device_baseline if baseline else None
            )
            scores["device"] = device_result.risk_score
            if device_result.anomalies:
                anomalies.extend([f"device: {a}" for a in device_result.anomalies])
        else:
            scores["device"] = 0.5

        # 4. Análise de Sessão
        session_score = self._analyze_session(session_data, baseline)
        scores["session"] = session_score

        # 5. Calcular score geral
        overall_score = self._calculate_overall_score(scores)

        # 6. Detectar bot
        bot_detected = self._detect_bot(
            keystroke_events or [],
            mouse_events or [],
            session_data
        )

        # 7. Detectar duress (coerção)
        duress_detected = self._detect_duress(
            keystroke_events or [],
            mouse_events or [],
            session_data,
            baseline
        )

        # 8. Calcular confiança
        confidence = self._calculate_confidence(
            keystroke_events,
            mouse_events,
            device_info,
            baseline
        )

        # 9. Gerar explicação
        explanation = self._generate_explanation(
            scores, anomalies, bot_detected, duress_detected
        )

        # 10. Atualizar baseline se comportamento normal
        if overall_score < 0.5 and confidence > 0.7:
            self._update_baseline(
                user_id, keystroke_events, mouse_events,
                device_info, session_data
            )

        return BehavioralScore(
            overall_score=overall_score,
            confidence=confidence,
            keystroke_score=scores["keystroke"],
            mouse_score=scores["mouse"],
            device_score=scores["device"],
            session_score=scores["session"],
            duress_detected=duress_detected,
            bot_detected=bot_detected,
            anomalies=anomalies,
            explanation=explanation
        )

    def _get_or_create_baseline(self, user_id: str) -> Optional[UserBaseline]:
        """Obter ou criar baseline do usuário"""

        if user_id in self.user_baselines:
            return self.user_baselines[user_id]

        # Em produção, carregar do banco de dados
        return None

    def _analyze_session(
        self,
        session_data: Dict[str, Any],
        baseline: Optional[UserBaseline]
    ) -> float:
        """Analisar padrões de sessão"""

        risk_score = 0.0

        # Duração da sessão
        duration = session_data.get("duration_seconds", 0)
        if duration < 5:  # Sessão muito curta
            risk_score += 0.3
        elif duration > 3600:  # Sessão muito longa (1h+)
            risk_score += 0.1

        # Tempo idle
        idle_ratio = session_data.get("idle_time_seconds", 0) / max(duration, 1)
        if idle_ratio > 0.8:  # Muito tempo inativo
            risk_score += 0.2

        # Mudanças de IP durante sessão
        ip_changes = session_data.get("ip_changes", 0)
        if ip_changes > 0:
            risk_score += min(0.3, ip_changes * 0.15)

        # Velocidade de navegação
        page_views = session_data.get("page_views", 0)
        if duration > 0:
            pages_per_minute = (page_views / duration) * 60
            if pages_per_minute > 10:  # Navegação muito rápida (bot)
                risk_score += 0.3

        # Comparar com baseline
        if baseline and baseline.session_baseline:
            avg_duration = baseline.session_baseline.get("avg_duration", duration)
            if abs(duration - avg_duration) > 2 * baseline.session_baseline.get("std_duration", 300):
                risk_score += 0.2

        return min(risk_score, 1.0)

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calcular score geral ponderado"""

        weighted_sum = sum(
            scores.get(component, 0.5) * weight
            for component, weight in self.COMPONENT_WEIGHTS.items()
        )

        return min(max(weighted_sum, 0.0), 1.0)

    def _detect_bot(
        self,
        keystroke_events: List[Dict[str, Any]],
        mouse_events: List[Dict[str, Any]],
        session_data: Dict[str, Any]
    ) -> bool:
        """Detectar se é um bot ou automação"""

        bot_indicators = 0

        # 1. Keystroke timing muito uniforme
        if keystroke_events:
            timings = self._extract_timings(keystroke_events)
            if len(timings) > 5:
                std_timing = np.std(timings) if len(timings) > 1 else 0
                if std_timing < 5:  # Muito uniforme (< 5ms de variação)
                    bot_indicators += 1

        # 2. Mouse movements muito lineares
        if mouse_events:
            linearity = self._calculate_mouse_linearity(mouse_events)
            if linearity > 0.95:  # Movimentos muito retos
                bot_indicators += 1

        # 3. Velocidade sobre-humana
        if keystroke_events:
            typing_speed = len(keystroke_events) / max(session_data.get("duration_seconds", 1), 1)
            if typing_speed > 20:  # Mais de 20 teclas/segundo
                bot_indicators += 1

        # 4. Sem variação de velocidade ao longo do tempo
        if mouse_events and len(mouse_events) > 10:
            velocities = self._calculate_mouse_velocities(mouse_events)
            if len(velocities) > 5:
                velocity_variation = np.std(velocities) / (np.mean(velocities) + 0.001)
                if velocity_variation < 0.1:  # Variação muito baixa
                    bot_indicators += 1

        # 5. Padrões repetitivos perfeitos
        if keystroke_events and len(keystroke_events) > 20:
            if self._detect_perfect_repetition(keystroke_events):
                bot_indicators += 1

        return bot_indicators >= 2

    def _detect_duress(
        self,
        keystroke_events: List[Dict[str, Any]],
        mouse_events: List[Dict[str, Any]],
        session_data: Dict[str, Any],
        baseline: Optional[UserBaseline]
    ) -> bool:
        """
        Detectar se usuário está sob coerção

        Indicadores:
        - Hesitação incomum
        - Tremores (mouse jitter)
        - Padrão de digitação alterado
        - Comportamento muito diferente do baseline
        """

        duress_indicators = 0

        if not baseline:
            return False  # Sem baseline, não podemos detectar

        # 1. Hesitação ao digitar valores (pausas longas)
        if keystroke_events:
            long_pauses = self._count_long_pauses(keystroke_events, threshold_ms=2000)
            normal_pauses = baseline.keystroke_baseline.get("avg_long_pauses", 0)
            if long_pauses > normal_pauses * 2:
                duress_indicators += 1

        # 2. Tremor no mouse (jitter aumentado)
        if mouse_events:
            jitter = self._calculate_mouse_jitter(mouse_events)
            normal_jitter = baseline.mouse_baseline.get("avg_jitter", jitter)
            if jitter > normal_jitter * 1.5:
                duress_indicators += 1

        # 3. Velocidade de digitação muito diferente
        if keystroke_events and baseline.keystroke_baseline:
            current_speed = self._calculate_typing_speed(keystroke_events)
            normal_speed = baseline.keystroke_baseline.get("avg_speed", current_speed)
            speed_deviation = abs(current_speed - normal_speed) / (normal_speed + 0.001)
            if speed_deviation > 0.5:  # 50% diferente
                duress_indicators += 1

        # 4. Erros de digitação aumentados
        if keystroke_events:
            error_rate = self._calculate_error_rate(keystroke_events)
            normal_error_rate = baseline.keystroke_baseline.get("avg_error_rate", error_rate)
            if error_rate > normal_error_rate * 2:
                duress_indicators += 1

        # 5. Tempo de sessão anormal
        duration = session_data.get("duration_seconds", 0)
        normal_duration = baseline.session_baseline.get("avg_duration", duration)
        if duration > normal_duration * 3:  # Muito mais longo
            duress_indicators += 1

        return duress_indicators >= 2

    def _calculate_confidence(
        self,
        keystroke_events: Optional[List],
        mouse_events: Optional[List],
        device_info: Optional[Dict],
        baseline: Optional[UserBaseline]
    ) -> float:
        """Calcular confiança na análise"""

        confidence = 0.0

        # Dados disponíveis
        if keystroke_events and len(keystroke_events) > 10:
            confidence += 0.25
        if mouse_events and len(mouse_events) > 20:
            confidence += 0.25
        if device_info:
            confidence += 0.15

        # Baseline disponível
        if baseline:
            confidence += 0.2
            if baseline.samples_count > 10:
                confidence += 0.15

        return min(confidence, 1.0)

    def _generate_explanation(
        self,
        scores: Dict[str, float],
        anomalies: List[str],
        bot_detected: bool,
        duress_detected: bool
    ) -> str:
        """Gerar explicação da análise"""

        parts = []

        if bot_detected:
            parts.append("ALERTA: Comportamento de bot/automação detectado.")

        if duress_detected:
            parts.append("ALERTA: Possível usuário sob coerção detectado.")

        # Scores por componente
        high_risk = [k for k, v in scores.items() if v > 0.7]
        if high_risk:
            parts.append(f"Componentes de alto risco: {', '.join(high_risk)}.")

        # Anomalias específicas
        if anomalies:
            parts.append(f"Anomalias: {'; '.join(anomalies[:5])}.")

        if not parts:
            parts.append("Comportamento dentro dos padrões normais.")

        return " ".join(parts)

    def _update_baseline(
        self,
        user_id: str,
        keystroke_events: Optional[List],
        mouse_events: Optional[List],
        device_info: Optional[Dict],
        session_data: Dict
    ):
        """Atualizar baseline do usuário"""

        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = UserBaseline(
                user_id=user_id,
                keystroke_baseline={},
                mouse_baseline={},
                session_baseline={},
                device_baseline={},
                samples_count=0,
                last_updated=datetime.now(),
                confidence=0.0
            )

        baseline = self.user_baselines[user_id]

        # Atualizar keystroke baseline
        if keystroke_events:
            self._update_keystroke_baseline(baseline, keystroke_events)

        # Atualizar mouse baseline
        if mouse_events:
            self._update_mouse_baseline(baseline, mouse_events)

        # Atualizar session baseline
        self._update_session_baseline(baseline, session_data)

        # Atualizar device baseline
        if device_info:
            self._update_device_baseline(baseline, device_info)

        baseline.samples_count += 1
        baseline.last_updated = datetime.now()
        baseline.confidence = min(baseline.samples_count / 20, 1.0)

    def _update_keystroke_baseline(
        self,
        baseline: UserBaseline,
        events: List[Dict]
    ):
        """Atualizar baseline de keystroke"""

        speed = self._calculate_typing_speed(events)
        error_rate = self._calculate_error_rate(events)
        long_pauses = self._count_long_pauses(events)

        # Média móvel exponencial
        alpha = 0.2

        if "avg_speed" not in baseline.keystroke_baseline:
            baseline.keystroke_baseline["avg_speed"] = speed
            baseline.keystroke_baseline["avg_error_rate"] = error_rate
            baseline.keystroke_baseline["avg_long_pauses"] = long_pauses
        else:
            baseline.keystroke_baseline["avg_speed"] = (
                alpha * speed + (1 - alpha) * baseline.keystroke_baseline["avg_speed"]
            )
            baseline.keystroke_baseline["avg_error_rate"] = (
                alpha * error_rate + (1 - alpha) * baseline.keystroke_baseline["avg_error_rate"]
            )
            baseline.keystroke_baseline["avg_long_pauses"] = (
                alpha * long_pauses + (1 - alpha) * baseline.keystroke_baseline["avg_long_pauses"]
            )

    def _update_mouse_baseline(
        self,
        baseline: UserBaseline,
        events: List[Dict]
    ):
        """Atualizar baseline de mouse"""

        jitter = self._calculate_mouse_jitter(events)
        velocities = self._calculate_mouse_velocities(events)
        avg_velocity = np.mean(velocities) if velocities else 0

        alpha = 0.2

        if "avg_jitter" not in baseline.mouse_baseline:
            baseline.mouse_baseline["avg_jitter"] = jitter
            baseline.mouse_baseline["avg_velocity"] = avg_velocity
        else:
            baseline.mouse_baseline["avg_jitter"] = (
                alpha * jitter + (1 - alpha) * baseline.mouse_baseline["avg_jitter"]
            )
            baseline.mouse_baseline["avg_velocity"] = (
                alpha * avg_velocity + (1 - alpha) * baseline.mouse_baseline["avg_velocity"]
            )

    def _update_session_baseline(
        self,
        baseline: UserBaseline,
        session_data: Dict
    ):
        """Atualizar baseline de sessão"""

        duration = session_data.get("duration_seconds", 0)

        alpha = 0.2

        if "avg_duration" not in baseline.session_baseline:
            baseline.session_baseline["avg_duration"] = duration
            baseline.session_baseline["std_duration"] = 0
        else:
            old_avg = baseline.session_baseline["avg_duration"]
            baseline.session_baseline["avg_duration"] = (
                alpha * duration + (1 - alpha) * old_avg
            )
            # Atualizar desvio padrão aproximado
            baseline.session_baseline["std_duration"] = (
                alpha * abs(duration - old_avg) +
                (1 - alpha) * baseline.session_baseline.get("std_duration", 0)
            )

    def _update_device_baseline(
        self,
        baseline: UserBaseline,
        device_info: Dict
    ):
        """Atualizar baseline de dispositivo"""

        # Armazenar dispositivos conhecidos
        if "known_devices" not in baseline.device_baseline:
            baseline.device_baseline["known_devices"] = []

        device_hash = self._hash_device(device_info)
        if device_hash not in baseline.device_baseline["known_devices"]:
            baseline.device_baseline["known_devices"].append(device_hash)
            # Manter apenas os últimos 10 dispositivos
            baseline.device_baseline["known_devices"] = (
                baseline.device_baseline["known_devices"][-10:]
            )

    # Métodos auxiliares

    def _extract_timings(self, events: List[Dict]) -> List[float]:
        """Extrair intervalos de tempo entre eventos"""
        timings = []
        for i in range(1, len(events)):
            t1 = events[i - 1].get("timestamp", 0)
            t2 = events[i].get("timestamp", 0)
            if t2 > t1:
                timings.append(t2 - t1)
        return timings

    def _calculate_mouse_linearity(self, events: List[Dict]) -> float:
        """Calcular linearidade dos movimentos de mouse"""
        if len(events) < 3:
            return 0.5

        # Calcular curvatura média
        curvatures = []
        for i in range(1, len(events) - 1):
            p1 = (events[i-1].get("x", 0), events[i-1].get("y", 0))
            p2 = (events[i].get("x", 0), events[i].get("y", 0))
            p3 = (events[i+1].get("x", 0), events[i+1].get("y", 0))

            # Ângulo entre vetores
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp
                curvatures.append(abs(cos_angle))

        return np.mean(curvatures) if curvatures else 0.5

    def _calculate_mouse_velocities(self, events: List[Dict]) -> List[float]:
        """Calcular velocidades de mouse"""
        velocities = []
        for i in range(1, len(events)):
            x1, y1 = events[i-1].get("x", 0), events[i-1].get("y", 0)
            x2, y2 = events[i].get("x", 0), events[i].get("y", 0)
            t1 = events[i-1].get("timestamp", 0)
            t2 = events[i].get("timestamp", 0)

            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            dt = t2 - t1

            if dt > 0:
                velocities.append(dist / dt)

        return velocities

    def _calculate_mouse_jitter(self, events: List[Dict]) -> float:
        """Calcular jitter (micro-movimentos) do mouse"""
        if len(events) < 3:
            return 0

        # Calcular variação de direção
        direction_changes = 0
        for i in range(2, len(events)):
            x1 = events[i-2].get("x", 0)
            x2 = events[i-1].get("x", 0)
            x3 = events[i].get("x", 0)
            y1 = events[i-2].get("y", 0)
            y2 = events[i-1].get("y", 0)
            y3 = events[i].get("y", 0)

            # Direção mudou?
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x3 - x2, y3 - y2

            if (dx1 * dx2 < 0) or (dy1 * dy2 < 0):
                direction_changes += 1

        return direction_changes / max(len(events) - 2, 1)

    def _detect_perfect_repetition(self, events: List[Dict]) -> bool:
        """Detectar padrões repetitivos perfeitos"""
        if len(events) < 10:
            return False

        timings = self._extract_timings(events)
        if len(timings) < 8:
            return False

        # Procurar por sequências idênticas
        for pattern_len in range(3, len(timings) // 2):
            pattern = timings[:pattern_len]
            matches = 0

            for i in range(pattern_len, len(timings) - pattern_len, pattern_len):
                subsequence = timings[i:i+pattern_len]
                if len(subsequence) == pattern_len:
                    diff = sum(abs(a - b) for a, b in zip(pattern, subsequence))
                    if diff < 10:  # Muito similar
                        matches += 1

            if matches >= 2:
                return True

        return False

    def _count_long_pauses(
        self,
        events: List[Dict],
        threshold_ms: int = 2000
    ) -> int:
        """Contar pausas longas na digitação"""
        timings = self._extract_timings(events)
        return sum(1 for t in timings if t > threshold_ms)

    def _calculate_typing_speed(self, events: List[Dict]) -> float:
        """Calcular velocidade de digitação (chars/segundo)"""
        if len(events) < 2:
            return 0

        first_ts = events[0].get("timestamp", 0)
        last_ts = events[-1].get("timestamp", 0)
        duration = (last_ts - first_ts) / 1000  # para segundos

        if duration > 0:
            return len(events) / duration
        return 0

    def _calculate_error_rate(self, events: List[Dict]) -> float:
        """Calcular taxa de erros (backspaces/total)"""
        backspaces = sum(1 for e in events if e.get("key") == "Backspace")
        total = len(events)

        if total > 0:
            return backspaces / total
        return 0

    def _hash_device(self, device_info: Dict) -> str:
        """Gerar hash único do dispositivo"""
        key_fields = ["user_agent", "screen_resolution", "timezone", "language"]
        values = [str(device_info.get(f, "")) for f in key_fields]
        return hashlib.md5("|".join(values).encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "users_with_baseline": len(self.user_baselines),
            "component_weights": self.COMPONENT_WEIGHTS,
            "thresholds": {
                "anomaly": self.ANOMALY_THRESHOLD,
                "bot": self.BOT_THRESHOLD,
                "duress": self.DURESS_THRESHOLD
            }
        }


def create_behavioral_analyzer(
    config: Optional[Dict[str, Any]] = None
) -> BehavioralAnalyzer:
    """Factory function para criar BehavioralAnalyzer"""
    return BehavioralAnalyzer(config)
