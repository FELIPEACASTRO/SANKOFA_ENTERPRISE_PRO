"""
Sankofa Enterprise Pro - Duress Detector
Detecção de sinais de coerção/pressão sobre o usuário

Baseado em:
- BioCatch duress detection research
- Academic papers on behavioral biometrics under stress
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DuressAnalysis:
    """Resultado da análise de duress/coerção"""
    session_id: str
    duress_probability: float
    duress_detected: bool
    indicators: List[str]
    behavioral_deviations: Dict[str, float]
    stress_markers: Dict[str, float]
    recommended_action: str
    explanation: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class DuressDetector:
    """
    Detector de Duress (Coerção)

    Detecta sinais de que o usuário está sendo forçado ou pressionado
    a realizar uma transação contra sua vontade.

    Indicadores de duress:
    - Hesitação anormal ao digitar
    - Movimentos de mouse erráticos
    - Tempo excessivo em campos de confirmação
    - Padrão de digitação alterado
    - Pausas longas antes de confirmar
    - Correções frequentes
    """

    VERSION = "1.0.0"

    # Thresholds para detecção
    THRESHOLDS = {
        "hesitation_score": 0.6,
        "typing_deviation": 2.0,  # Desvios padrão
        "mouse_jitter": 0.5,
        "confirmation_delay": 10.0,  # segundos
        "correction_rate": 0.3,
        "focus_loss_rate": 0.4
    }

    # Pesos dos indicadores
    INDICATOR_WEIGHTS = {
        "typing_hesitation": 0.20,
        "mouse_jitter": 0.15,
        "confirmation_delay": 0.20,
        "correction_pattern": 0.15,
        "focus_pattern": 0.10,
        "timing_anomaly": 0.10,
        "behavioral_deviation": 0.10
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("detection_threshold", 0.7)
        self.user_baselines: Dict[str, Dict] = {}

        logger.info(f"DuressDetector v{self.VERSION} initialized")

    def analyze(
        self,
        session_id: str,
        keystroke_data: Optional[Dict[str, Any]] = None,
        mouse_data: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> DuressAnalysis:
        """
        Analisa dados comportamentais para detectar duress

        Args:
            session_id: ID da sessão
            keystroke_data: Dados de dinâmica de digitação
            mouse_data: Dados de movimento de mouse
            session_data: Dados gerais da sessão
            user_id: ID do usuário (para comparação com baseline)

        Returns:
            DuressAnalysis com resultado da análise
        """
        indicators = []
        stress_markers = {}
        behavioral_deviations = {}

        # 1. Analisar padrão de digitação
        if keystroke_data:
            typing_result = self._analyze_typing_pattern(keystroke_data, user_id)
            stress_markers.update(typing_result["markers"])
            indicators.extend(typing_result["indicators"])
            behavioral_deviations["typing"] = typing_result["deviation"]

        # 2. Analisar padrão de mouse
        if mouse_data:
            mouse_result = self._analyze_mouse_pattern(mouse_data, user_id)
            stress_markers.update(mouse_result["markers"])
            indicators.extend(mouse_result["indicators"])
            behavioral_deviations["mouse"] = mouse_result["deviation"]

        # 3. Analisar padrão de sessão
        if session_data:
            session_result = self._analyze_session_pattern(session_data)
            stress_markers.update(session_result["markers"])
            indicators.extend(session_result["indicators"])

        # 4. Calcular probabilidade de duress
        duress_probability = self._calculate_duress_probability(
            stress_markers, indicators, behavioral_deviations
        )

        # 5. Determinar se duress foi detectado
        duress_detected = duress_probability >= self.threshold

        # 6. Determinar ação recomendada
        recommended_action = self._get_recommended_action(
            duress_probability, duress_detected
        )

        # 7. Gerar explicação
        explanation = self._generate_explanation(
            duress_probability, duress_detected, indicators, stress_markers
        )

        # 8. Calcular confiança
        confidence = self._calculate_confidence(
            keystroke_data, mouse_data, session_data, user_id
        )

        return DuressAnalysis(
            session_id=session_id,
            duress_probability=round(duress_probability, 4),
            duress_detected=duress_detected,
            indicators=indicators,
            behavioral_deviations=behavioral_deviations,
            stress_markers=stress_markers,
            recommended_action=recommended_action,
            explanation=explanation,
            confidence=confidence
        )

    def _analyze_typing_pattern(
        self,
        keystroke_data: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analisa padrão de digitação para sinais de stress"""
        result = {
            "markers": {},
            "indicators": [],
            "deviation": 0.0
        }

        # Extrair métricas de keystroke
        avg_hold_time = keystroke_data.get("avg_hold_time", 0)
        std_hold_time = keystroke_data.get("std_hold_time", 0)
        avg_flight_time = keystroke_data.get("avg_flight_time", 0)
        typing_speed = keystroke_data.get("typing_speed", 0)
        error_rate = keystroke_data.get("error_rate", 0)
        pause_count = keystroke_data.get("pause_count", 0)
        backspace_rate = keystroke_data.get("backspace_rate", 0)

        # Analisar hesitação
        hesitation_score = 0.0

        # Pausas longas indicam hesitação
        if pause_count > 3:
            hesitation_score += 0.3
            result["indicators"].append("frequent_pauses")

        # Alta variação no tempo de digitação
        if std_hold_time > avg_hold_time * 0.5:
            hesitation_score += 0.2
            result["indicators"].append("inconsistent_typing")

        # Taxa alta de erros/backspace
        if backspace_rate > 0.2:
            hesitation_score += 0.2
            result["indicators"].append("high_correction_rate")

        result["markers"]["typing_hesitation"] = hesitation_score

        # Comparar com baseline do usuário se disponível
        if user_id and user_id in self.user_baselines:
            baseline = self.user_baselines[user_id].get("typing", {})

            if baseline.get("avg_speed"):
                speed_deviation = abs(typing_speed - baseline["avg_speed"])
                speed_deviation /= (baseline.get("std_speed", 1) + 0.1)
                result["deviation"] = speed_deviation

                if speed_deviation > 2:
                    result["indicators"].append("speed_deviation")
                    result["markers"]["speed_deviation"] = min(speed_deviation / 3, 1.0)

        return result

    def _analyze_mouse_pattern(
        self,
        mouse_data: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analisa padrão de mouse para sinais de stress"""
        result = {
            "markers": {},
            "indicators": [],
            "deviation": 0.0
        }

        # Extrair métricas de mouse
        avg_velocity = mouse_data.get("avg_velocity", 0)
        jitter_score = mouse_data.get("jitter_score", 0)
        straightness_ratio = mouse_data.get("straightness_ratio", 1.0)
        click_accuracy = mouse_data.get("click_accuracy", 1.0)
        hover_time_avg = mouse_data.get("hover_time_avg", 0)

        # Analisar jitter (tremor)
        result["markers"]["mouse_jitter"] = jitter_score
        if jitter_score > self.THRESHOLDS["mouse_jitter"]:
            result["indicators"].append("mouse_tremor")

        # Movimentos não lineares (nervosismo)
        if straightness_ratio < 0.7:
            result["indicators"].append("erratic_movement")
            result["markers"]["erratic_movement"] = 1 - straightness_ratio

        # Clicks imprecisos
        if click_accuracy < 0.8:
            result["indicators"].append("imprecise_clicks")
            result["markers"]["click_imprecision"] = 1 - click_accuracy

        # Hover time excessivo (hesitação)
        if hover_time_avg > 3:
            result["indicators"].append("excessive_hovering")
            result["markers"]["hover_hesitation"] = min(hover_time_avg / 10, 1.0)

        # Comparar com baseline
        if user_id and user_id in self.user_baselines:
            baseline = self.user_baselines[user_id].get("mouse", {})

            if baseline.get("avg_velocity"):
                velocity_deviation = abs(avg_velocity - baseline["avg_velocity"])
                velocity_deviation /= (baseline.get("std_velocity", 1) + 0.1)
                result["deviation"] = velocity_deviation

        return result

    def _analyze_session_pattern(
        self,
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analisa padrões de sessão para sinais de stress"""
        result = {
            "markers": {},
            "indicators": []
        }

        # Tempo no campo de confirmação
        confirm_time = session_data.get("confirm_field_time", 0)
        if confirm_time > self.THRESHOLDS["confirmation_delay"]:
            result["indicators"].append("confirmation_hesitation")
            result["markers"]["confirmation_delay"] = min(confirm_time / 30, 1.0)

        # Tempo total da transação
        total_time = session_data.get("total_transaction_time", 0)
        page_count = session_data.get("page_count", 1)
        avg_time_per_step = total_time / max(page_count, 1)

        if avg_time_per_step > 60:
            result["indicators"].append("slow_progression")
            result["markers"]["slow_progression"] = min(avg_time_per_step / 120, 1.0)

        # Navegação para trás
        back_navigations = session_data.get("back_navigations", 0)
        if back_navigations > 2:
            result["indicators"].append("navigation_uncertainty")
            result["markers"]["back_navigation"] = min(back_navigations / 5, 1.0)

        # Perdas de foco
        focus_losses = session_data.get("focus_losses", 0)
        if focus_losses > 3:
            result["indicators"].append("attention_breaks")
            result["markers"]["focus_loss"] = min(focus_losses / 10, 1.0)

        # Scroll errático
        scroll_reversals = session_data.get("scroll_reversals", 0)
        if scroll_reversals > 5:
            result["indicators"].append("erratic_scrolling")
            result["markers"]["scroll_pattern"] = min(scroll_reversals / 15, 1.0)

        return result

    def _calculate_duress_probability(
        self,
        stress_markers: Dict[str, float],
        indicators: List[str],
        behavioral_deviations: Dict[str, float]
    ) -> float:
        """Calcula probabilidade de duress"""

        # Score base dos stress markers
        marker_score = 0.0
        for marker, value in stress_markers.items():
            weight = self.INDICATOR_WEIGHTS.get(marker, 0.05)
            marker_score += value * weight

        # Boost por quantidade de indicadores
        indicator_boost = len(indicators) * 0.03

        # Boost por desvio comportamental
        deviation_score = 0.0
        if behavioral_deviations:
            avg_deviation = np.mean(list(behavioral_deviations.values()))
            deviation_score = min(avg_deviation / 3, 0.3)

        # Combinar scores
        probability = marker_score + indicator_boost + deviation_score

        # Aplicar boost se múltiplos indicadores fortes
        strong_markers = sum(1 for v in stress_markers.values() if v > 0.5)
        if strong_markers >= 3:
            probability *= 1.2

        return min(probability, 1.0)

    def _get_recommended_action(
        self,
        probability: float,
        duress_detected: bool
    ) -> str:
        """Determina ação recomendada"""
        if duress_detected:
            if probability >= 0.9:
                return "BLOCK_AND_CALL"
            elif probability >= 0.8:
                return "CALL_VERIFICATION"
            else:
                return "ENHANCED_VERIFICATION"
        else:
            if probability >= 0.5:
                return "SOFT_WARNING"
            return "ALLOW"

    def _generate_explanation(
        self,
        probability: float,
        duress_detected: bool,
        indicators: List[str],
        stress_markers: Dict[str, float]
    ) -> str:
        """Gera explicação da análise"""
        if not duress_detected and probability < 0.3:
            return "Nenhum sinal significativo de coerção detectado."

        explanation_parts = []

        if duress_detected:
            explanation_parts.append(
                f"ALERTA: Sinais de possível coerção detectados "
                f"(probabilidade: {probability*100:.0f}%)."
            )
        else:
            explanation_parts.append(
                f"Alguns sinais de stress detectados "
                f"(probabilidade de coerção: {probability*100:.0f}%)."
            )

        # Descrever principais indicadores
        indicator_descriptions = {
            "frequent_pauses": "Pausas frequentes durante digitação",
            "inconsistent_typing": "Padrão de digitação inconsistente",
            "high_correction_rate": "Alta taxa de correções",
            "speed_deviation": "Velocidade de digitação anormal",
            "mouse_tremor": "Tremor detectado no mouse",
            "erratic_movement": "Movimentos de mouse erráticos",
            "confirmation_hesitation": "Hesitação ao confirmar transação",
            "slow_progression": "Progressão muito lenta entre etapas",
            "navigation_uncertainty": "Navegação incerta (muitos retornos)",
            "attention_breaks": "Múltiplas perdas de foco"
        }

        relevant_indicators = [
            indicator_descriptions.get(ind, ind)
            for ind in indicators[:4]
            if ind in indicator_descriptions
        ]

        if relevant_indicators:
            explanation_parts.append(
                "Sinais detectados: " + "; ".join(relevant_indicators) + "."
            )

        return " ".join(explanation_parts)

    def _calculate_confidence(
        self,
        keystroke_data: Optional[Dict],
        mouse_data: Optional[Dict],
        session_data: Optional[Dict],
        user_id: Optional[str]
    ) -> float:
        """Calcula confiança na análise"""
        confidence = 0.3  # Base

        # Mais tipos de dados = mais confiança
        if keystroke_data:
            confidence += 0.2
        if mouse_data:
            confidence += 0.2
        if session_data:
            confidence += 0.15

        # Baseline do usuário aumenta confiança
        if user_id and user_id in self.user_baselines:
            confidence += 0.15

        return min(confidence, 0.95)

    def update_user_baseline(
        self,
        user_id: str,
        keystroke_data: Optional[Dict[str, Any]] = None,
        mouse_data: Optional[Dict[str, Any]] = None
    ):
        """Atualiza baseline do usuário com dados de sessão normal"""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "typing": {},
                "mouse": {},
                "session_count": 0
            }

        baseline = self.user_baselines[user_id]
        baseline["session_count"] += 1
        alpha = 0.3  # Fator de atualização

        # Atualizar baseline de digitação
        if keystroke_data:
            if "avg_speed" not in baseline["typing"]:
                baseline["typing"]["avg_speed"] = keystroke_data.get("typing_speed", 0)
                baseline["typing"]["speeds"] = [keystroke_data.get("typing_speed", 0)]
            else:
                baseline["typing"]["speeds"].append(keystroke_data.get("typing_speed", 0))
                baseline["typing"]["speeds"] = baseline["typing"]["speeds"][-20:]
                baseline["typing"]["avg_speed"] = np.mean(baseline["typing"]["speeds"])
                baseline["typing"]["std_speed"] = np.std(baseline["typing"]["speeds"])

        # Atualizar baseline de mouse
        if mouse_data:
            if "avg_velocity" not in baseline["mouse"]:
                baseline["mouse"]["avg_velocity"] = mouse_data.get("avg_velocity", 0)
                baseline["mouse"]["velocities"] = [mouse_data.get("avg_velocity", 0)]
            else:
                baseline["mouse"]["velocities"].append(mouse_data.get("avg_velocity", 0))
                baseline["mouse"]["velocities"] = baseline["mouse"]["velocities"][-20:]
                baseline["mouse"]["avg_velocity"] = np.mean(baseline["mouse"]["velocities"])
                baseline["mouse"]["std_velocity"] = np.std(baseline["mouse"]["velocities"])


def create_duress_detector(config: Optional[Dict[str, Any]] = None) -> DuressDetector:
    """Factory function para criar DuressDetector"""
    return DuressDetector(config)
