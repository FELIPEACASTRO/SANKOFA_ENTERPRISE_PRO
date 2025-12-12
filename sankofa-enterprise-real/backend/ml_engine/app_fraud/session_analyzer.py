"""
Sankofa Enterprise Pro - Session Analyzer
Análise de padrões de sessão para detecção de APP Fraud

Baseado em:
- BioCatch session analysis
- Academic research on user behavior patterns
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SessionRisk:
    """Resultado da análise de risco da sessão"""
    session_id: str
    risk_score: float
    risk_level: str
    anomalies: List[str]
    navigation_pattern: str
    time_on_critical_pages: Dict[str, float]
    hesitation_detected: bool
    automation_detected: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class SessionAnalyzer:
    """
    Analisador de sessão para detectar comportamento suspeito

    Analisa:
    - Padrões de navegação
    - Tempo em páginas críticas
    - Sequência de ações
    - Hesitação em campos sensíveis
    - Sinais de automação/bot
    """

    VERSION = "1.0.0"

    # Páginas críticas para análise
    CRITICAL_PAGES = [
        "transfer",
        "pix",
        "confirm",
        "review",
        "beneficiary",
        "amount"
    ]

    # Tempos médios esperados em segundos
    EXPECTED_PAGE_TIMES = {
        "transfer": (10, 60),      # Min, Max
        "pix": (15, 90),
        "confirm": (5, 30),
        "review": (10, 45),
        "beneficiary": (20, 120),
        "amount": (5, 30)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.session_data: Dict[str, Dict] = {}
        self.user_baselines: Dict[str, Dict] = {}

        logger.info(f"SessionAnalyzer v{self.VERSION} initialized")

    def start_session(self, session_id: str, user_id: Optional[str] = None):
        """Inicia tracking de uma nova sessão"""
        self.session_data[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "pages": [],
            "events": [],
            "field_interactions": {},
            "clicks": [],
            "scrolls": [],
            "navigation_path": []
        }

    def track_page_visit(
        self,
        session_id: str,
        page_name: str,
        time_spent: float,
        scroll_depth: float = 0.0
    ):
        """Registra visita a uma página"""
        if session_id not in self.session_data:
            self.start_session(session_id)

        self.session_data[session_id]["pages"].append({
            "name": page_name,
            "time_spent": time_spent,
            "scroll_depth": scroll_depth,
            "timestamp": datetime.now()
        })

        self.session_data[session_id]["navigation_path"].append(page_name)

    def track_field_interaction(
        self,
        session_id: str,
        field_name: str,
        time_to_first_input: float,
        total_time: float,
        corrections: int,
        focus_losses: int
    ):
        """Registra interação com campo de formulário"""
        if session_id not in self.session_data:
            return

        self.session_data[session_id]["field_interactions"][field_name] = {
            "time_to_first_input": time_to_first_input,
            "total_time": total_time,
            "corrections": corrections,
            "focus_losses": focus_losses,
            "timestamp": datetime.now()
        }

    def track_event(
        self,
        session_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Registra evento genérico"""
        if session_id not in self.session_data:
            return

        self.session_data[session_id]["events"].append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now()
        })

    def analyze_session(self, session_id: str) -> SessionRisk:
        """
        Analisa uma sessão para detectar comportamento suspeito

        Returns:
            SessionRisk com resultado da análise
        """
        if session_id not in self.session_data:
            return SessionRisk(
                session_id=session_id,
                risk_score=0.5,
                risk_level="UNKNOWN",
                anomalies=["session_not_found"],
                navigation_pattern="unknown",
                time_on_critical_pages={},
                hesitation_detected=False,
                automation_detected=False,
                confidence=0.0
            )

        session = self.session_data[session_id]
        user_id = session.get("user_id")

        anomalies = []
        risk_components = []

        # 1. Analisar padrão de navegação
        nav_pattern, nav_score = self._analyze_navigation_pattern(session)
        risk_components.append(nav_score)
        if nav_score > 0.5:
            anomalies.append(f"unusual_navigation_pattern:{nav_pattern}")

        # 2. Analisar tempo em páginas críticas
        page_times, time_score = self._analyze_page_times(session)
        risk_components.append(time_score)
        if time_score > 0.5:
            anomalies.append("abnormal_page_times")

        # 3. Analisar interações com campos
        field_score, hesitation = self._analyze_field_interactions(session)
        risk_components.append(field_score)
        if hesitation:
            anomalies.append("field_hesitation_detected")

        # 4. Detectar automação
        automation_score, is_automated = self._detect_automation(session)
        risk_components.append(automation_score)
        if is_automated:
            anomalies.append("possible_automation")

        # 5. Comparar com baseline do usuário
        if user_id and user_id in self.user_baselines:
            baseline_score = self._compare_to_baseline(session, user_id)
            risk_components.append(baseline_score)
            if baseline_score > 0.6:
                anomalies.append("deviation_from_baseline")

        # 6. Analisar velocidade geral
        speed_score = self._analyze_session_speed(session)
        risk_components.append(speed_score)
        if speed_score > 0.7:
            anomalies.append("too_fast_session")
        elif speed_score < 0.2 and len(session["pages"]) > 3:
            anomalies.append("too_slow_session")

        # Calcular risk score final
        risk_score = np.mean(risk_components) if risk_components else 0.5

        # Boost se múltiplas anomalias
        if len(anomalies) >= 3:
            risk_score = min(risk_score * 1.2, 1.0)

        risk_level = self._get_risk_level(risk_score)
        confidence = self._calculate_confidence(session, len(risk_components))

        return SessionRisk(
            session_id=session_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            anomalies=anomalies,
            navigation_pattern=nav_pattern,
            time_on_critical_pages=page_times,
            hesitation_detected=hesitation,
            automation_detected=is_automated,
            confidence=confidence
        )

    def _analyze_navigation_pattern(
        self,
        session: Dict
    ) -> Tuple[str, float]:
        """Analisa padrão de navegação"""
        nav_path = session.get("navigation_path", [])

        if not nav_path:
            return "empty", 0.5

        # Padrões esperados para transações
        expected_patterns = [
            ["transfer", "beneficiary", "amount", "review", "confirm"],
            ["pix", "beneficiary", "amount", "confirm"],
            ["transfer", "amount", "review", "confirm"]
        ]

        # Verificar se segue padrão esperado
        nav_lower = [p.lower() for p in nav_path]

        for expected in expected_patterns:
            if self._is_subsequence(expected, nav_lower):
                return "normal", 0.2

        # Detectar comportamentos suspeitos

        # Muitos retornos
        back_count = sum(1 for i in range(1, len(nav_path))
                       if nav_path[i] == nav_path[i-1])
        if back_count > 3:
            return "excessive_backtracking", 0.7

        # Navegação muito direta (possível automação)
        critical_in_path = sum(1 for p in nav_lower if any(
            c in p for c in self.CRITICAL_PAGES
        ))
        if len(nav_path) == critical_in_path and len(nav_path) <= 4:
            return "suspiciously_direct", 0.6

        # Navegação errática
        unique_pages = len(set(nav_path))
        if len(nav_path) > unique_pages * 2:
            return "erratic", 0.65

        return "atypical", 0.4

    def _analyze_page_times(
        self,
        session: Dict
    ) -> Tuple[Dict[str, float], float]:
        """Analisa tempo gasto em páginas críticas"""
        pages = session.get("pages", [])
        page_times = {}
        anomaly_scores = []

        for page in pages:
            page_name = page["name"].lower()
            time_spent = page["time_spent"]

            # Verificar se é página crítica
            for critical in self.CRITICAL_PAGES:
                if critical in page_name:
                    page_times[critical] = time_spent

                    # Comparar com tempo esperado
                    if critical in self.EXPECTED_PAGE_TIMES:
                        min_time, max_time = self.EXPECTED_PAGE_TIMES[critical]

                        if time_spent < min_time * 0.5:
                            # Muito rápido
                            anomaly_scores.append(0.8)
                        elif time_spent > max_time * 2:
                            # Muito lento (possível hesitação ou distração)
                            anomaly_scores.append(0.5)
                        else:
                            anomaly_scores.append(0.2)
                    break

        avg_score = np.mean(anomaly_scores) if anomaly_scores else 0.3
        return page_times, avg_score

    def _analyze_field_interactions(
        self,
        session: Dict
    ) -> Tuple[float, bool]:
        """Analisa interações com campos de formulário"""
        interactions = session.get("field_interactions", {})

        if not interactions:
            return 0.3, False

        hesitation_detected = False
        risk_scores = []

        # Campos sensíveis que indicam hesitação
        sensitive_fields = ["amount", "valor", "receiver", "destinatario", "pix_key"]

        for field_name, data in interactions.items():
            field_lower = field_name.lower()

            # Verificar hesitação em campos sensíveis
            if any(sf in field_lower for sf in sensitive_fields):
                # Tempo para primeira entrada > 5s indica hesitação
                if data["time_to_first_input"] > 5:
                    hesitation_detected = True
                    risk_scores.append(0.7)

                # Muitas correções indicam incerteza
                if data["corrections"] > 2:
                    risk_scores.append(0.5)

                # Perda de foco indica distração ou coerção
                if data["focus_losses"] > 3:
                    hesitation_detected = True
                    risk_scores.append(0.6)

            # Tempo total muito curto (automação)
            if data["total_time"] < 0.5:
                risk_scores.append(0.8)

        avg_risk = np.mean(risk_scores) if risk_scores else 0.2
        return avg_risk, hesitation_detected

    def _detect_automation(self, session: Dict) -> Tuple[float, bool]:
        """Detecta sinais de automação/bot"""
        indicators = []

        # 1. Eventos muito regulares (timing perfeito)
        events = session.get("events", [])
        if len(events) >= 5:
            intervals = []
            for i in range(1, len(events)):
                delta = (events[i]["timestamp"] - events[i-1]["timestamp"]).total_seconds()
                intervals.append(delta)

            if intervals:
                std_intervals = np.std(intervals)
                # Intervalos muito regulares = suspeito
                if std_intervals < 0.1 and np.mean(intervals) < 0.5:
                    indicators.append(0.9)

        # 2. Velocidade sobre-humana
        pages = session.get("pages", [])
        for page in pages:
            if page["time_spent"] < 0.3:  # < 300ms por página
                indicators.append(0.85)
                break

        # 3. Nenhum scroll em páginas longas
        for page in pages:
            if page.get("scroll_depth", 0) == 0 and page["time_spent"] > 5:
                # Passou tempo na página mas não scrollou
                indicators.append(0.5)

        # 4. Falta de micro-movimentos
        field_interactions = session.get("field_interactions", {})
        if field_interactions:
            all_fast = all(
                data["time_to_first_input"] < 0.2
                for data in field_interactions.values()
            )
            if all_fast:
                indicators.append(0.7)

        if indicators:
            avg_score = np.mean(indicators)
            is_automated = avg_score > 0.6
            return avg_score, is_automated

        return 0.1, False

    def _compare_to_baseline(
        self,
        session: Dict,
        user_id: str
    ) -> float:
        """Compara sessão com baseline do usuário"""
        baseline = self.user_baselines.get(user_id, {})

        if not baseline:
            return 0.3

        deviations = []

        # Comparar tempo médio em páginas
        pages = session.get("pages", [])
        for page in pages:
            page_name = page["name"].lower()
            if page_name in baseline.get("avg_page_times", {}):
                expected = baseline["avg_page_times"][page_name]
                actual = page["time_spent"]
                deviation = abs(actual - expected) / (expected + 1)
                deviations.append(min(deviation, 1.0))

        # Comparar padrão de navegação
        nav_path = session.get("navigation_path", [])
        usual_paths = baseline.get("usual_paths", [])
        if nav_path and usual_paths:
            path_similarity = max(
                self._path_similarity(nav_path, usual)
                for usual in usual_paths
            )
            deviations.append(1 - path_similarity)

        return np.mean(deviations) if deviations else 0.3

    def _analyze_session_speed(self, session: Dict) -> float:
        """Analisa velocidade geral da sessão"""
        pages = session.get("pages", [])

        if not pages:
            return 0.5

        total_time = sum(p["time_spent"] for p in pages)
        num_pages = len(pages)

        if num_pages == 0:
            return 0.5

        avg_time_per_page = total_time / num_pages

        # Normalizar para score de risco
        if avg_time_per_page < 2:
            return 0.9  # Muito rápido
        elif avg_time_per_page < 5:
            return 0.6  # Rápido
        elif avg_time_per_page < 30:
            return 0.2  # Normal
        elif avg_time_per_page < 120:
            return 0.4  # Lento (possível hesitação)
        else:
            return 0.6  # Muito lento

    def _get_risk_level(self, risk_score: float) -> str:
        """Determina nível de risco"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        return "VERY_LOW"

    def _calculate_confidence(
        self,
        session: Dict,
        num_components: int
    ) -> float:
        """Calcula confiança na análise"""
        confidence = 0.5

        # Mais dados = mais confiança
        num_pages = len(session.get("pages", []))
        num_interactions = len(session.get("field_interactions", {}))
        num_events = len(session.get("events", []))

        confidence += min(num_pages * 0.05, 0.15)
        confidence += min(num_interactions * 0.05, 0.15)
        confidence += min(num_events * 0.01, 0.1)
        confidence += min(num_components * 0.02, 0.1)

        return min(confidence, 0.95)

    def _is_subsequence(self, sub: List, main: List) -> bool:
        """Verifica se sub é subsequência de main"""
        it = iter(main)
        return all(item in it for item in sub)

    def _path_similarity(self, path1: List, path2: List) -> float:
        """Calcula similaridade entre dois caminhos de navegação"""
        if not path1 or not path2:
            return 0.0

        # Longest Common Subsequence
        m, n = len(path1), len(path2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if path1[i-1].lower() == path2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        return (2 * lcs_length) / (m + n)

    def update_user_baseline(self, user_id: str, session: Dict):
        """Atualiza baseline do usuário com sessão legítima"""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "avg_page_times": {},
                "usual_paths": [],
                "session_count": 0
            }

        baseline = self.user_baselines[user_id]
        baseline["session_count"] += 1

        # Atualizar tempos médios
        for page in session.get("pages", []):
            page_name = page["name"].lower()
            time_spent = page["time_spent"]

            if page_name in baseline["avg_page_times"]:
                # Média móvel exponencial
                alpha = 0.3
                baseline["avg_page_times"][page_name] = (
                    alpha * time_spent +
                    (1 - alpha) * baseline["avg_page_times"][page_name]
                )
            else:
                baseline["avg_page_times"][page_name] = time_spent

        # Adicionar caminho usual
        nav_path = session.get("navigation_path", [])
        if nav_path:
            if len(baseline["usual_paths"]) < 5:
                baseline["usual_paths"].append(nav_path)
            elif baseline["session_count"] % 10 == 0:
                # Substituir caminho mais antigo periodicamente
                baseline["usual_paths"].pop(0)
                baseline["usual_paths"].append(nav_path)

    def end_session(self, session_id: str) -> Optional[SessionRisk]:
        """Finaliza e analisa sessão"""
        if session_id not in self.session_data:
            return None

        # Analisar sessão
        risk = self.analyze_session(session_id)

        # Limpar dados da sessão
        del self.session_data[session_id]

        return risk


def create_session_analyzer(config: Optional[Dict[str, Any]] = None) -> SessionAnalyzer:
    """Factory function para criar SessionAnalyzer"""
    return SessionAnalyzer(config)
