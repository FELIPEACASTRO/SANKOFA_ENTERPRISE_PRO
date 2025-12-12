"""
Sankofa Enterprise Pro - Device Analyzer
Análise de fingerprint e reputação de dispositivo

Baseado em:
- FingerprintJS techniques
- Device intelligence research
- Browser/device fingerprinting
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceFingerprint:
    """Fingerprint completo do dispositivo"""
    fingerprint_hash: str
    risk_score: float
    is_known_device: bool
    device_age_days: int
    is_emulator: bool
    is_headless: bool
    is_proxy_detected: bool
    is_vpn_detected: bool
    is_tor_detected: bool
    timezone_mismatch: bool
    language_mismatch: bool
    webgl_anomaly: bool
    canvas_anomaly: bool
    audio_anomaly: bool
    screen_anomaly: bool
    anomalies: List[str]
    confidence: float
    device_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeviceHistory:
    """Histórico de uso do dispositivo"""
    fingerprint_hash: str
    first_seen: datetime
    last_seen: datetime
    usage_count: int
    users_count: int  # Quantos usuários usaram este dispositivo
    fraud_count: int  # Quantas fraudes associadas
    fraud_rate: float


class DeviceAnalyzer:
    """
    Analisador de Dispositivo

    Detecta:
    - Dispositivos emulados
    - Navegadores headless
    - VPNs e proxies
    - Fingerprint spoofing
    - Dispositivos comprometidos
    - Anomalias de configuração
    """

    VERSION = "1.0.0"

    # Thresholds
    HIGH_FRAUD_RATE_THRESHOLD = 0.1  # 10%
    SUSPICIOUS_USERS_THRESHOLD = 5  # Muitos usuários no mesmo dispositivo

    # User agents conhecidos de bots
    BOT_USER_AGENTS = [
        "headless", "phantom", "selenium", "puppeteer", "playwright",
        "webdriver", "bot", "crawler", "spider", "scraper"
    ]

    # Resoluções de tela suspeitas (VMs, emuladores)
    SUSPICIOUS_RESOLUTIONS = [
        (800, 600),  # VM padrão
        (1024, 768),  # VM comum
        (1, 1),  # Headless
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Cache de dispositivos conhecidos
        self.device_history: Dict[str, DeviceHistory] = {}

        logger.info(f"DeviceAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        device_info: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> DeviceFingerprint:
        """
        Analisar informações do dispositivo

        Args:
            device_info: Informações coletadas do dispositivo
            baseline: Dispositivos conhecidos do usuário

        Returns:
            DeviceFingerprint com análise completa
        """
        anomalies = []
        risk_score = 0.0

        # 1. Gerar fingerprint hash
        fingerprint_hash = self._generate_fingerprint_hash(device_info)

        # 2. Verificar se é dispositivo conhecido
        is_known = self._is_known_device(fingerprint_hash, baseline)

        # 3. Obter histórico do dispositivo
        history = self.device_history.get(fingerprint_hash)
        device_age = self._calculate_device_age(history)

        # 4. Detectar emulador/VM
        is_emulator, emulator_details = self._detect_emulator(device_info)
        if is_emulator:
            anomalies.append(f"Emulador detectado: {emulator_details}")
            risk_score += 0.3

        # 5. Detectar navegador headless
        is_headless, headless_details = self._detect_headless(device_info)
        if is_headless:
            anomalies.append(f"Navegador headless: {headless_details}")
            risk_score += 0.35

        # 6. Detectar proxy/VPN/Tor
        is_proxy = self._detect_proxy(device_info)
        is_vpn = self._detect_vpn(device_info)
        is_tor = self._detect_tor(device_info)

        if is_proxy:
            anomalies.append("Proxy detectado")
            risk_score += 0.15
        if is_vpn:
            anomalies.append("VPN detectada")
            risk_score += 0.1
        if is_tor:
            anomalies.append("Rede Tor detectada")
            risk_score += 0.25

        # 7. Verificar inconsistências de timezone/language
        timezone_mismatch = self._check_timezone_mismatch(device_info)
        language_mismatch = self._check_language_mismatch(device_info)

        if timezone_mismatch:
            anomalies.append("Timezone inconsistente com IP")
            risk_score += 0.15
        if language_mismatch:
            anomalies.append("Idioma inconsistente com localização")
            risk_score += 0.1

        # 8. Verificar anomalias de fingerprinting
        webgl_anomaly = self._check_webgl_anomaly(device_info)
        canvas_anomaly = self._check_canvas_anomaly(device_info)
        audio_anomaly = self._check_audio_anomaly(device_info)
        screen_anomaly = self._check_screen_anomaly(device_info)

        if webgl_anomaly:
            anomalies.append("WebGL fingerprint anômalo")
            risk_score += 0.1
        if canvas_anomaly:
            anomalies.append("Canvas fingerprint anômalo")
            risk_score += 0.1
        if audio_anomaly:
            anomalies.append("Audio fingerprint anômalo")
            risk_score += 0.1
        if screen_anomaly:
            anomalies.append("Resolução de tela suspeita")
            risk_score += 0.1

        # 9. Verificar histórico do dispositivo
        if history:
            if history.fraud_rate > self.HIGH_FRAUD_RATE_THRESHOLD:
                anomalies.append(f"Dispositivo com histórico de fraude ({history.fraud_rate:.1%})")
                risk_score += 0.3
            if history.users_count > self.SUSPICIOUS_USERS_THRESHOLD:
                anomalies.append(f"Dispositivo usado por {history.users_count} usuários")
                risk_score += 0.2

        # 10. Verificar se é dispositivo novo para usuário conhecido
        if baseline and not is_known:
            anomalies.append("Novo dispositivo para este usuário")
            risk_score += 0.15

        # 11. Calcular confiança
        confidence = self._calculate_confidence(device_info, baseline)

        return DeviceFingerprint(
            fingerprint_hash=fingerprint_hash,
            risk_score=min(risk_score, 1.0),
            is_known_device=is_known,
            device_age_days=device_age,
            is_emulator=is_emulator,
            is_headless=is_headless,
            is_proxy_detected=is_proxy,
            is_vpn_detected=is_vpn,
            is_tor_detected=is_tor,
            timezone_mismatch=timezone_mismatch,
            language_mismatch=language_mismatch,
            webgl_anomaly=webgl_anomaly,
            canvas_anomaly=canvas_anomaly,
            audio_anomaly=audio_anomaly,
            screen_anomaly=screen_anomaly,
            anomalies=anomalies,
            confidence=confidence,
            device_info=device_info
        )

    def _generate_fingerprint_hash(self, device_info: Dict[str, Any]) -> str:
        """Gerar hash único do dispositivo"""
        # Campos estáveis para fingerprint
        stable_fields = [
            "user_agent",
            "screen_resolution",
            "color_depth",
            "timezone",
            "language",
            "platform",
            "hardware_concurrency",
            "device_memory",
            "webgl_vendor",
            "webgl_renderer",
            "canvas_hash",
            "audio_hash",
            "fonts"
        ]

        values = []
        for field in stable_fields:
            value = device_info.get(field, "")
            if isinstance(value, (list, dict)):
                value = str(value)
            values.append(str(value))

        fingerprint_string = "|".join(values)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:32]

    def _is_known_device(
        self,
        fingerprint_hash: str,
        baseline: Optional[Dict[str, Any]]
    ) -> bool:
        """Verificar se dispositivo é conhecido"""
        if not baseline:
            return False

        known_devices = baseline.get("known_devices", [])
        return fingerprint_hash in known_devices

    def _calculate_device_age(self, history: Optional[DeviceHistory]) -> int:
        """Calcular idade do dispositivo em dias"""
        if not history:
            return 0

        age = (datetime.now() - history.first_seen).days
        return age

    def _detect_emulator(
        self,
        device_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Detectar se é emulador/VM"""
        indicators = []

        # Verificar user agent
        user_agent = device_info.get("user_agent", "").lower()
        if any(vm in user_agent for vm in ["android sdk", "genymotion", "bluestacks"]):
            indicators.append("user_agent")

        # Verificar WebGL renderer
        webgl_renderer = device_info.get("webgl_renderer", "").lower()
        vm_renderers = ["swiftshader", "llvmpipe", "virtualbox", "vmware", "parallels"]
        if any(vm in webgl_renderer for vm in vm_renderers):
            indicators.append("webgl_renderer")

        # Verificar hardware
        hardware_concurrency = device_info.get("hardware_concurrency", 0)
        device_memory = device_info.get("device_memory", 0)

        if hardware_concurrency == 1 and device_memory <= 2:
            indicators.append("low_resources")

        # Verificar plugins
        plugins = device_info.get("plugins", [])
        if isinstance(plugins, list) and len(plugins) == 0:
            indicators.append("no_plugins")

        # Verificar touch points em desktop
        platform = device_info.get("platform", "").lower()
        max_touch_points = device_info.get("max_touch_points", 0)

        if "win" in platform and max_touch_points > 5:
            indicators.append("suspicious_touch")

        is_emulator = len(indicators) >= 2
        return is_emulator, ", ".join(indicators) if indicators else ""

    def _detect_headless(
        self,
        device_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Detectar navegador headless"""
        indicators = []

        user_agent = device_info.get("user_agent", "").lower()

        # Verificar user agent
        for bot_ua in self.BOT_USER_AGENTS:
            if bot_ua in user_agent:
                indicators.append(f"ua:{bot_ua}")

        # Verificar webdriver
        if device_info.get("webdriver", False):
            indicators.append("webdriver")

        # Verificar plugins
        plugins = device_info.get("plugins", [])
        if isinstance(plugins, list) and len(plugins) == 0:
            indicators.append("no_plugins")

        # Verificar languages
        languages = device_info.get("languages", [])
        if isinstance(languages, list) and len(languages) == 0:
            indicators.append("no_languages")

        # Verificar dimensões de viewport
        viewport_width = device_info.get("viewport_width", 0)
        viewport_height = device_info.get("viewport_height", 0)

        if viewport_width == 0 or viewport_height == 0:
            indicators.append("zero_viewport")

        # Verificar outerWidth/outerHeight
        outer_width = device_info.get("outer_width", 0)
        inner_width = device_info.get("inner_width", 0)

        if outer_width > 0 and inner_width > 0:
            if outer_width == inner_width:  # Headless não tem chrome
                indicators.append("no_chrome")

        is_headless = len(indicators) >= 2
        return is_headless, ", ".join(indicators) if indicators else ""

    def _detect_proxy(self, device_info: Dict[str, Any]) -> bool:
        """Detectar uso de proxy"""
        # Verificar headers suspeitos
        headers = device_info.get("headers", {})

        proxy_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "via",
            "x-proxy-id",
            "forwarded"
        ]

        for header in proxy_headers:
            if header.lower() in [h.lower() for h in headers.keys()]:
                return True

        # Verificar inconsistência de timezone com IP
        # (implementação simplificada)
        return False

    def _detect_vpn(self, device_info: Dict[str, Any]) -> bool:
        """Detectar uso de VPN"""
        # Verificar ASN do IP
        ip_info = device_info.get("ip_info", {})
        asn_org = ip_info.get("org", "").lower()

        vpn_providers = [
            "nordvpn", "expressvpn", "surfshark", "cyberghost",
            "private internet access", "protonvpn", "mullvad"
        ]

        for vpn in vpn_providers:
            if vpn in asn_org:
                return True

        # Verificar tipo de conexão
        connection_type = ip_info.get("type", "").lower()
        if connection_type in ["hosting", "proxy", "vpn"]:
            return True

        return False

    def _detect_tor(self, device_info: Dict[str, Any]) -> bool:
        """Detectar uso de Tor"""
        ip_info = device_info.get("ip_info", {})

        # Verificar se é exit node do Tor
        is_tor_exit = ip_info.get("is_tor_exit", False)
        if is_tor_exit:
            return True

        # Verificar ASN conhecido do Tor
        asn_org = ip_info.get("org", "").lower()
        if "tor" in asn_org:
            return True

        return False

    def _check_timezone_mismatch(self, device_info: Dict[str, Any]) -> bool:
        """Verificar inconsistência de timezone"""
        browser_timezone = device_info.get("timezone", "")
        ip_timezone = device_info.get("ip_info", {}).get("timezone", "")

        if browser_timezone and ip_timezone:
            # Simplificação: verificar se são diferentes
            return browser_timezone != ip_timezone

        return False

    def _check_language_mismatch(self, device_info: Dict[str, Any]) -> bool:
        """Verificar inconsistência de idioma"""
        browser_language = device_info.get("language", "").split("-")[0].lower()
        ip_country = device_info.get("ip_info", {}).get("country", "").lower()

        # Mapeamento simplificado
        country_languages = {
            "br": ["pt"],
            "us": ["en"],
            "uk": ["en"],
            "de": ["de"],
            "fr": ["fr"],
            "es": ["es"],
            "jp": ["ja"],
            "cn": ["zh"]
        }

        expected_languages = country_languages.get(ip_country, [])

        if expected_languages and browser_language:
            return browser_language not in expected_languages

        return False

    def _check_webgl_anomaly(self, device_info: Dict[str, Any]) -> bool:
        """Verificar anomalia de WebGL"""
        webgl_vendor = device_info.get("webgl_vendor", "")
        webgl_renderer = device_info.get("webgl_renderer", "")

        # WebGL desabilitado
        if not webgl_vendor and not webgl_renderer:
            return True

        # Valores genéricos (spoofing)
        generic_values = ["webgl", "mesa", "generic"]
        if any(g in webgl_renderer.lower() for g in generic_values):
            return True

        return False

    def _check_canvas_anomaly(self, device_info: Dict[str, Any]) -> bool:
        """Verificar anomalia de Canvas fingerprint"""
        canvas_hash = device_info.get("canvas_hash", "")

        # Canvas bloqueado ou vazio
        if not canvas_hash or canvas_hash == "blocked":
            return True

        # Hash muito comum (spoofing)
        common_hashes = device_info.get("known_common_hashes", [])
        if canvas_hash in common_hashes:
            return True

        return False

    def _check_audio_anomaly(self, device_info: Dict[str, Any]) -> bool:
        """Verificar anomalia de Audio fingerprint"""
        audio_hash = device_info.get("audio_hash", "")

        # Audio bloqueado
        if not audio_hash or audio_hash == "blocked":
            return True

        return False

    def _check_screen_anomaly(self, device_info: Dict[str, Any]) -> bool:
        """Verificar anomalia de resolução de tela"""
        width = device_info.get("screen_width", 0)
        height = device_info.get("screen_height", 0)

        # Resolução suspeita
        if (width, height) in self.SUSPICIOUS_RESOLUTIONS:
            return True

        # Resolução muito baixa para navegação normal
        if width < 320 or height < 240:
            return True

        # Aspect ratio muito estranho
        if width > 0 and height > 0:
            ratio = max(width, height) / min(width, height)
            if ratio > 4:  # Muito alongado
                return True

        return False

    def _calculate_confidence(
        self,
        device_info: Dict[str, Any],
        baseline: Optional[Dict]
    ) -> float:
        """Calcular confiança na análise"""
        confidence = 0.0

        # Dados disponíveis
        important_fields = [
            "user_agent", "screen_resolution", "timezone",
            "webgl_vendor", "canvas_hash", "plugins"
        ]

        available = sum(1 for f in important_fields if device_info.get(f))
        confidence += (available / len(important_fields)) * 0.5

        # Baseline disponível
        if baseline:
            confidence += 0.3
            if baseline.get("samples_count", 0) > 5:
                confidence += 0.1

        # Dados de IP enriquecidos
        if device_info.get("ip_info"):
            confidence += 0.1

        return min(confidence, 1.0)

    def register_device(
        self,
        fingerprint_hash: str,
        user_id: str,
        is_fraud: bool = False
    ):
        """Registrar uso de dispositivo"""
        now = datetime.now()

        if fingerprint_hash not in self.device_history:
            self.device_history[fingerprint_hash] = DeviceHistory(
                fingerprint_hash=fingerprint_hash,
                first_seen=now,
                last_seen=now,
                usage_count=1,
                users_count=1,
                fraud_count=1 if is_fraud else 0,
                fraud_rate=1.0 if is_fraud else 0.0
            )
        else:
            history = self.device_history[fingerprint_hash]
            history.last_seen = now
            history.usage_count += 1

            if is_fraud:
                history.fraud_count += 1

            history.fraud_rate = history.fraud_count / history.usage_count

    def get_device_reputation(
        self,
        fingerprint_hash: str
    ) -> Dict[str, Any]:
        """Obter reputação do dispositivo"""
        history = self.device_history.get(fingerprint_hash)

        if not history:
            return {
                "known": False,
                "reputation": "unknown",
                "risk_level": "medium"
            }

        # Calcular reputação
        if history.fraud_rate > 0.3:
            reputation = "bad"
            risk_level = "high"
        elif history.fraud_rate > 0.1:
            reputation = "suspicious"
            risk_level = "medium-high"
        elif history.fraud_rate > 0:
            reputation = "mixed"
            risk_level = "medium"
        else:
            reputation = "good"
            risk_level = "low"

        return {
            "known": True,
            "reputation": reputation,
            "risk_level": risk_level,
            "usage_count": history.usage_count,
            "users_count": history.users_count,
            "fraud_rate": history.fraud_rate,
            "first_seen": history.first_seen.isoformat(),
            "last_seen": history.last_seen.isoformat()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        total_devices = len(self.device_history)
        bad_devices = sum(
            1 for h in self.device_history.values()
            if h.fraud_rate > 0.1
        )

        return {
            "version": self.VERSION,
            "total_devices": total_devices,
            "bad_devices": bad_devices,
            "thresholds": {
                "high_fraud_rate": self.HIGH_FRAUD_RATE_THRESHOLD,
                "suspicious_users": self.SUSPICIOUS_USERS_THRESHOLD
            }
        }


def create_device_analyzer(
    config: Optional[Dict[str, Any]] = None
) -> DeviceAnalyzer:
    """Factory function para criar DeviceAnalyzer"""
    return DeviceAnalyzer(config)
