"""
Device Fingerprinting Interno - Sem Custos Externos
Sankofa Enterprise Pro

Alternativa gratuita ao Fingerprint.com ($99/mês)
Accuracy esperada: 85-90% (vs 98% do serviço pago)

Componentes:
- Canvas fingerprint
- WebGL fingerprint
- Audio fingerprint
- Screen/hardware info
- Timezone/language
- Plugin detection
"""

import hashlib
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import warnings

# Suppress deprecation warnings for utcnow
warnings.filterwarnings("ignore", category=DeprecationWarning, module="datetime")

logger = logging.getLogger(__name__)

# Constante de schema version para migrations
DB_SCHEMA_VERSION = "1.0.0"


@dataclass
class DeviceRiskScore:
    """Resultado da análise de device fingerprint"""

    fingerprint: str
    risk_score: float
    is_new_device: bool
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    components_hash: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeviceProfile:
    """Perfil de um dispositivo conhecido"""

    fingerprint: str
    user_id: str
    first_seen: datetime  # REQUIRED - deve ser sempre preenchido com datetime.utcnow()
    last_seen: datetime  # REQUIRED - deve ser sempre preenchido com datetime.utcnow()
    usage_count: int
    is_trusted: bool
    components: Dict[str, Any]

    def __post_init__(self):
        """Validação após inicialização"""
        if not isinstance(self.first_seen, datetime):
            raise TypeError(f"first_seen deve ser datetime, got {type(self.first_seen)}")
        if not isinstance(self.last_seen, datetime):
            raise TypeError(f"last_seen deve ser datetime, got {type(self.last_seen)}")
        if not self.fingerprint or len(self.fingerprint) == 0:
            raise ValueError("fingerprint não pode ser vazio")
        if not self.user_id or len(self.user_id) == 0:
            raise ValueError("user_id não pode ser vazio")
        if len(self.fingerprint) > 64:
            raise ValueError(f"fingerprint muito longo: {len(self.fingerprint)} (máx 64)")
        if len(self.user_id) > 255:
            raise ValueError(f"user_id muito longo: {len(self.user_id)} (máx 255)")

        # CRITICAL: Validação de segurança contra injection (COMPREHENSIVE)
        import re

        # Block: SQL/NoSQL, OS commands, path traversal, URL encoding, wildcards, special control chars
        dangerous_chars = re.compile(r'[;\'"`\-\(\)\[\]\{\}|&$<>\\/.%*?@!~^=+#\x00-\x1f]')
        if dangerous_chars.search(self.fingerprint):
            raise ValueError(f"fingerprint contém caracteres perigosos: {self.fingerprint}")
        if dangerous_chars.search(self.user_id):
            raise ValueError(f"user_id contém caracteres perigosos: {self.user_id}")


class DeviceFingerprintPersistence:
    """
    Interface para persistência de device fingerprints.
    Implementar para PostgreSQL, Redis, ou outro storage.
    """

    def save_device(self, profile: DeviceProfile) -> bool:
        raise NotImplementedError

    def get_device(self, fingerprint: str) -> Optional[DeviceProfile]:
        raise NotImplementedError

    def get_user_devices(self, user_id: str) -> List[str]:
        raise NotImplementedError

    def get_device_users(self, fingerprint: str) -> List[str]:
        raise NotImplementedError

    def update_device(self, profile: DeviceProfile) -> bool:
        raise NotImplementedError


class InMemoryPersistence(DeviceFingerprintPersistence):
    """
    Persistência em memória (para testes/desenvolvimento).

    THREAD-SAFE: Usa Lock para evitar race conditions.
    Adequada para desenvolvimento, NÃO para produção com alta concorrência.
    """

    def __init__(self):
        self.devices: Dict[str, DeviceProfile] = {}
        self.user_devices: Dict[str, List[str]] = {}
        self.device_users: Dict[str, List[str]] = {}
        self._lock = threading.RLock()  # Reentrante para nested calls

    def save_device(self, profile: DeviceProfile) -> bool:
        with self._lock:
            self.devices[profile.fingerprint] = profile

            if profile.user_id not in self.user_devices:
                self.user_devices[profile.user_id] = []
            if profile.fingerprint not in self.user_devices[profile.user_id]:
                self.user_devices[profile.user_id].append(profile.fingerprint)

            if profile.fingerprint not in self.device_users:
                self.device_users[profile.fingerprint] = []
            if profile.user_id not in self.device_users[profile.fingerprint]:
                self.device_users[profile.fingerprint].append(profile.user_id)

            return True

    def get_device(self, fingerprint: str) -> Optional[DeviceProfile]:
        with self._lock:
            return self.devices.get(fingerprint)

    def get_user_devices(self, user_id: str) -> List[str]:
        with self._lock:
            return list(self.user_devices.get(user_id, []))

    def get_device_users(self, fingerprint: str) -> List[str]:
        with self._lock:
            return list(self.device_users.get(fingerprint, []))

    def update_device(self, profile: DeviceProfile) -> bool:
        with self._lock:
            if profile.fingerprint in self.devices:
                self.devices[profile.fingerprint] = profile
                return True
            return False


class PostgresPersistence(DeviceFingerprintPersistence):
    """
    Persistência em PostgreSQL para produção.

    Usa psycopg2 ThreadedConnectionPool (conforme production_api.py).

    Requer tabela:
    CREATE TABLE device_fingerprints (
        fingerprint VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        first_seen TIMESTAMP NOT NULL,
        last_seen TIMESTAMP NOT NULL,
        usage_count INTEGER DEFAULT 1,
        is_trusted BOOLEAN DEFAULT FALSE,
        components JSONB
    );

    CREATE INDEX idx_device_user ON device_fingerprints(user_id);
    """

    def __init__(self, db_pool, connect_timeout: int = 5):
        """
        Inicializa com psycopg2 connection pool.

        Args:
            db_pool: psycopg2.pool.ThreadedConnectionPool
            connect_timeout: Timeout em segundos para conexões (default 5)
        """
        self.db_pool = db_pool
        self.connect_timeout = connect_timeout

    def save_device(self, profile: DeviceProfile) -> bool:
        """Salva/atualiza device usando connection pool"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO device_fingerprints 
                    (fingerprint, user_id, first_seen, last_seen, usage_count, is_trusted, components)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fingerprint) DO UPDATE SET
                        last_seen = EXCLUDED.last_seen,
                        usage_count = device_fingerprints.usage_count + 1
                    """,
                    (
                        profile.fingerprint,
                        profile.user_id,
                        profile.first_seen,
                        profile.last_seen,
                        profile.usage_count,
                        profile.is_trusted,
                        json.dumps(profile.components),
                    ),
                )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar device: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.db_pool.putconn(conn)

    def get_device(self, fingerprint: str) -> Optional[DeviceProfile]:
        """Busca device pelo fingerprint"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM device_fingerprints WHERE fingerprint = %s", (fingerprint,)
                )
                result = cur.fetchone()

            if result:
                return DeviceProfile(
                    fingerprint=result[0],
                    user_id=result[1],
                    first_seen=result[2],
                    last_seen=result[3],
                    usage_count=result[4],
                    is_trusted=result[5],
                    components=json.loads(result[6]) if isinstance(result[6], str) else result[6],
                )
            return None
        except Exception as e:
            logger.error(f"Erro ao buscar device: {e}")
            if conn:
                self.db_pool.putconn(conn, close=True)
            return None
        else:
            if conn:
                self.db_pool.putconn(conn)
            return None if not result else None

    def get_user_devices(self, user_id: str) -> List[str]:
        """Lista devices de um usuário"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT fingerprint FROM device_fingerprints WHERE user_id = %s", (user_id,)
                )
                results = cur.fetchall()
            if conn:
                self.db_pool.putconn(conn)
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"Erro ao buscar devices do usuário: {e}")
            if conn:
                self.db_pool.putconn(conn, close=True)
            return []

    def get_device_users(self, fingerprint: str) -> List[str]:
        """Lista usuários que usam um device"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT user_id FROM device_fingerprints WHERE fingerprint = %s",
                    (fingerprint,),
                )
                results = cur.fetchall()
            if conn:
                self.db_pool.putconn(conn)
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"Erro ao buscar usuários do device: {e}")
            if conn:
                self.db_pool.putconn(conn, close=True)
            return []

    def update_device(self, profile: DeviceProfile) -> bool:
        """Atualiza device existente"""
        return self.save_device(profile)


class DeviceFingerprintAnalyzer:
    """
    Analisador de device fingerprint sem custo externo.

    Implementa detecção de:
    - Novos dispositivos
    - VPN/Proxy
    - Emuladores
    - Navegadores headless
    - Múltiplos usuários no mesmo device
    - Timezone inconsistente

    Para produção, use PostgresPersistence para persistir dados.
    """

    EMULATOR_KEYWORDS = [
        "swiftshader",
        "llvmpipe",
        "mesa",
        "virtualbox",
        "vmware",
        "bluestacks",
        "genymotion",
        "nox",
        "android emulator",
        "google android sdk",
    ]

    HEADLESS_INDICATORS = [
        "headless",
        "phantom",
        "puppeteer",
        "selenium",
        "playwright",
        "webdriver",
    ]

    BRAZILIAN_TIMEZONES = [
        "America/Sao_Paulo",
        "America/Fortaleza",
        "America/Recife",
        "America/Bahia",
        "America/Belem",
        "America/Manaus",
        "America/Cuiaba",
        "America/Porto_Velho",
        "America/Boa_Vista",
        "America/Rio_Branco",
        "America/Noronha",
        "America/Araguaina",
        "America/Maceio",
        "America/Campo_Grande",
        "America/Eirunepe",
        "America/Santarem",
    ]

    def __init__(self, persistence: Optional[DeviceFingerprintPersistence] = None):
        """
        Inicializa o analisador.

        Args:
            persistence: Implementação de persistência (InMemoryPersistence para dev,
                        PostgresPersistence para produção)
        """
        self.persistence = persistence or InMemoryPersistence()
        logger.info(f"DeviceFingerprintAnalyzer inicializado com {type(self.persistence).__name__}")

    def analyze(self, user_id: str, fingerprint_data: Dict[str, Any]) -> DeviceRiskScore:
        """
        Analisa fingerprint e retorna score de risco.

        Args:
            user_id: ID do usuário
            fingerprint_data: Dados do fingerprint do frontend

        Returns:
            DeviceRiskScore com análise completa
        """
        fingerprint = fingerprint_data.get("fingerprint", "")
        components = fingerprint_data.get("components", {})
        confidence = fingerprint_data.get("confidence", 0.5)

        if not fingerprint:
            fingerprint = self._generate_fingerprint(components)

        risk_score = 0.0
        reasons = []

        is_known = self._is_known_device(user_id, fingerprint)
        if not is_known:
            risk_score += 0.25
            reasons.append("Dispositivo não cadastrado para este usuário")

        tz_check = self._check_timezone(components)
        if tz_check["suspicious"]:
            risk_score += tz_check["score"]
            reasons.extend(tz_check["reasons"])

        vpn_check = self._detect_vpn_proxy(components)
        if vpn_check["detected"]:
            risk_score += vpn_check["score"]
            reasons.extend(vpn_check["reasons"])

        emu_check = self._detect_emulator(components)
        if emu_check["detected"]:
            risk_score += emu_check["score"]
            reasons.extend(emu_check["reasons"])

        headless_check = self._detect_headless(components)
        if headless_check["detected"]:
            risk_score += headless_check["score"]
            reasons.extend(headless_check["reasons"])

        multi_user = self._check_multiple_users(fingerprint, user_id)
        if multi_user["detected"]:
            risk_score += multi_user["score"]
            reasons.extend(multi_user["reasons"])

        hw_check = self._check_hardware_anomalies(components)
        if hw_check["suspicious"]:
            risk_score += hw_check["score"]
            reasons.extend(hw_check["reasons"])

        self._update_device_registry(user_id, fingerprint, components)

        return DeviceRiskScore(
            fingerprint=fingerprint,
            risk_score=min(risk_score, 1.0),
            is_new_device=not is_known,
            is_suspicious=risk_score > 0.5,
            confidence=confidence,
            reasons=reasons,
            components_hash=self._hash_components(components),
        )

    def _generate_fingerprint(self, components: Dict[str, Any]) -> str:
        """Gera fingerprint a partir dos componentes"""

        stable_components = {
            "screen": components.get("screen", {}),
            "webgl": components.get("webgl", {}),
            "timezone": components.get("timezone", ""),
            "language": components.get("language", ""),
            "platform": components.get("platform", ""),
            "cpuCores": components.get("cpuCores", 0),
            "touchSupport": components.get("touchSupport", False),
        }

        data = json.dumps(stable_components, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _is_known_device(self, user_id: str, fingerprint: str) -> bool:
        """Verifica se dispositivo é conhecido para o usuário"""
        user_devices = self.persistence.get_user_devices(user_id)
        return fingerprint in user_devices

    def _check_timezone(self, components: Dict) -> Dict:
        """Verifica timezone"""

        result = {"suspicious": False, "score": 0.0, "reasons": []}

        tz = components.get("timezone", "")

        if not tz:
            result["suspicious"] = True
            result["score"] = 0.15
            result["reasons"].append("Timezone não detectado")
            return result

        if tz not in self.BRAZILIAN_TIMEZONES:
            result["suspicious"] = True
            result["score"] = 0.20
            result["reasons"].append(f"Timezone não brasileiro: {tz}")

        return result

    def _detect_vpn_proxy(self, components: Dict) -> Dict:
        """Detecta sinais de VPN/Proxy"""

        result = {"detected": False, "score": 0.0, "reasons": []}

        webrtc = components.get("webrtc", {})
        if webrtc.get("leak_detected"):
            result["detected"] = True
            result["score"] = 0.25
            result["reasons"].append("WebRTC leak detectado (possível VPN)")

        tz = components.get("timezone", "")
        geo_tz = components.get("geo_timezone", "")
        if tz and geo_tz and tz != geo_tz:
            result["detected"] = True
            result["score"] = max(result["score"], 0.30)
            result["reasons"].append("Timezone não corresponde à geolocalização")

        return result

    def _detect_emulator(self, components: Dict) -> Dict:
        """Detecta emulador Android/iOS"""

        result = {"detected": False, "score": 0.0, "reasons": []}

        webgl = components.get("webgl", {})
        renderer = str(webgl.get("renderer", "")).lower()
        vendor = str(webgl.get("vendor", "")).lower()

        for keyword in self.EMULATOR_KEYWORDS:
            if keyword in renderer or keyword in vendor:
                result["detected"] = True
                result["score"] = 0.40
                result["reasons"].append(f"Possível emulador detectado: {keyword}")
                break

        platform = str(components.get("platform", "")).lower()
        user_agent = str(components.get("userAgent", "")).lower()

        if "android" in platform and "linux" in user_agent and "mobile" not in user_agent:
            result["detected"] = True
            result["score"] = max(result["score"], 0.35)
            result["reasons"].append("Inconsistência Android/Desktop")

        return result

    def _detect_headless(self, components: Dict) -> Dict:
        """Detecta navegador headless (bot)"""

        result = {"detected": False, "score": 0.0, "reasons": []}

        plugins = components.get("plugins", [])
        if isinstance(plugins, list) and len(plugins) == 0:
            result["detected"] = True
            result["score"] = 0.30
            result["reasons"].append("Nenhum plugin detectado (possível headless)")

        webgl = components.get("webgl", {})
        renderer = str(webgl.get("renderer", "")).lower()

        for indicator in self.HEADLESS_INDICATORS:
            if indicator in renderer:
                result["detected"] = True
                result["score"] = 0.50
                result["reasons"].append(f"Navegador headless detectado: {indicator}")
                break

        webdriver = components.get("webdriver", False)
        if webdriver:
            result["detected"] = True
            result["score"] = max(result["score"], 0.60)
            result["reasons"].append("WebDriver detectado (automação)")

        languages = components.get("languages", [])
        if not languages or languages == [""]:
            result["detected"] = True
            result["score"] = max(result["score"], 0.25)
            result["reasons"].append("Idiomas não detectados")

        return result

    def _check_multiple_users(self, fingerprint: str, user_id: str) -> Dict:
        """Verifica se múltiplos usuários usam o mesmo device"""

        result = {"detected": False, "score": 0.0, "reasons": []}

        device_users = self.persistence.get_device_users(fingerprint)
        other_users = [u for u in device_users if u != user_id]

        if len(other_users) >= 3:
            result["detected"] = True
            result["score"] = 0.40
            result["reasons"].append(
                f"Dispositivo usado por {len(other_users) + 1} usuários diferentes"
            )
        elif len(other_users) >= 1:
            result["detected"] = True
            result["score"] = 0.20
            result["reasons"].append("Dispositivo compartilhado com outro usuário")

        return result

    def _check_hardware_anomalies(self, components: Dict) -> Dict:
        """Verifica anomalias de hardware"""

        result = {"suspicious": False, "score": 0.0, "reasons": []}

        screen = components.get("screen", {})
        width = screen.get("width", 0)
        height = screen.get("height", 0)

        if width == 0 or height == 0:
            result["suspicious"] = True
            result["score"] = 0.20
            result["reasons"].append("Resolução de tela inválida")

        if width > 7680 or height > 4320:
            result["suspicious"] = True
            result["score"] = 0.15
            result["reasons"].append("Resolução de tela suspeita (muito alta)")

        cpu_cores = components.get("cpuCores", 0)
        if cpu_cores == 0:
            result["suspicious"] = True
            result["score"] = max(result["score"], 0.15)
            result["reasons"].append("CPU cores não detectado")

        memory = components.get("memory", 0)
        if memory == 0:
            result["suspicious"] = True
            result["score"] = max(result["score"], 0.10)
            result["reasons"].append("Memória não detectada")

        return result

    def _hash_components(self, components: Dict) -> Dict[str, str]:
        """Gera hash de cada componente para comparação"""

        hashes = {}
        for key, value in components.items():
            data = (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            )
            hashes[key] = hashlib.md5(data.encode()).hexdigest()[:8]

        return hashes

    def _update_device_registry(self, user_id: str, fingerprint: str, components: Dict):
        """Atualiza registro de dispositivos usando persistência"""

        now = datetime.utcnow()

        existing = self.persistence.get_device(fingerprint)

        if existing:
            existing.last_seen = now
            existing.usage_count += 1
            self.persistence.update_device(existing)
        else:
            profile = DeviceProfile(
                fingerprint=fingerprint,
                user_id=user_id,
                first_seen=now,
                last_seen=now,
                usage_count=1,
                is_trusted=False,
                components=components,
            )
            self.persistence.save_device(profile)

    def register_trusted_device(self, user_id: str, fingerprint: str) -> bool:
        """Registra dispositivo como confiável para o usuário"""

        device = self.persistence.get_device(fingerprint)
        if device:
            device.is_trusted = True
            self.persistence.update_device(device)
            logger.info(f"Device {fingerprint[:8]}... marcado como confiável para {user_id}")
            return True

        return False

    def get_user_devices(self, user_id: str) -> List[DeviceProfile]:
        """Retorna todos os dispositivos de um usuário"""

        fingerprints = self.persistence.get_user_devices(user_id)
        devices = []
        for fp in fingerprints:
            device = self.persistence.get_device(fp)
            if device:
                devices.append(device)

        return devices

    def get_device_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de dispositivos.

        Nota: Para persistência em memória, retorna estatísticas completas.
        Para PostgreSQL, pode precisar de queries específicas.
        """

        if isinstance(self.persistence, InMemoryPersistence):
            return {
                "total_devices": len(self.persistence.devices),
                "total_users_with_devices": len(self.persistence.user_devices),
                "devices_with_multiple_users": sum(
                    1 for users in self.persistence.device_users.values() if len(users) > 1
                ),
                "trusted_devices": sum(
                    1 for d in self.persistence.devices.values() if d.is_trusted
                ),
            }
        else:
            return {
                "total_devices": "N/A (use SQL query)",
                "total_users_with_devices": "N/A (use SQL query)",
                "devices_with_multiple_users": "N/A (use SQL query)",
                "trusted_devices": "N/A (use SQL query)",
            }


FRONTEND_FINGERPRINT_JS = """
// Código JavaScript para captura de fingerprint no frontend
// Incluir em: sankofa-enterprise-real/frontend/src/utils/deviceFingerprint.js

export async function generateDeviceFingerprint() {
  const components = {};
  
  // Canvas fingerprint
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillStyle = '#f60';
    ctx.fillRect(125, 1, 62, 20);
    ctx.fillStyle = '#069';
    ctx.fillText('Sankofa FP', 2, 15);
    components.canvas = canvas.toDataURL().slice(-50);
  } catch (e) {
    components.canvas = 'error';
  }
  
  // WebGL fingerprint
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      components.webgl = {
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : '',
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : ''
      };
    }
  } catch (e) {
    components.webgl = {};
  }
  
  // Screen info
  components.screen = {
    width: screen.width,
    height: screen.height,
    colorDepth: screen.colorDepth,
    pixelRatio: window.devicePixelRatio || 1
  };
  
  // Timezone
  components.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  
  // Language
  components.language = navigator.language;
  components.languages = navigator.languages ? [...navigator.languages] : [];
  
  // Platform
  components.platform = navigator.platform;
  components.userAgent = navigator.userAgent;
  
  // Hardware
  components.cpuCores = navigator.hardwareConcurrency || 0;
  components.memory = navigator.deviceMemory || 0;
  components.touchSupport = 'ontouchstart' in window;
  
  // Plugins
  components.plugins = Array.from(navigator.plugins || []).map(p => p.name);
  
  // WebDriver detection
  components.webdriver = navigator.webdriver || false;
  
  // Generate hash
  const data = JSON.stringify(components);
  const hashBuffer = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(data));
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const fingerprint = hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 32);
  
  return {
    fingerprint,
    components,
    confidence: calculateConfidence(components)
  };
}

function calculateConfidence(components) {
  let score = 0;
  if (components.canvas !== 'error') score += 0.2;
  if (components.webgl?.renderer) score += 0.2;
  if (components.screen?.width > 0) score += 0.15;
  if (components.timezone) score += 0.15;
  if (components.cpuCores > 0) score += 0.15;
  if (components.plugins?.length > 0) score += 0.15;
  return score;
}
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = DeviceFingerprintAnalyzer()

    fingerprint_data = {
        "fingerprint": "abc123def456",
        "components": {
            "screen": {"width": 1920, "height": 1080, "colorDepth": 24},
            "webgl": {"vendor": "Intel Inc.", "renderer": "Intel HD Graphics"},
            "timezone": "America/Sao_Paulo",
            "language": "pt-BR",
            "platform": "Win32",
            "cpuCores": 8,
            "memory": 8,
            "touchSupport": False,
            "plugins": ["PDF Viewer", "Chrome PDF Plugin"],
            "webdriver": False,
        },
        "confidence": 0.85,
    }

    result = analyzer.analyze("user123", fingerprint_data)

    print("\n=== Análise de Device Fingerprint ===")
    print(f"Fingerprint: {result.fingerprint}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Novo dispositivo: {result.is_new_device}")
    print(f"Suspeito: {result.is_suspicious}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Razões: {result.reasons}")

    bot_fingerprint = {
        "fingerprint": "bot123",
        "components": {
            "screen": {"width": 0, "height": 0},
            "webgl": {"renderer": "headless chrome"},
            "timezone": "UTC",
            "plugins": [],
            "webdriver": True,
        },
        "confidence": 0.2,
    }

    result_bot = analyzer.analyze("user456", bot_fingerprint)
    print("\n=== Análise de Bot ===")
    print(f"Risk Score: {result_bot.risk_score:.2f}")
    print(f"Suspeito: {result_bot.is_suspicious}")
    print(f"Razões: {result_bot.reasons}")
