"""
Sankofa Enterprise Pro - Behavioral Biometrics Module
Análise comportamental para detecção de fraude

Baseado em:
- BioCatch behavioral biometrics
- IBM Behavioral Authentication
- Academic research on keystroke dynamics
"""

from .behavioral_analyzer import (
    BehavioralAnalyzer,
    BehavioralScore,
    create_behavioral_analyzer
)

from .keystroke_analyzer import (
    KeystrokeAnalyzer,
    KeystrokePattern,
    create_keystroke_analyzer
)

from .mouse_analyzer import (
    MouseAnalyzer,
    MousePattern,
    create_mouse_analyzer
)

from .device_analyzer import (
    DeviceAnalyzer,
    DeviceFingerprint,
    create_device_analyzer
)

__all__ = [
    "BehavioralAnalyzer",
    "BehavioralScore",
    "create_behavioral_analyzer",
    "KeystrokeAnalyzer",
    "KeystrokePattern",
    "create_keystroke_analyzer",
    "MouseAnalyzer",
    "MousePattern",
    "create_mouse_analyzer",
    "DeviceAnalyzer",
    "DeviceFingerprint",
    "create_device_analyzer",
]

__version__ = "1.0.0"
