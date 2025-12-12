"""
Sankofa Enterprise Pro - Mule Account Detection Module
Detecção de contas mula para prevenção de lavagem de dinheiro

Baseado em:
- BioCatch Mule Detection
- MuleHunter AI (RBI India)
- Metro Bank UK implementation (300% uplift)
"""

from .mule_detector import (
    MuleDetector,
    MuleScore,
    MuleIndicator,
    create_mule_detector
)

from .dormancy_analyzer import (
    DormancyAnalyzer,
    DormancyPattern,
    create_dormancy_analyzer
)

from .turnover_analyzer import (
    TurnoverAnalyzer,
    TurnoverPattern,
    create_turnover_analyzer
)

from .network_position_analyzer import (
    NetworkPositionAnalyzer,
    NetworkPosition,
    create_network_analyzer
)

__all__ = [
    # Main detector
    "MuleDetector",
    "MuleScore",
    "MuleIndicator",
    "create_mule_detector",

    # Dormancy analysis
    "DormancyAnalyzer",
    "DormancyPattern",
    "create_dormancy_analyzer",

    # Turnover analysis
    "TurnoverAnalyzer",
    "TurnoverPattern",
    "create_turnover_analyzer",

    # Network analysis
    "NetworkPositionAnalyzer",
    "NetworkPosition",
    "create_network_analyzer",
]

__version__ = "1.0.0"
