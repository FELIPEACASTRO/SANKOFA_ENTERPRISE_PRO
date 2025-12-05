"""
Sankofa Enterprise Pro - Config Store Service
Armazenamento persistente de configurações do sistema
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class ConfigStore:
    """Armazena configurações do sistema"""

    def __init__(self):
        self._config_file = DATA_DIR / "system_config.json"
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo"""
        default_config = {
            "hard_rules": [
                {
                    "id": 1,
                    "name": "Valor acima do limite",
                    "condition": "amount > 50000",
                    "action": "block",
                    "enabled": True,
                },
                {
                    "id": 2,
                    "name": "País de alto risco",
                    "condition": "country in ['XX', 'YY']",
                    "action": "review",
                    "enabled": True,
                },
                {
                    "id": 3,
                    "name": "Primeira transação grande",
                    "condition": "is_first_transaction and amount > 5000",
                    "action": "step_up",
                    "enabled": True,
                },
            ],
            "vip_list": [
                {
                    "id": 1,
                    "identifier": "12345678901",
                    "type": "cpf",
                    "reason": "VIP Customer",
                    "added_at": datetime.now().isoformat(),
                },
            ],
            "hot_list": [
                {
                    "id": 1,
                    "identifier": "98765432100",
                    "type": "cpf",
                    "reason": "Fraud confirmed",
                    "added_at": datetime.now().isoformat(),
                },
            ],
            "manual_review_queue": [],
            "settings": {
                "fraud_threshold": 0.7,
                "step_up_threshold": 0.5,
                "review_threshold": 0.6,
                "max_transaction_value": 100000,
                "enable_step_up": True,
                "enable_manual_review": True,
            },
        }

        try:
            if self._config_file.exists():
                with open(self._config_file, "r") as f:
                    saved = json.load(f)
                    default_config.update(saved)
        except Exception:
            pass

        return default_config

    def _save_config(self):
        """Salva configuração no arquivo"""
        try:
            with open(self._config_file, "w") as f:
                json.dump(self._config, f, indent=2, default=str)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value
        self._save_config()

    def update(self, key: str, item_id: int, data: Dict):
        items = self._config.get(key, [])
        for i, item in enumerate(items):
            if item.get("id") == item_id:
                items[i].update(data)
                break
        self._save_config()

    def add(self, key: str, item: Dict):
        items = self._config.get(key, [])
        max_id = max([it.get("id", 0) for it in items], default=0)
        item["id"] = max_id + 1
        items.append(item)
        self._config[key] = items
        self._save_config()
        return item

    def delete(self, key: str, item_id: int):
        items = self._config.get(key, [])
        self._config[key] = [it for it in items if it.get("id") != item_id]
        self._save_config()


config_store = ConfigStore()
