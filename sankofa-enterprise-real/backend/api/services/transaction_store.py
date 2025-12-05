"""
Sankofa Enterprise Pro - Transaction Store Service
Armazenamento de transações recentes para consulta
"""

import threading
from typing import Dict, List


class TransactionStore:
    """Armazena transações recentes para consulta"""

    def __init__(self, max_size: int = 1000):
        self._lock = threading.Lock()
        self._transactions: List[Dict] = []
        self._max_size = max_size

    def add(self, transaction: Dict):
        """Adiciona transação"""
        with self._lock:
            self._transactions.append(transaction)
            if len(self._transactions) > self._max_size:
                self._transactions = self._transactions[-self._max_size :]

    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Retorna transações recentes"""
        with self._lock:
            return list(reversed(self._transactions[-limit:]))


transaction_store = TransactionStore()
