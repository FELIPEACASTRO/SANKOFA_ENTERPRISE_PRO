"""
Sankofa Enterprise Pro - Hard Rules Engine
Motor de Regras Duras que retorna no MESMO FORMATO que o ML.
Quem chama não sabe se a resposta veio do ML ou das regras duras.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class HardRulePrediction:
    """
    Resultado de avaliação de regra dura.
    MESMO FORMATO que FraudPrediction do ML para ser indistinguível.
    """
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    model_version: str
    detection_reason: List[str]
    timestamp: str
    triggered_rules: Optional[List[Dict[str, Any]]] = None
    source: str = "HARD_RULES"

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário no formato ML"""
        result = asdict(self)
        result.pop('triggered_rules', None)
        result.pop('source', None)
        return result
    
    def to_unified_response(self) -> Dict[str, Any]:
        """
        Retorna resposta unificada INDISTINGUÍVEL do ML.
        Remove campos específicos de regras duras.
        """
        return {
            "transaction_id": self.transaction_id,
            "is_fraud": self.is_fraud,
            "fraud_probability": self.fraud_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "detection_reason": self.detection_reason,
            "timestamp": self.timestamp
        }


class HardRulesEngine:
    """
    Motor de Regras Duras Enterprise-Grade
    
    Features:
    - Avalia transações contra 216+ regras
    - Retorna no MESMO formato que FraudPrediction do ML
    - Suporte a múltiplas condições (AND/OR)
    - Cache de regras em memória
    - Suporte a 20 campos e 16 operadores
    """
    
    VERSION = "2.0.0"
    
    OPERATORS = {
        "==": lambda a, b: str(a).lower() == str(b).lower(),
        "!=": lambda a, b: str(a).lower() != str(b).lower(),
        ">": lambda a, b: float(a) > float(b),
        "<": lambda a, b: float(a) < float(b),
        ">=": lambda a, b: float(a) >= float(b),
        "<=": lambda a, b: float(a) <= float(b),
        "contains": lambda a, b: str(b).lower() in str(a).lower(),
        "not_contains": lambda a, b: str(b).lower() not in str(a).lower(),
        "starts_with": lambda a, b: str(a).lower().startswith(str(b).lower()),
        "ends_with": lambda a, b: str(a).lower().endswith(str(b).lower()),
        "in": lambda a, b: str(a).lower() in [x.strip().lower() for x in str(b).split(",")],
        "not_in": lambda a, b: str(a).lower() not in [x.strip().lower() for x in str(b).split(",")],
        "between": lambda a, b: float(b.split(",")[0]) <= float(a) <= float(b.split(",")[1]) if "," in str(b) else False,
        "regex": lambda a, b: bool(re.match(str(b), str(a))),
        "is_null": lambda a, b: a is None or str(a).lower() in ["none", "null", ""],
        "is_not_null": lambda a, b: a is not None and str(a).lower() not in ["none", "null", ""],
    }
    
    ACTION_TO_FRAUD = {
        "block": True,
        "review": True,
        "alert": False,
        "approve": False,
        "step_up": True,
        "score_adjust": False,
    }
    
    ACTION_TO_SCORE = {
        "block": 0.95,
        "review": 0.75,
        "alert": 0.50,
        "approve": 0.10,
        "step_up": 0.80,
        "score_adjust": 0.60,
    }
    
    def __init__(self):
        self._rules_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 30
        self._conn_string = os.environ.get("DATABASE_URL")
    
    def _get_connection(self):
        """Obtém conexão com o banco de dados"""
        return psycopg2.connect(self._conn_string, cursor_factory=RealDictCursor)
    
    def _load_rules(self) -> List[Dict]:
        """Carrega regras do PostgreSQL com cache"""
        now = time.time()
        if self._rules_cache and (now - self._cache_timestamp) < self._cache_ttl:
            return self._rules_cache
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, condition, conditions_json, logic_operator,
                               action, action_config, rule_type, priority, description,
                               enabled
                        FROM hard_rules
                        WHERE enabled = true
                        ORDER BY priority ASC, id ASC
                    """)
                    rows = cur.fetchall()
                    self._rules_cache = [dict(row) for row in rows]
                    self._cache_timestamp = now
                    return self._rules_cache
        except Exception as e:
            print(f"Error loading hard rules: {e}")
            return []
    
    def _evaluate_condition(self, condition: Dict, transaction: Dict) -> bool:
        """Avalia uma condição individual contra a transação"""
        try:
            field = condition.get("field", "")
            operator = condition.get("operator", "==")
            expected_value = condition.get("value", "")
            
            actual_value = transaction.get(field)
            
            if actual_value is None and operator not in ["is_null", "is_not_null"]:
                return False
            
            if operator in self.OPERATORS:
                return self.OPERATORS[operator](actual_value, expected_value)
            
            return False
        except Exception:
            return False
    
    def _evaluate_rule(self, rule: Dict, transaction: Dict) -> Tuple[bool, List[str]]:
        """
        Avalia uma regra contra a transação.
        Retorna (triggered, reasons)
        """
        conditions_json = rule.get("conditions_json")
        logic_operator = rule.get("logic_operator", "AND")
        
        if not conditions_json:
            return False, []
        
        if isinstance(conditions_json, str):
            import json
            try:
                conditions_json = json.loads(conditions_json)
            except (json.JSONDecodeError, TypeError, ValueError):
                return False, []
        
        if not conditions_json:
            return False, []
        
        results = []
        matched_conditions = []
        
        for condition in conditions_json:
            matched = self._evaluate_condition(condition, transaction)
            results.append(matched)
            if matched:
                field = condition.get("field", "")
                operator = condition.get("operator", "==")
                value = condition.get("value", "")
                matched_conditions.append(f"{field} {operator} {value}")
        
        if logic_operator == "AND":
            triggered = all(results) if results else False
        else:
            triggered = any(results) if results else False
        
        if triggered:
            reasons = [
                f"Rule '{rule.get('name')}' triggered",
                f"Conditions: {', '.join(matched_conditions)}",
            ]
            description = rule.get("description")
            if description:
                reasons.append(str(description)[:100])
            return True, reasons
        
        return False, []
    
    def evaluate(self, transaction: Dict) -> HardRulePrediction:
        """
        Avalia transação contra todas as regras duras.
        Retorna HardRulePrediction no MESMO formato que FraudPrediction do ML.
        """
        start_time = time.time()
        
        rules = self._load_rules()
        
        triggered_rules = []
        all_reasons = []
        highest_score = 0.0
        primary_action = "approve"
        
        for rule in rules:
            triggered, reasons = self._evaluate_rule(rule, transaction)
            
            if triggered:
                action = rule.get("action", "alert")
                score = self.ACTION_TO_SCORE.get(action, 0.5)
                
                triggered_rules.append({
                    "id": rule.get("id"),
                    "name": rule.get("name"),
                    "action": action,
                    "score": score,
                    "priority": rule.get("priority", 999),
                })
                
                all_reasons.extend(reasons)
                
                if score > highest_score:
                    highest_score = score
                    primary_action = action
        
        is_fraud = self.ACTION_TO_FRAUD.get(primary_action, False)
        
        if highest_score >= 0.95:
            risk_level = "CRITICAL"
        elif highest_score >= 0.80:
            risk_level = "HIGH"
        elif highest_score >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        processing_time = (time.time() - start_time) * 1000
        
        confidence = 0.95 if triggered_rules else 0.50
        
        transaction_id = transaction.get("id", transaction.get("transaction_id", f"TXN_{int(time.time()*1000)}"))
        
        return HardRulePrediction(
            transaction_id=str(transaction_id),
            is_fraud=is_fraud,
            fraud_probability=highest_score,
            risk_score=highest_score,
            risk_level=risk_level,
            confidence=confidence,
            processing_time_ms=round(processing_time, 2),
            model_version=f"HARD_RULES_{self.VERSION}",
            detection_reason=all_reasons[:5] if all_reasons else ["No rules triggered"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            triggered_rules=triggered_rules,
            source="HARD_RULES"
        )
    
    def evaluate_batch(self, transactions: List[Dict]) -> List[HardRulePrediction]:
        """Avalia múltiplas transações em batch"""
        return [self.evaluate(txn) for txn in transactions]
    
    def get_rules_count(self) -> int:
        """Retorna contagem de regras ativas"""
        rules = self._load_rules()
        return len(rules)
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Retorna resumo das regras por categoria"""
        rules = self._load_rules()
        
        by_action = {}
        by_type = {}
        
        for rule in rules:
            action = rule.get("action", "unknown")
            rule_type = rule.get("rule_type", "unknown")
            
            by_action[action] = by_action.get(action, 0) + 1
            by_type[rule_type] = by_type.get(rule_type, 0) + 1
        
        return {
            "total_rules": len(rules),
            "by_action": by_action,
            "by_type": by_type,
            "version": self.VERSION
        }


_hard_rules_engine = None

def get_hard_rules_engine() -> HardRulesEngine:
    """Singleton para HardRulesEngine"""
    global _hard_rules_engine
    if _hard_rules_engine is None:
        _hard_rules_engine = HardRulesEngine()
    return _hard_rules_engine


class UnifiedFraudEngine:
    """
    Motor Unificado de Fraude
    
    Combina ML + Regras Duras em uma única resposta.
    Quem chama NÃO SABE qual motor detectou a fraude.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, ml_engine=None, hard_rules_engine=None):
        self.ml_engine = ml_engine
        self.hard_rules_engine = hard_rules_engine or get_hard_rules_engine()
    
    def evaluate(self, transaction: Dict, use_ml: bool = True, use_rules: bool = True) -> Dict[str, Any]:
        """
        Avalia transação usando ML e/ou Regras Duras.
        Retorna resposta unificada indistinguível.
        """
        start_time = time.time()
        
        ml_result = None
        rules_result = None
        
        if use_rules:
            rules_result = self.hard_rules_engine.evaluate(transaction)
        
        if use_ml and self.ml_engine and hasattr(self.ml_engine, 'predict_single'):
            try:
                ml_result = self.ml_engine.predict_single(transaction)
            except Exception:
                pass
        
        if rules_result and rules_result.is_fraud:
            final_result = rules_result.to_unified_response()
        elif ml_result:
            final_result = ml_result
        elif rules_result:
            final_result = rules_result.to_unified_response()
        else:
            final_result = {
                "transaction_id": transaction.get("id", "unknown"),
                "is_fraud": False,
                "fraud_probability": 0.1,
                "risk_score": 0.1,
                "risk_level": "LOW",
                "confidence": 0.5,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "model_version": f"UNIFIED_{self.VERSION}",
                "detection_reason": ["No fraud indicators detected"],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        return final_result
