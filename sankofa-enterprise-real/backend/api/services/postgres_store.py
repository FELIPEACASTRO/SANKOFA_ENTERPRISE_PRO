"""
Sankofa Enterprise Pro - PostgreSQL Store Service
Armazenamento persistente usando PostgreSQL em vez de arquivo JSON
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


class SimpleCache:
    """Cache simples em memória com TTL para reduzir latência"""

    def __init__(self, default_ttl: int = 30):
        self._cache = {}
        self._default_ttl = default_ttl

    def get(self, key: str):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry["expires"]:
                return entry["value"]
            del self._cache[key]
        return None

    def set(self, key: str, value, ttl: int = None):
        self._cache[key] = {"value": value, "expires": time.time() + (ttl or self._default_ttl)}

    def invalidate(self, key: str = None):
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


_dashboard_cache = SimpleCache(default_ttl=30)


class PostgresStore:
    """Armazena configurações e dados no PostgreSQL"""

    def __init__(self):
        self._conn_string = os.environ.get("DATABASE_URL")

    def _get_connection(self):
        """Obtém conexão com o banco de dados"""
        return psycopg2.connect(self._conn_string, cursor_factory=RealDictCursor)

    def get_hard_rules(self) -> List[Dict]:
        """Retorna todas as hard rules do PostgreSQL
        Com cache em memória (TTL 30s) para reduzir latência
        """
        cache_key = "hard_rules"
        cached = _dashboard_cache.get(cache_key)
        if cached:
            return cached

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, name, condition, conditions_json, logic_operator,
                               action, action_config, rule_type, priority, description,
                               enabled, created_at, updated_at
                        FROM hard_rules
                        ORDER BY priority ASC, id ASC
                    """
                    )
                    rows = cur.fetchall()
                    result = [dict(row) for row in rows]
                    _dashboard_cache.set(cache_key, result, ttl=30)
                    return result
        except Exception as e:
            print(f"Error fetching hard_rules: {e}")
            return []

    def add_hard_rule(
        self,
        name: str,
        condition: str,
        action: str,
        enabled: bool = True,
        conditions_json: list = None,
        logic_operator: str = "AND",
        priority: int = 1,
        description: str = None,
        action_config: dict = None,
        rule_type: str = "blocking",
    ) -> Dict:
        """Adiciona nova hard rule com suporte a condições múltiplas"""
        try:
            import json

            conditions_json = conditions_json or []
            action_config = action_config or {}

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO hard_rules (name, condition, conditions_json, logic_operator,
                                               action, action_config, rule_type, priority, description,
                                               enabled, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id, name, condition, conditions_json, logic_operator,
                                  action, action_config, rule_type, priority, description,
                                  enabled, created_at, updated_at
                    """,
                        (
                            name,
                            condition,
                            json.dumps(conditions_json),
                            logic_operator,
                            action,
                            json.dumps(action_config),
                            rule_type,
                            priority,
                            description,
                            enabled,
                        ),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    _dashboard_cache.invalidate("hard_rules")
                    return dict(row)
        except Exception as e:
            print(f"Error adding hard_rule: {e}")
            raise

    def update_hard_rule(self, rule_id: int, data: Dict) -> bool:
        """Atualiza hard rule existente com suporte a condições múltiplas"""
        try:
            import json

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    fields = []
                    values = []

                    allowed_fields = [
                        "name",
                        "condition",
                        "action",
                        "enabled",
                        "logic_operator",
                        "priority",
                        "description",
                        "rule_type",
                    ]
                    json_fields = ["conditions_json", "action_config"]

                    for key in allowed_fields:
                        if key in data:
                            fields.append(f"{key} = %s")
                            values.append(data[key])

                    for key in json_fields:
                        if key in data:
                            fields.append(f"{key} = %s")
                            values.append(
                                json.dumps(data[key])
                                if isinstance(data[key], (list, dict))
                                else data[key]
                            )

                    if fields:
                        fields.append("updated_at = NOW()")
                        values.append(rule_id)
                        query = f"UPDATE hard_rules SET {', '.join(fields)} WHERE id = %s"
                        cur.execute(query, values)
                        conn.commit()
                        _dashboard_cache.invalidate("hard_rules")
                        return cur.rowcount > 0
            return False
        except Exception as e:
            print(f"Error updating hard_rule: {e}")
            return False

    def delete_hard_rule(self, rule_id: int) -> bool:
        """Remove hard rule"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM hard_rules WHERE id = %s", (rule_id,))
                    conn.commit()
                    _dashboard_cache.invalidate("hard_rules")
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting hard_rule: {e}")
            return False

    def get_vip_list(self) -> List[Dict]:
        """Retorna lista VIP do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, identifier, identifier_type as type, reason, 
                               added_by, created_at
                        FROM vip_list
                        ORDER BY id
                    """
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching vip_list: {e}")
            return []

    def add_vip(
        self, identifier: str, identifier_type: str, reason: str, added_by: str = "system"
    ) -> Dict:
        """Adiciona item à lista VIP"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO vip_list (identifier, identifier_type, reason, added_by, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, identifier, identifier_type as type, reason, added_by, created_at
                    """,
                        (identifier, identifier_type, reason, added_by),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding to vip_list: {e}")
            raise

    def delete_vip(self, item_id: int) -> bool:
        """Remove item da lista VIP"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM vip_list WHERE id = %s", (item_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting from vip_list: {e}")
            return False

    def get_hot_list(self) -> List[Dict]:
        """Retorna hot list (blacklist) do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, identifier, identifier_type as type, reason, 
                               added_by, created_at
                        FROM hot_list
                        ORDER BY id
                    """
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching hot_list: {e}")
            return []

    def add_hot(
        self, identifier: str, identifier_type: str, reason: str, added_by: str = "system"
    ) -> Dict:
        """Adiciona item à hot list"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO hot_list (identifier, identifier_type, reason, added_by, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, identifier, identifier_type as type, reason, added_by, created_at
                    """,
                        (identifier, identifier_type, reason, added_by),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding to hot_list: {e}")
            raise

    def delete_hot(self, item_id: int) -> bool:
        """Remove item da hot list"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM hot_list WHERE id = %s", (item_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting from hot_list: {e}")
            return False

    def get_settings(self) -> Dict:
        """Retorna configurações do sistema"""
        default_settings = {
            "fraud_threshold": 0.7,
            "step_up_threshold": 0.5,
            "review_threshold": 0.6,
            "max_transaction_value": 100000,
            "enable_step_up": True,
            "enable_manual_review": True,
        }
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT config_key, config_value
                        FROM system_configs
                        WHERE config_key = 'settings'
                    """
                    )
                    row = cur.fetchone()
                    if row:
                        return row["config_value"]
                    return default_settings
        except Exception as e:
            print(f"Error fetching settings: {e}")
            return default_settings

    def update_settings(self, settings: Dict) -> Dict:
        """Atualiza configurações do sistema"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO system_configs (config_key, config_value, updated_at)
                        VALUES ('settings', %s, NOW())
                        ON CONFLICT (config_key) DO UPDATE SET 
                            config_value = EXCLUDED.config_value,
                            updated_at = NOW()
                        RETURNING config_value
                    """,
                        (json.dumps(settings),),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return row["config_value"] if row else settings
        except Exception as e:
            print(f"Error updating settings: {e}")
            return settings

    def add_audit_log(
        self, action: str, user_id: str = None, details: str = None, ip_address: str = None
    ) -> Dict:
        """Registra ação no log de auditoria"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO audit_logs (action, user_id, details, ip_address, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, action, user_id, details, ip_address, created_at
                    """,
                        (action, user_id, details, ip_address),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding audit_log: {e}")
            return {}

    def get_audit_logs(
        self,
        limit: int = 100,
        action_filter: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """Retorna logs de auditoria"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, action, user_id as user, details, ip_address, created_at as timestamp
                        FROM audit_logs
                        WHERE 1=1
                    """
                    params = []

                    if action_filter:
                        query += " AND action = %s"
                        params.append(action_filter)

                    if start_date:
                        query += " AND created_at >= %s"
                        params.append(start_date)

                    if end_date:
                        query += " AND created_at <= %s"
                        params.append(end_date)

                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching audit_logs: {e}")
            return []

    def get_alerts(self, limit: int = 100, status: str = None) -> List[Dict]:
        """Retorna alertas do sistema"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, alert_id, title as titulo, description as descricao,
                               type as tipo, severity as severidade, status,
                               transaction_id as transacao_id, amount_involved as valor_envolvido,
                               recommended_action as acao_recomendada, investigator as investigador,
                               tags, created_at, updated_at
                        FROM alerts
                        WHERE 1=1
                    """
                    params = []

                    if status:
                        query += " AND status = %s"
                        params.append(status)

                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []

    def add_alert(
        self,
        title: str,
        description: str,
        alert_type: str,
        severity: str,
        transaction_id: str = None,
        amount_involved: float = None,
        recommended_action: str = None,
    ) -> Dict:
        """Adiciona novo alerta"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    import uuid

                    alert_id = f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8].upper()}"

                    cur.execute(
                        """
                        INSERT INTO alerts (alert_id, title, description, type, severity, 
                                          status, transaction_id, amount_involved, 
                                          recommended_action, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'novo', %s, %s, %s, NOW(), NOW())
                        RETURNING id, alert_id, title, description, type, severity, 
                                  status, transaction_id, amount_involved, created_at
                    """,
                        (
                            alert_id,
                            title,
                            description,
                            alert_type,
                            severity,
                            transaction_id,
                            amount_involved,
                            recommended_action,
                        ),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding alert: {e}")
            raise

    def update_alert_status(self, alert_id: int, status: str, investigator: str = None) -> bool:
        """Atualiza status de um alerta"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE alerts SET status = %s, investigator = COALESCE(%s, investigator),
                                        updated_at = NOW()
                        WHERE id = %s
                    """,
                        (status, investigator, alert_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error updating alert status: {e}")
            return False

    def add_feedback(
        self, transaction_id: str, is_fraud: bool, analyst_notes: str = None, analyst_id: str = None
    ) -> Dict:
        """Adiciona feedback sobre transação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO feedback (transaction_id, is_fraud, analyst_notes, analyst_id, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, transaction_id, is_fraud, analyst_notes, analyst_id, created_at
                    """,
                        (transaction_id, is_fraud, analyst_notes, analyst_id),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding feedback: {e}")
            raise

    def get_feedback_list(self, limit: int = 100) -> List[Dict]:
        """Retorna lista de feedbacks"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT f.id, f.transaction_id, f.is_fraud, f.analyst_notes, 
                               f.analyst_id, f.created_at,
                               t.amount, t.cpf as payer_document, t.risk_score
                        FROM feedback f
                        LEFT JOIN transactions t ON f.transaction_id = t.transaction_id
                        ORDER BY f.created_at DESC
                        LIMIT %s
                    """,
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching feedback: {e}")
            return []

    def get_feedback_analytics(self) -> Dict:
        """Retorna analytics de feedback"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN is_fraud = true THEN 1 END) as fraud_count,
                            COUNT(CASE WHEN is_fraud = false THEN 1 END) as legit_count
                        FROM feedback
                    """
                    )
                    row = cur.fetchone()

                    total = row["total"] if row else 0
                    fraud_count = row["fraud_count"] if row else 0
                    legit_count = row["legit_count"] if row else 0

                    return {
                        "total_feedback": total,
                        "fraud_confirmed": fraud_count,
                        "legit_confirmed": legit_count,
                        "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0,
                        "accuracy_improvement": 0,
                    }
        except Exception as e:
            print(f"Error fetching feedback analytics: {e}")
            return {
                "total_feedback": 0,
                "fraud_confirmed": 0,
                "legit_confirmed": 0,
                "fraud_rate": 0,
                "accuracy_improvement": 0,
            }

    def get_pending_reviews(self) -> List[Dict]:
        """Retorna transações pendentes de revisão manual"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT transaction_id, amount, cpf as payer_document, 
                               risk_score, status, created_at, channel
                        FROM transactions
                        WHERE status IN ('pending_review', 'review', 'PENDENTE')
                           OR (risk_score >= 0.5 AND risk_score < 0.7 AND status NOT IN ('APPROVED', 'FRAUD', 'approved', 'rejected'))
                        ORDER BY risk_score DESC, created_at DESC
                        LIMIT 100
                    """
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching pending reviews: {e}")
            return []

    def add_to_manual_review(self, transaction_id: str, reason: str = None) -> bool:
        """Adiciona transação à fila de revisão manual"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE transactions 
                        SET status = 'pending_review'
                        WHERE transaction_id = %s
                    """,
                        (transaction_id,),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error adding to manual review: {e}")
            return False

    def complete_review(
        self, transaction_id: str, decision: str, analyst_id: str = None, notes: str = None
    ) -> bool:
        """Completa revisão manual de transação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    new_status = "approved" if decision == "approve" else "rejected"

                    cur.execute(
                        """
                        UPDATE transactions 
                        SET status = %s
                        WHERE transaction_id = %s
                    """,
                        (new_status, transaction_id),
                    )

                    rows_updated = cur.rowcount

                    if notes or decision:
                        is_fraud = decision == "reject"
                        cur.execute(
                            """
                            DELETE FROM feedback WHERE transaction_id = %s
                        """,
                            (transaction_id,),
                        )
                        cur.execute(
                            """
                            INSERT INTO feedback (transaction_id, is_fraud, analyst_notes, analyst_id, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                        """,
                            (
                                transaction_id,
                                is_fraud,
                                notes or f"Decision: {decision}",
                                analyst_id,
                            ),
                        )

                    conn.commit()
                    return rows_updated > 0
        except Exception as e:
            print(f"Error completing review: {e}")
            return False

    def get_investigations(self) -> List[Dict]:
        """Retorna transações em investigação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT t.transaction_id, t.amount, t.cpf as payer_document, 
                               t.risk_score, t.status, t.created_at, t.channel,
                               t.is_fraud
                        FROM transactions t
                        WHERE t.status IN ('flagged', 'investigating', 'review', 'pending_review')
                           OR t.risk_score >= 0.7
                        ORDER BY t.risk_score DESC, t.created_at DESC
                        LIMIT 100
                    """
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching investigations: {e}")
            return []

    def get_recent_transactions(self, limit: int = 50) -> List[Dict]:
        """Retorna transações recentes do PostgreSQL
        Com cache em memória (TTL 30s) para reduzir latência
        """
        cache_key = f"recent_transactions_{limit}"
        cached = _dashboard_cache.get(cache_key)
        if cached:
            return cached

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT transaction_id, amount, channel, type, status,
                               risk_score, cpf, location, timestamp, created_at
                        FROM transactions
                        ORDER BY created_at DESC
                        LIMIT %s
                    """,
                        (limit,),
                    )
                    rows = cur.fetchall()
                    result = [dict(row) for row in rows]
                    _dashboard_cache.set(cache_key, result, ttl=30)
                    return result
        except Exception as e:
            print(f"Error fetching recent transactions: {e}")
            return []

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """Busca transação por ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT transaction_id, amount, channel, type, status,
                               risk_score, cpf, location, timestamp, created_at
                        FROM transactions
                        WHERE transaction_id = %s
                    """,
                        (transaction_id,),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            print(f"Error fetching transaction: {e}")
            return None

    def get_dashboard_kpis(self, date_from: str = None, date_to: str = None) -> Dict:
        """Retorna KPIs do dashboard baseados em dados reais do PostgreSQL
        Com cache em memória (TTL 30s) para reduzir latência

        Args:
            date_from: Data inicial (default: últimos 30 dias)
            date_to: Data final (default: agora)
        """
        cache_key = "dashboard_kpis"
        cached = _dashboard_cache.get(cache_key)
        if cached:
            return cached

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total_transactions,
                            COUNT(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN 1 END) as frauds_detected,
                            COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as approved,
                            COALESCE(AVG(processing_time_ms), 0) as avg_latency,
                            COALESCE(SUM(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN amount ELSE 0 END), 0) as value_protected,
                            MAX(created_at) as latest_transaction
                        FROM transactions
                    """
                    )
                    total = cur.fetchone()

                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total_transactions,
                            COUNT(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN 1 END) as frauds_detected,
                            COALESCE(AVG(processing_time_ms), 0) as avg_latency
                        FROM transactions
                        WHERE created_at >= (SELECT MAX(created_at) - INTERVAL '1 day' FROM transactions)
                    """
                    )
                    recent = cur.fetchone()

                    total_count = int(total["total_transactions"] or 0)
                    frauds_count = int(total["frauds_detected"] or 0)
                    approved_count = int(total["approved"] or 0)
                    latency_avg = float(total["avg_latency"] or 0)
                    value_protected = float(total["value_protected"] or 0)

                    recent_count = int(recent["total_transactions"] or 0)
                    recent_frauds = int(recent["frauds_detected"] or 0)
                    recent_latency = float(recent["avg_latency"] or 0)

                    approval_rate = 100.0
                    if total_count > 0:
                        approval_rate = (approved_count / total_count) * 100

                    fraud_rate = 0.0
                    if total_count > 0:
                        fraud_rate = (frauds_count / total_count) * 100

                    result = {
                        "transacoes_hoje": total_count,
                        "transacoes_variacao": 0.0,
                        "fraudes_detectadas": frauds_count,
                        "fraudes_variacao": 0.0,
                        "taxa_aprovacao": round(approval_rate, 1),
                        "aprovacao_variacao": 0.0,
                        "latencia_media": round(latency_avg, 2),
                        "latencia_variacao": 0.0,
                        "valor_protegido": round(value_protected, 2),
                        "valor_protegido_ano": round(value_protected * 12, 2),
                        "transacoes_ano": total_count * 12,
                        "taxa_fraude": round(fraud_rate, 2),
                        "transacoes_recentes": recent_count,
                        "ultima_atualizacao": str(total.get("latest_transaction", "")),
                    }

                    _dashboard_cache.set(cache_key, result, ttl=30)
                    return result
        except Exception as e:
            print(f"Error fetching dashboard KPIs: {e}")
            return {
                "transacoes_hoje": 0,
                "transacoes_variacao": 0,
                "fraudes_detectadas": 0,
                "fraudes_variacao": 0,
                "taxa_aprovacao": 100.0,
                "aprovacao_variacao": 0,
                "latencia_media": 0,
                "latencia_variacao": 0,
                "valor_protegido": 0,
                "valor_protegido_ano": 0,
                "transacoes_ano": 0,
            }

    def get_dashboard_timeseries(self) -> List[Dict]:
        """Retorna série temporal de transações por hora (todas as transações agrupadas por hora)
        Com cache em memória (TTL 30s) para reduzir latência
        """
        cache_key = "dashboard_timeseries"
        cached = _dashboard_cache.get(cache_key)
        if cached:
            return cached

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            EXTRACT(HOUR FROM created_at) as hour,
                            COUNT(*) as transactions,
                            COALESCE(AVG(processing_time_ms), 0) as avg_latency
                        FROM transactions
                        GROUP BY EXTRACT(HOUR FROM created_at)
                        ORDER BY hour
                    """
                    )
                    rows = cur.fetchall()

                    hourly_data = {int(row["hour"]): row for row in rows}

                    result = []
                    for hour in range(24):
                        data = hourly_data.get(hour, {"transactions": 0, "avg_latency": 0})
                        result.append(
                            {
                                "time": f"{hour:02d}:00",
                                "transactions": int(data.get("transactions", 0)),
                                "latency": round(float(data.get("avg_latency", 0)), 1),
                            }
                        )

                    _dashboard_cache.set(cache_key, result, ttl=30)
                    return result
        except Exception as e:
            print(f"Error fetching dashboard timeseries: {e}")
            return [{"time": f"{h:02d}:00", "transactions": 0, "latency": 0} for h in range(24)]

    def get_dashboard_channels(self) -> List[Dict]:
        """Retorna estatísticas por canal (todas as transações)
        Com cache em memória (TTL 30s) para reduzir latência
        """
        cache_key = "dashboard_channels"
        cached = _dashboard_cache.get(cache_key)
        if cached:
            return cached

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            COALESCE(channel, 'PIX') as channel,
                            COUNT(*) as transactions,
                            COUNT(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN 1 END) as frauds,
                            COALESCE(SUM(amount), 0) as total_value
                        FROM transactions
                        GROUP BY channel
                    """
                    )
                    rows = cur.fetchall()

                    channel_map = {"pix": "PIX", "card": "Cartão", "ted": "TED", "doc": "DOC"}
                    result = []

                    for row in rows:
                        channel_name = channel_map.get(str(row["channel"]).lower(), row["channel"])
                        result.append(
                            {
                                "name": channel_name,
                                "transactions": int(row["transactions"] or 0),
                                "frauds": int(row["frauds"] or 0),
                                "value": round(float(row["total_value"] or 0), 2),
                            }
                        )

                    if not result:
                        result = [
                            {"name": "PIX", "transactions": 0, "frauds": 0, "value": 0},
                            {"name": "Cartão", "transactions": 0, "frauds": 0, "value": 0},
                            {"name": "TED", "transactions": 0, "frauds": 0, "value": 0},
                            {"name": "DOC", "transactions": 0, "frauds": 0, "value": 0},
                        ]

                    _dashboard_cache.set(cache_key, result, ttl=30)
                    return result
        except Exception as e:
            print(f"Error fetching dashboard channels: {e}")
            return []

    def get_alerts_list(self, limit: int = 100) -> List[Dict]:
        """Retorna lista de alertas do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, alert_id, title, description, type, severity,
                               status, transaction_id, amount_involved, 
                               recommended_action, investigator, tags, created_at
                        FROM alerts
                        ORDER BY created_at DESC
                        LIMIT %s
                    """,
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []

    def add_alert(self, alert_data: Dict) -> Dict:
        """Adiciona novo alerta ao PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO alerts (alert_id, title, description, type, severity, 
                                          status, transaction_id, amount_involved, 
                                          recommended_action, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id, alert_id, title, type, severity, status, created_at
                    """,
                        (
                            alert_data.get(
                                "alert_id", f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            ),
                            alert_data.get("title", "Alerta do Sistema"),
                            alert_data.get("description", ""),
                            alert_data.get("type", "system"),
                            alert_data.get("severity", "medium"),
                            alert_data.get("status", "active"),
                            alert_data.get("transaction_id"),
                            alert_data.get("amount_involved", 0),
                            alert_data.get("recommended_action", ""),
                        ),
                    )
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding alert: {e}")
            return {}

    def update_alert_status(self, alert_id: str, status: str, investigator: str = None) -> bool:
        """Atualiza status de alerta"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE alerts 
                        SET status = %s, investigator = %s, updated_at = NOW()
                        WHERE alert_id = %s OR id::text = %s
                    """,
                        (status, investigator, alert_id, alert_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error updating alert: {e}")
            return False

    def update_transaction_status(self, transaction_id: str, new_status: str) -> bool:
        """Atualiza status de transação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE transactions 
                        SET status = %s
                        WHERE transaction_id = %s
                    """,
                        (new_status, transaction_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error updating transaction status: {e}")
            return False

    def get_model_metrics(self) -> List[Dict]:
        """Retorna métricas dos modelos"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT model_version, accuracy, precision_score, recall,
                               f1_score, roc_auc, threshold, samples_used, created_at
                        FROM model_metrics
                        ORDER BY created_at DESC
                        LIMIT 10
                    """
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching model metrics: {e}")
            return []

    def get_calibration_settings(self) -> Dict:
        """Retorna configurações de calibração do banco"""
        try:
            settings = self.get_settings()
            return settings.get(
                "calibration",
                {
                    "fraud_threshold": 0.5,
                    "high_risk_threshold": 0.7,
                    "auto_block_threshold": 0.9,
                    "review_band_low": 0.4,
                    "review_band_high": 0.7,
                },
            )
        except Exception as e:
            print(f"Error fetching calibration settings: {e}")
            return {}

    def save_calibration_settings(self, settings: Dict) -> bool:
        """Salva configurações de calibração"""
        try:
            current_settings = self.get_settings()
            current_settings["calibration"] = settings
            return self.update_settings(current_settings)
        except Exception as e:
            print(f"Error saving calibration settings: {e}")
            return False

    def get_datasets_catalog(self) -> List[Dict]:
        """Retorna catálogo de datasets disponíveis"""
        datasets = [
            {
                "id": "transactions_2024",
                "name": "Transações 2024",
                "records": 0,
                "type": "production",
                "status": "active",
            },
            {
                "id": "fraud_samples",
                "name": "Amostras de Fraude",
                "records": 0,
                "type": "training",
                "status": "active",
            },
            {
                "id": "elliptic_btc",
                "name": "Elliptic Bitcoin",
                "records": 203769,
                "type": "research",
                "status": "active",
            },
            {
                "id": "ieee_cis",
                "name": "IEEE-CIS Fraud",
                "records": 590540,
                "type": "benchmark",
                "status": "active",
            },
        ]
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM transactions")
                    row = cur.fetchone()
                    datasets[0]["records"] = int(row["count"] or 0)

                    cur.execute("SELECT COUNT(*) as count FROM transactions WHERE is_fraud = true")
                    row = cur.fetchone()
                    datasets[1]["records"] = int(row["count"] or 0)
        except Exception as e:
            print(f"Error fetching dataset counts: {e}")
        return datasets

    def get_monitoring_status(self) -> Dict:
        """Retorna status de monitoramento do sistema"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total_24h,
                            COUNT(CASE WHEN status = 'FRAUD' THEN 1 END) as frauds_24h,
                            AVG(processing_time_ms) as avg_latency,
                            MAX(processing_time_ms) as max_latency,
                            MIN(processing_time_ms) as min_latency
                        FROM transactions
                        WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """
                    )
                    stats = cur.fetchone()

                    cur.execute(
                        """
                        SELECT COUNT(*) as active_alerts
                        FROM alerts
                        WHERE status = 'active'
                    """
                    )
                    alerts = cur.fetchone()

                    return {
                        "system_status": "healthy",
                        "transactions_24h": int(stats["total_24h"] or 0),
                        "frauds_24h": int(stats["frauds_24h"] or 0),
                        "avg_latency_ms": round(float(stats["avg_latency"] or 0), 2),
                        "max_latency_ms": round(float(stats["max_latency"] or 0), 2),
                        "min_latency_ms": round(float(stats["min_latency"] or 0), 2),
                        "active_alerts": int(alerts["active_alerts"] or 0),
                        "model_status": "healthy",
                        "database_status": "connected",
                    }
        except Exception as e:
            print(f"Error fetching monitoring status: {e}")
            return {"system_status": "error", "error": str(e)}

    def generate_report(self, report_type: str, date_from: str = None, date_to: str = None) -> Dict:
        """Gera relatório baseado em dados reais do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    date_filter = ""
                    params = []
                    if date_from:
                        date_filter = " AND created_at >= %s"
                        params.append(date_from)
                    if date_to:
                        date_filter += " AND created_at <= %s"
                        params.append(date_to)

                    cur.execute(
                        f"""
                        SELECT 
                            COUNT(*) as total_transactions,
                            COUNT(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN 1 END) as total_frauds,
                            COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as total_approved,
                            SUM(amount) as total_value,
                            SUM(CASE WHEN status = 'FRAUD' THEN amount ELSE 0 END) as fraud_value,
                            AVG(risk_score) as avg_risk_score,
                            AVG(processing_time_ms) as avg_latency
                        FROM transactions
                        WHERE 1=1 {date_filter}
                    """,
                        params,
                    )
                    stats = cur.fetchone()

                    cur.execute(
                        f"""
                        SELECT channel, COUNT(*) as count, SUM(amount) as value
                        FROM transactions
                        WHERE 1=1 {date_filter}
                        GROUP BY channel
                    """,
                        params,
                    )
                    channels = cur.fetchall()

                    total = int(stats["total_transactions"] or 0)
                    frauds = int(stats["total_frauds"] or 0)

                    return {
                        "report_type": report_type,
                        "generated_at": datetime.now().isoformat() + "Z",
                        "period": {"from": date_from, "to": date_to},
                        "summary": {
                            "total_transactions": total,
                            "total_frauds": frauds,
                            "total_approved": int(stats["total_approved"] or 0),
                            "fraud_rate": round((frauds / total * 100) if total > 0 else 0, 2),
                            "total_value": float(stats["total_value"] or 0),
                            "fraud_value": float(stats["fraud_value"] or 0),
                            "avg_risk_score": round(float(stats["avg_risk_score"] or 0), 4),
                            "avg_latency_ms": round(float(stats["avg_latency"] or 0), 2),
                        },
                        "by_channel": [dict(c) for c in channels],
                        "status": "completed",
                    }
        except Exception as e:
            print(f"Error generating report: {e}")
            return {"status": "error", "error": str(e)}


postgres_store = PostgresStore()
