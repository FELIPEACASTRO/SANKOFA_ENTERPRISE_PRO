"""
Sankofa Enterprise Pro - PostgreSQL Store Service
Armazenamento persistente usando PostgreSQL em vez de arquivo JSON
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresStore:
    """Armazena configurações e dados no PostgreSQL"""
    
    def __init__(self):
        self._conn_string = os.environ.get("DATABASE_URL")
    
    def _get_connection(self):
        """Obtém conexão com o banco de dados"""
        return psycopg2.connect(self._conn_string, cursor_factory=RealDictCursor)
    
    def get_hard_rules(self) -> List[Dict]:
        """Retorna todas as hard rules do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, condition, action, enabled, 
                               created_at, updated_at
                        FROM hard_rules
                        ORDER BY id
                    """)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching hard_rules: {e}")
            return []
    
    def add_hard_rule(self, name: str, condition: str, action: str, enabled: bool = True) -> Dict:
        """Adiciona nova hard rule"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO hard_rules (name, condition, action, enabled, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        RETURNING id, name, condition, action, enabled, created_at, updated_at
                    """, (name, condition, action, enabled))
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding hard_rule: {e}")
            raise
    
    def update_hard_rule(self, rule_id: int, data: Dict) -> bool:
        """Atualiza hard rule existente"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    fields = []
                    values = []
                    for key in ['name', 'condition', 'action', 'enabled']:
                        if key in data:
                            fields.append(f"{key} = %s")
                            values.append(data[key])
                    
                    if fields:
                        fields.append("updated_at = NOW()")
                        values.append(rule_id)
                        query = f"UPDATE hard_rules SET {', '.join(fields)} WHERE id = %s"
                        cur.execute(query, values)
                        conn.commit()
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
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting hard_rule: {e}")
            return False
    
    def get_vip_list(self) -> List[Dict]:
        """Retorna lista VIP do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, identifier, identifier_type as type, reason, 
                               added_by, created_at
                        FROM vip_list
                        ORDER BY id
                    """)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching vip_list: {e}")
            return []
    
    def add_vip(self, identifier: str, identifier_type: str, reason: str, added_by: str = "system") -> Dict:
        """Adiciona item à lista VIP"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO vip_list (identifier, identifier_type, reason, added_by, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, identifier, identifier_type as type, reason, added_by, created_at
                    """, (identifier, identifier_type, reason, added_by))
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
                    cur.execute("""
                        SELECT id, identifier, identifier_type as type, reason, 
                               added_by, created_at
                        FROM hot_list
                        ORDER BY id
                    """)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching hot_list: {e}")
            return []
    
    def add_hot(self, identifier: str, identifier_type: str, reason: str, added_by: str = "system") -> Dict:
        """Adiciona item à hot list"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO hot_list (identifier, identifier_type, reason, added_by, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, identifier, identifier_type as type, reason, added_by, created_at
                    """, (identifier, identifier_type, reason, added_by))
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
                    cur.execute("""
                        SELECT config_key, config_value
                        FROM system_configs
                        WHERE config_key = 'settings'
                    """)
                    row = cur.fetchone()
                    if row:
                        return row['config_value']
                    return default_settings
        except Exception as e:
            print(f"Error fetching settings: {e}")
            return default_settings
    
    def update_settings(self, settings: Dict) -> Dict:
        """Atualiza configurações do sistema"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO system_configs (config_key, config_value, updated_at)
                        VALUES ('settings', %s, NOW())
                        ON CONFLICT (config_key) DO UPDATE SET 
                            config_value = EXCLUDED.config_value,
                            updated_at = NOW()
                        RETURNING config_value
                    """, (json.dumps(settings),))
                    conn.commit()
                    row = cur.fetchone()
                    return row['config_value'] if row else settings
        except Exception as e:
            print(f"Error updating settings: {e}")
            return settings
    
    def add_audit_log(self, action: str, user_id: str = None, details: str = None, ip_address: str = None) -> Dict:
        """Registra ação no log de auditoria"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_logs (action, user_id, details, ip_address, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, action, user_id, details, ip_address, created_at
                    """, (action, user_id, details, ip_address))
                    conn.commit()
                    row = cur.fetchone()
                    return dict(row)
        except Exception as e:
            print(f"Error adding audit_log: {e}")
            return {}
    
    def get_audit_logs(self, limit: int = 100, action_filter: str = None, 
                       start_date: str = None, end_date: str = None) -> List[Dict]:
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
    
    def add_alert(self, title: str, description: str, alert_type: str, 
                  severity: str, transaction_id: str = None, 
                  amount_involved: float = None, recommended_action: str = None) -> Dict:
        """Adiciona novo alerta"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    import uuid
                    alert_id = f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8].upper()}"
                    
                    cur.execute("""
                        INSERT INTO alerts (alert_id, title, description, type, severity, 
                                          status, transaction_id, amount_involved, 
                                          recommended_action, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'novo', %s, %s, %s, NOW(), NOW())
                        RETURNING id, alert_id, title, description, type, severity, 
                                  status, transaction_id, amount_involved, created_at
                    """, (alert_id, title, description, alert_type, severity, 
                          transaction_id, amount_involved, recommended_action))
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
                    cur.execute("""
                        UPDATE alerts SET status = %s, investigator = COALESCE(%s, investigator),
                                        updated_at = NOW()
                        WHERE id = %s
                    """, (status, investigator, alert_id))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error updating alert status: {e}")
            return False
    
    def add_feedback(self, transaction_id: str, is_fraud: bool, 
                    analyst_notes: str = None, analyst_id: str = None) -> Dict:
        """Adiciona feedback sobre transação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO feedback (transaction_id, is_fraud, analyst_notes, analyst_id, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id, transaction_id, is_fraud, analyst_notes, analyst_id, created_at
                    """, (transaction_id, is_fraud, analyst_notes, analyst_id))
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
                    cur.execute("""
                        SELECT f.id, f.transaction_id, f.is_fraud, f.analyst_notes, 
                               f.analyst_id, f.created_at,
                               t.amount, t.cpf as payer_document, t.risk_score
                        FROM feedback f
                        LEFT JOIN transactions t ON f.transaction_id = t.transaction_id
                        ORDER BY f.created_at DESC
                        LIMIT %s
                    """, (limit,))
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
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN is_fraud = true THEN 1 END) as fraud_count,
                            COUNT(CASE WHEN is_fraud = false THEN 1 END) as legit_count
                        FROM feedback
                    """)
                    row = cur.fetchone()
                    
                    total = row['total'] if row else 0
                    fraud_count = row['fraud_count'] if row else 0
                    legit_count = row['legit_count'] if row else 0
                    
                    return {
                        "total_feedback": total,
                        "fraud_confirmed": fraud_count,
                        "legit_confirmed": legit_count,
                        "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0,
                        "accuracy_improvement": 0
                    }
        except Exception as e:
            print(f"Error fetching feedback analytics: {e}")
            return {
                "total_feedback": 0,
                "fraud_confirmed": 0,
                "legit_confirmed": 0,
                "fraud_rate": 0,
                "accuracy_improvement": 0
            }
    
    def get_pending_reviews(self) -> List[Dict]:
        """Retorna transações pendentes de revisão manual"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT transaction_id, amount, cpf as payer_document, 
                               risk_score, status, created_at, channel
                        FROM transactions
                        WHERE status IN ('pending_review', 'review', 'PENDENTE')
                           OR (risk_score >= 0.5 AND risk_score < 0.7 AND status NOT IN ('APPROVED', 'FRAUD', 'approved', 'rejected'))
                        ORDER BY risk_score DESC, created_at DESC
                        LIMIT 100
                    """)
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
                    cur.execute("""
                        UPDATE transactions 
                        SET status = 'pending_review'
                        WHERE transaction_id = %s
                    """, (transaction_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            print(f"Error adding to manual review: {e}")
            return False
    
    def complete_review(self, transaction_id: str, decision: str, 
                       analyst_id: str = None, notes: str = None) -> bool:
        """Completa revisão manual de transação"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    new_status = 'approved' if decision == 'approve' else 'rejected'
                    
                    cur.execute("""
                        UPDATE transactions 
                        SET status = %s
                        WHERE transaction_id = %s
                    """, (new_status, transaction_id))
                    
                    rows_updated = cur.rowcount
                    
                    if notes or decision:
                        is_fraud = decision == 'reject'
                        cur.execute("""
                            DELETE FROM feedback WHERE transaction_id = %s
                        """, (transaction_id,))
                        cur.execute("""
                            INSERT INTO feedback (transaction_id, is_fraud, analyst_notes, analyst_id, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                        """, (transaction_id, is_fraud, notes or f"Decision: {decision}", analyst_id))
                    
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
                    cur.execute("""
                        SELECT t.transaction_id, t.amount, t.cpf as payer_document, 
                               t.risk_score, t.status, t.created_at, t.channel,
                               t.is_fraud
                        FROM transactions t
                        WHERE t.status IN ('flagged', 'investigating', 'review', 'pending_review')
                           OR t.risk_score >= 0.7
                        ORDER BY t.risk_score DESC, t.created_at DESC
                        LIMIT 100
                    """)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching investigations: {e}")
            return []
    
    def get_recent_transactions(self, limit: int = 50) -> List[Dict]:
        """Retorna transações recentes do PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT transaction_id, amount, channel, type, status,
                               risk_score, cpf, location, timestamp, created_at
                        FROM transactions
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching recent transactions: {e}")
            return []
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """Busca transação por ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT transaction_id, amount, channel, type, status,
                               risk_score, cpf, location, timestamp, created_at
                        FROM transactions
                        WHERE transaction_id = %s
                    """, (transaction_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            print(f"Error fetching transaction: {e}")
            return None


postgres_store = PostgresStore()
