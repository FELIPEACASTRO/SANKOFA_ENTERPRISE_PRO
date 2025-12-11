"""
Feedback Blueprint - Sankofa Enterprise Pro
Handles feedback-related endpoints (unified - removed duplicate submit_feedback_v2)
Refactored from production_api.py for better code organization
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

from config.settings import get_config
from utils.structured_logging import get_structured_logger
from utils.error_handling import ValidationError
from utils.log_sanitizer import sanitize_log_data

try:
    from api.schemas import FeedbackCreate
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    PydanticValidationError = Exception

config = get_config()
logger = get_structured_logger("feedback_blueprint", config.monitoring.log_level)

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')

# Database store reference (to be injected)
_postgres_store = None


def init_feedback_blueprint(postgres_store):
    """Initialize feedback blueprint with database dependencies"""
    global _postgres_store
    _postgres_store = postgres_store


@feedback_bp.route("", methods=["POST"])
@feedback_bp.route("/submit", methods=["POST"])
def submit_feedback():
    """
    Submete feedback sobre uma predição
    
    NOTA: Este endpoint unifica /api/feedback e /api/feedback/submit
    O endpoint duplicado submit_feedback_v2 foi removido.
    
    Body params:
    - transaction_id: str (required)
    - correct_label: str ("fraud" ou "legitimate") - via Pydantic
    - is_fraud: bool - fallback se Pydantic não disponível
    - comments/notes: str (optional)
    - feedback_type: str (optional, default: "correction")
    - confidence: float (optional, default: 1.0)
    """
    try:
        if not request.json:
            raise ValidationError("Request body is required", context={"endpoint": "/api/feedback"})

        # Pydantic validation if available
        if PYDANTIC_AVAILABLE:
            validated_request = FeedbackCreate(**request.json)
            transaction_id = validated_request.transaction_id
            is_fraud = validated_request.correct_label == "fraud"
            notes = validated_request.comments or ""
        else:
            # Fallback validation
            transaction_id = request.json.get("transaction_id")
            is_fraud = request.json.get("is_fraud", False)
            notes = request.json.get("notes", request.json.get("comments", ""))

            if not transaction_id:
                raise ValidationError("transaction_id is required")

    except PydanticValidationError as e:
        logger.warning("Pydantic validation failed on feedback submit", extra=sanitize_log_data({
            'endpoint': '/api/feedback', 'errors': e.errors()
        }))
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400

    if _postgres_store is None:
        return jsonify({"success": False, "error": "Feedback service not initialized"}), 500

    # Extra fields for v2 compatibility
    feedback_type = request.json.get("feedback_type", "correction")
    confidence = request.json.get("confidence", 1.0)

    result = _postgres_store.add_feedback(transaction_id, is_fraud, notes, None)
    
    if "created_at" in result and result["created_at"]:
        result["submitted_at"] = (
            result["created_at"].isoformat() + "Z"
            if hasattr(result["created_at"], "isoformat")
            else str(result["created_at"])
        )

    # Add extra fields for compatibility
    result["feedback_type"] = feedback_type
    result["confidence"] = confidence

    _postgres_store.add_audit_log(
        "FEEDBACK_SUBMIT",
        None,
        f"Feedback submitted for {transaction_id}: {'Fraud' if is_fraud else 'Legit'}",
        request.remote_addr,
    )

    return jsonify({"success": True, "message": "Feedback registrado com sucesso", "data": result})


@feedback_bp.route("/list", methods=["GET"])
def list_feedbacks():
    """Lista todos os feedbacks de analistas"""
    if _postgres_store is None:
        return jsonify({"success": False, "error": "Feedback service not initialized"}), 500

    limit = request.args.get("limit", 100, type=int)
    feedbacks = _postgres_store.get_feedback_list(limit=limit)

    for fb in feedbacks:
        if "created_at" in fb and fb["created_at"]:
            fb["feedback_timestamp"] = (
                fb["created_at"].isoformat() + "Z"
                if hasattr(fb["created_at"], "isoformat")
                else str(fb["created_at"])
            )

    return jsonify({"success": True, "feedbacks": feedbacks, "total": len(feedbacks)})


@feedback_bp.route("/analytics", methods=["GET"])
def feedback_analytics():
    """Retorna analytics dos feedbacks para melhoria do modelo"""
    if _postgres_store is None:
        return jsonify({"success": False, "error": "Feedback service not initialized"}), 500

    analytics = _postgres_store.get_feedback_analytics()

    return jsonify({
        "success": True,
        "total_feedbacks": analytics.get("total_feedback", 0),
        "fraud_confirmed": analytics.get("fraud_confirmed", 0),
        "legit_confirmed": analytics.get("legit_confirmed", 0),
        "fraud_rate": analytics.get("fraud_rate", 0),
        "accuracy_improvement": analytics.get("accuracy_improvement", 0),
    })


@feedback_bp.route("/export", methods=["GET"])
def export_feedbacks():
    """Exporta todos os feedbacks em formato CSV/JSON"""
    if _postgres_store is None:
        return jsonify({"success": False, "error": "Feedback service not initialized"}), 500

    export_format = request.args.get("format", "json")
    limit = request.args.get("limit", 1000, type=int)

    feedbacks = _postgres_store.get_feedback_list(limit=limit)

    for fb in feedbacks:
        if "created_at" in fb and fb["created_at"]:
            fb["feedback_timestamp"] = (
                fb["created_at"].isoformat() + "Z"
                if hasattr(fb["created_at"], "isoformat")
                else str(fb["created_at"])
            )

    if export_format == "csv":
        import io
        import csv
        import base64

        output = io.StringIO()
        if feedbacks:
            writer = csv.DictWriter(output, fieldnames=feedbacks[0].keys())
            writer.writeheader()
            writer.writerows(feedbacks)

        csv_content = output.getvalue()
        content_base64 = base64.b64encode(csv_content.encode()).decode()
        download_url = f"data:text/csv;base64,{content_base64}"

        return jsonify({
            "success": True,
            "format": "csv",
            "total": len(feedbacks),
            "filename": f"feedbacks_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            "download_url": download_url,
        })
    else:
        import json
        import base64

        content_json = json.dumps({
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "total": len(feedbacks),
            "feedbacks": feedbacks,
        }, indent=2, default=str)

        content_base64 = base64.b64encode(content_json.encode()).decode()
        download_url = f"data:application/json;base64,{content_base64}"

        _postgres_store.add_audit_log(
            "FEEDBACK_EXPORT", None, f"Exported {len(feedbacks)} feedbacks", request.remote_addr
        )

        return jsonify({
            "success": True,
            "format": "json",
            "total": len(feedbacks),
            "filename": f"feedbacks_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            "download_url": download_url,
        })
