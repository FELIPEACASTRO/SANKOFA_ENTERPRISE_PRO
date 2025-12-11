"""
DSR (Data Subject Rights) API Endpoints
LGPD Art. 18 compliance
"""

from flask import Blueprint, request, jsonify
from api.schemas import DSRAccessRequest, DSRDeletionRequest
from compliance.dsr_service import dsr_service
from pydantic import ValidationError
from utils.log_sanitizer import sanitize_log_data
import logging

logger = logging.getLogger(__name__)

dsr_bp = Blueprint('dsr', __name__)

@dsr_bp.route('/access', methods=['POST'])
async def dsr_access():
    """LGPD Art. 18, I - Access to personal data"""
    try:
        validated = DSRAccessRequest(**request.json)

        request_id = f"DSR-ACCESS-{int(time.time())}"
        report = await dsr_service.access_request(validated.cpf, request_id)

        logger.info(
            "DSR access request processed",
            extra=sanitize_log_data({'request_id': request_id})
        )

        return jsonify({'success': True, 'data': report})

    except ValidationError as e:
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400

@dsr_bp.route('/deletion', methods=['POST'])
async def dsr_deletion():
    """LGPD Art. 18, VI - Right to be forgotten"""
    try:
        validated = DSRDeletionRequest(**request.json)

        request_id = f"DSR-DELETE-{int(time.time())}"
        result = await dsr_service.deletion_request(validated.cpf, request_id)

        logger.info(
            "DSR deletion request processed",
            extra=sanitize_log_data({'request_id': request_id})
        )

        return jsonify(result)

    except ValidationError as e:
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400

@dsr_bp.route('/portability', methods=['POST'])
async def dsr_portability():
    """LGPD Art. 18, V - Data portability"""
    try:
        validated = DSRAccessRequest(**request.json)

        request_id = f"DSR-PORT-{int(time.time())}"
        data_bytes = await dsr_service.portability_request(validated.cpf, request_id)

        from flask import send_file
        import io

        return send_file(
            io.BytesIO(data_bytes),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'personal_data_{request_id}.json'
        )

    except ValidationError as e:
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400
