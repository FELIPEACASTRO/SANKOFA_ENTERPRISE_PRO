"""
Fraud Blueprint
Handles fraud-related endpoints
"""

from flask import Blueprint, request, jsonify

fraud_bp = Blueprint('fraud', __name__)

# TODO: Migrate endpoints from production_api.py to this blueprint
