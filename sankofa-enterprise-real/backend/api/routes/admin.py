"""
Admin Blueprint
Handles admin-related endpoints
"""

from flask import Blueprint, request, jsonify

admin_bp = Blueprint('admin', __name__)

# TODO: Migrate endpoints from production_api.py to this blueprint
