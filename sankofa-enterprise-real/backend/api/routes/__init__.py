"""
API Routes Module - Sankofa Enterprise Pro
Exports all blueprints for modular API organization
"""

from api.routes.auth import auth_bp, require_auth, require_permission, init_auth_blueprint
from api.routes.fraud import feedback_bp, init_feedback_blueprint
from api.routes.admin import admin_bp
from api.routes.observability import observability_bp
from api.routes.dsr import dsr_bp

__all__ = [
    # Blueprints
    'auth_bp',
    'feedback_bp',
    'admin_bp',
    'observability_bp',
    'dsr_bp',
    
    # Auth decorators
    'require_auth',
    'require_permission',
    
    # Init functions
    'init_auth_blueprint',
    'init_feedback_blueprint',
]
