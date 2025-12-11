"""
Flask Application Factory
Creates configured Flask app instances
"""

from flask import Flask
from flask_cors import CORS

def create_app(config_name='production'):
    """
    Application factory pattern

    Args:
        config_name: 'production', 'development', 'testing'

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)

    # Load configuration
    if config_name == 'testing':
        app.config['TESTING'] = True

    # Enable CORS
    CORS(app)

    # Register blueprints
    from api.routes.dsr import dsr_bp
    app.register_blueprint(dsr_bp, url_prefix='/api/dsr')

    # Apply middleware
    from api.middleware.security import SecurityHeadersMiddleware
    SecurityHeadersMiddleware(app)

    return app
