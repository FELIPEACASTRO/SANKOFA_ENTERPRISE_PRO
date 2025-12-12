"""
Sankofa Enterprise Pro - Unified Production Server
Serves both API and static frontend on a single port (5000)
Optimized for Replit Autoscale Deployment
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
from flask import send_from_directory, send_file

from backend.api.production_api import app as api_app

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

api_app.view_functions.pop("root", None)
if "static" in api_app.view_functions:
    api_app.view_functions.pop("static", None)


@api_app.route("/")
def serve_index():
    """Serve index.html for the root path"""
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return send_file(index_path)
    return {"error": "Frontend not built. Run 'npm run build' first."}, 404


@api_app.route("/<path:path>")
def serve_frontend(path):
    """Serve static frontend files, fallback to index.html for SPA routing"""
    if path.startswith("api/"):
        return {"error": "API endpoint not found"}, 404

    file_path = FRONTEND_DIST / path

    if file_path.exists() and file_path.is_file():
        return send_from_directory(FRONTEND_DIST, path)

    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return send_file(index_path)

    return {"error": "Frontend not built. Run 'npm run build' first."}, 404


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))

    print(f"=== Sankofa Enterprise Pro - Unified Server ===")
    print(f"API + Frontend running on port {port}")
    print(f"Frontend dist: {FRONTEND_DIST}")
    print(f"Frontend exists: {FRONTEND_DIST.exists()}")
    if FRONTEND_DIST.exists():
        print(f"Frontend files: {list(FRONTEND_DIST.iterdir())}")

    api_app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
