"""
Main Flask Application — Entry point for the AI + Quantum Materials
Discovery Platform backend.

Run with:
    cd Backend
    python -m app.main

Or:
    cd Backend
    python app/main.py
"""

from flask import Flask
from flask_cors import CORS
from app.api.routes import api


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS for frontend dev server (React on localhost)
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",   # Vite default
                "http://localhost:3000",   # Common React dev port
                "http://localhost:8080",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3000",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    })

    # Register API blueprint
    app.register_blueprint(api)

    return app


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 60)
    print("  AI + Quantum Materials Discovery Platform")
    print("  Flask server starting on http://localhost:5000")
    print("=" * 60 + "\n")
    print("Endpoints:")
    print("  GET  /health    — Health check")
    print("  POST /discover  — Run discovery pipeline")
    print("  GET  /elements  — List all elements")
    print()
    app.run(host="0.0.0.0", port=5000, debug=True)
