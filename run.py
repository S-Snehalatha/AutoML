#!/usr/bin/env python3
"""
AutoML Studio — Launcher Script
Starts both the Streamlit UI and FastAPI backend.
"""

import subprocess
import sys
import os
import threading
import time


def run_fastapi():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    print("=" * 60)
    print("  ⚡  AutoML Studio")
    print("  Streamlit UI  → http://localhost:8501")
    print("  FastAPI Docs  → http://localhost:8000/docs")
    print("=" * 60)

    # Start FastAPI in background thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    time.sleep(2)

    # Run Streamlit in foreground
    run_streamlit()
