"""Test bootstrap: point STATE_DIR at a throwaway directory BEFORE importing
the app so tests never touch real state, and make the repo root importable."""
import os
import sys
import tempfile

os.environ["STATE_DIR"] = tempfile.mkdtemp(prefix="papertrail-test-state-")
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # silence chromadb telemetry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
