"""
Legacy entry point for Project 2: Iris Classifier.
Delegates to the modular enterprise ML experimentation pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
