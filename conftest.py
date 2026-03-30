"""conftest.py — pytest configuration for zk-gpu-accelerator."""

import sys
import os

# Ensure the project root is on sys.path so `from src.xxx import ...` works
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
