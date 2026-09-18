"""Гарантирует, что корень репозитория виден в sys.path независимо от того,
как запущен pytest (`pytest tests/` или `python -m pytest`) — иначе
`import config` из корневых модулей падает при прямом вызове `pytest`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
