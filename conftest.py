"""Pytest configuration.

Make the repository root and the ``scripts/`` directory importable so tests can
``import`` the helper modules (at the root) and the pipeline scripts (in
``scripts/``) by their plain module names, regardless of the current directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "scripts"):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
