"""pytest configuration: add the project venv to sys.path so tests can
import rollio_device_nero (installed in .venv, not system-wide)."""
import sys
from pathlib import Path

_VENV_SITE = (
    Path(__file__).resolve().parents[2]
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
)
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
