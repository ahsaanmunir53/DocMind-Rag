"""
Celery is optional.

The package is only imported when CELERY_ENABLED is on, so the project
installs and runs without a broker, a worker, or the celery package itself.
Importing it unconditionally meant a missing optional dependency stopped
Django from starting at all.
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

if os.getenv("CELERY_ENABLED", "False").strip().lower() in ("1", "true", "yes", "on"):
    try:
        from .celery import app as celery_app

        __all__ = ("celery_app",)
    except ImportError:  # pragma: no cover
        celery_app = None
        __all__ = ()
else:
    celery_app = None
    __all__ = ()
