# Expose the Celery app so tasks are registered when Django starts.
from .celery import app as celery_app

__all__ = ("celery_app",)
