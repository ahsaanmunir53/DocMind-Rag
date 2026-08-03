"""
Celery app. Only actually used when CELERY_ENABLED=True (with Redis + a worker).
When disabled, settings.CELERY_TASK_ALWAYS_EAGER makes .delay() run inline, so
the same code path works with or without Celery.
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("docqa")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
