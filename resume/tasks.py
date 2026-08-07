"""
Background tasks.

Mirrors the documents app: when Celery is off, .delay() runs inline so the
project works with no broker. Turn CELERY_ENABLED on and the same call goes
to a worker with no code change.
"""

from django.conf import settings

from resume.services import process_resume, process_tailoring

if settings.CELERY_ENABLED:
    from config.celery import app as celery_app

    @celery_app.task(name="resume.process_resume")
    def process_resume_task(resume_id: int):
        process_resume(resume_id)

    @celery_app.task(name="resume.process_tailoring")
    def process_tailoring_task(tailoring_id: int):
        process_tailoring(tailoring_id)

else:

    class _Inline:
        def __init__(self, fn):
            self._fn = fn

        def delay(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    process_resume_task = _Inline(process_resume)
    process_tailoring_task = _Inline(process_tailoring)
