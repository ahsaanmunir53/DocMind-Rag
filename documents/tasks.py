"""
Background tasks.

With Celery disabled .delay() runs inline, so the project works with no broker
and no worker. Turn CELERY_ENABLED on and the same call goes to a queue with
no change at the call site.
"""

from django.conf import settings

from .services import process_document

if settings.CELERY_ENABLED:
    from celery import shared_task

    @shared_task(name="documents.process_document")
    def process_document_task(document_id: int) -> None:
        process_document(document_id)

else:

    class _Inline:
        def __init__(self, fn):
            self._fn = fn

        def delay(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    process_document_task = _Inline(process_document)
