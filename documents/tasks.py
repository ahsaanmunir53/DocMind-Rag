"""
Celery task wrapper around the ingestion pipeline. With Celery disabled this
still runs (eagerly/inline); with Celery enabled it runs on a worker.
"""
from celery import shared_task

from .services import process_document


@shared_task
def process_document_task(document_id: int) -> None:
    process_document(document_id)
