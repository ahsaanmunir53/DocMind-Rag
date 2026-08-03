"""
Data model for documents and their embedded chunks.

The Chunk.embedding field adapts to the database:
  - On Postgres with pgvector -> a real VectorField (fast similarity search).
  - On SQLite (dev default)    -> a JSONField holding the vector as a list
                                  (we compute cosine similarity in Python).
Same code; only the field type differs by DB.
"""
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models

if settings.USING_PGVECTOR:
    from pgvector.django import VectorField

    def embedding_field():
        return VectorField(dimensions=settings.EMBEDDING_DIM, null=True, blank=True)
else:
    def embedding_field():
        return models.JSONField(null=True, blank=True)


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")
    num_chunks = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.status})"


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index = models.PositiveIntegerField()
    text = models.TextField()
    embedding = embedding_field()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "index"]
        unique_together = ("document", "index")

    def __str__(self):
        return f"{self.document.title} #{self.index}"
