"""
Data model.

Beyond the original Document/Chunk pair this now records what a page is and
what is drawn on it, because a PDF is not a stream of text - it is pages that
happen to contain text, and figures that contain none.

Chunk.embedding still adapts to the database:
  - Postgres + pgvector -> a real VectorField (index-backed similarity search)
  - SQLite (dev default) -> JSONField holding the vector as a list
Same code path; only the column type differs.
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
        UPLOADING = "uploading", "Uploading"
        EXTRACTING = "extracting", "Reading pages"
        ANALYSING = "analysing", "Analysing figures"
        INDEXING = "indexing", "Building index"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")

    # progress, so a 600-page upload shows movement instead of a spinner
    stage_detail = models.CharField(max_length=120, blank=True, default="")
    pages_total = models.PositiveIntegerField(default=0)
    pages_done = models.PositiveIntegerField(default=0)
    num_chunks = models.PositiveIntegerField(default=0)
    num_figures = models.PositiveIntegerField(default=0)

    # file identity and shape
    sha256 = models.CharField(max_length=64, blank=True, default="", db_index=True)
    size_bytes = models.BigIntegerField(default=0)
    page_count = models.PositiveIntegerField(default=0)
    is_scanned = models.BooleanField(default=False)
    has_signatures = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["owner", "status"])]

    def __str__(self):
        return f"{self.title} ({self.status})"

    @property
    def progress_percent(self) -> int:
        if self.status == self.Status.READY:
            return 100
        if not self.pages_total:
            return 0
        return min(99, int(self.pages_done / self.pages_total * 100))


class Page(models.Model):
    """One page. Keeps citations honest - answers cite a page, not an offset."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="pages")
    number = models.PositiveIntegerField()
    text = models.TextField(blank=True, default="")
    width = models.FloatField(default=0)
    height = models.FloatField(default=0)
    has_text_layer = models.BooleanField(default=True)

    class Meta:
        ordering = ["document_id", "number"]
        unique_together = ("document", "number")
        indexes = [models.Index(fields=["document", "number"])]

    def __str__(self):
        return f"{self.document_id} p{self.number}"


class Figure(models.Model):
    """A non-text region: photo, chart, vector diagram, stamp or signature."""

    class Kind(models.TextChoices):
        SIGNATURE = "signature", "Signature"
        STAMP = "stamp_or_seal", "Stamp or seal"
        CHART = "chart", "Chart"
        DIAGRAM = "diagram", "Diagram"
        FLOWCHART = "flowchart", "Flowchart"
        TABLE_IMAGE = "table_image", "Table image"
        PHOTO = "photo", "Photo"
        SCREENSHOT = "screenshot", "Screenshot"
        LOGO = "logo", "Logo"
        LETTERHEAD = "letterhead", "Letterhead"
        HANDWRITING = "handwriting", "Handwriting"
        FORMULA = "formula", "Formula"
        MAP = "map", "Map"
        FLOOR_PLAN = "floor_plan", "Floor plan"
        ID_DOCUMENT = "id_document", "ID document"
        BARCODE = "barcode_or_qr", "Barcode or QR"
        IMAGE = "image", "Image"
        OTHER = "other", "Other"

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="figures")
    page_number = models.PositiveIntegerField()
    kind = models.CharField(max_length=24, choices=Kind.choices, default=Kind.OTHER)

    image = models.ImageField(upload_to="figures/", blank=True, null=True)
    caption = models.TextField(blank=True, default="")
    ocr_text = models.TextField(blank=True, default="")
    labels = models.JSONField(default=list, blank=True)

    has_signature = models.BooleanField(default=False)
    has_stamp = models.BooleanField(default=False)
    is_decorative = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    analysed_by = models.CharField(max_length=16, default="heuristics")

    # geometry, in PDF points, so a viewer can highlight the exact region
    x0 = models.FloatField(default=0)
    y0 = models.FloatField(default=0)
    x1 = models.FloatField(default=0)
    y1 = models.FloatField(default=0)
    source = models.CharField(max_length=8, default="raster")   # raster | vector
    sha1 = models.CharField(max_length=40, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "page_number", "id"]
        indexes = [
            models.Index(fields=["document", "page_number"]),
            models.Index(fields=["document", "kind"]),
        ]

    def __str__(self):
        return f"{self.get_kind_display()} p{self.page_number}"

    @property
    def bbox(self):
        return [self.x0, self.y0, self.x1, self.y1]


class Chunk(models.Model):
    """A retrievable passage. Figures become chunks too, via their caption."""

    class Kind(models.TextChoices):
        TEXT = "text", "Text"
        FIGURE = "figure", "Figure"

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index = models.PositiveIntegerField()
    text = models.TextField()
    kind = models.CharField(max_length=8, choices=Kind.choices, default=Kind.TEXT)
    page_number = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    figure = models.ForeignKey(
        Figure, on_delete=models.CASCADE, related_name="chunks", null=True, blank=True
    )
    token_estimate = models.PositiveIntegerField(default=0)
    embedding = embedding_field()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "index"]
        unique_together = ("document", "index")
        indexes = [
            models.Index(fields=["document", "kind"]),
            models.Index(fields=["document", "page_number"]),
        ]

    def __str__(self):
        return f"{self.document_id} #{self.index}"


class UploadSession(models.Model):
    """Resumable upload state.

    Browsers and free-tier hosts both give up on very long single requests.
    Splitting the file into parts means a dropped connection costs one part,
    not the whole upload.
    """

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="upload_sessions")
    upload_id = models.CharField(max_length=40, unique=True, db_index=True)
    filename = models.CharField(max_length=255)
    total_size = models.BigIntegerField(default=0)
    total_parts = models.PositiveIntegerField(default=0)
    received_parts = models.JSONField(default=list, blank=True)
    completed = models.BooleanField(default=False)
    document = models.ForeignKey(
        Document, on_delete=models.SET_NULL, null=True, blank=True, related_name="upload_session"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.filename} ({len(self.received_parts)}/{self.total_parts})"

    @property
    def missing_parts(self):
        return [i for i in range(self.total_parts) if i not in set(self.received_parts)]
