from django.conf import settings
from rest_framework import serializers

from .models import Document, Figure


class FigureSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    kind_label = serializers.CharField(source="get_kind_display", read_only=True)

    class Meta:
        model = Figure
        fields = [
            "id", "page_number", "kind", "kind_label", "caption", "ocr_text",
            "labels", "has_signature", "has_stamp", "is_decorative",
            "confidence", "analysed_by", "bbox", "source", "image_url",
        ]

    def get_image_url(self, obj):
        return obj.image.url if obj.image else None


class DocumentSerializer(serializers.ModelSerializer):
    progress = serializers.IntegerField(source="progress_percent", read_only=True)
    figure_summary = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id", "title", "status", "error", "stage_detail", "progress",
            "pages_total", "pages_done", "page_count", "num_chunks",
            "num_figures", "size_bytes", "is_scanned", "has_signatures",
            "figure_summary", "created_at", "updated_at",
        ]
        read_only_fields = fields

    def get_figure_summary(self, obj):
        if not obj.num_figures:
            return {}
        counts = {}
        for kind in obj.figures.values_list("kind", flat=True):
            counts[kind] = counts.get(kind, 0) + 1
        return counts


class DocumentUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "title", "file"]

    def validate_file(self, f):
        allowed = (".pdf", ".txt", ".md", ".docx")
        if not f.name.lower().endswith(allowed):
            raise serializers.ValidationError(
                f"Unsupported file type. Allowed: {', '.join(allowed)}"
            )
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        if f.size > limit:
            raise serializers.ValidationError(
                f"File is {f.size / 1024 / 1024:.0f} MB; the limit is "
                f"{settings.MAX_UPLOAD_MB} MB. Raise MAX_UPLOAD_MB in .env, or use "
                "the resumable upload endpoint for very large files."
            )
        return f

    def validate(self, attrs):
        if not attrs.get("title"):
            attrs["title"] = attrs["file"].name
        return attrs


class UploadInitSerializer(serializers.Serializer):
    filename = serializers.CharField(max_length=255)
    total_size = serializers.IntegerField(min_value=1)
    total_parts = serializers.IntegerField(min_value=1, max_value=10000)

    def validate_filename(self, v):
        if not v.lower().endswith((".pdf", ".txt", ".md", ".docx")):
            raise serializers.ValidationError("Unsupported file type.")
        return v

    def validate(self, attrs):
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        if attrs["total_size"] > limit:
            raise serializers.ValidationError(
                f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit."
            )
        return attrs
