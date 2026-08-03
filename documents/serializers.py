from rest_framework import serializers

from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "title", "status", "error", "num_chunks", "created_at", "updated_at"]
        read_only_fields = ["status", "error", "num_chunks", "created_at", "updated_at"]


class DocumentUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "title", "file"]

    def validate_file(self, f):
        allowed = (".pdf", ".txt", ".md")
        if not f.name.lower().endswith(allowed):
            raise serializers.ValidationError(f"Unsupported file type. Allowed: {', '.join(allowed)}")
        if f.size > 20 * 1024 * 1024:
            raise serializers.ValidationError("File too large (max 20 MB).")
        return f

    def validate(self, attrs):
        if not attrs.get("title"):
            attrs["title"] = attrs["file"].name
        return attrs
