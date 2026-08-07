from rest_framework import serializers

from .models import Resume, Tailoring


class ResumeSerializer(serializers.ModelSerializer):
    issues = serializers.SerializerMethodField()

    class Meta:
        model = Resume
        fields = [
            "id", "title", "status", "error", "contact", "skills",
            "page_count", "word_count", "base_score", "issues", "created_at",
        ]
        read_only_fields = fields

    def get_issues(self, obj):
        return (obj.format_report or {}).get("issues", [])


class ResumeUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = ["title", "file"]

    def validate_file(self, value):
        name = (value.name or "").lower()
        if not name.endswith((".pdf", ".docx", ".doc", ".txt", ".md")):
            raise serializers.ValidationError(
                "Upload a PDF, DOCX or plain-text CV."
            )
        return value


class TailoringSerializer(serializers.ModelSerializer):
    docx_url = serializers.SerializerMethodField()
    pdf_url = serializers.SerializerMethodField()
    txt_url = serializers.SerializerMethodField()

    class Meta:
        model = Tailoring
        fields = [
            "id", "resume", "job_title", "company", "status", "error",
            "requirements", "gap_analysis", "before_score", "after_score",
            "tailored", "change_log", "questions", "answers", "evidence",
            "fabrication", "docx_url", "pdf_url", "txt_url", "created_at",
        ]
        read_only_fields = fields

    def get_docx_url(self, obj):
        return obj.docx_file.url if obj.docx_file else None

    def get_pdf_url(self, obj):
        return obj.pdf_file.url if obj.pdf_file else None

    def get_txt_url(self, obj):
        return obj.txt_file.url if obj.txt_file else None


class AnswersSerializer(serializers.Serializer):
    """Interview answers, keyed by question index as a string."""

    answers = serializers.DictField(child=serializers.CharField(allow_blank=True, max_length=800))

    def validate_answers(self, value):
        if len(value) > 40:
            raise serializers.ValidationError("Too many answers submitted.")
        return value


class TailoringCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tailoring
        fields = ["resume", "job_title", "company", "job_description"]

    def validate_job_description(self, value):
        if len(value.strip()) < 80:
            raise serializers.ValidationError(
                "Paste the full job description - at least a few sentences. "
                "A short blurb produces a weak keyword match."
            )
        return value
