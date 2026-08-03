from django.contrib import admin

from .models import Chunk, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "owner", "status", "num_chunks", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("title", "owner__username")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "index", "created_at")
    search_fields = ("document__title", "text")
