"""
Document API.

Two upload paths:

  POST /api/documents/                  one-shot, fine up to tens of MB
  POST /api/documents/upload/init|part|complete
                                        resumable, for large files and
                                        connections that drop

The resumable path exists because a single multi-hundred-MB POST fails often
enough to matter: free hosts cap request duration, mobile connections drop,
and a failure at 95% costs the whole upload. Parts are small, retryable, and
the client can resume by asking which parts are still missing.
"""

from __future__ import annotations

import os
import uuid

from django.conf import settings
from django.core.files.base import File
from rest_framework import status
from rest_framework.generics import RetrieveDestroyAPIView
from rest_framework.parsers import FileUploadParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, Figure, UploadSession
from .serializers import (
    DocumentSerializer,
    DocumentUploadSerializer,
    FigureSerializer,
    UploadInitSerializer,
)
from .tasks import process_document_task

PART_DIR = "uploads_parts"


class DocumentListCreateView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        docs = Document.objects.filter(owner=request.user)
        return Response(DocumentSerializer(docs, many=True).data)

    def post(self, request):
        ser = DocumentUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        doc = ser.save(owner=request.user, status=Document.Status.PENDING)
        process_document_task.delay(doc.id)
        doc.refresh_from_db()
        return Response(DocumentSerializer(doc).data, status=status.HTTP_201_CREATED)


class DocumentDetailView(RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)


class DocumentReprocessView(APIView):
    def post(self, request, pk):
        doc = Document.objects.filter(owner=request.user, id=pk).first()
        if not doc:
            return Response({"detail": "Not found."}, status=404)
        process_document_task.delay(doc.id)
        doc.refresh_from_db()
        return Response(DocumentSerializer(doc).data)


class DocumentFiguresView(APIView):
    """Everything non-textual found in the document."""

    def get(self, request, pk):
        doc = Document.objects.filter(owner=request.user, id=pk).first()
        if not doc:
            return Response({"detail": "Not found."}, status=404)

        figures = doc.figures.all()
        kind = request.query_params.get("kind")
        if kind:
            figures = figures.filter(kind=kind)
        if request.query_params.get("signatures") == "1":
            figures = figures.filter(has_signature=True)
        if request.query_params.get("include_decorative") != "1":
            figures = figures.filter(is_decorative=False)

        return Response({
            "document": doc.id,
            "count": figures.count(),
            "has_signatures": doc.has_signatures,
            "figures": FigureSerializer(figures, many=True).data,
        })


# --------------------------------------------------------------- resumable

def _part_path(upload_id: str, index: int) -> str:
    folder = os.path.join(settings.MEDIA_ROOT, PART_DIR, upload_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{index:06d}.part")


class UploadInitView(APIView):
    def post(self, request):
        ser = UploadInitSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        session = UploadSession.objects.create(
            owner=request.user,
            upload_id=uuid.uuid4().hex,
            filename=ser.validated_data["filename"],
            total_size=ser.validated_data["total_size"],
            total_parts=ser.validated_data["total_parts"],
            received_parts=[],
        )
        return Response(
            {
                "upload_id": session.upload_id,
                "total_parts": session.total_parts,
                "part_url": f"/api/documents/upload/{session.upload_id}/part/",
                "complete_url": f"/api/documents/upload/{session.upload_id}/complete/",
            },
            status=status.HTTP_201_CREATED,
        )


class UploadPartView(APIView):
    parser_classes = [MultiPartParser, FormParser, FileUploadParser]

    def post(self, request, upload_id):
        session = UploadSession.objects.filter(
            upload_id=upload_id, owner=request.user, completed=False
        ).first()
        if not session:
            return Response({"detail": "Unknown or completed upload."}, status=404)

        try:
            index = int(request.data.get("index", request.query_params.get("index", -1)))
        except (TypeError, ValueError):
            return Response({"detail": "index must be an integer."}, status=400)
        if not 0 <= index < session.total_parts:
            return Response({"detail": "index out of range."}, status=400)

        blob = request.data.get("file") or request.data.get("chunk")
        if blob is None:
            return Response({"detail": "No part data received."}, status=400)

        with open(_part_path(upload_id, index), "wb") as fh:
            for piece in blob.chunks():
                fh.write(piece)

        # idempotent: re-sending a part is safe, which is what makes retry work
        if index not in session.received_parts:
            session.received_parts = sorted(session.received_parts + [index])
            session.save(update_fields=["received_parts"])

        return Response({
            "received": len(session.received_parts),
            "total": session.total_parts,
            "missing": session.missing_parts[:50],
        })


class UploadStatusView(APIView):
    def get(self, request, upload_id):
        session = UploadSession.objects.filter(
            upload_id=upload_id, owner=request.user
        ).first()
        if not session:
            return Response({"detail": "Unknown upload."}, status=404)
        return Response({
            "upload_id": session.upload_id,
            "received": len(session.received_parts),
            "total": session.total_parts,
            "missing": session.missing_parts,
            "completed": session.completed,
            "document": session.document_id,
        })


class UploadCompleteView(APIView):
    def post(self, request, upload_id):
        session = UploadSession.objects.filter(
            upload_id=upload_id, owner=request.user
        ).first()
        if not session:
            return Response({"detail": "Unknown upload."}, status=404)
        if session.completed and session.document_id:
            return Response(DocumentSerializer(session.document).data)

        missing = session.missing_parts
        if missing:
            return Response(
                {"detail": "Upload incomplete.", "missing": missing[:100]}, status=409
            )

        folder = os.path.join(settings.MEDIA_ROOT, PART_DIR, upload_id)
        assembled = os.path.join(folder, "assembled")
        with open(assembled, "wb") as out:
            for i in range(session.total_parts):
                with open(_part_path(upload_id, i), "rb") as part:
                    for block in iter(lambda p=part: p.read(1024 * 1024), b""):
                        out.write(block)

        title = request.data.get("title") or session.filename
        doc = Document(owner=request.user, title=title, status=Document.Status.PENDING)
        with open(assembled, "rb") as fh:
            doc.file.save(session.filename, File(fh), save=True)

        session.completed = True
        session.document = doc
        session.save(update_fields=["completed", "document"])

        # parts are large; remove them as soon as the file is assembled
        for i in range(session.total_parts):
            try:
                os.remove(_part_path(upload_id, i))
            except OSError:
                pass
        try:
            os.remove(assembled)
            os.rmdir(folder)
        except OSError:
            pass

        process_document_task.delay(doc.id)
        doc.refresh_from_db()
        return Response(DocumentSerializer(doc).data, status=status.HTTP_201_CREATED)
