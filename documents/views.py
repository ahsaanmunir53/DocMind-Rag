"""
Document API: upload (triggers processing), list, detail/status, delete.
"""
from django.conf import settings
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveDestroyAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document
from .serializers import DocumentSerializer, DocumentUploadSerializer
from .tasks import process_document_task


class DocumentListCreateView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        docs = Document.objects.filter(owner=request.user)
        return Response(DocumentSerializer(docs, many=True).data)

    def post(self, request):
        ser = DocumentUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        doc = ser.save(owner=request.user, status=Document.Status.PENDING)
        # runs inline if Celery disabled, on a worker if enabled
        process_document_task.delay(doc.id)
        return Response(DocumentSerializer(doc).data, status=status.HTTP_201_CREATED)


class DocumentDetailView(RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)
