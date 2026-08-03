"""
Chat API: ask a question (retrieve -> generate -> save), list/read conversations.
"""
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from core.llm import generate_answer
from documents.models import Document
from documents.retrieval import retrieve

from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer


class AskView(APIView):
    """POST {question, document?, conversation?} -> assistant Message with sources."""

    def post(self, request):
        question = (request.data.get("question") or "").strip()
        if not question:
            return Response({"error": "question is required"}, status=status.HTTP_400_BAD_REQUEST)

        document_id = request.data.get("document") or None
        if document_id is not None:
            # ensure the document belongs to this user
            get_object_or_404(Document, pk=document_id, owner=request.user)

        # get or create the conversation
        conv_id = request.data.get("conversation") or None
        if conv_id:
            conversation = get_object_or_404(Conversation, pk=conv_id, owner=request.user)
        else:
            conversation = Conversation.objects.create(
                owner=request.user,
                document_id=document_id,
                title=question[:60],
            )

        # save the user's message
        Message.objects.create(conversation=conversation, role=Message.Role.USER, content=question)

        # RAG: retrieve -> generate
        chunks = retrieve(request.user, question, document_id=document_id)
        answer = generate_answer(question, chunks)
        sources = [
            {"document": c.document_title, "index": c.chunk_index, "score": c.score, "text": c.text[:300]}
            for c in chunks
        ]
        msg = Message.objects.create(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content=answer,
            sources=sources,
        )
        conversation.save(update_fields=["updated_at"])

        return Response(
            {"conversation": conversation.id, "message": MessageSerializer(msg).data},
            status=status.HTTP_201_CREATED,
        )


class ConversationListView(ListAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(owner=self.request.user)


class ConversationDetailView(RetrieveAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(owner=self.request.user)
