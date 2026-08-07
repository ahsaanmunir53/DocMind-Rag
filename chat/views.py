"""
Chat API: ask a question (condense -> retrieve -> generate -> save).
"""
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from core.embeddings import embed_query
from core.llm import RetrievedChunk, condense_question, generate_answer
from documents import retrieval
from documents.models import Chunk, Document

from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer


class AskView(APIView):
    """POST {question, document?, conversation?} -> assistant Message with sources."""

    def post(self, request):
        question = (request.data.get("question") or "").strip()
        if not question:
            return Response({"error": "question is required"},
                            status=status.HTTP_400_BAD_REQUEST)

        document_id = request.data.get("document") or None
        if document_id is not None:
            get_object_or_404(Document, pk=document_id, owner=request.user)

        conv_id = request.data.get("conversation") or None
        if conv_id:
            conversation = get_object_or_404(Conversation, pk=conv_id, owner=request.user)
        else:
            conversation = Conversation.objects.create(
                owner=request.user, document_id=document_id, title=question[:60]
            )

        history = [
            {"role": m.role, "content": m.content}
            for m in conversation.messages.order_by("created_at")[:12]
        ]
        Message.objects.create(
            conversation=conversation, role=Message.Role.USER, content=question
        )

        search_query = condense_question(question, history)

        chunk_qs = Chunk.objects.filter(document__owner=request.user,
                                        document__status=Document.Status.READY)
        if document_id:
            chunk_qs = chunk_qs.filter(document_id=document_id)

        try:
            qvec = embed_query(search_query)
        except Exception:
            qvec = None

        hits = retrieval.search(
            search_query, chunk_qs, query_vec=qvec,
            top_k=settings.TOP_K, use_rerank=settings.USE_RERANK,
        )

        chunks = [
            RetrievedChunk(
                text=h.text, document_title=h.document_title, chunk_index=h.chunk_index,
                score=h.score, page_number=h.page_number, kind=h.kind,
                figure_id=h.figure_id,
            )
            for h in hits
        ]
        answer = generate_answer(question, chunks)

        sources = [
            {
                "document": h.document_title,
                "document_id": h.document_id,
                "index": h.chunk_index,
                "page": h.page_number,
                "kind": h.kind,
                "figure_id": h.figure_id,
                "score": round(h.score, 4),
                "matched_by": _matched_by(h),
                "text": h.text[:320],
            }
            for h in hits
        ]

        msg = Message.objects.create(
            conversation=conversation, role=Message.Role.ASSISTANT,
            content=answer, sources=sources,
        )
        conversation.save(update_fields=["updated_at"])

        return Response(
            {
                "conversation": conversation.id,
                "message": MessageSerializer(msg).data,
                "search_query": search_query if search_query != question else None,
            },
            status=status.HTTP_201_CREATED,
        )


def _matched_by(hit) -> str:
    if hit.lexical_rank and hit.vector_rank:
        return "keyword + meaning"
    if hit.lexical_rank:
        return "keyword"
    if hit.vector_rank:
        return "meaning"
    return "ranked"


class ConversationListView(ListAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(owner=self.request.user)


class ConversationDetailView(RetrieveAPIView):
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(owner=self.request.user)
