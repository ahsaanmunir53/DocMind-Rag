"""CV tailoring API."""

from django.conf import settings
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveDestroyAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Resume, Tailoring
from .serializers import (
    AnswersSerializer,
    ResumeSerializer,
    ResumeUploadSerializer,
    TailoringCreateSerializer,
    TailoringSerializer,
)
from .tasks import process_resume_task, process_tailoring_task


class ResumeListCreateView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        qs = Resume.objects.filter(owner=request.user)
        return Response(ResumeSerializer(qs, many=True).data)

    def post(self, request):
        ser = ResumeUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        resume = ser.save(owner=request.user, status=Resume.Status.PENDING)
        process_resume_task.delay(resume.id)
        resume.refresh_from_db()
        return Response(ResumeSerializer(resume).data, status=status.HTTP_201_CREATED)


class ResumeDetailView(RetrieveDestroyAPIView):
    serializer_class = ResumeSerializer

    def get_queryset(self):
        return Resume.objects.filter(owner=self.request.user)


class ResumeReparseView(APIView):
    def post(self, request, pk):
        resume = Resume.objects.filter(owner=request.user, id=pk).first()
        if not resume:
            return Response({"detail": "Not found."}, status=404)
        process_resume_task.delay(resume.id)
        resume.refresh_from_db()
        return Response(ResumeSerializer(resume).data)


class TailoringListCreateView(APIView):
    def get(self, request):
        qs = Tailoring.objects.filter(owner=request.user)
        resume_id = request.query_params.get("resume")
        if resume_id:
            qs = qs.filter(resume_id=resume_id)
        return Response(TailoringSerializer(qs, many=True).data)

    def post(self, request):
        ser = TailoringCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        resume = ser.validated_data["resume"]
        if resume.owner_id != request.user.id:
            return Response({"detail": "Not your CV."}, status=403)
        if resume.status != Resume.Status.READY:
            return Response(
                {"detail": f"That CV is still {resume.get_status_display().lower()}."},
                status=409,
            )
        job = ser.save(owner=request.user, status=Tailoring.Status.PENDING)
        process_tailoring_task.delay(job.id)
        job.refresh_from_db()
        return Response(TailoringSerializer(job).data, status=status.HTTP_201_CREATED)


class TailoringDetailView(RetrieveDestroyAPIView):
    serializer_class = TailoringSerializer

    def get_queryset(self):
        return Tailoring.objects.filter(owner=self.request.user)


class TailoringAnswersView(APIView):
    """Submit gap-interview answers and re-run the tailoring with them.

    The answers become source material, so the rewrite can finally add
    bullets - and the fabrication check treats them as evidence the user
    supplied rather than something the model invented.
    """

    def post(self, request, pk):
        job = Tailoring.objects.filter(pk=pk, owner=request.user).first()
        if not job:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if job.status == Tailoring.Status.WORKING:
            return Response(
                {"detail": "This tailoring is still running. Wait for it to finish."},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = AnswersSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        merged = dict(job.answers or {})
        merged.update(serializer.validated_data["answers"])

        job.answers = merged
        job.status = Tailoring.Status.PENDING
        job.error = ""
        job.save(update_fields=["answers", "status", "error"])

        process_tailoring_task(job.id)
        job.refresh_from_db()
        return Response(TailoringSerializer(job).data)


class TailoringQuestionsView(APIView):
    """Regenerate the interview questions without touching existing answers."""

    def post(self, request, pk):
        job = Tailoring.objects.filter(pk=pk, owner=request.user).first()
        if not job:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        from resume import interview

        job.questions = interview.build_questions(job.requirements, job.gap_analysis)
        job.save(update_fields=["questions"])
        return Response({"questions": job.questions})
