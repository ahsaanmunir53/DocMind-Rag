from django.urls import path

from . import views

urlpatterns = [
    path("", views.ResumeListCreateView.as_view(), name="resume-list"),
    path("<int:pk>/", views.ResumeDetailView.as_view(), name="resume-detail"),
    path("<int:pk>/reparse/", views.ResumeReparseView.as_view(), name="resume-reparse"),
    path("tailorings/", views.TailoringListCreateView.as_view(), name="tailoring-list"),
    path("tailorings/<int:pk>/", views.TailoringDetailView.as_view(), name="tailoring-detail"),
    path("tailorings/<int:pk>/answers/", views.TailoringAnswersView.as_view(), name="tailoring-answers"),
    path("tailorings/<int:pk>/questions/", views.TailoringQuestionsView.as_view(), name="tailoring-questions"),
    # Downloads go through a view so they work with DEBUG=False and are
    # checked against the signed-in user.
    path("tailorings/<int:pk>/download/<str:kind>/", views.TailoringDownloadView.as_view(),
         name="tailoring-download"),
]
