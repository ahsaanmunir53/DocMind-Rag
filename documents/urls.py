from django.urls import path

from . import views

urlpatterns = [
    path("", views.DocumentListCreateView.as_view(), name="document-list"),
    path("<int:pk>/", views.DocumentDetailView.as_view(), name="document-detail"),
    path("<int:pk>/figures/", views.DocumentFiguresView.as_view(), name="document-figures"),
    path("<int:pk>/reprocess/", views.DocumentReprocessView.as_view(), name="document-reprocess"),

    # resumable upload, for files too large or connections too flaky for one POST
    path("upload/init/", views.UploadInitView.as_view(), name="upload-init"),
    path("upload/<str:upload_id>/part/", views.UploadPartView.as_view(), name="upload-part"),
    path("upload/<str:upload_id>/complete/", views.UploadCompleteView.as_view(), name="upload-complete"),
    path("upload/<str:upload_id>/status/", views.UploadStatusView.as_view(), name="upload-status"),
]
