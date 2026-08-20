from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

from . import views

urlpatterns = [
    path("admin/", admin.site.urls),

    # pages
    path("", views.home, name="home"),                 # public 3D landing
    path("app/", views.app, name="app"),               # the chat app (login required)
    path("cv/", views.cv, name="cv"),                  # CV tailoring workspace

    # auth
    path("accounts/signup/", views.signup, name="signup"),
    path("accounts/login/", views.RememberMeLoginView.as_view(), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),

    # REST API
    path("api/documents/", include("documents.urls")),
    path("api/chat/", include("chat.urls")),
    path("api/resume/", include("resume.urls")),
    path("api-auth/", include("rest_framework.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
