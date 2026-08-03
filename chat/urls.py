from django.urls import path

from .views import AskView, ConversationDetailView, ConversationListView

urlpatterns = [
    path("ask/", AskView.as_view(), name="ask"),
    path("conversations/", ConversationListView.as_view(), name="conversation-list"),
    path("conversations/<int:pk>/", ConversationDetailView.as_view(), name="conversation-detail"),
]
