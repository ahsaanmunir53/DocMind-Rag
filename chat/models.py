"""
Chat history: a Conversation is tied to a user and (optionally) a document.
Each Message is either the user's question or the assistant's answer. Assistant
messages keep the list of source chunks they cited, so the UI can show citations.
"""
from django.contrib.auth.models import User
from django.db import models

from documents.models import Document


class Conversation(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="conversations")
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="conversations",
        null=True, blank=True,  # null => ask across ALL of the user's documents
    )
    title = models.CharField(max_length=255, default="New conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title


class Message(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=12, choices=Role.choices)
    content = models.TextField()
    # for assistant messages: [{"document": "...", "index": 0, "score": 0.83, "text": "..."}]
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"
