"""
CV tailoring data model.

A Resume is the uploaded original plus everything we could parse out of it.
A Tailoring is one attempt to match that CV to one job description, kept
separately so a person can target several roles from the same base CV and
compare the results.
"""

from django.contrib.auth.models import User
from django.db import models


class Resume(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PARSING = "parsing", "Parsing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="resumes")
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="resumes/")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")

    raw_text = models.TextField(blank=True, default="")
    sections = models.JSONField(default=dict, blank=True)     # parsed CV structure
    contact = models.JSONField(default=dict, blank=True)
    skills = models.JSONField(default=list, blank=True)
    format_report = models.JSONField(default=dict, blank=True)  # ATS parsing risks
    base_score = models.JSONField(default=dict, blank=True)     # score before any JD

    page_count = models.PositiveIntegerField(default=0)
    word_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class Tailoring(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        WORKING = "working", "Working"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name="tailorings")
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tailorings")

    job_title = models.CharField(max_length=255, blank=True, default="")
    company = models.CharField(max_length=255, blank=True, default="")
    job_description = models.TextField()

    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")

    requirements = models.JSONField(default=dict, blank=True)   # parsed from the JD
    gap_analysis = models.JSONField(default=dict, blank=True)
    before_score = models.JSONField(default=dict, blank=True)
    after_score = models.JSONField(default=dict, blank=True)
    tailored = models.JSONField(default=dict, blank=True)       # rewritten CV structure
    change_log = models.JSONField(default=list, blank=True)

    questions = models.JSONField(default=list, blank=True)      # gap interview
    answers = models.JSONField(default=dict, blank=True)        # user replies
    evidence = models.JSONField(default=list, blank=True)       # answered pairs
    fabrication = models.JSONField(default=dict, blank=True)    # verify.check report

    docx_file = models.FileField(upload_to="tailored/", blank=True, null=True)
    pdf_file = models.FileField(upload_to="tailored/", blank=True, null=True)
    txt_file = models.FileField(upload_to="tailored/", blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.resume.title} -> {self.job_title or 'role'}"
