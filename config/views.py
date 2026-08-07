"""
Page views: public landing, the (login-required) app, and public signup.
"""
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db import IntegrityError
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie


def home(request):
    """Public 3D landing page. Logged-in users go straight to the app."""
    if request.user.is_authenticated:
        return redirect("app")
    return render(request, "landing.html")


@ensure_csrf_cookie
@login_required
def app(request):
    """The chat-with-your-documents interface (CSRF cookie guaranteed for the JS)."""
    return render(request, "index.html")


@ensure_csrf_cookie
@login_required
def cv(request):
    """The CV tailoring workspace."""
    return render(request, "cv.html")


def signup(request):
    """Public self-serve registration."""
    if request.user.is_authenticated:
        return redirect("app")
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # A double-clicked submit button fires two POSTs at once. Both pass
            # is_valid() (neither user exists yet), the first one saves, and the
            # second hits the UNIQUE constraint on auth_user.username. Catching
            # it here turns a 500 into a normal "username taken" message.
            try:
                user = form.save()
            except IntegrityError:
                form.add_error(
                    "username",
                    "That username was just taken. Please pick a different one.",
                )
            else:
                login(request, user)
                return redirect("app")
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})
