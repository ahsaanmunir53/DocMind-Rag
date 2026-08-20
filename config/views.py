"""
Page views: public landing, the (login-required) app, and public signup.
"""
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.db import IntegrityError
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie


class RememberMeLoginView(LoginView):
    """Django's login view plus a "keep me signed in" choice.

    Unticked, the session cookie is dropped when the browser closes — which is
    what you want on a shared or public machine. Ticked, it lives for
    SESSION_COOKIE_AGE and is refreshed on every visit.

    Nothing about the login is written to localStorage. The session cookie is
    HttpOnly, so no script on the page can read it; a token in localStorage
    would be readable by any script that runs, and one XSS would be enough to
    take the account.
    """

    template_name = "login.html"

    def form_valid(self, form):
        remember = self.request.POST.get("remember") == "on"
        response = super().form_valid(form)
        if not remember:
            self.request.session.set_expiry(0)   # until the browser closes
        return response


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
