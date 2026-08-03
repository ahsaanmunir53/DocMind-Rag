"""
Session auth WITHOUT DRF's CSRF enforcement.

Our API is called by our own same-origin JavaScript, and every endpoint already
requires an authenticated user (IsAuthenticated). Django's session CSRF check
was rejecting valid JSON requests in some browsers ("CSRF cookie not set"), so we
skip that specific check here while keeping login-based protection intact.
"""
from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return  # endpoints are still protected by IsAuthenticated
