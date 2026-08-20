"""
Session auth for our own same-origin JavaScript.

This file used to export a SessionAuthentication subclass with enforce_csrf()
stubbed out, on the reasoning that "endpoints are still protected by
IsAuthenticated". That reasoning is backwards: a CSRF attack works *because*
the visitor is authenticated. The browser attaches the session cookie to a
cross-site request automatically, so IsAuthenticated passes and the request
goes through. Disabling the CSRF check is what makes the attack possible.

The original symptom ("CSRF cookie not set") was fixed elsewhere and some time
ago: config.views puts @ensure_csrf_cookie on the pages, and both templates
send X-CSRFToken on every call through their api() wrapper. Nothing needs the
exemption any more, so DRF's stock SessionAuthentication is used directly in
settings.py and this module keeps only a deprecation shim for old imports.
"""
from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """Deprecated. Kept so an old import cannot silently re-open the hole."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CsrfExemptSessionAuthentication has been removed because it "
            "disabled CSRF protection. Use "
            "rest_framework.authentication.SessionAuthentication instead."
        )
