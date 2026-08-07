"""Vocabulary equivalence for keyword matching.

A CV that says "purchasing" and a posting that says "procurement" describe the
same work. Exact-token matching scores that as a missing requirement, which is
both wrong and demoralising - it sends people off inventing experience they
already have.

Each group below is a set of terms that mean the same thing in a hiring
context. Membership is symmetric: any member matches any other. Groups are
deliberately conservative. "Lead" and "manage" are here; "Python" and "Ruby"
are not, and never will be.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

GROUPS: List[Set[str]] = [
    # --- procurement / supply chain ---
    {"procurement", "purchasing", "sourcing", "buying"},
    {"inventory", "stock", "stock control", "inventory management"},
    {"warehouse", "warehousing", "storage", "stores"},
    {"logistics", "distribution", "supply chain"},
    {"vendor", "supplier", "vendor management", "supplier management"},
    {"tender", "bid", "rfp", "rfq", "solicitation"},
    {"freight", "shipping", "forwarding", "haulage"},
    {"customs", "clearance", "import", "export"},
    {"reconciliation", "reconcile", "stock take", "stocktake", "cycle count"},

    # --- data / reporting ---
    {"reporting", "reports", "dashboards", "dashboard"},
    {"analysis", "analytics", "analytical"},
    {"excel", "spreadsheets", "spreadsheet", "ms excel"},
    {"sql", "queries", "querying"},
    {"visualisation", "visualization", "charts", "charting"},
    {"forecasting", "forecast", "demand planning"},

    # --- systems ---
    {"erp", "sap", "oracle", "netsuite", "dynamics"},
    {"crm", "salesforce", "hubspot"},
    {"database", "databases", "rdbms"},

    # --- engineering ---
    {"ci/cd", "cicd", "ci", "continuous integration", "pipelines", "pipeline"},
    {"kubernetes", "k8s", "eks", "aks", "gke"},
    {"containers", "docker", "containerisation", "containerization"},
    {"iac", "terraform", "cloudformation", "infrastructure as code"},
    {"observability", "monitoring", "telemetry", "instrumentation"},
    {"aws", "amazon web services"},
    {"gcp", "google cloud"},
    {"azure", "microsoft azure"},
    {"api", "apis", "rest", "restful", "endpoints"},
    {"testing", "tests", "qa", "quality assurance"},
    {"debugging", "troubleshooting", "diagnosis", "root cause"},

    # --- people / process ---
    {"lead", "led", "leading", "leadership", "manage", "managed", "managing"},
    {"mentor", "mentored", "coaching", "coached", "training", "trained"},
    {"stakeholder", "stakeholders", "clients", "customers"},
    {"coordinate", "coordinated", "coordination", "liaise", "liaison"},
    {"documentation", "documented", "sops", "runbooks", "procedures"},
    {"compliance", "regulatory", "audit", "auditing", "governance"},
    {"budget", "budgets", "budgeting", "cost control", "cost"},
    {"negotiation", "negotiate", "negotiated"},
    {"improvement", "optimisation", "optimization", "efficiency", "streamlining"},
    {"cross-functional", "cross functional", "interdepartmental"},

    # --- generic delivery verbs ---
    {"implement", "implemented", "deploy", "deployed", "rolled out", "rollout"},
    {"develop", "developed", "build", "built", "created"},
    {"maintain", "maintained", "support", "supported", "upkeep"},
]

# term -> canonical id, built once at import
_INDEX: Dict[str, int] = {}
for _i, _group in enumerate(GROUPS):
    for _term in _group:
        _INDEX[_term] = _i


def aliases(term: str) -> Set[str]:
    """Every term that means the same thing, including the term itself."""
    t = (term or "").strip().lower()
    gid = _INDEX.get(t)
    if gid is None:
        return {t} if t else set()
    return set(GROUPS[gid])


def _stem(word: str) -> str:
    """Crude suffix trim. 'managing' and 'managed' should not be two skills."""
    w = word.lower()
    for suf in ("ising", "izing", "ing", "ised", "ized", "ers", "er", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def matches(term: str, haystack: Set[str]) -> bool:
    """Does the CV evidence this requirement, allowing for wording differences?

    Three passes, cheapest first: exact, alias group, then shared stem.
    """
    t = (term or "").strip().lower()
    if not t:
        return False

    if t in haystack:
        return True

    for alt in aliases(t):
        if alt in haystack:
            return True
        if " " in alt and any(alt in h for h in haystack):
            return True

    if len(t) > 4:
        st = _stem(t)
        if len(st) > 3 and any(_stem(h) == st for h in haystack if " " not in h):
            return True

    return False


def expand_text(text: str) -> Set[str]:
    """All alias forms present in a body of text, for coverage reporting."""
    found: Set[str] = set()
    low = (text or "").lower()
    for group in GROUPS:
        for term in group:
            if re.search(rf"\b{re.escape(term)}\b", low):
                found |= group
                break
    return found
