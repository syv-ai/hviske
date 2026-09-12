"""Privacy-safe classification of Hugging Face Hub failures."""

from __future__ import annotations

import re
from dataclasses import dataclass

from huggingface_hub.errors import HfHubHTTPError

_SAFE_PHASE = re.compile(r"^[a-z_]{1,32}$")
_ENDPOINT_PATTERNS = (
    ("preupload", re.compile(r"/preupload(?:/|$)")),
    ("lfs_batch", re.compile(r"/info/lfs/objects/batch(?:[?#]|$)")),
    (
        "xet",
        re.compile(r"(?:xethub\.hf\.co|/xet-|/api/.*/xet(?:/|$)|xet.*(?:token|write))"),
    ),
    ("commit", re.compile(r"/commit(?:/|$)")),
    ("paths_info", re.compile(r"/paths-info(?:/|$)")),
    ("repo_tree", re.compile(r"/tree(?:/|$)")),
    ("repo_create", re.compile(r"/api/repos/create(?:[?#]|$)")),
)
_ERROR_CODE_REASONS = {
    "RevisionNotFound": "revision_not_found",
    "EntryNotFound": "entry_not_found",
    "RepoNotFound": "repository_not_found",
    "GatedRepo": "authorisation",
}


def annotate_hub_error(
    error: HfHubHTTPError, *, phase: str, reason: str | None = None
) -> None:
    """Attach only validated classifier hints to an exception."""
    if _SAFE_PHASE.fullmatch(phase):
        setattr(error, "_p1_hub_phase", phase)
    if reason in {"stale_parent", "xet_unavailable"}:
        setattr(error, "_p1_hub_reason", reason)


@dataclass(frozen=True)
class HubErrorDiagnostic:
    """Bounded, allowlisted diagnostic fields safe for operational logs."""

    status_code: int | None
    phase: str
    reason: str
    retryable: bool


def classify_hub_error(error: BaseException) -> HubErrorDiagnostic:
    """Classify a Hub error without returning any transport-controlled text.

    Returns:
        Status, coarse request phase, allowlisted reason and retry decision. Unknown
        fields are represented by ``None`` or ``"unknown"``.
    """
    if not isinstance(error, HfHubHTTPError):
        return HubErrorDiagnostic(None, "unknown", "unknown", False)

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    status_code = status if isinstance(status, int) else None
    phase = _phase_from_response(response)
    if phase == "unknown":
        tagged_phase = getattr(error, "_p1_hub_phase", None)
        if isinstance(tagged_phase, str) and _SAFE_PHASE.fullmatch(tagged_phase):
            phase = tagged_phase
    reason = _tagged_reason(error) or _reason_from_response(response, status_code)
    retryable = _is_retryable(status_code=status_code, reason=reason)
    return HubErrorDiagnostic(status_code, phase, reason, retryable)


def _is_retryable(*, status_code: int | None, reason: str) -> bool:
    if reason in {"stale_parent", "xet_unavailable"}:
        return True
    return status_code in {408, 409, 425, 429} or (
        status_code is not None and status_code >= 500
    )


def _phase_from_response(response: object) -> str:
    request = getattr(response, "request", None)
    url = getattr(request, "url", None)
    if url is None:
        return "unknown"
    value = str(url).casefold()
    for phase, pattern in _ENDPOINT_PATTERNS:
        if pattern.search(value):
            return phase
    return "unknown"


def _reason_from_response(response: object, status_code: int | None) -> str:
    headers = getattr(response, "headers", None)
    error_code = headers.get("X-Error-Code") if hasattr(headers, "get") else None
    if isinstance(error_code, str) and error_code in _ERROR_CODE_REASONS:
        return _ERROR_CODE_REASONS[error_code]

    fragments = _response_fragments(response)
    if any("a commit has happened since" in item for item in fragments) or any(
        "parent commit" in item
        and any(word in item for word in ("mismatch", "does not match", "stale"))
        for item in fragments
    ):
        return "stale_parent"
    if any(
        "xet" in item
        and any(word in item for word in ("temporar", "unavailable", "timeout"))
        for item in fragments
    ):
        return "xet_unavailable"
    if status_code == 429:
        return "rate_limited"
    if status_code is not None and status_code >= 500:
        return "server_error"
    if status_code in {401, 403}:
        return "authorisation"
    return "unknown"


def _response_fragments(response: object) -> tuple[str, ...]:
    values: list[str] = []
    headers = getattr(response, "headers", None)
    if hasattr(headers, "get"):
        message = headers.get("X-Error-Message")
        if isinstance(message, str):
            values.append(message.casefold())
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            payload = json_method()
        except Exception:
            payload = None
        _collect_strings(payload, values=values, depth=0)
    return tuple(values)


def _collect_strings(value: object, *, values: list[str], depth: int) -> None:
    if depth > 3 or len(values) >= 16:
        return
    if isinstance(value, str):
        values.append(value[:512].casefold())
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, values=values, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, values=values, depth=depth + 1)


def _tagged_reason(error: HfHubHTTPError) -> str | None:
    reason = getattr(error, "_p1_hub_reason", None)
    return reason if reason in {"stale_parent", "xet_unavailable"} else None
