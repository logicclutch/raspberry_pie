"""Stream addresses the dashboard may switch the engine to, and how they are shown.

Allowed: a local video file (absolute path), an rtsp:// / rtsps:// link, a webcam index ("0") or
"picamera". An empty string means "the camera from config.yaml". Passwords in RTSP links are
stored as given (the engine needs them) but never shown or logged: use `mask_source`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".ts", ".mpg", ".mpeg"})
STREAM_SCHEMES = frozenset({"rtsp", "rtsps"})
MAX_LEN = 1024


class SourceError(ValueError):
    """The address can't be used; the message is written for the dashboard user."""


def source_kind(src: str) -> str:
    s = src.strip()
    if not s:
        return "default"
    if s.isdigit():
        return "webcam"
    if s.lower() == "picamera":
        return "picamera"
    if "://" in s:
        return "rtsp"
    return "file"


def _unquote_path(s: str) -> str:
    """Accept paths pasted from Finder/Terminal: surrounding quotes, or `\\ ` escaped spaces."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    if "\\ " in s and not Path(s).expanduser().exists():
        s = s.replace("\\ ", " ")
    return s


def normalize_source(raw: str) -> str:
    """Validate a user-entered address and return the canonical form to store. Raises SourceError."""
    s = raw.strip()
    if not s:
        return ""
    if len(s) > MAX_LEN:
        raise SourceError(f"address is too long (max {MAX_LEN} characters)")
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        raise SourceError("address contains control characters")
    if s.isdigit():
        if len(s) > 2:
            raise SourceError("webcam index must be a small number such as 0")
        return str(int(s))
    if s.lower() == "picamera":
        return "picamera"
    if "://" in s:
        return _normalize_link(s)
    return _normalize_file(_unquote_path(s))


def _normalize_link(s: str) -> str:
    try:
        u = urlsplit(s)
        _ = u.port  # raises ValueError on a bad port
    except ValueError:
        raise SourceError("the link is not a valid address") from None
    if u.scheme.lower() not in STREAM_SCHEMES:
        raise SourceError("only rtsp:// or rtsps:// camera links are supported")
    if not u.hostname:
        raise SourceError("the link has no camera address (e.g. rtsp://192.168.1.64:554/…)")
    if any(c.isspace() for c in s):
        raise SourceError("the link must not contain spaces")
    return s


def _normalize_file(s: str) -> str:
    p = Path(s).expanduser()
    if not p.is_absolute():
        raise SourceError("use the full path of the video file, e.g. /Users/you/Videos/gate.mp4")
    if p.suffix.lower() not in VIDEO_SUFFIXES:
        raise SourceError("not a video file (use " + ", ".join(sorted(VIDEO_SUFFIXES)) + ")")
    try:
        resolved = p.resolve()
    except (OSError, RuntimeError):
        raise SourceError("video file not found") from None
    if not resolved.is_file():
        raise SourceError("video file not found")
    if resolved.suffix.lower() not in VIDEO_SUFFIXES:  # a symlink named .mp4 pointing elsewhere
        raise SourceError("not a video file")
    return str(resolved)


def mask_source(src: str) -> str:
    """Safe to show or log: an RTSP password becomes ***."""
    s = src.strip()
    if "://" not in s:
        return s
    try:
        u = urlsplit(s)
    except ValueError:
        return "rtsp://…"
    if u.password is None:
        return s
    host = u.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = u.port
    except ValueError:
        port = None
    netloc = f"{u.username or ''}:***@{host}" + (f":{port}" if port else "")
    return urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
