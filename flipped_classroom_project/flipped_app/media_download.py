"""Serve study materials from local disk or Cloudinary.

Root cause of download failure (fixed here):
1. PATH TRIPLING: material.file.name stored as "materials/DS_notes.pdf"
   → _public_id_candidates prefixed "media/" → "media/materials/DS_notes.pdf"
   → Cloudinary secure_url returned "…/media/materials/DS_notes.pdf"
   → fl_attachment injected → "…/media/materials/materials/DS_notes.pdf"  ← WRONG
   Fix: strip one leading "media/" before building candidates.

2. fl_attachment on image-type PDF: Cloudinary NEEDS fl_attachment to return
   actual PDF bytes (without it, image/upload serves an image preview).
   The ERR_INVALID_RESPONSE in the earlier screenshot was caused by the
   tripled path, NOT by fl_attachment itself.
   Fix: keep fl_attachment but proxy the bytes through Django so the
   browser never redirects to a broken Cloudinary URL.

3. Proxy-streaming: Django fetches the file, sets Content-Disposition and
   Content-Type itself, then streams to the browser. No direct redirects.
"""

import mimetypes
import os
import re
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse, HttpResponseRedirect


# ── Helpers ───────────────────────────────────────────────────────────────────

def uses_cloudinary() -> bool:
    return bool(os.environ.get('CLOUDINARY_URL', '').strip())


def _strip_media_prefix(name: str) -> str:
    """Remove exactly one leading 'media/' (if present)."""
    name = name.replace('\\', '/').lstrip('/')
    if name.startswith('media/'):
        name = name[len('media/'):]
    return name


def _public_id_candidates(file_name: str) -> list:
    """
    Build public_id strings to try against the Cloudinary Admin API.

    django-cloudinary-storage typically stores files under a path like:
        materials/DS_complete_notes_abc123.pdf   (raw)
    or without extension for some uploads.

    We try both the normalised path and a 'media/' prefixed variant,
    plus stem-only variants for uploads that strip the extension.
    """
    name = _strip_media_prefix(file_name)
    ext = Path(name).suffix.lower().lstrip('.')   # e.g. "pdf"

    candidates = []

    # Primary: as-is normalised
    candidates.append(name)

    # Without extension (Cloudinary raw often strips it on upload)
    if ext:
        no_ext = name[: -(len(ext) + 1)]   # strip ".ext"
        candidates.append(no_ext)

    # With legacy "media/" prefix (older uploads)
    candidates.append(f'media/{name}')
    if ext:
        candidates.append(f'media/{name[: -(len(ext) + 1)]}')

    # De-duplicate, preserve order
    seen: set = set()
    result = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _inject_fl_attachment(url: str) -> str:
    """
    Inject fl_attachment into a Cloudinary delivery URL so the server
    returns the actual file bytes (required for PDFs stored as image type).
    Idempotent — won't double-inject.
    """
    if '/raw/' in url or '/raw/upload/' in url:
        return url
    if '/fl_attachment/' in url or '/fl_attachment:' in url:
        return url
    if '/upload/' in url:
        return url.replace('/upload/', '/upload/fl_attachment/', 1)
    return url


# ── Cloudinary resolution ─────────────────────────────────────────────────────

def resolve_cloudinary_resource(file_name: str) -> dict:
    """
    Look up the asset in the Cloudinary Admin API.
    Returns the resource dict (contains the authoritative secure_url with
    correct version number — NOT the URL stored in the DB which can be stale).
    Tries 'raw' first (PDFs should be raw), then 'image', then 'video'.
    """
    import cloudinary.api

    last_error = None
    for public_id in _public_id_candidates(file_name):
        for resource_type in ('raw', 'image', 'video'):
            try:
                return cloudinary.api.resource(public_id, resource_type=resource_type)
            except Exception as exc:
                last_error = exc
                continue

    raise Http404(f'Material not found in Cloudinary: {last_error}')


def _get_download_url(info: dict) -> str:
    """
    Build the final download URL from the Cloudinary API response.
    Injects fl_attachment so Cloudinary returns raw file bytes.
    """
    url = info.get('secure_url') or info.get('url')
    if not url:
        raise Http404('Cloudinary asset has no delivery URL')
    return _inject_fl_attachment(url)


# ── Proxy streaming ───────────────────────────────────────────────────────────

def _proxy_stream(url: str, filename: str) -> HttpResponse | None:
    """
    Fetch the file from Cloudinary and stream it back through Django.
    Sets Content-Disposition so the browser downloads (not previews) it.
    Returns None if the fetch fails; caller should fall back to redirect.
    """
    try:
        import requests as _requests
        resp = _requests.get(url, timeout=60, stream=True)
    except Exception:
        return None

    if resp.status_code != 200 or not resp.content:
        return None

    content_type = resp.headers.get(
        'content-type',
        mimetypes.guess_type(filename)[0] or 'application/octet-stream',
    )
    # Enforce PDF MIME type only if it is a real PDF (starts with %PDF)
    is_real_pdf = resp.content.startswith(b'%PDF')
    if filename.lower().endswith('.pdf'):
        if is_real_pdf:
            content_type = 'application/pdf'
        else:
            content_type = 'text/plain; charset=utf-8'

    response = HttpResponse(resp.content, content_type=content_type)
    safe = filename.replace('"', '_')
    response['Content-Disposition'] = f'attachment; filename="{safe}"'
    response['Content-Length'] = len(resp.content)
    return response


# ── Public entry point ────────────────────────────────────────────────────────

def serve_material_file(material, as_attachment: bool = True):
    """Return an HTTP file response for a StudyMaterial instance (LOCAL DISK ONLY)."""
    if not material.file or not material.file.name:
        raise Http404('Material file not found')

    filename = Path(material.file.name).name or f'{material.title[:80]}.pdf'

    # ── Local disk path ONLY ──────────────────────────────────────────────────
    norm = _strip_media_prefix(material.file.name)
    local_path = Path(settings.MEDIA_ROOT) / norm
    if not local_path.is_file():
        alt = Path(settings.MEDIA_ROOT) / 'materials' / filename
        if alt.is_file():
            local_path = alt
        else:
            raise Http404(f'Material file not found at {local_path} or {alt}')

    # Detect mock PDFs (which are text files starting with '==Start of PDF==')
    try:
        with local_path.open('rb') as f:
            header = f.read(4)
        is_real_pdf = header.startswith(b'%PDF')
    except Exception:
        is_real_pdf = True

    response_content_type = None
    if filename.lower().endswith('.pdf') and not is_real_pdf:
        response_content_type = 'text/plain; charset=utf-8'

    return FileResponse(
        local_path.open('rb'),
        as_attachment=as_attachment,
        filename=filename,
        content_type=response_content_type,
    )
