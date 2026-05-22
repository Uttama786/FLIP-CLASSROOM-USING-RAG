"""Serve study materials from local disk or Cloudinary."""

import mimetypes
import os
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse, HttpResponseRedirect


def uses_cloudinary() -> bool:
    return bool(os.environ.get('CLOUDINARY_URL', '').strip())


def _normalize_name(file_name: str) -> str:
    return file_name.replace('\\', '/').lstrip('/')


def _public_id_candidates(file_name: str) -> list:
    """Build public_id variants stored in Cloudinary (with/without media/ prefix)."""
    name = _normalize_name(file_name)
    base = Path(name).stem
    ext = Path(name).suffix.lower().lstrip('.')

    candidates = [name]
    if not name.startswith('media/'):
        candidates.append(f'media/{name}')
    if ext and not name.endswith(f'.{ext}'):
        candidates.append(f'{name}.{ext}')
        if not name.startswith('media/'):
            candidates.append(f'media/{name}.{ext}')
    # Cloudinary often stores public_id without extension for image uploads
    if ext:
        for prefix in ('', 'media/'):
            stem_path = f'{prefix}{base}' if prefix else base
            if stem_path not in candidates:
                candidates.append(stem_path)
            inner = name.replace(f'.{ext}', '')
            if inner and inner not in candidates:
                candidates.append(inner)
            if not inner.startswith('media/'):
                candidates.append(f'media/{inner}')

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def resolve_cloudinary_resource(file_name: str) -> dict:
    """Look up the asset in Cloudinary Admin API (correct version + URL)."""
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


def _delivery_url(info: dict, attachment: bool = False) -> str:
    """Use secure_url from API (has the real version, not v1)."""
    url = info.get('secure_url') or info.get('url')
    if not url:
        raise Http404('Cloudinary asset has no delivery URL')
    if attachment and '/upload/' in url:
        # fl_attachment only — never fl_attachment:filename.pdf (causes HTTP 400)
        if '/fl_attachment/' not in url:
            url = url.replace('/upload/', '/upload/fl_attachment/', 1)
    return url


def _proxy_download(url: str, filename: str) -> HttpResponse | None:
    """Stream file through Django when Cloudinary allows delivery."""
    import requests

    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException:
        return None

    if resp.status_code != 200 or not resp.content:
        return None

    content_type = resp.headers.get(
        'content-type',
        mimetypes.guess_type(filename)[0] or 'application/octet-stream',
    )
    response = HttpResponse(resp.content, content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def serve_material_file(material, as_attachment: bool = True):
    """Return file response for a StudyMaterial instance."""
    if not material.file or not material.file.name:
        raise Http404('Material file not found')

    filename = Path(material.file.name).name
    if not filename and material.title:
        filename = f'{material.title[:80]}.pdf'

    if uses_cloudinary():
        info = resolve_cloudinary_resource(material.file.name)
        url = _delivery_url(info, attachment=as_attachment)

        # Proxy works for raw/txt; PDF-as-image needs Cloudinary PDF delivery enabled
        proxied = _proxy_download(url, filename)
        if proxied is not None:
            return proxied

        # Fallback: redirect to canonical URL (user may need Cloudinary security setting)
        return HttpResponseRedirect(url)

    local_path = Path(settings.MEDIA_ROOT) / _normalize_name(material.file.name)
    if not local_path.is_file():
        alt = Path(settings.MEDIA_ROOT) / 'materials' / filename
        if alt.is_file():
            local_path = alt
        else:
            raise Http404('Material file is unavailable on this server')

    return FileResponse(
        local_path.open('rb'),
        as_attachment=as_attachment,
        filename=filename,
    )
