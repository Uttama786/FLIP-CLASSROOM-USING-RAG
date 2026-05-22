"""Helpers for serving uploaded files from local disk or Cloudinary."""

import os
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponseRedirect


def uses_cloudinary() -> bool:
    return bool(os.environ.get('CLOUDINARY_URL', '').strip())


def cloudinary_download_url(file_name: str, download_filename: str | None = None) -> str:
    """
    Build a signed Cloudinary URL for downloading a stored file.
    Avoids storage.exists() HEAD requests that return 401 on Render.
    """
    import cloudinary.utils
    from flipped_classroom_project.media_storage import FlipLearnMediaStorage

    storage = FlipLearnMediaStorage()
    public_id = storage._normalise_name(file_name)
    public_id = storage._prepend_prefix(public_id)
    resource_type = storage._get_resource_type(file_name)

    flags = None
    if download_filename:
        flags = f'attachment:{download_filename}'

    # cloudinary_url returns (url_string, options_dict) — not a plain string
    url, _opts = cloudinary.utils.cloudinary_url(
        public_id,
        resource_type=resource_type,
        secure=True,
        sign_url=True,
        type='upload',
        flags=flags,
    )
    if not url or not str(url).startswith(('http://', 'https://')):
        raise Http404('Could not build Cloudinary download URL')
    return str(url)


def serve_material_file(material, as_attachment: bool = True):
    """
    Return redirect URL (Cloudinary) or FileResponse (local MEDIA_ROOT).
    """
    if not material.file or not material.file.name:
        raise Http404('Material file not found')

    filename = Path(material.file.name).name

    if uses_cloudinary():
        url = cloudinary_download_url(
            material.file.name,
            download_filename=filename if as_attachment else None,
        )
        return HttpResponseRedirect(url)

    storage = material.file.storage
    local_path = Path(settings.MEDIA_ROOT) / material.file.name.replace('\\', '/')
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
