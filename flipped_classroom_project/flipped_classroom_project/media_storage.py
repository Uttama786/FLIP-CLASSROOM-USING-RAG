"""
Cloudinary storage that picks resource_type from file extension.

Default MediaCloudinaryStorage only uploads as 'image', which breaks PDF/TXT/video files.

Path-normalisation fix
──────────────────────
django-cloudinary-storage's MediaCloudinaryStorage prepends 'media/' to the
public_id before uploading.  Django's model FileField also stores the name as
returned by storage._save(), which already contains the 'media/' prefix.  On a
subsequent re-save (e.g. fix_cloudinary_pdfs, or re-uploading) the DB name
'media/materials/file.pdf' goes through _save() again which becomes
'media/media/materials/file.pdf' in Cloudinary.

Fix: override _save() to strip any leading 'media/' from the name BEFORE
calling the parent, because the parent will add it back exactly once.
"""

import re as _re
from cloudinary_storage.storage import MediaCloudinaryStorage, RESOURCE_TYPES

_RAW_EXTENSIONS = {
    'pdf', 'txt', 'md', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
    'zip', 'rar', 'csv', 'json', 'xml', 'py', 'java', 'c', 'cpp', 'h',
}
_VIDEO_EXTENSIONS = {
    'mp4', 'webm', 'mov', 'avi', 'mkv', 'ogv', 'flv', 'wmv', 'm4v',
}


def _normalise_upload_name(name: str) -> str:
    """
    Remove any leading 'media/' prefix and collapse repeated folder names.

    Examples:
        'media/materials/foo.pdf'           -> 'materials/foo.pdf'
        'media/materials/materials/foo.pdf' -> 'materials/foo.pdf'
        'materials/materials/foo.pdf'       -> 'materials/foo.pdf'
        'materials/foo.pdf'                 -> 'materials/foo.pdf'  (unchanged)
    """
    n = name.replace('\\', '/').lstrip('/')
    # Strip one or more leading 'media/' segments
    n = _re.sub(r'^(media/)+', '', n)
    # Collapse repeated identical folder names at the start (e.g. materials/materials/)
    n = _re.sub(r'^([^/]+/)\1+', r'\1', n)
    return n


class FlipLearnMediaStorage(MediaCloudinaryStorage):
    """Upload videos as video, documents as raw, images as image."""

    def _get_resource_type(self, name):
        ext = ''
        if name and '.' in name:
            ext = name.rsplit('.', 1)[-1].lower()
        if ext in _VIDEO_EXTENSIONS:
            return RESOURCE_TYPES['VIDEO']
        if ext in _RAW_EXTENSIONS:
            return RESOURCE_TYPES['RAW']
        return RESOURCE_TYPES['IMAGE']

    def _save(self, name, content):
        # Normalise before the parent adds its own 'media/' prefix
        clean = _normalise_upload_name(name)
        return super()._save(clean, content)

