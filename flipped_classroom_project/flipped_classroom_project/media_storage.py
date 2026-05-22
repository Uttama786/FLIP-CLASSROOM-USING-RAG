"""
Cloudinary storage that picks resource_type from file extension.

Default MediaCloudinaryStorage only uploads as 'image', which breaks PDF/TXT/video files.
"""

from cloudinary_storage.storage import MediaCloudinaryStorage, RESOURCE_TYPES

_RAW_EXTENSIONS = {
    'pdf', 'txt', 'md', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
    'zip', 'rar', 'csv', 'json', 'xml', 'py', 'java', 'c', 'cpp', 'h',
}
_VIDEO_EXTENSIONS = {
    'mp4', 'webm', 'mov', 'avi', 'mkv', 'ogv', 'flv', 'wmv', 'm4v',
}


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
