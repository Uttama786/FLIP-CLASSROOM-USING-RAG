"""
Upload local media files (videos + study materials) to Cloudinary and update DB paths.

Run on your development machine with production credentials:

  set CLOUDINARY_URL=cloudinary://key:secret@cloud_name
  python manage.py push_media_to_cloudinary

Free Cloudinary plan: max 10 MB per file. Larger videos are skipped — use a YouTube URL
on that lecture instead, or compress the file before uploading.
"""

import os
import pathlib

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand

# Cloudinary free tier default (bytes). Override: CLOUDINARY_MAX_UPLOAD_MB=10
DEFAULT_MAX_BYTES = 10 * 1024 * 1024


def _max_upload_bytes():
    mb = os.environ.get('CLOUDINARY_MAX_UPLOAD_MB', '10')
    try:
        return int(float(mb) * 1024 * 1024)
    except ValueError:
        return DEFAULT_MAX_BYTES


def _human_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f'{num_bytes / (1024 * 1024):.1f} MB'
    return f'{num_bytes / 1024:.1f} KB'


class Command(BaseCommand):
    help = "Upload local video/material files to Cloudinary for production deployment."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='List files that would be uploaded without uploading.',
        )
        parser.add_argument(
            '--force-large',
            action='store_true',
            help='Attempt upload even when file exceeds size limit (will fail on free tier).',
        )

    def handle(self, *args, **options):
        if not os.environ.get('CLOUDINARY_URL', '').strip():
            self.stderr.write(
                self.style.ERROR(
                    'CLOUDINARY_URL is not set. Add it to .env or environment first.'
                )
            )
            return

        from flipped_app.models import VideoLecture, StudyMaterial

        try:
            from cloudinary.exceptions import BadRequest as CloudinaryBadRequest
        except ImportError:
            CloudinaryBadRequest = Exception

        media_root = pathlib.Path(settings.MEDIA_ROOT)
        dry = options['dry_run']
        force_large = options['force_large']
        max_bytes = _max_upload_bytes()

        uploaded = 0
        skipped = 0
        missing = 0
        too_large = []

        self.stdout.write(
            f'Cloudinary upload limit: {_human_size(max_bytes)} per file '
            f'(set CLOUDINARY_MAX_UPLOAD_MB to override)\n'
        )

        def _local_path(file_field):
            if not file_field or not file_field.name:
                return None
            name = file_field.name.replace('\\', '/')
            candidates = [
                media_root / name,
                media_root / 'materials' / pathlib.PurePosixPath(name).name,
                media_root / 'videos' / pathlib.PurePosixPath(name).name,
            ]
            for p in candidates:
                if p.is_file():
                    return p
            return None

        def _upload_instance(obj, field_name, label, is_video=False):
            nonlocal uploaded, skipped, missing
            f = getattr(obj, field_name)
            if not f or not f.name:
                skipped += 1
                return

            # Skip only if URL is already a Cloudinary (or other remote) link
            try:
                url = str(f.url)
                if url.startswith('http') and 'cloudinary.com' in url:
                    self.stdout.write(f'  [skip] {label} already on cloud: {f.name}')
                    skipped += 1
                    return
            except Exception:
                pass

            local = _local_path(f)
            if not local:
                self.stdout.write(self.style.WARNING(f'  [miss] {label} no local file: {f.name}'))
                missing += 1
                return

            size = local.stat().st_size
            rel_name = f.name.replace('\\', '/')

            if size > max_bytes and not force_large:
                msg = (
                    f'  [large] {label}: {_human_size(size)} > limit {_human_size(max_bytes)} — skipped'
                )
                if is_video:
                    msg += ' (add a YouTube URL on this video in admin, or compress the file)'
                self.stdout.write(self.style.WARNING(msg))
                too_large.append((label, local, size))
                skipped += 1
                return

            if dry:
                self.stdout.write(
                    f'  [dry]  would upload {label}: {local} ({_human_size(size)})'
                )
                uploaded += 1
                return

            try:
                with local.open('rb') as fh:
                    f.save(rel_name, File(fh), save=False)
                obj.save(update_fields=[field_name])
                self.stdout.write(
                    self.style.SUCCESS(f'  [ok]   {label}: {rel_name} ({_human_size(size)})')
                )
                uploaded += 1
            except CloudinaryBadRequest as exc:
                err = str(exc).lower()
                if 'too large' in err or 'file size' in err:
                    self.stdout.write(self.style.WARNING(
                        f'  [large] {label}: Cloudinary rejected ({exc})'
                    ))
                    too_large.append((label, local, size))
                    skipped += 1
                else:
                    self.stdout.write(self.style.ERROR(f'  [err]  {label}: {exc}'))
                    skipped += 1
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f'  [err]  {label}: {exc}'))
                skipped += 1

        self.stdout.write('Uploading video files …')
        for video in VideoLecture.objects.exclude(video_file='').exclude(video_file__isnull=True):
            _upload_instance(
                video, 'video_file', f'Video #{video.id} {video.title[:40]}', is_video=True
            )

        self.stdout.write('Uploading study materials …')
        for mat in StudyMaterial.objects.exclude(file='').exclude(file__isnull=True):
            _upload_instance(mat, 'file', f'Material #{mat.id} {mat.title[:40]}')

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(
            f'Done. uploaded={uploaded} skipped={skipped} missing_local={missing} '
            f'too_large={len(too_large)}'
            + (' (dry-run)' if dry else '')
        ))

        if too_large:
            self.stdout.write('')
            self.stdout.write(self.style.WARNING('Files over Cloudinary limit:'))
            for label, path, size in too_large:
                self.stdout.write(f'  - {label}: {path} ({_human_size(size)})')
            self.stdout.write(
                '\nFor large videos: edit the lecture in Django admin and set '
                '"YouTube URL" instead of uploading the MP4, or compress below 10 MB.'
            )
