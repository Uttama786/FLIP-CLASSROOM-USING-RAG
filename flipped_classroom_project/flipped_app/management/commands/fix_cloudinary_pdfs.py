"""
Re-upload PDF study materials to Cloudinary as resource_type=raw so downloads work.

PDFs uploaded earlier as 'image' are blocked from delivery until you enable
PDF delivery in Cloudinary Security settings. Uploading as 'raw' fixes downloads.

  python manage.py fix_cloudinary_pdfs
  python manage.py fix_cloudinary_pdfs --dry-run
"""

import pathlib

import cloudinary
import cloudinary.uploader
from django.conf import settings
from django.core.management.base import BaseCommand

from flipped_app.media_download import resolve_cloudinary_resource
from flipped_app.models import StudyMaterial


class Command(BaseCommand):
    help = 'Re-upload PDF materials on Cloudinary as raw resources for reliable download.'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        if not cloudinary.config().cloud_name:
            self.stderr.write(self.style.ERROR('CLOUDINARY_URL is not configured.'))
            return

        dry = options['dry_run']
        fixed = skipped = failed = 0

        for mat in StudyMaterial.objects.exclude(file='').exclude(file__isnull=True):
            fname = (mat.file.name or '').lower()
            if not fname.endswith('.pdf') and 'pdf' not in mat.title.lower():
                skipped += 1
                continue

            try:
                info = resolve_cloudinary_resource(mat.file.name)
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f'  [skip] #{mat.id} not in Cloudinary: {exc}'))
                skipped += 1
                continue

            if info.get('resource_type') == 'raw':
                self.stdout.write(f'  [ok]   #{mat.id} already raw')
                skipped += 1
                continue

            public_id = info['public_id']
            fmt = info.get('format') or 'pdf'

            if dry:
                self.stdout.write(f'  [dry]  #{mat.id} would re-upload {public_id} as raw')
                fixed += 1
                continue

            try:
                media_root = pathlib.Path(settings.MEDIA_ROOT)
                local = None
                for candidate in [
                    media_root / mat.file.name,
                    media_root / 'materials' / pathlib.Path(mat.file.name).name,
                ]:
                    if candidate.is_file():
                        local = candidate
                        break

                upload_source = str(local) if local else info['secure_url']
                result = cloudinary.uploader.upload(
                    upload_source,
                    resource_type='raw',
                    public_id=public_id,
                    format=fmt,
                    overwrite=True,
                    invalidate=True,
                )
                new_name = result.get('public_id', public_id)
                if fmt and not new_name.endswith(f'.{fmt}'):
                    stored = f'{new_name}.{fmt}'
                else:
                    stored = new_name
                mat.file.name = stored.replace('\\', '/')
                mat.save(update_fields=['file'])
                self.stdout.write(self.style.SUCCESS(
                    f'  [ok]   #{mat.id} {mat.title[:40]} -> raw/{stored}'
                ))
                fixed += 1
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f'  [err]  #{mat.id}: {exc}'))
                failed += 1

        self.stdout.write(self.style.SUCCESS(
            f'Done. fixed={fixed} skipped={skipped} failed={failed}'
            + (' (dry-run)' if dry else '')
        ))
        if failed:
            self.stdout.write(
                'If upload-from-URL fails, enable PDF delivery in Cloudinary → Settings → Security, '
                'then run this command again.'
            )
