"""
Re-upload study materials to Cloudinary as resource_type=raw so downloads work.

PDFs uploaded as 'image' are blocked from delivery (HTTP 401) unless you
enable PDF delivery in Cloudinary Security settings.  Uploading as 'raw' is
the reliable fix and works on all Cloudinary plans.

NOTE: The command no longer pre-filters by filename extension because
Cloudinary strips extensions from public_ids, making .pdf checks unreliable.
Instead it asks the Cloudinary Admin API for each asset's actual resource_type.

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
            # ── Ask Cloudinary what resource_type this asset actually is ──────────
            # Do NOT pre-filter by filename extension: Cloudinary strips .pdf from
            # public_ids, so the DB stores names like 'media/materials/notes_abc123'
            # (no .pdf suffix) and the old extension check skipped everything.
            try:
                info = resolve_cloudinary_resource(mat.file.name)
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f'  [skip] #{mat.id} not in Cloudinary: {exc}'))
                skipped += 1
                continue

            if info.get('resource_type') == 'raw':
                self.stdout.write(f'  [ok]   #{mat.id} already raw — skipping')
                skipped += 1
                continue

            # Only re-upload assets that are NOT raw (image/video type blocks PDF delivery)
            self.stdout.write(
                f'  [fix]  #{mat.id} {mat.title[:40]} '
                f'is resource_type={info.get("resource_type")} — will re-upload as raw'
            )

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
